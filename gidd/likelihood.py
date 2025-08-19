import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm.auto as tqdm
import torch.distributed as dist
from transformers import AutoModelForCausalLM


class ELBO(nn.Module):
    def __init__(self, config, model, noise_schedule, loss_fn):
        super().__init__()
        self.config = config
        self.model = model
        self.noise_schedule = noise_schedule
        self.loss_fn = loss_fn

    def forward(self, input_ids, attention_mask, t):
        # Sample z_t from the noise schedule
        z_t = self.noise_schedule.sample_zt(input_ids, t)
        
        # Get model outputs (logits)
        outputs = self.model(z_t, t)
        
        # Compute loss using the loss function
        # The loss function expects: (logits, input_ids, attention_mask, z_t, t)
        _, elbo, _ = self.loss_fn(outputs, input_ids, attention_mask, z_t, t, reduction="none")
        return elbo


def compute_elbo(elbo_fn: ELBO, batch, num_samples=128, t_eps=1e-4, return_token_nlls=False, reduce_metrics=False, show_progress=True):
    device = batch["input_ids"].device
    ts = torch.linspace(t_eps, 1 - t_eps, num_samples, device=device)

    # Stream the mean over time samples to avoid stacking on GPU
    sum_elbo_cpu = None  # accumulated sum of per-token NLLs on CPU
    with torch.no_grad():
        for i in tqdm.trange(num_samples, disable=not show_progress):
            t = ts[i, None].expand(batch["input_ids"].shape[0])
            elbo = elbo_fn(batch["input_ids"], batch["attention_mask"], t)
            elbo_cpu = elbo.detach().to(torch.float32).cpu()
            if sum_elbo_cpu is None:
                sum_elbo_cpu = elbo_cpu
            else:
                sum_elbo_cpu += elbo_cpu

    # Mean over time samples per token (CPU tensor: [batch, seq_len])
    token_nlls = sum_elbo_cpu / float(num_samples)

    # Compute totals on CPU to minimize GPU memory
    attention_mask_cpu = batch["attention_mask"].cpu()
    total_nll = (token_nlls * attention_mask_cpu).sum()
    total_tokens = attention_mask_cpu.sum()
    total_batch_size = torch.tensor(batch["input_ids"].size(0), dtype=torch.long)

    # Optional distributed reduction on device
    if reduce_metrics and dist.is_available() and dist.is_initialized():
        total_nll = total_nll.to(device)
        total_tokens = total_tokens.to(device)
        total_batch_size = total_batch_size.to(device)
        dist.all_reduce(total_nll)
        dist.all_reduce(total_tokens)
        dist.all_reduce(total_batch_size)

    # Compute averages
    nll = total_nll / total_tokens
    seq_nll = total_nll / total_batch_size

    metrics = {
        "nll": nll,
        "ppl": nll.exp(),
        "seq_nll": seq_nll,
        "token_count": total_tokens,
        "token_nll_sum": total_nll,
        "batch_size": total_batch_size,
    }

    # Return token_nlls on CPU to avoid unnecessary GPU memory use
    return (metrics, token_nlls) if return_token_nlls else metrics


def compute_causal_nll(model, batch, reduce_metrics=False, return_token_nlls=False):
    labels = batch["input_ids"][:, 1:]
    loss_mask = batch["attention_mask"][:, :-1]

    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits[:, :-1]
    loss = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
    
    total_nll = (loss * loss_mask).sum()
    total_tokens = loss_mask.sum()
    total_batch_size = torch.tensor(batch["input_ids"].size(0), device=loss.device)

    if reduce_metrics and dist.is_available() and dist.is_initialized():
        dist.all_reduce(total_nll)
        dist.all_reduce(total_tokens)
        dist.all_reduce(total_batch_size)

    nll = total_nll / total_tokens
    seq_nll = total_nll / total_batch_size

    metrics = {
        "nll": nll,
        "ppl": nll.exp(),
        "seq_nll": seq_nll,
        "seq_ppl": seq_nll.exp(),
    }

    return (metrics, loss) if return_token_nlls else metrics