import torch
import torch.nn as nn
import tqdm.auto as tqdm
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer

from gidd.diffusion_process import HybridDiffusion
from gidd.sampling import GiddSampler
from gidd.utils import sample_categorical


class GiddPipeline(nn.Module):
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
        config = model.config
        noise_schedule = HybridDiffusion(tokenizer, p_uniform=config.p_uniform)
        return cls(model, noise_schedule, tokenizer, config)
    
    def __init__(self, model, noise_schedule, tokenizer, config):
        super().__init__()
        self.model = model
        self.noise_schedule = noise_schedule
        self.tokenizer = tokenizer
        self.config = config

        self.sampler = GiddSampler(model, tokenizer, noise_schedule, t_eps=config.t_eps, compile_step=False)

    @torch.compiler.disable
    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm.tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm.tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    @torch.no_grad()
    def generate(
        self,
        num_samples: int = 1,
        num_inference_steps: int = 128,
        show_progress: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ) -> list[str]:
        device = next(self.model.parameters()).device
        with torch.autocast(device.type, dtype):
            return self.sampler.generate(
                num_samples=num_samples,
                num_denoising_steps=num_inference_steps,
                max_length=self.config.max_seq_len,
                decode=True,
                show_progress=show_progress,
            )
    
    @torch.no_grad()
    def self_correction(
        self,
        texts: list[str],
        num_inference_steps: int = 128,
        temperature: float = 0.1,
        temperature_schedule: str = "progressive",  # "fixed", "progressive", "exponential"
        temp_start: float = 0.5,
        temp_end: float = 0.1,
        t0: float = 0.01,
        tokens_per_step: int = 3,
        early_stopping: bool = True,
        early_stopping_patience: int = 32,
        show_progress: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        return_metrics: bool = False,
    ) -> tuple[list[str], list[float]]:
        """
        Self-correction with progressive temperature scheduling:
        
        Args:
            temperature_schedule: "fixed" (use temperature param), "progressive" (linear), "exponential" (exp decay)
            temp_start: Starting temperature for progressive/exponential schedules
            temp_end: Ending temperature for progressive/exponential schedules
            
        Returns:
            corrected_texts: list of corrected samples
            self_accuracies: list of self-accuracy for each sample
        """
        
        def get_temperature_for_step(step, total_steps):
            """Calculate temperature for current step based on schedule"""
            if temperature_schedule == "fixed":
                return temperature
            elif temperature_schedule == "progressive":
                # Linear interpolation from temp_start to temp_end
                progress = step / max(total_steps - 1, 1)
                return temp_start + (temp_end - temp_start) * progress
            elif temperature_schedule == "exponential":
                # Exponential decay from temp_start to temp_end
                progress = step / max(total_steps - 1, 1)
                decay_factor = np.log(temp_end / temp_start)
                return temp_start * np.exp(decay_factor * progress)
            else:
                return temperature
        def _correction_step(model, tokenizer, z_t, t, temp, tokens_per_step=3):
            logits = model(z_t, t)
            logits[..., tokenizer.mask_token_id] = -1e6
            p_t = (logits / temp).softmax(-1)
            z_tm1 = sample_categorical(p_t)
            score = (z_tm1 != z_t) * p_t.gather(-1, z_tm1.unsqueeze(-1)).squeeze(-1)
            # Multi-token parallel correction: select top-k tokens to modify
            num_changes = min(tokens_per_step, (score > 0).sum().item())
            if num_changes > 0:
                ids = torch.topk(score, num_changes, dim=-1).indices
                z_tm1 = z_t.scatter(-1, ids, z_tm1.gather(-1, ids))
            else:
                z_tm1 = z_t  # No changes if no valid modifications
            acc = (z_tm1 == logits.argmax(-1)).float().mean().item()
            return z_tm1, acc

        device = next(self.model.parameters()).device
        z_ts = self.tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=self.config.max_seq_len)["input_ids"]
        corrected_zts = []
        self_accuracies = []
        
        # Log temperature schedule info
        if show_progress:
            print(f"Temperature schedule: {temperature_schedule}")
            if temperature_schedule != "fixed":
                print(f"Temperature range: {temp_start:.3f} → {temp_end:.3f}")
                # Show example temperatures at key steps
                example_steps = [0, num_inference_steps//4, num_inference_steps//2, 3*num_inference_steps//4, num_inference_steps-1]
                temp_examples = [get_temperature_for_step(step, num_inference_steps) for step in example_steps]
                print(f"Temperature examples: {[f'{t:.3f}' for t in temp_examples]}")
        
        with tqdm.tqdm(total=len(texts) * num_inference_steps, disable=not show_progress) as pbar:
            for z_t in z_ts:
                max_acc = 0
                curr_patience = 0
                z_t = z_t.unsqueeze(0).to(device)
                t = torch.full((z_t.shape[0],), device=device, fill_value=t0)
                logits = self.model(z_t, t)
                logits[..., self.tokenizer.mask_token_id] = -1e6
                for i in range(num_inference_steps):
                    # Calculate dynamic temperature for current step
                    current_temp = get_temperature_for_step(i, num_inference_steps)
                    with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
                        z_t_next, acc = _correction_step(self.model, self.tokenizer, z_t, t, current_temp, tokens_per_step)
                        if early_stopping:
                            if acc > max_acc:
                                max_acc = acc
                                curr_patience = 0
                            else:
                                curr_patience += 1
                                if curr_patience > early_stopping_patience:
                                    break
                            if (z_t == z_t_next).all():
                                break
                        z_t = z_t_next
                    pbar.update(1)
                corrected_zts.append(z_t)
                final_logits = self.model(z_t, t)
                final_argmax = final_logits.argmax(-1)
                self_acc = (z_t == final_argmax).float().mean().item()
                self_accuracies.append(self_acc)
            corrected_zts = torch.cat(corrected_zts, dim=0)
            corrected_samples = self.tokenizer.batch_decode(corrected_zts, skip_special_tokens=True)
            if return_metrics:
                return corrected_samples, self_accuracies
            else:
                return corrected_samples
