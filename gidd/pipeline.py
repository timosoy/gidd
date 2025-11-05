import torch
import torch.nn as nn
import tqdm.auto as tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
import math

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
        sampling_temperature: float = None,  # Temperature for sampling replacement tokens
        sampling_temperature_schedule: str | None = None,  # None | "decrease" | "triangular"
        sampling_temperature_min: float | None = None,  # used by schedules
        sampling_temperature_max: float | None = None,  # used by schedules
        position_sampling_temperature: float = None,  # Temperature for stochastic position selection (NLL-based)
        t0: float = 0.01,
        tokens_per_step: int | None = None,
        selection_strategy: str = "nll",
        selection_mode: str = "topk",  # "topk", "threshold", "dyn_threshold", "stochastic"， "topk_masked". or "dyn_ratio"
        conf_threshold: float = 0.2,    # used when selection_mode == "threshold"
        dynamic_threshold_ratio: float = 0.7,  # used when selection_mode == "dyn_ratio"(0..1 of max NLL)
        early_stopping: bool = True,
        early_stopping_patience: int = 32,
        show_progress: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        return_metrics: bool = False,
        return_change_counts: bool = False,
    ) -> tuple[list[str], list[float]]:
        """
        Self-correction with metrics:
        Returns:
            corrected_texts: list of corrected samples
            self_accuracies: list of self-accuracy for each sample
        """
        def _correction_step(
            model,
            tokenizer,
            z_t,
            t,
            temp,
            sampling_temp,
            pos_temp,
            tokens_per_step=None,
            strategy: str = "nll",
            selection_mode: str = "topk",
            conf_threshold: float = 0.2,
            dyn_ratio: float = 0.7,
        ):
            logits = model(z_t, t)
            logits[..., tokenizer.mask_token_id] = -1e6
            # Base distribution for scoring/selection (untampered by temperature)
            p_base = torch.softmax(logits, dim=-1)
            
            # Sample replacement tokens with separate temperature for NLL-based approach
            if sampling_temp is not None:
                if sampling_temp == 0.0:
                    z_proposal = logits.argmax(-1)
                else:
                    p_sampling = (logits / sampling_temp).softmax(-1)
                    z_proposal = sample_categorical(p_sampling)
            else:
                # Default sampling uses base distribution
                z_proposal = sample_categorical(p_base)

            # Compute scores used by selection strategies
            current_token_probs = p_base.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)
            nll_scores = -torch.log(current_token_probs + 1e-9)

            # Select scoring strategy
            if strategy == "nll":
                score = nll_scores
            else:
                # Fallback to original confidence-based change score (use base probs)
                score = (z_proposal != z_t) * p_base.gather(-1, z_proposal.unsqueeze(-1)).squeeze(-1)

            # Low-confidence / stochastic selection modes
            if selection_mode == "threshold":
                # Use current-token probability as confidence; replace all below threshold
                low_mask = (current_token_probs < conf_threshold)
                if low_mask.any():
                    # Build indices tensor with shape [batch, num_changes]
                    idx = torch.nonzero(low_mask[0], as_tuple=False).squeeze(-1)
                    ids = idx.unsqueeze(0)
                    z_tm1 = z_t.scatter(-1, ids, z_proposal.gather(-1, ids))
                else:
                    # Fallback: change at least one lowest-confidence position
                    ids = torch.topk(-current_token_probs, k=min(1, z_t.size(-1)), dim=-1).indices
                    z_tm1 = z_t.scatter(-1, ids, z_proposal.gather(-1, ids))
            elif selection_mode == "dyn_threshold":
                # Robust threshold + upper clipping by current tokens_per_step (k)
                # Quantile method: with valid length L and target k, rho = k/L,
                # thr = quantile(nll_valid, q=1-rho). Select positions with nll >= thr,
                # then cap to top-k by NLL to handle ties.
                pad_id = getattr(tokenizer, 'pad_token_id', None)
                valid_mask = torch.ones_like(z_t, dtype=torch.bool)
                if pad_id is not None:
                    valid_mask = valid_mask & (z_t != pad_id)
                # Assume batch size == 1 here as upstream uses per-sample correction
                if z_t.size(0) != 1:
                    masked_scores = nll_scores.masked_fill(~valid_mask, float('-inf'))
                    if tokens_per_step is None or (isinstance(tokens_per_step, int) and tokens_per_step <= 0):
                        # Ratio-based selection across batch: nll >= dyn_ratio * max(nll)
                        max_vals, _ = torch.max(masked_scores, dim=-1, keepdim=True)
                        thr_vals = dyn_ratio * max_vals
                        sel = masked_scores >= thr_vals
                        # Fallback ensure at least one per row
                        any_sel = sel.any(dim=-1, keepdim=True)
                        top1 = torch.topk(masked_scores, k=1, dim=-1).indices
                        ids = torch.where(any_sel, torch.argmax(sel.to(torch.int), dim=-1, keepdim=True), top1)
                        z_tm1 = z_t.scatter(-1, ids, z_proposal.gather(-1, ids))
                    else:
                        # Quantile/top-k path
                        k = max(1, int(tokens_per_step))
                        k = min(k, z_t.size(-1))
                        ids = torch.topk(masked_scores, k, dim=-1).indices
                        z_tm1 = z_t.scatter(-1, ids, z_proposal.gather(-1, ids))
                else:
                    valid_idx = torch.nonzero(valid_mask[0], as_tuple=False).squeeze(-1)
                    L = int(valid_idx.numel())
                    if L == 0:
                        # No valid tokens (all pads) → fallback to global argmax
                        ids = nll_scores.argmax(dim=-1, keepdim=True)
                        z_tm1 = z_t.scatter(-1, ids, z_proposal.gather(-1, ids))
                    else:
                        nll_valid = nll_scores[0, valid_idx]
                        if tokens_per_step is None or (isinstance(tokens_per_step, int) and tokens_per_step <= 0):
                            # Ratio-based: select all with nll >= dyn_ratio * max(nll)
                            max_val = torch.max(nll_valid)
                            thr_val = dyn_ratio * max_val
                            sel_in_valid = nll_valid >= thr_val
                            cand_idx_in_valid = torch.nonzero(sel_in_valid, as_tuple=False).squeeze(-1)
                            if cand_idx_in_valid.numel() == 0:
                                cand_idx_in_valid = torch.topk(nll_valid, 1, dim=-1).indices
                            final_ids = valid_idx[cand_idx_in_valid]
                            ids = final_ids.unsqueeze(0)
                            z_tm1 = z_t.scatter(-1, ids, z_proposal.gather(-1, ids))
                        else:
                            k = max(1, int(tokens_per_step))
                            if k > L:
                                k = L
                            # Sort descending to obtain kth-largest as threshold
                            sorted_vals, _ = torch.sort(nll_valid, descending=True)
                            thr_val = sorted_vals[min(k - 1, L - 1)]
                            # Select all positions >= threshold (robust to ties)
                            sel_in_valid = nll_valid >= thr_val
                            cand_idx_in_valid = torch.nonzero(sel_in_valid, as_tuple=False).squeeze(-1)
                            if cand_idx_in_valid.numel() > k:
                                # Cap to top-k among candidates by actual NLL
                                cand_vals = nll_valid[cand_idx_in_valid]
                                topk_in_cand = torch.topk(cand_vals, k, dim=-1).indices
                                final_idx_in_valid = cand_idx_in_valid[topk_in_cand]
                            else:
                                final_idx_in_valid = cand_idx_in_valid
                            final_ids = valid_idx[final_idx_in_valid]
                            # Shape to [batch, num_changes]
                            ids = final_ids.unsqueeze(0)
                            z_tm1 = z_t.scatter(-1, ids, z_proposal.gather(-1, ids))
            elif selection_mode == "dyn_ratio":
                # Threshold at a ratio of max NLL: select positions with nll >= dyn_ratio * max(nll)
                pad_id = getattr(tokenizer, 'pad_token_id', None)
                valid_mask = torch.ones_like(z_t, dtype=torch.bool)
                if pad_id is not None:
                    valid_mask = valid_mask & (z_t != pad_id)
                if z_t.size(0) != 1:
                    # Fallback: vectorized masked top-k if unexpected batch; approximate by top-k
                    masked_scores = nll_scores.masked_fill(~valid_mask, float('-inf'))
                    if tokens_per_step is None or (isinstance(tokens_per_step, int) and tokens_per_step <= 0):
                        max_vals, _ = torch.max(masked_scores, dim=-1, keepdim=True)
                        thr_vals = dyn_ratio * max_vals
                        sel = masked_scores >= thr_vals
                        any_sel = sel.any(dim=-1, keepdim=True)
                        top1 = torch.topk(masked_scores, k=1, dim=-1).indices
                        ids = torch.where(any_sel, torch.argmax(sel.to(torch.int), dim=-1, keepdim=True), top1)
                        z_tm1 = z_t.scatter(-1, ids, z_proposal.gather(-1, ids))
                    else:
                        k = max(1, int(tokens_per_step))
                        k = min(k, z_t.size(-1))
                        ids = torch.topk(masked_scores, k, dim=-1).indices
                        z_tm1 = z_t.scatter(-1, ids, z_proposal.gather(-1, ids))
                else:
                    valid_idx = torch.nonzero(valid_mask[0], as_tuple=False).squeeze(-1)
                    L = int(valid_idx.numel())
                    if L == 0:
                        ids = nll_scores.argmax(dim=-1, keepdim=True)
                        z_tm1 = z_t.scatter(-1, ids, z_proposal.gather(-1, ids))
                    else:
                        nll_valid = nll_scores[0, valid_idx]
                        max_val = torch.max(nll_valid)
                        thr_val = dyn_ratio * max_val
                        sel_in_valid = nll_valid >= thr_val
                        cand_idx_in_valid = torch.nonzero(sel_in_valid, as_tuple=False).squeeze(-1)
                        if cand_idx_in_valid.numel() == 0:
                            # Ensure at least one change: take the max NLL
                            top1 = torch.topk(nll_valid, 1, dim=-1).indices
                            cand_idx_in_valid = top1
                        # Optional cap by tokens_per_step to avoid too many simultaneous edits
                        if tokens_per_step is None or (isinstance(tokens_per_step, int) and tokens_per_step <= 0):
                            final_idx_in_valid = cand_idx_in_valid
                        else:
                            k = max(1, int(tokens_per_step))
                            if cand_idx_in_valid.numel() > k:
                                cand_vals = nll_valid[cand_idx_in_valid]
                                topk_in_cand = torch.topk(cand_vals, k, dim=-1).indices
                                final_idx_in_valid = cand_idx_in_valid[topk_in_cand]
                            else:
                                final_idx_in_valid = cand_idx_in_valid
                        final_ids = valid_idx[final_idx_in_valid]
                        ids = final_ids.unsqueeze(0)
                        z_tm1 = z_t.scatter(-1, ids, z_proposal.gather(-1, ids))
            elif selection_mode == "stochastic":
                # Stochastic single-position selection using NLL as logits with temperature
                # Exclude pad positions from being selected
                pad_id = getattr(tokenizer, 'pad_token_id', None)
                valid_mask = torch.ones_like(z_t, dtype=torch.bool)
                if pad_id is not None:
                    valid_mask = valid_mask & (z_t != pad_id)
                # Build logits over positions; mask invalid
                if pos_temp is not None and pos_temp == 0.0:
                    pos_logits = nll_scores.masked_fill(~valid_mask, float('-inf'))
                    # Deterministic: pick highest NLL valid position
                    ids = pos_logits.argmax(dim=-1, keepdim=True)
                else:
                    temp_val = 1.0 if pos_temp is None else pos_temp
                    pos_logits = (nll_scores / temp_val).masked_fill(~valid_mask, float('-inf'))
                    pos_probs = torch.softmax(pos_logits, dim=-1)
                    # Fallback if all probs are NaN/zero on masked sequence: use argmax
                    if not torch.isfinite(pos_probs).any() or (pos_probs.sum(dim=-1) == 0).any():
                        ids = nll_scores.argmax(dim=-1, keepdim=True)
                    else:
                        sampled_idx = sample_categorical(pos_probs)
                        if sampled_idx.dim() == 1:
                            sampled_idx = sampled_idx.unsqueeze(-1)
                        ids = sampled_idx
                z_tm1 = z_t.scatter(-1, ids, z_proposal.gather(-1, ids))
            elif selection_mode == "topk_masked":
                # Same as default top-k, but exclude PAD positions from selection
                pad_id = getattr(tokenizer, 'pad_token_id', None)
                valid_mask = torch.ones_like(z_t, dtype=torch.bool)
                if pad_id is not None:
                    valid_mask = valid_mask & (z_t != pad_id)
                masked_score = score.masked_fill(~valid_mask, float('-inf'))
                k_val = 1 if (tokens_per_step is None or (isinstance(tokens_per_step, int) and tokens_per_step <= 0)) else int(tokens_per_step)
                num_changes = min(k_val, int((masked_score > 0).sum().item()))
                if num_changes > 0:
                    ids = torch.topk(masked_score, num_changes, dim=-1).indices
                    z_tm1 = z_t.scatter(-1, ids, z_proposal.gather(-1, ids))
                else:
                    z_tm1 = z_t
            else:
                # Default: top-k selection by score
                k_val = 1 if (tokens_per_step is None or (isinstance(tokens_per_step, int) and tokens_per_step <= 0)) else int(tokens_per_step)
                num_changes = min(k_val, int((score > 0).sum().item()))
                if num_changes > 0:
                    ids = torch.topk(score, num_changes, dim=-1).indices
                    z_tm1 = z_t.scatter(-1, ids, z_proposal.gather(-1, ids))
                else:
                    z_tm1 = z_t  # No changes if no valid modifications
            acc = (z_tm1 == logits.argmax(-1)).float().mean().item()
            return z_tm1, acc

        device = next(self.model.parameters()).device
        z_ts = self.tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=self.config.max_seq_len)["input_ids"]
        corrected_zts = []
        self_accuracies = []
        total_token_changes_list = []
        final_token_diff_list = []
        with tqdm.tqdm(total=len(texts) * num_inference_steps, disable=not show_progress) as pbar:
            for z_t in z_ts:
                max_acc = 0
                curr_patience = 0
                z_t = z_t.unsqueeze(0).to(device)
                z_t_orig = z_t.clone()
                per_sample_total_changes = 0
                t = torch.full((z_t.shape[0],), device=device, fill_value=t0)
                logits = self.model(z_t, t)
                logits[..., self.tokenizer.mask_token_id] = -1e6
                for i in range(num_inference_steps):
                    # Linear decay of tokens_per_step from initial value down to 1
                    if tokens_per_step is None:
                        current_k = None
                    else:
                        current_k = max(1, math.ceil(tokens_per_step * (1 - i / max(num_inference_steps, 1))))
                    # Compute per-step sampling temperature based on optional schedule
                    if sampling_temperature_schedule is None or sampling_temperature_schedule == "none":
                        current_sampling_temp = sampling_temperature
                    elif sampling_temperature_schedule == "decrease":
                        # Linear decrease from start (max) to end (min) over steps
                        steps_denom = max(num_inference_steps - 1, 1)
                        start_temp = sampling_temperature_max if sampling_temperature_max is not None else (
                            sampling_temperature if sampling_temperature is not None else 0.7
                        )
                        end_temp = sampling_temperature_min if sampling_temperature_min is not None else 0.0
                        alpha = i / steps_denom
                        current_sampling_temp = start_temp - (start_temp - end_temp) * alpha
                    elif sampling_temperature_schedule == "triangular":
                        # Single-cycle triangular: low -> high -> low across all steps
                        steps_denom = max(num_inference_steps - 1, 1)
                        low_temp = sampling_temperature_min if sampling_temperature_min is not None else 0.1
                        high_temp = sampling_temperature_max if sampling_temperature_max is not None else (
                            sampling_temperature if sampling_temperature is not None else 0.7
                        )
                        phase = i / steps_denom
                        if phase <= 0.5:
                            # rising edge
                            beta = phase / 0.5
                            current_sampling_temp = low_temp + (high_temp - low_temp) * beta
                        else:
                            # falling edge
                            beta = (phase - 0.5) / 0.5
                            current_sampling_temp = high_temp - (high_temp - low_temp) * beta
                    else:
                        # Unknown schedule; fallback to provided constant
                        current_sampling_temp = sampling_temperature
                    with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
                        z_t_next, acc = _correction_step(
                            self.model,
                            self.tokenizer,
                            z_t,
                            t,
                            temperature,
                            current_sampling_temp,
                            position_sampling_temperature,
                            tokens_per_step=current_k,
                            strategy=selection_strategy,
                            selection_mode=selection_mode,
                            conf_threshold=conf_threshold,
                            dyn_ratio=dynamic_threshold_ratio,
                        )
                        # Decide whether to commit this step and count changes accordingly
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
                        # Commit update and count token changes for this step
                        step_changes = int((z_t_next != z_t).sum().item())
                        per_sample_total_changes += step_changes
                        z_t = z_t_next
                    pbar.update(1)
                corrected_zts.append(z_t)
                final_logits = self.model(z_t, t)
                final_argmax = final_logits.argmax(-1)
                self_acc = (z_t == final_argmax).float().mean().item()
                self_accuracies.append(self_acc)
                # Final diff vs original tokens
                final_diff = int((z_t != z_t_orig).sum().item())
                total_token_changes_list.append(per_sample_total_changes)
                final_token_diff_list.append(final_diff)
            corrected_zts = torch.cat(corrected_zts, dim=0)
            corrected_samples = self.tokenizer.batch_decode(corrected_zts, skip_special_tokens=True)
            if return_metrics:
                if return_change_counts:
                    return corrected_samples, self_accuracies, total_token_changes_list, final_token_diff_list
                else:
                    return corrected_samples, self_accuracies
            else:
                return corrected_samples
