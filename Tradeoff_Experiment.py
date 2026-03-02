"""
=============================================================================
Trade-off Experiment: Generation Steps vs Self-Correction Steps
=============================================================================

Research Question: 
    Given a fixed compute budget, how should we allocate steps between 
    generation and self-correction to maximize quality?

Usage:
    python Tradeoff_Experiment.py

Output:
    - Samples: Tradeoff_Samples/
    - Metrics: Tradeoff_Samples_Metrics/
    
=============================================================================
"""

# Add correct path for gidd package
import sys
import os
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Add the gidd subdirectory to Python path (so it finds E:\GIDD\gidd\gidd)
GIDD_PATH = os.path.join(SCRIPT_DIR, "gidd")
if GIDD_PATH not in sys.path:
    sys.path.insert(0, GIDD_PATH)

import torch
import json
import os
import logging
from datetime import datetime
from collections import Counter
from gidd.pipeline import GiddPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import torch.nn.functional as F
from gidd.loss import get_loss
from gidd.likelihood import ELBO, compute_elbo
from omegaconf import OmegaConf
import evaluate
import pandas as pd

# =============================================================================
# CONFIGURATION - MODIFY HERE
# =============================================================================

CONFIG = {
    # ----- Sample Settings -----
    "num_samples": 64,           # Samples per configuration
    "batch_size": 32,            # Generation batch size (reduce if OOM)
    
    # ----- Self-Correction Settings -----
    "corr_algorithm": "nll",     # "nll" or "conf" (confidence-based)
    "corr_temp": 0.1,            # Sampling temperature for correction
    "corr_tokens": 8,            # Tokens to modify per step
    "selection_mode": "topk",    # "topk", "dyn_threshold", etc.
    
    # ----- Output Directories -----
    "samples_dir": "Tradeoff_Samples",
    "metrics_dir": "Tradeoff_Samples_Metrics",
    
    # =========================================================================
    # EXPERIMENT CONFIGURATIONS - MODIFY TO CHANGE ALLOCATIONS
    # Format: (generation_steps, correction_steps)
    # =========================================================================
    "experiments": [
        # ===== Budget = 64 total steps =====
        (64, 0),      # Pure generation, no correction
        (48, 16),     # Light correction
        (32, 32),     # Balanced
        (16, 48),     # More correction
        (8, 56),      # Heavy correction
        
        # ===== Budget = 128 total steps =====
        (128, 0),     # Pure generation
        (96, 32),     # Light correction
        (64, 64),     # Balanced
        (32, 96),     # More correction
        (16, 112),    # Heavy correction
        
        # ===== Budget = 256 total steps =====
        (256, 0),     # Pure generation
        (192, 64),    # Light correction
        (128, 128),   # Balanced
        (64, 192),    # More correction
        (32, 224),    # Heavy correction
    ],
}

# =============================================================================
# LOGGING SETUP
# =============================================================================

os.makedirs("Logs", exist_ok=True)

def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"Logs/tradeoff_experiment_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"=== Trade-off Experiment Started ===")
    logger.info(f"Log file: {log_filename}")
    return logger

logger = setup_logging()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_samples_in_batches(pipe, total_samples, batch_size, num_inference_steps):
    """Generate samples in batches to avoid memory overflow."""
    all_texts = []
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        current_batch_size = min(batch_size, total_samples - i * batch_size)
        logger.info(f"  Generating batch {i+1}/{num_batches} ({current_batch_size} samples)...")
        
        batch_texts = pipe.generate(
            num_samples=current_batch_size,
            num_inference_steps=num_inference_steps,
            show_progress=False
        )
        all_texts.extend(batch_texts)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return all_texts


def compute_self_accuracy(pipeline, texts, t0=0.01, batch_size=16):
    """Compute per-sample self-accuracy for given texts."""
    device = next(pipeline.model.parameters()).device
    mask_id = pipeline.tokenizer.mask_token_id
    accuracies = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            tokenized = pipeline.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=pipeline.config.max_seq_len,
            )
            input_ids = tokenized["input_ids"].to(device)
            t = torch.full((input_ids.shape[0],), device=device, fill_value=t0)
            logits = pipeline.model(input_ids, t)
            logits[..., mask_id] = -1e6
            argmax_ids = logits.argmax(-1)
            batch_acc = (argmax_ids == input_ids).float().mean(dim=1).tolist()
            accuracies.extend(batch_acc)
    
    return accuracies


def compute_self_ppl_with_elbo(pipeline, texts, num_samples=32, t_eps=1e-4, batch_size=16, gpu_chunk_size=8):
    """Compute self-PPL using the ELBO method."""
    device = next(pipeline.model.parameters()).device
    
    config_path = "gidd/gidd/configs/gidd.yaml"
    if os.path.exists(config_path):
        config = OmegaConf.load(config_path)
        if hasattr(pipeline.config, 'max_seq_len'):
            config.max_seq_len = pipeline.config.max_seq_len
        if hasattr(pipeline.config, 'p_uniform'):
            config.model.p_uniform = pipeline.config.p_uniform
        if hasattr(pipeline.config, 't_eps'):
            config.model.t_eps = pipeline.config.t_eps
        loss_fn = get_loss(config, pipeline.tokenizer, pipeline.noise_schedule)
    else:
        from gidd.loss import GiddLoss
        class MockLossConfig:
            loss_weighting = "uniform"
            min_loss_weight = 0.0
            max_loss_weight = 2.0
        class MockConfig:
            def __init__(self, real_config):
                for attr in ['max_seq_len', 'p_uniform', 't_eps']:
                    if hasattr(real_config, attr):
                        setattr(self, attr, getattr(real_config, attr))
                self.loss = MockLossConfig()
        config = MockConfig(pipeline.config)
        loss_fn = GiddLoss(config, pipeline.tokenizer, pipeline.noise_schedule)
    
    elbo_fn = ELBO(config, pipeline.model, pipeline.noise_schedule, loss_fn)
    all_metrics = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch = pipeline.tokenizer(
                batch_texts, return_tensors="pt", padding="max_length",
                truncation=True, max_length=pipeline.config.max_seq_len
            )
            batch = {k: v.to(device) for k, v in batch.items()}
            
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            bs = input_ids.size(0)
            
            for j in range(0, bs, gpu_chunk_size):
                sub_batch = {
                    "input_ids": input_ids[j:j+gpu_chunk_size],
                    "attention_mask": attention_mask[j:j+gpu_chunk_size],
                }
                sub_metrics = compute_elbo(
                    elbo_fn, sub_batch, num_samples=num_samples,
                    t_eps=t_eps, return_token_nlls=False,
                    reduce_metrics=False, show_progress=False,
                )
                all_metrics.append(sub_metrics)
    
    total_token_nll = float(sum(m["token_nll_sum"].item() for m in all_metrics))
    total_tokens = int(sum(m["token_count"].item() for m in all_metrics))
    
    avg_nll = total_token_nll / max(total_tokens, 1)
    avg_ppl = float(np.exp(avg_nll))
    
    return {"average_nll": avg_nll, "average_perplexity": avg_ppl}


def evaluate_with_gpt2(texts, model_name="gpt2-large", batch_size=8, max_length=512):
    """Evaluate texts using GPT-2 as external LM."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    
    total_nll = 0
    total_acc = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch = tokenizer(batch_texts, padding=True, return_tensors="pt",
                             truncation=True, max_length=max_length).to(device)
            attn_mask = batch["attention_mask"]
            logits = model(input_ids=batch["input_ids"], attention_mask=attn_mask,
                          use_cache=False).logits[:, :-1]
            labels = batch["input_ids"][:, 1:]
            loss_mask = attn_mask[:, :-1]
            nll = F.cross_entropy(logits.flatten(0, 1), labels.flatten(0, 1),
                                 reduction='none').view_as(labels)
            total_nll += (nll * loss_mask).sum().item()
            acc = (logits.argmax(-1) == labels).float()
            total_acc += (acc * loss_mask).sum().item()
            total_tokens += loss_mask.sum().item()
    
    del model
    torch.cuda.empty_cache()
    
    avg_nll = total_nll / total_tokens
    return {
        "ppl": np.exp(avg_nll),
        "accuracy": total_acc / total_tokens,
        "avg_nll": avg_nll
    }


def compute_bleu(original_texts, corrected_texts):
    """Compute BLEU score between original and corrected texts."""
    if len(original_texts) == 0 or len(corrected_texts) == 0:
        return 1.0
    
    pair_count = min(len(original_texts), len(corrected_texts))
    references = [[ref] for ref in original_texts[:pair_count]]
    predictions = corrected_texts[:pair_count]
    
    bleu = evaluate.load("bleu")
    result = bleu.compute(predictions=predictions, references=references)
    return result.get("bleu", 0.0)


def compute_shannon_entropy(texts, tokenizer, max_length=512):
    """Compute Shannon entropy for text samples (same as Sample_Generate.py)."""
    tokenized = tokenizer(
        texts, return_tensors="pt", padding="max_length",
        truncation=True, max_length=max_length
    )
    z_ts = tokenized["input_ids"]
    
    total_ent = 0
    total_token_ent = 0
    total_tokens = 0
    
    for z_t in z_ts:
        counts = Counter(z_t.tolist())
        num_tokens = len(z_t)
        
        if tokenizer.pad_token_id in counts:
            num_tokens -= counts[tokenizer.pad_token_id]
            del counts[tokenizer.pad_token_id]
        
        if len(counts) == 0:
            continue
        
        prs = torch.tensor(list(counts.values()), dtype=torch.float32)
        prs = prs / prs.sum()
        ent = -torch.sum(prs * torch.log(prs))
        
        total_ent += ent.item()
        total_token_ent += ent.item() * num_tokens
        total_tokens += num_tokens
    
    return {
        "ent_per_seq": total_ent / len(z_ts) if len(z_ts) > 0 else 0.0,
        "ent_per_token": total_token_ent / total_tokens if total_tokens > 0 else 0.0,
        "total_tokens": total_tokens,
        "samples_analyzed": len(z_ts)
    }


# =============================================================================
# MAIN EXPERIMENT FUNCTION
# =============================================================================

def run_single_experiment(pipe, gen_steps, corr_steps, entropy_tokenizer):
    """Run a single experiment configuration."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Configuration: Gen={gen_steps}, Corr={corr_steps}, Total={gen_steps+corr_steps}")
    logger.info(f"{'='*60}")
    
    # Step 1: Generate samples
    logger.info(f"[1/5] Generating {CONFIG['num_samples']} samples with {gen_steps} steps...")
    generated_texts = generate_samples_in_batches(
        pipe, CONFIG["num_samples"], CONFIG["batch_size"], gen_steps
    )
    
    # Step 2: Apply self-correction (if corr_steps > 0)
    total_changes = None
    final_diffs = None
    
    if corr_steps > 0:
        logger.info(f"[2/5] Applying self-correction with {corr_steps} steps...")
        final_texts, self_accuracies, total_changes, final_diffs = pipe.self_correction(
            generated_texts,
            num_inference_steps=corr_steps,
            sampling_temperature=CONFIG["corr_temp"],
            tokens_per_step=CONFIG["corr_tokens"],
            selection_strategy=CONFIG["corr_algorithm"],
            selection_mode=CONFIG["selection_mode"],
            early_stopping=True,
            return_metrics=True,
            return_change_counts=True,
            show_progress=True
        )
    else:
        logger.info("[2/5] No correction (corr_steps=0)")
        final_texts = generated_texts
        self_accuracies = compute_self_accuracy(pipe, generated_texts)
    
    # Step 3: Compute internal metrics
    logger.info("[3/5] Computing internal metrics (self-PPL)...")
    self_ppl = compute_self_ppl_with_elbo(pipe, final_texts, num_samples=32, batch_size=16)
    avg_self_accuracy = np.mean(self_accuracies)
    
    # Step 4: Compute external metrics
    logger.info("[4/5] Computing external metrics (GPT-2 PPL)...")
    ext_metrics = evaluate_with_gpt2(final_texts, batch_size=8)
    
    # Step 5: Compute similarity and entropy
    logger.info("[5/5] Computing BLEU and entropy...")
    bleu_score = compute_bleu(generated_texts, final_texts) if corr_steps > 0 else 1.0
    entropy_result = compute_shannon_entropy(final_texts, entropy_tokenizer)
    
    # Build results
    results = {
        "config": {
            "gen_steps": gen_steps,
            "corr_steps": corr_steps,
            "total_steps": gen_steps + corr_steps,
            "gen_ratio": gen_steps / (gen_steps + corr_steps) if (gen_steps + corr_steps) > 0 else 1.0,
            "corr_algorithm": CONFIG["corr_algorithm"] if corr_steps > 0 else "none",
            "corr_temp": CONFIG["corr_temp"],
            "corr_tokens": CONFIG["corr_tokens"],
            "num_samples": CONFIG["num_samples"],
        },
        "internal_metrics": {
            "self_ppl": self_ppl["average_perplexity"],
            "self_nll": self_ppl["average_nll"],
            "avg_self_accuracy": avg_self_accuracy,
        },
        "external_metrics": {
            "generative_ppl": ext_metrics["ppl"],
            "generative_accuracy": ext_metrics["accuracy"],
            "generative_nll": ext_metrics["avg_nll"],
        },
        "similarity_metrics": {
            "bleu_vs_generated": bleu_score,
        },
        "entropy_metrics": {
            "entropy_per_token": entropy_result["ent_per_token"],
            "entropy_per_seq": entropy_result["ent_per_seq"],
        },
    }
    
    if total_changes is not None:
        results["token_metrics"] = {
            "avg_total_changes": float(np.mean(total_changes)),
            "avg_final_diffs": float(np.mean(final_diffs)),
        }
    
    logger.info(f"Results: Gen_PPL={ext_metrics['ppl']:.2f}, Self_PPL={self_ppl['average_perplexity']:.2f}, "
                f"Self_Acc={avg_self_accuracy:.4f}, BLEU={bleu_score:.4f}")
    
    return results, generated_texts, final_texts


def main():
    """Main experiment runner."""
    
    # Create output directories
    os.makedirs(CONFIG["samples_dir"], exist_ok=True)
    os.makedirs(CONFIG["metrics_dir"], exist_ok=True)
    
    logger.info("\n" + "="*80)
    logger.info("TRADE-OFF EXPERIMENT: Generation Steps vs Self-Correction Steps")
    logger.info("="*80)
    logger.info(f"Samples output: {CONFIG['samples_dir']}/")
    logger.info(f"Metrics output: {CONFIG['metrics_dir']}/")
    logger.info(f"Configurations to test: {len(CONFIG['experiments'])}")
    logger.info(f"Samples per config: {CONFIG['num_samples']}")
    logger.info(f"Correction algorithm: {CONFIG['corr_algorithm']}")
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"\nLoading GIDD model on {device}...")
    pipe = GiddPipeline.from_pretrained("dvruette/gidd-base-p_unif-0.2", trust_remote_code=True)
    if device != "cpu":
        pipe.to(device)
    logger.info("Model loaded successfully!")
    
    # Load entropy tokenizer
    entropy_tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    if entropy_tokenizer.pad_token_id is None:
        entropy_tokenizer.pad_token = entropy_tokenizer.eos_token
    
    # Run experiments
    all_results = []
    summary_rows = []
    
    for idx, (gen_steps, corr_steps) in enumerate(CONFIG["experiments"]):
        logger.info(f"\n[Experiment {idx+1}/{len(CONFIG['experiments'])}]")
        
        try:
            results, gen_texts, final_texts = run_single_experiment(
                pipe, gen_steps, corr_steps, entropy_tokenizer
            )
            
            # Save samples
            exp_id = f"gen{gen_steps}_corr{corr_steps}"
            
            gen_file = os.path.join(CONFIG["samples_dir"], f"{exp_id}_generated.txt")
            with open(gen_file, "w", encoding="utf-8") as f:
                for i, text in enumerate(gen_texts):
                    f.write(f"Sample {i+1}:\n{text}\n\n")
            
            final_file = os.path.join(CONFIG["samples_dir"], f"{exp_id}_final.txt")
            with open(final_file, "w", encoding="utf-8") as f:
                for i, text in enumerate(final_texts):
                    f.write(f"Sample {i+1}:\n{text}\n\n")
            
            # Save detailed metrics
            metrics_file = os.path.join(CONFIG["metrics_dir"], f"{exp_id}_metrics.json")
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            
            # Add to summary
            all_results.append(results)
            
            row = {
                "exp_id": exp_id,
                "gen_steps": gen_steps,
                "corr_steps": corr_steps,
                "total_steps": gen_steps + corr_steps,
                "gen_ratio": results["config"]["gen_ratio"],
                "self_ppl": results["internal_metrics"]["self_ppl"],
                "self_accuracy": results["internal_metrics"]["avg_self_accuracy"],
                "generative_ppl": results["external_metrics"]["generative_ppl"],
                "generative_accuracy": results["external_metrics"]["generative_accuracy"],
                "bleu": results["similarity_metrics"]["bleu_vs_generated"],
                "entropy_per_token": results["entropy_metrics"]["entropy_per_token"],
                "entropy_per_seq": results["entropy_metrics"]["entropy_per_seq"],
            }
            if "token_metrics" in results:
                row["avg_token_changes"] = results["token_metrics"]["avg_total_changes"]
                row["avg_final_diffs"] = results["token_metrics"]["avg_final_diffs"]
            summary_rows.append(row)
            
            # Save intermediate summary
            pd.DataFrame(summary_rows).to_csv(
                os.path.join(CONFIG["metrics_dir"], "tradeoff_summary.csv"), index=False
            )
            
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Final summary
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(CONFIG["metrics_dir"], "tradeoff_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    
    with open(os.path.join(CONFIG["metrics_dir"], "tradeoff_all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print best configurations
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT COMPLETE - BEST CONFIGURATIONS BY BUDGET")
    logger.info("="*80)
    
    for budget in sorted(summary_df["total_steps"].unique()):
        budget_df = summary_df[summary_df["total_steps"] == budget]
        if len(budget_df) > 0:
            best = budget_df.loc[budget_df["generative_ppl"].idxmin()]
            logger.info(f"\nBudget {int(budget)}: Best = Gen={int(best['gen_steps'])}, Corr={int(best['corr_steps'])}")
            logger.info(f"  Generative PPL: {best['generative_ppl']:.2f}")
            logger.info(f"  Self PPL: {best['self_ppl']:.2f}")
            logger.info(f"  Self Accuracy: {best['self_accuracy']:.4f}")
    
    logger.info(f"\nSummary saved to: {summary_csv}")
    return summary_df


if __name__ == "__main__":
    main()

