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

# Create Samples directory if it doesn't exist
os.makedirs("Samples", exist_ok=True)
os.makedirs("Logs", exist_ok=True)

# Set up logging configuration
def setup_logging():
    """Setup logging to save to both console and file with timestamp"""
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"Logs/sample_generation_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()  # This keeps console output
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"=== Starting new sample generation session ===")
    logger.info(f"Log file: {log_filename}")
    return logger

# Initialize logging
logger = setup_logging()

# Override print function to also log to file
original_print = print
def print(*args, **kwargs):
    message = ' '.join(str(arg) for arg in args)
    logger.info(message)
    original_print(*args, **kwargs)

def load_samples_from_file(filename):
    """Load samples from a text file. Supports headers 'Sample N:' and 'Corrected Sample N:'"""
    samples = []
    current_sample = ""
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if (line.startswith("Sample ") or line.startswith("Corrected Sample ")) and current_sample:
                samples.append(current_sample.strip())
                current_sample = ""
            elif not (line.startswith("Sample ") or line.startswith("Corrected Sample ")) and line.strip():
                current_sample += line
        if current_sample.strip():
            samples.append(current_sample.strip())
    return samples

def compute_shannon_entropy(texts, name, tokenizer, max_length=512):
    """
    Compute Shannon entropy for text samples
    
    Args:
        texts: List of text strings
        name: Name for logging (e.g., "Generated", "Corrected")
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length for tokenization
    
    Returns:
        dict: Contains entropy per sequence, per token, and total tokens
    """
    print(f"\nComputing Shannon entropy for {name} samples...")
    
    # Tokenize all texts
    tokenized = tokenizer(
        texts, 
        return_tensors="pt", 
        padding="max_length", 
        truncation=True, 
        max_length=max_length
    )
    z_ts = tokenized["input_ids"]
    
    total_ent = 0
    total_token_ent = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i, z_t in enumerate(z_ts):
            if (i + 1) % 100 == 0 or (i == 0 and len(z_ts) > 1):
                print(f"Processing sample {i+1}/{len(z_ts)}...")
            
            counts = Counter(z_t.tolist())
            num_tokens = len(z_t)
            
            # Remove padding tokens
            if tokenizer.pad_token_id in counts:
                num_tokens -= counts[tokenizer.pad_token_id]
                del counts[tokenizer.pad_token_id]

            if len(counts) == 0:
                # entropy of current seq is 0
                continue
            
            # Calculate Shannon entropy using natural log
            prs = torch.tensor(list(counts.values()), dtype=torch.float32)
            prs = prs / prs.sum()
            ent = -torch.sum(prs * torch.log(prs))
            
            total_ent += ent.item()
            total_token_ent += ent.item() * num_tokens
            total_tokens += num_tokens
    
    ent_per_seq = total_ent / len(z_ts)
    ent_per_token = total_token_ent / total_tokens
    
    print(f"{name} entropy per sequence: {ent_per_seq:.4f}")
    print(f"{name} entropy per token: {ent_per_token:.4f}")
    print(f"Total tokens: {total_tokens}")
    print(f"Samples analyzed: {len(z_ts)}")
    
    return {
        "ent_per_seq": ent_per_seq,
        "ent_per_token": ent_per_token,
        "total_tokens": total_tokens,
        "samples_analyzed": len(z_ts)
    }


def compute_bleu_metrics(original_texts, corrected_texts):
    """Compute BLEU metrics comparing corrected texts against original references."""
    pair_count = min(len(original_texts), len(corrected_texts))
    if pair_count == 0:
        return {
            "pair_count": 0,
            "corrected_vs_original": None,
            "identity_original_vs_original": None,
        }

    trimmed_original = original_texts[:pair_count]
    trimmed_corrected = corrected_texts[:pair_count]
    references = [[ref] for ref in trimmed_original]

    corrected_metric = evaluate.load("bleu")
    corrected_vs_original = corrected_metric.compute(predictions=trimmed_corrected, references=references)

    identity_metric = evaluate.load("bleu")
    identity_original = identity_metric.compute(predictions=trimmed_original, references=references)

    return {
        "pair_count": pair_count,
        "corrected_vs_original": corrected_vs_original,
        "identity_original_vs_original": identity_original,
    }

 


def compute_self_accuracy_for_texts(pipeline, texts, t0=0.01, batch_size=16):
    """Compute per-sample self-accuracy for given texts without running correction."""
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

def generate_samples_in_batches(pipe, total_samples=1000, batch_size=16, num_inference_steps=128):
    """Generate samples in batches to avoid memory overflow"""
    all_texts = []
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    logger.info(f"Starting sample generation: {total_samples} samples in {num_batches} batches of {batch_size}")
    logger.info(f"Model device: {next(pipe.model.parameters()).device}")
    logger.info(f"Number of inference steps: {num_inference_steps}")
    
    print(f"Generating {total_samples} samples in {num_batches} batches of {batch_size}")
    
    for i in range(num_batches):
        current_batch_size = min(batch_size, total_samples - i * batch_size)
        logger.info(f"Processing batch {i+1}/{num_batches}: generating {current_batch_size} samples")
        print(f"Batch {i+1}/{num_batches}: generating {current_batch_size} samples...")
        
        try:
            batch_texts = pipe.generate(
                num_samples=current_batch_size, 
                num_inference_steps=num_inference_steps
            )
            all_texts.extend(batch_texts)
            logger.info(f"Batch {i+1} completed successfully, total samples so far: {len(all_texts)}")
        except Exception as e:
            logger.error(f"Error in batch {i+1}: {str(e)}")
            raise
        
        # Clear GPU cache between batches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    logger.info(f"Sample generation completed. Total samples generated: {len(all_texts)}")
    return all_texts

def compute_self_ppl_with_elbo(pipeline, texts, num_samples=32, t_eps=1e-4, batch_size=16, gpu_chunk_size=8):
    """
    Compute self-PPL using the theoretically correct ELBO method from likelihood.py
    
    Args:
        pipeline: GiddPipeline instance
        texts: List of text strings to evaluate
        num_samples: Number of time steps to sample for ELBO integration
        t_eps: Small epsilon for time bounds
        batch_size: Batch size for processing
    
    Returns:
        dict: Contains per-sample and average ELBO-based perplexity metrics
    """
    device = next(pipeline.model.parameters()).device
    
    # Load the real GIDD config file
    config_path = "gidd/gidd/configs/gidd.yaml"
    if os.path.exists(config_path):
        print(f"Loading real config from: {config_path}")
        # Load the complete config with all defaults resolved
        config = OmegaConf.load(config_path)
        print(f"Config loaded - loss_type: {config.loss.loss_type}, loss_weighting: {config.loss.loss_weighting}")
        
        # Copy necessary attributes from the pipeline config
        if hasattr(pipeline.config, 'max_seq_len'):
            config.max_seq_len = pipeline.config.max_seq_len
        if hasattr(pipeline.config, 'p_uniform'):
            config.model.p_uniform = pipeline.config.p_uniform
        if hasattr(pipeline.config, 't_eps'):
            config.model.t_eps = pipeline.config.t_eps
            
        # Get the loss function using the real config
        print("Creating loss function with real config...")
        loss_fn = get_loss(config, pipeline.tokenizer, pipeline.noise_schedule)
        print("Loss function created successfully")
    else:
        # Fallback to mock config if file not found
        print(f"Config file not found at {config_path}, using fallback mock config")
        from gidd.loss import GiddLoss
        
        class MockLossConfig:
            def __init__(self):
                self.loss_weighting = "uniform"
                self.min_loss_weight = 0.0
                self.max_loss_weight = 2.0
        
        class MockConfig:
            def __init__(self, real_config):
                for attr in ['max_seq_len', 'p_uniform', 't_eps']:
                    if hasattr(real_config, attr):
                        setattr(self, attr, getattr(real_config, attr))
                self.loss = MockLossConfig()
        
        config = MockConfig(pipeline.config)
        loss_fn = GiddLoss(config, pipeline.tokenizer, pipeline.noise_schedule)
    
    # Create ELBO function
    elbo_fn = ELBO(config, pipeline.model, pipeline.noise_schedule, loss_fn)
    
    all_metrics = []
    
    print(f"Computing ELBO-based self-PPL for {len(texts)} texts with {num_samples} time samples...")
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize the batch
            batch = pipeline.tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding="max_length", 
                truncation=True, 
                max_length=pipeline.config.max_seq_len
            )
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Further split the batch into smaller GPU chunks to reduce peak memory.
            # This does not change results, only memory footprint.
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            bs = input_ids.size(0)
            for j in range(0, bs, gpu_chunk_size):
                sub_batch = {
                    "input_ids": input_ids[j:j+gpu_chunk_size],
                    "attention_mask": attention_mask[j:j+gpu_chunk_size],
                }
                sub_metrics = compute_elbo(
                    elbo_fn,
                    sub_batch,
                    num_samples=num_samples,
                    t_eps=t_eps,
                    return_token_nlls=False,
                    reduce_metrics=False,
                    show_progress=False,
                )
                all_metrics.append(sub_metrics)
    
    # Aggregate metrics across all batches with proper weighting
    total_token_nll = float(sum(m["token_nll_sum"].item() for m in all_metrics))
    total_tokens = int(sum(m["token_count"].item() for m in all_metrics))
    total_sequences = int(sum(m["batch_size"].item() for m in all_metrics))

    # Token-weighted averages
    avg_nll = total_token_nll / max(total_tokens, 1)
    avg_ppl = float(np.exp(avg_nll))
    # Sequence-average NLL: total token NLL per sequence count
    avg_seq_nll = total_token_nll / max(total_sequences, 1)
    
    return {
        "average_nll": avg_nll,
        "average_perplexity": avg_ppl,
        "average_seq_nll": avg_seq_nll,
        "num_samples": len(texts),
        "num_time_samples": num_samples,
        "method": "ELBO"
    }


# Check if generated samples file exists
samples_file = "Samples/generated_samples.txt"
if os.path.exists(samples_file):
    logger.info(f"Found existing generated samples file: {samples_file}")
    print("Loading existing generated samples from file...")
    texts = load_samples_from_file(samples_file)
    logger.info(f"Successfully loaded {len(texts)} samples from file")
    print(f"Loaded {len(texts)} samples")
else:
    logger.info(f"No existing samples file found at {samples_file}, generating new samples")
    print("Generating new samples...")
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Selected device: {device}")
    
    # Load model and move to appropriate device
    logger.info("Loading model...")
    pipe = GiddPipeline.from_pretrained("dvruette/gidd-base-p_unif-0.2", trust_remote_code=True)
    if device != "cpu":
        logger.info(f"Moving model to {device}")
        pipe.to(device)
    
    model_device = next(pipe.model.parameters()).device
    logger.info(f"Model loaded on device: {model_device}")

    # Generate samples in batches
    texts = generate_samples_in_batches(pipe, total_samples=64, batch_size=32)

    # Save the samples
    logger.info(f"Saving generated samples to: {samples_file}")
    with open(samples_file, "w", encoding="utf-8") as f:
        for i, text in enumerate(texts):
            f.write(f"Sample {i+1}:\n{text}\n\n")
    logger.info(f"Generated samples saved successfully to {samples_file}")

# Load model for self-correction and evaluation
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Loading model for self-correction and evaluation on device: {device}")

# Load model and move to appropriate device
logger.info("Loading model...")
pipe = GiddPipeline.from_pretrained("dvruette/gidd-base-p_unif-0.2", trust_remote_code=True)
if device != "cpu":
    logger.info(f"Moving model to {device}")
    pipe.to(device)

model_device = next(pipe.model.parameters()).device
logger.info(f"Model loaded on device: {model_device}")

# Self-correction or reuse existing corrected samples
corrected_samples_file = "Samples/corrected_samples_1_steps.txt"
total_token_changes_list = None
final_token_diff_list = None
if os.path.exists(corrected_samples_file):
    logger.info(f"Found existing corrected samples at {corrected_samples_file}. Skipping self-correction and proceeding to metrics analysis.")
    corrected_texts = load_samples_from_file(corrected_samples_file)
    # Compute self-accuracies directly on corrected texts
    self_accuracies = compute_self_accuracy_for_texts(pipe, corrected_texts, t0=0.01, batch_size=16)
    logger.info(f"Loaded {len(corrected_texts)} corrected samples from file")
else:
    # Perform self-correction
    logger.info(f"Starting self-correction on {len(texts)} samples")
    logger.info("Self-correction parameters: num_inference_steps=128, early_stopping=True")
    corrected_texts, self_accuracies, total_token_changes_list, final_token_diff_list = pipe.self_correction(
        texts,
        num_inference_steps=1,
        early_stopping=True,
        temperature=0.1,
        return_metrics=True,
        return_change_counts=True,
    )
    logger.info(f"Self-correction completed. Processed {len(corrected_texts)} samples")

    # Save the corrected samples
    logger.info(f"Saving corrected samples to: {corrected_samples_file}")
    with open(corrected_samples_file, "w", encoding="utf-8") as f:
        for i, text in enumerate(corrected_texts):
            f.write(f"Corrected Sample {i+1}:\n{text}\n\n")
    logger.info(f"Corrected samples saved successfully")

# Compare the original and corrected samples
comparison_file = "Samples/comparison_1_steps.json"
logger.info(f"Saving comparison data to: {comparison_file}")
with open(comparison_file, "w", encoding="utf-8") as f:
    comparison = {
        "samples": [
            {
                "original": orig,
                "corrected": corr
            }
            for orig, corr in zip(texts, corrected_texts)
        ]
    }
    json.dump(comparison, f, ensure_ascii=False, indent=2)
logger.info(f"Comparison data saved successfully")

# =====================
# Quantitative Evaluation (PPL & Accuracy)
# =====================

 

# Compute self-PPL using ELBO method for original samples
print("\nComputing self-PPL for generated samples (ELBO method)...")
gen_self_ppl = compute_self_ppl_with_elbo(pipe, texts, num_samples=32, batch_size=16)
print(f"Generated samples self-PPL: {gen_self_ppl['average_perplexity']:.2f}")

# Compute self-PPL using ELBO method for corrected samples
print("\nComputing self-PPL for corrected samples (ELBO method)...")
corr_self_ppl = compute_self_ppl_with_elbo(pipe, corrected_texts, num_samples=32, batch_size=16)
print(f"Corrected samples self-PPL: {corr_self_ppl['average_perplexity']:.2f}")

# Calculate improvement in self-PPL (ELBO method)
self_ppl_improvement = gen_self_ppl['average_perplexity'] - corr_self_ppl['average_perplexity']
self_ppl_improvement_ratio = corr_self_ppl['average_perplexity'] / gen_self_ppl['average_perplexity']
print(f"Self-PPL improvement: {self_ppl_improvement:.2f} (ratio: {self_ppl_improvement_ratio:.3f})")


print("\nComputing BLEU scores for corrected samples against original samples...")
logger.info("Computing BLEU metrics for corrected samples")
bleu_metrics = compute_bleu_metrics(texts, corrected_texts)
bleu_main = bleu_metrics.get("corrected_vs_original") or {}
if bleu_main:
    print(f"Corrected vs original BLEU: {bleu_main.get('bleu', float('nan')):.6f}")
else:
    print("BLEU metrics unavailable (no overlapping samples).")

def evaluate_texts(texts, model_name="gpt2-large", batch_size=4, max_length=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.eval()
    total_nll = 0
    total_acc = 0
    total_tokens = 0
    all_nlls = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch = tokenizer(batch_texts, padding=True, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            attn_mask = batch["attention_mask"]
            logits = model(input_ids=batch["input_ids"], attention_mask=attn_mask, use_cache=False).logits[:, :-1]
            labels = batch["input_ids"][:, 1:]
            loss_mask = attn_mask[:, :-1]
            nll = torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(0, 1), reduction='none').view_as(labels)
            all_nlls.extend(nll[loss_mask == 1].cpu().numpy().tolist())
            total_nll += (nll * loss_mask).sum().item()
            acc = (logits.argmax(-1) == labels).float()
            total_acc += (acc * loss_mask).sum().item()
            total_tokens += loss_mask.sum().item()
    avg_nll = total_nll / total_tokens
    ppl = np.exp(avg_nll)
    accuracy = total_acc / total_tokens
    median_nll = float(np.median(all_nlls))
    return {
        "ppl": ppl,
        "accuracy": accuracy,
        "avg_nll": avg_nll, 
        "median_nll": median_nll,
        "tokens": total_tokens
    }

# Evaluate generated samples
print("\nEvaluating generated samples...")
logger.info("Starting external evaluation of generated samples")
gen_metrics = evaluate_texts(texts)
logger.info(f"Generated samples evaluation completed: PPL={gen_metrics['ppl']:.2f}, Accuracy={gen_metrics['accuracy']:.4f}")
print("Generated samples metrics:", json.dumps(gen_metrics, indent=2))

gen_metrics_file = "Samples/generated_samples_metrics_1_steps.json"
logger.info(f"Saving generated samples metrics to: {gen_metrics_file}")
with open(gen_metrics_file, "w", encoding="utf-8") as f:
    json.dump({
        "external_metrics": gen_metrics,
        "self_ppl_metrics": gen_self_ppl,
        "bleu": bleu_metrics
    }, f, indent=2)
logger.info("Generated samples metrics saved successfully")

# Evaluate self-corrected samples
print("\nEvaluating self-corrected samples...")
logger.info("Starting external evaluation of corrected samples")
corr_metrics = evaluate_texts(corrected_texts)
logger.info(f"Corrected samples evaluation completed: PPL={corr_metrics['ppl']:.2f}, Accuracy={corr_metrics['accuracy']:.4f}")
print("Self-corrected samples metrics:", json.dumps(corr_metrics, indent=2))

# Calculate average self_accuracy
avg_self_accuracy = np.mean(self_accuracies) if self_accuracies else 0.0
logger.info(f"Average self-accuracy calculated: {avg_self_accuracy:.4f}")
print(f"Average self_accuracy: {avg_self_accuracy:.4f}")

corr_metrics_file = "Samples/corrected_samples_metrics_1_steps.json"
logger.info(f"Saving corrected samples metrics to: {corr_metrics_file}")
with open(corr_metrics_file, "w", encoding="utf-8") as f:
    token_change_summary = None
    if (total_token_changes_list is not None) or (final_token_diff_list is not None):
        token_change_summary = {}
        if total_token_changes_list is not None and len(total_token_changes_list) > 0:
            t = np.array(total_token_changes_list, dtype=float)
            token_change_summary["total_token_changes_summary"] = {
                "count": int(t.size),
                "sum": float(t.sum()),
                "mean": float(t.mean()),
                "median": float(np.median(t)),
                "min": float(t.min()),
                "max": float(t.max()),
            }
        if final_token_diff_list is not None and len(final_token_diff_list) > 0:
            d = np.array(final_token_diff_list, dtype=float)
            token_change_summary["final_token_diffs_summary"] = {
                "count": int(d.size),
                "sum": float(d.sum()),
                "mean": float(d.mean()),
                "median": float(np.median(d)),
                "min": float(d.min()),
                "max": float(d.max()),
            }
    # Build dict in desired order: external -> self_accuracies -> average_self_accuracy -> token_change_summary -> others
    base_corr = {}
    base_corr["external_metrics"] = corr_metrics
    base_corr["self_accuracies"] = self_accuracies
    base_corr["average_self_accuracy"] = avg_self_accuracy
    if token_change_summary is not None:
        base_corr["token_change_summary"] = token_change_summary
    base_corr["bleu"] = bleu_metrics
    base_corr["self_ppl_metrics"] = corr_self_ppl
    base_corr["self_ppl_improvement"] = {
        "absolute_improvement": self_ppl_improvement,
        "improvement_ratio": self_ppl_improvement_ratio
    }
    if total_token_changes_list is not None:
        base_corr["total_token_changes"] = total_token_changes_list
    if final_token_diff_list is not None:
        base_corr["final_token_diffs"] = final_token_diff_list
    json.dump(base_corr, f, indent=2)
logger.info("Corrected samples metrics saved successfully")

# =====================
# Shannon Entropy Analysis
# =====================

print("\n" + "="*60)
print("Shannon Entropy Analysis")
print("="*60)

# We need a tokenizer for entropy analysis - use gpt2 as a standard
entropy_tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
if entropy_tokenizer.pad_token_id is None:
    entropy_tokenizer.pad_token = entropy_tokenizer.eos_token

# Compute Shannon entropy for generated samples
logger.info("Starting Shannon entropy analysis for generated samples")
gen_entropy = compute_shannon_entropy(texts, "Generated", entropy_tokenizer, max_length=512)

# Compute Shannon entropy for corrected samples
logger.info("Starting Shannon entropy analysis for corrected samples")
corr_entropy = compute_shannon_entropy(corrected_texts, "Corrected", entropy_tokenizer, max_length=512)

# Calculate entropy changes
seq_change = corr_entropy['ent_per_seq'] - gen_entropy['ent_per_seq']
token_change = corr_entropy['ent_per_token'] - gen_entropy['ent_per_token']

print(f"\nEntropy per sequence change: {seq_change:+.4f}")
print(f"Entropy per token change: {token_change:+.4f}")

# Integrate entropy and reorder keys for readability in metrics files
try:
    # Rebuild generated metrics with desired ordering
    if 'gen_metrics_file' in globals() and os.path.exists(gen_metrics_file):
        gen_file = {
            "external_metrics": gen_metrics,
            "self_ppl_metrics": gen_self_ppl,
            "self_ppl_improvement": {
                "absolute_improvement": self_ppl_improvement,
                "improvement_ratio": self_ppl_improvement_ratio
            },
            "bleu": bleu_metrics,
            "entropy_metrics": gen_entropy,
        }
        with open(gen_metrics_file, "w", encoding="utf-8") as f:
            json.dump(gen_file, f, indent=2)
        logger.info(f"Reordered and updated: {gen_metrics_file}")
    else:
        logger.warning("Generated metrics file not found when reordering; skipping.")

    # Rebuild corrected metrics with desired ordering
    if 'corr_metrics_file' in globals() and os.path.exists(corr_metrics_file):
        # Build token change summary to place after self-accuracy
        token_change_summary = None
        if (total_token_changes_list is not None) or (final_token_diff_list is not None):
            token_change_summary = {}
            if total_token_changes_list is not None and len(total_token_changes_list) > 0:
                t = np.array(total_token_changes_list, dtype=float)
                token_change_summary["total_token_changes_summary"] = {
                    "count": int(t.size),
                    "sum": float(t.sum()),
                    "mean": float(t.mean()),
                    "median": float(np.median(t)),
                    "min": float(t.min()),
                    "max": float(t.max()),
                }
            if final_token_diff_list is not None and len(final_token_diff_list) > 0:
                d = np.array(final_token_diff_list, dtype=float)
                token_change_summary["final_token_diffs_summary"] = {
                    "count": int(d.size),
                    "sum": float(d.sum()),
                    "mean": float(d.mean()),
                    "median": float(np.median(d)),
                    "min": float(d.min()),
                    "max": float(d.max()),
                }
        # Build dict in the desired order
        corr_file = {}
        corr_file["external_metrics"] = corr_metrics
        corr_file["self_ppl_metrics"] = corr_self_ppl
        corr_file["self_ppl_improvement"] = {
            "absolute_improvement": self_ppl_improvement,
            "improvement_ratio": self_ppl_improvement_ratio
        }
        corr_file["average_self_accuracy"] = avg_self_accuracy
        if token_change_summary is not None:
            corr_file["token_change_summary"] = token_change_summary
        corr_file["bleu"] = bleu_metrics
        corr_file["entropy_metrics"] = corr_entropy
        corr_file["entropy_improvements"] = {
            "seq_change": seq_change,
            "token_change": token_change,
            "generated_entropy": gen_entropy,
        }
        # Place per-sample arrays last
        corr_file["self_accuracies"] = self_accuracies
        if total_token_changes_list is not None:
            corr_file["total_token_changes"] = total_token_changes_list
        if final_token_diff_list is not None:
            corr_file["final_token_diffs"] = final_token_diff_list
        with open(corr_metrics_file, "w", encoding="utf-8") as f:
            json.dump(corr_file, f, indent=2)
        logger.info(f"Reordered and updated: {corr_metrics_file}")
    else:
        logger.warning("Corrected metrics file not found when reordering; skipping.")
except Exception as e:
    logger.error(f"Failed to reorder metrics files: {str(e)}")

# Log final summary

logger.info("=== Session Summary ===")
logger.info(f"Generated samples: {len(texts)}")
logger.info(f"Corrected samples: {len(corrected_texts)}")
logger.info(f"Generated PPL: {gen_metrics['ppl']:.2f}")
logger.info(f"Corrected PPL: {corr_metrics['ppl']:.2f}")
logger.info(f"PPL improvement: {gen_metrics['ppl'] - corr_metrics['ppl']:.2f}")
logger.info(f"Self-PPL improvement: {self_ppl_improvement:.2f}")
logger.info(f"Shannon entropy per sequence change: {seq_change:+.4f}")
logger.info(f"Shannon entropy per token change: {token_change:+.4f}")
logger.info("=== Session completed successfully ===") 
logger.info(f"BLEU corrected vs original: {bleu_main.get('bleu', float('nan')):.6f}")