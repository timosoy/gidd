import torch
import json
import os
import logging
from datetime import datetime
from gidd.pipeline import GiddPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import torch.nn.functional as F
from gidd.loss import get_loss
from gidd.likelihood import ELBO, compute_elbo
from omegaconf import OmegaConf

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
    """Load samples from a text file"""
    samples = []
    current_sample = ""
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Sample ") and current_sample:
                samples.append(current_sample.strip())
                current_sample = ""
            elif not line.startswith("Sample ") and line.strip():
                current_sample += line
        if current_sample.strip():
            samples.append(current_sample.strip())
    return samples

def compute_self_surprisal(pipeline, texts, t_value=0.01, batch_size=4):
    """
    Compute self-surprisal using the GIDD model itself.
    
    Args:
        pipeline: GiddPipeline instance
        texts: List of text strings to evaluate
        t_value: Time value for diffusion model (lower = closer to clean data)
        batch_size: Batch size for processing
    
    Returns:
        dict: Contains per-sample and average self-surprisal metrics
    """
    device = next(pipeline.model.parameters()).device
    all_perplexities = []
    all_nlls = []
    
    print(f"Computing self-surprisal for {len(texts)} texts...")
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize the batch
            tokenized = pipeline.tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding="max_length", 
                truncation=True, 
                max_length=pipeline.config.max_seq_len
            )
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
            
            # Create time tensor - using small t_value for high quality evaluation
            batch_size_actual = input_ids.shape[0]
            t = torch.full((batch_size_actual,), fill_value=t_value, device=device)
            
            # Get model predictions
            logits = pipeline.model(input_ids, t)
            
            # Mask out the [MASK] token to prevent the model from predicting it
            logits[..., pipeline.tokenizer.mask_token_id] = -1e6
            
            # Compute cross-entropy loss for each position
            # Shift inputs: predict next token based on previous tokens
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_attention = attention_mask[..., :-1].contiguous()
            
            # Compute negative log-likelihood for each token
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1), 
                reduction='none'
            )
            loss = loss.view(shift_labels.shape)  # [batch_size, seq_len-1]
            
            # Compute per-sample metrics
            for j in range(batch_size_actual):
                sample_loss = loss[j]
                sample_attention = shift_attention[j]
                
                # Only consider non-padding tokens
                valid_tokens = sample_attention.sum().item()
                if valid_tokens > 0:
                    # Average NLL for this sample
                    sample_nll = (sample_loss * sample_attention).sum().item() / valid_tokens
                    sample_ppl = np.exp(sample_nll)
                    
                    all_nlls.append(sample_nll)
                    all_perplexities.append(sample_ppl)
    
    # Compute aggregate metrics
    avg_nll = np.mean(all_nlls)
    avg_ppl = np.mean(all_perplexities)
    median_ppl = np.median(all_perplexities)
    
    return {
        "per_sample_surprisals": all_perplexities,
        "per_sample_nlls": all_nlls,
        "average_nll": avg_nll,
        "average_surprisal": avg_ppl,
        "median_surprisal": median_ppl,
        "num_samples": len(all_perplexities)
    }

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

def compute_self_ppl_with_elbo(pipeline, texts, num_samples=32, t_eps=1e-4, batch_size=16):
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
            
            # Compute ELBO for this batch
            batch_metrics = compute_elbo(
                elbo_fn, 
                batch, 
                num_samples=num_samples, 
                t_eps=t_eps, 
                return_token_nlls=False, 
                reduce_metrics=False, 
                show_progress=False
            )
            
            all_metrics.append(batch_metrics)
    
    # Aggregate metrics across all batches
    # Simplified aggregation 
    avg_nll = np.mean([m["nll"].item() for m in all_metrics])
    avg_ppl = np.mean([m["ppl"].item() for m in all_metrics])
    avg_seq_nll = np.mean([m["seq_nll"].item() for m in all_metrics])
    
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

# Perform self-correction
logger.info(f"Starting self-correction on {len(texts)} samples")
logger.info("Self-correction parameters: num_inference_steps=128, early_stopping=True, temperature=0.1")
corrected_texts, self_accuracies = pipe.self_correction(
    texts, num_inference_steps=128, early_stopping=True, temperature=0.1, return_metrics=True
)
logger.info(f"Self-correction completed. Processed {len(corrected_texts)} samples")

# Save the corrected samples
corrected_samples_file = "Samples/corrected_samples.txt"
logger.info(f"Saving corrected samples to: {corrected_samples_file}")
with open(corrected_samples_file, "w", encoding="utf-8") as f:
    for i, text in enumerate(corrected_texts):
        f.write(f"Corrected Sample {i+1}:\n{text}\n\n")
logger.info(f"Corrected samples saved successfully")

# Compare the original and corrected samples
comparison_file = "Samples/comparison.json"
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

# Compute self-surprisal for original samples (simplified method)
print("\nComputing self-surprisal for generated samples (simplified method)...")
gen_self_surprisal = compute_self_surprisal(pipe, texts, t_value=0.01, batch_size=16)
print(f"Generated samples self-surprisal: {gen_self_surprisal['average_surprisal']:.2f}")

# Compute self-surprisal for corrected samples (simplified method)
print("\nComputing self-surprisal for corrected samples (simplified method)...")
corr_self_surprisal = compute_self_surprisal(pipe, corrected_texts, t_value=0.01, batch_size=16)
print(f"Corrected samples self-surprisal: {corr_self_surprisal['average_surprisal']:.2f}")

# Calculate improvement in self-surprisal (simplified method)
self_surprisal_improvement = gen_self_surprisal['average_surprisal'] - corr_self_surprisal['average_surprisal']
self_surprisal_improvement_ratio = corr_self_surprisal['average_surprisal'] / gen_self_surprisal['average_surprisal']
print(f"Self-surprisal improvement: {self_surprisal_improvement:.2f} (ratio: {self_surprisal_improvement_ratio:.3f})")

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

gen_metrics_file = "Samples/generated_samples_metrics.json"
logger.info(f"Saving generated samples metrics to: {gen_metrics_file}")
with open(gen_metrics_file, "w", encoding="utf-8") as f:
    json.dump({
        "external_metrics": gen_metrics,
        "self_surprisal_metrics": gen_self_surprisal,
        "self_ppl_metrics": gen_self_ppl
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

corr_metrics_file = "Samples/corrected_samples_metrics.json"
logger.info(f"Saving corrected samples metrics to: {corr_metrics_file}")
with open(corr_metrics_file, "w", encoding="utf-8") as f:
    json.dump({
        "external_metrics": corr_metrics,
        "self_accuracies": self_accuracies,
        "average_self_accuracy": avg_self_accuracy,
        "self_surprisal_metrics": corr_self_surprisal,
        "self_surprisal_improvement": {
            "absolute_improvement": self_surprisal_improvement,
            "improvement_ratio": self_surprisal_improvement_ratio
        },
        "self_ppl_metrics": corr_self_ppl,
        "self_ppl_improvement": {
            "absolute_improvement": self_ppl_improvement,
            "improvement_ratio": self_ppl_improvement_ratio
        }
    }, f, indent=2)
logger.info("Corrected samples metrics saved successfully")

# Log final summary

logger.info("=== Session Summary ===")
logger.info(f"Generated samples: {len(texts)}")
logger.info(f"Corrected samples: {len(corrected_texts)}")
logger.info(f"Generated PPL: {gen_metrics['ppl']:.2f}")
logger.info(f"Corrected PPL: {corr_metrics['ppl']:.2f}")
logger.info(f"PPL improvement: {gen_metrics['ppl'] - corr_metrics['ppl']:.2f}")
logger.info(f"Self-surprisal improvement: {self_surprisal_improvement:.2f}")
logger.info(f"Self-PPL improvement: {self_ppl_improvement:.2f}")
logger.info("=== Session completed successfully ===") 