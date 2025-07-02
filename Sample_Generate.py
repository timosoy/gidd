import torch
import json
import os
from gidd import GiddPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

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

# Check if generated_samples.txt exists
if os.path.exists("generated_samples.txt"):
    print("Loading existing generated samples from generated_samples.txt...")
    texts = load_samples_from_file("generated_samples.txt")
    print(f"Loaded {len(texts)} samples")
else:
    print("Generating new samples...")
    # load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = GiddPipeline.from_pretrained("dvruette/gidd-base-p_unif-0.2", trust_remote_code=True)
    pipe.to(device)

    # Generate Samples
    texts = pipe.generate(num_samples=4, num_inference_steps=128)

    # save the samples
    with open("generated_samples.txt", "w", encoding="utf-8") as f:
        for i, text in enumerate(texts):
            f.write(f"Sample {i+1}:\n{text}\n\n")

# Load model for self-correction and evaluation
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = GiddPipeline.from_pretrained("dvruette/gidd-base-p_unif-0.2", trust_remote_code=True)
pipe.to(device)

# do the self-correction (with oscillation avoidance)
corrected_texts = pipe.self_correction(texts, num_inference_steps=128, early_stopping=True, temperature=0.1)

# save the corrected version
with open("corrected_samples.txt", "w", encoding="utf-8") as f:
    for i, text in enumerate(corrected_texts):
        f.write(f"Corrected Sample {i+1}:\n{text}\n\n")

# compare the uncorrected and corrected version
with open("comparison.json", "w", encoding="utf-8") as f:
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

# =====================
# Quantitative Evaluation (PPL & Accuracy)
# =====================
def evaluate_texts(texts, model_name="gpt2", batch_size=4, max_length=512):
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
gen_metrics = evaluate_texts(texts)
print("Generated samples metrics:", json.dumps(gen_metrics, indent=2))
with open("generated_samples_metrics.json", "w", encoding="utf-8") as f:
    json.dump(gen_metrics, f, indent=2)

# Evaluate self-corrected samples
print("\nEvaluating self-corrected samples...")
corr_metrics = evaluate_texts(corrected_texts)
print("Self-corrected samples metrics:", json.dumps(corr_metrics, indent=2))
with open("corrected_samples_metrics.json", "w", encoding="utf-8") as f:
    json.dump(corr_metrics, f, indent=2) 