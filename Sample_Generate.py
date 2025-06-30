import torch
import json
from gidd import GiddPipeline

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

# do the self-correction
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