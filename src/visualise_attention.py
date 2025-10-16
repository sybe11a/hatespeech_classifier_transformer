"""
visualise_attention.py
----------------------
Compares attention patterns and predictions between Full Fine-Tuning and LoRA fine-tuning
for DistilBERT on HateXplain. 
Now excludes special tokens ([CLS], [SEP], [PAD]) for cleaner visualization.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
FULL_FT_PATH = "outputs/full_ft/best_model"
LORA_PATH = "outputs/lora_ft/best_model"
MODEL_NAME = "distilbert-base-uncased"
LABELS = ["hate", "normal", "offensive"]

SAMPLES = [
    "Immigrants are ruining this country", # hate
    "Foreigners are ruining this country", # hate
    "Muslims are ruining this country", # hate
    "Stupid ugly people shouldn’t be allowed on television.", # offensive
    "Gay people deserve equal rights", # normal (tests false positives)
    "Gays are ruining this country", # hate (can compare against ^)
    "I don’t hate these people at all", # normal (negation; tests whether model understands “don’t hate”)
]

SAVE_DIR = "outputs/attention_vis"
os.makedirs(SAVE_DIR, exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ---------------------------------------------------------------------
# HELPER — Get attention and prediction
# ---------------------------------------------------------------------
def get_attention_and_pred(model_path, text, tokenizer, is_lora=False):
    """Returns (tokens, attention_weights, predicted_label) excluding special tokens."""
    if is_lora:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=3, output_attentions=True
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, output_attentions=True
        )

    model.to(device)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # Prediction
    probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
    pred_label = LABELS[int(np.argmax(probs))]

    # Attention (last layer, averaged across heads)
    attn = outputs.attentions[-1][0].mean(dim=0).cpu().numpy()  # (seq_len, seq_len)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    token_scores = attn[0] / attn[0].sum()

    # Filter out [CLS], [SEP], [PAD]
    filtered = [(t, s) for t, s in zip(tokens, token_scores) if t not in ["[CLS]", "[SEP]", "[PAD]"]]
    tokens, token_scores = zip(*filtered)
    tokens = list(tokens)
    token_scores = np.array(token_scores)
    token_scores /= token_scores.sum()  # re-normalize after removing specials

    return tokens, token_scores, pred_label


# ---------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------
def main():
    for i, text in enumerate(SAMPLES, 1):
        print(f"\n[INFO] Processing sample {i}/{len(SAMPLES)}: {text}")

        tokens_full, scores_full, label_full = get_attention_and_pred(FULL_FT_PATH, text, tokenizer, is_lora=False)
        tokens_lora, scores_lora, label_lora = get_attention_and_pred(LORA_PATH, text, tokenizer, is_lora=True)

        min_len = min(len(scores_full), len(scores_lora))
        tokens = tokens_full[:min_len]
        scores_full = scores_full[:min_len]
        scores_lora = scores_lora[:min_len]

        plt.figure(figsize=(10, 3))
        x = np.arange(len(tokens))
        plt.bar(x - 0.2, scores_full, width=0.4, label=f"Full FT ({label_full})", alpha=0.8)
        plt.bar(x + 0.2, scores_lora, width=0.4, label=f"LoRA ({label_lora})", alpha=0.8)
        plt.xticks(x, tokens, rotation=45, ha="right")
        plt.ylabel("Normalized attention weight")
        plt.title(f"Example {i}: {text[:50]}...", fontsize=10)
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(SAVE_DIR, f"attention_case{i}.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"[SAVED] {save_path} | Full FT: {label_full} | LoRA: {label_lora}")

    print("\n All attention visualizations saved!")


if __name__ == "__main__":
    main()
