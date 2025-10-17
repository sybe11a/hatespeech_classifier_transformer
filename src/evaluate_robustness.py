"""
evaluate_robustness.py
----------------------
Evaluates robustness of Full Fine-Tuning vs LoRA models on HateXplain
under lexical noise (typos, slang, casing changes).
"""

import os
import csv
import random
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
FULL_FT_PATH = "outputs/full_ft/best_model"
LORA_PATH = "outputs/lora_ft/best_model"
MODEL_NAME = "distilbert-base-uncased"
LABELS = ["hate", "normal", "offensive"]

# Save path now inside outputs/noise_tests/
SAVE_DIR = "outputs/noise_tests"
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_PATH = os.path.join(SAVE_DIR, "robustness_results.csv")

device = "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ---------------------------------------------------------------------
# TEST SAMPLES
# ---------------------------------------------------------------------
SAMPLES = [
    # HATE (targeted exclusion / blame)
    "Muslims are ruining this country.",      

    # OFFENSIVE (insults / generalized rudeness)
    "Stupid ugly people shouldn’t be allowed on television.",                                       

    # NORMAL (clean baselines)
    "Everyone deserves to be treated with respect, even when we disagree.",      
]


# ---------------------------------------------------------------------
# NOISE FUNCTIONS
# ---------------------------------------------------------------------
def random_typo(word):
    """Randomly swaps adjacent characters in a word (25% chance)."""
    if len(word) > 3 and random.random() < 0.25:
        i = random.randint(0, len(word) - 2)
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word


def add_noise(sentence):
    """Applies typos, slang replacements, casing, and occasional word drops."""
    words = sentence.split()
    noisy_words = [random_typo(w) for w in words]
    noisy = " ".join(noisy_words)

    # Simple corruption patterns
    noisy = noisy.lower()
    noisy = noisy.replace("people", "ppl").replace("hate", "h8").replace("great", "gr8")

    # Randomly drop one word
    if len(noisy_words) > 3 and random.random() < 0.3:
        idx = random.randint(0, len(noisy_words) - 1)
        noisy_words.pop(idx)
        noisy = " ".join(noisy_words)

    return noisy


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def predict(model, text):
    """Returns predicted label for input text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
    return LABELS[int(np.argmax(probs))]


def load_model(model_path, is_lora=False):
    """Loads full fine-tuned or LoRA-adapted model."""
    if is_lora:
        base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    random.seed(42)

    print("[INFO] Loading models...")
    full_model = load_model(FULL_FT_PATH, is_lora=False)
    lora_model = load_model(LORA_PATH, is_lora=True)

    rows = []
    for model_name, model in [("Full FT", full_model), ("LoRA", lora_model)]:
        print(f"\n[INFO] Evaluating {model_name}...")
        for text in SAMPLES:
            noisy = add_noise(text)
            clean_pred = predict(model, text)
            noisy_pred = predict(model, noisy)
            changed = "Yes" if clean_pred != noisy_pred else "No"

            rows.append({
                "Model": model_name,
                "Original Text": text,
                "Noisy Text": noisy,
                "Clean Prediction": clean_pred,
                "Noisy Prediction": noisy_pred,
                "Changed?": changed
            })

            print(f"→ {model_name:<6} | Clean: {clean_pred:<9} | Noisy: {noisy_pred:<9} | Changed: {changed}")

    # Save results
    with open(SAVE_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nRobustness evaluation complete. Results saved to: {SAVE_PATH}")


if __name__ == "__main__":
    main()
