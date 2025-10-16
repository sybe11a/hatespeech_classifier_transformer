"""
train_full_ft.py
----------------
Fine-tune DistilBERT on HateXplain (full fine-tuning baseline).
"""

import numpy as np
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_from_disk
import torch
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef, cohen_kappa_score, f1_score

# ---------------------------------------------------------------------
# SEEDING + DEVICE
# ---------------------------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    acc = (preds == labels).mean()
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    
    mcc = matthews_corrcoef(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    
    # Optional: class-wise precision/recall/f1 for your report
    report = classification_report(labels, preds, target_names=["hate", "normal", "offensive"], digits=3)
    print("\n[CLASSIFICATION REPORT]\n", report)
    print("[CONFUSION MATRIX]\n", confusion_matrix(labels, preds))
    
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "mcc": mcc,
        "kappa": kappa
    }

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    # Load encoded dataset
    dataset = load_from_disk("data/encoded_hatexplain")

    model_name = "distilbert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Training setup
    args = TrainingArguments(
        output_dir="outputs/full_ft",
        evaluation_strategy="epoch",
        save_strategy="epoch",        # changed to match
        num_train_epochs=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir="outputs/logs",
    )


    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train + eval
    print("[INFO] Starting training...")
    trainer.train()  
    print("[INFO] Training complete.")

    # Save training logs
    metrics = trainer.state.log_history   # includes all loss/metric logs
    import json, os

    os.makedirs("outputs/history", exist_ok=True)
    with open("outputs/history/full_ft_logs.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("[INFO] Evaluating on test set...")
    results = trainer.evaluate(dataset["test"])
    print(results)

    trainer.save_model("outputs/full_ft/best_model")
    print("[INFO] Model saved to outputs/full_ft/best_model")

if __name__ == "__main__":
    main()
