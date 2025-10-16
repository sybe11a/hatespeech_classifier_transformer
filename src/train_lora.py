"""
train_lora.py
-------------
Parameter-efficient fine-tuning (LoRA) of DistilBERT on HateXplain.
Compares against full fine-tuning baseline.
"""

import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
)
from peft import LoraConfig, get_peft_model

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

    report = classification_report(labels, preds, target_names=["hate", "normal", "offensive"], digits=3)
    print("\n[CLASSIFICATION REPORT]\n", report)
    print("[CONFUSION MATRIX]\n", confusion_matrix(labels, preds))

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "mcc": mcc,
        "kappa": kappa,
    }


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    print("[INFO] Loading encoded HateXplain dataset...")
    dataset = load_from_disk("data/encoded_hatexplain")

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 1️⃣ Base model (frozen)
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    base_model.to(device)

    # 2️⃣ Apply LoRA configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_lin", "v_lin"],  # works for DistilBERT
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
    )

    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()  # sanity check: only small % trainable

    # Training setup
    args = TrainingArguments(
        output_dir="outputs/lora_ft",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-4,  # usually higher LR works better for LoRA
        weight_decay=0.01,
        logging_dir="outputs/logs_lora",
        logging_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        label_smoothing_factor=0.05,
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
    print("[INFO] Starting LoRA fine-tuning...")
    trainer.train()
    print("[INFO] LoRA fine-tuning complete.")

    # Save training logs
    metrics = trainer.state.log_history   # includes all loss/metric logs
    import json, os

    os.makedirs("outputs/history", exist_ok=True)
    with open("outputs/history/lora_ft_logs.json", "w") as f:
        json.dump(metrics, f, indent=2)


    print("[INFO] Evaluating on test set...")
    results = trainer.evaluate(dataset["test"])
    print(results)

    trainer.save_model("outputs/lora_ft/best_model")
    print("[INFO] LoRA model saved to outputs/lora_ft/best_model")


if __name__ == "__main__":
    main()
