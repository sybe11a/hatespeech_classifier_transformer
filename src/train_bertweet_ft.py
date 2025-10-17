"""
train_bertweet_ft.py
--------------------
Fine-tune BERTweet-base on HateXplain (full fine-tuning baseline).
"""

import os, json, numpy as np, torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
)

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
    dataset = load_from_disk("data/encoded_hatexplain_bertweet")

    # Drop token_type_ids if present (RoBERTa/BERTweet doesn’t use them)
    for split in dataset.keys():
        if "token_type_ids" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns(["token_type_ids"])

    model_name = "vinai/bertweet-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.to(device)

    args = TrainingArguments(
        output_dir="outputs/bertweet_full_ft",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir="outputs/logs_bertweet_ft",
        logging_steps=500,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    data_collator = DataCollatorWithPadding(tokenizer, padding="max_length", max_length=128, return_tensors="pt")
    trainer.data_collator = data_collator

    print("[INFO] Starting BERTweet full fine-tuning...")
    trainer.train()
    print("[INFO] Training complete.")

    # save logs
    metrics = trainer.state.log_history
    os.makedirs("outputs/history", exist_ok=True)
    with open("outputs/history/bertweet_full_ft_logs.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("[INFO] Evaluating on test set...")
    results = trainer.evaluate(dataset["test"])
    print(results)

    trainer.save_model("outputs/bertweet_full_ft/best_model")
    print("[INFO] Model saved to outputs/bertweet_full_ft/best_model")


if __name__ == "__main__":
    main()
