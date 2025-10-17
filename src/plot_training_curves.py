"""
plot_training_curves.py
-----------------------
Visualize training & evaluation metrics (loss and F1) for:
- Full fine-tuning
- LoRA fine-tuning

Works for both DistilBERT and BERTweet runs by changing LOGS below.
"""

import json
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DISTILBERT_LOGS = {
    "Full FT": "outputs/history/full_ft_logs.json",
    "LoRA": "outputs/history/lora_ft_logs.json",
}

BERTWEET_LOGS = {
    "Full FT": "outputs/history/bertweet_full_ft_logs.json",
    "LoRA": "outputs/history/bertweet_lora_ft_logs.json",
}

SAVE_DIR = "outputs/plots"
os.makedirs(SAVE_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# HELPER — extract metric history
# ---------------------------------------------------------------------
def extract_metrics(path):
    with open(path, "r") as f:
        logs = json.load(f)

    train_loss, eval_loss, f1_macro, epochs = [], [], [], []
    for entry in logs:
        if "loss" in entry and "epoch" in entry:
            train_loss.append(entry["loss"])
            epochs.append(entry["epoch"])
        if "eval_loss" in entry:
            eval_loss.append(entry["eval_loss"])
        if "eval_f1_macro" in entry:
            f1_macro.append(entry["eval_f1_macro"])
    return epochs, train_loss, eval_loss, f1_macro


# ---------------------------------------------------------------------
# PLOTTING FUNCTION
# ---------------------------------------------------------------------
def plot_training_curves(logs, title_prefix, save_name):
    """Generate training curves plot for a given set of logs."""
    plt.figure(figsize=(10, 5))

    # 1️⃣ Training vs Evaluation Loss
    plt.subplot(1, 2, 1)
    for name, path in logs.items():
        epochs, train_loss, eval_loss, _ = extract_metrics(path)

        # --- Aggregate training loss per epoch ---
        num_epochs = len(eval_loss)
        if num_epochs > 0:
            points_per_epoch = len(train_loss) // num_epochs
            train_loss_epoch = [
                np.mean(train_loss[i * points_per_epoch:(i + 1) * points_per_epoch])
                for i in range(num_epochs)
            ]
        else:
            train_loss_epoch = train_loss

        # --- Plot ---
        plt.plot(range(1, num_epochs + 1), train_loss_epoch,
                 linestyle="--", label=f"{name} – Train")
        plt.plot(range(1, num_epochs + 1), eval_loss,
                 marker="o", linewidth=2, label=f"{name} – Eval")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0.5, 1.0)
    plt.title(f"Training vs Evaluation Loss ({title_prefix})", fontsize=13)
    plt.grid(alpha=0.3)
    plt.legend()

    # 2️⃣ F1 Macro
    plt.subplot(1, 2, 2)
    for name, path in logs.items():
        _, _, _, f1_macro = extract_metrics(path)
        plt.plot(range(1, len(f1_macro) + 1), f1_macro,
                 marker='o', linewidth=2, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("F1 (Macro)")
    plt.ylim(0.5, 1.0)
    plt.title(f"F1 Macro over Epochs ({title_prefix})", fontsize=13)
    plt.grid(alpha=0.3)
    plt.legend()

    # Save
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"[SAVED] {title_prefix} plot exported to {save_path}")


# ---------------------------------------------------------------------
# GENERATE PLOTS
# ---------------------------------------------------------------------
# Plot for DistilBERT
plot_training_curves(DISTILBERT_LOGS, "DistilBERT", "training_curves_distilbert.png")

# Plot for BERTweet
plot_training_curves(BERTWEET_LOGS, "BERTweet", "training_curves_bertweet.png")
