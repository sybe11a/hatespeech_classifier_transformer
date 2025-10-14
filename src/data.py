"""
data.py
-------
Utility script to load and preprocess the HateXplain dataset
for text classification (majority-vote version).

Usage:
    from data import load_hatexplain_dataset
    dataset = load_hatexplain_dataset()
"""

from collections import Counter
from datasets import load_dataset


def _apply_majority_vote(dataset):
    """
    For each example, derive the majority label among the annotators.
    Also join post_tokens into a single string under key 'text'.
    """
    def majority_label(example):
        labels = example["annotators"]["label"]  # updated: access dict of lists
        majority = Counter(labels).most_common(1)[0][0]
        example["label"] = majority
        example["text"] = " ".join(example["post_tokens"])
        return example

    dataset = dataset.map(majority_label)
    dataset = dataset.remove_columns(["annotators", "rationales", "post_tokens"])
    return dataset



def _check_class_balance(dataset):
    label_counts = Counter(dataset["train"]["label"])
    total = sum(label_counts.values())
    print("\n[INFO] Label distribution (training set):")
    for label_id, count in sorted(label_counts.items()):
        pct = count / total * 100
        label_name = {0: "hatespeech", 1: "normal", 2: "offensive"}.get(label_id, str(label_id))
        print(f"  {label_id} ({label_name}): {count:>6} ({pct:5.2f}%)")
    print("-" * 50)


def load_hatexplain_dataset():
    print("[INFO] Loading HateXplain dataset from Hugging Face Datasets hub...")
    dataset = load_dataset("Hate-speech-CNERG/hatexplain")
    dataset = _apply_majority_vote(dataset)
    _check_class_balance(dataset)
    print("[INFO] HateXplain dataset loaded successfully.")
    print(dataset)
    return dataset


if __name__ == "__main__":
    data = load_hatexplain_dataset()
    print("\nSample example:")
    print(data["train"][0])
