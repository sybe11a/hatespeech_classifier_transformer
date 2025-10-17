"""
tokenize_data_bertweet.py
----------------
Tokenize and encode the HateXplain dataset using BERTweet tokenizer.
"""

from transformers import AutoTokenizer
from datasets import DatasetDict
from data import load_hatexplain_dataset

def encode_dataset(dataset: DatasetDict, model_name="vinai/bertweet-base", max_length=128):
    """
    Tokenizes and encodes the dataset for BERTweet.
    Args:
        dataset: DatasetDict from data.py
        model_name: name of the pretrained model
        max_length: truncation/padding length
    Returns:
        encoded DatasetDict
    """
    print(f"[INFO] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    print("[INFO] Tokenizing dataset...")
    encoded = dataset.map(tokenize_fn, batched=True)
    encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    print("[INFO] Tokenization complete.")
    return encoded


if __name__ == "__main__":
    dataset = load_hatexplain_dataset()
    encoded = encode_dataset(dataset)
    print(encoded)
    # optional: save for quick reload
    encoded.save_to_disk("data/encoded_hatexplain_bertweet")
    print("[INFO] Saved encoded dataset to data/encoded_hatexplain_bertweet/")
