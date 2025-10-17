# DSA4213 Assignment 3: Hate Speech Detection with Fine-Tuning

This repository implements and compares different fine-tuning strategies for hate speech detection using the HateXplain dataset.

## Overview

This project explores:
- **Full Fine-Tuning** vs **LoRA (Low-Rank Adaptation)** for parameter-efficient training
- **DistilBERT** vs **BERTweet** model comparison
- **Robustness evaluation** under noisy input conditions
- **Attention visualization** for model interpretability

## Quick Start

### 1. Prerequisites

- Python 3.11.9 (recommended for compatibility)
- GPU with CUDA support or Apple Silicon (MPS) recommended

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/sybe11a/sybella-dsa4213-assignment3.git
cd sybella-dsa4213-assignment3

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Data Preparation

The HateXplain dataset will be automatically downloaded from Hugging Face hub when you run the experiments. No manual download required!

The tokenized data will be cached in:
```bash
data/
  ├── encoded_hatexplain/       # DistilBERT tokenized data
  └── encoded_hatexplain_bertweet/  # BERTweet tokenized data
```

### 4. Run All Experiments

Use the provided Python script to reproduce all results:

```bash
python main.py
```

Or run individual steps manually (see [Manual Execution](#manual-execution) below).

## Project Structure

```
.
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── main.py                      # Main entry point to reproduce experiments
├── data/                        # Dataset directory (gitignored)
│   ├── encoded_hatexplain/      # Tokenized DistilBERT data
│   └── encoded_hatexplain_bertweet/  # Tokenized BERTweet data
├── src/                         # Source code
│   ├── data.py                  # Data loading utilities
│   ├── tokenize_data.py         # DistilBERT tokenization
│   ├── tokenize_data_bertweet.py  # BERTweet tokenization
│   ├── train_full_ft.py         # Full fine-tuning (DistilBERT)
│   ├── train_lora.py            # LoRA fine-tuning (DistilBERT)
│   ├── train_bertweet_ft.py     # Full fine-tuning (BERTweet)
│   ├── train_bertweet_lora.py   # LoRA fine-tuning (BERTweet)
│   ├── evaluate_robustness.py   # Robustness testing
│   ├── visualise_attention.py   # Attention visualization
│   └── plot_training_curves.py  # Training metrics plotting
└── outputs/                     # Results directory
    ├── attention_vis/           # Attention heatmaps
    ├── history/                 # Training logs (JSON)
    ├── noise_tests/             # Robustness evaluation results
    └── plots/                   # Training curve visualizations
```

## Manual Execution

If you prefer to run experiments step-by-step:

### Step 1: Tokenize Data

```bash
# For DistilBERT
python src/tokenize_data.py

# For BERTweet
python src/tokenize_data_bertweet.py
```

### Step 2: Train Models

```bash
# DistilBERT - Full Fine-Tuning
python src/train_full_ft.py

# DistilBERT - LoRA
python src/train_lora.py

# BERTweet - Full Fine-Tuning
python src/train_bertweet_ft.py

# BERTweet - LoRA
python src/train_bertweet_lora.py
```

### Step 3: Evaluate and Visualize

```bash
# Robustness evaluation
python src/evaluate_robustness.py

# Attention visualization
python src/visualise_attention.py

# Plot training curves
python src/plot_training_curves.py
```

## Results

After running the experiments, you'll find:

- **Training logs**: `outputs/history/*.json`
- **Training curves**: `outputs/plots/training_curves.png`
- **Attention visualizations**: `outputs/attention_vis/*.png`
- **Robustness results**: `outputs/noise_tests/robustness_results.csv`

## Notes

- Training times vary based on hardware (GPU/MPS/CPU)
- Model checkpoints are gitignored due to size constraints
- Encoded datasets are gitignored; run tokenization scripts to regenerate

## Troubleshooting

**Issue**: CUDA out of memory
- Reduce batch size in training scripts (default: 16)

**Issue**: MPS backend errors on Mac
- Set `device = "cpu"` in training scripts

**Issue**: Missing dataset
- Ensure `data/dataset.json` exists before tokenization

## References

- HateXplain Dataset: [Paper](https://arxiv.org/abs/2012.10289)
- LoRA: [Paper](https://arxiv.org/abs/2106.09685)
- DistilBERT: [Paper](https://arxiv.org/abs/1910.01108)
- BERTweet: [Paper](https://arxiv.org/abs/2005.10200)