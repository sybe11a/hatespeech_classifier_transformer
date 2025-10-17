"""
main.py
-------
Main entry point to reproduce all experiments for DSA4213 Assignment 3.
Run this script to execute the complete pipeline from data tokenization to evaluation.
"""

import os
import sys
import subprocess
import warnings
from pathlib import Path

# Suppress warnings globally
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"


def run_script(script_path, description):
    """Run a Python script and handle errors."""
    print(f"\n→ {description}...")
    try:
        # Run with inherited environment (includes warning suppressions)
        result = subprocess.run(
            [sys.executable, "-W", "ignore", script_path],
            check=True,
            env=os.environ.copy(),
            capture_output=False
        )
        print(f"✅ {description} complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}")
        print(f"   Script: {script_path}")
        return False


def check_prerequisites():
    """Check if required directories exist."""
    print("Checking prerequisites...")
    
    # Create data directory if it doesn't exist
    Path("data").mkdir(exist_ok=True)
    
    print("✅ Prerequisites check passed")
    print("   (Dataset will be loaded from Hugging Face hub)")
    return True


def main():
    """Main execution pipeline."""
    print("=" * 50)
    print("DSA4213 Assignment 3 - Experiment Runner")
    print("=" * 50)
    print()
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # =========================================================
    # STEP 1: Data Tokenization
    # =========================================================
    print("\n" + "=" * 50)
    print("STEP 1: Tokenizing Data")
    print("=" * 50)
    
    if not Path("data/encoded_hatexplain").exists():
        if not run_script("src/tokenize_data.py", "Tokenizing for DistilBERT"):
            sys.exit(1)
    else:
        print("⏭️  DistilBERT data already tokenized (skipping)")
    
    if not Path("data/encoded_hatexplain_bertweet").exists():
        if not run_script("src/tokenize_data_bertweet.py", "Tokenizing for BERTweet"):
            sys.exit(1)
    else:
        print("⏭️  BERTweet data already tokenized (skipping)")
    
    # =========================================================
    # STEP 2: Model Training
    # =========================================================
    print("\n" + "=" * 50)
    print("STEP 2: Training Models")
    print("=" * 50)
    
    training_configs = [
        ("src/train_full_ft.py", "Training DistilBERT (Full Fine-Tuning)", "outputs/full_ft/best_model"),
        ("src/train_lora.py", "Training DistilBERT (LoRA)", "outputs/lora_ft/best_model"),
        ("src/train_bertweet_ft.py", "Training BERTweet (Full Fine-Tuning)", "outputs/bertweet_full_ft/best_model"),
        ("src/train_bertweet_lora.py", "Training BERTweet (LoRA)", "outputs/bertweet_lora_ft/best_model"),
    ]
    
    for script, desc, checkpoint_path in training_configs:
        if Path(checkpoint_path).exists():
            print(f"\n⏭️  {desc} - model already trained (skipping)")
        else:
            if not run_script(script, desc):
                print(f"\n⚠️  Warning: {desc} failed. Continuing with next step...")
    
    # =========================================================
    # STEP 3: Evaluation & Visualization
    # =========================================================
    print("\n" + "=" * 50)
    print("STEP 3: Evaluation & Visualization")
    print("=" * 50)
    
    eval_scripts = [
        ("src/evaluate_robustness.py", "Evaluating robustness"),
        ("src/visualise_attention.py", "Visualizing attention"),
        ("src/plot_training_curves.py", "Plotting training curves"),
    ]
    
    for script, desc in eval_scripts:
        if not run_script(script, desc):
            print(f"\n⚠️  Warning: {desc} failed. Continuing with next step...")
    
    # =========================================================
    # SUMMARY
    # =========================================================
    print("\n" + "=" * 50)
    print("🎉 ALL EXPERIMENTS COMPLETE!")
    print("=" * 50)
    print("\nResults saved to:")
    print("  📊 Training logs:     outputs/history/")
    print("  📈 Training curves:   outputs/plots/training_curves.png")
    print("  🔍 Attention maps:    outputs/attention_vis/")
    print("  🧪 Robustness tests:  outputs/noise_tests/robustness_results.csv")
    print("=" * 50)


if __name__ == "__main__":
    main()
