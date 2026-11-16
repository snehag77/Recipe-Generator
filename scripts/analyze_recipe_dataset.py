# -*- coding: utf-8 -*-
"""Dataset Analysis and Token Length Profiling for GPT-2.

Analyzes character lengths, vocabulary diversity, and token distribution
to recommend optimal max_sequence_length and batch sizing for 16GB Unified RAM.

Usage:
    source venv_tf/bin/activate
    python scripts/analyze_recipe_dataset.py --dataset_dir data/processed
"""

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def analyze_dataset(dataset_dir: str):
    print("=" * 60)
    print("RecipeGen AI: Dataset Length & Token Distribution Analysis")
    print(f"Target: {dataset_dir}")
    print("=" * 60)

    train_path = os.path.join(dataset_dir, "recipes_train.jsonl")
    if not os.path.isfile(train_path):
        print(f"❌ Training split not found at: {train_path}")
        sys.exit(1)

    lengths = []
    titles = []
    ing_counts = []
    step_counts = []

    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            fmt_text = rec.get("formatted_text", "")
            lengths.append(len(fmt_text))
            titles.append(rec.get("title", ""))
            ing_counts.append(len(rec.get("ingredients", [])))
            step_counts.append(len(rec.get("directions", [])))

    total_records = len(lengths)
    if total_records == 0:
        print("❌ Dataset contains zero records.")
        sys.exit(1)

    avg_chars = sum(lengths) / total_records
    min_chars = min(lengths)
    max_chars = max(lengths)

    # Approximate token count (roughly 3.8 to 4 chars per BPE token in English)
    approx_tokens = [int(l / 3.8) for l in lengths]
    avg_tokens = sum(approx_tokens) / total_records

    # Percentiles
    sorted_tokens = sorted(approx_tokens)
    p50 = sorted_tokens[int(total_records * 0.50)]
    p90 = sorted_tokens[int(total_records * 0.90)]
    p95 = sorted_tokens[int(total_records * 0.95)]
    p99 = sorted_tokens[int(min(total_records - 1, int(total_records * 0.99)))]

    print(f"📊 Dataset Size Analyzed: {total_records} training recipes\n")
    print("📏 Character Metrics:")
    print(f"   - Average characters:  {avg_chars:.1f}")
    print(f"   - Min characters:      {min_chars}")
    print(f"   - Max characters:      {max_chars}\n")

    print("🔢 Token Distribution (Approx. BPE tokens):")
    print(f"   - Median (50th %ile):  {p50} tokens")
    print(f"   - 90th percentile:     {p90} tokens")
    print(f"   - 95th percentile:     {p95} tokens")
    print(f"   - 99th percentile:     {p99} tokens")
    print(f"   - Max tokens:          {max(approx_tokens)} tokens\n")

    print("🍳 Recipe Structural Characteristics:")
    print(f"   - Avg ingredients/recipe: {sum(ing_counts)/total_records:.1f}")
    print(f"   - Avg directions/recipe:  {sum(step_counts)/total_records:.1f}\n")

    print("💡 Recommendations for GPT-2 Fine-Tuning (16 GB Mac M4):")
    if p95 <= 256:
        rec_seq = 256
    elif p95 <= 384:
        rec_seq = 384
    else:
        rec_seq = 512

    print(f"   - Recommended Sequence Length:  {rec_seq} tokens (covers >=95% of recipes without truncation)")
    print("   - Recommended Batch Size:       4 to 8 (low memory footprint, fast throughput on Apple Silicon)")
    print("   - Recommended Precision:        float32 or mixed_float16")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Analyze Recipe Dataset Statistics")
    parser.add_argument(
        "--dataset_dir",
        default=str(PROJECT_ROOT / "data" / "processed"),
        help="Directory containing JSONL splits",
    )
    args = parser.parse_args()

    analyze_dataset(args.dataset_dir)


if __name__ == "__main__":
    main()
