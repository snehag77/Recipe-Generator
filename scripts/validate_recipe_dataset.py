# -*- coding: utf-8 -*-
"""Dataset Validation Suite for RecipeGen AI.

Verifies:
1. JSONL files exist and are non-empty.
2. Required fields ('recipe_id', 'ingredients', 'title', 'directions', 'formatted_text') exist.
3. Formatted recipe text complies with delimiters (<INGREDIENTS>, <END_ING>, <RECIPE>, <TITLE>, <QUANTITIES>, <STEPS>, </RECIPE>).
4. No cross-split duplicate leakage between Train, Validation, and Test sets.
5. No empty ingredient lists or empty directions.
6. Statistics report matches actual line counts.
"""

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Dict, List, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FIELDS = ["recipe_id", "title", "ingredients", "directions", "formatted_text"]
REQUIRED_TAGS = ["<INGREDIENTS>", "<END_ING>", "<RECIPE>", "<TITLE>", "<QUANTITIES>", "<STEPS>", "</RECIPE>"]


class SplitResults:
    def __init__(self):
        self.count = 0
        self.fingerprints = set()
        self.errors = []


def run_validation(dataset_dir: str) -> bool:
    print("=" * 60)
    print("RecipeGen AI: Dataset Validation & Quality Check")
    print(f"Directory: {dataset_dir}")
    print("=" * 60)

    splits = ["recipes_train.jsonl", "recipes_validation.jsonl", "recipes_test.jsonl"]
    split_data: Dict[str, SplitResults] = {}

    all_passed = True

    for s in splits:
        path = os.path.join(dataset_dir, s)
        res = SplitResults()
        split_data[s] = res

        if not os.path.isfile(path):
            print(f"❌ Missing expected split file: {s}")
            all_passed = False
            continue

        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                res.count += 1
                try:
                    record = json.loads(line)
                except Exception as e:
                    res.errors.append(f"Line {line_no}: Malformed JSON ({e})")
                    continue

                # Field check
                for rf in REQUIRED_FIELDS:
                    if rf not in record or not record[rf]:
                        res.errors.append(f"Line {line_no}: Missing required field '{rf}'")

                # Delimiter check
                fmt = record.get("formatted_text", "")
                for tag in REQUIRED_TAGS:
                    if tag not in fmt:
                        res.errors.append(f"Line {line_no}: Missing tag '{tag}' in formatted_text")

                # Fingerprint check
                title = record.get("title", "")
                ings = record.get("ingredients", [])
                fp = f"{title.lower()}|{sorted([str(i).lower() for i in ings[:4]])}"
                res.fingerprints.add(fp)

        if res.errors:
            print(f"❌ Split '{s}': {len(res.errors)} errors encountered (showing first 3):")
            for err in res.errors[:3]:
                print(f"   - {err}")
            all_passed = False
        else:
            print(f"✅ Split '{s}': {res.count} records verified successfully.")

    # Cross-split duplicate check
    print("\n🔍 Checking for cross-split duplicate leakage...")
    train_fps = split_data.get("recipes_train.jsonl", SplitResults()).fingerprints
    val_fps = split_data.get("recipes_validation.jsonl", SplitResults()).fingerprints
    test_fps = split_data.get("recipes_test.jsonl", SplitResults()).fingerprints

    train_val_leak = train_fps.intersection(val_fps)
    train_test_leak = train_fps.intersection(test_fps)
    val_test_leak = val_fps.intersection(test_fps)

    if train_val_leak:
        print(f"❌ Cross-split leak detected between Train and Validation: {len(train_val_leak)} duplicates!")
        all_passed = False
    if train_test_leak:
        print(f"❌ Cross-split leak detected between Train and Test: {len(train_test_leak)} duplicates!")
        all_passed = False
    if val_test_leak:
        print(f"❌ Cross-split leak detected between Validation and Test: {len(val_test_leak)} duplicates!")
        all_passed = False

    if not train_val_leak and not train_test_leak and not val_test_leak:
        print("✅ Zero cross-split leakage! All splits are strictly disjoint.")

    # Validate dataset_statistics.json
    stats_path = os.path.join(dataset_dir, "dataset_statistics.json")
    if os.path.isfile(stats_path):
        with open(stats_path, "r", encoding="utf-8") as sf:
            stats = json.load(sf)
            if stats.get("train_count") != split_data.get("recipes_train.jsonl", SplitResults()).count:
                print("❌ Statistics mismatch for train_count")
                all_passed = False
            else:
                print("✅ Statistics record matches actual line counts.")
    else:
        print("⚠️ dataset_statistics.json not found in output directory.")

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL VALIDATION CHECKS PASSED: Dataset is clean, structured, and leak-free!")
    else:
        print("❌ Dataset validation encountered failures.")
    print("=" * 60)
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Validate Prepared Recipe Dataset")
    parser.add_argument(
        "--dataset_dir",
        default=str(PROJECT_ROOT / "data" / "processed"),
        help="Directory containing JSONL splits",
    )
    args = parser.parse_args()

    success = run_validation(args.dataset_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
