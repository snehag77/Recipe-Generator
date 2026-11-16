# -*- coding: utf-8 -*-
"""Recipe Dataset Preparation Pipeline for GPT-2 Fine-Tuning.

Features:
- Reproducible Reservoir Sampling (seed 42) across the full dataset to prevent ordering/source bias.
- Safe AST literal parsing (ast.literal_eval, zero eval()).
- Comprehensive deduplication before splitting.
- Strict 90:5:5 train/validation/test disjoint split (zero leakage).
- Preservation of authentic ingredient quantities and directions (zero fabrication).
- Full accounting of rows read, rejected, deduplicated, and retained.

Usage:
    source venv_tf/bin/activate
    python scripts/prepare_recipe_dataset.py --input "data/archive 2/RecipeNLG_dataset.csv" --max_recipes 5000
"""

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_SEED = 42


def parse_list_field(field_val: Any) -> List[str]:
    """Parse stringified Python lists safely using ast.literal_eval (never eval)."""
    if not field_val:
        return []
    if isinstance(field_val, list):
        return [str(x).strip() for x in field_val if str(x).strip()]

    val_str = str(field_val).strip()
    if val_str.startswith("[") and val_str.endswith("]"):
        try:
            parsed = ast.literal_eval(val_str)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except (ValueError, SyntaxError):
            pass
        # Fallback safe regex extraction of quoted string items
        items = re.findall(r"['\"](.*?)['\"]", val_str)
        if items:
            return [i.strip() for i in items if i.strip()]

    # Fallback to newline/semicolon/comma split
    return [x.strip() for x in re.split(r"[;\n,]", val_str) if x.strip()]


def clean_text(text: Any) -> str:
    """Normalize whitespace and strip illegal characters."""
    if not text:
        return ""
    cleaned = re.sub(r"[\r\t\f\v]+", " ", str(text))
    cleaned = re.sub(r" +", " ", cleaned)
    return cleaned.strip()


def validate_recipe_record(record: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate completeness and quality of a recipe record."""
    title = clean_text(record.get("title", ""))
    if not title or len(title) < 2:
        return False, "missing_or_short_title"

    ings_raw = parse_list_field(record.get("ingredients", []))
    ings_ner = parse_list_field(record.get("NER", []))
    if not ings_raw and not ings_ner:
        return False, "missing_ingredients"

    dirs = parse_list_field(record.get("directions", []))
    if not dirs or len(dirs) == 0:
        return False, "missing_directions"

    # Reject directions that are trivially short (< 15 chars total)
    total_dir_len = sum(len(d) for d in dirs)
    if total_dir_len < 15:
        return False, "directions_too_short"

    # Reject corrupted, excessively long directions (> 4000 chars)
    if total_dir_len > 4000:
        return False, "excessively_long_directions"

    return True, None


def compute_recipe_fingerprint(title: str, ingredients: List[str]) -> str:
    """Compute deduplication fingerprint based on normalized title and ingredients."""
    norm_title = re.sub(r"[^a-z0-9]", "", title.lower())
    norm_ings = sorted(re.sub(r"[^a-z0-9]", "", i.lower()) for i in ingredients[:5])
    raw = f"{norm_title}|{'|'.join(norm_ings)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def format_recipe_for_gpt2(
    title: str,
    ingredients_ner: List[str],
    ingredients_raw: List[str],
    directions: List[str],
) -> str:
    """Format recipe into standard delimiter structure for GPT-2 causal modeling.

    Structure:
    <INGREDIENTS>
    ingredient 1
    ingredient 2
    <END_ING>
    <RECIPE>
    <TITLE>
    Recipe Title
    </TITLE>
    <QUANTITIES>
    - 2 cups flour
    - 1 tsp salt
    </QUANTITIES>
    <STEPS>
    1. First direction.
    2. Second direction.
    </STEPS>
    </RECIPE>
    """
    # 1. Ingredient names (NER or raw names)
    ing_names = [clean_text(i) for i in ingredients_ner if clean_text(i)]
    if not ing_names:
        ing_names = [clean_text(i) for i in ingredients_raw if clean_text(i)]
    ing_names_block = "\n".join(ing_names)

    # 2. Quantities / detailed ingredients
    quantities = [clean_text(q) for q in ingredients_raw if clean_text(q)]
    quantities_block = "\n".join([f"- {q}" for q in quantities])

    # 3. Steps
    cleaned_steps = []
    for idx, step in enumerate(directions, 1):
        s = clean_text(step)
        if s:
            s_clean = re.sub(r"^\d+[\.\)]\s*", "", s)
            cleaned_steps.append(f"{idx}. {s_clean}")
    steps_block = "\n".join(cleaned_steps)

    formatted = (
        f"<INGREDIENTS>\n{ing_names_block}\n<END_ING>\n"
        f"<RECIPE>\n"
        f"<TITLE>\n{clean_text(title)}\n</TITLE>\n"
        f"<QUANTITIES>\n{quantities_block}\n</QUANTITIES>\n"
        f"<STEPS>\n{steps_block}\n</STEPS>\n"
        f"</RECIPE>"
    )
    return formatted


def process_dataset_with_reservoir_sampling(
    input_path: str,
    output_dir: str,
    target_sample_size: int = 5000,
    scan_limit: Optional[int] = None,
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
):
    """Clean, deduplicate, and perform reproducible reservoir sampling across the dataset."""
    os.makedirs(output_dir, exist_ok=True)
    random.seed(RANDOM_SEED)

    stats = {
        "total_rows_read": 0,
        "valid_candidate_count": 0,
        "duplicates_removed": 0,
        "discarded_reasons": {},
        "target_sample_size": target_sample_size,
        "final_retained_count": 0,
        "train_count": 0,
        "val_count": 0,
        "test_count": 0,
    }

    seen_fingerprints = set()
    # Reservoir will hold up to target_sample_size records
    reservoir: List[Dict[str, Any]] = []

    print(f"📖 Streaming dataset from: {input_path}")
    print(f"🎯 Target reservoir sample size: {target_sample_size} (Random seed: {RANDOM_SEED})")

    # Increase CSV field size limit for large recipe direction fields
    csv.field_size_limit(min(sys.maxsize, 2147483646))

    with open(input_path, mode="r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["total_rows_read"] += 1

            # Validate record
            is_valid, reason = validate_recipe_record(row)
            if not is_valid:
                stats["discarded_reasons"][reason] = stats["discarded_reasons"].get(reason, 0) + 1
                continue

            title = clean_text(row.get("title", ""))
            ings_raw = parse_list_field(row.get("ingredients", []))
            ings_ner = parse_list_field(row.get("NER", []))
            dirs = parse_list_field(row.get("directions", []))

            # Deduplication before reservoir insertion
            fp = compute_recipe_fingerprint(title, ings_ner or ings_raw)
            if fp in seen_fingerprints:
                stats["duplicates_removed"] += 1
                continue
            seen_fingerprints.add(fp)

            stats["valid_candidate_count"] += 1
            valid_idx = stats["valid_candidate_count"]

            cleaned_item = {
                "title": title,
                "ingredients": ings_ner if ings_ner else ings_raw,
                "quantities": ings_raw,
                "directions": dirs,
                "source": clean_text(row.get("source", "RecipeNLG")),
            }

            # Reservoir Sampling Algorithm (Algorithm R)
            if len(reservoir) < target_sample_size:
                reservoir.append(cleaned_item)
            else:
                # Pick a random integer from 0 to valid_idx - 1
                j = random.randint(0, valid_idx - 1)
                if j < target_sample_size:
                    reservoir[j] = cleaned_item

            if stats["total_rows_read"] % 100000 == 0:
                print(
                    f"  Rows read: {stats['total_rows_read']:,} | "
                    f"Valid candidates: {stats['valid_candidate_count']:,} | "
                    f"Duplicates dropped: {stats['duplicates_removed']:,}"
                )

            if scan_limit and stats["total_rows_read"] >= scan_limit:
                print(f"Reached scan limit of {scan_limit:,} rows.")
                break

    actual_sample_size = len(reservoir)
    stats["final_retained_count"] = actual_sample_size
    print(
        f"\n✅ Finished stream processing.\n"
        f"   Total rows read: {stats['total_rows_read']:,}\n"
        f"   Valid candidates discovered: {stats['valid_candidate_count']:,}\n"
        f"   Duplicates removed: {stats['duplicates_removed']:,}\n"
        f"   Retained in reservoir: {actual_sample_size:,}"
    )

    # Format the selected sample records
    formatted_records = []
    for idx, item in enumerate(reservoir, 1):
        formatted_text = format_recipe_for_gpt2(
            title=item["title"],
            ingredients_ner=item["ingredients"],
            ingredients_raw=item["quantities"],
            directions=item["directions"],
        )
        rec = {
            "recipe_id": f"rec_{idx:06d}",
            "title": item["title"],
            "ingredients": item["ingredients"],
            "quantities": item["quantities"],
            "directions": item["directions"],
            "formatted_text": formatted_text,
            "source": item["source"],
        }
        formatted_records.append(rec)

    # Deterministic Split 90:5:5
    random.shuffle(formatted_records)
    n_total = len(formatted_records)
    stats["valid_kept"] = n_total
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    train_records = formatted_records[:n_train]
    val_records = formatted_records[n_train: n_train + n_val]
    test_records = formatted_records[n_train + n_val:]

    stats["train_count"] = len(train_records)
    stats["val_count"] = len(val_records)
    stats["test_count"] = len(test_records)

    # Output file paths
    train_path = os.path.join(output_dir, "recipes_train.jsonl")
    val_path = os.path.join(output_dir, "recipes_validation.jsonl")
    test_path = os.path.join(output_dir, "recipes_test.jsonl")

    for path, data_split in [(train_path, train_records), (val_path, val_records), (test_path, test_records)]:
        with open(path, "w", encoding="utf-8") as out_f:
            for item in data_split:
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Character & approximate token stats
    char_lengths = [len(r["formatted_text"]) for r in formatted_records]
    approx_tokens = [int(len(r["formatted_text"]) / 3.8) for r in formatted_records]

    stats["avg_character_length"] = round(sum(char_lengths) / max(len(char_lengths), 1), 1)
    stats["avg_approx_tokens"] = round(sum(approx_tokens) / max(len(approx_tokens), 1), 1)
    stats["max_approx_tokens"] = max(approx_tokens) if approx_tokens else 0
    stats["min_approx_tokens"] = min(approx_tokens) if approx_tokens else 0

    stats_path = os.path.join(output_dir, "dataset_statistics.json")
    with open(stats_path, "w", encoding="utf-8") as sf:
        json.dump(stats, sf, indent=2)

    metadata = {
        "dataset_name": "RecipeNLG Processed Pilot (Reservoir Sample)",
        "source_dataset": os.path.basename(input_path),
        "sampling_method": "Reservoir Sampling (Algorithm R)",
        "random_seed": RANDOM_SEED,
        "format": "GPT-2 Causal LM Delimited Recipe Format",
        "license_notice": "For non-commercial research and educational use only in accordance with RecipeNLG terms.",
        "splits": {
            "train": {"file": os.path.basename(train_path), "count": stats["train_count"], "percentage": "90%"},
            "validation": {"file": os.path.basename(val_path), "count": stats["val_count"], "percentage": "5%"},
            "test": {"file": os.path.basename(test_path), "count": stats["test_count"], "percentage": "5%"},
        },
        "statistics_file": os.path.basename(stats_path),
    }
    meta_path = os.path.join(output_dir, "dataset_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(metadata, mf, indent=2)

    print("\nDataset preparation completed successfully!")
    print(f" - Train:      {stats['train_count']:,} ({train_path})")
    print(f" - Validation: {stats['val_count']:,} ({val_path})")
    print(f" - Test:       {stats['test_count']:,} ({test_path})")
    print(f" - Statistics: {stats_path}")
    print(f" - Metadata:   {meta_path}")


# Public alias for external test suites and pipelines
process_dataset = process_dataset_with_reservoir_sampling


def main():
    parser = argparse.ArgumentParser(description="Prepare Recipe Dataset for GPT-2 Fine-Tuning with Reservoir Sampling")
    parser.add_argument("--input", required=True, help="Path to raw RecipeNLG CSV file")
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "data" / "processed"),
        help="Output directory for JSONL splits",
    )
    parser.add_argument("--max_recipes", type=int, default=5000, help="Target reservoir sample size")
    parser.add_argument("--scan_limit", type=int, default=None, help="Optional row scan limit")
    args = parser.parse_args()

    process_dataset_with_reservoir_sampling(
        input_path=args.input,
        output_dir=args.output_dir,
        target_sample_size=args.max_recipes,
        scan_limit=args.scan_limit,
    )


if __name__ == "__main__":
    main()
