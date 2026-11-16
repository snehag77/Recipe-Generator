# -*- coding: utf-8 -*-
"""Stage 2G: Held-Out Test Set Generation and Evaluation Script.

Evaluates Base GPT-2 vs. Stage 2F Full Fine-Tuned Checkpoint on all 250 records
in data/processed/recipes_test.jsonl using identical decoding settings:
- Temperature: 0.7
- Top-K: 40
- Seed: 42
- Max length: 220 tokens
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_HOME", str(Path(tempfile.gettempdir()) / ".keras"))

import keras
import keras_hub
import numpy as np
import tensorflow as tf
from utils import prompt


def compute_file_md5(filepath: str) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def parse_structure(text: str) -> dict:
    """Analyze the structural integrity of generated output."""
    has_title_tags = "<TITLE>" in text and "</TITLE>" in text
    has_quantities_tags = "<QUANTITIES>" in text and "</QUANTITIES>" in text
    has_steps_tags = "<STEPS>" in text
    has_steps_close = "</STEPS>" in text
    has_recipe_close = "</RECIPE>" in text

    # Extract title
    title = ""
    if has_title_tags:
        m = re.search(r"<TITLE>\s*(.*?)\s*</TITLE>", text, re.DOTALL)
        if m:
            title = m.group(1).strip()

    # Extract quantities
    quantities = []
    if has_quantities_tags:
        m = re.search(r"<QUANTITIES>\s*(.*?)\s*</QUANTITIES>", text, re.DOTALL)
        if m:
            quantities = [q.strip("- *").strip() for q in m.group(1).splitlines() if q.strip("- *").strip()]

    # Extract steps
    steps = []
    if "<STEPS>" in text:
        steps_text = text.split("<STEPS>")[-1]
        if "</STEPS>" in steps_text:
            steps_text = steps_text.split("</STEPS>")[0]
        steps = [s.strip() for s in steps_text.splitlines() if s.strip()]

    # Determine status
    has_core_sections = bool(title) and len(quantities) > 0 and len(steps) > 0
    is_fully_closed = has_core_sections and has_steps_close and has_recipe_close
    is_partially_structured = has_core_sections and not is_fully_closed
    is_empty = len(text.strip()) == 0 or (not title and not quantities and not steps)
    is_incomplete = not has_recipe_close

    # Repetition check (repeated phrases of 4+ words or repeated lines)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    has_line_repetition = len(lines) > len(set(lines)) and any(lines.count(ln) >= 3 for ln in set(lines))
    words = re.findall(r"\b\w+\b", text.lower())
    four_grams = [tuple(words[i:i+4]) for i in range(len(words)-3)] if len(words) >= 4 else []
    has_ngram_repetition = any(four_grams.count(fg) >= 4 for fg in set(four_grams)) if four_grams else False
    is_repetitive = has_line_repetition or has_ngram_repetition

    return {
        "title": title,
        "quantities": quantities,
        "steps": steps,
        "has_title_tags": has_title_tags,
        "has_quantities_tags": has_quantities_tags,
        "has_steps_tags": has_steps_tags,
        "has_steps_close": has_steps_close,
        "has_recipe_close": has_recipe_close,
        "is_fully_closed": is_fully_closed,
        "is_partially_structured": is_partially_structured,
        "is_empty": is_empty,
        "is_incomplete": is_incomplete,
        "is_repetitive": is_repetitive,
    }


def evaluate_ingredients(supplied_ingredients: list, generated_text: str, parsed: dict) -> dict:
    """Assess ingredient inclusion and unprovided ingredient introduction."""
    text_lower = generated_text.lower()
    
    # Check supplied ingredient recall
    found = []
    missing = []
    for ing in supplied_ingredients:
        ing_clean = ing.strip().lower()
        # Look for full ingredient string or individual words
        pattern = r"\b" + re.escape(ing_clean) + r"\b"
        if re.search(pattern, text_lower):
            found.append(ing_clean)
        else:
            # Check individual main token (e.g. 'potato' in 'potatoes')
            tokens = [t for t in ing_clean.split() if len(t) > 3]
            if tokens and any(re.search(r"\b" + re.escape(t[:4]), text_lower) for t in tokens):
                found.append(ing_clean)
            else:
                missing.append(ing_clean)

    recall = len(found) / len(supplied_ingredients) if supplied_ingredients else 1.0
    full_inclusion = len(missing) == 0

    # Common kitchen staples not flagged as hallucinations
    staples = {"salt", "pepper", "water", "oil", "olive oil", "butter", "sugar", "flour"}
    supplied_set = {s.lower() for s in supplied_ingredients} | staples

    # Count extra items in quantities
    extra_ingredients = []
    for q in parsed["quantities"]:
        q_lower = q.lower()
        # strip digits, units
        cleaned = re.sub(r"[\d/]+|\b(cup|cups|tbsp|tsp|tablespoon|teaspoon|oz|ounce|pound|lb|can|package|bag|stick|clove|cloves)\b", "", q_lower).strip()
        cleaned_words = [w for w in re.findall(r"\b[a-z]{3,}\b", cleaned) if w not in {"chopped", "diced", "minced", "sliced", "peeled", "melted", "divided", "large", "medium", "small", "fresh"}]
        # Check if any word matches supplied
        if cleaned_words and not any(any(w in s for s in supplied_set) for w in cleaned_words):
            extra_ingredients.append(" ".join(cleaned_words[:3]))

    return {
        "supplied_count": len(supplied_ingredients),
        "found_count": len(found),
        "missing_count": len(missing),
        "found_ingredients": found,
        "missing_ingredients": missing,
        "recall": recall,
        "full_inclusion": full_inclusion,
        "extra_ingredients": list(set(extra_ingredients)),
        "has_extra_ingredients": len(extra_ingredients) > 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Base GPT-2 vs Stage 2F on Held-Out Test Set")
    parser.add_argument(
        "--test_file",
        default=str(PROJECT_ROOT / "data" / "processed" / "recipes_test.jsonl"),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "models" / "recipe_gpt2_full" / "recipe_gpt2_full_best.weights.h5"),
    )
    parser.add_argument(
        "--output_file",
        default=str(PROJECT_ROOT / "data" / "outputs" / "stage2g_generation_eval.json"),
    )
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=220)
    args = parser.parse_args()

    print("=" * 70)
    print("RecipeGen AI — Stage 2G: Held-Out Test Set Generation Evaluation")
    print("=" * 70)

    # 1. Load test data
    with open(args.test_file, "r", encoding="utf-8") as f:
        test_records = [json.loads(line) for line in f]
    print(f"Loaded {len(test_records)} test records from {args.test_file}")

    prompts = [prompt.build_prompt(r["ingredients"]) for r in test_records]

    # 2. Setup preset and models
    cached_path = Path.home() / ".cache" / "kagglehub" / "models" / "keras" / "gpt2" / "keras" / "gpt2_base_en" / "3"
    preset = str(cached_path) if cached_path.is_dir() else "gpt2_base_en"
    preprocessor = keras_hub.models.GPT2CausalLMPreprocessor.from_preset(preset, sequence_length=512)

    sampler = keras_hub.samplers.TopKSampler(k=40, temperature=0.7, seed=42)

    # Base Model Generation
    print("\n⏳ Initializing Base GPT-2 model...")
    base_model = keras_hub.models.GPT2CausalLM.from_preset(preset, preprocessor=preprocessor)
    base_model.compile(sampler=sampler)

    print("🚀 Generating recipes with Base GPT-2 (250 records)...")
    base_start = time.time()
    base_outputs = []
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i:i+args.batch_size]
        outs = base_model.generate(batch, max_length=args.max_length)
        base_outputs.extend(outs)
        print(f"   Base: generated {len(base_outputs)}/{len(prompts)}...")
    base_duration = time.time() - base_start
    print(f"✅ Base generation complete in {base_duration:.1f}s")

    # Clean up base model from memory
    del base_model
    keras.backend.clear_session()

    # Stage 2F Full Fine-Tuned Model Generation
    print("\n⏳ Initializing Stage 2F Full Fine-Tuned model...")
    ft_model = keras_hub.models.GPT2CausalLM.from_preset(preset, preprocessor=preprocessor)
    ft_model.load_weights(args.checkpoint)
    ft_model.compile(sampler=sampler)

    print(f"🚀 Generating recipes with Stage 2F Checkpoint ({args.checkpoint})...")
    ft_start = time.time()
    ft_outputs = []
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i:i+args.batch_size]
        outs = ft_model.generate(batch, max_length=args.max_length)
        ft_outputs.extend(outs)
        print(f"   Stage 2F: generated {len(ft_outputs)}/{len(prompts)}...")
    ft_duration = time.time() - ft_start
    print(f"✅ Stage 2F generation complete in {ft_duration:.1f}s")

    # 3. Analyze all outputs
    print("\n🔍 Analyzing structural and ingredient metrics across 250 test samples...")
    detailed_results = []
    base_stats = {
        "fully_closed": 0,
        "partially_structured": 0,
        "unstructured": 0,
        "empty": 0,
        "repetitive": 0,
        "total_recall": 0.0,
        "full_inclusion_count": 0,
        "extra_ing_count": 0,
    }
    ft_stats = {
        "fully_closed": 0,
        "partially_structured": 0,
        "unstructured": 0,
        "empty": 0,
        "repetitive": 0,
        "total_recall": 0.0,
        "full_inclusion_count": 0,
        "extra_ing_count": 0,
    }

    for idx, r in enumerate(test_records):
        b_raw = base_outputs[idx]
        f_raw = ft_outputs[idx]

        b_struct = parse_structure(b_raw)
        b_ing = evaluate_ingredients(r["ingredients"], b_raw, b_struct)

        f_struct = parse_structure(f_raw)
        f_ing = evaluate_ingredients(r["ingredients"], f_raw, f_struct)

        # Base stats update
        if b_struct["is_fully_closed"]:
            base_stats["fully_closed"] += 1
        elif b_struct["is_partially_structured"]:
            base_stats["partially_structured"] += 1
        else:
            base_stats["unstructured"] += 1
        if b_struct["is_empty"]:
            base_stats["empty"] += 1
        if b_struct["is_repetitive"]:
            base_stats["repetitive"] += 1
        base_stats["total_recall"] += b_ing["recall"]
        if b_ing["full_inclusion"]:
            base_stats["full_inclusion_count"] += 1
        if b_ing["has_extra_ingredients"]:
            base_stats["extra_ing_count"] += 1

        # Stage 2F stats update
        if f_struct["is_fully_closed"]:
            ft_stats["fully_closed"] += 1
        elif f_struct["is_partially_structured"]:
            ft_stats["partially_structured"] += 1
        else:
            ft_stats["unstructured"] += 1
        if f_struct["is_empty"]:
            ft_stats["empty"] += 1
        if f_struct["is_repetitive"]:
            ft_stats["repetitive"] += 1
        ft_stats["total_recall"] += f_ing["recall"]
        if f_ing["full_inclusion"]:
            ft_stats["full_inclusion_count"] += 1
        if f_ing["has_extra_ingredients"]:
            ft_stats["extra_ing_count"] += 1

        detailed_results.append({
            "recipe_id": r["recipe_id"],
            "reference_title": r["title"],
            "supplied_ingredients": r["ingredients"],
            "base": {
                "generated_text": b_raw,
                "structure": b_struct,
                "ingredients": b_ing,
            },
            "stage2f": {
                "generated_text": f_raw,
                "structure": f_struct,
                "ingredients": f_ing,
            }
        })

    n = len(test_records)
    summary = {
        "num_test_records": n,
        "decoding": {
            "temperature": 0.7,
            "top_k": 40,
            "seed": 42,
            "max_length": args.max_length,
        },
        "timings": {
            "base_seconds": round(base_duration, 1),
            "stage2f_seconds": round(ft_duration, 1),
        },
        "base_metrics": {
            "fully_closed_rate": round(base_stats["fully_closed"] / n, 4),
            "partially_structured_rate": round(base_stats["partially_structured"] / n, 4),
            "unstructured_rate": round(base_stats["unstructured"] / n, 4),
            "empty_rate": round(base_stats["empty"] / n, 4),
            "repetitive_rate": round(base_stats["repetitive"] / n, 4),
            "mean_ingredient_recall": round(base_stats["total_recall"] / n, 4),
            "full_ingredient_adherence_rate": round(base_stats["full_inclusion_count"] / n, 4),
            "unprovided_ingredients_rate": round(base_stats["extra_ing_count"] / n, 4),
        },
        "stage2f_metrics": {
            "fully_closed_rate": round(ft_stats["fully_closed"] / n, 4),
            "partially_structured_rate": round(ft_stats["partially_structured"] / n, 4),
            "unstructured_rate": round(ft_stats["unstructured"] / n, 4),
            "empty_rate": round(ft_stats["empty"] / n, 4),
            "repetitive_rate": round(ft_stats["repetitive"] / n, 4),
            "mean_ingredient_recall": round(ft_stats["total_recall"] / n, 4),
            "full_ingredient_adherence_rate": round(ft_stats["full_inclusion_count"] / n, 4),
            "unprovided_ingredients_rate": round(ft_stats["extra_ing_count"] / n, 4),
        },
        "detailed": detailed_results,
    }

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("STAGE 2G GENERATION EVALUATION RESULTS")
    print("=" * 70)
    print("Metric                           | Base GPT-2 | Stage 2F Full FT")
    print("-" * 65)
    print(f"Fully Closed (<RECIPE>)          | {summary['base_metrics']['fully_closed_rate']*100:6.1f}%    | {summary['stage2f_metrics']['fully_closed_rate']*100:6.1f}%")
    print(f"Partially Structured             | {summary['base_metrics']['partially_structured_rate']*100:6.1f}%    | {summary['stage2f_metrics']['partially_structured_rate']*100:6.1f}%")
    print(f"Unstructured / Failed Tags       | {summary['base_metrics']['unstructured_rate']*100:6.1f}%    | {summary['stage2f_metrics']['unstructured_rate']*100:6.1f}%")
    print(f"Repetitive / Degenerate          | {summary['base_metrics']['repetitive_rate']*100:6.1f}%    | {summary['stage2f_metrics']['repetitive_rate']*100:6.1f}%")
    print(f"Mean Ingredient Recall           | {summary['base_metrics']['mean_ingredient_recall']*100:6.1f}%    | {summary['stage2f_metrics']['mean_ingredient_recall']*100:6.1f}%")
    print(f"Full Ingredient Adherence (100%) | {summary['base_metrics']['full_ingredient_adherence_rate']*100:6.1f}%    | {summary['stage2f_metrics']['full_ingredient_adherence_rate']*100:6.1f}%")
    print(f"Introduced Extra Ingredients     | {summary['base_metrics']['unprovided_ingredients_rate']*100:6.1f}%    | {summary['stage2f_metrics']['unprovided_ingredients_rate']*100:6.1f}%")
    print("=" * 70)
    print(f"Detailed results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
