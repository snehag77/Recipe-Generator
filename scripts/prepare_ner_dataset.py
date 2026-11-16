# -*- coding: utf-8 -*-
"""RecipeGen AI — Weakly Supervised Ingredient NER Dataset Preparation.

Extracts token-level BIO annotations for ingredient entity recognition using
phrase alignment from RecipeNLG ground-truth entity lists.

Strict Guarantees:
1. Zero Data Leakage: Splits strictly match recipes_train, recipes_validation,
   and recipes_test disjoint recipe IDs.
2. Robust Phrase Alignment: Handles case normalization, hyphenation, punctuation,
   and resolves overlapping entity spans (longest-match first).
3. Traceability: Every sentence preserves its originating recipe_id.
4. Transparency: Formally labeled as 'weakly_supervised_phrase_alignment'.
"""

import argparse
import json
from pathlib import Path
import random
import re
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "ner_dataset"


def normalize_token_text(text: str) -> str:
    """Normalize whitespace and punctuation quotes."""
    text = re.sub(r"[\r\t\f\v]+", " ", str(text))
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201C\u201D]", '"', text)
    return text.strip()


def find_entity_spans(
    text: str,
    entities: List[str],
) -> List[Tuple[int, int, str]]:
    """Locate non-overlapping character spans [start, end) for ingredient entities.

    Uses longest-match-first to prevent sub-phrase shadowing (e.g. 'extra virgin olive oil'
    before 'olive oil'). Handles hyphenation and word boundaries.
    """
    if not text or not entities:
        return []

    clean_ents = []
    seen_ents = set()
    for e in entities:
        e_clean = normalize_token_text(e).strip().lower()
        if len(e_clean) >= 2 and e_clean not in seen_ents:
            seen_ents.add(e_clean)
            clean_ents.append(e_clean)

    # Sort descending by length so longer phrases match before shorter constituents
    clean_ents.sort(key=lambda s: len(s), reverse=True)

    occupied_spans = []  # List of (start, end, entity_name)

    def is_overlapping(s: int, e: int) -> bool:
        for os, oe, _ in occupied_spans:
            if max(s, os) < min(e, oe):
                return True
        return False

    for entity in clean_ents:
        # Split entity into words to handle whitespace and hyphens flexibly
        pattern_tokens = [re.escape(w) for w in re.split(r"[\s\-]+", entity) if w]
        if not pattern_tokens:
            continue
        pattern_str = r"[\s\-]+".join(pattern_tokens)
        regex = re.compile(r"(?<![a-zA-Z0-9])" + pattern_str + r"(?![a-zA-Z0-9])", re.IGNORECASE)

        for match in regex.finditer(text):
            start, end = match.span()
            if not is_overlapping(start, end):
                occupied_spans.append((start, end, entity))

    # Sort spans chronologically by start character
    occupied_spans.sort(key=lambda x: x[0])
    return occupied_spans


def tokenize_with_spans(text: str) -> List[Tuple[str, int, int]]:
    """Split text into words/tokens with exact character spans [start, end)."""
    tokens = []
    for m in re.finditer(r"\w+|[^\w\s]", text):
        tokens.append((m.group(0), m.start(), m.end()))
    return tokens


def build_bio_annotated_example(
    text: str,
    entities: List[str],
    recipe_id: Any,
) -> Optional[Dict[str, Any]]:
    """Build a token-level BIO annotated dictionary from sentence and entity list."""
    text_clean = normalize_token_text(text)
    if not text_clean or len(text_clean) < 3:
        return None

    entity_spans = find_entity_spans(text_clean, entities)
    token_spans = tokenize_with_spans(text_clean)
    if not token_spans or not entity_spans:
        return None

    tokens = []
    ner_tags = []
    matched_entities = set()

    for token, t_start, t_end in token_spans:
        tag = "O"
        for e_start, e_end, ent_name in entity_spans:
            if max(t_start, e_start) < min(t_end, e_end):
                # Token overlaps entity span
                # Check if this token is the first token of the entity
                is_first = True
                for prev_tok, p_start, p_end in token_spans:
                    if p_end <= t_start and max(p_start, e_start) < min(p_end, e_end):
                        is_first = False
                        break
                tag = "B-INGREDIENT" if is_first else "I-INGREDIENT"
                matched_entities.add(ent_name)
                break

        tokens.append(token)
        ner_tags.append(tag)

    return {
        "recipe_id": recipe_id,
        "text": text_clean,
        "tokens": tokens,
        "ner_tags": ner_tags,
        "extracted_entities": sorted(list(matched_entities)),
        "annotation_type": "weakly_supervised_phrase_alignment",
    }


def create_conversational_sentence(ingredients: List[str], quantities: List[str]) -> str:
    """Synthesize realistic conversational recipe instructions."""
    templates = [
        "Add {items} into the large mixing bowl.",
        "In a hot pan, saute {items} until fragrant.",
        "Combine {items} and mix thoroughly.",
        "You will need {items} to prepare this dish.",
        "Gently stir in {items} and simmer on low heat.",
        "Chop and prepare {items} before cooking.",
    ]
    chosen = quantities if quantities else ingredients
    sample_size = min(len(chosen), random.randint(2, 5))
    selected = random.sample(chosen, sample_size)
    if len(selected) == 1:
        items_str = selected[0]
    elif len(selected) == 2:
        items_str = f"{selected[0]} and {selected[1]}"
    else:
        items_str = ", ".join(selected[:-1]) + f", and {selected[-1]}"

    tmpl = random.choice(templates)
    return tmpl.format(items=items_str)


def process_split(
    input_jsonl_path: Path,
    output_jsonl_path: Path,
    max_sentences_per_recipe: int = 3,
) -> Tuple[int, int, set]:
    """Process a single JSONL split into BIO annotated sentences."""
    total_sentences = 0
    recipe_ids = set()

    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(input_jsonl_path, "r", encoding="utf-8") as in_f, open(output_jsonl_path, "w", encoding="utf-8") as out_f:
        for line in in_f:
            if not line.strip():
                continue
            record = json.loads(line)
            r_id = record.get("recipe_id")
            entities = record.get("ingredients", [])
            quantities = record.get("quantities", [])
            directions = record.get("directions", [])

            if not entities:
                continue

            recipe_ids.add(r_id)
            sentences_created = 0

            # 1. Raw ingredient lines
            for q in quantities[:4]:
                ex = build_bio_annotated_example(q, entities, r_id)
                if ex and any(t != "O" for t in ex["ner_tags"]):
                    out_f.write(json.dumps(ex) + "\n")
                    total_sentences += 1
                    sentences_created += 1
                    if sentences_created >= max_sentences_per_recipe:
                        break

            # 2. Authentic direction sentences
            if sentences_created < max_sentences_per_recipe and directions:
                for d in directions[:3]:
                    ex = build_bio_annotated_example(d, entities, r_id)
                    if ex and any(t != "O" for t in ex["ner_tags"]):
                        out_f.write(json.dumps(ex) + "\n")
                        total_sentences += 1
                        sentences_created += 1
                        if sentences_created >= max_sentences_per_recipe:
                            break

            # 3. Conversational synthesis
            if sentences_created < max_sentences_per_recipe:
                conv_text = create_conversational_sentence(entities, quantities)
                ex = build_bio_annotated_example(conv_text, entities, r_id)
                if ex and any(t != "O" for t in ex["ner_tags"]):
                    out_f.write(json.dumps(ex) + "\n")
                    total_sentences += 1
                    sentences_created += 1

    return len(recipe_ids), total_sentences, recipe_ids


def main():
    parser = argparse.ArgumentParser(description="Prepare Weakly Supervised Ingredient NER Dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max-sentences", type=int, default=2, help="Max sentences extracted per recipe")
    args = parser.parse_args()

    random.seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("======================================================================")
    print("RecipeGen AI: Preparing Weakly Supervised BERT NER Dataset")
    print("======================================================================")

    splits = {
        "train": (PROJECT_ROOT / "data" / "processed" / "recipes_train.jsonl", OUTPUT_DIR / "ner_train.jsonl"),
        "val": (PROJECT_ROOT / "data" / "processed" / "recipes_validation.jsonl", OUTPUT_DIR / "ner_val.jsonl"),
        "test": (PROJECT_ROOT / "data" / "processed" / "recipes_test.jsonl", OUTPUT_DIR / "ner_test.jsonl"),
    }

    all_split_ids = {}

    for split_name, (in_path, out_path) in splits.items():
        if not in_path.is_file():
            raise FileNotFoundError(f"Missing required source split: {in_path}")

        n_recipes, n_sentences, r_ids = process_split(in_path, out_path, max_sentences_per_recipe=args.max_sentences)
        all_split_ids[split_name] = r_ids
        print(f"✅ {split_name.upper()} Split: {n_recipes} recipes -> {n_sentences} annotated sentences ({out_path})")

    # Strict split isolation verification
    train_val_overlap = all_split_ids["train"] & all_split_ids["val"]
    train_test_overlap = all_split_ids["train"] & all_split_ids["test"]
    val_test_overlap = all_split_ids["val"] & all_split_ids["test"]

    print("\n🔍 Split Isolation & Recipe ID Audit:")
    print(f"   - Train / Validation ID Overlap: {len(train_val_overlap)}")
    print(f"   - Train / Test ID Overlap:       {len(train_test_overlap)}")
    print(f"   - Val / Test ID Overlap:         {len(val_test_overlap)}")

    assert len(train_val_overlap) == 0, "FATAL: Data leakage detected between train and val!"
    assert len(train_test_overlap) == 0, "FATAL: Data leakage detected between train and test!"
    assert len(val_test_overlap) == 0, "FATAL: Data leakage detected between val and test!"

    print("🎉 Verification SUCCESSFUL: Zero recipe-ID overlap across all three NER splits.")


if __name__ == "__main__":
    main()
