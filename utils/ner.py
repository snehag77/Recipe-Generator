# -*- coding: utf-8 -*-
"""U3: Ingredient Tokenizer and Named Entity Recognition (NER).

Supports:
1. Neural BERT-based Ingredient NER (Fine-tuned google-bert/bert-base-cased)
   extracting ingredient entity spans from unstructured culinary text.
2. Rule-based delimiter and pattern matcher as a reliable, zero-dependency fallback.

Preserves the standard extract_ingredients() interface for 100% backward compatibility.
"""

import json
from pathlib import Path
import re
import subprocess
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "ingredient_ner"
VENV_PT_PYTHON = PROJECT_ROOT / "venv_pt" / "bin" / "python"
RUN_NER_SCRIPT = PROJECT_ROOT / "scripts" / "run_ner.py"


def is_bert_ner_available() -> bool:
    """Check if fine-tuned BERT NER model and venv_pt environment are available."""
    has_weights = (MODEL_DIR / "config.json").is_file() and (
        (MODEL_DIR / "model.safetensors").is_file() or (MODEL_DIR / "pytorch_model.bin").is_file()
    )
    has_runtime = VENV_PT_PYTHON.is_file() and RUN_NER_SCRIPT.is_file()
    return bool(has_weights and has_runtime)


def extract_ingredients_rule_based(text: str) -> List[str]:
    """Extract ingredients using the rule-based delimiter & list parser."""
    if not text or not isinstance(text, str):
        return []

    # Split on combinations of newlines, commas, and semicolons
    raw_tokens = re.split(r"[\n,;]+", text)

    seen = set()
    cleaned = []

    for token in raw_tokens:
        # Strip common markdown/bullet/number list prefixes (e.g. '-', '*', '•', '1.', '1)')
        item = re.sub(r"^\s*(?:[-*•–—]|\d+[.)])\s*", "", token).strip()
        if not item:
            continue

        item_key = item.lower()
        if item_key not in seen:
            seen.add(item_key)
            cleaned.append(item)

    return cleaned


def extract_ingredients_bert(text: str) -> Optional[List[str]]:
    """Extract ingredient entities using the fine-tuned BERT token classifier."""
    if not is_bert_ner_available():
        return None

    try:
        res = subprocess.run(
            [str(VENV_PT_PYTHON), str(RUN_NER_SCRIPT), "--text", text],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if res.returncode == 0:
            for line in reversed(res.stdout.strip().splitlines()):
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        data = json.loads(line)
                        if data.get("status") == "ok":
                            return data.get("ingredients", [])
                    except json.JSONDecodeError:
                        continue
    except Exception:
        pass

    return None


def extract_ingredients_with_metadata(
    text: str,
    method: str = "auto",
) -> Tuple[List[str], str]:
    """Extract ingredients and report which extraction engine was utilized.

    Args:
        text: Free-text ingredient input string.
        method: 'auto' (prioritize BERT with fallback), 'bert', or 'rule'.

    Returns:
        Tuple of (extracted_ingredients_list, active_method_name).
    """
    if not text or not isinstance(text, str):
        return [], "rule"

    clean_text = text.strip()
    if not clean_text:
        return [], "rule"

    if method in ("auto", "bert"):
        bert_res = extract_ingredients_bert(clean_text)
        if bert_res is not None and len(bert_res) > 0:
            return bert_res, "bert"
        if method == "bert":
            # If user explicitly requested BERT but it yielded nothing, fall back
            pass

    # Rule-based fallback
    return extract_ingredients_rule_based(clean_text), "rule"


def extract_ingredients(text: str, method: str = "rule") -> List[str]:
    """Extract a clean, deduplicated list of ingredient strings from free-text input.

    Standard public API method maintaining complete backward compatibility.
    """
    ingredients, _ = extract_ingredients_with_metadata(text, method=method)
    return ingredients
