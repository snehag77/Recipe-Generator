# -*- coding: utf-8 -*-
"""U1: Structured ingredient-to-recipe prompt design and output parsing helpers.

Aligned exactly with RecipeNLG training format:
<INGREDIENTS>
ingredient 1
ingredient 2
<END_ING>
<RECIPE>
<TITLE>
[Model generates title, </TITLE>, <QUANTITIES>, </QUANTITIES>, <STEPS>, </STEPS>, </RECIPE>]
"""

import re
from typing import Dict, List, Optional


def build_prompt(
    ingredients: List[str],
    title_hint: Optional[str] = None,
) -> str:
    """Construct an inference prompt matching the exact training format.

    Deduplicates ingredients while preserving order, formats one ingredient
    per line inside `<INGREDIENTS> ... <END_ING>`, and terminates cleanly at
    the generation prefix `<TITLE>\n` (or `<TITLE>\n{hint}` if user provided a style hint).

    Args:
        ingredients: Raw list of ingredient strings.
        title_hint: Optional dish style/title hint.

    Returns:
        Structured prompt string.
    """
    seen = set()
    cleaned_ings = []
    for raw_ing in ingredients:
        clean = raw_ing.strip().lower()
        if clean and clean not in seen:
            seen.add(clean)
            cleaned_ings.append(clean)

    if not cleaned_ings:
        cleaned_ings = ["ingredients not specified"]

    ings_block = "\n".join(cleaned_ings)

    prompt = (
        f"<INGREDIENTS>\n"
        f"{ings_block}\n"
        f"<END_ING>\n"
        f"<RECIPE>\n"
        f"<TITLE>\n"
    )
    if title_hint and title_hint.strip():
        prompt += title_hint.strip().title() + "\n"

    return prompt


def parse_recipe_output(text: str) -> Dict[str, Optional[str]]:
    """Safe post-processing utility: converts raw generated text with tags into clean sections.

    Extracts:
    - title
    - quantities (as a clean list of lines)
    - steps (as a clean list of instructions)
    """
    out = {
        "title": None,
        "quantities": [],
        "steps": [],
        "raw_text": text,
    }

    # Extract Title between <TITLE> and </TITLE>
    title_match = re.search(r"<TITLE>(.*?)</TITLE>", text, re.DOTALL | re.IGNORECASE)
    if title_match:
        out["title"] = title_match.group(1).strip()
    else:
        # Fallback: check if title starts after <TITLE>
        t_fallback = re.search(r"<TITLE>\s*(.*?)(?=\n|<QUANTITIES>|</RECIPE>|$)", text, re.DOTALL | re.IGNORECASE)
        if t_fallback:
            out["title"] = t_fallback.group(1).strip()

    # Extract Quantities between <QUANTITIES> and </QUANTITIES>
    qty_match = re.search(r"<QUANTITIES>(.*?)</QUANTITIES>", text, re.DOTALL | re.IGNORECASE)
    if qty_match:
        raw_qtys = qty_match.group(1).strip().splitlines()
        clean_qtys = [q.lstrip("-* •").strip() for q in raw_qtys if q.strip()]
        out["quantities"] = clean_qtys

    # Extract Steps between <STEPS> and </STEPS>
    steps_match = re.search(r"<STEPS>(.*?)</STEPS>", text, re.DOTALL | re.IGNORECASE)
    if steps_match:
        raw_steps = steps_match.group(1).strip().splitlines()
        clean_steps = [s.strip() for s in raw_steps if s.strip()]
        out["steps"] = clean_steps

    return out


def format_recipe_markdown(parsed: Dict[str, Optional[str]]) -> str:
    """Format parsed recipe dictionary into beautiful, clean Markdown for display."""
    md_lines = []
    if parsed.get("title"):
        md_lines.append(f"### {parsed['title']}\n")

    if parsed.get("quantities"):
        md_lines.append("#### 🛒 Ingredients & Quantities")
        for q in parsed["quantities"]:
            md_lines.append(f"- {q}")
        md_lines.append("")

    if parsed.get("steps"):
        md_lines.append("#### 👩‍🍳 Cooking Directions")
        for s in parsed["steps"]:
            # If step already has a number prefix, keep it, otherwise format as bullet/number
            md_lines.append(s if re.match(r"^\d+\.", s) else f"1. {s}")
        md_lines.append("")

    if not md_lines:
        return parsed.get("raw_text", "No recipe generated.")

    return "\n".join(md_lines)
