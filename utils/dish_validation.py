# -*- coding: utf-8 -*-
"""Lightweight culinary validation for ingredient and dish title alignment.

Provides non-blocking advisory warnings when an optional dish title style
(e.g., 'Dal', 'Biryani', 'Omelette') is specified without its quintessential
culinary base ingredients (e.g., lentils, rice, eggs).

Operational Guidelines:
- Advisory only: never blocks recipe generation.
- Never automatically mutates or injects ingredients into user lists.
- Avoids large brittle knowledge bases; uses a concise set of standard culinary archetypes.
"""

import re
from typing import List, Optional

DISH_ALIGNMENT_RULES = [
    {
        "patterns": [r"\bdals?\b", r"\bdaals?\b", r"\bdahls?\b", r"\blentils?\b", r"\bsambars?\b"],
        "display_name": "Dal",
        "expected_component": "lentils, pulses, or beans",
        "keywords": [
            "lentil", "lentils", "dal", "daal", "dahl", "moong", "toor", "urad",
            "chana", "masoor", "split pea", "pigeon pea", "gram", "beans", "chickpea", "pulses",
        ],
    },
    {
        "patterns": [r"\bbiryanis?\b", r"\bpulaos?\b", r"\bfried\s+rice\b", r"\brisottos?\b"],
        "display_name": "Rice dishes (e.g. Biryani/Pulao)",
        "expected_component": "rice or culinary grains",
        "keywords": ["rice", "basmati", "jasmine", "arborio", "grain", "grains"],
    },
    {
        "patterns": [r"\bpastas?\b", r"\bspaghettis?\b", r"\bmacaronis?\b", r"\blasagnas?\b", r"\bnoodles?\b", r"\bramen\b"],
        "display_name": "Pasta or Noodle dishes",
        "expected_component": "pasta, noodles, or macaroni",
        "keywords": ["pasta", "noodle", "noodles", "spaghetti", "macaroni", "penne", "fusilli", "fettuccine", "lasagna", "ramen", "linguine"],
    },
    {
        "patterns": [r"\bbreads?\b", r"\bcakes?\b", r"\bpancakes?\b", r"\bcookies?\b", r"\bmuffins?\b", r"\bwaffles?\b"],
        "display_name": "Baked goods / Batters",
        "expected_component": "flour, baking meal, or a grain base",
        "keywords": ["flour", "maida", "atta", "meal", "oats", "almond flour", "cornstarch", "dough", "batter"],
    },
    {
        "patterns": [r"\bomelettes?\b", r"\bomelets?\b", r"\bfrittatas?\b", r"\bscrambled\s+eggs?\b"],
        "display_name": "Egg dishes (e.g. Omelette/Frittata)",
        "expected_component": "eggs",
        "keywords": ["egg", "eggs", "egg white", "egg whites", "yolk", "yolks"],
    },
    {
        "patterns": [r"\bpaneers?\b"],
        "display_name": "Paneer dishes",
        "expected_component": "paneer or cottage cheese",
        "keywords": ["paneer", "cottage cheese", "tofu", "cheese"],
    },
    {
        "patterns": [r"\bpizzas?\b"],
        "display_name": "Pizza",
        "expected_component": "flour, pizza dough, or crust base",
        "keywords": ["flour", "dough", "crust", "pita", "flatbread", "bread"],
    },
    {
        "patterns": [r"\bsmoothies?\b", r"\bshakes?\b"],
        "display_name": "Smoothies or Shakes",
        "expected_component": "a liquid or fruit base (milk, yogurt, juice, or fruit)",
        "keywords": ["milk", "yogurt", "juice", "water", "banana", "berries", "apple", "mango", "fruit", "ice cream"],
    },
]


def validate_dish_title_alignment(
    dish_title: Optional[str],
    ingredients: List[str],
) -> Optional[str]:
    """Check for possible culinary mismatches between an optional dish title and ingredients.

    Args:
        dish_title: User-specified dish style or title hint (e.g., 'Dal', 'Biryani').
        ingredients: List of parsed ingredient strings.

    Returns:
        Advisory warning string if an expected core component is missing, or None if aligned.
    """
    if not dish_title or not isinstance(dish_title, str) or not dish_title.strip():
        return None

    if not ingredients:
        return None

    title_clean = dish_title.strip().lower()
    combined_ingredients = " ".join(ing.lower() for ing in ingredients)

    for rule in DISH_ALIGNMENT_RULES:
        # Check if the title matches any pattern in this rule
        matched_pattern = any(re.search(pat, title_clean) for pat in rule["patterns"])
        if matched_pattern:
            # Check if any expected keyword is present in the ingredients
            has_expected = any(kw in combined_ingredients for kw in rule["keywords"])
            if not has_expected:
                return (
                    f"⚠️ **Advisory:** '{rule['display_name']}' typically requires {rule['expected_component']}, "
                    f"but these were not found in your ingredient list. Consider adding them or choosing another dish style."
                )

    return None
