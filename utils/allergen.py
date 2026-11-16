# -*- coding: utf-8 -*-
"""U5a: Allergen detection logic.

Rule-based keyword and compound phrase detection across standard allergen categories,
with common false-positive mitigations and support for declared allergen-free product claims.

Important Safety Notes:
- Recognizes declared product claims (e.g., 'gluten-free flour', 'dairy-free cheese', 'egg-free mayo')
  to prevent false alarms on plant-based and substitute products.
- Does NOT treat 'lactose-free' as dairy-free: lactose-free cow's milk still contains milk proteins (casein, whey).
- Does NOT suppress an allergen warning if an explicit contradicting allergen is detected.
- Allergen detection is rule-based and advisory only. It does not replace manufacturer packaging checks.
"""

import re
from typing import Dict, List, Set

# Comprehensive major allergen dictionary
ALLERGENS: Dict[str, List[str]] = {
    "Milk/Dairy": [
        "milk",
        "butter",
        "ghee",
        "cream",
        "cheese",
        "paneer",
        "curd",
        "yogurt",
        "yoghurt",
        "whey",
        "casein",
        "buttermilk",
        "mozzarella",
        "parmesan",
        "cheddar",
        "ricotta",
        "custard",
    ],
    "Peanuts": ["peanut", "peanuts", "groundnut"],
    "Tree Nuts": [
        "almond",
        "walnut",
        "cashew",
        "pistachio",
        "pecan",
        "hazelnut",
        "macadamia",
        "brazil nut",
        "pine nut",
    ],
    "Wheat/Gluten": [
        "wheat",
        "flour",
        "maida",
        "atta",
        "semolina",
        "suji",
        "rava",
        "bread",
        "pasta",
        "spaghetti",
        "macaroni",
        "noodle",
        "noodles",
        "couscous",
        "barley",
        "rye",
        "soy sauce",
        "roti",
        "naan",
        "chapati",
    ],
    "Soy": [
        "soy",
        "soya",
        "soybean",
        "tofu",
        "edamame",
        "miso",
        "tempeh",
        "tamari",
    ],
    "Eggs": ["egg", "eggs", "mayonnaise", "mayo", "meringue", "albumen"],
    "Fish": [
        "fish",
        "salmon",
        "tuna",
        "cod",
        "anchovy",
        "anchovies",
        "sardine",
        "mackerel",
        "trout",
        "tilapia",
    ],
    "Shellfish": [
        "shrimp",
        "prawn",
        "prawns",
        "crab",
        "lobster",
        "crayfish",
        "oyster",
        "clam",
        "mussel",
        "scallop",
        "squid",
        "octopus",
    ],
    "Sesame": ["sesame", "sesame oil", "tahini", "sesame seeds"],
}

# Neutralize phrases that would otherwise cause false positives for dairy/wheat
SAFE_SUBSTITUTIONS = {
    "peanut butter": "peanut_spread",
    "almond butter": "almond_spread",
    "cocoa butter": "cocoa_fat",
    "shea butter": "shea_fat",
    "coconut milk": "coconut_liquid",
    "coconut cream": "coconut_fat",
    "almond milk": "almond_liquid",
    "soy milk": "soy_liquid",
    "oat milk": "oat_liquid",
    "rice milk": "rice_liquid",
    "almond flour": "almond_meal",
    "coconut flour": "coconut_meal",
    "cream of tartar": "tartar_powder",
    "eggplant": "aubergine",
    "butternut squash": "winter_squash",
}

# Explicit free-from patterns per category.
# NOTE: "lactose-free" is deliberately NOT listed under Milk/Dairy because lactose-free milk contains dairy proteins.
EXPLICIT_FREE_CLAIMS: Dict[str, List[str]] = {
    "Wheat/Gluten": [r"\bgluten[- ]free\b", r"\bwheat[- ]free\b"],
    "Milk/Dairy": [r"\bdairy[- ]free\b", r"\bnon[- ]dairy\b", r"\bvegan\b"],
    "Eggs": [r"\begg[- ]free\b", r"\beggless\b", r"\bvegan\b"],
    "Soy": [r"\bsoy[- ]free\b"],
    "Peanuts": [r"\bpeanut[- ]free\b"],
    "Tree Nuts": [r"\btree[- ]nut[- ]free\b", r"\bnut[- ]free\b"],
}

# Critical raw source terms that, if explicitly present, contradict a free-from claim
CONTRADICTION_KEYWORDS: Dict[str, List[str]] = {
    "Wheat/Gluten": ["wheat", "barley", "rye", "semolina", "maida"],
    "Milk/Dairy": ["cow milk", "milk powder", "whey", "casein", "butter", "heavy cream"],
    "Eggs": ["egg yolk", "egg white", "albumen"],
    "Soy": ["soybean", "edamame"],
    "Peanuts": ["peanut", "peanuts"],
    "Tree Nuts": ["walnut", "cashew", "almond", "pecan"],
}


def _clean_and_neutralize(text: str) -> str:
    """Lowercase text and swap safe false-positive phrases."""
    t = text.lower()
    for phrase, replacement in SAFE_SUBSTITUTIONS.items():
        t = t.replace(phrase, replacement)
    return t


def detect_allergens(ingredients: List[str]) -> List[str]:
    """Detect presence of common allergens across a list of ingredient strings.

    Recognizes explicit 'free-from' claims (e.g. 'gluten-free flour', 'dairy-free cheese',
    'egg-free mayonnaise') without suppressing true positives (e.g. 'lactose-free milk'
    still triggers Milk/Dairy, and 'gluten-free flour with wheat' still triggers Wheat/Gluten).

    Args:
        ingredients: List of ingredient strings.

    Returns:
        Sorted list of detected allergen category names.
    """
    detected: Set[str] = set()

    for raw_ing in ingredients:
        raw_lower = raw_ing.lower()
        cleaned = _clean_and_neutralize(raw_ing)
        words = [w.strip(" ,.-/()[]") for w in cleaned.split()]

        for allergen_cat, keywords in ALLERGENS.items():
            # Check if this category matches keywords
            matched = False
            for kw in keywords:
                if " " in kw:
                    if kw in cleaned:
                        matched = True
                        break
                else:
                    if kw in words or any(w.startswith(kw) for w in words):
                        matched = True
                        break

            if matched:
                # Check for explicit free-from claim
                has_free_claim = False
                if allergen_cat in EXPLICIT_FREE_CLAIMS:
                    for pattern in EXPLICIT_FREE_CLAIMS[allergen_cat]:
                        if re.search(pattern, raw_lower):
                            has_free_claim = True
                            break

                # Check if the text explicitly contradicts the free claim
                has_contradiction = False
                if has_free_claim and allergen_cat in CONTRADICTION_KEYWORDS:
                    for c_kw in CONTRADICTION_KEYWORDS[allergen_cat]:
                        # Match contradiction keyword if not part of the claim itself
                        if re.search(r"\b" + re.escape(c_kw) + r"\b", raw_lower):
                            # Make sure it's not the 'wheat' in 'wheat-free'
                            if f"{c_kw}-free" not in raw_lower and f"{c_kw} free" not in raw_lower:
                                has_contradiction = True
                                break

                if has_free_claim and not has_contradiction:
                    # Explicit free-from claim declared and not contradicted: do not flag
                    continue

                detected.add(allergen_cat)

    return sorted(detected)
