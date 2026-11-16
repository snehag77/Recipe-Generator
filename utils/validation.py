# -*- coding: utf-8 -*-
"""U5b: Quantity and measurement validation.

Parses integers, decimals, and fractional quantities (including unicode fractions).
Validates bounds (negative, zero, unrealistically large) and standard culinary units.
Normalizes singular and plural units so boundary checks remain consistent.
"""

import re
from typing import List, Optional, Tuple

UNICODE_FRACTION_MAP = {
    "½": " 1/2 ",
    "⅓": " 1/3 ",
    "⅔": " 2/3 ",
    "¼": " 1/4 ",
    "¾": " 3/4 ",
    "⅛": " 1/8 ",
    "⅜": " 3/8 ",
    "⅝": " 5/8 ",
    "⅞": " 7/8 ",
}

# Mapping all unit variations to their canonical singular forms
UNIT_NORMALIZATION = {
    # Metric Volume
    "ml": "ml",
    "milliliter": "ml",
    "milliliters": "ml",
    "millilitre": "ml",
    "millilitres": "ml",
    "l": "l",
    "liter": "l",
    "liters": "l",
    "litre": "l",
    "litres": "l",
    # Metric Mass
    "mg": "mg",
    "milligram": "mg",
    "milligrams": "mg",
    "g": "g",
    "gram": "g",
    "grams": "g",
    "kg": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    # US / Imperial Volume & Mass
    "tsp": "tsp",
    "teaspoon": "tsp",
    "teaspoons": "tsp",
    "tbsp": "tbsp",
    "tablespoon": "tbsp",
    "tablespoons": "tbsp",
    "cup": "cup",
    "cups": "cup",
    "c": "cup",
    "fl oz": "fl oz",
    "fluid ounce": "fl oz",
    "fluid ounces": "fl oz",
    "pint": "pint",
    "pints": "pint",
    "pt": "pint",
    "quart": "quart",
    "quarts": "quart",
    "qt": "quart",
    "gallon": "gallon",
    "gallons": "gallon",
    "gal": "gallon",
    "oz": "oz",
    "ounce": "oz",
    "ounces": "oz",
    "lb": "lb",
    "lbs": "lb",
    "pound": "lb",
    "pounds": "lb",
    # Portions / Counts
    "pinch": "pinch",
    "pinches": "pinch",
    "dash": "pinch",
    "dashes": "pinch",
    "clove": "clove",
    "cloves": "clove",
    "can": "can",
    "cans": "can",
    "slice": "slice",
    "slices": "slice",
    "piece": "piece",
    "pieces": "piece",
    "pack": "pack",
    "package": "pack",
    "packages": "pack",
    "bunch": "bunch",
    "bunches": "bunch",
    "stick": "stick",
    "sticks": "stick",
}

VALID_UNITS = set(UNIT_NORMALIZATION.keys())

# Upper bound thresholds mapped to normalized canonical units
UPPER_BOUNDS = {
    "kg": 10.0,
    "g": 5000.0,
    "mg": 50000.0,
    "l": 8.0,
    "ml": 8000.0,
    "cup": 15.0,
    "tbsp": 40.0,
    "tsp": 60.0,
    "pint": 16.0,
    "quart": 8.0,
    "gallon": 4.0,
    "lb": 15.0,
    "oz": 160.0,
    "fl oz": 160.0,
    "pinch": 20.0,
    "clove": 30.0,
    "can": 20.0,
    "slice": 30.0,
    "piece": 50.0,
    "pack": 10.0,
    "bunch": 15.0,
    "stick": 20.0,
    "default_count": 50.0,
}

# Common non-culinary units (distance, area, electrical, time) to explicitly flag
NON_CULINARY_UNITS = {
    "kilometer", "kilometers", "km",
    "meter", "meters", "m",
    "centimeter", "centimeters", "cm",
    "millimeter", "millimeters", "mm",
    "mile", "miles",
    "yard", "yards",
    "foot", "feet", "ft",
    "inch", "inches",
    "square-foot", "square-feet", "sq ft",
    "square-meter", "square-meters",
    "acre", "acres",
    "volt", "volts",
    "watt", "watts",
    "hour", "hours",
    "minute", "minutes",
    "second", "seconds",
}


def _parse_numeric_fraction(token: str) -> Optional[float]:
    """Parse numeric strings like '1/2', '3/4', '1.5', '2'."""
    token = token.strip()
    if "/" in token:
        parts = token.split("/")
        if len(parts) == 2:
            try:
                num = float(parts[0])
                denom = float(parts[1])
                if denom == 0:
                    return None
                return num / denom
            except ValueError:
                return None
    try:
        return float(token)
    except ValueError:
        return None


def parse_quantity(ingredient: str) -> Tuple[Optional[float], Optional[str]]:
    """Extract leading numeric quantity and unit from an ingredient string.

    Handles mixed numbers ('1 1/2'), standalone fractions ('1/2', '3/4'),
    unicode fractions ('½', '¾'), decimals ('1.5'), and standard units.
    Returns:
        (quantity_float or None, unit_str or None)
    """
    s = ingredient.strip()
    for u_char, val_str in UNICODE_FRACTION_MAP.items():
        s = s.replace(u_char, val_str)

    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return None, None

    # Check for negative numbers
    is_negative = False
    if s.startswith("-"):
        is_negative = True
        s = s.lstrip("-").strip()

    # Regex matching:
    # 1) Whole number + space + fraction (e.g. "1 1/2")
    match_mixed = re.match(r"^(\d+)\s+(\d+/\d+)\s*(.*)$", s)
    if match_mixed:
        whole = float(match_mixed.group(1))
        frac = _parse_numeric_fraction(match_mixed.group(2)) or 0.0
        qty = whole + frac
        remainder = match_mixed.group(3).strip()
    else:
        # 2) Standalone fraction OR decimal OR integer.
        # CRITICAL: \d+/\d+ MUST precede \d+(?:\.\d+)? to prevent greedy integer matching on fractions.
        match_single = re.match(r"^(\d+/\d+|\d+(?:\.\d+)?)\s*(.*)$", s)
        if match_single:
            qty = _parse_numeric_fraction(match_single.group(1))
            remainder = match_single.group(2).strip()
        else:
            return None, None

    if qty is None:
        return None, None

    if is_negative:
        qty = -qty

    # Look for unit in the remainder
    unit = None
    if remainder:
        words = remainder.split()
        first_word = words[0].lower().rstrip(",.;")
        if first_word in VALID_UNITS or first_word in NON_CULINARY_UNITS:
            unit = first_word
        elif len(words) >= 2:
            two_words = f"{words[0].lower()} {words[1].lower()}".rstrip(",.;")
            if two_words in VALID_UNITS or two_words in NON_CULINARY_UNITS:
                unit = two_words

    return qty, unit


def validate_quantities(ingredients: List[str]) -> List[str]:
    """Validate quantity and measurement consistency across ingredient items.

    Returns a list of clear warning messages for:
    - Zero quantities
    - Negative quantities
    - Unusually large quantities exceeding plausible upper bounds (singular & plural normalized)
    - Unrecognized or non-culinary measurement units
    """
    warnings: List[str] = []

    for ing in ingredients:
        raw = ing.strip()
        if not raw:
            continue

        qty, unit = parse_quantity(raw)

        if qty is not None:
            if qty < 0:
                warnings.append(
                    f"Negative quantity ({qty}) detected for '{raw}'. Quantities must be positive."
                )
            elif qty == 0:
                warnings.append(
                    f"Zero quantity ({qty}) specified for '{raw}'. Please specify a usable amount."
                )
            else:
                # Check for non-culinary unit
                if unit and unit in NON_CULINARY_UNITS:
                    warnings.append(
                        f"Non-culinary measurement unit '{unit}' detected for '{raw}'. Please use standard culinary units."
                    )
                    continue

                # Normalize unit to canonical form for upper bound lookup
                canonical_unit = UNIT_NORMALIZATION.get(unit, unit) if unit else None
                threshold = (
                    UPPER_BOUNDS.get(canonical_unit, UPPER_BOUNDS["default_count"])
                    if canonical_unit
                    else UPPER_BOUNDS["default_count"]
                )
                if qty > threshold:
                    display_unit = unit or "units"
                    warnings.append(
                        f"Unusually large quantity ({qty:g} {display_unit}) for '{raw}'. Please verify recipe scale."
                    )
        else:
            # Check if there is an attempted but malformed fraction or negative
            if re.search(r"-\d", raw):
                warnings.append(
                    f"Possible negative amount in '{raw}'. Quantities must be positive."
                )
            elif re.search(r"\b0\s+[a-zA-Z]", raw):
                warnings.append(
                    f"Possible zero quantity in '{raw}'. Please specify a usable amount."
                )

    return warnings
