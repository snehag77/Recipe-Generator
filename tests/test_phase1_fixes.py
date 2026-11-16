# -*- coding: utf-8 -*-
"""Phase 1 Automated Regression Test Suite.

Covers:
1. Fraction parsing tests
2. Decimal & integer parsing tests
3. Negative & zero quantity validation tests
4. Singular/plural unit normalization tests
5. Upper-bound validation tests
6. Allergen true-positive tests
7. Allergen false-positive / declared claims tests
8. Model Training page configuration tests
"""

import json
import os
import sys
import unittest

# Ensure project root is in path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils import allergen, validation


class TestQuantityParsingAndBounds(unittest.TestCase):
    """Test quantity parsing, fraction resolution, and boundary validation."""

    def test_fractions(self):
        cases = [
            ("1/2 tsp salt", 0.5, "tsp"),
            ("3/4 cup milk", 0.75, "cup"),
            ("1 1/2 cups flour", 1.5, "cups"),
            ("2 1/4 tsp sugar", 2.25, "tsp"),
            ("½ tsp salt", 0.5, "tsp"),
            ("¾ cup milk", 0.75, "cup"),
            ("1 ½ cups flour", 1.5, "cups"),
        ]
        for raw, exp_qty, exp_unit in cases:
            qty, unit = validation.parse_quantity(raw)
            self.assertIsNotNone(qty, f"Failed to parse quantity for '{raw}'")
            self.assertAlmostEqual(qty, exp_qty, places=3, msg=f"Mismatch for '{raw}'")
            self.assertEqual(unit, exp_unit, f"Unit mismatch for '{raw}'")

    def test_decimals_and_integers(self):
        cases = [
            ("1.5 cups water", 1.5, "cups"),
            ("2 cups rice", 2.0, "cups"),
            ("10 grams salt", 10.0, "grams"),
            ("1 apple", 1.0, None),
        ]
        for raw, exp_qty, exp_unit in cases:
            qty, unit = validation.parse_quantity(raw)
            self.assertAlmostEqual(qty, exp_qty, places=3)
            self.assertEqual(unit, exp_unit)

    def test_negative_and_zero_quantities(self):
        warn_zero = validation.validate_quantities(["0 cups flour"])
        self.assertTrue(any("Zero quantity" in w for w in warn_zero), "Failed to flag zero quantity")

        warn_neg = validation.validate_quantities(["-2 cups sugar"])
        self.assertTrue(any("Negative quantity" in w for w in warn_neg), "Failed to flag negative quantity")

    def test_plural_unit_normalization_and_bounds(self):
        # 50 cups salt should trigger upper bound warning (threshold 15 cups)
        warn_plural = validation.validate_quantities(["50 cups salt"])
        self.assertTrue(
            any("Unusually large quantity" in w and "50 cups" in w for w in warn_plural),
            f"Expected upper bound warning for 50 cups salt, got: {warn_plural}"
        )

        # 30 teaspoons should NOT trigger warning (threshold 60 tsp)
        warn_normal = validation.validate_quantities(["30 teaspoons salt"])
        self.assertEqual(len(warn_normal), 0, f"Unexpected warning for 30 teaspoons: {warn_normal}")

        # 70 teaspoons should trigger warning
        warn_tsp_high = validation.validate_quantities(["70 teaspoons salt"])
        self.assertTrue(any("Unusually large quantity" in w for w in warn_tsp_high))

        # 1000 kg chicken should trigger warning (threshold 10 kg)
        warn_kg = validation.validate_quantities(["1000 kg chicken"])
        self.assertTrue(any("Unusually large quantity" in w for w in warn_kg))

        # 100 grams and 5000 ml are within safe thresholds
        self.assertEqual(len(validation.validate_quantities(["100 grams salt"])), 0)
        self.assertEqual(len(validation.validate_quantities(["5000 ml oil"])), 0)

    def test_non_culinary_units(self):
        warn_km = validation.validate_quantities(["2 kilometers milk"])
        self.assertTrue(any("Non-culinary measurement unit" in w for w in warn_km))


class TestAllergenDetection(unittest.TestCase):
    """Test allergen detection for true positives, declared free claims, and contradictions."""

    def test_true_positives(self):
        self.assertIn("Wheat/Gluten", allergen.detect_allergens(["wheat flour"]))
        self.assertIn("Milk/Dairy", allergen.detect_allergens(["regular milk"]))
        self.assertIn("Peanuts", allergen.detect_allergens(["peanut butter"]))
        self.assertIn("Tree Nuts", allergen.detect_allergens(["almond flour"]))
        self.assertIn("Milk/Dairy", allergen.detect_allergens(["butter"]))

        soy_res = allergen.detect_allergens(["soy sauce"])
        self.assertIn("Soy", soy_res)
        self.assertIn("Wheat/Gluten", soy_res)

        # Lactose-free milk IS STILL DAIRY (cow's milk contains dairy proteins)
        lactose_res = allergen.detect_allergens(["lactose-free milk"])
        self.assertIn("Milk/Dairy", lactose_res, "Lactose-free milk must still flag Milk/Dairy")

    def test_false_positive_mitigations(self):
        self.assertEqual(allergen.detect_allergens(["gluten-free flour"]), [])
        self.assertEqual(allergen.detect_allergens(["dairy-free cheese"]), [])
        self.assertEqual(allergen.detect_allergens(["egg-free mayonnaise"]), [])
        self.assertEqual(allergen.detect_allergens(["coconut milk"]), [])
        self.assertEqual(allergen.detect_allergens(["eggplant"]), [])
        self.assertEqual(allergen.detect_allergens(["butternut squash"]), [])

    def test_contradictions_preserve_warnings(self):
        # If an item claims gluten-free but explicitly contains wheat, warning must trigger
        self.assertIn("Wheat/Gluten", allergen.detect_allergens(["gluten-free flour with wheat"]))
        self.assertIn("Milk/Dairy", allergen.detect_allergens(["dairy-free cheese with whey"]))


class TestModelTrainingPageExtraction(unittest.TestCase):
    """Test that Page 4 extracts nested config values without displaying 'None'."""

    def test_pilot_metrics_extraction(self):
        path = os.path.join(ROOT_DIR, "data", "outputs", "finetuned_metrics.json")
        self.assertTrue(os.path.isfile(path))
        with open(path) as f:
            data = json.load(f)
        cfg = data.get("config", {})
        lr = cfg.get("initial_learning_rate") or cfg.get("learning_rate")
        batch = cfg.get("batch_size")
        epochs = cfg.get("epochs")
        self.assertIsNotNone(lr)
        self.assertIsNotNone(batch)
        self.assertIsNotNone(epochs)
        self.assertEqual(batch, 4)
        self.assertEqual(epochs, 1)

    def test_full_metrics_extraction(self):
        path = os.path.join(ROOT_DIR, "data", "outputs", "recipe_gpt2_full_metrics.json")
        self.assertTrue(os.path.isfile(path))
        with open(path) as f:
            data = json.load(f)
        cfg = data.get("config", {})
        lr = cfg.get("initial_learning_rate") or cfg.get("learning_rate")
        batch = cfg.get("batch_size")
        epochs = cfg.get("epochs")
        max_len = cfg.get("max_len")
        self.assertIsNotNone(lr)
        self.assertEqual(batch, 4)
        self.assertEqual(epochs, 1)
        self.assertEqual(max_len, 512)


if __name__ == "__main__":
    unittest.main()
