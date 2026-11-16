# -*- coding: utf-8 -*-
"""Comprehensive verification suite for Stage 2K application corrections and enhancements."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import allergen, gpt2, ner, prompt, validation


def compute_md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class TestStage2KEnhancements(unittest.TestCase):
    """Test suite verifying all Stage 2K requirements."""

    def test_ingredient_parser_delimiters(self):
        """Test 1: Comma, newline, and mixed input handling in utils/ner.py."""
        # Comma input
        self.assertEqual(ner.extract_ingredients("potato, onion, tomato"), ["potato", "onion", "tomato"])

        # Newline input
        self.assertEqual(ner.extract_ingredients("potato\nonion\ntomato"), ["potato", "onion", "tomato"])

        # Mixed input with bullets
        mixed = """potato
onion, tomato
- cumin
* 1 tsp salt"""
        self.assertEqual(
            ner.extract_ingredients(mixed),
            ["potato", "onion", "tomato", "cumin", "1 tsp salt"],
        )

        # Empty / whitespace
        self.assertEqual(ner.extract_ingredients(""), [])
        self.assertEqual(ner.extract_ingredients("   \n\n  "), [])

        # Deduplication
        self.assertEqual(ner.extract_ingredients("apple, banana, Apple, BANANA"), ["apple", "banana"])

    def test_formatted_recipe_parsing(self):
        """Test 2: Recipe parsing handles complete and partial outputs safely."""
        full_sample = """<RECIPE>
<TITLE>
Potato Curry
</TITLE>
<QUANTITIES>
- 2 potatoes
- 1 onion
- 1 tsp salt
</QUANTITIES>
<STEPS>
1. Peel and cube potatoes.
2. Cook with onions and salt.
3. Serve hot.
</STEPS>
</RECIPE>"""
        parsed = prompt.parse_recipe_output(full_sample)
        self.assertEqual(parsed["title"], "Potato Curry")
        self.assertEqual(len(parsed["quantities"]), 3)
        self.assertEqual(len(parsed["steps"]), 3)

        # Unstructured text handling (Base GPT-2 simulation)
        unstructured = "This is a random text about cooking without tags."
        parsed_bad = prompt.parse_recipe_output(unstructured)
        self.assertIsNone(parsed_bad["title"])
        self.assertEqual(parsed_bad["quantities"], [])
        self.assertEqual(parsed_bad["steps"], [])
        self.assertEqual(parsed_bad["raw_text"], unstructured)

    def test_stage2g_held_out_metrics_loading(self):
        """Test 3: Stage 2G held-out metrics match source JSON files."""
        # Dynamically import page 4
        p4_path = PROJECT_ROOT / "app" / "pages" / "4_Model_Training.py"
        spec = importlib.util.spec_from_file_location("p4_test", str(p4_path))
        p4 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(p4)

        metrics = p4._load_stage2g_metrics()
        self.assertIsNotNone(metrics, "Failed to load Stage 2G held-out metrics!")
        self.assertEqual(metrics["num_records"], 250)

        s2f = metrics["stage2f"]
        self.assertAlmostEqual(s2f["test_loss"], 0.8855, places=3)
        self.assertAlmostEqual(s2f["test_perplexity"], 2.42, places=1)
        self.assertAlmostEqual(s2f["structural_validity"], 90.4, places=1)
        self.assertAlmostEqual(s2f["ingredient_recall"], 89.21, places=1)

    def test_pages_import_cleanly(self):
        """Test 4: All 5 application pages import and define run() methods."""
        pages = [
            ("main", PROJECT_ROOT / "app" / "main.py"),
            ("p1", PROJECT_ROOT / "app" / "pages" / "1_Recipe_Generator.py"),
            ("p2", PROJECT_ROOT / "app" / "pages" / "2_Ingredient_Analysis.py"),
            ("p3", PROJECT_ROOT / "app" / "pages" / "3_Model_Comparison.py"),
            ("p4", PROJECT_ROOT / "app" / "pages" / "4_Model_Training.py"),
        ]
        for name, path in pages:
            self.assertTrue(path.is_file(), f"Missing page: {path}")
            spec = importlib.util.spec_from_file_location(name, str(path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if "pages" in str(path):
                self.assertTrue(hasattr(mod, "run"), f"Page {name} missing run() method")

    def test_safety_validation_integration(self):
        """Test 5: Allergen screening and quantity bounds on custom inputs."""
        custom_input = "50 cups milk, 1/2 tsp salt, 3 eggs"
        ings = ner.extract_ingredients(custom_input)
        self.assertEqual(len(ings), 3)

        allergens = allergen.detect_allergens(ings)
        self.assertIn("Milk/Dairy", allergens)
        self.assertIn("Eggs", allergens)

        qty_warns = validation.validate_quantities(ings)
        self.assertTrue(any("50 cups" in w for w in qty_warns))

    def test_checkpoints_unchanged(self):
        """Test 6: Strict protection - model checkpoints remain byte-for-byte identical."""
        expected_md5s = {
            PROJECT_ROOT / "models" / "recipe_gpt2_full" / "recipe_gpt2_full_best.weights.h5": "d95eb61d471f183a67a0e7fb920369d0",
            PROJECT_ROOT / "models" / "gpt2_finetuned.weights.h5": "a0ddb523b21e820c2fdcc871f2fae2a1",
            PROJECT_ROOT / "models" / "recipe_gpt2" / "recipe_gpt2_best.weights.h5": "0f733e894f4a83087164dfc1b7fe9a9a",
        }
        for path, exp_md5 in expected_md5s.items():
            self.assertTrue(path.is_file(), f"Missing checkpoint: {path}")
            actual_md5 = compute_md5(path)
            self.assertEqual(actual_md5, exp_md5, f"Checkpoint MD5 mismatch for {path.name}")

    def test_datasets_unchanged(self):
        """Test 7: Strict protection - processed datasets remain byte-for-byte identical."""
        expected_md5s = {
            PROJECT_ROOT / "data" / "processed" / "recipes_train.jsonl": "6d2487aa1719146437892318f5820dc9",
            PROJECT_ROOT / "data" / "processed" / "recipes_validation.jsonl": "01b7bf6c2d0ee71093f38e01c000a045",
            PROJECT_ROOT / "data" / "processed" / "recipes_test.jsonl": "11b581b58e352672e9ef2f7130cd10bb",
        }
        for path, exp_md5 in expected_md5s.items():
            self.assertTrue(path.is_file(), f"Missing dataset split: {path}")
            actual_md5 = compute_md5(path)
            self.assertEqual(actual_md5, exp_md5, f"Dataset split MD5 mismatch for {path.name}")


if __name__ == "__main__":
    unittest.main()
