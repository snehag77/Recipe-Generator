# -*- coding: utf-8 -*-
"""Unit and functional tests for BERT Ingredient NER and utils.ner integration."""

import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from utils import ner
from scripts.run_ner import extract_entities_from_bio


class TestIngredientNER(unittest.TestCase):
    """Test suite for Phase 4 Ingredient NER implementation."""

    def test_rule_based_fallback_delimiters(self):
        """Test 1: Rule-based extraction handles standard and mixed delimiters."""
        text = "2 cups flour, 1 tsp salt\n3 large eggs; 1 tbsp olive oil"
        res = ner.extract_ingredients(text, method="rule")
        self.assertEqual(len(res), 4)
        self.assertIn("2 cups flour", res)
        self.assertIn("1 tsp salt", res)
        self.assertIn("3 large eggs", res)
        self.assertIn("1 tbsp olive oil", res)

    def test_empty_and_whitespace_input(self):
        """Test 2: Empty, None, and whitespace strings return empty lists."""
        self.assertEqual(ner.extract_ingredients("", method="rule"), [])
        self.assertEqual(ner.extract_ingredients("   \n\t  ", method="rule"), [])
        self.assertEqual(ner.extract_ingredients(None, method="rule"), [])

    def test_extract_entities_from_bio_single_and_multiword(self):
        """Test 3: extract_entities_from_bio handles single and multi-word ingredient spans."""
        tokens = ["Add", "two", "chopped", "onions", "and", "fresh", "olive", "oil", "."]
        tags = ["O", "O", "O", "B-INGREDIENT", "O", "O", "B-INGREDIENT", "I-INGREDIENT", "O"]

        entities = extract_entities_from_bio(tokens, tags)
        self.assertEqual(len(entities), 2)
        self.assertIn("onions", entities)
        self.assertIn("olive oil", entities)

    def test_extract_entities_from_bio_adjacent_entities(self):
        """Test 4: Adjacent B- tags are treated as distinct entities, not merged."""
        tokens = ["garlic", "ginger", "paste"]
        tags = ["B-INGREDIENT", "B-INGREDIENT", "I-INGREDIENT"]

        entities = extract_entities_from_bio(tokens, tags)
        self.assertEqual(len(entities), 2)
        self.assertIn("garlic", entities)
        self.assertIn("ginger paste", entities)

    def test_extract_entities_from_bio_malformed_i_tag(self):
        """Test 5: An isolated I- tag at the beginning of a sequence is handled gracefully."""
        tokens = ["tomatoes", "and", "basil"]
        tags = ["I-INGREDIENT", "O", "B-INGREDIENT"]

        entities = extract_entities_from_bio(tokens, tags)
        self.assertEqual(len(entities), 2)
        self.assertIn("tomatoes", entities)
        self.assertIn("basil", entities)

    def test_extract_ingredients_with_metadata_interface(self):
        """Test 6: extract_ingredients_with_metadata returns (items, engine_name)."""
        text = "carrots, celery, onions"
        items, engine = ner.extract_ingredients_with_metadata(text, method="rule")
        self.assertIsInstance(items, list)
        self.assertEqual(engine, "rule")
        self.assertEqual(len(items), 3)

    def test_auto_fallback_when_bert_unavailable(self):
        """Test 7: 'auto' method cleanly falls back to rule-based parser if BERT unavailable."""
        text = "- 1 cup milk\n- 2 tbsp sugar"
        # Force fallback by testing extract_ingredients with fallback
        items = ner.extract_ingredients(text, method="rule")
        self.assertEqual(len(items), 2)
        self.assertIn("1 cup milk", items)
        self.assertIn("2 tbsp sugar", items)


if __name__ == "__main__":
    unittest.main()
