# -*- coding: utf-8 -*-
"""Unit tests for robust ingredient parsing in utils/ner.py."""

import unittest
from utils.ner import extract_ingredients


class TestIngredientParser(unittest.TestCase):
    """Test suite for extract_ingredients() covering commas, newlines, bullets, and edge cases."""

    def test_comma_separated(self):
        text = "potato, onion, tomato"
        self.assertEqual(extract_ingredients(text), ["potato", "onion", "tomato"])

    def test_newline_separated(self):
        text = "potato\nonion\ntomato"
        self.assertEqual(extract_ingredients(text), ["potato", "onion", "tomato"])

    def test_mixed_delimiters(self):
        text = """potato
onion, tomato
cumin
salt"""
        self.assertEqual(
            extract_ingredients(text),
            ["potato", "onion", "tomato", "cumin", "salt"],
        )

    def test_bullet_prefixed(self):
        text = """- 2 potatoes
* 1 onion
• 3 tomatoes
- salt"""
        self.assertEqual(
            extract_ingredients(text),
            ["2 potatoes", "1 onion", "3 tomatoes", "salt"],
        )

    def test_numbered_list(self):
        text = """1. 2 cups flour
2) 1 tsp salt
3. 3 large eggs"""
        self.assertEqual(
            extract_ingredients(text),
            ["2 cups flour", "1 tsp salt", "3 large eggs"],
        )

    def test_semicolon_separated(self):
        text = "chicken; garlic; ginger; soy sauce"
        self.assertEqual(
            extract_ingredients(text),
            ["chicken", "garlic", "ginger", "soy sauce"],
        )

    def test_whitespace_and_empty_lines(self):
        text = "   \n\n  potato   ,  \n   onion  \n\n  "
        self.assertEqual(extract_ingredients(text), ["potato", "onion"])

    def test_empty_and_none_input(self):
        self.assertEqual(extract_ingredients(""), [])
        self.assertEqual(extract_ingredients("   "), [])
        self.assertEqual(extract_ingredients(None), [])

    def test_duplicate_deduplication(self):
        text = "potato, onion, Potato, ONION, tomato, potato"
        self.assertEqual(extract_ingredients(text), ["potato", "onion", "tomato"])


if __name__ == "__main__":
    unittest.main()
