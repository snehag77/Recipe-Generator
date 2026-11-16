# -*- coding: utf-8 -*-
"""Unit tests for dish title and ingredient alignment validation in utils/dish_validation.py."""

import unittest
from utils.dish_validation import validate_dish_title_alignment


class TestDishValidation(unittest.TestCase):
    """Test suite verifying advisory warnings for dish title and ingredient mismatches."""

    def test_dal_mismatch_warning(self):
        """User's exact example: dal requested with potatoes, onions, tomatoes (no lentils)."""
        ings = ["potato", "onion", "tomato", "cumin", "oil", "salt"]
        warning = validate_dish_title_alignment("dal", ings)
        self.assertIsNotNone(warning)
        self.assertIn("Dal", warning)
        self.assertIn("lentils", warning)

    def test_dal_with_lentils_no_warning(self):
        """Dal requested with lentils included should raise no warning."""
        ings = ["yellow moong dal", "onion", "tomato", "cumin", "ghee"]
        warning = validate_dish_title_alignment("Dal Tadka", ings)
        self.assertIsNone(warning)

    def test_biryani_mismatch_warning(self):
        """Biryani requested without rice."""
        ings = ["chicken", "yogurt", "garam masala", "mint", "onions"]
        warning = validate_dish_title_alignment("Chicken Biryani", ings)
        self.assertIsNotNone(warning)
        self.assertIn("rice", warning)

    def test_biryani_with_rice_no_warning(self):
        """Biryani requested with basmati rice."""
        ings = ["chicken", "basmati rice", "yogurt", "spices"]
        warning = validate_dish_title_alignment("Chicken Biryani", ings)
        self.assertIsNone(warning)

    def test_omelette_mismatch_warning(self):
        """Omelette requested without eggs."""
        ings = ["bell pepper", "onion", "cheese", "salt"]
        warning = validate_dish_title_alignment("Cheese Omelette", ings)
        self.assertIsNotNone(warning)
        self.assertIn("eggs", warning)

    def test_empty_title_no_warning(self):
        """Empty or None title hint should never produce a warning."""
        ings = ["potato", "onion"]
        self.assertIsNone(validate_dish_title_alignment("", ings))
        self.assertIsNone(validate_dish_title_alignment("   ", ings))
        self.assertIsNone(validate_dish_title_alignment(None, ings))

    def test_generic_title_no_warning(self):
        """Generic style like 'Curry' or 'Soup' without archetype mismatch."""
        ings = ["potato", "onion", "tomato"]
        self.assertIsNone(validate_dish_title_alignment("Curry", ings))
        self.assertIsNone(validate_dish_title_alignment("Stir Fry", ings))


if __name__ == "__main__":
    unittest.main()
