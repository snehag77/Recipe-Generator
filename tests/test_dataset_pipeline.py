# -*- coding: utf-8 -*-
"""Test Suite for Dataset Preparation Pipeline (Unit Fixtures).

Creates a temporary test fixture to prove that:
- Malformed records, duplicates, and missing fields are rejected.
- Delimiter tags are formatted properly.
- Splits are disjoint (no duplicate leakage).
- All tests pass deterministically.
"""

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_recipe_dataset import process_dataset
from scripts.validate_recipe_dataset import run_validation


def test_pipeline_with_mock_fixture():
    temp_dir = tempfile.mkdtemp()
    raw_csv = os.path.join(temp_dir, "test_raw.csv")
    out_dir = os.path.join(temp_dir, "processed")

    # Sample rows: valid rows, duplicate row, missing direction row, empty row
    csv_content = """title,ingredients,directions,NER,source
"Potato Onion Curry","['2 potatoes', '1 onion', '1 tsp salt']","['Chop veggies.', 'Cook in pan for 15 mins.']","['potato', 'onion', 'salt']","RecipeNLG"
"Potato Onion Curry","['2 potatoes', '1 onion', '1 tsp salt']","['Chop veggies.', 'Cook in pan for 15 mins.']","['potato', 'onion', 'salt']","RecipeNLG"
"Tomato Soup","['3 tomatoes', '1 cup water']","['Blend tomatoes.', 'Simmer with water.']","['tomato', 'water']","RecipeNLG"
"Broken Recipe","['salt']","[]","['salt']","RecipeNLG"
"Empty Title","['sugar']","['Mix.']","['sugar']","RecipeNLG"
"Garlic Bread","['1 loaf bread', '2 tbsp garlic butter']","['Spread butter on bread.', 'Toast until crispy.']","['bread', 'garlic butter']","RecipeNLG"
"Pancake Medley","['1 cup flour', '1 egg', '1 cup milk']","['Whisk ingredients.', 'Cook on griddle until bubbles form.', 'Flip and serve.']","['flour', 'egg', 'milk']","RecipeNLG"
"Spicy Paneer","['200g paneer', '1 tsp chili powder']","['Cube paneer.', 'Sauté with chili.']","['paneer', 'chili powder']","RecipeNLG"
"Oatmeal Bowl","['1 cup oats', '2 cups water']","['Boil oats in water for 5 minutes.', 'Serve warm.']","['oats', 'water']","RecipeNLG"
"Fresh Salad","['1 cup lettuce', '1 tomato', '1 tbsp olive oil']","['Toss lettuce and tomato.', 'Drizzle with olive oil.']","['lettuce', 'tomato', 'olive oil']","RecipeNLG"
"Vegetable Soup","['1 carrot', '1 celery', '4 cups broth']","['Chop vegetables.', 'Simmer in broth for 20 minutes.']","['carrot', 'celery', 'broth']","RecipeNLG"
"Fried Rice","['2 cups cooked rice', '1 cup mixed vegetables', '2 tbsp soy sauce']","['Heat oil in wok.', 'Add vegetables and rice.', 'Stir fry with soy sauce.']","['rice', 'vegetables', 'soy sauce']","RecipeNLG"
"""
    with open(raw_csv, "w", encoding="utf-8") as f:
        f.write(csv_content.strip())

    try:
        # Run preparation
        process_dataset(input_path=raw_csv, output_dir=out_dir)

        # Run validation
        is_valid = run_validation(out_dir)
        assert is_valid, "Validation suite failed on pipeline output!"

        # Verify duplicate was dropped (12 rows - 1 duplicate - 2 malformed = 9 valid)
        with open(os.path.join(out_dir, "dataset_statistics.json")) as sf:
            stats = json.load(sf)
            assert stats["duplicates_removed"] == 1, f"Expected 1 duplicate removed, got {stats['duplicates_removed']}"
            assert stats["valid_kept"] == 9, f"Expected 9 valid kept, got {stats['valid_kept']}"

        print("\n✅ Fixture Unit Test Passed: Pipeline correctly cleaned, deduplicated, and split records!")
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_pipeline_with_mock_fixture()
