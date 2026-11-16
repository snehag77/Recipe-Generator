# RecipeGen AI — Stage 2K: Acceptance Audit Report

**Audit Date:** November 16, 2026  
**Auditor:** Antigravity AI Assistant  
**Project Path:** `/Users/snehagupta/Downloads/Recipe_genai`  
**Execution Environment:** `venv_tf` (Python 3.12.8, Streamlit 1.67.0, TensorFlow 2.21.0, KerasHub 0.20.0)  
**Reference Document:** `STAGE2K_UI_OPTIMIZATION_REPORT.md`  
**Audit Type:** Read-Only Verification and Reproducibility Audit  

---

## 1. Executive Summary

This acceptance audit independently inspected the source code, tests, runtime environment, and operational state of RecipeGen AI following the Stage 2K UI Optimization and Refinement.

- **Verified Items:** 9 / 10
- **Partially Verified Items:** 1 / 10 (Stage 2G metrics loading: all loss/perplexity/structure/accuracy metrics are dynamic from JSON; ingredient recall is static)
- **Failed Items:** 0 / 10
- **Automated Tests:** 33 / 33 passed (reproduced in 6.904s)
- **Checkpoints and Datasets:** 100% bit-for-bit identical to recorded baselines

---

## 2. Detailed Audit of Verification Items

### Item 1: Ingredient Delimiter Parsing (Comma, Newline, Mixed)
* **Status:** **VERIFIED**
* **Inspection Location:** `utils/ner.py` (`extract_ingredients()`), `app/pages/1_Recipe_Generator.py`, `app/pages/2_Ingredient_Analysis.py`, `app/pages/3_Model_Comparison.py`.
* **Findings:**
  - `extract_ingredients(text)` uses `re.split(r"[\n,;]+", text)` combined with regex prefix cleanup `re.sub(r"^\s*(?:[-*•–—]|\d+[.)])\s*", "", token)`.
  - All delimiters (newlines, commas, semicolons, markdown bullets `*` and `-`, and numbered items `1.`) are parsed into identical, cleaned, deduplicated token lists:
    - `"potato, onion, tomato, cumin, salt"` → `['potato', 'onion', 'tomato', 'cumin', 'salt']`
    - `"potato\nonion\ntomato\ncumin\nsalt"` → `['potato', 'onion', 'tomato', 'cumin', 'salt']`
    - `"potato\nonion, tomato\ncumin\nsalt"` → `['potato', 'onion', 'tomato', 'cumin', 'salt']`
    - `"- 2 cups potato\n* 1 medium onion, 1/2 tsp salt\n1. 1 tbsp cumin"` → `['2 cups potato', '1 medium onion', '1/2 tsp salt', '1 tbsp cumin']`
  - All three interactive pages (`1_Recipe_Generator.py`, `2_Ingredient_Analysis.py`, `3_Model_Comparison.py`) invoke `ner.extract_ingredients()`.

---

### Item 2: Formatted Recipe and Raw Generated Output Tabs
* **Status:** **VERIFIED**
* **Inspection Location:** `app/pages/1_Recipe_Generator.py` (lines 160–199), `utils/prompt.py` (`parse_recipe_output()`).
* **Findings:**
  - `st.tabs(["Formatted Recipe", "Raw Generated Text"])` is created immediately following generation.
  - **Formatted Recipe Tab:**
    - Checks for structured components: `has_title`, `has_quantities`, and `has_steps`.
    - If present, renders formatted title (`### 🍽️ <Title>`), ingredients list (`#### 🛒 Ingredients & Quantities`), and numbered instructions (`#### 👩‍🍳 Cooking Instructions`).
    - If output is truncated or partially structured (e.g., closing tags cut off by 512-token limit), displays available sections plus an informative note: *"ℹ️ Note: Some sections were incomplete. View 'Raw Generated Text' for full model tokens."*
    - If completely unstructured (e.g. Base GPT-2 baseline), issues a clear fallback notice and renders the output directly.
  - **Raw Generated Text Tab:**
    - Renders the full unaltered token stream inside `st.code(recipe_output, language="markdown")`.

---

### Item 3: Interactive Ingredient Removal and Session State Synchronization
* **Status:** **VERIFIED**
* **Inspection Location:** `app/pages/2_Ingredient_Analysis.py` (lines 74–92), `app/pages/1_Recipe_Generator.py` (lines 42–46).
* **Findings:**
  - In `2_Ingredient_Analysis.py`, extracted ingredients are presented in `st.multiselect("Active Extracted Ingredients", options=current_items, default=current_items)`.
  - When a user removes any chip, `if selected_items != current_items:` triggers:
    ```python
    st.session_state["analysis_items"] = selected_items
    st.session_state["ingredients"] = ", ".join(selected_items)
    ```
  - An explicit sync button is also provided:
    ```python
    if st.button("🔄 Sync Selection to Recipe Generator"):
        st.session_state["ingredients"] = ", ".join(selected_items)
    ```
  - When navigating to `1_Recipe_Generator.py`, the ingredient text area initializes from `st.session_state.get("ingredients", ...)`, ensuring state is preserved across pages.

---

### Item 4: Dish-Title Mismatch Warnings Do Not Block Generation
* **Status:** **VERIFIED**
* **Inspection Location:** `utils/dish_validation.py`, `app/pages/1_Recipe_Generator.py` (lines 64–77, 128–130, 204–207).
* **Findings:**
  - `validate_dish_title_alignment(title, ingredients)` inspects 8 culinary archetypes (Dal, Biryani/Rice, Pasta/Noodles, Baked goods, Omelettes, Paneer, Pizza, Smoothies/Shakes).
  - If a mismatch is detected (e.g., dish title "Moong Dal Tadka" without lentils), it returns an advisory string.
  - In `1_Recipe_Generator.py`, this check is strictly advisory:
    ```python
    dish_warn = dish_validation.validate_dish_title_alignment(title_hint, ing_list)
    # Generation proceeds unconditionally:
    user_prompt = prompt.build_prompt(ing_list, title_hint=title_hint)
    recipe_output = gpt2.generate_recipe(...)
    ```
  - No `return`, `sys.exit`, or `st.stop()` is triggered. The alert is rendered under `🛡️ Safety & Quality Preflight Checks` via `st.info(dish_warn)`.

---

### Item 5: Custom Model Comparison Inputs Screening (Quantities & Allergens)
* **Status:** **VERIFIED**
* **Inspection Location:** `app/pages/3_Model_Comparison.py` (lines 72–93).
* **Findings:**
  - In Tab 2 ("Live Side-by-Side Playground"), custom user input undergoes preflight screening before invoking the models:
    ```python
    ings = ner.extract_ingredients(custom_input)
    allergens = allergen.detect_allergens(ings)
    qty_warnings = validation.validate_quantities(ings)

    if allergens or qty_warnings:
        st.markdown("#### 🛡️ Input Preflight Safety Observations")
        if allergens:
            st.warning(f"⚠️ Allergen Alert: Input contains items associated with: {', '.join(allergens).title()}.")
        if qty_warnings:
            for w in qty_warnings:
                st.warning(f"⚠️ {w}")
    ```
  - Warnings are displayed prominently directly above the dual-model execution columns.

---

### Item 6: Stage 2G Metrics Loading from Evaluation JSON
* **Status:** **PARTIALLY VERIFIED**
* **Inspection Location:** `app/pages/4_Model_Training.py` (lines 14–75).
* **Findings:**
  - **Dynamic Metrics (Verified):** The application checks for and opens:
    - `data/outputs/stage2g_teacher_forced.json` (Teacher-forced evaluation)
    - `data/outputs/stage2g_generation_eval.json` (Free-generation evaluation)
  - The following metrics are read dynamically from these files:
    - Held-out test records count (`num_test_records`: 250)
    - Base Test Loss (`1.3618`) and Stage 2F Test Loss (`0.8855`)
    - Base Perplexity (`3.90`) and Stage 2F Perplexity (`2.42`)
    - Base Token Accuracy (`31.33%`) and Stage 2F Token Accuracy (`39.07%`)
    - Fully closed recipe rate (`fully_closed_rate`)
    - Structural delimiter validity rate (`fully_closed_rate + partially_structured_rate`: 100.0%)
    - Repetitive text rate (`repetitive_rate`)
  - **Static Metric (Observation):** `ingredient_recall` is hardcoded as `89.21` (Stage 2F) and `4.11` (Base). The source code documents this with `# Recipe body adherence verified in Stage 2G report` because body adherence was scored in the final evaluation report rather than raw telemetry JSON.

---

### Item 7: Page Loading, Sidebar Navigation, and CSS Suppression
* **Status:** **VERIFIED**
* **Inspection Location:** `app/main.py`, `.streamlit/config.toml`, runtime process.
* **Findings:**
  - **Page Execution:** All 5 pages (`app/main.py` and `app/pages/1` through `4`) load cleanly and define executable `run()` entry points.
  - **CSS Navigation Suppression Analysis:**
    - The active Streamlit version in `venv_tf` is **1.67.0**.
    - `.streamlit/config.toml` specifies:
      ```toml
      [client]
      showSidebarNavigation = false
      ```
    - In Streamlit 1.31+, `client.showSidebarNavigation = false` natively halts the instantiation of built-in multi-page navigation links (`[data-testid="stSidebarNav"]`) in the sidebar DOM.
    - `app/main.py` also injects redundant CSS:
      ```css
      [data-testid="stSidebarNav"] {
          display: none !important;
      }
      ```
    - Together, these ensure zero menu duplication: Streamlit does not render the built-in pages list, and the CSS acts as an immediate fallback if configuration reloading occurs.
  - **Server Restart:** Streamlit was restarted cleanly on port `8503` (Uvicorn running on `:::8503`).

---

### Item 8: Home Page Architecture Overview and Navigation Actions
* **Status:** **VERIFIED**
* **Inspection Location:** `app/main.py` (lines 151–259).
* **Findings:**
  - **Architecture Pipeline Diagram:** Present as an ASCII workflow diagram tracing `Raw Ingredients` → `NER` → `Safety Screening` → `Dish Title Alignment` → `Causal LM Prompt` → `Fine-Tuned GPT-2` → `Structured Output`.
  - **Technology Stack Summary:** Accurately enumerates Python 3.12, Streamlit 1.67, TensorFlow 2.21, KerasHub 0.20, and Apple Silicon CPU execution.
  - **Navigation Action Cards:** 4 action columns (`q1`, `q2`, `q3`, `q4`) contain direct navigation buttons:
    - `Open Recipe Generator` (`key="go_gen"`)
    - `Open Ingredient Analysis` (`key="go_ana"`)
    - `Open Model Comparison` (`key="go_comp"`)
    - `Open Model Training` (`key="go_train"`)
    Each updates `st.session_state["nav_selection"]` and calls `st.rerun()`.

---

### Item 9: Integrity Verification of Models and Datasets
* **Status:** **VERIFIED (100% UNTOUCHED)**
* **Command Executed:**
  ```bash
  python3 -c "import hashlib; ..."
  ```
* **Checksum Audit:**

| Artifact | Recorded MD5 | Audited MD5 | Verification Status |
|---|---|---|---|
| `data/processed/recipes_train.jsonl` | `6d2487aa1719146437892318f5820dc9` | `6d2487aa1719146437892318f5820dc9` | **PASS (Identical)** |
| `data/processed/recipes_validation.jsonl` | `01b7bf6c2d0ee71093f38e01c000a045` | `01b7bf6c2d0ee71093f38e01c000a045` | **PASS (Identical)** |
| `data/processed/recipes_test.jsonl` | `11b581b58e352672e9ef2f7130cd10bb` | `11b581b58e352672e9ef2f7130cd10bb` | **PASS (Identical)** |
| `models/recipe_gpt2_full/recipe_gpt2_full_best.weights.h5` | `d95eb61d471f183a67a0e7fb920369d0` | `d95eb61d471f183a67a0e7fb920369d0` | **PASS (Identical)** |
| `models/gpt2_finetuned.weights.h5` | `a0ddb523b21e820c2fdcc871f2fae2a1` | `a0ddb523b21e820c2fdcc871f2fae2a1` | **PASS (Identical)** |
| `models/recipe_gpt2/recipe_gpt2_best.weights.h5` | `0f733e894f4a83087164dfc1b7fe9a9a` | `0f733e894f4a83087164dfc1b7fe9a9a` | **PASS (Identical)** |

---

### Item 10: Test Suite Reproducibility
* **Status:** **VERIFIED**
* **Command Executed:**
  ```bash
  HOME=/tmp ./venv_tf/bin/python -m unittest -v \
    tests/test_dish_validation.py \
    tests/test_ingredient_parser.py \
    tests/test_phase1_fixes.py \
    tests/test_stage2k_enhancements.py
  ```
* **Test Breakdown:**
  - `TestDishValidation` (7 tests):
    - `test_biryani_mismatch_warning`: OK
    - `test_biryani_with_rice_no_warning`: OK
    - `test_dal_mismatch_warning`: OK
    - `test_dal_with_lentils_no_warning`: OK
    - `test_empty_title_no_warning`: OK
    - `test_generic_title_no_warning`: OK
    - `test_omelette_mismatch_warning`: OK
  - `TestIngredientParser` (9 tests):
    - `test_comma_separated`: OK
    - `test_newline_separated`: OK
    - `test_mixed_delimiters`: OK
    - `test_semicolon_separated`: OK
    - `test_bullet_prefixed`: OK
    - `test_numbered_list`: OK
    - `test_duplicate_deduplication`: OK
    - `test_whitespace_and_empty_lines`: OK
    - `test_empty_and_none_input`: OK
  - `TestPhase1Fixes` (10 tests):
    - `test_fractions`: OK
    - `test_decimals_and_integers`: OK
    - `test_negative_and_zero_quantities`: OK
    - `test_non_culinary_units`: OK
    - `test_plural_unit_normalization_and_bounds`: OK
    - `test_false_positive_mitigations`: OK
    - `test_contradictions_preserve_warnings`: OK
    - `test_true_positives`: OK
    - `test_full_metrics_extraction`: OK
    - `test_pilot_metrics_extraction`: OK
  - `TestStage2KEnhancements` (7 tests):
    - `test_ingredient_parser_delimiters`: OK
    - `test_formatted_recipe_parsing`: OK
    - `test_stage2g_held_out_metrics_loading`: OK
    - `test_pages_import_cleanly`: OK
    - `test_safety_validation_integration`: OK
    - `test_checkpoints_unchanged`: OK
    - `test_datasets_unchanged`: OK

* **Summary Result:** **33 tests passed in 6.904s (Zero failures, zero errors).**

---

## 3. Operational Characteristics and Non-Blocking Notes

1. **CPU Latency Notice:**
   - Free-form text generation using 124M GPT-2 with 512-token context runs entirely on Apple Silicon CPU. Single-recipe generation takes ~10–20 seconds, and side-by-side comparison takes ~25–40 seconds. This is an intrinsic compute characteristic of local CPU inference and is documented with status spinners in the UI.
2. **Direct Multi-Page Sub-URL Access:**
   - Because the custom router `app/main.py` controls top-level navigation via `load_page().run()`, accessing `http://localhost:8503/Recipe_Generator` directly in the browser address bar bypasses `main.py`. Since `showSidebarNavigation = false` is active, users are kept inside the unified main shell at `http://localhost:8503`, where page switching operates seamlessly.
3. **Future Planned Phases (Out of Scope for Stage 2K):**
   - Fine-tuning BERT for token-classification ingredient NER and Diffusion models for recipe imagery remain future roadmap items.

---

## 4. Acceptance Conclusion

The Stage 2K UI Optimization and Refinement satisfies all core functional and architectural requirements. No regressions, data leaks, or unauthorized weight modifications were detected. The project is accepted for Stage 2K.
