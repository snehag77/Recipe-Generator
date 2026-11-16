# -*- coding: utf-8 -*-
"""Recipe Generator page.

Connects Stage 2F Full Fine-Tuned GPT-2 generation, allergen warnings,
quantity validation, dish-title alignment checking, and rollback support.
"""

import os
from pathlib import Path
import streamlit as st
from utils import allergen, dish_validation, gpt2, ner, prompt, validation


def run():
    st.header("🍳 Recipe Generator")
    st.write(
        "Enter available ingredients to generate a structured recipe. The system verifies "
        "measurements, screens for allergens, and conditions **Fine-Tuned GPT-2** to generate "
        "a customized recipe with quantities and instructions."
    )

    full_exists = os.path.isfile(gpt2.FULL_WEIGHTS_PATH)
    pilot_exists = os.path.isfile(gpt2.PILOT_WEIGHTS_PATH)

    if full_exists:
        rel_path = os.path.relpath(gpt2.FULL_WEIGHTS_PATH, gpt2.ROOT_DIR)
        st.success(f"✅ **Active Engine:** Stage 2F Fine-Tuned Model (`{rel_path}`, Seq Len: 512).")
    elif pilot_exists:
        rel_path = os.path.relpath(gpt2.PILOT_WEIGHTS_PATH, gpt2.ROOT_DIR)
        st.info(f"ℹ️ **Active Engine:** Pilot Model (Rollback) (`{rel_path}`).")
    else:
        st.warning("ℹ️ Model checkpoint not found. Operating in Base GPT-2 baseline mode.")

    col_input, col_ctrl = st.columns([3, 2], gap="large")

    with col_input:
        st.subheader("1. Ingredients & Style")
        ingredient_input = st.text_area(
            "Available Ingredients",
            value=st.session_state.get(
                "ingredients", "potato, onion, tomato, cumin, oil, salt"
            ),
            placeholder=(
                "e.g.,\n"
                "2 potatoes\n"
                "1 onion\n"
                "2 cups flour\n"
                "1/2 tsp salt\n"
                "3 eggs\n"
                "(Enter comma-separated or newline-separated items with optional quantities)"
            ),
            height=160,
            help="Enter items you have available. You can include standard culinary quantities (e.g. '2 cups flour', '1/2 tsp salt').",
        )

        title_hint = st.text_input(
            "Dish Style / Title Hint (Optional)",
            value=st.session_state.get("title_hint", ""),
            placeholder="e.g., Curry, Dal, Soup, Stir Fry, Stew (leave blank for open-ended generation)",
            help="Optional culinary style guide. If left empty, the model autonomously invents an appropriate dish title.",
        )

        # Parse ingredients early for real-time dish title alignment check
        parsed_current_ings = ner.extract_ingredients(ingredient_input) if ingredient_input else []
        dish_alignment_warn = dish_validation.validate_dish_title_alignment(title_hint, parsed_current_ings)
        if dish_alignment_warn:
            st.info(dish_alignment_warn)

    with col_ctrl:
        st.subheader("2. Model & Generation Settings")

        model_options = []
        if full_exists:
            model_options.append("Fine-Tuned GPT-2 (Stage 2F Full)")
        if pilot_exists:
            model_options.append("Pilot GPT-2 (Rollback)")
        model_options.append("Base GPT-2 (Pretrained Baseline)")

        model_choice = st.radio(
            "Model Checkpoint Selection",
            model_options,
            index=0 if model_options else 0,
            help="Choose between the full 4,500-recipe fine-tuned model, the 200-recipe pilot rollback model, or the raw pretrained baseline.",
        )

        temperature = st.slider(
            "Sampling Temperature",
            min_value=0.3,
            max_value=1.2,
            value=0.7,
            step=0.05,
            help="Controls randomness: Lower values (0.3 - 0.6) produce more predictable, focused recipes; higher values (0.7 - 1.1) produce more creative, varied recipes.",
        )

        st.caption(
            "💡 Tip: Temperature `0.70` provides an optimal balance between structural tag discipline and culinary creativity."
        )

        generate_button = st.button(
            "🍳 Generate Recipe",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.get("is_generating", False),
        )

    # Generation Workflow
    if generate_button:
        if not ingredient_input or not ingredient_input.strip():
            st.error("⚠️ Please provide at least one valid ingredient to generate a recipe.")
            return

        # Persist session state
        st.session_state["ingredients"] = ingredient_input.strip()
        st.session_state["title_hint"] = title_hint.strip()

        # 1. Parse ingredients
        ing_list = ner.extract_ingredients(ingredient_input)
        if not ing_list:
            st.error("Could not parse any ingredients from input.")
            return

        # 2. Allergen screening (U5a)
        allergens = allergen.detect_allergens(ing_list)

        # 3. Quantity bounds validation (U5b)
        qty_warnings = validation.validate_quantities(ing_list)

        # 4. Dish title culinary alignment check
        dish_warn = dish_validation.validate_dish_title_alignment(title_hint, ing_list)

        # 5. Build structured prompt (U1)
        user_prompt = prompt.build_prompt(ing_list, title_hint=title_hint)

        # 6. Resolve checkpoint
        if model_choice.startswith("Fine-Tuned"):
            use_ft = True
            ckpt_path = gpt2.FULL_WEIGHTS_PATH
        elif model_choice.startswith("Pilot"):
            use_ft = True
            ckpt_path = gpt2.PILOT_WEIGHTS_PATH
        else:
            use_ft = False
            ckpt_path = None

        # Lock button during execution
        st.session_state["is_generating"] = True
        try:
            with st.spinner(f"⏳ Generating recipe with {model_choice} (512 token budget)..."):
                recipe_output = gpt2.generate_recipe(
                    user_prompt,
                    use_finetuned=use_ft,
                    temperature=temperature,
                    max_length=512,
                    checkpoint_path=ckpt_path,
                )
        finally:
            st.session_state["is_generating"] = False

        st.markdown("---")
        st.subheader("📋 Generated Recipe")

        parsed_recipe = prompt.parse_recipe_output(recipe_output)

        tab_formatted, tab_raw = st.tabs(["Formatted Recipe", "Raw Generated Text"])

        with tab_formatted:
            has_title = bool(parsed_recipe.get("title"))
            has_quantities = bool(parsed_recipe.get("quantities"))
            has_steps = bool(parsed_recipe.get("steps"))

            if not (has_title or has_quantities or has_steps):
                st.warning(
                    "⚠️ Notice: Structured recipe tags were not detected or the output is unstructured. "
                    "Please view the 'Raw Generated Text' tab."
                )
                st.code(recipe_output, language="markdown")
            else:
                if has_title:
                    st.markdown(f"### 🍽️ {parsed_recipe['title']}")
                else:
                    st.caption("*(Title tag missing in generation)*")

                if has_quantities:
                    st.markdown("#### 🛒 Ingredients & Quantities")
                    for q in parsed_recipe["quantities"]:
                        st.markdown(f"- {q}")

                if has_steps:
                    st.markdown("#### 👩‍🍳 Cooking Instructions")
                    for idx, s in enumerate(parsed_recipe["steps"], 1):
                        step_text = s if s.lstrip().startswith(tuple(f"{i}." for i in range(1, 100))) else f"{idx}. {s}"
                        st.markdown(step_text)

                if not (has_quantities and has_steps):
                    st.info("ℹ️ Note: Some sections were incomplete. View 'Raw Generated Text' for full model tokens.")

        with tab_raw:
            st.code(recipe_output, language="markdown")

        # Safety & Quality Section
        st.markdown("---")
        st.subheader("🛡️ Safety & Quality Preflight Checks")

        # Dish title alignment alert
        if dish_warn:
            st.info(dish_warn)

        # Allergen alerts
        if allergens:
            st.warning(
                f"⚠️ **Allergen Alert:** Input contains items associated with: **{', '.join(allergens).title()}**."
            )
        else:
            st.success("✅ **Allergens:** No common allergens detected in input list.")

        st.caption(
            "Disclaimer: Allergen checking is advisory only and recognizes declared product claims. "
            "Verify with packaging for clinical food allergy safety."
        )

        # Quantity alerts
        if qty_warnings:
            for w in qty_warnings:
                st.warning(f"⚠️ {w}")
        else:
            st.info("✅ **Quantities:** Quantities and measurement units appear plausible.")

        with st.expander("🔍 View Structured Prompt Sent to Model"):
            st.code(user_prompt)
