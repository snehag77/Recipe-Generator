# -*- coding: utf-8 -*-
"""Model Comparison page.

Compares Base GPT-2 vs Fine-Tuned Recipe GPT-2 across the 5 reference benchmark sets
and custom user inputs with integrated allergen screening, quantity validation,
and CPU progress telemetry.
"""

import streamlit as st
from utils import allergen, gpt2, ner, prompt, validation

BENCHMARK_SETS = [
    ("Set 1: Chicken & Rice", ["chicken", "rice", "garlic", "soy sauce"]),
    ("Set 2: Paneer & Tomato", ["paneer", "tomato", "onion", "cream", "garam masala"]),
    ("Set 3: Baking Essentials", ["flour", "butter", "sugar", "eggs", "vanilla"]),
    ("Set 4: Veggie & Legume", ["potato", "spinach", "chickpeas", "cumin"]),
    ("Set 5: Healthy Breakfast", ["banana", "peanut butter", "oats", "honey"]),
]


def run():
    st.header("⚖️ Model Comparison: Base vs Fine-Tuned GPT-2")
    st.write(
        "Evaluate the qualitative and structural differences between the pretrained **Base GPT-2** "
        "and our **Fine-Tuned Recipe GPT-2** across reference benchmarks or custom ingredients."
    )

    st.caption("ℹ️ CPU Execution Note: Two-model sequential inference takes ~8–14 seconds on Apple Silicon CPU.")

    tab1, tab2 = st.tabs(["5 Benchmark Sets Comparison", "Live Side-by-Side Playground"])

    with tab1:
        st.subheader("1. Benchmark Recipe Generation (5 Reference Sets)")
        selected_set_name = st.selectbox(
            "Choose a benchmark ingredient set",
            [name for name, _ in BENCHMARK_SETS],
            help="Select one of the 5 standardized benchmark sets used throughout project evaluation.",
        )

        selected_ings = next(ings for name, ings in BENCHMARK_SETS if name == selected_set_name)
        st.markdown(f"**Conditioning Ingredients:** `{', '.join(selected_ings)}`")

        if st.button("Generate Benchmark Comparison", type="primary", key="btn_bench", use_container_width=True):
            test_prompt = prompt.build_prompt(selected_ings)

            col1, col2 = st.columns(2, gap="medium")
            with col1:
                st.markdown("### ⚪ Base GPT-2")
                st.caption("Pretrained Baseline (KerasHub gpt2_base_en • 124M)")
                with st.spinner("⏳ Step 1/2: Generating with Base Model..."):
                    base_out = gpt2.generate_recipe(test_prompt, use_finetuned=False, temperature=0.7)
                st.code(base_out, language="markdown")

            with col2:
                st.markdown("### 🟢 Fine-Tuned GPT-2")
                st.caption("Stage 2F Full Fine-Tuned Checkpoint (4,500 Recipes)")
                with st.spinner("⏳ Step 2/2: Generating with Fine-Tuned Model..."):
                    ft_out = gpt2.generate_recipe(test_prompt, use_finetuned=True, temperature=0.7)
                st.code(ft_out, language="markdown")

    with tab2:
        st.subheader("2. Live Custom Ingredients Comparison")
        custom_input = st.text_area(
            "Enter custom ingredients for comparison",
            value="potato, cheese, rosemary, olive oil",
            placeholder="e.g.,\npotato\ncheese\nrosemary\nolive oil",
            help="Supports comma-separated or newline-separated ingredients with optional quantities.",
            height=100,
        )
        temp = st.slider("Comparison Temperature", 0.4, 1.2, 0.7, 0.05, key="comp_temp")

        if st.button("Run Live Side-by-Side Comparison", type="primary", key="btn_custom", use_container_width=True):
            ings = ner.extract_ingredients(custom_input)
            if not ings:
                st.error("⚠️ Please provide at least one valid ingredient.")
                return

            # Safety screening before generation
            allergens = allergen.detect_allergens(ings)
            qty_warnings = validation.validate_quantities(ings)

            if allergens or qty_warnings:
                st.markdown("#### 🛡️ Input Preflight Safety Observations")
                if allergens:
                    st.warning(f"⚠️ **Allergen Alert:** Input contains items associated with: **{', '.join(allergens).title()}**.")
                if qty_warnings:
                    for w in qty_warnings:
                        st.warning(f"⚠️ {w}")
                st.caption(
                    "Disclaimer: Allergen and quantity checks are advisory and rule-based. "
                    "They do not constitute medical or clinical dietary verification."
                )

            cust_prompt = prompt.build_prompt(ings)

            col_base, col_ft = st.columns(2, gap="medium")
            with col_base:
                st.markdown("### ⚪ Base GPT-2")
                st.caption("Pretrained Baseline (Vanilla weights)")
                with st.spinner("⏳ Step 1/2: Running Base Model..."):
                    b_res = gpt2.generate_recipe(cust_prompt, use_finetuned=False, temperature=temp)
                st.code(b_res, language="markdown")

            with col_ft:
                st.markdown("### 🟢 Fine-Tuned GPT-2")
                st.caption("Stage 2F Full Checkpoint (Recipe-conditioned)")
                with st.spinner("⏳ Step 2/2: Running Fine-Tuned Model..."):
                    f_res = gpt2.generate_recipe(cust_prompt, use_finetuned=True, temperature=temp)
                st.code(f_res, language="markdown")
