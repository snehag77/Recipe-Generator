# -*- coding: utf-8 -*-
"""Ingredient Analysis page.

Allows inspecting extracted ingredients, tokenization, allergen risks,
and quantity validation rules with neural BERT token-classification,
interactive ingredient management, and cross-page session state synchronization.
"""

import streamlit as st
from utils import allergen, ner, validation


def run():
    st.header("🔎 Ingredient Analysis & Named Entity Recognition")
    st.write(
        "Analyze raw culinary text using **BERT-based Token Classification (NER)** or a deterministic "
        "rule-based parser to extract clean ingredient entities, screen for allergens, and validate "
        "measurements before recipe generation."
    )

    # Extraction Engine Selection
    bert_available = ner.is_bert_ner_available()
    engine_col1, engine_col2 = st.columns([3, 2])

    with engine_col1:
        engine_option = st.radio(
            "Extraction Engine",
            [
                "🤖 Neural BERT Token Classifier (Fine-Tuned)",
                "⚡ Rule-Based Pattern Matcher (Regex)",
            ],
            index=0 if bert_available else 1,
            horizontal=True,
            help="Select BERT for natural language token-level entity spans, or Rule-Based for delimiter splitting.",
        )

    requested_method = "bert" if "BERT" in engine_option else "rule"

    with engine_col2:
        if bert_available:
            st.success("✅ **BERT Model Status:** Fine-Tuned Checkpoint Ready (`bert-base-cased` on Apple M4 MPS)")
        else:
            st.info("ℹ️ **BERT Model Status:** Initializing/Training (Rule-based fallback active)")

    # Pre-populate with session state if available
    default_text = st.session_state.get(
        "ingredients", "potato, onion, tomato, cumin, oil, salt"
    )
    user_input = st.text_area(
        "Enter ingredients or unstructured recipe text to analyze",
        value=default_text,
        placeholder="e.g.,\nAdd two chopped onions, three ripe tomatoes, fresh minced ginger, garlic cloves, and 1 tbsp olive oil.",
        help="Supports natural sentences ('Add 2 onions and garlic'), lists, commas, or newlines.",
        height=140,
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        analyze_clicked = st.button("🔎 Analyze Ingredients", type="primary", use_container_width=True)

    # Maintain parsed list in session state for interactive adjustments
    if analyze_clicked or "analysis_items" not in st.session_state:
        if user_input.strip():
            parsed_items, actual_method = ner.extract_ingredients_with_metadata(
                user_input, method=requested_method
            )
            st.session_state["analysis_items"] = parsed_items
            st.session_state["analysis_engine"] = actual_method
            st.session_state["ingredients"] = user_input.strip()
        else:
            st.session_state["analysis_items"] = []
            st.session_state["analysis_engine"] = "rule"

    current_items = st.session_state.get("analysis_items", [])
    used_engine = st.session_state.get("analysis_engine", "rule")

    if not current_items:
        if analyze_clicked:
            st.warning("⚠️ Please enter at least one valid ingredient to analyze.")
        return

    # Engine Badge
    if used_engine == "bert":
        st.success("🤖 **Active Extraction Engine:** Fine-Tuned BERT (`google-bert/bert-base-cased` Token Classifier)")
    else:
        st.info("⚡ **Active Extraction Engine:** Rule-Based Pattern Matcher")

    # Pre-calculate diagnostics
    detected_allergens = allergen.detect_allergens(current_items)
    qty_warnings = validation.validate_quantities(current_items)

    st.markdown("---")
    st.subheader("📊 Analysis Overview")
    m1, m2, m3 = st.columns(3)
    m1.metric("Extracted Ingredients", f"{len(current_items)} items")
    m2.metric("Detected Allergen Risks", f"{len(detected_allergens)} categories")
    m3.metric("Quantity Alerts", f"{len(qty_warnings)} warnings")

    st.markdown("---")

    # 1. Interactive Extracted Ingredient Management
    st.subheader("1. Extracted Ingredients & Interactive Management")
    st.caption("Remove or uncheck ingredients below to refine your list for recipe generation.")

    selected_items = st.multiselect(
        "Active Extracted Ingredients",
        options=current_items,
        default=current_items,
        help="Click the 'x' on any chip to remove that ingredient.",
    )

    # Option to synchronize updated list back to session state
    if selected_items != current_items:
        st.session_state["analysis_items"] = selected_items
        st.session_state["ingredients"] = ", ".join(selected_items)
        st.info("ℹ️ Active ingredient list updated.")

    col_sync1, col_sync2 = st.columns([2, 3])
    with col_sync1:
        if st.button("🔄 Sync Selection to Recipe Generator", use_container_width=True):
            st.session_state["ingredients"] = ", ".join(selected_items)
            st.success(f"✅ Synchronized {len(selected_items)} ingredients to Recipe Generator!")

    # Display clean list view
    if selected_items:
        st.markdown(f"**Current Items ({len(selected_items)}):** " + " • ".join(f"`{item}`" for item in selected_items))
    else:
        st.warning("⚠️ All ingredients removed. Add ingredients above to continue.")
        return

    # Recalculate based on active selection
    active_allergens = allergen.detect_allergens(selected_items)
    active_warnings = validation.validate_quantities(selected_items)

    # 2. Quantity & Measurement Check
    st.markdown("---")
    st.subheader("2. Quantity & Unit Validation")
    if active_warnings:
        for w in active_warnings:
            st.warning(f"⚠️ {w}")
    else:
        st.success("✅ All parsed quantities and measurement units appear plausible.")

    # 3. Allergen Analysis
    st.markdown("---")
    st.subheader("3. Allergen Screening")
    if active_allergens:
        st.warning(
            f"🚨 **Potential Allergens Detected:** **{', '.join(active_allergens).title()}**"
        )
    else:
        st.success("✅ No common allergens identified from standard watchlists.")

    st.caption(
        "⚠️ Disclaimer: Allergen screening is rule-based and informational only. "
        "It recognizes declared product claims (e.g., 'gluten-free flour') but does not "
        "guarantee absence of cross-contamination or substitute allergens."
    )
