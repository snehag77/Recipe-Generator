# -*- coding: utf-8 -*-
"""RecipeGen AI Application Entry Point and Dynamic Router.

Pages load on-demand with strict fault isolation; a broken page never crashes the application.
"""

import importlib.util
import os
from pathlib import Path
import sys
import traceback
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure TensorFlow backend is set before any Keras modules are imported
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PAGES_DIR = APP_DIR / "pages"

NAV_OPTIONS = [
    "🏠 Home",
    "🍳 Recipe Generator",
    "🔎 Ingredient Analysis",
    "⚖️ Model Comparison",
    "📈 Model Training",
]

PAGE_MAP = {
    "🏠 Home": None,
    "🍳 Recipe Generator": "1_Recipe_Generator.py",
    "🔎 Ingredient Analysis": "2_Ingredient_Analysis.py",
    "⚖️ Model Comparison": "3_Model_Comparison.py",
    "📈 Model Training": "4_Model_Training.py",
}

# Resolve active navigation choice before first Streamlit call for browser tab title
current_nav_label = st.session_state.get("nav_choice", "🏠 Home")
raw_title = current_nav_label.split(" ", 1)[-1] if " " in current_nav_label else current_nav_label

st.set_page_config(
    page_title=f"RecipeGen AI — {raw_title}",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling to remove duplicate built-in navigation and unify aesthetics
st.markdown(
    """
    <style>
    /* 1. Eliminate duplicate built-in Streamlit multi-page navigation */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }

    /* 2. Brand header and sidebar spacing */
    .brand-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1B5E20;
        margin-bottom: 0.2rem;
    }
    .brand-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        background-color: #F3E5AB;
        color: #1B5E20;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 1.2rem;
        border: 1px solid #E0D39B;
    }

    /* 3. Subtle card container style */
    .recipe-card {
        background-color: #FFFDF7;
        border: 1px solid #E8DFBF;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }

    /* 4. Sidebar footer details */
    .sidebar-footer {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #E8DFBF;
        font-size: 0.78rem;
        color: #4E6E4E;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_page(filename: str):
    """Dynamically load and return a page module."""
    path = PAGES_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Page file not found: {path}")
    spec = importlib.util.spec_from_file_location("page_" + path.stem, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "run"):
        raise AttributeError(f"{filename} must define a run() function")
    return mod


# Sidebar Brand Header
st.sidebar.markdown('<div class="brand-header">🍳 RecipeGen AI</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="brand-badge">Stage 2F Full • 512 Tokens</div>', unsafe_allow_html=True)

# Navigation Radio (Single source of navigation)
default_index = 0
if current_nav_label in NAV_OPTIONS:
    default_index = NAV_OPTIONS.index(current_nav_label)

selected_nav = st.sidebar.radio(
    "Navigation Menu",
    NAV_OPTIONS,
    index=default_index,
    key="nav_selection",
    label_visibility="collapsed",
)
st.session_state["nav_choice"] = selected_nav

# Sidebar Status & Environment Footer
st.sidebar.markdown(
    """
    <div class="sidebar-footer">
        <strong>Engine:</strong> TensorFlow 2.21 + KerasHub<br>
        <strong>Architecture:</strong> GPT-2 (124M Base)<br>
        <strong>Checkpoint:</strong> Stage 2F Full (4.5k Recipes)<br>
        <strong>Device:</strong> Apple Silicon CPU<br>
        <strong>Status:</strong> All Systems Operational
    </div>
    """,
    unsafe_allow_html=True,
)

# Page Router
page_file = PAGE_MAP.get(selected_nav)

if page_file is None:
    # -------------------------------------------------------------
    # HOME PAGE
    # -------------------------------------------------------------
    st.title("🍳 RecipeGen AI")
    st.subheader("Intelligent Ingredient-Conditioned Recipe Generation & Safety Platform")

    st.markdown(
        "RecipeGen AI is an end-to-end culinary intelligence platform that transforms available pantry "
        "ingredients into coherent, structured recipes while actively screening for allergens, "
        "validating measurement quantities, and verifying dish-title culinary alignment."
    )

    st.markdown("---")
    st.subheader("🏗️ Architecture & Processing Pipeline")

    st.markdown(
        """
```text
[Raw Ingredients] ──> [Ingredient Extraction (NER)]
                                 │
                                 ▼
                     [Allergen & Bounds Screening]
                                 │
                                 ▼
                    [Dish Title Alignment Check]
                                 │
                                 ▼
                       [GPT-2 Causal LM Prompt]
                                 │
                                 ▼
                  [Fine-Tuned Recipe Generation]
                                 │
                                 ▼
                     [Structured Recipe Output]
                (Title, Quantities & Cooking Steps)
```
        """
    )

    c_step1, c_step2, c_step3 = st.columns(3)
    with c_step1:
        st.markdown(
            "**1. Input & Safety Preflight**\n"
            "- Parses comma- or newline-separated items.\n"
            "- Screens against 10 common allergen categories.\n"
            "- Validates realistic culinary units and bounds.\n"
            "- Advisory dish title alignment validation."
        )
    with c_step2:
        st.markdown(
            "**2. Fine-Tuned GPT-2 Generation**\n"
            "- Conditioned on structured `<INGREDIENTS>` tags.\n"
            "- Full Stage 2F checkpoint trained on 4,500 recipes.\n"
            "- 512-token context sequence window on CPU."
        )
    with c_step3:
        st.markdown(
            "**3. Output Structuring**\n"
            "- Dual view: Formatted Markdown & Raw Token inspection.\n"
            "- Clean separation of Recipe Title, Quantities, and Directions.\n"
            "- Full transparency into underlying prompt delimiters."
        )

    st.markdown("---")
    st.subheader("💻 Implemented Technology Stack")
    st.markdown(
        "- **Runtime & Web UI:** Python 3.12, Streamlit 1.67\n"
        "- **Deep Learning Engine:** TensorFlow 2.21 & Keras 3.15\n"
        "- **Causal LM Architecture:** KerasHub 0.33 (`gpt2_base_en` Causal LM, 124M parameters)\n"
        "- **Fine-Tuning Execution:** 4,500 RecipeNLG recipes on Apple Silicon CPU (164.3 min, AdamW, 5% warmup, seq len 512)\n"
        "- **Safety & Preflight:** Rule-based keyword/compound allergen matching with declared claims & unit normalization"
    )

    st.markdown("---")
    st.subheader("🚀 Quick Navigation")
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.markdown("#### 🍳 Generator")
        st.caption("Generate recipes from ingredients with allergen and quantity checks.")
        if st.button("Open Recipe Generator", key="go_gen", use_container_width=True):
            st.session_state["nav_selection"] = "🍳 Recipe Generator"
            st.session_state["nav_choice"] = "🍳 Recipe Generator"
            st.rerun()

    with q2:
        st.markdown("#### 🔎 Analysis")
        st.caption("Inspect token extraction, unit checks, and allergen screening.")
        if st.button("Open Ingredient Analysis", key="go_ana", use_container_width=True):
            st.session_state["nav_selection"] = "🔎 Ingredient Analysis"
            st.session_state["nav_choice"] = "🔎 Ingredient Analysis"
            st.rerun()

    with q3:
        st.markdown("#### ⚖️ Comparison")
        st.caption("Compare Base GPT-2 vs Fine-Tuned Recipe GPT-2 side-by-side.")
        if st.button("Open Model Comparison", key="go_comp", use_container_width=True):
            st.session_state["nav_selection"] = "⚖️ Model Comparison"
            st.session_state["nav_choice"] = "⚖️ Model Comparison"
            st.rerun()

    with q4:
        st.markdown("#### 📈 Training")
        st.caption("View training loss, perplexity, and 250 held-out test evaluation benchmarks.")
        if st.button("Open Model Training", key="go_train", use_container_width=True):
            st.session_state["nav_selection"] = "📈 Model Training"
            st.session_state["nav_choice"] = "📈 Model Training"
            st.rerun()

else:
    try:
        load_page(page_file).run()
    except Exception as e:
        st.error(f"Could not show '{selected_nav}': {e}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())
