# -*- coding: utf-8 -*-
"""Streamlit Cloud Entry Point for RecipeGen AI.

Redirects directly to app/main.py so that both default 'streamlit_app.py'
and custom 'app/main.py' paths work seamlessly in Streamlit Community Cloud.
"""

from pathlib import Path
import runpy
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Launch app/main.py
MAIN_APP_PATH = PROJECT_ROOT / "app" / "main.py"
runpy.run_path(str(MAIN_APP_PATH), run_name="__main__")
