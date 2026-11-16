# -*- coding: utf-8 -*-
"""GPT-2 Inference & Model Loader for RecipeGen AI.

Loads either:
1. Stage 2F Full-Dataset Fine-Tuned Model: models/recipe_gpt2_full/recipe_gpt2_full_best.weights.h5 (Primary)
2. Stage 2D Pilot Fine-Tuned Model: models/gpt2_finetuned.weights.h5 (Rollback Option)
3. Base GPT-2: gpt2_base_en (Pretrained Baseline)
"""

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_HOME", str(Path(tempfile.gettempdir()) / ".keras"))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROOT_DIR = str(PROJECT_ROOT)

# Stage 2F primary checkpoint
FULL_WEIGHTS_PATH = str(
    PROJECT_ROOT / "models" / "recipe_gpt2_full" / "recipe_gpt2_full_best.weights.h5"
)

# Stage 2D pilot checkpoint (preserved as rollback option)
PILOT_WEIGHTS_PATH = str(PROJECT_ROOT / "models" / "gpt2_finetuned.weights.h5")

# Default active weights: primary is Stage 2F full checkpoint, fallback to pilot
WEIGHTS_PATH = FULL_WEIGHTS_PATH if os.path.isfile(FULL_WEIGHTS_PATH) else PILOT_WEIGHTS_PATH

_MODELS_CACHE = {}


def is_valid_preset_dir(preset_path: Path) -> bool:
    """Validate that a candidate preset directory contains all required KerasHub GPT-2 files."""
    if not preset_path.is_dir():
        return False
    required = [
        "config.json",
        "model.weights.h5",
        "tokenizer.json",
        "assets/tokenizer/vocabulary.json",
    ]
    return all((preset_path / f).is_file() for f in required)


def get_cached_preset_path() -> str:
    """Return local cached preset directory if present and valid, following strict priority order.

    Lookup Priority:
    1. Explicitly configured local project preset: models/gpt2_base_en/ or KERAS_PRESET_DIR
    2. Existing KaggleHub cache (~/.cache/kagglehub/.../gpt2_base_en/3)
    3. Other explicitly configured cache locations
    4. Remote preset fallback ('gpt2_base_en'), if no valid local copy exists
    """
    candidate_paths = []

    # Priority 1: Explicitly configured local project preset
    if "KERAS_PRESET_DIR" in os.environ:
        candidate_paths.append(Path(os.environ["KERAS_PRESET_DIR"]).resolve())
    if "GPT2_PRESET_DIR" in os.environ:
        candidate_paths.append(Path(os.environ["GPT2_PRESET_DIR"]).resolve())
    candidate_paths.append(PROJECT_ROOT / "models" / "gpt2_base_en")

    # Priority 2: Existing KaggleHub cache
    if "KAGGLE_HUB_CACHE" in os.environ:
        candidate_paths.append(
            Path(os.environ["KAGGLE_HUB_CACHE"]) / "models" / "keras" / "gpt2" / "keras" / "gpt2_base_en" / "3"
        )
    candidate_paths.append(
        Path.home() / ".cache" / "kagglehub" / "models" / "keras" / "gpt2" / "keras" / "gpt2_base_en" / "3"
    )
    # Check physical user home if Path.home() was overridden (e.g. HOME=/tmp)
    try:
        username = os.environ.get("USER") or os.environ.get("LOGNAME")
        if username and sys.platform != "win32":
            import pwd
            pw_home = Path(pwd.getpwnam(username).pw_dir)
            candidate_paths.append(
                pw_home / ".cache" / "kagglehub" / "models" / "keras" / "gpt2" / "keras" / "gpt2_base_en" / "3"
            )
    except Exception:
        pass

    # Priority 3: Other explicitly configured cache locations
    if "XDG_CACHE_HOME" in os.environ:
        candidate_paths.append(
            Path(os.environ["XDG_CACHE_HOME"]) / "kagglehub" / "models" / "keras" / "gpt2" / "keras" / "gpt2_base_en" / "3"
        )
    candidate_paths.append(
        Path.home() / ".cache" / "keras" / "gpt2_base_en"
    )

    for p in candidate_paths:
        if is_valid_preset_dir(p):
            return str(p)

    return "gpt2_base_en"


def load_model(use_finetuned: bool = True, checkpoint_path: str = None):
    """Load GPT2CausalLM with preprocessor configured for sequence_length=512."""
    import keras_hub

    # Determine target weights and cache key
    if not use_finetuned:
        model_key = "base"
        target_path = None
    else:
        target_path = checkpoint_path or WEIGHTS_PATH
        model_key = f"ft_{os.path.basename(target_path)}"

    if model_key in _MODELS_CACHE:
        return _MODELS_CACHE[model_key]

    preset = get_cached_preset_path()
    try:
        preprocessor = keras_hub.models.GPT2CausalLMPreprocessor.from_preset(
            preset,
            sequence_length=512,
        )
        model = keras_hub.models.GPT2CausalLM.from_preset(
            preset,
            preprocessor=preprocessor,
        )
    except Exception as err:
        if preset == "gpt2_base_en":
            raise RuntimeError(
                f"Failed to load base GPT-2 preset from '{preset}'. If running offline or in a restricted container, "
                f"please run 'python scripts/export_base_preset.py' to export the local preset to 'models/gpt2_base_en/' "
                f"or ensure internet connectivity. Root cause: {err}"
            ) from err
        raise

    if use_finetuned and target_path:
        if os.path.isfile(target_path):
            model.load_weights(target_path)
            print(f"✅ Loaded fine-tuned weights from: {target_path}")
        else:
            print(f"⚠️ Fine-tuned weights not found at {target_path}. Using base weights.")

    _MODELS_CACHE[model_key] = model
    return model


def generate_recipe(
    prompt: str,
    use_finetuned: bool = True,
    temperature: float = 0.7,
    max_length: int = 512,
    checkpoint_path: str = None,
) -> str:
    """Generate recipe text given a prompt using KerasHub Top-K sampling."""
    try:
        import keras_hub

        model = load_model(use_finetuned=use_finetuned, checkpoint_path=checkpoint_path)
        model.compile(sampler=keras_hub.samplers.TopKSampler(k=40, temperature=float(temperature)))
        output = model.generate(prompt, max_length=max_length)
        if isinstance(output, (list, tuple)):
            text = str(output[0])
        elif hasattr(output, "numpy"):
            text = str(output.numpy())
        else:
            text = str(output)
        return text.strip()
    except Exception as e:
        # Fallback to demo template if model fails to load
        return f"⚠️ Model generation fallback ({e}).\n\n{generate_demo_recipe(prompt)}"


def generate_demo_recipe(prompt: str, ingredients: list = None, title_hint: str = None) -> str:
    """Return a structured demo recipe matching the aligned U1 format."""
    ings_list = ingredients or ["Sample ingredient 1", "Sample ingredient 2"]
    title = (
        title_hint.strip().title()
        if title_hint and title_hint.strip()
        else "Homestyle " + (ings_list[0].title() if ings_list else "Medley")
    )

    quantities_formatted = "\n".join([f"- 1 portion {ing.strip()}" for ing in ings_list])

    return (
        f"<RECIPE>\n"
        f"<TITLE>\n{title}\n</TITLE>\n"
        f"<QUANTITIES>\n"
        f"{quantities_formatted}\n"
        f"- 2 tbsp cooking oil\n"
        f"- 1 tsp salt (to taste)\n"
        f"</QUANTITIES>\n"
        f"<STEPS>\n"
        f"1. Wash and prep ingredients ({', '.join(ings_list[:3])}).\n"
        f"2. Heat oil in a pan over medium heat.\n"
        f"3. Sauté key ingredients until tender.\n"
        f"4. Season with salt and simmer for 15-20 minutes.\n"
        f"5. Serve warm.\n"
        f"</STEPS>\n"
        f"</RECIPE>"
    )
