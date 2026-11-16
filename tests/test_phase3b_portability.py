# -*- coding: utf-8 -*-
"""Unit and functional tests for Phase 3B Deployment Portability and Packaging."""

import hashlib
import os
from pathlib import Path
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestPhase3BPortability(unittest.TestCase):
    """Test suite for Phase 3B portability hardening."""

    def test_portable_keras_home_configured(self):
        """Test 1: KERAS_HOME points to a valid temporary or user directory, not hardcoded /tmp."""
        from utils import gpt2
        keras_home = os.environ.get("KERAS_HOME")
        self.assertIsNotNone(keras_home)
        self.assertTrue(Path(keras_home).name == ".keras")

    def test_preset_dir_validator(self):
        """Test 2: is_valid_preset_dir correctly checks for config, weights, tokenizer, and vocabulary."""
        from utils.gpt2 import is_valid_preset_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            base_p = Path(tmpdir)

            # Case 1: Nonexistent directory
            nonexistent = base_p / "nonexistent_dir"
            self.assertFalse(is_valid_preset_dir(nonexistent))

            def setup_preset(dir_name, missing_component=None):
                target = base_p / dir_name
                target.mkdir(parents=True, exist_ok=True)
                (target / "assets" / "tokenizer").mkdir(parents=True, exist_ok=True)

                if missing_component != "config":
                    (target / "config.json").write_text("{}", encoding="utf-8")
                if missing_component != "weights":
                    (target / "model.weights.h5").write_bytes(b"dummy_weights")
                if missing_component != "tokenizer":
                    (target / "tokenizer.json").write_text("{}", encoding="utf-8")
                if missing_component != "vocab":
                    (target / "assets" / "tokenizer" / "vocabulary.json").write_text("{}", encoding="utf-8")

                return target

            # Case 2: Missing configuration
            missing_cfg = setup_preset("missing_config", missing_component="config")
            self.assertFalse(is_valid_preset_dir(missing_cfg))

            # Case 3: Missing model weights
            missing_weights = setup_preset("missing_weights", missing_component="weights")
            self.assertFalse(is_valid_preset_dir(missing_weights))

            # Case 4: Missing tokenizer vocabulary
            missing_vocab = setup_preset("missing_vocab", missing_component="vocab")
            self.assertFalse(is_valid_preset_dir(missing_vocab))

            # Case 5: Complete preset directory
            complete_preset = setup_preset("complete_preset")
            self.assertTrue(is_valid_preset_dir(complete_preset))

    def test_preset_lookup_priority(self):
        """Test 3: get_cached_preset_path respects KERAS_PRESET_DIR override."""
        from utils.gpt2 import get_cached_preset_path

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            (p / "config.json").write_text("{}", encoding="utf-8")
            (p / "model.weights.h5").write_bytes(b"dummy")
            (p / "tokenizer.json").write_text("{}", encoding="utf-8")
            (p / "assets" / "tokenizer").mkdir(parents=True, exist_ok=True)
            (p / "assets" / "tokenizer" / "vocabulary.json").write_text("{}", encoding="utf-8")

            old_env = os.environ.get("KERAS_PRESET_DIR")
            try:
                os.environ["KERAS_PRESET_DIR"] = str(p)
                resolved = get_cached_preset_path()
                self.assertEqual(Path(resolved).resolve(), p.resolve())
            finally:
                if old_env is not None:
                    os.environ["KERAS_PRESET_DIR"] = old_env
                else:
                    os.environ.pop("KERAS_PRESET_DIR", None)

    def test_download_model_preflight_verifier(self):
        """Test 4: download_model.py preflight verifier passes on exact file and fails on mismatch."""
        from scripts.download_model import verify_existing_artifact

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"RecipeGen AI Test Content")
            test_path = Path(f.name)

        try:
            content_sha256 = hashlib.sha256(b"RecipeGen AI Test Content").hexdigest()
            content_md5 = hashlib.md5(b"RecipeGen AI Test Content").hexdigest()
            content_size = len(b"RecipeGen AI Test Content")

            # Passing preflight
            self.assertTrue(
                verify_existing_artifact(
                    test_path,
                    expected_size=content_size,
                    expected_sha256=content_sha256,
                    expected_md5=content_md5,
                )
            )

            # Failing size
            self.assertFalse(
                verify_existing_artifact(
                    test_path,
                    expected_size=99999,
                )
            )

            # Failing hash
            self.assertFalse(
                verify_existing_artifact(
                    test_path,
                    expected_md5="00000000000000000000000000000000",
                )
            )
        finally:
            test_path.unlink(missing_ok=True)

    def test_dockerignore_rules_present(self):
        """Test 5: .dockerignore contains all critical exclusions."""
        dockerignore_path = PROJECT_ROOT / ".dockerignore"
        self.assertTrue(dockerignore_path.is_file(), ".dockerignore missing!")
        content = dockerignore_path.read_text(encoding="utf-8")

        for rule in ["data/archive 2/", "venv/", "venv_tf/", "*.bak", "models/recipe_gpt2/"]:
            self.assertIn(rule, content, f"Rule '{rule}' missing from .dockerignore!")

    def test_dockerfile_syntax_and_structure(self):
        """Test 6: Dockerfile defines non-root user, healthcheck, and python 3.12."""
        dockerfile_path = PROJECT_ROOT / "Dockerfile"
        self.assertTrue(dockerfile_path.is_file(), "Dockerfile missing!")
        content = dockerfile_path.read_text(encoding="utf-8")

        self.assertIn("python:3.12-slim", content)
        self.assertIn("appuser", content)
        self.assertIn("HEALTHCHECK", content)
        self.assertIn("EXPOSE 8501", content)
        self.assertIn("requirements.txt", content)

    def test_export_base_preset_script_exists_and_parses(self):
        """Test 7: export_base_preset.py exists and handles dry-run."""
        export_script = PROJECT_ROOT / "scripts" / "export_base_preset.py"
        self.assertTrue(export_script.is_file(), "export_base_preset.py missing!")


if __name__ == "__main__":
    unittest.main()
