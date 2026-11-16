# -*- coding: utf-8 -*-
"""Optional Base GPT-2 Preset Export Utility for Offline Deployment.

Exports the cached KerasHub 'gpt2_base_en' preset (model architecture, weights,
and tokenizer vocabulary) from the user's local cache into 'models/gpt2_base_en/'.

This enables 100% offline self-contained deployments (e.g., Docker containers,
air-gapped clusters) without requiring internet connectivity or Kaggle credentials.

Strict Operational Guidelines:
1. User-invoked only: Never executes automatically during imports or app startup.
2. Preserves file contents verbatim (bit-for-bit exact copy).
3. Verifies every copied file against its source using SHA-256.
4. Refuses to overwrite existing preset files unless --force is explicitly provided.
5. Reports total size and disk space requirements (~475 - 498 MB).

Usage:
    python scripts/export_base_preset.py --dry-run
    python scripts/export_base_preset.py
    python scripts/export_base_preset.py --force
"""

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = PROJECT_ROOT / "models" / "gpt2_base_en"


def compute_sha256(filepath: Path) -> str:
    """Compute cryptographic SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def discover_source_preset(custom_source: str = None) -> Path:
    """Find the source preset directory from KaggleHub cache or custom source."""
    if custom_source:
        p = Path(custom_source).resolve()
        if p.is_dir() and (p / "config.json").is_file():
            return p
        raise FileNotFoundError(f"Custom preset source '{custom_source}' is not a valid preset directory.")

    # Search standard KaggleHub cache locations
    candidates = []
    if "KAGGLE_HUB_CACHE" in os.environ:
        candidates.append(
            Path(os.environ["KAGGLE_HUB_CACHE"]) / "models" / "keras" / "gpt2" / "keras" / "gpt2_base_en" / "3"
        )
    candidates.append(
        Path.home() / ".cache" / "kagglehub" / "models" / "keras" / "gpt2" / "keras" / "gpt2_base_en" / "3"
    )
    try:
        username = os.environ.get("USER") or os.environ.get("LOGNAME")
        if username and sys.platform != "win32":
            import pwd
            pw_home = Path(pwd.getpwnam(username).pw_dir)
            candidates.append(
                pw_home / ".cache" / "kagglehub" / "models" / "keras" / "gpt2" / "keras" / "gpt2_base_en" / "3"
            )
    except Exception:
        pass
    if "XDG_CACHE_HOME" in os.environ:
        candidates.append(
            Path(os.environ["XDG_CACHE_HOME"]) / "kagglehub" / "models" / "keras" / "gpt2" / "keras" / "gpt2_base_en" / "3"
        )

    for c in candidates:
        if c.is_dir() and (c / "config.json").is_file() and (c / "model.weights.h5").is_file():
            return c

    raise FileNotFoundError(
        "Could not locate cached 'gpt2_base_en' preset in standard cache directories. "
        "Run the application once online or pass --source-dir to specify location."
    )


def collect_preset_files(source_dir: Path) -> List[Tuple[Path, str, int]]:
    """Recursively collect all files in the preset directory with relative paths and byte sizes."""
    file_list = []
    for root, _, files in os.walk(source_dir):
        for fname in sorted(files):
            full_path = Path(root) / fname
            rel_path = full_path.relative_to(source_dir)
            size = full_path.stat().st_size
            file_list.append((full_path, str(rel_path), size))
    return file_list


def main():
    parser = argparse.ArgumentParser(
        description="Export cached base GPT-2 preset into models/gpt2_base_en for offline deployment."
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Custom path to cached source preset (defaults to auto-discovery in ~/.cache/kagglehub/...)",
    )
    parser.add_argument(
        "--dest-dir",
        default=str(DEFAULT_DEST),
        help=f"Target destination directory (default: {DEFAULT_DEST})",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite existing files in destination directory without prompt",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview source files, target paths, and disk requirements without copying",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("RecipeGen AI: Base GPT-2 Offline Preset Exporter")
    print("=" * 70)

    try:
        source_dir = discover_source_preset(args.source_dir)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

    dest_dir = Path(args.dest_dir).resolve()
    print(f"Source Preset:      {source_dir}")
    print(f"Target Destination: {dest_dir}")

    files_to_copy = collect_preset_files(source_dir)
    total_bytes = sum(sz for _, _, sz in files_to_copy)
    total_mb = total_bytes / (1024 * 1024)

    print(f"Files Found:        {len(files_to_copy)} files")
    print(f"Total Size:         {total_mb:.2f} MB ({total_bytes:,} bytes)")
    print("=" * 70)

    for src_file, rel_name, sz in files_to_copy:
        print(f"  • {rel_name:<30} ({sz / (1024 * 1024):>6.2f} MB)")

    if args.dry_run:
        print("\n🔍 Dry-run complete. No files were written to disk.")
        print(f"   Executing without --dry-run will copy {total_mb:.2f} MB to '{dest_dir}'.")
        sys.exit(0)

    # Check for existing destination files
    if dest_dir.exists() and any(dest_dir.iterdir()):
        if not args.force:
            print(f"\n⚠️ Destination directory '{dest_dir}' already contains files.")
            print("   Pass --force to explicitly allow overwriting the destination.")
            sys.exit(1)
        else:
            print(f"\n⚠️ Destination directory exists. Overwrite authorized via --force.")

    dest_dir.mkdir(parents=True, exist_ok=True)

    print("\n⏳ Copying files and verifying SHA-256 checksums...")
    verified_files = 0

    for src_file, rel_name, sz in files_to_copy:
        target_file = dest_dir / rel_name
        target_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"  📦 Copying {rel_name} ({sz / (1024 * 1024):.2f} MB)...", end="", flush=True)
        shutil.copy2(src_file, target_file)

        src_hash = compute_sha256(src_file)
        dst_hash = compute_sha256(target_file)

        if src_hash != dst_hash:
            print(" ❌ SHA-256 CHECKSUM MISMATCH!")
            print(f"     Source: {src_hash}")
            print(f"     Dest:   {dst_hash}")
            sys.exit(1)

        print(f" ✅ Verified ({src_hash[:12]}...)")
        verified_files += 1

    print("\n" + "=" * 70)
    print(f"🎉 Successfully exported and verified {verified_files} preset files!")
    print(f"Destination: {dest_dir}")
    print("RecipeGen AI can now load the base GPT-2 architecture in 100% offline environments.")
    print("=" * 70)


if __name__ == "__main__":
    main()
