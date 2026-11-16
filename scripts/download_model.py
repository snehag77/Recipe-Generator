# -*- coding: utf-8 -*-
"""Model Artifact Download and Integrity Verification Utility for RecipeGen AI.

Safely downloads large model weight artifacts (e.g., GPT-2 fine-tuned checkpoints)
from user-specified remote hosting, verifies checksum integrity, and safely
deploys weights into the local models directory.

Strict Operational Standards:
1. Never executes automatically during imports or app startup.
2. Never overwrites existing checkpoint files unless explicitly requested with --overwrite.
3. Downloads to a temporary staging file before verifying integrity.
4. Validates SHA-256 (or MD5) checksum before moving to final destination.
5. Deletes temporary staging file if download or checksum fails.
6. Does NOT hardcode fabricated URLs or fake checksums.

Usage:
    python scripts/download_model.py \
        --url "https://storage.example.com/models/recipe_gpt2_full_best.weights.h5" \
        --dest "models/recipe_gpt2_full/recipe_gpt2_full_best.weights.h5" \
        --sha256 "EXPECTED_HEX_SHA256_STRING"
"""

import argparse
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def compute_hash(filepath: Path, algorithm: str = "sha256") -> str:
    """Compute cryptographic hash of a file efficiently using 64KB blocks."""
    if algorithm.lower() == "sha256":
        hasher = hashlib.sha256()
    elif algorithm.lower() == "md5":
        hasher = hashlib.md5()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def verify_existing_artifact(
    filepath: Path,
    expected_size: Optional[int] = None,
    expected_sha256: Optional[str] = None,
    expected_md5: Optional[str] = None,
) -> bool:
    """Perform preflight integrity verification on an existing local model file.

    Args:
        filepath: Path to the local file to verify.
        expected_size: Optional expected size in bytes.
        expected_sha256: Optional expected SHA-256 hex string.
        expected_md5: Optional expected MD5 hex string.

    Returns:
        True if all specified criteria pass, False otherwise.
    """
    filepath = filepath.resolve()
    print("=" * 70)
    print("RecipeGen AI: Checkpoint Preflight Integrity Verification")
    print("=" * 70)
    print(f"Target File: {filepath}")

    if not filepath.exists():
        print(f"❌ Error: Target file does not exist at '{filepath}'.")
        return False
    if not filepath.is_file():
        print(f"❌ Error: Target path '{filepath}' is not a file.")
        return False

    actual_size = filepath.stat().st_size
    print(f"Actual Size: {actual_size:,} bytes ({actual_size / (1024 * 1024):.2f} MB)")

    if expected_size is not None:
        if actual_size != expected_size:
            print(f"❌ SIZE MISMATCH: Expected {expected_size:,} bytes, found {actual_size:,} bytes.")
            return False
        print(f"✅ Size verification passed: {actual_size:,} bytes.")

    if expected_sha256:
        print("🔍 Computing SHA-256 checksum...")
        actual_sha = compute_hash(filepath, "sha256")
        if actual_sha.lower() != expected_sha256.strip().lower():
            print(f"❌ SHA-256 MISMATCH!")
            print(f"   Expected: {expected_sha256.strip().lower()}")
            print(f"   Actual:   {actual_sha}")
            return False
        print(f"✅ SHA-256 checksum verified: {actual_sha}")

    if expected_md5:
        print("🔍 Computing MD5 checksum...")
        actual_md5 = compute_hash(filepath, "md5")
        if actual_md5.lower() != expected_md5.strip().lower():
            print(f"❌ MD5 MISMATCH!")
            print(f"   Expected: {expected_md5.strip().lower()}")
            print(f"   Actual:   {actual_md5}")
            return False
        print(f"✅ MD5 checksum verified: {actual_md5}")

    if not expected_sha256 and not expected_md5 and expected_size is None:
        print("ℹ️ File exists. (No checksum or size criteria were specified for verification).")

    print("🎉 Verification SUCCESSFUL: File meets all specified integrity criteria.")
    return True


def download_artifact(
    url: str,
    dest_path: Path,
    expected_size: Optional[int] = None,
    expected_sha256: Optional[str] = None,
    expected_md5: Optional[str] = None,
    overwrite: bool = False,
    timeout: int = 60,
) -> bool:
    """Download a file with progress reporting and checksum verification.

    Args:
        url: Remote HTTP/HTTPS URL of the model artifact.
        dest_path: Target local file path.
        expected_sha256: Optional expected SHA-256 hex string.
        expected_md5: Optional expected MD5 hex string.
        overwrite: If True, allow replacing an existing file at dest_path.
        timeout: Network socket timeout in seconds.

    Returns:
        True if download and verification succeeded, False otherwise.
    """
    dest_path = dest_path.resolve()

    # Pre-condition checks
    if dest_path.exists():
        if not overwrite:
            print(f"❌ Error: Target file already exists at '{dest_path}'.")
            print("   Pass --overwrite (or -f) to explicitly allow replacing this file.")
            return False
        else:
            print(f"⚠️ Target file exists. Overwrite authorized for '{dest_path}'.")

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Use a staging file in the same directory to allow atomic rename
    temp_file = dest_path.with_suffix(dest_path.suffix + f".tmp_{int(time.time())}")

    print("=" * 70)
    print("RecipeGen AI: Model Artifact Downloader & Integrity Verifier")
    print("=" * 70)
    print(f"Source URL:        {url}")
    print(f"Destination Path:  {dest_path}")
    if expected_sha256:
        print(f"Expected SHA-256:  {expected_sha256}")
    if expected_md5:
        print(f"Expected MD5:      {expected_md5}")
    print("=" * 70)

    try:
        import requests
    except ImportError:
        print("❌ Error: 'requests' library is not installed. Please install requirements.txt.")
        return False

    print("⏳ Connecting to remote server...")
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
    except Exception as e:
        print(f"❌ Connection or HTTP error: {e}")
        return False

    total_bytes = int(response.headers.get("content-length", 0))
    if total_bytes > 0:
        print(f"📦 Total artifact size: {total_bytes / (1024 * 1024):.2f} MB ({total_bytes:,} bytes)")
    else:
        print("📦 Content length not reported by server (streaming mode).")

    print(f"⬇️ Downloading to temporary staging file: {temp_file.name}")
    downloaded_bytes = 0
    start_time = time.time()
    last_print = start_time

    try:
        with open(temp_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    downloaded_bytes += len(chunk)

                    now = time.time()
                    if now - last_print > 1.0 or (total_bytes and downloaded_bytes == total_bytes):
                        elapsed = max(now - start_time, 0.001)
                        speed_mb = (downloaded_bytes / (1024 * 1024)) / elapsed
                        if total_bytes > 0:
                            pct = (downloaded_bytes / total_bytes) * 100
                            sys.stdout.write(
                                f"\r   [{pct:5.1f}%] {downloaded_bytes / (1024 * 1024):.1f} MB "
                                f"/ {total_bytes / (1024 * 1024):.1f} MB ({speed_mb:.1f} MB/s)"
                            )
                        else:
                            sys.stdout.write(
                                f"\r   Downloaded: {downloaded_bytes / (1024 * 1024):.1f} MB ({speed_mb:.1f} MB/s)"
                            )
                        sys.stdout.flush()
                        last_print = now
        print("\n✅ Download stream completed successfully.")
    except Exception as e:
        print(f"\n❌ Error during download stream: {e}")
        if temp_file.exists():
            temp_file.unlink(missing_ok=True)
            print("   🧹 Temporary staging file cleaned up.")
        return False

    # Check for empty file
    if downloaded_bytes == 0 or not temp_file.exists():
        print("❌ Error: Downloaded 0 bytes. Artifact is empty.")
        if temp_file.exists():
            temp_file.unlink(missing_ok=True)
        return False

    # Check expected content length if available
    if expected_size is not None and downloaded_bytes != expected_size:
        print(f"❌ Size mismatch: expected {expected_size:,} bytes, received {downloaded_bytes:,} bytes.")
        temp_file.unlink(missing_ok=True)
        return False
    elif total_bytes > 0 and downloaded_bytes != total_bytes:
        print(f"❌ Size mismatch: expected {total_bytes:,} bytes, received {downloaded_bytes:,} bytes.")
        temp_file.unlink(missing_ok=True)
        return False

    # Checksum Verification
    if expected_sha256 or expected_md5:
        print("🔍 Verifying cryptographic checksums...")

        if expected_sha256:
            actual_sha = compute_hash(temp_file, "sha256")
            print(f"   Computed SHA-256: {actual_sha}")
            if actual_sha.lower() != expected_sha256.strip().lower():
                print(f"❌ SHA-256 MISMATCH! Rejecting download.")
                print(f"   Expected: {expected_sha256.strip().lower()}")
                print(f"   Actual:   {actual_sha}")
                temp_file.unlink(missing_ok=True)
                print("   🧹 Staging file removed.")
                return False
            print("   ✅ SHA-256 checksum verified.")

        if expected_md5:
            actual_md5 = compute_hash(temp_file, "md5")
            print(f"   Computed MD5:    {actual_md5}")
            if actual_md5.lower() != expected_md5.strip().lower():
                print(f"❌ MD5 MISMATCH! Rejecting download.")
                print(f"   Expected: {expected_md5.strip().lower()}")
                print(f"   Actual:   {actual_md5}")
                temp_file.unlink(missing_ok=True)
                print("   🧹 Staging file removed.")
                return False
            print("   ✅ MD5 checksum verified.")
    else:
        print("⚠️ Warning: No checksum provided. Integrity verification skipped.")

    # Atomic move
    try:
        temp_file.replace(dest_path)
        print(f"🎉 Checkpoint successfully placed at: {dest_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to move staging file to final destination: {e}")
        if temp_file.exists():
            temp_file.unlink(missing_ok=True)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Secure Model Checkpoint Downloader & Integrity Verifier for RecipeGen AI"
    )
    parser.add_argument(
        "--dest",
        required=True,
        help="Local target file path (e.g., models/recipe_gpt2_full/recipe_gpt2_full_best.weights.h5)",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Remote HTTP/HTTPS URL of the model weights artifact (required unless --verify-only)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Preflight verification mode: verify local file existence, size, and checksum without downloading",
    )
    parser.add_argument(
        "--expected-size",
        type=int,
        default=None,
        help="Expected exact size of the artifact in bytes",
    )
    parser.add_argument("--sha256", default=None, help="Expected SHA-256 hexadecimal checksum")
    parser.add_argument("--md5", default=None, help="Expected MD5 hexadecimal checksum")
    parser.add_argument(
        "--overwrite",
        "-f",
        action="store_true",
        help="Explicitly allow overwriting an existing local file",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Socket connection and read timeout in seconds (default: 60)",
    )
    args = parser.parse_args()

    dest = Path(args.dest)
    if not dest.is_absolute():
        dest = PROJECT_ROOT / dest

    if args.verify_only:
        success = verify_existing_artifact(
            filepath=dest,
            expected_size=args.expected_size,
            expected_sha256=args.sha256,
            expected_md5=args.md5,
        )
        sys.exit(0 if success else 1)

    if not args.url:
        parser.error("--url is required when not running in --verify-only mode.")

    success = download_artifact(
        url=args.url,
        dest_path=dest,
        expected_size=args.expected_size,
        expected_sha256=args.sha256,
        expected_md5=args.md5,
        overwrite=args.overwrite,
        timeout=args.timeout,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
