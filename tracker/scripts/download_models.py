#!/usr/bin/env python3
"""
Model download script for HandForge.
Downloads the necessary MediaPipe Tasks models.
"""

import sys
import urllib.request
from pathlib import Path

# Constants
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_FILENAME = "hand_landmarker.task"


def download_progress(block_num: int, block_size: int, total_size: int) -> None:
    """Callback to display download progress."""
    if total_size <= 0:
        return

    downloaded = block_num * block_size
    progress = min(100, (downloaded / total_size) * 100)

    # Progress bar style
    bar_length = 30
    filled_length = int(bar_length * progress // 100)
    bar = "=" * filled_length + "-" * (bar_length - filled_length)

    sys.stdout.write(
        f"\rDownloading: [{bar}] {progress:.1f}% ({downloaded / (1024 * 1024):.1f} MB)"
    )
    sys.stdout.flush()


def main() -> None:
    # Ensure the models directory exists in the project root
    root_dir = Path(__file__).parent.parent
    models_dir = root_dir / "models"
    models_dir.mkdir(exist_ok=True)

    target_path = models_dir / MODEL_FILENAME

    if target_path.exists():
        print(f"Model already exists at: {target_path}")
        print("Delete it if you want to re-download.")
        return

    print(f"Starting download of {MODEL_FILENAME}...")
    print(f"Source: {MODEL_URL}")
    print(f"Target: {target_path}")

    try:
        urllib.request.urlretrieve(MODEL_URL, target_path, download_progress)
        print("\n\nDownload complete!")
    except Exception as e:
        print(f"\n\nError downloading model: {e}")
        if target_path.exists():
            target_path.unlink()
        sys.exit(1)


if __name__ == "__main__":
    main()
