"""
HandForge_python — real-time hand tracker entry point script.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap — enables `python scripts/run_tracker.py` without
# requiring the package to be installed into the venv first.
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

try:
    from hand_tracker.cli import main
except ImportError as exc:
    print(
        f"Error: Could not import hand_tracker package. Ensure you are running from the project root or it is installed. ({exc})"
    )
    sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())
