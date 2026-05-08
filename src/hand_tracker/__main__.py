"""Enables `python -m hand_tracker` invocation."""

from __future__ import annotations

import sys

from hand_tracker.cli import main

if __name__ == "__main__":
    sys.exit(main())
