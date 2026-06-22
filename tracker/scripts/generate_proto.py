"""
Generate Python protobuf bindings from the shared wire schema.

Re-run this whenever that schema changes; the generated ``handtracking_pb2.py``
and ``handtracking_pb2.pyi`` are derived artifacts, never hand-edited.

Usage (from the ``tracker/`` directory)::

    uv run python scripts/generate_proto.py

The compiler is ``protoc`` bundled inside ``grpcio-tools`` (a dev dependency),
invoked via ``grpc_tools.protoc`` so the entire toolchain lives in the project
lockfile — no system ``protoc`` install required.
"""

from __future__ import annotations

import sys
from pathlib import Path

from grpc_tools import protoc

# This file is tracker/scripts/generate_proto.py
TRACKER_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = TRACKER_ROOT.parent
PROTO_DIR = REPO_ROOT / "proto"
PROTO_FILE = PROTO_DIR / "handtracking.proto"
# Output directly into the package so `from hand_tracker import handtracking_pb2`
# resolves (the Python module name is the .proto filename, not its package).
OUT_DIR = TRACKER_ROOT / "src" / "hand_tracker"


def main() -> int:
    if not PROTO_FILE.is_file():
        print(f"proto schema not found: {PROTO_FILE}", file=sys.stderr)
        return 1

    # argv[0] is the program name and is ignored by protoc.main.
    args = [
        "grpc_tools.protoc",
        f"-I{PROTO_DIR}",
        f"--python_out={OUT_DIR}",
        f"--pyi_out={OUT_DIR}",
        str(PROTO_FILE),
    ]
    rc: int = protoc.main(args)
    if rc != 0:
        print(f"protoc failed (exit code {rc})", file=sys.stderr)
        return rc

    print(f"generated handtracking_pb2.py(+.pyi) in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
