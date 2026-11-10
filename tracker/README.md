# HandForge Tracker

The Python tracking component of [HandForge](../README.md).
Real-time hand pose tracking pipeline using MediaPipe Tasks API and OpenCV.

## Requirements

- **Python** — 3.11, 3.12, or 3.13.
- **OS** — Windows (the project's verified development environment). Linux/macOS are not verified; see [`TUNING.md`](TUNING.md) for backend caveats.
- **Webcam** — a built-in or USB camera is required at runtime. The tracker gives MediaPipe the camera frame without mirroring it, so `Left` is the subject's left hand and `Right` the subject's right hand, as the [`HandLandmarkerResult` documentation](https://github.com/google-ai-edge/mediapipe/blob/8c2f96c001beaf8d3e8eeebfe64f3bcfd1e8ac03/mediapipe/tasks/python/vision/hand_landmarker.py#L250-L252) defines them. With MediaPipe 0.10.33 and a Logitech C922, the performer's left hand was labelled `Left`. If a camera driver mirrors its own output, the two labels swap, and no configuration setting corrects that.
- **MediaPipe model file** — `hand_landmarker.task`, downloaded by `scripts/download_models.py` on first setup (requires network access).

## Setup

From the `tracker/` directory:

1. **Install dependencies:**
   ```bash
   uv sync --all-extras
   ```

2. **Generate the protobuf bindings:**
   The wire types are generated from [`/proto/handtracking.proto`](../proto/handtracking.proto)
   — the single source of truth — and are **not** committed. Regenerate them after
   cloning, and again whenever the schema changes:
   ```bash
   uv run python scripts/generate_proto.py
   ```

3. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

4. **Download the MediaPipe model:**
   ```bash
   uv run python scripts/download_models.py
   ```

## Usage

The tracking pipeline is driven by `config.yaml`. Every value can be overridden via environment variables following the pattern `HANDFORGE__<SECTION>__<KEY>` — see [`.env.example`](.env.example) for reference.

### Run the tracker

```bash
# Via developer script (recommended)
uv run python scripts/run_tracker.py

# As a Python module
uv run python -m hand_tracker

# Via installed entry point
uv run handforge-tracker
```

### CLI Options

| Flag                | Default       | Description                                              |
| ------------------- | ------------- | -------------------------------------------------------- |
| `--view`            | off           | Enable real-time OpenCV debug visualization              |
| `--full-dump`       | off           | Print all 21 landmarks per frame to stdout               |
| `--log-level LEVEL` | from config   | Override log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `--config PATH`     | `config.yaml` | Path to configuration file                               |

**Example** — run with visualization and debug logging:

```bash
uv run python scripts/run_tracker.py --view --log-level DEBUG
```

### Runtime Stats

The tracker emits performance stats throughout each session and a structured summary on exit, giving FPS and per-frame latency for your specific hardware.

- **On exit** — the console prints a summary covering effective FPS, per-frame latency, detection rate, and total frames. The same numbers are emitted as a JSON log entry to the configured log directory.
- **Per-frame warnings** — when a consumer-loop iteration exceeds `LATENCY_WARN_MS` (a soft diagnostic threshold defined in `cli.py`), a per-frame warning is logged with the offending value.

Methodology and reproduction commands are in [`PERFORMANCE.md`](PERFORMANCE.md).

## Further Reading

| Document | Purpose |
|---|---|
| [`PERFORMANCE.md`](PERFORMANCE.md) | Performance evidence — KPIs, benchmark results, measurement methodology. |
| [`TUNING.md`](TUNING.md) | System & camera tuning guide — USB bandwidth/FOURCC, OS backends, auto-focus/exposure rationale. |

## Development

### Lint & Format

```bash
uv run ruff check src tests scripts
uv run ruff format src tests scripts
```

### Static Type Checking

Mypy runs in strict mode — all code must be fully annotated.

```bash
uv run mypy src tests scripts
```

### Tests

```bash
# Run tests with coverage
uv run pytest

# Run benchmarks only
uv run pytest --benchmark-only
```

### CI

GitHub Actions runs lint, format check, and mypy on every push/PR against Python 3.11, 3.12, and 3.13. See [`.github/workflows/lint.yml`](../.github/workflows/lint.yml).

## License

[MIT](../LICENSE)
