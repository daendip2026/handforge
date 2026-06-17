# HandForge Tracker

The Python tracking component of [HandForge](../README.md).
Real-time hand pose tracking pipeline using MediaPipe Tasks API and OpenCV.

## Requirements

- **Python** — 3.11, 3.12, or 3.13.
- **OS** — Windows (the project's verified development environment). Linux/macOS are not verified; see [`TUNING.md`](TUNING.md) §3 for backend caveats.
- **Webcam** — a built-in or USB camera is required at runtime. The default configuration is calibrated for the project's development setup; cameras with different native mirror orientation may invert `Left`/`Right` handedness labels (configuration toggles do not currently cover all camera types).
- **MediaPipe model file** — `hand_landmarker.task`, downloaded by `scripts/download_models.py` on first setup (requires network access).

## Setup

From the `tracker/` directory:

1. **Install dependencies:**
   ```bash
   uv sync --all-extras
   ```

2. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

3. **Download the MediaPipe model:**
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

The tracker emits performance stats throughout each session and a structured summary on exit. This is the canonical way to inspect MediaPipe inference latency, FPS, and end-to-end latency for your specific hardware — no portable benchmark covers these.

- **On exit** — the console prints a summary covering effective FPS, end-to-end latency, detection rate, and total frames. The same numbers are emitted as a JSON log entry to the configured log directory.
- **Per-frame warnings** — when end-to-end latency exceeds `LATENCY_WARN_MS` (50 ms by default, defined in `cli.py`), a per-frame warning is logged with the offending value.

Methodology, reproduction commands, and the rationale for the no-portable-benchmark choice are in [`PERFORMANCE.md`](PERFORMANCE.md).

## Further Reading

| Document | Purpose |
|---|---|
| [`docs/adr/tracker/`](../docs/adr/tracker/) | Architecture Decision Records — *why* the tracker is shaped this way (pipeline structure, MediaPipe boundary). |
| [`PERFORMANCE.md`](PERFORMANCE.md) | Performance evidence — KPIs, benchmark results, measurement methodology, coverage gaps. |
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
