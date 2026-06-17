# HandForge Tracker — Performance Measurements

* Last Modified: 2026-06-17
* Last Measurement Run: 2026-05-27 (§4 benchmarks)

## §1 Purpose & Scope

This document is the *performance evidence record* for the HandForge tracker. It collects the KPIs the tracker is designed to meet, the methodology used to measure them, the measured benchmark values, and the pointers needed to reproduce those measurements. ADRs in `docs/adr/tracker/` cite values from this document when making quantitative claims.

**In scope**
- Performance KPIs (target values + verification source).
- Measurement methodology (hardware, tool settings, mock policy, run date).
- Measured benchmark results (current values from the pytest-benchmark suite).
- Commands to reproduce measurements locally.

**Out of scope**
- System and camera tuning guidance → see [`TUNING.md`](TUNING.md).
- Architectural optimization rationale → see [tracker ADRs](../docs/adr/tracker/README.md).
- MediaPipe inference latency → not captured here. The §4 benchmarks mock the detector, and the current runtime instrumentation does not measure inference latency reliably (single mutable submission timestamp, plus per-frame fan-out of one cached value); reliable measurement is tracked as project work.

## §2 Performance Targets (KPIs)

| Metric | Target | Verification mechanism |
|---|---|---|
| Pipeline frame rate | sustain the configured `target_fps` | Runtime `effective_fps` vs `target_fps`, reported by `_PipelineStats` in `cli.py` |
| Per-frame consumer-loop latency | report-only; soft diagnostic warning past `LATENCY_WARN_MS` | mean / min / max via `_print_exit_summary` in `cli.py`; per-frame warning logged when an iteration exceeds `LATENCY_WARN_MS` |
| Detection rate | report-only (no fixed target) | Runtime `detection_rate_pct` from `_PipelineStats` in `cli.py` |

Each row is traceable to a code constant or a runtime measurement mechanism. Note: the latency metric is the consumer-loop iteration time (async inference excluded); `LATENCY_WARN_MS` is a loose diagnostic ceiling, not a tuned target.

## §3 Measurement Methodology

**Hardware (reference run)**
- CPU: Intel Core Ultra 7 258V @ 3.30 GHz (laptop, 8 cores)
- OS: Windows 11
- Power state: AC adapter connected (no battery throttling)

**Tool**: `pytest-benchmark`, pedantic mode. Per-benchmark `iterations`, `rounds`, and `warmup_rounds` are defined in the test functions (see `tests/test_*.py`).

**Mock policy** — *all benchmarks in §4 use mocks*:
- `tracker-*`: real `MediaPipeTracker.process()` code path with a `MagicMock` detector. Inference itself is not exercised.
- `landmark-processor-*`: real `LandmarkProcessor.update()` / `console_summary()` on a synthetic `FrameResult`. MediaPipe is not involved.
- `capture-*`: real `WebcamCapture` queue / synchronisation logic with `cv2.VideoCapture` patched out. Hardware camera is not involved.

## §4 Measured Benchmark Results

Values below are from the `pytest-benchmark` run on the reference hardware (§3). Re-running on different hardware will produce different values; the *methodology* is the stable contract, not the specific numbers.

### tracker (groups: `tracker-hot-path`, `tracker-polling`)

| Benchmark | Mean | Min | Max | OPS | Measures | Mocked |
|---|---|---|---|---|---|---|
| `test_benchmark_processing_hot_path` | 0.88 ms | 0.76 ms | 0.99 ms | ~1,138 | `MediaPipeTracker.process()` full mapping path on a new frame | Detector (`MagicMock`) |
| `test_benchmark_polling_efficiency` | 0.73 µs | 0.58 µs | 1.59 µs | ~1,364,000 | `process()` zero-redundancy bypass when the same frame is observed | Detector (`MagicMock`) |

### landmark-processor (groups: `landmark-processor-update`, `landmark-processor-summary`)

| Benchmark | Mean | Min | Max | OPS | Measures | Mocked |
|---|---|---|---|---|---|---|
| `test_benchmark_processor_update` | 1.59 µs | 1.45 µs | 2.75 µs | ~628,000 | `LandmarkProcessor.update()` on a synthetic `FrameResult` | MediaPipe layer (synthetic input replaces real detection) |
| `test_benchmark_console_summary` | 13.23 µs | 12.11 µs | 17.68 µs | ~75,560 | `console_summary()` string formatting | MediaPipe layer (synthetic input replaces real detection) |

### capture (groups: `capture-latency-unthrottled`, `capture-latency-throttled`)

| Benchmark | Mean | Min | Max | OPS | Measures | Mocked |
|---|---|---|---|---|---|---|
| `test_benchmark_latency_unthrottled` | 68.97 µs | 39.39 µs | 326.77 µs | ~14,500 | Producer→consumer queue transfer overhead, unthrottled | `cv2.VideoCapture` |
| `test_benchmark_latency_240fps` | 4.66 ms | 3.48 ms | 6.35 ms | ~215 | Same, simulated at 240 FPS (4.16 ms inter-frame interval) | `cv2.VideoCapture` |
| `test_benchmark_latency_60fps` | 17.45 ms | 15.05 ms | 20.79 ms | ~57 | Same, simulated at 60 FPS — value is throttle-bound, not code-bound | `cv2.VideoCapture` |
| `test_benchmark_latency_30fps` | 34.86 ms | 32.65 ms | 60.34 ms | ~29 | Same, simulated at 30 FPS — value is throttle-bound, not code-bound | `cv2.VideoCapture` |

**Reading note**: the `latency_30fps` / `latency_60fps` numbers approximate the simulated inter-frame interval *by design*; they validate that the throttling mechanism produces the expected delay, not that the code itself is that slow. The `unthrottled` and `240fps` rows reflect actual code-path latency.

## §5 How to Reproduce

**Run the `pytest-benchmark` suite**:
```bash
cd tracker
uv run pytest --benchmark-only
```

**Save results to JSON** (for historical tracking / CI):
```bash
cd tracker
uv run pytest --benchmark-only --benchmark-json=logs/benchmark_results.json
```

**Collect per-frame latency runtime stats**:
```bash
cd tracker
uv run python -m hand_tracker
# Run for a representative duration, then Ctrl+C.
# The exit-summary table prints per-frame consumer-loop latency (mean / min / max) and FPS.
```

**Deeper profiling with cProfile** (optional, for diagnosing specific hotspots):
```bash
cd tracker
uv run python -m cProfile -o logs/tracker_profile.stats -m hand_tracker
# Visualise with snakeviz (install separately):
# uv run snakeviz logs/tracker_profile.stats
```

## §6 References

**ADRs that cite this document**
- [tracker/ADR-0001 — Pipeline Architecture](../docs/adr/tracker/0001-tracker-architecture-overview.md) — the §4 benchmarks and §3 methodology backing its Validation Targets (FPS sustain, consumer-loop processing cost).

**Related project documents**
- [`tracker/TUNING.md`](TUNING.md) — system and camera tuning guidance (separate concern).
- [`tracker/config.yaml`](config.yaml) and [`tracker/src/hand_tracker/config.py`](src/hand_tracker/config.py) — full reference for every configurable setting.
