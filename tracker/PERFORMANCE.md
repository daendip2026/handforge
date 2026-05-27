# HandForge Tracker — Performance Measurements

* Last Modified: 2026-05-27
* Last Measurement Run: 2026-05-27

## §1 Purpose & Scope

This document is the *performance evidence record* for the HandForge tracker. It collects the KPIs the tracker is designed to meet, the methodology used to measure them, the measured benchmark values, and the pointers needed to reproduce those measurements. ADRs in `docs/adr/tracker/` cite values from this document when making quantitative claims.

**In scope**
- Performance KPIs (target values + verification source).
- Measurement methodology (hardware, tool settings, mock policy, run date).
- Measured benchmark results (current values from the pytest-benchmark suite).
- Coverage gaps (explicitly: things this document does *not* measure).
- Commands to reproduce measurements locally.

**Out of scope**
- System and camera tuning guidance → see [`TUNING.md`](TUNING.md).
- Architectural optimization rationale → see [tracker ADRs](../docs/adr/tracker/README.md).
- Actual MediaPipe inference latency → not measured by any benchmark here; see §5.

## §2 Performance Targets (KPIs)

| Metric | Target | Verification mechanism |
|---|---|---|
| Pipeline frame rate | ≥ 30 FPS (sustained) | Runtime `effective_fps` reported by `_PipelineStats` in `cli.py` |
| End-to-end latency (per-frame, mean) | ≤ 50 ms | `LATENCY_WARN_MS` constant in `cli.py`; runtime mean reported on exit by `_print_exit_summary` in `cli.py` |
| Detection rate | report-only (no fixed target) | Runtime `detection_rate_pct` from `_PipelineStats` in `cli.py` |

*Hot-path heap allocation* is a separate design target inherited from [tracker/ADR-0003](../docs/adr/tracker/0003-zero-allocation-memory-pooling.md); it has no current measurement mechanism, see §5.

Each target is traceable to either a code constant or a runtime measurement mechanism.

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

For the *date* of the §4 measurement values, see the `Last Measurement Run` field at the top of this document.

## §4 Measured Benchmark Results

Values below are from the `pytest-benchmark` run on the reference hardware (§3), on the date noted there. Re-running on different hardware will produce different values; the *methodology* is the stable contract, not the specific numbers.

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

## §5 Coverage Gaps

The benchmarks in §4 do not cover every dimension this tracker is designed for. The gaps below are recorded explicitly so readers do not assume §4 is complete.

### MediaPipe Inference Latency

*No benchmark in §4 measures actual MediaPipe inference time.* All `tracker-*` benchmarks substitute a `MagicMock` detector. This is intentional, not omission: real MediaPipe inference latency depends on the model file, the hardware, and the input image content, and per-environment variance is too large for a portable benchmark value.

**Substitute measurement mechanism — already implemented**:
- `_latest_inference_time_us` in `mediapipe_tracker.py` is recorded on each callback; this is propagated into per-frame `FrameResult.inference_time_us` and `RawHandResult.inference_time_us` (types defined in `types.py`).
- `_PipelineStats` in `cli.py` accumulates end-to-end latency (via its `record_latency` method), reports mean / min / max on exit through `_print_exit_summary`, and emits a structured JSON log entry on shutdown from `main()`.
- `LATENCY_WARN_MS` constant in `cli.py` (currently `50.0`) triggers per-frame warning logs when the threshold is exceeded.

To obtain a real-MediaPipe inference latency number for an ADR or design discussion: run the tracker for a representative session and read the end-of-run summary (see §6).

### Hot-Path Heap Allocation

The patterns described in [tracker/ADR-0003](../docs/adr/tracker/0003-zero-allocation-memory-pooling.md) (pre-allocated NumPy buffers, RGB pool, in-place `cv2.cvtColor`) target *near-zero heap allocation on the inference / processing hot path in steady state*.

**Status: design intent, not currently measured.** No `tracemalloc`-based test or benchmark exists in this project. The claim is currently inferred from the code patterns, not verified at runtime.

**Substitute measurement (not yet implemented)**: a `tracemalloc`-based test wrapping repeated `process()` calls in steady state would produce a quantitative allocation count. Adding this test is tracked as project work; until it lands, treat the zero-allocation claim as a design statement rather than a measurement.

## §6 How to Reproduce

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

**Collect real runtime stats** (the substitute for inference-latency benchmarks; §5):
```bash
cd tracker
uv run python -m hand_tracker
# Run for a representative duration, then Ctrl+C.
# Read the end-of-run summary table printed to the console,
# or find the corresponding JSON log entry in the logs/ directory.
```

**Deeper profiling with cProfile** (optional, for diagnosing specific hotspots):
```bash
cd tracker
uv run python -m cProfile -o logs/tracker_profile.stats -m hand_tracker
# Visualise with snakeviz (install separately):
# uv run snakeviz logs/tracker_profile.stats
```

## §7 References

**ADRs that cite this document**
- [tracker/ADR-0001 — Pipeline Architecture](../docs/adr/tracker/0001-tracker-architecture-overview.md) — `LATENCY_WARN_MS` and FPS targets.
- [tracker/ADR-0002 — Async Capture](../docs/adr/tracker/0002-async-capture-single-slot-buffer.md) — capture-latency benchmarks.
- [tracker/ADR-0003 — Zero-Allocation Hot Path](../docs/adr/tracker/0003-zero-allocation-memory-pooling.md) — heap-allocation target.

**Related project documents**
- [`tracker/TUNING.md`](TUNING.md) — system and camera tuning guidance (separate concern).
- [`tracker/config.yaml`](config.yaml) and [`tracker/src/hand_tracker/config.py`](src/hand_tracker/config.py) — full reference for every configurable setting.
