# Tracker Pipeline Architecture: Multi-Threaded Stages with Drop-Oldest Backpressure

* Status: accepted
* Deciders: daendip2026
* Consulted: Claude Opus 4.8 (structure review)
* Created: 2026-05-26
* Last Modified: 2026-06-17

## Context and Problem Statement

This ADR is **post-hoc documentation**. The tracker is already implemented; this record captures the architectural framework of the tracker pipeline.

The tracker is the producer side of the wire schema defined in [system/ADR-0003](../system/0003-hand-data-structure.md). It must consume webcam frames and produce hand landmarks at frame rate while staying within the constraints inherited from [system/ADR-0001](../system/0001-architecture.md):

* 16.6ms / 33.3ms per-frame budget (60 / 30 FPS).
* No GC stalls on the hot path.
* Tracking-to-render latency < 100ms end-to-end.
* Single OS process per stage (the tracker itself runs as **one OS process**; cross-stage IPC to the avatar is handled by the system-level transport, out of scope here).

## Considered Options

### Option 1: Synchronous Monolithic Loop

Read frame → run inference → process → output, all sequentially on one thread inside a single loop.
* **Good**: Simplest mental model; no locks, no thread coordination, no queue invariants.
* **Bad**: Frame budget cannot be met. The slowest stage (MediaPipe inference) gates the entire loop, and any I/O blocking in capture or output stalls inference.

### Option 2: Multi-Process Pipeline

Run capture and inference in separate OS processes, communicate via local IPC (pipe, shared memory, or socket).
* **Good**: True parallelism unaffected by Python's GIL; process isolation contains crashes to one stage.
* **Bad**: Per-frame serialization of large NumPy arrays across the process boundary introduces both CPU cost and latency. The tracker is bound to a single machine ([system/ADR-0001](../system/0001-architecture.md)) and not horizontally scaled, so the parallelism gain that would justify the serialization cost does not materialize.

### Option 3: Multi-Threaded Single Process with Drop-Oldest Backpressure

Decompose the pipeline into stages connected by in-process queues. Use a dedicated worker thread for blocking I/O (capture), let MediaPipe's own callback thread feed an internal buffer, and run the consumer loop on the main thread. Apply drop-oldest backpressure between stages so the slowest stage cannot accumulate latency.
* **Good**: I/O blocking in capture does not stall inference; in-process memory sharing avoids serialization cost; drop-oldest backpressure structurally prevents latency growth (the loop always sees the freshest frame).
* **Bad**: Lock and queue invariants must be designed carefully. Intermediate frames are dropped under inference load — accepted because real-time tracking prefers the freshest frame over historical completeness.

## Decision Outcome

Chosen option: **Option 3**, because it is designed to meet the frame-budget constraint without paying the serialization cost of multi-process IPC, and because drop-oldest backpressure is the structural mechanism that prevents latency accumulation between stages.

### Pipeline Stages

| Stage | Owner module | Input | Output |
|---|---|---|---|
| Capture | `capture.py` | webcam (OpenCV) | `Frame` (raw BGR + acquisition timestamp) |
| Inference | `mediapipe_tracker.py` | `Frame` | `FrameResult` (tuple of `RawHandResult`) |
| Processing | `landmark_processor.py` | `FrameResult` | `ProcessedFrame` (tuple of `ProcessedHand` + FPS) |
| Output | `cli.py` / future wire emitter | `ProcessedFrame` | console / future protobuf wire |

Each stage is implemented as its own module with a single responsibility. Stages communicate only through the data-flow vocabulary above; they do not reach into each other's internals.

### Threading Model

* **Capture worker thread** (`WebcamCapture._read_loop`) — blocking `cv2.VideoCapture.read()` and frame timestamp anchoring; pushes to a `maxsize=1` drop-oldest queue.
* **MediaPipe internal worker thread** — invoked by `vision.HandLandmarker.detect_async`; results delivered via callback to a locked single-slot buffer.
* **Main thread** — consumes the capture queue, polls the MediaPipe result buffer, runs landmark processing, emits output, handles graceful shutdown (SIGINT/SIGTERM/SIGBREAK).
* **Logging listener thread** (`AsyncLoggerLifecycle`) — background drain of a bounded log-record queue.

All threads share memory in one process; no IPC, no serialization between stages.

### Cross-Cutting Patterns

* **Drop-oldest backpressure on in-process queues.** Both the capture-to-main frame queue and the logging-record queue apply the same policy: if a downstream consumer is slow, drop the oldest item rather than block or grow unbounded. Applied at `WebcamCapture._push_to_queue` (`capture.py`) and `ZeroLatencyQueueHandler.enqueue` (`logger.py`).
* **Immutable hand-off.** All inter-stage data types (`Frame`, `RawHandResult`, `FrameResult`, `ProcessedHand`, `ProcessedFrame`) are `frozen` dataclasses with `tuple` collections (not lists). A producer cannot mutate data after handing it off, eliminating one whole class of cross-thread bugs.
* **Pooled RGB conversion buffer.** The dominant per-frame allocation (BGR→RGB conversion) is removed via `MediaPipeTracker._rgb_pool` (`mediapipe_tracker.py`): a cyclic 5-slot pool written in place via `cv2.cvtColor(..., dst=slot)`. Smaller per-hand `(LANDMARK_COUNT, 3) float32` arrays are still allocated per detected hand; accepted as residual.
* **Context-managed lifecycle.** Resource-owning stages (`WebcamCapture`, `MediaPipeTracker`, `AsyncLoggerLifecycle`) are context managers; resource release is deterministic on exit, regardless of exception path.
* **High-resolution monotonic timing.** A single `_TimeAnchor` (wall-clock anchored to `perf_counter`) is captured once at capture open; all subsequent per-frame timestamps are derived from `perf_counter` deltas added to the wall-clock anchor. This sidesteps platform-specific `time.time()` granularity quirks (e.g. ~15ms on legacy Windows) while keeping a meaningful absolute reference.

## Consequences

### Accepted Trade-offs

* **Concurrency complexity.** Multi-threading introduces locks (`_hands_lock`, queue internals) and ordering invariants that must be respected by every subsystem that reads or writes shared state. Mitigated by keeping lock scopes minimal and using `frozen` types for cross-thread payloads.
* **Frames dropped under inference load.** When inference takes longer than a frame interval, intermediate capture frames are discarded. Accepted because real-time tracking prefers the freshest frame over historical completeness.
* **Python-side processing under the GIL.** Inference itself runs in MediaPipe's C++ layer and is GIL-free, but Python-side per-frame work (data wrapping, FPS tracking) runs under the GIL. Per-landmark data is held in NumPy arrays and passed by reference through the processor, so the current 21-landmark hand model has acceptable processing time within the frame budget.

### Validation Targets

* Per-frame consumer-loop latency is logged with a soft warning past `LATENCY_WARN_MS` (a diagnostic ceiling, not a tuned target; async inference is excluded from this measurement).
* Dominant per-frame allocation (BGR→RGB conversion) eliminated via pooling on the hot path — currently design intent.
* FPS sustained at or above `camera.target_fps` — the CLI surfaces actual vs target FPS on exit.

Benchmark measurements supporting these targets and the measurement methodology are recorded in [`tracker/PERFORMANCE.md`](../../../tracker/PERFORMANCE.md).

### Re-review Conditions

* Per-frame consumer-loop latency consistently approaches the frame interval under representative load → re-examine stage decomposition or threading model.
* MediaPipe inference latency becomes the dominant frame-budget consumer → consider GPU-accelerated inference (MediaPipe Tasks API supports a GPU mode) or model-variant changes.
* A second pipeline consumer (besides the avatar wire) materialises with different freshness/ordering requirements → revisit the drop-oldest assumption at the affected queue.
