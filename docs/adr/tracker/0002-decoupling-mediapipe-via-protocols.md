# Decoupling MediaPipe via Structural Typing Protocols

* Status: accepted
* Deciders: daendip2026
* Consulted: Claude Opus 4.8 (structure review)
* Created: 2026-05-27
* Last Modified: 2026-06-16

## Context and Problem Statement

This ADR is **post-hoc documentation** of how the tracker isolates itself from the MediaPipe Tasks API (`mediapipe.tasks.vision.HandLandmarker`).

MediaPipe Tasks API is an *external dependency* with three properties that, if left unmanaged, propagate hazards throughout the tracker codebase:

* **Opaque** — the Python surface is a thin wrapper over compiled C++ extensions; static analysers cannot introspect the underlying types, so MediaPipe-typed values appear as `Any` to type-checkers.
* **Evolving** — the Tasks API is a relatively new MediaPipe surface; its result shapes can change between releases.
* **Heavy** — the runtime depends on TFLite and supporting libraries; pulling it in solely to test or develop unrelated code is wasteful.

Without an explicit boundary between the tracker and the MediaPipe surface, every site that touches a MediaPipe result is *coupled to all three properties at once*: opacity contaminates static analysis project-wide; evolution requires multi-file edits; runtime weight is paid by every test that imports the consuming module. The architectural requirement is therefore a *managed boundary* with three properties of its own:

* **Explicitness** — the MediaPipe shape we depend on is declared in our own codebase, not implicit in scattered usage.
* **Locality** — a MediaPipe shape change touches one place, not many.
* **Verifiability** — we can confirm the boundary holds. (mypy strict mode and a no-real-landmarker test path are the verification *mechanisms* the tracker happens to use; they are how we know the boundary works, not the reason it exists.)

## Considered Options

### Option 1: No Explicit Boundary

Use MediaPipe types directly across the tracker codebase.
* **Good**: No abstraction layer; minimal code at the boundary.
* **Bad**: All three hazards above propagate as cross-cutting concerns. Opaque types become `Any` everywhere; any field rename forces edits across many files; tests cannot run without the MediaPipe runtime.

### Option 2: Data-Wrapping Boundary

Wrap MediaPipe results in our own classes (e.g. ABC + concrete subclasses) at the boundary; hand only our wrappers downstream.
* **Good**: Downstream code never touches MediaPipe types directly; the boundary is concrete and visible.
* **Bad**: Wrapping happens *per object, per frame*. At 21 landmarks × 2 hands × 30 FPS that is 1260 wrapper allocations per second on the hot path — directly hostile to the no-GC-stalls constraint inherited from [system/ADR-0001](../system/0001-architecture.md). The boundary cost is paid by every call site, not just once.

### Option 3: Contract-Typing Boundary (`typing.Protocol`)

Declare the *shapes* the tracker depends on as `Protocol` classes in our own code; pass the native MediaPipe objects through unchanged; the boundary lives in the *type contract*, not in *data shape*. Application code instantiates the real `HandLandmarker`; test code injects an alternative factory that returns objects satisfying the same Protocol.
* **Good**: Zero per-object cost — no wrapping, no allocation. The boundary is enforced statically (mypy checks every usage site against the Protocol). Tests inject dataclass mocks that satisfy the Protocol without the MediaPipe runtime.
* **Bad**: The Protocol is hand-maintained against MediaPipe's actual shape. An upstream rename or new required field is not caught at static analysis time; the only guard is the integration test against the real `HandLandmarker`.

## Decision Outcome

Chosen option: **Option 3** — a contract-typing boundary. The MediaPipe surface is captured as a *type contract* rather than as wrapped data; native MediaPipe objects flow through the tracker unchanged, while every usage site is statically constrained by Protocol declarations. `mediapipe_tracker.py` declares the contract via four `Protocol` classes (`MPLandmark`, `MPCategory`, `MPHandLandmarkerResult`, `MPHandLandmarker`).

### Code Realization

* The tracker accepts an optional factory parameter:
    ```python
    def __init__(
        self,
        mp_cfg: MediaPipeConfig,
        tracker_cfg: TrackerConfig,
        camera_cfg: CameraConfig,
        hand_landmarker_factory: type[MPHandLandmarker] | None = None,
    ) -> None:
    ```
* When a factory is provided (tests), it is invoked in place of `vision.HandLandmarker.create_from_options`. When `None` (application), the tracker instantiates the real Tasks API landmarker.
* All MediaPipe-typed values internal to the tracker are typed against the Protocols, not against `Any`. This propagates a strict type signature from the boundary downward.

## Consequences

### Accepted Trade-offs

* **Manual Protocol maintenance.** The Protocols mirror MediaPipe's current shape; an upstream change is not caught at static analysis time. The codebase's existing integration tests against the real `HandLandmarker` are the only line of defence; any change to the Protocol must be paired with a re-run of those tests.

### Validation Targets

* mypy strict mode passes against the full `hand_tracker` package with zero `Any` leakage from the MediaPipe boundary.
* The unit-test suite for `MediaPipeTracker` constructs mock results without instantiating the real `HandLandmarker` — verified by test fixtures that inject lightweight dataclass mocks satisfying the Protocols. (The MediaPipe library itself is still imported transitively through `mediapipe_tracker.py`.)

### Re-review Conditions

* MediaPipe Tasks API publishes a breaking change in result shape (e.g. renames `category_name`) → the Protocols must be updated and the integration tests re-validated; treat as a tracker-level migration.
* The MediaPipe project ships official Python type stubs covering the Tasks API → the Protocol layer becomes redundant; consider deleting it in favour of the upstream stubs.
