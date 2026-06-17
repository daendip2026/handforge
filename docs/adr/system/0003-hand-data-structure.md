# Hand-Tracking Data Wire Representation Semantics

* Status: accepted
* Deciders: daendip2026
* Consulted: Claude Opus 4.8 (structure review)
* Created: 2026-05-22
* Last Modified: 2026-06-17

## Context and Problem Statement

This ADR is **post-hoc documentation**. The tracker (Python, MediaPipe Tasks API, `vision.HandLandmarker`, using the full landmark model `hand_landmarker.task`) is already implemented, and the semantics recorded below were already fixed during that implementation and at initial tracker setup. This is documentation of settled facts, not a new decision. No alternative was reverse-engineered for this record; only the encoder alternatives actually weighed at the time (see Considered Options) are described.

### Documented Semantics

- **Landmarks.** 21 points per hand, matching the MediaPipe HandLandmarker model.
- **Two coordinate semantics.** The data carries two distinct, well-defined coordinate frames:
  - *image-space*: normalized `[0,1]` for (x, y); z is wrist-relative depth.
  - *metric*: hand-centric, origin at the hand's approximate geometric center, in metres ([MediaPipe HandLandmarker Python guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)). This is **not** an absolute world coordinate.
- **Handedness.** Each detected hand is classified by a `Handedness` enum. On the wire, values are `Left`, `Right`, or `Unknown` — the last is an explicit fallback used when the side cannot be determined from the model output. The label's ground-truth meaning depends on the tracker's selfie-mode handling (`mirror_input` configuration) and MediaPipe's handedness convention.
- **Time.** Time is represented as a `uint64` microsecond value. Which messages carry a timestamp is a field-layout matter, out of scope here.

### Hard constraints

- The data model is a **changing axis**: the hand-keypoint set may evolve (21 points → additional hand keypoints). Lowering the cost of change along this axis is the objective of the representation design.

## Considered Options

### Option 1: Protobuf

Adopt Protobuf as the default serialization format.
* **Good**: Schema-first codegen for both Python and C# (prevents producer/consumer drift), and backward-compatible additive fields matching the evolving hand-keypoint model.
* **Bad**: Introduces dependency on Protobuf library and codegen compiler.

### Option 2: Manual Binary / JSON (non-schema formats)

Manual binary serialization or JSON format.
* **Good**: Zero compilation step or external library dependency.
* **Bad**: Without schema-first code generation, the two stages must be kept in sync by hand, which invites producer/consumer version drift.

> FlatBuffers, MessagePack, and similar encoders were **not** compared at this time. Because the encoder is reversible behind `ISerializer`, they remain candidates for a future swap rather than alternatives weighed here. They are referenced only under Re-review Conditions.

## Decision Outcome

Chosen option: **Option 1**, because a schema-first encoder prevents producer/consumer version drift, and Protobuf's backward-compatible field addition matches the requirement to evolve the hand-keypoint model additively.

## Consequences

### Accepted Trade-offs

* **Dependency on a serialization framework.** Protobuf introduces a codegen compiler step and runtime libraries. This cost is accepted because it is isolated behind the `ISerializer` abstraction, allowing a future encoder swap if needed.

### Validation Targets

* Not applicable. The documented semantics and the encoder selection are not themselves measurable. The encoder's hot-path GC cost is, however, validated by the GC performance stress testing defined in [ADR-0001](0001-architecture.md).

### Re-review Conditions

* **GC performance stress testing violates the frame budget** ➔ swap the encoder behind `ISerializer`. Only at this point are FlatBuffers and similar encoders compared.
* **The hand-landmark model changes from 21 points** (e.g. additional hand keypoints) ➔ evolve the schema additively; do not reuse tag numbers.
* **Absolute world coordinates become necessary** ➔ supplement the metric semantics with a separate `WorldTransform`; the hand-centric meaning defined in this ADR is retained, not replaced.

