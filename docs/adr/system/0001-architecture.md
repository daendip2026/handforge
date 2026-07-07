# Separate-Process Architecture with Data/Control Plane Channel Separation

* Status: accepted
* Deciders: daendip2026
* Consulted: Claude Opus 4.7 (architecture proposal)
* Created: 2026-05-21
* Last Modified: 2026-05-25

## Context and Problem Statement

HandForge couples a Python-based hand-tracking stage (MediaPipe) to a Unity-based interpretation-and-render stage on a single-machine, loopback-only host. This ADR records the decision on the process boundary and communication topology.

### Hard Constraints
* **Single-machine, loopback-only.** Remote tracking is an explicit non-goal. This constraint exists specifically to prevent the architecture from being justified by, or drifting toward, distributed deployment.
* **Frame budget.** 60 FPS = 16.6ms, 120 FPS = 8.3ms per frame. The hot path must stay within budget.
* **No GC spikes.** A garbage-collection stall during a live session is a broadcast failure, not a performance footnote.
* **Tracking-to-render latency < 100ms.** Beyond this, expressive fidelity collapses.

### Key Context
* **Output path.** Unity performs final rendering; OBS captures the render output for streaming. No VMC in this path — Unity *is* the render application, so an inter-application avatar protocol has no slot in the delivery chain.
* **Unity's role.** Unity is an **interpretation layer** (IK solving, VRM pose mapping), not a transport passthrough. The interpretation responsibility is what justifies the second process.

## Considered Options

### Option 1: Single Process (Python Embedded in Unity)
Embedding MediaPipe directly within Unity via native plugins or an embedded Python runtime.
* **Good**: Eliminates IPC latency and serialization overhead on the hot path.
* **Bad**: MediaPipe carries native dependencies that make in-process embedding fragile. The IPC saving does not offset the cross-stack integration cost and runtime risk (a native crash terminates the entire rendering process).

### Option 2: Separate Processes, Single Channel
Both landmark data and control signaling over a single transport channel.
* **Good**: Simplest transport topology — single socket, single port.
* **Bad**: Data plane requires stale-drop semantics (an old frame is worthless; deliver the newest or nothing). Control plane requires lossless delivery. A single channel cannot honor both simultaneously: a transport tuned for lossless ordered delivery buffers stale frames the data plane wants discarded, while a transport tuned for latest-frame-wins drops control messages that must not be lost.

### Option 3: Separate Processes, Channel Separation — UDP Data / TCP Control (Chosen)
UDP for high-frequency landmark data, TCP for reliable control signaling.
* **Good**: Satisfies both delivery requirements simultaneously — stale-drop for data, lossless for control — with process isolation (tracker crash does not take down the renderer).
* **Bad**: Two transports instead of one. Channel separation adds operational surface (two sockets, two failure modes) relative to a single-channel design.

## Decision Outcome

Chosen option: **Option 3**, because the data-plane (stale-drop) and control-plane (lossless) delivery requirements are incompatible on a single transport, and channel separation resolves this using only OS-native network primitives.

### Transport Channels
* **Data plane (UDP):** Per-frame landmark/pose data. Latest-frame-wins; stale frames are dropped rather than buffered.
* **Control plane (TCP):** Messages that cannot be lost (e.g. calibration, lifecycle/control signaling).

### Design Rationale

* **`ITransport` is grounded in the current dual-channel need.** The abstraction exists because two distinct transports (UDP data, TCP control) coexist today. That it would also accommodate a future remote deployment is an incidental side effect, not the design driver — the single-machine constraint means remote distribution is not a current requirement.

* **`ISerializer` makes the wire format reversible; only the schema semantics are irreversible.** Protobuf sits behind `ISerializer`, so the serialization format can be swapped at bounded cost. What is irreversible is the *semantics* of the wire schema (field meaning, tag assignment), not the choice of Protobuf as the encoder.

* **VMC is isolated as an output-backend change axis.** The output backend lives inside the avatar stage with zero impact on the tracker↔avatar protocol. Only a design-level seam is recorded; materializing it as code is deferred until a second output backend has a production-grounded reason to exist.

## Consequences

### Accepted Trade-offs
* **Two transports instead of one.** Accepted as the simplest resolution for conflicting delivery requirements (stale-drop data vs. lossless control).
* **IPC across the process boundary.** Serialization and loopback transport cost on the hot path. Accepted in exchange for process isolation and the language boundary (Python ML ↔ Unity rendering).

### Validation Targets (Performance Stress Testing)
* Communication-layer latency < 15ms.
* Zero per-frame GC allocation on the hot path.

### Re-review Conditions
* GC measurements under load violate the frame budget → evaluate serializer swap via `ISerializer`.
* A production-grounded reason breaks the single-machine assumption → re-evaluate transport decision.

### Non-Goals
* Protobuf concrete implementation.
* Message bodies (sole exception: `timestamp`).
* VMC concrete implementation or VMC/OSC output.

### Prediction vs. Actual
To be completed after stress testing measurements.
* Prediction: [TBD]
* Actual: [TBD]
* Cause of divergence: [TBD]
