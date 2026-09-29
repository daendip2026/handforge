# Separate-Process Architecture with Data/Control Plane Channel Separation

* Status: accepted
* Deciders: daendip2026
* Consulted: Claude Opus 4.7 (architecture proposal)
* Created: 2026-05-21
* Last Modified: 2026-09-23

## Context and Problem Statement

HandForge tracks hands with MediaPipe in Python. Unity interprets the result, solving IK and mapping the hand pose onto the VRM avatar's bones, and renders it. Tracking and interpretation therefore sit in two stacks on one machine, and this ADR records how they are connected: the process boundary and the communication topology.

Per-frame hand data flows between the two stacks, and only the newest frame is of any use. When this decision was made, control messages such as calibration and lifecycle signaling were also expected to cross the boundary, and those were taken to be messages that must not be lost.

### Hard Constraints
These are the constraints assumed when this decision was made. The current release targets are set by the [PRD](../../PRD.md#6-acceptance-criteria).

* **Single-machine, loopback-only.** Remote tracking is an explicit non-goal. This constraint exists specifically to prevent the architecture from being justified by, or drifting toward, distributed deployment.
* **Frame budget.** 60 FPS = 16.67ms, 120 FPS = 8.33ms per frame. The hot path must stay within budget.
* **No GC spikes.** A garbage-collection stall during a live session is a broadcast failure, not a performance footnote.
* **Tracking-to-render latency < 100ms.**

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

> Same-machine IPC mechanisms such as shared memory and named pipes were **not** compared.

## Decision Outcome

Chosen option: **Option 3**. We will run tracking and rendering as separate processes on one machine, send per-frame hand data over loopback UDP and drop stale frames, and send control messages that must not be lost over a separate TCP channel. Each kind of message gets the delivery it needs, and a tracker crash does not stop rendering. The cost is two transports instead of one, and the serialization and transport work of crossing a process boundary on every frame.

### Transport Channels
* **Data plane (UDP):** Per-frame landmark/pose data. Latest-frame-wins; stale frames are dropped rather than buffered.
* **Control plane (TCP):** Messages that cannot be lost (e.g. calibration, lifecycle/control signaling).

## Consequences

### Accepted Trade-offs
* **Two transports instead of one.** Accepted as the simplest resolution for conflicting delivery requirements (stale-drop data vs. lossless control).
* **IPC across the process boundary.** Serialization and loopback transport cost on the hot path. Accepted because tracking and rendering run on different runtimes, and in exchange for process isolation.

### Validation Targets
* No garbage-collection pause shows up in frame time under sustained load. This is judged from the output frame time the avatar application records itself. Whether a stall came from garbage collection is told apart by attaching the profiler to a development build.

### Re-review Conditions
* A production-grounded reason breaks the single-machine assumption → re-evaluate transport decision.
* A release target is missed and the diagnostics attribute the miss to crossing the process boundary → re-evaluate the process split.

### Non-Goals
* Protobuf concrete implementation.
* Message bodies (sole exception: `timestamp`).
* VMC concrete implementation or VMC/OSC output.
