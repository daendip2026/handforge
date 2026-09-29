# One Protobuf Schema for Both Sides of the Wire

* Status: accepted
* Deciders: daendip2026
* Consulted: Claude Opus 4.7 (structure review)
* Created: 2026-05-22
* Last Modified: 2026-09-29

## Context and Problem Statement

The tracker is written in Python and the avatar in C#. Both must read the bytes of every message the same way. More messages will be added, because [ADR-0001](0001-architecture.md) leaves the control-plane messages undefined. Each new message must work in both languages.

The avatar parses a message for every tracker frame. [ADR-0001](0001-architecture.md) requires that no garbage-collection pause show up in frame time.

The two sides are deployed together from one version of the repository ([ADR-0002](0002-monorepo-structure.md)), so the format does not have to work between different versions.

## Considered Options

### Option 1: Hand-written Encoding (Fixed Byte Layout or JSON)
* **Good**: Neither side adds a library or a code-generation step.
* **Good, fixed layout only**: The reader can read into the same arrays every frame, so parsing allocates nothing.
* **Bad**: Each language keeps its own hand-written copy of the layout, so every change is made twice and the two copies are kept in agreement by hand.

### Option 2: Protobuf (Chosen)
* **Good**: Both sides' code is generated from one schema. A message or field is added in one place, and regenerating shows whether the two sides still match.
* **Bad**: Both sides depend on the Protobuf runtime and a code-generation step. The generated C# parser allocates a new object for each message element it reads.

> Other schema-first encoders such as FlatBuffers were **not** compared.

## Decision Outcome

Chosen option: **Option 2**. We will define the wire format in one Protobuf schema and generate the tracker's and the avatar's code from it, so the format has one definition. The cost is the Protobuf runtime and code generation on both sides, and an allocation each time the avatar parses a message.

## Consequences

### Accepted Trade-offs
* **Runtime and code generation on both sides.** Accepted because the alternative is two hand-written copies of the same layout that must change together.
* **Allocation while parsing.** Accepted until the re-review condition below is met.

### Re-review Conditions
* A release target is missed and the diagnostics attribute the miss to message parsing → re-open this decision.
