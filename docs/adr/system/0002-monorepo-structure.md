# One Repository for the Tracker, the Avatar, and the Wire Schema

* Status: accepted
* Deciders: daendip2026
* Consulted: Claude Opus 4.7 (structure review)
* Created: 2026-05-18
* Last Modified: 2026-09-24

## Context and Problem Statement

HandForge has two stacks with unrelated toolchains, a Python tracker and a Unity avatar, and both build their bindings from one wire schema. When the schema changes, both sides must follow it, or they disagree about the wire.

### Hard Constraints

* **Deployed together.** The tracker and the avatar run on one machine over loopback ([ADR-0001](0001-architecture.md)), and neither has a consumer other than the other, so they are deployed together, not as independent deployment units.

## Considered Options

### Option 1: Multi-repo (Separate tracker and avatar Repositories)
* **Good**: A hard language and toolchain boundary, and a release cadence per repository.
* **Bad**: Each side has its own version history, so which tracker version works with which avatar version has to be tracked. Without that, the two sides can be built against different versions of the schema.

### Option 2: Single Monorepo (Chosen)
* **Good**: The schema and both sides share one version history, so one version of the repository identifies the schema and both sides together.
* **Bad**: The two sides share one repository lifecycle.

## Decision Outcome

Chosen option: **Option 2**. We will keep the tracker, the avatar, and the wire schema in one repository, so that they share one version history and no compatibility between repositories has to be tracked. The cost is one repository lifecycle for both sides.

## Consequences

### Accepted Trade-offs
* **Coupled repository lifecycle.** Accepted because, under the Hard Constraint, a separate release of one side is not needed.

### Re-review Conditions
* A production-grounded reason makes the tracker or the avatar an independent deployment unit, for example a remote or distributed deployment → re-open this decision.
