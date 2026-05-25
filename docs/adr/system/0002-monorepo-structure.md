# Single Monorepo Structure (tracker/ + avatar/ + proto/ + docs/)

* Status: accepted
* Deciders: daendip2026
* Consulted: Claude Opus 4.7 (structure review)
* Created: 2026-05-18
* Last Modified: 2026-05-25

## Context and Problem Statement

This ADR is **post-hoc documentation**. The decision was made during planning and committed at the start of implementation; this record captures it after the fact, describing only what actually happened.

### What actually happened

- A multi-repo layout was a genuinely reviewed alternative during planning, not a strawman. The review cost was real.
- No multi-repo artifacts were ever physically created. In particular, the Unity (`avatar`) repository was never created. The pre-existing `handforge-python` repository was public.
- The plan was changed to a single monorepo before implementation proceeded, and implementation began against that single repository.
- The actual mechanical action was: rename `handforge-python` → `handforge` and add the additional top-level folders (`avatar/`, `proto/`, `docs/`) while moving the existing Python sources under `tracker/` with history preserved.
- Because no separate repositories existed, there was **no code or repository sunk cost**. This is therefore a *commitment* to the final structure, not an integration of two existing repos and not a reversal of a deployed layout.

### Final structure

```
handforge/
├── tracker/   # Python / MediaPipe
├── avatar/    # Unity / VRM
├── proto/     # single root .proto (wire schema source of truth)
└── docs/      # ADRs, documentation
```

### Hard constraints

- `tracker` (Python) and `avatar` (Unity) run on a single machine over loopback (see [ADR-0001](0001-architecture.md)). Under this constraint, they are **deployed together, not as independent units** (though the architecture remains capable of independent deployment if the constraint is lifted).
- `proto/` is the single source of truth for the wire schema. Both stages generate code from it.

## Considered Options

### Option 1: Multi-repo (Separate tracker and avatar Repositories)

Reviewed during planning, then not adopted.
* **Good**: Hard language/toolchain boundary, per-repository release cadence, and isolation of Unity history-bloat risk from the tracker.
* **Bad**: Under the single-machine constraint ([ADR-0001](0001-architecture.md)), per-repository release cadence does not materialize, leaving only coordination overhead. Splitting `proto/` and its two generated consumers across repository boundaries permits version skew (schema drift) between separate commits/merges.

### Option 2: Single Monorepo

Adopt a single monorepo containing `tracker/`, `avatar/`, `proto/`, and `docs/`.
* **Good**: Enables atomic cross-stage changes (a wire-schema change is a single commit updating both generated codebases). Prevents schema skew and avoids coordination overhead of multiple repositories.
* **Bad**: Couples repository lifecycles. Risks Git history bloat as the avatar stage gains large binary content (VRM models, textures) and Unity `Library/` artifacts.

## Decision Outcome

Chosen option: **Option 2**, because a wire-schema change requires atomic cross-stage updates to prevent version skew, and the independent release cycle benefits of a multi-repo layout do not materialize under the single-machine loopback constraint.

## Consequences

### Accepted Trade-offs

* **No independent CI or release boundary per stage.** A monorepo couples the two stages' repository lifecycle. This cost is accepted because the independent-release benefit it would buy does not exist under the single-machine constraint, so the coupling costs nothing that is actually used today.

### Validation Targets

Not applicable. The failure mode this decision prevents — schema skew between `tracker` and `avatar` — is structurally impossible in a single repository, so there is no runtime metric to measure.

### Re-review Conditions

* Re-open this decision if a production-grounded reason makes either `tracker` or `avatar` an independent deployment unit — for example, a remote/distributed deployment. Note that such a deployment is currently a non-goal under the single-machine hard constraint ([ADR-0001](0001-architecture.md)).

### Non-Goals

* A Git LFS / large-binary strategy (VRM, textures, Unity `Library/`). Deferred until before the first commit introducing such assets to `avatar/`. Stated here so history is not silently polluted before a decision is made.
