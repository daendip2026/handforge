# HandForge System — Architecture Decision Records (ADRs)

This directory contains the ADRs whose scope crosses the **tracker** (Python) and **avatar** (Unity, .NET) boundary, including the wire contract between them.

Decisions contained in the Python side live in [`../tracker/`](../tracker/README.md).

## Document Index

| Number | Title | Status |
|:---:|:---|:---:|
| [0001](0001-architecture.md) | Separate-Process Architecture with Data/Control Plane Channel Separation | accepted |
| [0002](0002-monorepo-structure.md) | Single Monorepo Structure (tracker/ + avatar/ + proto/ + docs/) | accepted |
| [0003](0003-hand-data-structure.md) | Hand-Tracking Data Wire Representation Semantics | accepted |
