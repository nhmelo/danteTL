# Engineering guide

DanteTL captures a selected Windows window, recognizes the selected source language locally and overlays translations from
an explicitly selected local, Ollama or cloud translator. These documents describe current behavior.

| Read when changing | Guidance |
| --- | --- |
| Ownership, processing, UI, models, setup or security | [Architecture](architecture.md) |
| Tests or native acceptance | [Verification](verification.md) |
| Packaging, dependencies or CI | [Building](building.md) |
| Dependency advisories, security scans or signing | [Security maintenance](security.md) |
| Performance or model comparisons | [Measurement](performance.md), [benchmark inputs](benchmarks/README.md) |

Keep each contract in one place. Code owns implementation constants; regression tests specify behavior.
Docs explain ownership and constraints needed to change them safely. Update stale guidance rather
than accumulating plans/results. Store disposable evidence under ignored `artifacts/`.
