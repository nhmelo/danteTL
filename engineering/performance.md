# Measurement

Measure before changing scheduling, retention or accuracy. Current contracts are in
[architecture](architecture.md); acceptance commands are in [verification](verification.md).

- Record hardware, runtime versions, resolution, DPI, workload, warm-up and competing activity.
  Separate load/first compilation from warm inference; report median/p95 and memory where measured.
- Separate capture-to-start, recognition, visual work, readiness, queueing, translation and rendering.
  A faster isolated crop/model does not establish lower overlay latency or game overhead.
- Compare identical frames/replays against a baseline. Keep raw text/boxes, edit errors, actual
  coverage, retries and retained English, including first observations and negative cases.
  Confidence alone is not accuracy. Never trade coverage/retention for speed.
- Use explicit time for policy tests; generous real deadlines only for blocked-worker/cancel checks.
  Benchmark idle service owners as well as cached GUI reads. Polling unchanged state must not cause
  repeated discovery, inference, classification or repaint; real invalidations must still work.
- Bound queues/images/templates/caches; test dense/reordered content, slow OCR, typewriter growth
  and static WGC frames. Use isolated environments and owned synthetic windows. Demo results do
  not establish game, HDR, mixed-DPI or anti-cheat compatibility.

Tools include `benchmark_scan_selection.py`, `benchmark_crop_recovery.py`, `benchmark_regions.py`,
`benchmark_tracking.py`, `benchmark_icons.py` and [translation comparison](benchmarks/README.md). Read `--help`.
Keep evidence in ignored `artifacts/`, label its limits and remove disposable environments safely.
OCR consensus stays experimental; changing retention requires explicit replay/native acceptance.

Tracking similarity uses the bit-parallel LCS recurrence of
[Allison and Dix](https://www.sciencedirect.com/science/article/pii/0020019086900918), normalized as
[Indel similarity](https://rapidfuzz.github.io/RapidFuzz/Usage/distance/Indel.html).
This differs from [SequenceMatcher's alignment heuristic](https://docs.python.org/3/library/difflib.html):
validate assignment/retention replays, not just scores or speed. The implementation reuses character
masks within one spatially bounded matching pass and adds no native dependency.
SQLite eviction uses a transactional row count and indexed excess-row deletion because
[OFFSET traverses skipped rows](https://www.sqlite.org/rowvalue.html#scrolling_window_queries).
Measure full cache operations and hot-entry survival, not only the eviction query.
