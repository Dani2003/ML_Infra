# Initial architecture

The API validates bounded input and submits requests through a bounded Tokio
channel. The scheduler owns the active request set, calls the runner once per
decode iteration, removes completed or disconnected requests, and admits new
work at the next iteration boundary. This demonstrates the scheduler contract;
the mock runner does not perform model inference.

The CPU KV block pool owns a fixed set of block IDs. A request block table maps
logical token blocks to those IDs. The pool is deliberately single-owner and
requires `&mut self`; a future execution design must choose where synchronization
belongs before adding shared access or FFI. No KV tensor storage or GPU pointers
exist in this milestone.

The simple scheduler currently creates a batch input vector and clones prompts
at each iteration. Those costs are visible and intentionally unoptimized until
there is a real backend and profile to guide changes. A GPU backend must not
block Tokio runtime workers while waiting for device work.

The `/metrics` endpoint currently exposes lifecycle counters and active request
count. It does not measure queue wait, time-to-first-token, per-token latency,
GPU memory, or device utilization. Add these only with measurement points tied
to the real execution lifecycle; the mock path cannot establish GPU behavior.
