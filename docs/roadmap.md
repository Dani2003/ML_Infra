# Engineering roadmap

This roadmap is ordered around evidence. Each stage must work and be measurable
before the next stage adds lower-level complexity.

## Gate 1: Operational Rust service (current)

- Typed configuration and startup validation.
- Bounded queue, active request cap, request/body limits, cancellation, deadline,
  and shutdown paths.
- Mock decode runner for exercising scheduler lifecycle.
- Prometheus lifecycle counters and active request gauge.

**Exit evidence:** compile on the documented Rust toolchain; automated tests for
allocator invariants, queue saturation, cancellation, deadlines, runner failure,
and continuous slot refill; a repeatable API smoke run. Performance numbers are
not meaningful with the mock runner.

## Gate 2: One real model backend

First record the target GPU model, VRAM, driver, CUDA version, and supported
inference libraries. Then select one decoder-only model and precision that fit
that hardware. Keep tokenization, model loading, prefill, and decode behind the
runner interface. Load model weights once at process startup, fail readiness if
initialization fails, and report model identity and precision in startup info.

The runner contract should distinguish prompt prefill from one-token decode.
Track request context length, EOS, cancellation, and maximum context. Define
device ownership and stream synchronization before crossing an FFI boundary.

**Exit evidence:** correctness comparison against the chosen backend's reference
generation for a fixed corpus, bounded VRAM behavior, repeatable startup and
shutdown, and measured request latency on the declared GPU.

## Gate 3: Scheduler and cache under real load

- Add separate prefill and decode queues with explicit scheduling policy.
- Integrate per-request block tables with the backend's KV writes and reads.
- Ensure a finished/cancelled request releases blocks exactly once.
- Define allocator exhaustion behavior (reject, pause, or preempt) and expose
  block usage, allocation failures, queue wait, time-to-first-token, and
  inter-token latency.
- Compare iteration scheduling with a fixed/static batching baseline using the
  same model, tokenized request trace, limits, warm-up, and GPU.

**Exit evidence:** correctness under varied request lengths and cancellation,
allocator accounting that returns to baseline after each run, and published
latency/throughput curves with hardware and workload details.

## Gate 4: Production deployment controls

Add authentication and network policy, TLS termination guidance, readiness and
liveness semantics, bounded graceful-drain behavior, structured request IDs,
configuration versioning, dependency and container scanning, and deployment
resource limits. Document tested driver/runtime combinations. Treat custom
attention kernels as a separate optimization requiring profiler evidence and
correctness comparison against the selected backend.

## Current next action

Record the target GPU and software stack, then select the one model/backend for
Gate 2. Without that hardware fact, model selection, CUDA memory budgeting, and
valid benchmark targets would be guesswork.
