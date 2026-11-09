# Benchmarks & Load Testing

The benchmark suite exercises the `infer-rs` request pipeline, iteration scheduler, and PagedAttention block allocator under concurrent asynchronous load.

## Scope & Purpose

* **What is measured:** Tokio async admission backpressure, iteration-level continuous slot refills, request latency distributions, and C++ physical block allocator accounting (allocations vs. frees).
* **Current status:** The engine currently drives the scheduler lifecycle with a deterministic mock runner. Performance numbers reflect scheduler and control-plane dispatching capability. GPU tensor execution, prefill phases, and real-model comparisons against static batching baselines will be conducted in subsequent roadmap gates once hardware is attached.

## Running the Load Test

1. **Start the `infer-rs` server** in one terminal:
   ```powershell
   cargo run -p api-server --release
   ```

2. **Run the high-concurrency load test** in a second terminal:
   ```powershell
   python benchmarks/load_test.py --requests 100 --concurrency 20 --min-tokens 8 --max-tokens 32
   ```

## Output Metrics

* **Request Completion & Failure Rate:** Confirms bounded channel admission and timeout handling.
* **Round-Trip Latency Percentiles (p50, p90, p95, p99):** Measures end-to-end client latency under bursty concurrency.
* **Block Allocator Accounting:** Verifies that all physical blocks allocated during request generation are cleanly returned to the free pool upon completion.
