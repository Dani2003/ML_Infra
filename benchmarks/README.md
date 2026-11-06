# Benchmarks & Load Testing

The benchmark suite exercises the inference engine under high concurrent async load to measure throughput, latency distributions, and memory allocator invariants.

## Running the Benchmark

1. **Start the `infer-rs` server** in one terminal:
   ```powershell
   cargo run -p api-server --release
   ```

2. **Run the high-concurrency load test** in a second terminal:
   ```powershell
   python benchmarks/load_test.py --requests 100 --concurrency 20 --min-tokens 8 --max-tokens 32
   ```

## Measured Metrics

* **Tokens per Second (Throughput):** Aggregate generation throughput under multi-client concurrency.
* **Latency Percentiles (p50, p90, p95, p99):** Round-trip end-to-end request latency distribution.
* **Average Inter-Token Latency (ITL):** Average delay between successive token generations.
* **PagedAttention Memory Invariant:** Asserts that 100% of physical blocks assigned to requests during inference are reclaimed upon completion, returning free blocks to baseline without leaks.
* **Continuous Batching vs. Static Batching Speedup:** Computes the throughput gain achieved by continuous iteration-level slot refills over fixed-batch padding bubbles.
