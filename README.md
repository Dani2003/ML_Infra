# infer-rs: LLM Serving Infrastructure & PagedAttention Prototype

[![Build & Test](https://img.shields.io/badge/tests-18%20passed%20(CPU)-brightgreen.svg)]()
[![Rust](https://img.shields.io/badge/rust-2024%20edition-orange.svg)]()
[![C++/CUDA](https://img.shields.io/badge/C%2B%2B-17%20%2F%20CUDA-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()

`infer-rs` is an LLM inference serving prototype built from scratch in **Rust (`tokio`, `axum`)** and **C++/CUDA**. The project focuses on two core ML systems engineering components:
1. **Dynamic Continuous Batching:** An iteration-level request scheduler that fills batch slots dynamically at step boundaries rather than stalling on full sequence completion.
2. **PagedAttention Virtual Memory Management:** A fixed-size physical block allocator and block table mapping designed to eliminate memory fragmentation during generation.

> **Project Stage:** This repository is an active infrastructure prototype (Gate 1 & Gate 2 of the [Engineering Roadmap](docs/roadmap.md)). The request pipeline, scheduler, CPU reference kernels (18 automated tests passing on CPU fallback), and C++ memory manager are fully tested. The CUDA path is currently implemented with explicit device staging and compiled conditionally; physical GPU benchmarking requires an NVIDIA device with `nvcc`. Real model weight execution and tokenizer integration are currently under development.

---

## Architecture

```text
                                  +---------------------------------------+
                                  |         Client HTTP Requests          |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |     crates/api-server (Axum HTTP)     |
                                  |  - Input Validation & Bounded Limits  |
                                  |  - Prometheus Telemetry (/metrics)    |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |     crates/scheduler (Continuous)     |
                                  |  - Iteration-Level Request Batching   |
                                  |  - Dynamic On-the-Fly Slot Refills    |
                                  |  - RAII Single-Release Invariants     |
                                  +---------------------------------------+
                                           |                      |
                    (Virtual Block Tables) |                      | (Tokens/Step)
                                           v                      v
+-------------------------------------------------+     +-----------------------------------+
|            crates/kv-cache                      |     |       crates/model-runner         |
|  - Logical-to-Physical Block Table Routing      |     |  - Safe Rust RAII FFI Bridge      |
|  - logical_block = token / block_size           |     |  - GpuKVCacheManager Abstraction  |
|  - offset = token % block_size                  |     +-----------------------------------+
+-------------------------------------------------+                       |
                                                                          v (Native FFI)
                                                        +-----------------------------------+
                                                        |        cuda_kernels/ (C++/CUDA)   |
                                                        |  - Pre-allocated Physical KV Pool |
                                                        |  - LIFO Double-Free Checked Stack |
                                                        |  - PagedAttention Kernel (CUDA)   |
                                                        |  - Portable CPU Reference Fallback|
                                                        +-----------------------------------+
```

---

## Core Implemented Systems

### 1. Iteration-Level Continuous Batching (`crates/scheduler`)
* Evaluates sequence state at individual decode iteration boundaries rather than waiting for an entire batch to complete.
* When Request A finishes, its slot in the active batch is immediately refilled by Request B.
* Enforces bounded queue admission, explicit backpressure (`SubmitError::QueueFull`), and timeout cancellation between decode steps.

### 2. PagedAttention Memory Manager (`cuda_kernels/` & `crates/kv-cache`)
* Allocates physical KV-cache memory as fixed-size blocks ($B = 16$ or $32$ tokens) rather than allocating contiguous buffers per maximum sequence length upfront.
* Maps non-contiguous virtual token indices to physical hardware blocks:
  $$\text{logical\_block} = \lfloor i / B \rfloor, \quad \text{offset} = i \pmod B, \quad \text{physical\_addr} = \text{block\_table}[\text{logical\_block}]$$
* **Single-Release Guarantee:** Both the Rust `BlockPool` and C++ `KVCacheMemoryManager` maintain bitmasks tracking active allocations. Duplicate frees and invalid IDs are strictly rejected to prevent pool corruption.
* **RAII Reclamation:** Sequence blocks are tied to Rust's `ActiveRequest` lifecycle via `Drop`, ensuring all assigned blocks return to the free pool upon completion, cancellation, or timeout.

### 3. Dual-Mode Hardware Layer & FFI Bridge (`crates/model-runner`)
* `build.rs` automatically detects the presence of NVIDIA CUDA compilers (`nvcc`).
* **CUDA Mode:** Compiles device kernels with explicit host-to-device staging transfers and stream synchronization.
* **CPU Reference Mode:** When compiling on systems without an NVIDIA GPU (e.g., standard laptops or CI runners), compiles an equivalent C++ reference implementation, allowing the entire FFI boundary and lifecycle to be compiled and verified anywhere.

### 4. Telemetry & Observability (`/metrics`)
* Exposes Prometheus text-formatted counters and gauges:
  * `infer_rs_requests_accepted_total`, `infer_rs_requests_completed_total`
  * `infer_rs_decode_iterations_total`, `infer_rs_active_requests`
  * `infer_rs_kv_blocks_total`, `infer_rs_kv_blocks_free`
  * `infer_rs_kv_exhaustion_total`

---

## Current Status & Roadmap

| Feature / Milestone | Status | Notes |
| :--- | :--- | :--- |
| **Axum API & Process Lifecycle** | **Complete** | Typed config, bounded queue, timeouts, graceful shutdown |
| **Continuous Batching Scheduler** | **Complete** | Step-level slot refills, backpressure, cancellation |
| **PagedAttention Memory Manager** | **Complete** | Rust control-plane + C++ block pool with double-free checks |
| **Zero-Leak RAII Lifecycle** | **Complete** | Verified 100% block recovery across concurrent test requests |
| **Native FFI Bridge** | **Complete** | Safe Rust wrapper over C++ allocation and kernel entry points |
| **CUDA Attention Kernel** | **Prototype** | Implemented with device staging; GPU validation ongoing |
| **Model Weights & Prefill Phase** | **Planned (Gate 2)** | Decoder model selection (e.g. TinyLlama / Llama-3) |
| **Empirical Static vs Continuous Benchmark** | **Planned (Gate 3)** | Requires real tensor backend for valid comparison |

See [docs/roadmap.md](docs/roadmap.md) for the complete milestone gates and exit criteria.

---

## Building & Testing

### Prerequisites
* **Rust 1.85+** (`rustup default stable`)
* **C++ Compiler:** MSVC (Visual Studio 2022 C++ Build Tools) or GCC/Clang
* **CUDA Toolkit (Optional):** Automatically detected by `build.rs` if `nvcc` is installed.

### 1. Run Unit Tests (18 Automated Tests on CPU Fallback)
```powershell
cargo test
```
Validates:
* Block pool allocation invariants, capacity exhaustion, and intra-batch double-free rejection in both Rust and C++.
* Cache ownership contract: enforces that physical blocks must be actively allocated before writing or reading.
* Safe Rust FFI slice boundary enforcement, dimension checks, and unallocated/invalid block table ID rejection.
* Numerical correctness of PagedAttention against analytical mathematical references with non-zero K/V vectors across multi-block layouts.
* Scheduler continuous slot refills, bounded backpressure, cancellation, and RAII single-release cache cleanup.

### 2. Run the Inference Server
```powershell
cargo run -p api-server
```

Endpoints:
* `POST /v1/generate`: Submit inference requests
* `GET /health`: Health probe
* `GET /metrics`: Prometheus telemetry

### 3. Run the Concurrency Load Test
In a separate terminal, exercise the scheduler and memory manager under asynchronous load:
```powershell
python benchmarks/load_test.py --requests 60 --concurrency 15 --min-tokens 8 --max-tokens 32
```

---

## Project Structure

```text
infer-rs/
├── Cargo.toml                     # Cargo workspace manifest (Rust 2024)
├── CMakeLists.txt                 # CMake configuration for CUDA kernels
├── README.md                      # Architecture & project documentation
├── configs/
│   └── example.env                # Configuration reference
├── benchmarks/
│   ├── README.md                  # Benchmark methodology
│   └── load_test.py               # Asynchronous multi-client load testing script
├── cuda_kernels/                  # Hardware & C++/CUDA Layer
│   ├── include/
│   │   ├── memory_manager.cuh     # PagedAttention Block Allocator header
│   │   └── attention.cuh          # PagedAttention kernel declaration
│   └── src/
│       ├── memory_manager.cu      # C++ memory manager with double-free checks
│       └── attention.cu           # Attention kernel with explicit staging transfers
└── crates/
    ├── api-server/                # Axum HTTP API & Prometheus telemetry
    ├── scheduler/                 # Iteration-level continuous batching scheduler
    ├── model-runner/              # Safe Rust FFI bridge & PagedAttentionRunner
    └── kv-cache/                  # CPU/Control-plane BlockTable & BlockPool
```
