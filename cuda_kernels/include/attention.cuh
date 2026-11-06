#pragma once

#include <cstdint>
#include <cstddef>

namespace infer_rs {

/// PagedAttention decode parameters for a batch of requests
struct PagedAttentionParams {
    const float* query;           // [num_seqs, num_heads, head_dim]
    const void* k_cache;          // Physical K pool: [num_blocks, block_size, num_heads, head_dim]
    const void* v_cache;          // Physical V pool: [num_blocks, block_size, num_heads, head_dim]
    const int32_t* block_tables;  // [num_seqs, max_blocks_per_seq]
    const int32_t* context_lens;  // [num_seqs]
    float* output;                // [num_seqs, num_heads, head_dim]

    int32_t num_seqs;
    int32_t num_heads;
    int32_t head_dim;
    int32_t block_size;
    int32_t max_blocks_per_seq;
    float scale;
};

extern "C" {
    /// Launches the PagedAttention decode operation across the active batch.
    /// Operates with CUDA kernel when compiled with CUDA; falls back to optimized CPU loop otherwise.
    void paged_attention_decode(const PagedAttentionParams* params);
}

} // namespace infer_rs

