#include "../include/attention.cuh"
#include <cmath>
#include <vector>
#include <algorithm>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace infer_rs {

__global__ void paged_attention_kernel(
    const float* __restrict__ query,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ context_lens,
    float* __restrict__ output,
    int32_t num_seqs,
    int32_t num_heads,
    int32_t head_dim,
    int32_t block_size,
    int32_t max_blocks_per_seq,
    float scale)
{
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    if (seq_idx >= num_seqs || head_idx >= num_heads) return;

    int context_len = context_lens[seq_idx];
    if (context_len <= 0) return;

    const float* q_ptr = query + (seq_idx * num_heads + head_idx) * head_dim;
    float* out_ptr = output + (seq_idx * num_heads + head_idx) * head_dim;
    const int32_t* block_table = block_tables + seq_idx * max_blocks_per_seq;

    // Strides inside physical cache:
    // Layout: [num_blocks, block_size, num_heads, head_dim]
    int block_stride = block_size * num_heads * head_dim;
    int token_stride = num_heads * head_dim;
    int head_stride = head_dim;

    extern __shared__ float shared_mem[];
    float* q_shared = shared_mem; // [head_dim]
    float* logits = shared_mem + head_dim; // [context_len]

    // Load Q into shared memory
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        q_shared[d] = q_ptr[d];
    }
    __syncthreads();

    // Step 1: Compute attention scores Q * K^T
    for (int t = threadIdx.x; t < context_len; t += blockDim.x) {
        int logical_block = t / block_size;
        int block_offset = t % block_size;
        int physical_block = block_table[logical_block];

        const half* k_vec = k_cache + physical_block * block_stride 
                                    + block_offset * token_stride 
                                    + head_idx * head_stride;

        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q_shared[d] * __half2float(k_vec[d]);
        }
        logits[t] = score * scale;
    }
    __syncthreads();

    // Step 2: Softmax over logits (find max, subtract, exp, sum)
    float max_logit = -1e20f;
    for (int t = 0; t < context_len; ++t) {
        max_logit = fmaxf(max_logit, logits[t]);
    }

    float exp_sum = 0.0f;
    for (int t = 0; t < context_len; ++t) {
        logits[t] = expf(logits[t] - max_logit);
        exp_sum += logits[t];
    }
    float inv_sum = (exp_sum > 0.0f) ? (1.0f / exp_sum) : 0.0f;
    for (int t = 0; t < context_len; ++t) {
        logits[t] *= inv_sum;
    }
    __syncthreads();

    // Step 3: Weighted sum over V
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < context_len; ++t) {
            int logical_block = t / block_size;
            int block_offset = t % block_size;
            int physical_block = block_table[logical_block];

            const half* v_vec = v_cache + physical_block * block_stride 
                                        + block_offset * token_stride 
                                        + head_idx * head_stride;

            acc += logits[t] * __half2float(v_vec[d]);
        }
        out_ptr[d] = acc;
    }
}

} // namespace infer_rs
#endif

namespace infer_rs {

extern "C" void paged_attention_decode(const PagedAttentionParams* params) {
    if (!params || params->num_seqs <= 0 || params->num_heads <= 0) return;

#ifdef USE_CUDA
    dim3 grid(params->num_seqs, params->num_heads);
    dim3 block(std::min(params->head_dim, 256));
    size_t shared_bytes = (params->head_dim + 2048) * sizeof(float);

    paged_attention_kernel<<<grid, block, shared_bytes>>>(
        params->query,
        reinterpret_cast<const half*>(params->k_cache),
        reinterpret_cast<const half*>(params->v_cache),
        params->block_tables,
        params->context_lens,
        params->output,
        params->num_seqs,
        params->num_heads,
        params->head_dim,
        params->block_size,
        params->max_blocks_per_seq,
        params->scale
    );
    cudaDeviceSynchronize();
#else
    // CPU reference implementation of PagedAttention
    int block_stride = params->block_size * params->num_heads * params->head_dim;
    int token_stride = params->num_heads * params->head_dim;
    int head_stride = params->head_dim;

    const uint16_t* k_cache = reinterpret_cast<const uint16_t*>(params->k_cache);
    const uint16_t* v_cache = reinterpret_cast<const uint16_t*>(params->v_cache);

    for (int s = 0; s < params->num_seqs; ++s) {
        int context_len = params->context_lens[s];
        if (context_len <= 0) continue;

        const int32_t* block_table = params->block_tables + s * params->max_blocks_per_seq;

        for (int h = 0; h < params->num_heads; ++h) {
            const float* q = params->query + (s * params->num_heads + h) * params->head_dim;
            float* out = params->output + (s * params->num_heads + h) * params->head_dim;

            std::vector<float> logits(context_len, 0.0f);
            float max_val = -1e30f;

            for (int t = 0; t < context_len; ++t) {
                int logical_block = t / params->block_size;
                int block_offset = t % params->block_size;
                int phys_block = block_table[logical_block];

                const uint16_t* k_vec = k_cache + phys_block * block_stride 
                                                + block_offset * token_stride 
                                                + h * head_stride;

                float dot = 0.0f;
                for (int d = 0; d < params->head_dim; ++d) {
                    // Approximate float conversion from uint16_t for host mode
                    dot += q[d] * static_cast<float>(k_vec[d]);
                }
                logits[t] = dot * params->scale;
                max_val = std::max(max_val, logits[t]);
            }

            float sum_exp = 0.0f;
            for (int t = 0; t < context_len; ++t) {
                logits[t] = std::exp(logits[t] - max_val);
                sum_exp += logits[t];
            }
            float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

            for (int d = 0; d < params->head_dim; ++d) {
                float acc = 0.0f;
                for (int t = 0; t < context_len; ++t) {
                    int logical_block = t / params->block_size;
                    int block_offset = t % params->block_size;
                    int phys_block = block_table[logical_block];

                    const uint16_t* v_vec = v_cache + phys_block * block_stride 
                                                    + block_offset * token_stride 
                                                    + h * head_stride;
                    acc += (logits[t] * inv_sum) * static_cast<float>(v_vec[d]);
                }
                out[d] = acc;
            }
        }
    }
#endif
}

} // namespace infer_rs

