#pragma once

#include <cstdint>
#include <vector>
#include <mutex>
#include <memory>
#include <stdexcept>
#include <iostream>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA Runtime Error at: ") + 
                                 file + ":" + std::to_string(line) + 
                                 " '" + func + "' \n" + cudaGetErrorString(err));
    }
}
using half_type = half;
#else
// Host-side representation of 16-bit floating point for portable CPU mode
using half_type = uint16_t;
#endif

namespace infer_rs {

class KVCacheMemoryManager {
private:
    int32_t num_blocks_;
    int32_t block_size_;
    int32_t num_heads_;
    int32_t head_dim_;

    // Pointers to physical KV cache memory pool
    half_type* d_k_cache_;
    half_type* d_v_cache_;

    // Host-side fast LIFO stack of available physical block IDs
    std::vector<int32_t> free_blocks_;
    std::mutex mtx_;

    bool is_cuda_allocated_;

public:
    KVCacheMemoryManager(int32_t num_blocks, int32_t block_size, int32_t num_heads, int32_t head_dim);
    ~KVCacheMemoryManager();

    // Allocate a physical block ID. Returns -1 if Out Of Memory (OOM).
    int32_t allocate_block();

    // Release a single physical block back to the free pool.
    bool free_block(int32_t block_id);

    // Release multiple physical blocks at once when a sequence finishes.
    void free_blocks(const int32_t* block_ids, size_t count);

    // Block status inspection
    int32_t get_free_block_count();
    int32_t get_total_block_count() const { return num_blocks_; }
    int32_t get_block_size() const { return block_size_; }
    int32_t get_num_heads() const { return num_heads_; }
    int32_t get_head_dim() const { return head_dim_; }

    // Direct access to the raw device/host cache pointers for kernel execution
    half_type* get_k_cache_ptr() const { return d_k_cache_; }
    half_type* get_v_cache_ptr() const { return d_v_cache_; }
};

// C-compatible factory functions for FFI
extern "C" {
    KVCacheMemoryManager* kv_cache_manager_create(
        int32_t num_blocks, int32_t block_size, int32_t num_heads, int32_t head_dim);
    void kv_cache_manager_destroy(KVCacheMemoryManager* mgr);
    int32_t kv_cache_manager_allocate(KVCacheMemoryManager* mgr);
    bool kv_cache_manager_free(KVCacheMemoryManager* mgr, int32_t block_id);
    void kv_cache_manager_free_batch(KVCacheMemoryManager* mgr, const int32_t* block_ids, size_t count);
    int32_t kv_cache_manager_free_count(KVCacheMemoryManager* mgr);
    void* kv_cache_manager_k_ptr(KVCacheMemoryManager* mgr);
    void* kv_cache_manager_v_ptr(KVCacheMemoryManager* mgr);
}

} // namespace infer_rs

