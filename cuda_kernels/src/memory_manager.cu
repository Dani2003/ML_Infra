#include "../include/memory_manager.cuh"
#include <cstdlib>
#include <algorithm>

namespace infer_rs {

KVCacheMemoryManager::KVCacheMemoryManager(
    int32_t num_blocks, int32_t block_size, int32_t num_heads, int32_t head_dim)
    : num_blocks_(num_blocks),
      block_size_(block_size),
      num_heads_(num_heads),
      head_dim_(head_dim),
      d_k_cache_(nullptr),
      d_v_cache_(nullptr),
      is_cuda_allocated_(false)
{
    if (num_blocks <= 0 || block_size <= 0 || num_heads <= 0 || head_dim <= 0) {
        throw std::invalid_argument("KVCache dimensions must be positive integers");
    }

    // Total elements per cache = num_blocks * block_size * num_heads * head_dim
    size_t total_elements = static_cast<size_t>(num_blocks_) * block_size_ * num_heads_ * head_dim_;
    size_t total_bytes = total_elements * sizeof(half_type);

#ifdef USE_CUDA
    CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&d_k_cache_), total_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(reinterpret_cast<void**>(&d_v_cache_), total_bytes));
    is_cuda_allocated_ = true;
    std::cout << "[KVCache] Allocated " << (total_bytes * 2) / (1024 * 1024)
              << " MB on CUDA device." << std::endl;
#else
    // Host fallback for CPU/portable builds
    d_k_cache_ = static_cast<half_type*>(std::malloc(total_bytes));
    d_v_cache_ = static_cast<half_type*>(std::malloc(total_bytes));
    if (!d_k_cache_ || !d_v_cache_) {
        throw std::bad_alloc();
    }
    std::fill_n(d_k_cache_, total_elements, 0);
    std::fill_n(d_v_cache_, total_elements, 0);
    is_cuda_allocated_ = false;
#endif

    // Populate free blocks stack (LIFO: 0, 1, ..., num_blocks-1)
    free_blocks_.reserve(num_blocks_);
    for (int32_t i = num_blocks_ - 1; i >= 0; --i) {
        free_blocks_.push_back(i);
    }
}

KVCacheMemoryManager::~KVCacheMemoryManager() {
#ifdef USE_CUDA
    if (is_cuda_allocated_) {
        if (d_k_cache_) cudaFree(d_k_cache_);
        if (d_v_cache_) cudaFree(d_v_cache_);
    }
#else
    if (d_k_cache_) std::free(d_k_cache_);
    if (d_v_cache_) std::free(d_v_cache_);
#endif
    d_k_cache_ = nullptr;
    d_v_cache_ = nullptr;
}

int32_t KVCacheMemoryManager::allocate_block() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (free_blocks_.empty()) {
        return -1; // Out of memory
    }
    int32_t block_id = free_blocks_.back();
    free_blocks_.pop_back();
    return block_id;
}

bool KVCacheMemoryManager::free_block(int32_t block_id) {
    if (block_id < 0 || block_id >= num_blocks_) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mtx_);
    free_blocks_.push_back(block_id);
    return true;
}

void KVCacheMemoryManager::free_blocks(const int32_t* block_ids, size_t count) {
    if (!block_ids || count == 0) return;
    std::lock_guard<std::mutex> lock(mtx_);
    for (size_t i = 0; i < count; ++i) {
        int32_t id = block_ids[i];
        if (id >= 0 && id < num_blocks_) {
            free_blocks_.push_back(id);
        }
    }
}

int32_t KVCacheMemoryManager::get_free_block_count() {
    std::lock_guard<std::mutex> lock(mtx_);
    return static_cast<int32_t>(free_blocks_.size());
}

// C-compatible FFI wrappers
extern "C" {

KVCacheMemoryManager* kv_cache_manager_create(
    int32_t num_blocks, int32_t block_size, int32_t num_heads, int32_t head_dim) 
{
    try {
        return new KVCacheMemoryManager(num_blocks, block_size, num_heads, head_dim);
    } catch (...) {
        return nullptr;
    }
}

void kv_cache_manager_destroy(KVCacheMemoryManager* mgr) {
    delete mgr;
}

int32_t kv_cache_manager_allocate(KVCacheMemoryManager* mgr) {
    return mgr ? mgr->allocate_block() : -1;
}

bool kv_cache_manager_free(KVCacheMemoryManager* mgr, int32_t block_id) {
    return mgr ? mgr->free_block(block_id) : false;
}

void kv_cache_manager_free_batch(KVCacheMemoryManager* mgr, const int32_t* block_ids, size_t count) {
    if (mgr) {
        mgr->free_blocks(block_ids, count);
    }
}

int32_t kv_cache_manager_free_count(KVCacheMemoryManager* mgr) {
    return mgr ? mgr->get_free_block_count() : 0;
}

void* kv_cache_manager_k_ptr(KVCacheMemoryManager* mgr) {
    return mgr ? static_cast<void*>(mgr->get_k_cache_ptr()) : nullptr;
}

void* kv_cache_manager_v_ptr(KVCacheMemoryManager* mgr) {
    return mgr ? static_cast<void*>(mgr->get_v_cache_ptr()) : nullptr;
}

} // extern "C"

} // namespace infer_rs

