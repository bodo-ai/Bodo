#pragma once

#include <cassert>

#include "_memory.h"

#define ALIGNMENT 64  // preferred alignment for AVX512

// ******** copied from Numba
// NRT = NumbaRunTime
// Related to managing memory between Numba and C++.
// TODO: make Numba C library

typedef void (*NRT_dtor_function)(void *ptr, size_t size, void *dtor_info);
typedef void *(*NRT_malloc_func)(size_t size);
typedef void *(*NRT_realloc_func)(void *ptr, size_t new_size);
typedef void (*NRT_free_func)(void *ptr);

struct MemInfoAllocator {
    MemInfoAllocator(NRT_malloc_func malloc_, NRT_realloc_func realloc_,
                     NRT_free_func free_)
        : malloc(malloc_), realloc(realloc_), free(free_) {}
    NRT_malloc_func malloc;
    NRT_realloc_func realloc;
    NRT_free_func free;
};

struct MemSys {
    MemSys(NRT_malloc_func malloc_, NRT_realloc_func realloc_,
           NRT_free_func free_)
        : mi_allocator(malloc_, realloc_, free_) {}

    /* Stats */
    size_t stats_alloc, stats_free, stats_mi_alloc, stats_mi_free;
    /* MemInfo allocation functions */
    MemInfoAllocator mi_allocator;
};

typedef struct MemSys NRT_MemSys;

// Pointer to singleton MemSys object, set in bodo_common_init().
// Used for finding memory leaks in unit tests.
// All the C extensions will use the same object.
inline MemSys *global_memsys = nullptr;

inline size_t NRT_MemSys_get_stats_alloc() {
    return global_memsys->stats_alloc;
}

inline size_t NRT_MemSys_get_stats_free() { return global_memsys->stats_free; }

inline size_t NRT_MemSys_get_stats_mi_alloc() {
    return global_memsys->stats_mi_alloc;
}

inline size_t NRT_MemSys_get_stats_mi_free() {
    return global_memsys->stats_mi_free;
}

/**
 * @brief Stores metadata needed to destroy an allocation. See MemInfo
 */
struct DtorInfo {
    void *other;             // Any additional info needed for destructor
    bodo::IBufferPool *pool; /* Store a pointer to the to the BufferPool
                             that was used for allocating the data */
};

/**
 * @brief Stores allocation metadata used by the NRT.
 *
 * WARNING: DO NOT MODIFY STRUCT DEFINITION
 *
 * This is a 1-to-1 recreation of the MemInfo struct in Numba.
 * We currently don't replace all usages for NRT in Numba, so
 * it is possible for a Bodo MemInfo to be deallocated in Numba. If you
 * need to store more data, do so in `dtor_info`.
 *
 */
struct MemInfo {
    int64_t refct;
    NRT_dtor_function dtor;
    // This is a DtorInfo struct under the hood. We use a void* here
    // for Numba compatibility
    void *dtor_info;
    void *data;
    // Numba uses for NRT allocated memory
    // Bodo only uses for memory allocated through the buffer pool
    size_t size;
    // Numba types this as NRT_ExternalAllocator*. We never use so it should be
    // fine
    void *external_allocator;
};

typedef struct MemInfo NRT_MemInfo;

/* ------- Wrappers around MemSys.mi_allocator functions ------- */

/**
 * @brief Allocate a MemInfo struct. This goes through
 * the MemInfoAllocator (i.e. malloc).
 *
 */
inline NRT_MemInfo *NRT_MemInfo_allocate() {
    void *ptr = global_memsys->mi_allocator.malloc(sizeof(NRT_MemInfo));
    if (!ptr) {
        throw std::runtime_error("bad mi alloc: possible Out of Memory error");
    }

    // Custom to Bodo: Allocate the DtorInfo struct
    void *dtor_info_ptr = global_memsys->mi_allocator.malloc(sizeof(DtorInfo));
    if (!dtor_info_ptr) {
        throw std::runtime_error("bad mi alloc: possible Out of Memory error");
    }

    NRT_MemInfo *mi = (NRT_MemInfo *)ptr;
    mi->dtor_info = dtor_info_ptr;

    /* Update stats */
    global_memsys->stats_alloc++;
    global_memsys->stats_mi_alloc++;
    return mi;
}

/**
 * @brief Destroy a MemInfo struct. This only frees the MemInfo struct, and not
 * the underlying data (since they are allocated separately).
 *
 * This is only called from C++ deallocation code. Python allocations go through
 * Numba's version. Thus, we can't deallocate DtorInfo in this function.
 * The destructors are responsible for cleaning up the info.
 *
 *
 * @param mi MemInfo struct to free.
 */
inline void NRT_MemInfo_destroy(NRT_MemInfo *mi) {
    global_memsys->mi_allocator.free(static_cast<void *>(mi));
    /* Update stats */
    global_memsys->stats_free++;
    global_memsys->stats_mi_free++;
}

/* ------------------------------------------------------------- */

/**
 * @brief Destructor for the Meminfo struct which will also free the underlying
 * data through the dtor in the meminfo.
 *
 * NOTE: This function is to be called only from C++.
 *       For Python the NUMBA decref functions are called.
 *       Assert statements are made for controlling that the reference count is
 *       0.
 *
 *
 * @param mi MemInfo to destroy.
 */
inline void NRT_MemInfo_call_dtor(NRT_MemInfo *mi) {
    assert(mi->refct == 0);  // The reference count should be exactly 0
    if (mi->dtor) {
        // If we have a custom destructor (which we always should, and it should
        // always be nrt_internal_custom_dtor), first call
        // that. This dtor will free the underlying data. Should always be
        // either nrt_internal_custom_dtor or arrow_buffer_dtor.
        mi->dtor(mi->data, mi->size, mi->dtor_info);
    }
    // Clear and release MemInfo struct.
    NRT_MemInfo_destroy(mi);
}

/**
 * @brief Set attributes of a newly created Meminfo struct.
 * Must be called right after a Meminfo struct is created.
 *
 */
inline void NRT_MemInfo_init(NRT_MemInfo *mi, void *data, size_t size,
                             NRT_dtor_function dtor, bodo::IBufferPool *pool,
                             void *dtor_other) {
    mi->refct = 1; /* starts with 1 refcount */
    mi->dtor = dtor;

    auto dtor_info = static_cast<DtorInfo *>(mi->dtor_info);
    dtor_info->other = dtor_other;
    dtor_info->pool = pool;

    mi->data = data;
    mi->size = size;
    mi->external_allocator = nullptr;
}

/* ----------- Buffer Pool Data Allocation / Deallocation Helpers ----------- */

/**
 * @brief Allocate 'size' bytes (with 'align' bytes alignment) through the
 * BufferPool for the MemInfo.
 *
 * @param size Bytes to allocated.
 * @param align Byte alignment to use.
 * @param mi MemInfo for which the data is being allocated. This is required
 * since we need to point the "Swip" in the BufferPool to point to the
 * data pointer in the MemInfo (for safe evictions).
 * @param pool IBufferPool to use for the allocation.
 */
inline void buffer_pool_aligned_data_alloc(size_t size, unsigned align,
                                           NRT_MemInfo *mi,
                                           bodo::IBufferPool *const pool) {
    if (align && ((align > ::arrow::kDefaultBufferAlignment) ||
                  ((align & (align - 1)) != 0))) {
        // TODO Add compiler hint to mark this as an unlikely branch.
        throw std::runtime_error(
            std::string("Requested alignment (") + std::to_string(align) +
            std::string(") is either greater than default alignment (") +
            std::to_string(::arrow::kDefaultBufferAlignment) +
            std::string(") or not a power of 2."));
    }
    // Pass a pointer to the data pointer in the MemInfo to be used as
    // the Swip by the BufferPool.
    // We use the default alignment (64B) for all MemInfo data allocations
    // since we don't provide the alignment information during Free/Realloc.
    // Using the default value at Free/Realloc time and using another
    // value at Alloc time can lead to issues since the BufferPool might
    // calculate the sizes incorrectly and search in the incorrect SizeClass.
    auto status =
        pool->Allocate(size, reinterpret_cast<uint8_t **>(&(mi->data)));

    CHECK_ARROW_MEM(status, "Allocation failed!");
    // Update stats for memory_leak_check.
    global_memsys->stats_alloc++;
}

/**
 * @brief Free memory allocated for a MemInfo through the BufferPool.
 *
 * @param ptr Pointer to the allocated memory region.
 * @param size Number of bytes originally allocated for the data. This is the
 * size stored in the MemInfo struct. The actual allocation (from a BufferPool
 * perspective) might have been larger, but the BufferPool will handle it
 * correctly.
 * @param pool IBufferPool to deallocate through.
 */
inline void buffer_pool_aligned_data_free(void *ptr, size_t size,
                                          bodo::IBufferPool *const pool) {
    if (pool == nullptr) {
        return;
    }
    pool->Free(static_cast<uint8_t *>(ptr), size);
    global_memsys->stats_free++;
}

/* -------------------------------------------------------------------------- */

/// @brief Destructor for Meminfo (and underlying data) allocated
/// through NRT_MemInfo_alloc_common.
inline void nrt_internal_custom_dtor(void *ptr, size_t size,
                                     void *dtor_info_raw) {
    DtorInfo *dtor_info = (DtorInfo *)dtor_info_raw;

    // Free the data buffer
    buffer_pool_aligned_data_free(ptr, size, dtor_info->pool);

    // Free the DtorInfo struct
    global_memsys->mi_allocator.free(dtor_info_raw);
}

/**
 * @brief Allocate a MemInfo with a data buffer of 'size' bytes.
 * The allocated memory will be aligned as per 'align'. A custom
 * constructor can be provided. This will be used in
 * nrt_internal_custom_dtor when destroying the MemInfo.
 *
 * @param size Number of bytes to allocate
 * @param align Byte-alignment for the allocated data. Set to 0
 *  in case there are no alignment restrictions.
 * @param pool IBufferPool to allocate underlying data buffer through.
 * @return NRT_MemInfo* Pointer to fully initialized MemInfo struct.
 */
inline NRT_MemInfo *NRT_MemInfo_alloc_common(size_t size, unsigned align,
                                             bodo::IBufferPool *const pool) {
    // Allocate the MemInfo object using standard allocator.
    NRT_MemInfo *mi = NRT_MemInfo_allocate();

    // Allocate the data buffer using the bodo::IBufferPool and assign it to
    // mi->data. This will also store a pointer to mi->data in the
    // bodo::IBufferPool for eviction purposes later.
    buffer_pool_aligned_data_alloc(size, align, mi, pool);

    // Initialize the MemInfo object. We assign our custom
    // destructor (nrt_internal_custom_dtor) which will be used
    // to destroy the data buffer when destroying the MemInfo struct
    // (e.g. see NRT_MemInfo_call_dtor). We store a pointer to the
    // pool in 'dtor_info', which nrt_internal_custom_dtor will then
    // use during deallocation.
    NRT_MemInfo_init(mi, mi->data, size, nrt_internal_custom_dtor, pool,
                     nullptr);
    return mi;
}

/* ---------- MemInfo + Data Allocation functions ----------- */

/// @brief Allocate MemInfo and data without a custom pool and without any
/// alignment constraints).
inline NRT_MemInfo *NRT_MemInfo_alloc_safe(size_t size) {
    return NRT_MemInfo_alloc_common(size, 0, bodo::BufferPool::DefaultPtr());
}

/// @brief Allocate MemInfo and data without a custom pool, aligned as per
/// 'align'.
inline NRT_MemInfo *NRT_MemInfo_alloc_safe_aligned(size_t size,
                                                   unsigned align) {
    return NRT_MemInfo_alloc_common(size, align,
                                    bodo::BufferPool::DefaultPtr());
}

/// @brief Allocate MemInfo and data through a custom pool, aligned
/// as per 'align'.
inline NRT_MemInfo *NRT_MemInfo_alloc_safe_aligned_pool(
    size_t size, unsigned align, bodo::IBufferPool *const pool) {
    return NRT_MemInfo_alloc_common(size, align, pool);
}

/* ---------------------------------------------------------- */

inline void NRT_MemInfo_Pin(NRT_MemInfo *mi) {
    auto dtor_info = static_cast<DtorInfo *>(mi->dtor_info);
    if (dtor_info->pool == nullptr) {
        return;
    }

    auto status = dtor_info->pool->Pin(
        reinterpret_cast<uint8_t **>(&(mi->data)), mi->size, ALIGNMENT);
    CHECK_ARROW_MEM(status, "Failed to Pin MemInfo Object");
}

inline void NRT_MemInfo_Unpin(NRT_MemInfo *mi) {
    auto dtor_info = static_cast<DtorInfo *>(mi->dtor_info);
    if (dtor_info->pool == nullptr) {
        return;
    }

    dtor_info->pool->Unpin(reinterpret_cast<uint8_t *>(mi->data), mi->size,
                           ALIGNMENT);
}
