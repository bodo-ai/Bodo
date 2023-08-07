// Copyright (C) 2023 Bodo Inc. All rights reserved.
#ifndef BODO_MEMINFO_INCLUDED
#define BODO_MEMINFO_INCLUDED

#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <iostream>
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

    /// @brief Get pointer to singleton MemSys object.
    /// Used for finding memory leaks in unit tests.
    /// All the C extensions will use the same object.
    /// Ref:
    /// https://betterprogramming.pub/3-tips-for-using-singletons-in-c-c6822dc42649
    static MemSys *instance() {
        static MemSys base(malloc, realloc, free);
        return &base;
    }
};

typedef struct MemSys NRT_MemSys;

inline size_t NRT_MemSys_get_stats_alloc() {
    return NRT_MemSys::instance()->stats_alloc;
}

inline size_t NRT_MemSys_get_stats_free() {
    return NRT_MemSys::instance()->stats_free;
}

inline size_t NRT_MemSys_get_stats_mi_alloc() {
    return NRT_MemSys::instance()->stats_mi_alloc;
}

inline size_t NRT_MemSys_get_stats_mi_free() {
    return NRT_MemSys::instance()->stats_mi_free;
}

struct MemInfo {
    int64_t refct;
    NRT_dtor_function dtor;
    void *dtor_info; /* Used for storing pointer to the BufferPool that was used
                        for allocating the data */
    void *data;
    size_t size; /* only used for memory allocated through the buffer pool */
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
    void *ptr =
        NRT_MemSys::instance()->mi_allocator.malloc(sizeof(NRT_MemInfo));
    if (!ptr) {
        throw std::runtime_error("bad mi alloc: possible Out of Memory error");
    }
    /* Update stats */
    NRT_MemSys::instance()->stats_alloc++;
    NRT_MemSys::instance()->stats_mi_alloc++;
    return (NRT_MemInfo *)ptr;
}

/**
 * @brief Destroy a MemInfo struct. This only frees the MemInfo struct, and not
 * the underlying data (since they are allocated separately).
 *
 * @param mi MemInfo struct to free.
 */
inline void NRT_MemInfo_destroy(NRT_MemInfo *mi) {
    NRT_MemSys::instance()->mi_allocator.free(static_cast<void *>(mi));
    /* Update stats */
    NRT_MemSys::instance()->stats_free++;
    NRT_MemSys::instance()->stats_mi_free++;
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
        // always be nrt_internal_custom_dtor), first call that. This dtor will
        // free the underlying data.
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
                             NRT_dtor_function dtor, void *dtor_info,
                             void *external_allocator) {
    mi->refct = 1; /* starts with 1 refct */
    mi->dtor = dtor;
    mi->dtor_info = dtor_info;
    mi->data = data;
    mi->size = size;
    mi->external_allocator = external_allocator;
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
    CHECK_ARROW_MEM(
        pool->Allocate(size, reinterpret_cast<uint8_t **>(&(mi->data))),
        "Allocation failed!");
    // Update stats for memory_leak_check.
    NRT_MemSys::instance()->stats_alloc++;
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
    pool->Free(static_cast<uint8_t *>(ptr), size);
    NRT_MemSys::instance()->stats_free++;
}

/* -------------------------------------------------------------------------- */

/// @brief Destructor for Meminfo (and underlying data) allocated
/// through NRT_MemInfo_alloc_common.
inline void nrt_internal_custom_dtor(void *ptr, size_t size, void *pool) {
    // Free the data buffer
    buffer_pool_aligned_data_free(ptr, size, (bodo::IBufferPool *)pool);
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

    // Allocate the data buffer using the bodo::BufferPool and assign it to
    // mi->data. This will also store a pointer to mi->data in the
    // bodo::BufferPool for eviction purposes later.
    buffer_pool_aligned_data_alloc(size, align, mi, pool);

    // Initialize the MemInfo object. We assign our custom
    // destructor (nrt_internal_custom_dtor) which will be used
    // to destroy the data buffer when destroying the MemInfo struct
    // (e.g. see NRT_MemInfo_call_dtor). We store a pointer to the
    // pool in 'dtor_info', which nrt_internal_custom_dtor will then
    // use during deallocation.
    NRT_MemInfo_init(mi, mi->data, size, nrt_internal_custom_dtor, (void *)pool,
                     NULL);
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
    auto pool = reinterpret_cast<bodo::IBufferPool *>(mi->dtor_info);
    auto status = pool->Pin(reinterpret_cast<uint8_t **>(&(mi->data)), mi->size,
                            ALIGNMENT);
    CHECK_ARROW_MEM(status, "Failed to Pin MemInfo Object");
}

inline void NRT_MemInfo_Unpin(NRT_MemInfo *mi) {
    auto pool = reinterpret_cast<bodo::IBufferPool *>(mi->dtor_info);
    pool->Unpin(reinterpret_cast<uint8_t *>(mi->data), mi->size, ALIGNMENT);
}

#endif  // #ifndef BODO_MEMINFO_INCLUDED
