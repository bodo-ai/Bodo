// Copyright (C) 2019 Bodo Inc. All rights reserved.
#ifndef BODO_MEMINFO_INCLUDED
#define BODO_MEMINFO_INCLUDED

// #include "_import_py.h"
// #include <numba/runtime/nrt.h>

// /* Import MemInfo_* from numba.runtime._nrt_python.
//  */
// static void *
// import_meminfo_func(const char * func) {
// #define CHECK(expr, msg) if(!(expr)){std::cerr << msg << std::endl;
// PyGILState_Release(gilstate); return NULL;}
//     auto gilstate = PyGILState_Ensure();
//     PyObject * helperdct = import_sym("numba.runtime._nrt_python",
//     "c_helpers");
//     CHECK(helperdct, "getting numba.runtime._nrt_python.c_helpers failed");
//     /* helperdct[func] */
//     PyObject * mi_rel_fn = PyDict_GetItemString(helperdct, func);
//     CHECK(mi_rel_fn, "getting meminfo func failed");
//     void * fnptr = PyLong_AsVoidPtr(mi_rel_fn);

//     Py_XDECREF(helperdct);
//     PyGILState_Release(gilstate);
//     return fnptr;
// #undef CHECK
// }

// typedef void (*MemInfo_release_type)(void*);
// typedef MemInfo* (*MemInfo_alloc_aligned_type)(size_t size, unsigned align);
// typedef void* (*MemInfo_data_type)(MemInfo* mi);

#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <iostream>

extern "C" {

// ******** copied from Numba
// NRT = NumbaRunTime
// Related to managing memory between Numba and C++.
// TODO: make Numba C library

typedef void (*NRT_dtor_function)(void *ptr, size_t size, void *info);
typedef size_t (*NRT_atomic_inc_dec_func)(size_t *ptr);
typedef int (*NRT_atomic_cas_func)(void *volatile *ptr, void *cmp, void *repl,
                                   void **oldptr);

typedef void *(*NRT_malloc_func)(size_t size);
typedef void *(*NRT_realloc_func)(void *ptr, size_t new_size);
typedef void (*NRT_free_func)(void *ptr);
typedef int (*atomic_meminfo_cas_func)(void **ptr, void *cmp, void *repl,
                                       void **oldptr);

struct Allocator {
    Allocator(NRT_malloc_func malloc_, NRT_realloc_func realloc_,
              NRT_free_func free_)
        : malloc(malloc_), realloc(realloc_), free(free_) {}
    NRT_malloc_func malloc;
    NRT_realloc_func realloc;
    NRT_free_func free;
};

struct MemSys {
    MemSys(NRT_malloc_func malloc_, NRT_realloc_func realloc_,
           NRT_free_func free_)
        : allocator(malloc_, realloc_, free_) {}

    /* Atomic increment and decrement function */
    NRT_atomic_inc_dec_func atomic_inc, atomic_dec;
    /* Atomic CAS */
    atomic_meminfo_cas_func atomic_cas;
    /* Shutdown flag */
    int shutting;
    /* Stats */
    size_t stats_alloc, stats_free, stats_mi_alloc, stats_mi_free;
    /* System allocation functions */
    Allocator allocator;

    /// @brief Get pointer to singleton MemSys object.
    /// Used to allow finding memory leaks in unit tests.
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
    void *dtor_info;
    void *data;
    size_t size; /* only used for NRT allocated memory */
    void *external_allocator;
};

typedef struct MemInfo NRT_MemInfo;

#if !defined MIN
#define MIN(a, b) ((a) < (b)) ? (a) : (b)
#endif

/* ------- Wrappers around MemSys.allocator functions ------- */

inline void *NRT_Allocate(size_t size) {
    void *ptr = NRT_MemSys::instance()->allocator.malloc(size);
    if (!ptr) {
        std::cerr << "bad alloc: possible Out of Memory error\n";
        exit(9);
    }

    NRT_MemSys::instance()->stats_alloc++;
    return ptr;
}

inline void NRT_Free(void *ptr) {
    NRT_MemSys::instance()->allocator.free(ptr);
    NRT_MemSys::instance()->stats_free++;
}

/* ----------------------------------------------------------- */

/**
 * @brief This only frees the MemInfo struct, and not the underlying
 * data (since they are allocated separately).
 *
 * @param mi MemInfo struct to free.
 */
inline void NRT_MemInfo_destroy(NRT_MemInfo *mi) {
    NRT_Free(mi);
    NRT_MemSys::instance()->stats_mi_free++;
}

/**
 * @brief Destructor for the Meminfo struct, which will also free the underlying
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
    if (mi->dtor && !NRT_MemSys::instance()->shutting) {
        // We have a destructor and the system is not shutting down.
        // This dtor will free the underlying data.
        mi->dtor(mi->data, mi->size, mi->dtor_info);
    }
    // Clear and release MemInfo struct.
    NRT_MemInfo_destroy(mi);
}

/**
 * @brief Main function where memory for the Meminfo struct and the
 * underlying data is allocated.
 * We make separate allocations for the MemInfo struct itself and the underlying
 * data.
 * In the future, the data allocation will go through our buffer pool.
 *
 * @param size Size of data to allocate.
 * @param [out] mi_out Allocated MemInfo struct.
 * @return void* Pointer to allocated data buffer.
 */
inline void *nrt_allocate_meminfo_and_data(size_t size, NRT_MemInfo **mi_out) {
    NRT_MemInfo *mi;
    char *data = (char *)NRT_Allocate(size);
    mi = (NRT_MemInfo *)NRT_Allocate(sizeof(NRT_MemInfo));
    *mi_out = mi;
    return data;
}

/**
 * @brief Wrapper around nrt_allocate_meminfo_and_data (previous function) which
 * additionally ensures that the allocated underlying data is aligned.
 * It returns a pointer to the "aligned" address. Address to the original
 * allocation address is returned through the base_data_ptr parameter for
 * use during deallocation.
 *
 * @param size Allocation size.
 * @param align Alignment for the data buffer.
 * @param [out] mi Allocated MemInfo struct.
 * @param [out] base_data_ptr Address to the data allocation that can be used
 * with 'free' during de-allocation.
 * @return void* Aligned data pointer.
 */
inline void *nrt_allocate_meminfo_and_data_align(size_t size, unsigned align,
                                                 NRT_MemInfo **mi,
                                                 void **base_data_ptr) {
    size_t offset, intptr, remainder;
    size_t capacity = size + 2 * align;
    char *base = (char *)nrt_allocate_meminfo_and_data(capacity, mi);
    intptr = (size_t)base;
    /* See if we are aligned */
    remainder = intptr % align;
    if (remainder == 0) { /* Yes */
        offset = 0;
    } else { /* No, move forward `offset` bytes */
        offset = align - remainder;
    }

    // Zero-pad to match Arrow
    // https://github.com/apache/arrow/blob/5b2fbade23eda9bc95b1e3854b19efff177cd0bd/cpp/src/arrow/memory_pool.cc#L932
    // https://github.com/apache/arrow/blob/5b2fbade23eda9bc95b1e3854b19efff177cd0bd/cpp/src/arrow/buffer.h#L125
    memset(base + offset + size, 0, capacity - size - offset);

    *base_data_ptr = base;
    return base + offset;
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
    /* Update stats */
    NRT_MemSys::instance()->stats_mi_alloc++;
}

/* ------ MemInfo + Data Allocation CodePath 1 ------ */

/// @brief Destructor for Meminfo (and underlying data) allocated
/// through  NRT_MemInfo_alloc_dtor_safe.
inline void nrt_internal_custom_safe_dtor(void *ptr, size_t size,
                                          void *custom_dtor) {
    NRT_dtor_function dtor = (NRT_dtor_function)custom_dtor;

    if (dtor) {
        dtor(ptr, size, NULL);
    }

    // In this case, ptr points to the underlying data.
    // Since there was no alignment offset, we can call 'free'
    // directly on this location.
    memset(ptr, 0xDE, MIN(size, 256));  // XXX Get rid of this?
    NRT_Free(ptr);
}

/// @brief Allocate MemInfo and data (no alignment constraints)
/// with a custom destructor.
inline NRT_MemInfo *NRT_MemInfo_alloc_dtor_safe(size_t size,
                                                NRT_dtor_function dtor) {
    NRT_MemInfo *mi;

    // Since we are not re-aligning the data, we don't need to store
    // additional information about the base data pointer (unlike
    // NRT_MemInfo_alloc_safe_aligned). In the destructor, we will
    // call 'free' on the data pointer directly.
    void *data = nrt_allocate_meminfo_and_data(size, &mi);
    /* Only fill up a couple cachelines with debug markers, to minimize
       overhead. */
    memset(data, 0xCB, MIN(size, 256));
    NRT_MemInfo_init(mi, data, size, nrt_internal_custom_safe_dtor,
                     (void *)dtor, NULL);
    return mi;
}

/// @brief Wrapper around NRT_MemInfo_alloc_dtor_safe when no
/// custom destructor is required.
inline NRT_MemInfo *NRT_MemInfo_alloc_safe(size_t size) {
    return NRT_MemInfo_alloc_dtor_safe(size, NULL);
}

/* -------------------------------------------------- */

/* ------ MemInfo + Data Allocation CodePath 2 ------ */

/// @brief Destructor for MemInfo (and underlying data) allocated through
/// the NRT_MemInfo_alloc_safe_aligned function.
inline void nrt_internal_safe_aligned_dtor(void *ptr, size_t size,
                                           void *dtor_info) {
    /* See NRT_MemInfo_alloc_safe() */
    memset(ptr, 0xDE, MIN(size, 256));  // XXX Get rid of this?

    // Free the underlying data.
    void *base_data_ptr = dtor_info;
    NRT_Free(base_data_ptr);
}

/// @brief Allocate MemInfo and data (aligned as per 'align').
inline NRT_MemInfo *NRT_MemInfo_alloc_safe_aligned(size_t size,
                                                   unsigned align) {
    NRT_MemInfo *mi;
    // nrt_allocate_meminfo_and_data_align will return a data pointer
    // with some offset (for alignment purposes). However, we need
    // to retain a pointer to the original buffer since that's
    // what we'll need to eventually call "free" on.
    void *base_data_ptr;
    void *data =
        nrt_allocate_meminfo_and_data_align(size, align, &mi, &base_data_ptr);
    /* Only fill up a couple cachelines with debug markers, to minimize
       overhead. */
    memset(data, 0xCB, MIN(size, 256));
    NRT_MemInfo_init(mi, data, size, nrt_internal_safe_aligned_dtor,
                     base_data_ptr, NULL);
    return mi;
}

/* -------------------------------------------------- */

}  // extern "C"

#endif  // #ifndef BODO_MEMINFO_INCLUDED
