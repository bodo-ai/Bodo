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
};

typedef struct MemSys NRT_MemSys;

/* The Memory System object */
extern NRT_MemSys TheMSys;

inline static size_t nrt_testing_atomic_inc(size_t *ptr) {
    /* non atomic */
    size_t out = *ptr;
    out += 1;
    *ptr = out;
    return out;
}

inline static size_t nrt_testing_atomic_dec(size_t *ptr) {
    /* non atomic */
    size_t out = *ptr;
    out -= 1;
    *ptr = out;
    return out;
}

inline static int nrt_testing_atomic_cas(void *volatile *ptr, void *cmp,
                                         void *val, void **oldptr) {
    /* non atomic */
    void *old = *ptr;
    *oldptr = old;
    if (old == cmp) {
        *ptr = val;
        return 1;
    }
    return 0;
}

inline void NRT_MemSys_set_atomic_inc_dec(NRT_atomic_inc_dec_func inc,
                                          NRT_atomic_inc_dec_func dec) {
    TheMSys.atomic_inc = inc;
    TheMSys.atomic_dec = dec;
}

inline void NRT_MemSys_set_atomic_inc_dec_stub(void) {
    NRT_MemSys_set_atomic_inc_dec(nrt_testing_atomic_inc,
                                  nrt_testing_atomic_dec);
}

inline void NRT_MemSys_set_atomic_cas(NRT_atomic_cas_func cas) {
    TheMSys.atomic_cas = (atomic_meminfo_cas_func)cas;
}

inline void NRT_MemSys_set_atomic_cas_stub(void) {
    NRT_MemSys_set_atomic_cas(nrt_testing_atomic_cas);
}

inline size_t NRT_MemSys_get_stats_alloc() { return TheMSys.stats_alloc; }

inline size_t NRT_MemSys_get_stats_free() { return TheMSys.stats_free; }

inline size_t NRT_MemSys_get_stats_mi_alloc() { return TheMSys.stats_mi_alloc; }

inline size_t NRT_MemSys_get_stats_mi_free() { return TheMSys.stats_mi_free; }

struct MemInfo {
    int64_t refct;
    NRT_dtor_function dtor;
    void *dtor_info;
    void *data;
    size_t size; /* only used for NRT allocated memory */
    void *external_allocator;
};

typedef struct MemInfo NRT_MemInfo;

inline void nrt_debug_print(char *fmt, ...) {
    va_list args;

    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
}

#if !defined MIN
#define MIN(a, b) ((a) < (b)) ? (a) : (b)
#endif

inline void NRT_Free(void *ptr) {
    TheMSys.allocator.free(ptr);
    TheMSys.stats_free++;
}

inline void NRT_MemInfo_destroy(NRT_MemInfo *mi) {
    NRT_Free(mi);
    TheMSys.stats_mi_free++;
}

/* This function is to be called only from C++.
   For Python the NUMBA decref functions are called.
   Assert statements are made for controlling that the reference count is 0.
 */
inline void NRT_MemInfo_call_dtor(NRT_MemInfo *mi) {
    assert(mi->refct == 0);  // The reference count should be exactly 0
    if (mi->dtor && !TheMSys.shutting)
        /* We have a destructor and the system is not shutting down */
        mi->dtor(mi->data, mi->size, mi->dtor_info);
    /* Clear and release MemInfo */
    NRT_MemInfo_destroy(mi);
}

inline void *NRT_Allocate(size_t size) {
    void *ptr = TheMSys.allocator.malloc(size);
    if (!ptr) {
        std::cerr << "bad alloc: possible Out of Memory error\n";
        exit(9);
    }

    TheMSys.stats_alloc++;
    return ptr;
}

inline void *nrt_allocate_meminfo_and_data(size_t size, NRT_MemInfo **mi_out) {
    NRT_MemInfo *mi;
    char *base = (char *)NRT_Allocate(sizeof(NRT_MemInfo) + size);
    mi = (NRT_MemInfo *)base;
    *mi_out = mi;
    return base + sizeof(NRT_MemInfo);
}

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
    TheMSys.stats_mi_alloc++;
}

inline void nrt_internal_dtor_safe(void *ptr, size_t size, void *info) {
    /* See NRT_MemInfo_alloc_safe() */
    memset(ptr, 0xDE, MIN(size, 256));
}

inline void nrt_internal_custom_dtor_safe(void *ptr, size_t size, void *info) {
    NRT_dtor_function dtor = (NRT_dtor_function)info;

    if (dtor) {
        dtor(ptr, size, NULL);
    }

    nrt_internal_dtor_safe(ptr, size, NULL);
}

inline NRT_MemInfo *NRT_MemInfo_alloc_dtor_safe(size_t size,
                                                NRT_dtor_function dtor) {
    NRT_MemInfo *mi;
    void *data = nrt_allocate_meminfo_and_data(size, &mi);
    /* Only fill up a couple cachelines with debug markers, to minimize
       overhead. */
    memset(data, 0xCB, MIN(size, 256));
    NRT_MemInfo_init(mi, data, size, nrt_internal_custom_dtor_safe,
                     (void *)dtor, NULL);
    return mi;
}

inline NRT_MemInfo *NRT_MemInfo_alloc_safe(size_t size) {
    return NRT_MemInfo_alloc_dtor_safe(size, NULL);
}

inline void *nrt_allocate_meminfo_and_data_align(size_t size, unsigned align,
                                                 NRT_MemInfo **mi) {
    size_t offset, intptr, remainder;
    char *base = (char *)nrt_allocate_meminfo_and_data(size + 2 * align, mi);
    intptr = (size_t)base;
    /* See if we are aligned */
    remainder = intptr % align;
    if (remainder == 0) { /* Yes */
        offset = 0;
    } else { /* No, move forward `offset` bytes */
        offset = align - remainder;
    }
    return base + offset;
}

inline NRT_MemInfo *NRT_MemInfo_alloc_aligned(size_t size, unsigned align) {
    NRT_MemInfo *mi;
    void *data = nrt_allocate_meminfo_and_data_align(size, align, &mi);
    NRT_MemInfo_init(mi, data, size, NULL, NULL, NULL);
    return mi;
}

inline NRT_MemInfo *NRT_MemInfo_alloc_safe_aligned(size_t size,
                                                   unsigned align) {
    NRT_MemInfo *mi;
    void *data = nrt_allocate_meminfo_and_data_align(size, align, &mi);
    /* Only fill up a couple cachelines with debug markers, to minimize
       overhead. */
    memset(data, 0xCB, MIN(size, 256));
    NRT_MemInfo_init(mi, data, size, nrt_internal_dtor_safe, (void *)size,
                     NULL);
    return mi;
}

}  // extern "C"

#endif  // #ifndef BODO_MEMINFO_INCLUDED
