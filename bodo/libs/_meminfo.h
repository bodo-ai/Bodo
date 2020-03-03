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

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <iostream>

// ******** copied from Numba
// TODO: make Numba C library
typedef void (*NRT_dtor_function)(void *ptr, size_t size, void *info);
struct MemInfo {
    size_t refct;
    NRT_dtor_function dtor;
    void *dtor_info;
    void *data;
    size_t size; /* only used for NRT allocated memory */
};

typedef struct MemInfo NRT_MemInfo;

inline void nrt_debug_print(char *fmt, ...) {
    va_list args;

    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
}

#if 0
#define BODO_DEBUG
#else
#undef BODO_DEBUG
#endif

#if !defined MIN
#define MIN(a, b) ((a) < (b)) ? (a) : (b)
#endif

inline void NRT_Free(void *ptr) {
#ifdef BODO_DEBUG
    std::cerr << "NRT_Free " << ptr << "\n";
#endif
    free(ptr);
    // TheMSys.allocator.free(ptr);
    // TheMSys.atomic_inc(&TheMSys.stats_free);
}

inline void NRT_MemInfo_destroy(NRT_MemInfo *mi) {
    NRT_Free(mi);
    // TheMSys.atomic_inc(&TheMSys.stats_mi_free);
}

inline void NRT_MemInfo_call_dtor(NRT_MemInfo *mi) {
#ifdef BODO_DEBUG
    std::cerr << "NRT_MemInfo_call_dtor " << mi << "\n";
#endif
    if (mi->dtor)  // && !TheMSys.shutting)
        /* We have a destructor and the system is not shutting down */
        mi->dtor(mi->data, mi->size, mi->dtor_info);
    /* Clear and release MemInfo */
    NRT_MemInfo_destroy(mi);
}

inline void *NRT_Allocate(size_t size) {
    // void *ptr = TheMSys.allocator.malloc(size);
    void *ptr = malloc(size);
#ifdef BODO_DEBUG
    std::cerr << "NRT_Allocate bytes=" << size << " ptr=" << ptr << "\n";
#endif
    // TheMSys.atomic_inc(&TheMSys.stats_alloc);
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
                             NRT_dtor_function dtor, void *dtor_info) {
    mi->refct = 1; /* starts with 1 refct */
    mi->dtor = dtor;
    mi->dtor_info = dtor_info;
    mi->data = data;
    mi->size = size;
    /* Update stats */
    // TheMSys.atomic_inc(&TheMSys.stats_mi_alloc);
}

inline void nrt_internal_dtor_safe(void *ptr, size_t size, void *info) {
#ifdef BODO_DEBUG
    std::cerr << "nrt_internal_dtor_safe " << ptr << ", " << info << "\n";
#endif
    /* See NRT_MemInfo_alloc_safe() */
    memset(ptr, 0xDE, MIN(size, 256));
}

inline void nrt_internal_custom_dtor_safe(void *ptr, size_t size, void *info) {
    NRT_dtor_function dtor = (NRT_dtor_function)info;
#ifdef BODO_DEBUG
    std::cerr << "nrt_internal_custom_dtor_safe " << ptr << ", " << info
              << "\n";
#endif
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
#ifdef BODO_DEBUG
    std::cerr << "NRT_MemInfo_alloc_dtor_safe " << data << " " << size << "\n";
#endif
    NRT_MemInfo_init(mi, data, size, nrt_internal_custom_dtor_safe,
                     (void *)dtor);
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
#ifdef BODO_DEBUG
    std::cerr << "NRT_MemInfo_alloc_aligned " << data << "\n";
#endif
    NRT_MemInfo_init(mi, data, size, NULL, NULL);
    return mi;
}

inline NRT_MemInfo *NRT_MemInfo_alloc_safe_aligned(size_t size,
                                                   unsigned align) {
    NRT_MemInfo *mi;
    void *data = nrt_allocate_meminfo_and_data_align(size, align, &mi);
    /* Only fill up a couple cachelines with debug markers, to minimize
       overhead. */
    memset(data, 0xCB, MIN(size, 256));
#ifdef BODO_DEBUG
    std::cerr << "NRT_MemInfo_alloc_safe_aligned " << data << " " << size
              << "\n";
#endif
    NRT_MemInfo_init(mi, data, size, nrt_internal_dtor_safe, (void *)size);
    return mi;
}

#endif  // #ifndef BODO_MEMINFO_INCLUDED
