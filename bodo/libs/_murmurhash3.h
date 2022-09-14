// Copyright (C) 2019 Bodo Inc. All rights reserved.
//-----------------------------------------------------------------------------
// based on MurmurHash3 written by Austin Appleby
// https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp

#ifndef _MURMURHASH3_H_
#define _MURMURHASH3_H_

#define USE_MURMUR_32 0  // MurmurHash3_x86_32
#define USE_MURMUR_128 \
    0                   // MurmurHash3_x64_128 getting 32 least significant bits
#define USE_XXH3_LOW 1  // 64-bit hash with xxHash3 and get 32 lsb

#if USE_XXH3_LOW
// xxHash code from https://github.com/Cyan4973/xxHash
// and corresponds to commit
// https://github.com/Cyan4973/xxHash/commit/4a20afceb625f62278e11f630156475aee40b055
#include "xxh3.h"
#endif

//-----------------------------------------------------------------------------
// Platform-specific functions and macros

// Microsoft Visual Studio

#if defined(_MSC_VER) && (_MSC_VER < 1600)

typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef unsigned __int64 uint64_t;

// Other compilers

#else  // defined(_MSC_VER)

#include <stdint.h>

#endif  // !defined(_MSC_VER)

//-----------------------------------------------------------------------------

void MurmurHash3_x86_32(const void* key, int len, uint32_t seed, void* out);

void MurmurHash3_x64_32(const void* key, int len, uint32_t seed, void* out);

// out hash: uint32_t*, 32 bits
inline void hash_string_32(const char* str, const int len, const uint32_t seed,
                           uint32_t* out_hash) {
    if (len == 0)
        *out_hash = 0;
    else
#if USE_MURMUR_32 || USE_MURMUR_128
        MurmurHash3_x64_32(str, len, seed, (void*)out_hash);
#endif
#if USE_XXH3_LOW
    *out_hash =
        static_cast<uint32_t>(XXH3_64bits_withSeed(str, (size_t)len, seed));
#endif
}

template <class T>
inline void hash_inner_32(T* data, const uint32_t seed, uint32_t* out_hash) {
#if USE_MURMUR_32 || USE_MURMUR_128
    MurmurHash3_x64_32((const void*)data, sizeof(T), seed, (void*)out_hash);
#endif
#if USE_XXH3_LOW
    *out_hash = static_cast<uint32_t>(
        XXH3_64bits_withSeed((const void*)data, sizeof(T), seed));
#endif
}

// We need the MurMurHash3_x86_32 implementation for Iceberg.
// (https://iceberg.apache.org/spec/#bucket-transform-details)
// (https://iceberg.apache.org/spec/#appendix-b-32-bit-hash-requirements)

inline void hash_string_murmurhash3_x86_32(const char* str, const int len,
                                           const uint32_t seed,
                                           uint32_t* out_hash) {
    if (len == 0)
        *out_hash = 0;
    else
        MurmurHash3_x86_32(str, len, seed, (void*)out_hash);
}

template <class T>
inline void hash_inner_murmurhash3_x86_32(T* data, const uint32_t seed,
                                          uint32_t* out_hash) {
    MurmurHash3_x86_32((const void*)data, sizeof(T), seed, (void*)out_hash);
}

//-----------------------------------------------------------------------------

#endif  // _MURMURHASH3_H_
