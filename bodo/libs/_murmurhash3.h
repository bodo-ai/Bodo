// Copyright (C) 2019 Bodo Inc. All rights reserved.
//-----------------------------------------------------------------------------
// based on MurmurHash3 written by Austin Appleby
// https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp

#ifndef _MURMURHASH3_H_
#define _MURMURHASH3_H_

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

void hash_string_32(const char* str, const int len, uint32_t* out_hash);

template <class T>
void hash_inner_32(T* data, uint32_t* out_hash);

void MurmurHash3_x64_32(const void* key, int len, uint32_t seed, void* out);

//-----------------------------------------------------------------------------

#endif  // _MURMURHASH3_H_
