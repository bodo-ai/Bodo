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

#else	// defined(_MSC_VER)

#include <stdint.h>

#endif // !defined(_MSC_VER)

//-----------------------------------------------------------------------------

void MurmurHash3_x64_32 ( const void * key, int len, uint32_t seed, void * out );

<<<<<<< HEAD
<<<<<<< HEAD
void MurmurHash3_x64_128 ( const void * key, int len, uint32_t seed, void * out );

=======
>>>>>>> adds murmurhash3_x62_32 and hash_string
=======
void MurmurHash3_x64_128 ( const void * key, int len, uint32_t seed, void * out );

>>>>>>> adds hash string array
//-----------------------------------------------------------------------------

#endif // _MURMURHASH3_H_
