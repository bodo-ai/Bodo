#include "_murmurhash3.h"
//#include <numpy/arrayobject.h>
#include <iostream> //TODO: take this out 
#include <bitset> //TODO: take this out 

#define HASHSEED 0xb0d01289

//hash string
//out hash: uint32_t*, 32 bits
void hash_string_32(const char* str, const int len, uint32_t* out_hash){
	MurmurHash3_x64_32 (str, len, HASHSEED, (void*)out_hash);
}

template <class T>
void hash_inner_32(T* data, uint32_t* out_hash){
    MurmurHash3_x64_32 ((const void *)data, sizeof(T), HASHSEED, (void*)out_hash);
}
