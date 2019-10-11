#include "_murmurhash3.h"
//#include <numpy/arrayobject.h>
#include <iostream> //TODO: take this out 
#include <bitset> //TODO: take this out 

void hash_string(const char* str, const int len, const uint32_t seed, uint32_t* out_hash){
	MurmurHash3_x64_32 ((const void*)str, len, seed, (void*)out_hash);
}

int main()
{
  const char* str = "what";
  const int len = strlen(str);
  const uint32_t seed = 0x1234abcd;
  uint32_t* out_hash = (uint32_t*)std::malloc(sizeof(uint32_t));
  hash_string(str, len, seed, out_hash);
  std::bitset<32> y(*out_hash);
  std::cout << "hash is " << y << '\n';
}
