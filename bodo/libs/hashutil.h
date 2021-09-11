// Source: https://github.com/FastFilter/fastfilter_cpp/commit/4264fbb307b2d6a5a3ada255d6f53978beb90dc9
// Bodo change: every rank has to use the same seed when building the same filter,
// so we use a fixed seed for now (instead of random seed)
#ifndef HASHUTIL_H_
#define HASHUTIL_H_

#include <stdint.h>
#include <stdlib.h>
#include <sys/types.h>

#include <string>

//#include <random>

namespace hashing {

class SimpleMixSplit {

 public:
  uint64_t seed;
  SimpleMixSplit() {
    //::std::random_device random;
    //seed = random();
    //seed <<= 32;
    //seed |= random();
    seed = 739;
  }

  inline static uint64_t murmur64(uint64_t h) {
    h ^= h >> 33;
    h *= UINT64_C(0xff51afd7ed558ccd);
    h ^= h >> 33;
    h *= UINT64_C(0xc4ceb9fe1a85ec53);
    h ^= h >> 33;
    return h;
  }

  inline uint64_t operator()(uint64_t key) const {
    return murmur64(key + seed);
  }
};

}

#endif  // HASHUTIL_H_
