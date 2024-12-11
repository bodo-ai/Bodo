#pragma once

#include "_memory.h"

// Choose default implementation for unordered map and set
#undef USE_STD
#define USE_ANKERL

#ifdef USE_STD
#include <unordered_map>
#include <unordered_set>
#define UNORD_MAP_CONTAINER std::unordered_map
#define UNORD_SET_CONTAINER std::unordered_set
#define UNORD_HASH std::hash
#endif

#ifdef USE_ANKERL
#define UNORDERED_MAP_MAX_LOAD_FACTOR 0.8
#include "vendored/ankerl/unordered_dense.h"
#define UNORD_MAP_CONTAINER ankerl::unordered_dense::map
#define UNORD_SET_CONTAINER ankerl::unordered_dense::set
#define UNORD_HASH ankerl::unordered_dense::hash
#endif

namespace bodo {

template <typename Key, typename T, typename Hash = UNORD_HASH<Key>,
          class KeyEqual = std::equal_to<Key>,
          class Allocator = bodo::STLBufferPoolAllocator<std::pair<Key, T>>>
using unord_map_container =
    UNORD_MAP_CONTAINER<Key, T, Hash, KeyEqual, Allocator>;

template <typename Key, typename Hash = UNORD_HASH<Key>,
          class KeyEqual = std::equal_to<Key>>
using unord_set_container =
    UNORD_SET_CONTAINER<Key, Hash, KeyEqual, bodo::STLBufferPoolAllocator<Key>>;

}  // namespace bodo
