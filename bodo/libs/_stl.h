// Copyright (C) 2023 Bodo Inc. All rights reserved.
#pragma once

#include "_memory.h"

// Choose default implementation for unordered map and set
#undef USE_STD
#undef USE_TSL_ROBIN
#undef USE_TSL_SPARSE
#undef USE_TSL_HOPSCOTCH
#define USE_ANKERL
#undef USE_ROBIN_HOOD_FLAT
#undef USE_ROBIN_HOOD_NODE

#ifdef USE_STD
#include <unordered_map>
#include <unordered_set>
#define UNORD_MAP_CONTAINER std::unordered_map
#define UNORD_SET_CONTAINER std::unordered_set
#define UNORD_HASH std::hash
#endif

#ifdef USE_TSL_ROBIN
// The robin_map can store hashes internally, which helps improve performance.
// To enable this, search for UNORD_MAP_CONTAINER and add a `true` template
// parameter and allocation. E.g.
//
// UNORD_MAP_CONTAINER<size_t, size_t, HashHashJoinTable,
//                     KeyEqualHashJoinTable,
//                     std::allocator<std::pair<size_t, size_t>>,
//                     true>  // StoreHash
#define UNORDERED_MAP_MAX_LOAD_FACTOR 0.5
#include <include/tsl/robin_set.h>
#define UNORD_MAP_CONTAINER tsl::robin_map
#define UNORD_SET_CONTAINER tsl::robin_set
#define UNORD_HASH std::hash
#endif

#ifdef USE_TSL_SPARSE
#define UNORDERED_MAP_MAX_LOAD_FACTOR 0.9
#include <include/tsl/sparse_map.h>
#include <include/tsl/sparse_set.h>
#define UNORD_MAP_CONTAINER tsl::sparse_map
#define UNORD_SET_CONTAINER tsl::sparse_set
#define UNORD_HASH std::hash
#endif

#ifdef USE_TSL_HOPSCOTCH
#include <include/tsl/hopscotch_set.h>
#define UNORD_MAP_CONTAINER tsl::hopscotch_map
#define UNORD_SET_CONTAINER tsl::hopscotch_set
#define UNORD_HASH std::hash
#endif

#ifdef USE_ANKERL
#define UNORDERED_MAP_MAX_LOAD_FACTOR 0.8
#include "vendored/ankerl/unordered_dense.h"
#define UNORD_MAP_CONTAINER ankerl::unordered_dense::map
#define UNORD_SET_CONTAINER ankerl::unordered_dense::set
#define UNORD_HASH ankerl::unordered_dense::hash
#endif

#ifdef USE_ROBIN_HOOD_FLAT
#define UNORDERED_MAP_MAX_LOAD_FACTOR 0.8
#include "vendored/robin_hood.h"
#define UNORD_MAP_CONTAINER robin_hood::unordered_flat_map
#define UNORD_SET_CONTAINER robin_hood::unordered_flat_set
#define UNORD_HASH robin_hood::hash
#endif

#ifdef USE_ROBIN_HOOD_NODE
#define UNORDERED_MAP_MAX_LOAD_FACTOR 0.8
#include "vendored/robin_hood.h"
#define UNORD_MAP_CONTAINER robin_hood::unordered_node_map
#define UNORD_SET_CONTAINER robin_hood::unordered_node_set
#define UNORD_HASH robin_hood::hash
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
