// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include "_bodo_common.h"

#define SEED_HASH_PARTITION           0xb0d01289
#define SEED_HASH_MULTIKEY            0xb0d01288
#define SEED_HASH_DROPDUPLI           0xb0d01287
#define SEED_HASH_JOIN                0xb0d01286
#define SEED_HASH_GROUPBY_SHUFFLE     0xb0d01285
#define SEED_HASH_CONTAINER           0xb0d01284

uint32_t* hash_keys(std::vector<array_info*> const& key_arrs,
                    const uint32_t seed);

void hash_array(uint32_t* out_hashes, array_info* array, size_t n_rows,
                const uint32_t seed);

inline uint32_t* hash_keys_table(table_info* in_table, size_t num_keys, uint32_t seed)
{
    std::vector<array_info*> key_arrs(in_table->columns.begin(), in_table->columns.begin() + num_keys);
    return hash_keys(key_arrs, seed);
}
