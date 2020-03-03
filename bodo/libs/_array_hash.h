// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include "_bodo_common.h"

#define SEED_HASH_PARTITION 0xb0d01289

uint32_t* hash_keys(std::vector<array_info*> const& key_arrs,
                    const uint32_t seed);

void hash_array(uint32_t* out_hashes, array_info* array, size_t n_rows,
                const uint32_t seed);
