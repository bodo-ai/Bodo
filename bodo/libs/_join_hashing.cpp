#include "_join_hashing.h"
#include "_array_hash.h"

uint32_t* hash_data_cols_table(const std::vector<array_info*>& in_table,
                               uint64_t* col_nums, size_t n_cols, uint32_t seed,
                               bool is_parallel) {
    if (n_cols > 0) {
        std::vector<array_info*> key_arrs(n_cols);
        for (size_t i = 0; i < n_cols; i++) {
            size_t col_num = col_nums[i];
            key_arrs[i] = in_table[col_num];
        }
        return hash_keys(key_arrs, seed, is_parallel);
    } else {
        // If there are no unique columns, everything should hash
        // to a unique group, so return all 0;
        size_t n_rows = in_table[0]->length;
        return new uint32_t[n_rows]();
    }
}
