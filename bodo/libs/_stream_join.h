#pragma once
#include "_array_utils.h"
#include "_bodo_common.h"

/**
 * @brief allocate an empty table with provided column types
 *
 * @param arr_c_types vector of ints for column types (in Bodo_CTypes format)
 * @return std::shared_ptr<table_info> allocated table
 */
std::shared_ptr<table_info> alloc_table(std::vector<int8_t> arr_c_types);
struct JoinState;

struct HashHashJoinTable {
    /**
     * provides row hashes for join hash table (std::unordered_multimap)
     *
     * Input row number iRow can refer to either build or probe table.
     * If iRow >= 0 then it is in the build table at index iRow.
     * If iRow < 0 then it is in the probe table
     *    at index (-iRow - 1).
     *
     * @param iRow row number
     * @return hash of row iRow
     */
    uint32_t operator()(const int64_t iRow) const;
    JoinState* join_state;
};

struct KeyEqualHashJoinTable {
    /**
     * provides row comparison for join hash table (std::unordered_multimap)
     *
     * Input row number iRow can refer to either build or probe table.
     * If iRow >= 0 then it is in the build table at index iRow.
     * If iRow < 0 then it is in the probe table
     *    at index (-iRow - 1).
     *
     * @param iRowA is the first row index for the comparison
     * @param iRowB is the second row index for the comparison
     * @return true if equal else false
     */
    bool operator()(const int64_t iRowA, const int64_t iRowB) const;
    JoinState* join_state;
    bool is_na_equal;
};

struct JoinState {
    // TODO[BSE-352]: use proper build buffer
    std::shared_ptr<table_info> build_table_buffer;
    int64_t n_keys;

    // state for hashing and comparison classes
    std::shared_ptr<table_info> probe_table;
    std::vector<uint32_t> build_table_hashes;
    std::shared_ptr<uint32_t[]> probe_table_hashes;

    // join hash table (key row number -> matching row numbers)
    std::unordered_multimap<int64_t, int64_t, HashHashJoinTable,
                            KeyEqualHashJoinTable>
        build_table;

    JoinState(std::vector<int8_t> arr_c_types, int64_t n_keys_)
        : n_keys(n_keys_),
          build_table({}, HashHashJoinTable(this),
                      KeyEqualHashJoinTable(this, false)) {
        build_table_buffer = alloc_table(arr_c_types);
    }
};
