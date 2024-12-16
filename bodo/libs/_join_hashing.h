#pragma once

#include "_array_utils.h"

/** This code tests if two rows in the same "table" are equal given a list
 * of column numbers. The table is defined as a
 * std::vector<std::shared_ptr<array_info>>.
 *
 * @param table: the table
 * @param iRow1: the row of the first key
 * @param iRow2: the row of the second key
 * @param col_nums: the columns considered for the comparison
 * @param n_cols: The number of columns to compare
 * @return True if they are equal and false otherwise.
 *
 */
inline bool TestRowsEqualGivenColumns(
    const std::vector<std::shared_ptr<array_info>>& table, size_t const& iRow1,
    size_t const& iRow2, const uint64_t* col_nums, const uint64_t n_cols) {
    // iteration over the list of columns for the comparison.
    for (uint64_t i = 0; i < n_cols; i++) {
        size_t col_num = col_nums[i];
        bool test =
            TestEqualColumn(table[col_num], iRow1, table[col_num], iRow2, true);
        if (!test)
            return false;
    }
    // If all columns are equal then we are ok and the rows are equals.
    return true;
}

namespace joinHashFcts {
/**
 * Compute hash for `hash_join_table`
 *
 * Don't use std::function to reduce call overhead.
 */
struct HashHashJoinTable {
    /**
     * This is a function for comparing the rows.
     * This is the first lambda used as argument for the unordered map
     * container.
     *
     * A row can refer to either the short or the long table.
     * If iRow < short_table_rows then it is in the short table
     *    at index iRow.
     * If iRow >= short_table_rows then it is in the long table
     *    at index (iRow - short_table_rows).
     *
     * @param iRow is the first row index for the comparison
     * @return true/false depending on the case.
     */
    uint32_t operator()(const size_t iRow) const {
        if (iRow < short_table_rows) {
            return short_table_hashes[iRow];
        } else {
            return long_table_hashes[iRow - short_table_rows];
        }
    }
    size_t short_table_rows;
    std::shared_ptr<uint32_t[]>& short_table_hashes;
    std::shared_ptr<uint32_t[]>& long_table_hashes;
};

/**
 * Key comparison for `hash_join_table`
 *
 * Don't use std::function to reduce call overhead.
 */
struct KeyEqualHashJoinTable {
    /** This is a function for testing equality of rows.
     * This is used as second argument for the unordered map container.
     *
     * A row can refer to either the build (short) or the probe (long) table.
     * If iRow < build_table_rows then it is in the build table
     *    at index iRow.
     * If iRow >= build_table_rows then it is in the probe table
     *    at index (iRow - build_table_rows).
     *
     * NOTE: Trying to minimize the overhead of this function does not
     * show significant speedup in TPC-H. For example, if there is a single
     * key column and the key array dtype is int64, doing this:
     * `return colA->at<int64_t>(iRowA) == colB->at<int64_t>(iRowB);`
     * only shows about 5% speedup in the portion of join that populates
     * ListPairWrite, and it looks like trying to have multiple versions
     * of equal_fct in a performance-correct way would significantly
     * complicate the code. The most expensive part of the code in the
     * computation of matching pairs is the loop over the probe table,
     * but it looks like the bottleneck right now is not the key equals
     * function (equal_fct).
     *
     * @param iRowA is the first row index for the comparison
     * @param iRowB is the second row index for the comparison
     * @return true/false depending on equality or not.
     */
    bool operator()(const size_t iRowA, const size_t iRowB) const {
        const std::shared_ptr<const table_info>& table_A =
            iRowA < build_table_rows ? build_table : probe_table;
        const std::shared_ptr<const table_info>& table_B =
            iRowB < build_table_rows ? build_table : probe_table;

        const size_t jRowA =
            iRowA < build_table_rows ? iRowA : iRowA - build_table_rows;
        const size_t jRowB =
            iRowB < build_table_rows ? iRowB : iRowB - build_table_rows;

        // Determine if NA columns should match. They should always
        // match when populating the hash map with the short table.
        // When comparing the short and long tables this depends on
        // is_na_equal.
        // TODO: Eliminate groups with NA columns with is_na_equal=False
        // from the hashmap.
        bool set_na_equal = is_na_equal || (table_A == table_B);
        bool test =
            TestEqualJoin(table_A, table_B, jRowA, jRowB, n_key, set_na_equal);
        return test;
    }
    size_t build_table_rows;
    size_t n_key;
    const std::shared_ptr<const table_info> build_table;
    const std::shared_ptr<const table_info> probe_table;
    bool is_na_equal;
};

/**
 * Compute hash for the second level hash maps. These are used
 * for non-equality cases.
 *
 * Don't use std::function to reduce call overhead.
 */
struct SecondLevelHashHashJoinTable {
    /**
     * Hash function used with second level groups. These are used to implement
     * non-equality conditions
     */
    uint32_t operator()(const size_t iRow) const {
        return short_nonequal_key_hashes[iRow];
    }
    std::shared_ptr<uint32_t[]> short_nonequal_key_hashes;
};

/**
 * Key comparison for the second level hash maps. These are used
 * for non-equality cases.
 *
 * Don't use std::function to reduce call overhead.
 */
struct SecondLevelKeyEqualHashJoinTable {
    /**
     * Equality function used with second level groups. This tests that columns
     * that aren't key columns but are used in conditions are equal.
     */
    bool operator()(const size_t iRowA, const size_t iRowB) const {
        return TestRowsEqualGivenColumns(build_table->columns, iRowA, iRowB,
                                         build_data_key_cols,
                                         build_data_key_n_cols);
    }
    const std::shared_ptr<const table_info> build_table;
    const uint64_t* build_data_key_cols;
    const uint64_t build_data_key_n_cols;
};
}  // namespace joinHashFcts

/**
 * Function for hashing specific data columns and returning its hashes
 *
 * @param in_table: the input table
 * @param col_nums: the column indices to hash
 * @param n_cols : the number of data cols
 * @param seed: the seed of the computation.
 * @return hash keys
 *
 */
std::unique_ptr<uint32_t[]> hash_data_cols_table(
    const std::vector<std::shared_ptr<array_info>>& in_table,
    uint64_t* col_nums, size_t n_cols, uint32_t seed, bool is_parallel);
