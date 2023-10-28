// Copyright (C) 2023 Bodo Inc. All rights reserved.
#pragma once

#include "_groupby.h"

/**
 * This file declares the functions that are responsible for organizing
 * the rows into groups or providing information about accessing the groups.
 *
 */

/**
 * @brief Given a set of tables with n key columns, this function calculates the
 * row to group mapping for every row based on its key. For every row in the
 * tables, this only does *one* lookup in the hash map.
 *
 * @param[in] tables the tables
 * @param[in] hashes hashes if they have already been calculated. nullptr
 * otherwise
 * @param[in] nunique_hashes estimated number of unique hashes if hashes are
 * provided
 * @param[out] grp_infos is grouping_info structures that map row numbers to
 * group numbers
 * @param[in] check_for_null_keys whether to check for null keys. If a key is
 * null and key_dropna=True that row will not be mapped to any group
 * @param[in] key_dropna whether to allow NA values in group keys or not.
 * @param[in] is_parallel: true if data is distributed
 */
void get_group_info(
    std::vector<std::shared_ptr<table_info>>& tables,
    std::shared_ptr<uint32_t[]>& hashes, size_t nunique_hashes,
    std::vector<grouping_info>& grp_infos, const int64_t n_keys,
    bool check_for_null_keys, bool key_dropna, bool is_parallel,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr());

/**
 * Given a set of tables with n key columns, this function calculates the row to
 * group mapping for every row based on its key. For every row in the tables,
 * this only does *one* lookup in the hash map.
 *
 * @param tables: the tables
 * @param[in] hashes hashes for first table in tables, if they have already
 * been calculated. nullptr otherwise.
 * @param[in] nunique_hashes estimated number of unique hashes if hashes are
 * provided (for first table)
 * @param[out] grouping_info structures that map row numbers to group numbers
 * @param[in] consider_missing: whether to return the list of missing rows or
 * not
 * @param[in] key_dropna whether to allow NA values in group keys or not.
 * @param[in] is_parallel: true if data is distributed
 * @param[in] pool: TODO
 */
void get_group_info_iterate(
    std::vector<std::shared_ptr<table_info>>& tables,
    std::shared_ptr<uint32_t[]>& hashes, size_t nunique_hashes,
    std::vector<grouping_info>& grp_infos, const int64_t n_keys,
    const bool consider_missing, bool key_dropna, bool is_parallel,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr());

/**
 * @brief Get total number of groups for input key arrays. This
 * is used by groupby apply.
 *
 * @param table a table of all key arrays
 * @param[out] out_labels output array to fill
 * @param[out] sort_idx sorted group indices
 * @param[in] key_dropna whether to allow NA values in group keys or not.
 * @param is_parallel: true if data is distributed
 * @return int64_t total number of groups
 */
int64_t get_groupby_labels(std::shared_ptr<table_info> table,
                           int64_t* out_labels, int64_t* sort_idx,
                           bool key_dropna, bool is_parallel);

// Python entry point for get_groupby_labels
int64_t get_groupby_labels_py_entry(table_info* table, int64_t* out_labels,
                                    int64_t* sort_idx, bool key_dropna,
                                    bool is_parallel);
