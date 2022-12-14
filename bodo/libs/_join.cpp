// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include "_join.h"
#include "_array_hash.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_distributed.h"
#include "_join_hashing.h"
#include "_shuffle.h"

// An overview of the join design can be found on Confluence:
// https://bodo.atlassian.net/wiki/spaces/B/pages/821624833/Join+Code+Design

// There are several heuristic in this code:
// ---We can use shuffle-join or broadcast-join. We have one constant for the
// maximal
//    size of the broadcasted table. It is set to 10 MB by default (same as
//    Spark). Variable name "CritMemorySize".
// ---For the join, we need to construct one hash map for the keys. If we take
// the left
//    table and is_left=T then we need to build a more complicated hash-map.
//    Thus the size of the table is just one parameter in the choice. We put a
//    factor of 6.0 in this choice. Variable is CritQuotientNrows.

table_info* hash_join_table(
    table_info* left_table, table_info* right_table, bool left_parallel,
    bool right_parallel, int64_t n_key_t, int64_t n_data_left_t,
    int64_t n_data_right_t, int64_t* vect_same_key, bool* key_in_output,
    int64_t* vect_need_typechange, bool is_left, bool is_right, bool is_join,
    bool extra_data_col, bool indicator, bool is_na_equal,
    cond_expr_fn_t cond_func, uint64_t* cond_func_left_columns,
    uint64_t cond_func_left_column_len, uint64_t* cond_func_right_columns,
    uint64_t cond_func_right_column_len, uint64_t* num_rows_ptr) {
    // XXX Not sure if this is needed. has_global_dictionary
    // should be set by the compiler automatically for
    // replicated data
    if (!left_parallel) {
        for (array_info* a : left_table->columns) {
            if (a->arr_type == bodo_array_type::DICT) {
                a->has_global_dictionary = true;
            }
        }
    }
    if (!right_parallel) {
        for (array_info* a : right_table->columns) {
            if (a->arr_type == bodo_array_type::DICT) {
                a->has_global_dictionary = true;
            }
        }
    }

    // Does this join need an additional cond_func
    const bool uses_cond_func = cond_func != nullptr;
    try {
        bool parallel_trace = (left_parallel || right_parallel);
        using BloomFilter = SimdBlockFilterFixed<::hashing::SimpleMixSplit>;
        tracing::Event ev("hash_join_table", parallel_trace);
        // Reading the MPI settings
        int n_pes, myrank;
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        // Doing checks and basic assignments.
        size_t n_key = size_t(n_key_t);
        size_t n_data_left = size_t(n_data_left_t);
        size_t n_data_right = size_t(n_data_right_t);
        size_t n_tot_left = n_key + n_data_left;
        size_t n_tot_right = n_key + n_data_right;
        for (size_t iKey = 0; iKey < n_key; iKey++) {
            // Check that all of the key pairs have matching types.
            CheckEqualityArrayType(left_table->columns[iKey],
                                   right_table->columns[iKey]);
        }
        // in the case of merging on index and one column, it can only be one
        // column
        if (n_key_t > 1 && extra_data_col) {
            throw std::runtime_error(
                "Error in join.cpp::hash_join_table: if extra_data_col=true "
                "then "
                "we must have n_key_t=1.");
        }

        if (ev.is_tracing()) {
            ev.add_attribute("in_left_table_nrows",
                             static_cast<size_t>(left_table->nrows()));
            ev.add_attribute("in_right_table_nrows",
                             static_cast<size_t>(right_table->nrows()));
            ev.add_attribute("g_left_parallel", left_parallel);
            ev.add_attribute("g_right_parallel", right_parallel);
            ev.add_attribute("g_n_key", n_key_t);
            ev.add_attribute("g_n_data_cols_left", n_data_left_t);
            ev.add_attribute("g_n_data_cols_right", n_data_right_t);
            ev.add_attribute("g_is_left", is_left);
            ev.add_attribute("g_is_right", is_right);
            ev.add_attribute("g_extra_data_col", extra_data_col);
        }
        // Unify dictionaries of DICT key columns (required for key comparison)
        // IMPORTANT: need to do this before computing the hashes
        //
        // This implementation of hashing dictionary encoded data is based on
        // the values in the indices array. To do this, we make and enforce
        // a few assumptions
        //
        // 1. Both arrays are dictionary encoded. This is enforced in join.py
        // where determine_table_cast_map requires either both inputs to be
        // dictionary encoded or neither.
        //
        // 2. Both arrays share the exact same dictionary. This occurs in
        // unify_dictionaries.
        //
        // 3. The dictionary does not contain any duplicate values. This is
        // enforced by drop_duplicates_local_dictionary. In particular,
        // drop_duplicates_local_dictionary contains a drop duplicates step
        // that ensures all values are unique. If the dictionary is not global
        // then convert_local_dictionary_to_global will also drop duplicate and
        // drop_duplicates_local_dictionary will be a no-op.
        for (size_t i = 0; i < n_key; i++) {
            array_info* arr1 = left_table->columns[i];
            array_info* arr2 = right_table->columns[i];
            if ((arr1->arr_type == bodo_array_type::DICT) &&
                (arr2->arr_type == bodo_array_type::DICT)) {
                make_dictionary_global_and_unique(arr1, left_parallel);
                make_dictionary_global_and_unique(arr2, right_parallel);
                unify_dictionaries(arr1, arr2, left_parallel, right_parallel);
            }
        }
        // Create sets for cond func columns. These columns act
        // like key columns and are contained inside of key_in_output,
        // so we need an efficient lookup.
        UNORD_SET_CONTAINER<int64_t> left_cond_func_cols_set;
        left_cond_func_cols_set.reserve(cond_func_left_column_len);

        UNORD_SET_CONTAINER<int64_t> right_cond_func_cols_set;
        right_cond_func_cols_set.reserve(cond_func_right_column_len);

        // Non-keys used in cond_func need global + unique dictionaries
        // for hashing, but no unifying is necessary.
        for (size_t i = 0; i < cond_func_left_column_len; i++) {
            uint64_t col_num = cond_func_left_columns[i];
            if (col_num >= n_key) {
                auto arr = left_table->columns[col_num];
                if (arr->arr_type == bodo_array_type::DICT) {
                    make_dictionary_global_and_unique(arr, left_parallel);
                }
                left_cond_func_cols_set.insert(col_num);
            }
        }
        for (size_t i = 0; i < cond_func_right_column_len; i++) {
            uint64_t col_num = cond_func_right_columns[i];
            if (col_num >= n_key) {
                auto arr = right_table->columns[col_num];
                if (arr->arr_type == bodo_array_type::DICT) {
                    make_dictionary_global_and_unique(arr, right_parallel);
                }
                right_cond_func_cols_set.insert(col_num);
            }
        }

        // Now deciding how we handle the parallelization of the tables
        // and doing the allgather/shuffle exchanges as needed.
        // There are two possible algorithms:
        // --- shuffle join where we shuffle both tables and then do the join.
        // --- broadcast join where one table is small enough to be gathered and
        //     broadcast to all nodes.
        // Note that Spark has other join algorithms, e.g. sort-shuffle join.
        //
        // Relative complexity of each algorithms:
        // ---Shuffle join requires shuffling which implies the use of
        // MPI_alltoallv
        //    and uses a lot of memory.
        // ---Broadcast join requires one table to be gathered which may be
        // advantageous
        //    if one table is much smaller. Still making a table replicated
        //    should give pause to users.
        //
        // Both algorithms require the construction of a hash map for the keys.
        //
        table_info *work_left_table, *work_right_table;
        bool free_work_left = false, free_work_right = false;
        // Default replicated values are opposite of parallel. These
        // are updated if we broadcast a table.
        bool left_replicated = !left_parallel,
             right_replicated = !right_parallel;

        if (left_parallel && right_parallel) {
            // If both tables are parallel then we need to decide between
            // shuffle and broadcast join.

            // Determine the memory size of each table
            int64_t left_total_memory = table_global_memory_size(left_table);
            int64_t right_total_memory = table_global_memory_size(right_table);
            // Determine the threshold for broadcast join. We default to 10MB,
            // which matches Spark, unless to user specifies a specific
            // threshold for their infrastructure.
            int CritMemorySize = 10 * 1024 * 1024;  // in bytes
            char* bcast_threshold = std::getenv("BODO_BCAST_JOIN_THRESHOLD");
            if (bcast_threshold) {
                CritMemorySize = std::stoi(bcast_threshold);
            }
            if (CritMemorySize < 0) {
                throw std::runtime_error("hash_join: CritMemorySize < 0");
            }
            if (ev.is_tracing()) {
                ev.add_attribute("g_left_total_memory", left_total_memory);
                ev.add_attribute("g_right_total_memory", right_total_memory);
                ev.add_attribute("g_bloom_filter_supported",
                                 bloom_filter_supported());
                ev.add_attribute("CritMemorySize", CritMemorySize);
            }
            bool all_gather = true;
            // Broadcast the smaller table if its replicated size is below a
            // size limit (CritMemorySize) and not similar or larger than the
            // size of the local "large" table (the latter is to avoid
            // detrimental impact on parallelization/scaling, note that bloom
            // filter approach also reduces cost of shuffle and does not have
            // sequential bottleneck)
            // We should tune this doing a more extensive study, see:
            // https://bodo.atlassian.net/browse/BE-1030
            float global_short_to_local_long_ratio_limit;
            if (bloom_filter_supported())
                global_short_to_local_long_ratio_limit = 0.8;
            else
                // Since we can't rely on bloom filters to reduce the shuffle
                // cost we give some more headroom to use broadcast join
                global_short_to_local_long_ratio_limit = 2.0;
            if (left_total_memory < right_total_memory &&
                left_total_memory < CritMemorySize &&
                left_total_memory < (right_total_memory / double(n_pes) *
                                     global_short_to_local_long_ratio_limit)) {
                work_left_table =
                    gather_table(left_table, -1, all_gather, parallel_trace);
                free_work_left = true;
                work_right_table = right_table;
                left_replicated = true;
            } else if (right_total_memory <= left_total_memory &&
                       right_total_memory < CritMemorySize &&
                       right_total_memory <
                           (left_total_memory / double(n_pes) *
                            global_short_to_local_long_ratio_limit)) {
                work_left_table = left_table;
                work_right_table =
                    gather_table(right_table, -1, all_gather, parallel_trace);
                free_work_right = true;
                right_replicated = true;
            } else {
                // If the smaller table is larger than the threshold
                // we do a shuffle-join. To shuffle the tables we build
                // a hash table (ensuring that comparable
                // types hash to the same values).

                // only do filters for inner join for now
                BloomFilter* bloom_left = nullptr;
                BloomFilter* bloom_right = nullptr;
                uint32_t* hashes_left = nullptr;
                uint32_t* hashes_right = nullptr;
                if (bloom_filter_supported() && !is_left && !is_right) {
                    bool make_bloom_left = true;
                    bool make_bloom_right = true;
                    hashes_left =
                        coherent_hash_keys_table(left_table, right_table, n_key,
                                                 SEED_HASH_PARTITION, true);
                    hashes_right =
                        coherent_hash_keys_table(right_table, left_table, n_key,
                                                 SEED_HASH_PARTITION, true);
                    const int64_t left_table_nrows = left_table->nrows();
                    const int64_t right_table_nrows = right_table->nrows();
                    int64_t left_table_global_nrows;
                    int64_t right_table_global_nrows;
                    // TODO do this in a single reduction?
                    MPI_Allreduce(&left_table_nrows, &left_table_global_nrows,
                                  1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
                    MPI_Allreduce(&right_table_nrows, &right_table_global_nrows,
                                  1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
                    // Filter built from table A is used to filter table
                    // B. If A is much larger than B we don't make a filter
                    // from A because the cost of making the bloom filter will
                    // probably be larger than what we save from filtering B
                    // Also see https://bodo.atlassian.net/browse/BE-1321
                    // regarding tuning these values
                    if (left_table_global_nrows >
                        (right_table_global_nrows * 10))
                        make_bloom_left = false;
                    if (right_table_global_nrows >
                        (left_table_global_nrows * 10))
                        make_bloom_right = false;
                    static constexpr size_t MAX_BLOOM_SIZE = 100 * 1024 * 1024;
                    size_t left_cardinality = 0;
                    size_t right_cardinality = 0;
                    // We are going to build global bloom filters, and we need
                    // to know how many unique elements we are inserting into
                    // each to set a bloom filter size that has good false
                    // positive probability. We only get global cardinality
                    // estimate if there is a reasonable chance that the bloom
                    // filter will be smaller than MAX_BLOOM_SIZE.
                    // Also see https://bodo.atlassian.net/browse/BE-1321
                    // regarding tuning these values
                    make_bloom_left =
                        make_bloom_left &&
                        (bloom_size_bytes(double(left_table_global_nrows) *
                                          0.3) < MAX_BLOOM_SIZE);
                    make_bloom_right =
                        make_bloom_right &&
                        (bloom_size_bytes(double(right_table_global_nrows) *
                                          0.3) < MAX_BLOOM_SIZE);
                    // Additionally we don't make a bloom filter if it is larger
                    // than MAX_BLOOM_SIZE or much larger than the data that
                    // it's supposed to filter (see previous reasoning)
                    // Also see https://bodo.atlassian.net/browse/BE-1321
                    // regarding tuning these values
                    if (make_bloom_left) {
                        left_cardinality =
                            std::get<1>(get_nunique_hashes_global(
                                hashes_left, left_table_nrows, parallel_trace));
                        size_t bloom_left_bytes =
                            bloom_size_bytes(left_cardinality);
                        if ((bloom_left_bytes > MAX_BLOOM_SIZE) ||
                            (bloom_left_bytes /
                                 (right_total_memory / double(n_pes)) >
                             2.0))
                            make_bloom_left = false;
                    }
                    if (make_bloom_right) {
                        right_cardinality =
                            std::get<1>(get_nunique_hashes_global(
                                hashes_right, right_table_nrows,
                                parallel_trace));
                        size_t bloom_right_bytes =
                            bloom_size_bytes(right_cardinality);
                        if ((bloom_right_bytes > MAX_BLOOM_SIZE) ||
                            (bloom_right_bytes /
                                 (left_total_memory / double(n_pes)) >
                             2.0))
                            make_bloom_right = false;
                    }
                    ev.add_attribute("g_make_bloom_left", make_bloom_left);
                    ev.add_attribute("g_make_bloom_right", make_bloom_right);

                    if (make_bloom_left) {
                        tracing::Event ev_bloom("make_bloom", parallel_trace);
                        // A bloom filter is a set, and we need to know how
                        // many unique elements we are inserting so that the
                        // implementation chooses a buffer size that guarantees
                        // low false positive probability. We are inserting
                        // the hashes, and this is going to be a global set
                        // (based on hashes from all the ranks) so we use the
                        // estimated global cardinality of the hashes
                        bloom_left = new BloomFilter(left_cardinality);
                        bloom_left->AddAll(hashes_left, 0, left_table_nrows);
                        // Do a union of the filters from all ranks (performs a
                        // bitwise OR allreduce of the bloom filter's buffer)
                        bloom_left->union_reduction();
                        ev_bloom.add_attribute("input_len", left_table_nrows);
                        ev_bloom.add_attribute("g_size", left_cardinality);
                        ev_bloom.add_attribute(
                            "g_size_bytes",
                            static_cast<size_t>(bloom_left->SizeInBytes()));
                        ev_bloom.add_attribute("which", "left");
                    }
                    if (make_bloom_right) {
                        tracing::Event ev_bloom("make_bloom", parallel_trace);
                        bloom_right = new BloomFilter(right_cardinality);
                        bloom_right->AddAll(hashes_right, 0, right_table_nrows);
                        bloom_right->union_reduction();
                        ev_bloom.add_attribute("input_len", right_table_nrows);
                        ev_bloom.add_attribute("g_size", right_cardinality);
                        ev_bloom.add_attribute(
                            "g_size_bytes",
                            static_cast<size_t>(bloom_right->SizeInBytes()));
                        ev_bloom.add_attribute("which", "right");
                    }
                }

                work_left_table = coherent_shuffle_table(
                    left_table, right_table, n_key, hashes_left, bloom_right);
                work_right_table = coherent_shuffle_table(
                    right_table, left_table, n_key, hashes_right, bloom_left);
                free_work_left = true;
                free_work_right = true;
                if (hashes_left != nullptr) delete[] hashes_left;
                if (hashes_right != nullptr) delete[] hashes_right;
                if (bloom_left != nullptr) delete bloom_left;
                if (bloom_right != nullptr) delete bloom_right;
            }
        } else {
            // If either table is already replicated then we
            // always do a broadcast-join (without the broadcast)
            work_left_table = left_table;
            work_right_table = right_table;
        }
        ev.add_attribute("g_left_replicated", left_replicated);
        ev.add_attribute("g_right_replicated", right_replicated);

        // At this point we always refer to the work tables, which abstracts
        // away the original left and right tables. We cannot garbage
        // collect either table because they may be reused in the Python code,
        // so we must store both the original table and the shuffled/broadcast
        // table.

        size_t n_rows_left = work_left_table->nrows();
        size_t n_rows_right = work_right_table->nrows();

        // Create a data structure containing the columns to match the format
        // expected by cond_func. We create two pairs of vectors, one with
        // the array_infos, which handle general types, and one with just data1
        // as a fast path for accessing numeric data. These include both keys
        // and data columns as either can be used in the cond_func.
        std::vector<array_info*> left_table_infos = work_left_table->columns;
        std::vector<array_info*> right_table_infos = work_right_table->columns;
        std::vector<void*> col_ptrs_left(n_tot_left);
        std::vector<void*> col_ptrs_right(n_tot_right);
        // Vectors for null bitmaps for fast null checking from the cfunc
        std::vector<void*> null_bitmap_left(n_tot_left);
        std::vector<void*> null_bitmap_right(n_tot_right);
        for (size_t i = 0; i < n_tot_left; i++) {
            col_ptrs_left[i] =
                static_cast<void*>(work_left_table->columns[i]->data1);
            null_bitmap_left[i] =
                static_cast<void*>(work_left_table->columns[i]->null_bitmask);
        }
        for (size_t i = 0; i < n_tot_right; i++) {
            col_ptrs_right[i] =
                static_cast<void*>(work_right_table->columns[i]->data1);
            null_bitmap_right[i] =
                static_cast<void*>(work_right_table->columns[i]->null_bitmask);
        }

        // Now computing the hashes that will be used in the hash map
        // or compared to the hash map.
        //
        uint32_t* hashes_left = hash_keys_table(work_left_table, n_key,
                                                SEED_HASH_JOIN, parallel_trace);
        uint32_t* hashes_right = hash_keys_table(
            work_right_table, n_key, SEED_HASH_JOIN, parallel_trace);
        // Compute the ratio of the table sizes on the current rank.
        // This is used when determining which table populates the
        // hash map.
        double quot1 = double(n_rows_left) / double(n_rows_right);
        double quot2 = double(n_rows_right) / double(n_rows_left);

        // In the computation of the join we are building some hash map on one
        // of the tables. Which one we choose influences the running time of
        // course. The parameters to weigh are:
        //
        // --- Selecting the smaller table. The smaller the table,
        // the smaller the hash map, so this is usually a good heuristic
        // --- Which tables require an outer merge. Outer merges require
        // tracking which rows found matches in the other table. This
        // complicates the data structure as we now need to track if
        // a row is used.
        //
        // To make this decision, we compare
        int MethodChoice;  // 1: Populate the hash map with the table with fewer
                           // rows. 2: Populate the hash map with the table that
                           // uniquely
                           //    requires an outer join (i.e. is_left/is_right).
        if (is_left == is_right) {
            // In that case we are doing either inner or full outer merge
            // the only relevant metric is table size.
            MethodChoice = 1;
        } else {
            // In the case of is_left <> is_right we must decide if the size
            // of the tables or the outer join property are more important.
            // To do this, we set a threshhold K (CritQuotientNrows) and compare
            // the size of the table. If either table is has K times more rows
            // we will always make the smaller table the table that populates
            // the hash map, regardless of if that table needs an outer join.
            // If the ratio of rows is less than K, then populate the hash map
            // with the table that doesn't require an outer join.
            //
            // TODO: Justify our threshhold of 1 table being 6x larger than
            // the other.
            double CritQuotientNrows = 6.0;
            if (quot2 < CritQuotientNrows && quot1 < CritQuotientNrows)
                // In that case the large table is not so large comparable to
                // the short one This means that we can use the is_left /
                // is_right for making the choice
                MethodChoice = 2;
            else
                // In that case one table is much larger than the other,
                // therefore the choice by the number of rows is the best here.
                MethodChoice = 1;
        }
        // We have chosen our criteria for selecting our table to populate
        // the hash map, now we map this to a variable selecting the table.
        // 0: left table
        // 1: right table
        int ChoiceOpt;
        if (MethodChoice == 1) {
            // We choose by the number of rows.
            if (n_rows_left < n_rows_right) {
                ChoiceOpt = 0;
            } else {
                ChoiceOpt = 1;
            }
        } else {
            // When is_left <> is_right
            // and the tables are similarly sized
            // we take the table without the outer join.
            // to avoid tracking if a row has a match in the hash table.
            if (is_right) {  // Thus is_left = false
                ChoiceOpt = 0;
            } else {
                ChoiceOpt = 1;
            }
        }
        ev.add_attribute("MethodChoice", MethodChoice);
        ev.add_attribute("ChoiceOpt", ChoiceOpt);

        // Select the "short" and "long" table based upon the choice opt
        // that we just set. For the short table we construct a hash map, for
        // the long table, we simply iterate over the rows and see if the keys
        // are in the hash map.
        size_t short_table_rows, long_table_rows;  // the number of rows
        uint32_t *short_table_hashes, *long_table_hashes;
        // This corresponds to is_left/is_right and determines
        // if the short/long tables are outer joins.
        bool short_table_outer, long_table_outer;
        table_info *short_table, *long_table;
        bool short_replicated, long_replicated;
        // Flag set to true if left table is assigned as short table,
        // used in equal_fct below
        bool short_is_left;
        if (ChoiceOpt == 0) {
            // short = left and long = right
            short_is_left = true;
            short_table_outer = is_left;
            long_table_outer = is_right;
            short_table = work_left_table;
            long_table = work_right_table;
            short_table_rows = n_rows_left;
            long_table_rows = n_rows_right;
            short_table_hashes = hashes_left;
            long_table_hashes = hashes_right;
            short_replicated = left_replicated;
            long_replicated = right_replicated;
        } else {
            // short = right and long = left
            short_is_left = false;
            short_table_outer = is_right;
            long_table_outer = is_left;
            short_table = work_right_table;
            long_table = work_left_table;
            short_table_rows = n_rows_right;
            long_table_rows = n_rows_left;
            short_table_hashes = hashes_right;
            long_table_hashes = hashes_left;
            short_replicated = right_replicated;
            long_replicated = left_replicated;
        }
        // If exactly one table is replicated and that table uses an outer
        // join (i.e. is_left), we do not have enough information in one
        // rank to determine that a row has no matches. As a result,
        // these variables track if we will need to perform a reduction
        // or if a row has a match.
        bool short_miss_needs_reduction =
            short_table_outer && short_replicated && !long_replicated;
        bool long_miss_needs_reduction =
            long_table_outer && long_replicated && !short_replicated;

        tracing::Event ev_tuples("compute_tuples", parallel_trace);
        if (ev_tuples.is_tracing()) {
            ev_tuples.add_attribute("short_table_rows", short_table_rows);
            ev_tuples.add_attribute("long_table_rows", long_table_rows);
            ev_tuples.add_attribute("short_table_outer", short_table_outer);
            ev_tuples.add_attribute("long_table_outer", long_table_outer);
        }

        joinHashFcts::HashHashJoinTable hash_fct{
            short_table_rows, short_table_hashes, long_table_hashes};
        joinHashFcts::KeyEqualHashJoinTable equal_fct{
            short_table_rows, n_key, short_table, long_table, is_na_equal};
        tracing::Event ev_alloc_map("alloc_hashmap", parallel_trace);

        // The entList contains the identical keys with the corresponding rows.
        // We address the entry by the row index. We store all the rows which
        // are identical in a "group". The hashmap stores the group number
        // and the groups are stored in the `groups` std::vector.
        //
        // The groups are stored in the hash map from values 1 to n,
        // although the groups indices are still 0 to n - 1.
        // This is because 0 is a reserved value that we use to indicate
        // that a value is not found in the hash map when inserting the short
        // table.
        //
        // NOTE: we don't store the group vectors in the map because it makes
        // map (re)allocs and deallocation very expensive, and possibly also
        // makes insertion and/or lookups slower.
        // StoreHash=true speeds up join code overall in TPC-H.
        // If we also need to do short_table_outer then we store an index in the
        // first position of the std::vector
        UNORD_MAP_CONTAINER<size_t, size_t, joinHashFcts::HashHashJoinTable,
                            joinHashFcts::KeyEqualHashJoinTable>
            entList({}, hash_fct, equal_fct);
        // reserving space is very important to avoid expensive reallocations
        // (at the cost of using more memory)
        // [BE-1078]: Should this use statistics to influence how much we
        // reserve?
        entList.reserve(short_table_rows);
        std::vector<std::vector<size_t>> groups;
        groups.reserve(short_table_rows);

        // Define additional information needed for non-equality conditions,
        // determined by 'uses_cond_func'. We need a vector of hash maps for the
        // short table groups. In addition, we also need hashes for the short
        // table on all columns that are not since they will insert into the
        // hash map.
        uint32_t* short_nonequal_key_hashes = nullptr;
        std::vector<UNORD_MAP_CONTAINER<
            size_t, size_t, joinHashFcts::SecondLevelHashHashJoinTable,
            joinHashFcts::SecondLevelKeyEqualHashJoinTable>>
            second_level_hash_maps;
        // Keep track of which table to use to populate the second level hash
        // table if 'uses_cond_func'
        uint64_t* short_data_key_cols = nullptr;
        uint64_t short_data_key_n_cols = 0;

        if (uses_cond_func) {
            if (short_is_left) {
                short_data_key_cols = cond_func_left_columns;
                short_data_key_n_cols = cond_func_left_column_len;
            } else {
                short_data_key_cols = cond_func_right_columns;
                short_data_key_n_cols = cond_func_right_column_len;
            }
            short_nonequal_key_hashes = hash_data_cols_table(
                short_table->columns, short_data_key_cols,
                short_data_key_n_cols, SEED_HASH_JOIN, parallel_trace);
            // [BE-1078]: Should this use statistics to influence how much we
            // reserve?
            second_level_hash_maps.reserve(short_table_rows);
        }
        joinHashFcts::SecondLevelHashHashJoinTable second_level_hash_fct{
            short_nonequal_key_hashes};
        joinHashFcts::SecondLevelKeyEqualHashJoinTable second_level_equal_fct{
            short_table, short_data_key_cols, short_data_key_n_cols};

        // ListPairWrite is the table used for the output
        // It precises the index used for the writing of the output table.
        std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> ListPairWrite;
        // [BE-1078]: how much should we reserve?
        ListPairWrite.reserve(long_table_rows);
        ev_alloc_map.finalize();

        // If we will need to perform a reduction on long table matches,
        // specify an allocation size for the vector.
        size_t n_bytes_long = 0;
        if (long_miss_needs_reduction) {
            n_bytes_long = (long_table_rows + 7) >> 3;
        }
        // V_long_map and V_short_map takes similar roles.
        // They indicate if an entry in the short or long table
        // has been matched to the other side.
        // This is needed only if said table is replicated
        // (i.e long_miss_needs_reduction/short_miss_needs_reduction).
        std::vector<uint8_t> V_long_map(n_bytes_long, 255);
        if (short_table_outer) {
            // The loop over the short table.
            // entries are stored one by one and all of them are put in a group
            // even if they are identical in value.
            // The first entry is going to be the index for the boolean array.
            // This code path will be selected whenever we have an OUTER merge.

            tracing::Event ev_groups("calc_groups", parallel_trace);
            // TODO: Refactor code paths into helper functions.
            if (uses_cond_func) {
                // If 'uses_cond_func' we have a separate insertion process. We
                // place the condition before the loop to avoid overhead.
                for (size_t i_short = 0; i_short < short_table_rows;
                     i_short++) {
                    // Check if the group already exists, if it doesn't this
                    // will insert a value.
                    size_t& first_level_group_id = entList[i_short];
                    // first_level_group_id==0 means the equality condition
                    // doesn't have a match
                    if (first_level_group_id == 0) {
                        // Update the value of first_level_group_id stored in
                        // the hash map as well since its pass by reference.
                        first_level_group_id =
                            second_level_hash_maps.size() + 1;
                        UNORD_MAP_CONTAINER<
                            size_t, size_t,
                            joinHashFcts::SecondLevelHashHashJoinTable,
                            joinHashFcts::SecondLevelKeyEqualHashJoinTable>
                            group_map({}, second_level_hash_fct,
                                      second_level_equal_fct);
                        second_level_hash_maps.emplace_back(group_map);
                    }
                    auto& group_map =
                        second_level_hash_maps[first_level_group_id - 1];
                    // Check if the group already exists, if it doesn't this
                    // will insert a value.
                    size_t& second_level_group_id = group_map[i_short];
                    if (second_level_group_id == 0) {
                        // Update the value of group_id stored in the hash map
                        // as well since its pass by reference.
                        second_level_group_id = groups.size() + 1;
                        groups.emplace_back();
                    }
                    groups[second_level_group_id - 1].emplace_back(i_short);
                }
            } else {
                for (size_t i_short = 0; i_short < short_table_rows;
                     i_short++) {
                    // Check if the group already exists, if it doesn't this
                    // will insert a value.
                    size_t& group_id = entList[i_short];
                    // group_id==0 means key doesn't exist in map
                    if (group_id == 0) {
                        // Update the value of group_id stored in the hash map
                        // as well since its pass by reference.
                        group_id = groups.size() + 1;
                        groups.emplace_back();
                    }
                    groups[group_id - 1].emplace_back(i_short);
                }
            }
            ev_groups.finalize();
            //
            // Now iterating and determining how many entries we have to do.
            //
            size_t n_bytes = (groups.size() + 7) >> 3;
            std::vector<uint8_t> V_short_map(n_bytes, 0);
            // We now iterate over all entries of the long table in order to get
            // the entries in the ListPairWrite.

            // TODO: Refactor code paths into helper functions.
            if (uses_cond_func) {
                // If 'uses_cond_func' we have a separate check to search
                // second level hashes. We place the condition before the loop
                // to avoid overhead.
                for (size_t i_long = 0; i_long < long_table_rows; i_long++) {
                    size_t i_long_shift = i_long + short_table_rows;
                    auto iter = entList.find(i_long_shift);
                    if (iter == entList.end()) {
                        if (long_table_outer) {
                            if (long_miss_needs_reduction) {
                                SetBitTo(V_long_map.data(), i_long, false);
                            } else {
                                ListPairWrite.emplace_back(-1, i_long);
                            }
                        }
                    } else {
                        // If the first level matches, check each second level
                        // hash.
                        UNORD_MAP_CONTAINER<
                            size_t, size_t,
                            joinHashFcts::SecondLevelHashHashJoinTable,
                            joinHashFcts::SecondLevelKeyEqualHashJoinTable>
                            group_map =
                                second_level_hash_maps[iter->second - 1];

                        bool has_match = false;
                        // Iterate over all of the keys and compare each group.
                        // TODO [BE-1300]: Explore tsl:sparse_map
                        for (auto& item : group_map) {
                            size_t pos = item.second - 1;
                            std::vector<size_t>& group = groups[pos];
                            // Select a single member
                            size_t cmp_row = group[0];
                            size_t left_ind = 0;
                            size_t right_ind = 0;
                            if (short_is_left) {
                                left_ind = cmp_row;
                                right_ind = i_long;
                            } else {
                                left_ind = i_long;
                                right_ind = cmp_row;
                            }
                            bool match = cond_func(
                                left_table_infos.data(),
                                right_table_infos.data(), col_ptrs_left.data(),
                                col_ptrs_right.data(), null_bitmap_left.data(),
                                null_bitmap_right.data(), left_ind, right_ind);
                            if (match) {
                                // If our group matches, add every row and
                                // update the bitmap
                                SetBitTo(V_short_map.data(), pos, true);
                                has_match = true;
                                for (size_t idx = 0; idx < group.size();
                                     idx++) {
                                    size_t j_short = group[idx];
                                    ListPairWrite.emplace_back(j_short, i_long);
                                }
                            }
                        }
                        if (!has_match && long_table_outer) {
                            // If there is no match, update the long table row.
                            if (long_miss_needs_reduction) {
                                SetBitTo(V_long_map.data(), i_long, false);
                            } else {
                                ListPairWrite.emplace_back(-1, i_long);
                            }
                        }
                    }
                }
            } else {
                for (size_t i_long = 0; i_long < long_table_rows; i_long++) {
                    size_t i_long_shift = i_long + short_table_rows;
                    auto iter = entList.find(i_long_shift);
                    if (iter == entList.end()) {
                        if (long_table_outer) {
                            if (long_miss_needs_reduction) {
                                SetBitTo(V_long_map.data(), i_long, false);
                            } else {
                                ListPairWrite.emplace_back(-1, i_long);
                            }
                        }
                    } else {
                        // If the short table entry are present in output as
                        // well, then we need to keep track whether they are
                        // used or not by the long table.
                        std::vector<size_t>& group = groups[iter->second - 1];
                        size_t pos = iter->second - 1;
                        SetBitTo(V_short_map.data(), pos, true);
                        for (size_t idx = 0; idx < group.size(); idx++) {
                            size_t j_short = group[idx];
                            ListPairWrite.emplace_back(j_short, i_long);
                        }
                    }
                }
            }

            // Perform the reduction on short table misses if necessary
            if (short_miss_needs_reduction) {
                MPI_Allreduce_bool_or(V_short_map);
            }
            int pos_short_disp = 0;
            // Add missing rows for outer joins when there are no matching short
            // table groups.
            for (size_t pos = 0; pos < groups.size(); pos++) {
                std::vector<size_t>& group = groups[pos];
                bool bit = GetBit(V_short_map.data(), pos);
                if (!bit) {
                    for (size_t idx = 0; idx < group.size(); idx++) {
                        size_t j_short = group[idx];
                        // For short_miss_needs_reduction=True, the output table
                        // is distributed. Since the table in input is
                        // replicated, we dispatch it by rank.
                        if (short_miss_needs_reduction) {
                            int node = pos_short_disp % n_pes;
                            if (node == myrank)
                                ListPairWrite.emplace_back(j_short, -1);
                            pos_short_disp++;
                        } else {
                            ListPairWrite.emplace_back(j_short, -1);
                        }
                    }
                }
            }
        } else {
            // The loop over the short table.
            // entries are stored one by one and all of them are put even if
            // identical in value.
            // No need to keep track of the usage of the short table.
            // This code path is selected whenever the short table is an inner
            // join.

            tracing::Event ev_groups("calc_groups", parallel_trace);
            // TODO: Refactor code paths into helper functions.
            if (uses_cond_func) {
                // If 'uses_cond_func' we have a separate check to search
                // second level hashes. We place the condition before the loop
                // to avoid overhead..
                for (size_t i_short = 0; i_short < short_table_rows;
                     i_short++) {
                    // Check if the group already exists, if it doesn't this
                    // will insert a value.
                    size_t& first_level_group_id = entList[i_short];
                    // first_level_group_id==0 means the equality condition
                    // doesn't have a match
                    if (first_level_group_id == 0) {
                        // Update the value of first_level_group_id stored in
                        // the hash map as well since its pass by reference.
                        first_level_group_id =
                            second_level_hash_maps.size() + 1;
                        UNORD_MAP_CONTAINER<
                            size_t, size_t,
                            joinHashFcts::SecondLevelHashHashJoinTable,
                            joinHashFcts::SecondLevelKeyEqualHashJoinTable>
                            group_map({}, second_level_hash_fct,
                                      second_level_equal_fct);
                        second_level_hash_maps.emplace_back(group_map);
                    }
                    auto& group_map =
                        second_level_hash_maps[first_level_group_id - 1];
                    // Check if the group already exists, if it doesn't this
                    // will insert a value.
                    size_t& second_level_group_id = group_map[i_short];
                    if (second_level_group_id == 0) {
                        // Update the value of group_id stored in the hash map
                        // as well since its pass by reference.
                        second_level_group_id = groups.size() + 1;
                        groups.emplace_back();
                    }
                    groups[second_level_group_id - 1].emplace_back(i_short);
                }

            } else {
                for (size_t i_short = 0; i_short < short_table_rows;
                     i_short++) {
                    // Check if the group already exists, if it doesn't this
                    // will insert a value.
                    size_t& group_id = entList[i_short];
                    // group_id==0 means key doesn't exist in map
                    if (group_id == 0) {
                        // Update the value of group_id stored in the hash map
                        // as well since its pass by reference.
                        group_id = groups.size() + 1;
                        groups.emplace_back();
                    }
                    groups[group_id - 1].emplace_back(i_short);
                }
            }
            ev_groups.finalize();
            //
            // Now iterating and determining how many entries we have to do.
            //

            // We now iterate over all entries of the long table in order to get
            // the entries in the ListPairWrite.

            // TODO: Refactor code paths into helper functions.
            if (uses_cond_func) {
                // If 'uses_cond_func' we have a separate check to search
                // second level hashes. We place the condition before the loop
                // to avoid overhead.
                for (size_t i_long = 0; i_long < long_table_rows; i_long++) {
                    size_t i_long_shift = i_long + short_table_rows;
                    auto iter = entList.find(i_long_shift);
                    if (iter == entList.end()) {
                        if (long_table_outer) {
                            if (long_miss_needs_reduction) {
                                SetBitTo(V_long_map.data(), i_long, false);
                            } else {
                                ListPairWrite.emplace_back(-1, i_long);
                            }
                        }
                    } else {
                        // If the first level matches, check each second level
                        // hash.
                        UNORD_MAP_CONTAINER<
                            size_t, size_t,
                            joinHashFcts::SecondLevelHashHashJoinTable,
                            joinHashFcts::SecondLevelKeyEqualHashJoinTable>
                            group_map =
                                second_level_hash_maps[iter->second - 1];

                        bool has_match = false;
                        // Iterate over all of the keys and compare each group.
                        // TODO [BE-1300]: Explore tsl:sparse_map
                        for (auto& item : group_map) {
                            size_t pos = item.second - 1;
                            std::vector<size_t>& group = groups[pos];
                            // Select a single member
                            size_t cmp_row = group[0];
                            size_t left_ind = 0;
                            size_t right_ind = 0;
                            if (short_is_left) {
                                left_ind = cmp_row;
                                right_ind = i_long;
                            } else {
                                left_ind = i_long;
                                right_ind = cmp_row;
                            }
                            bool match = cond_func(
                                left_table_infos.data(),
                                right_table_infos.data(), col_ptrs_left.data(),
                                col_ptrs_right.data(), null_bitmap_left.data(),
                                null_bitmap_right.data(), left_ind, right_ind);
                            if (match) {
                                // If our group matches, add every row
                                has_match = true;
                                for (size_t idx = 0; idx < group.size();
                                     idx++) {
                                    size_t j_short = group[idx];
                                    ListPairWrite.emplace_back(j_short, i_long);
                                }
                            }
                        }
                        if (!has_match && long_table_outer) {
                            // If there is no match, update the long table row.
                            if (long_miss_needs_reduction) {
                                SetBitTo(V_long_map.data(), i_long, false);
                            } else {
                                ListPairWrite.emplace_back(-1, i_long);
                            }
                        }
                    }
                }

            } else {
                for (size_t i_long = 0; i_long < long_table_rows; i_long++) {
                    size_t i_long_shift = i_long + short_table_rows;
                    auto iter = entList.find(i_long_shift);
                    if (iter == entList.end()) {
                        if (long_table_outer) {
                            if (long_miss_needs_reduction) {
                                SetBitTo(V_long_map.data(), i_long, false);
                            } else {
                                ListPairWrite.emplace_back(-1, i_long);
                            }
                        }
                    } else {
                        // If the short table entry are present in output as
                        // well, then we need to keep track whether they are
                        // used or not by the long table.
                        std::vector<size_t>& group = groups[iter->second - 1];
                        for (auto& j_short : group) {
                            ListPairWrite.emplace_back(j_short, i_long);
                        }
                    }
                }
            }
        }
        tracing::Event ev_clear_map("dealloc_hashmap", parallel_trace);
        // Data structures used during computation of the tuples can become
        // quite large and their deallocation can take a non-negligible amount
        // of time. We dealloc them here to free memory for the next stage and
        // also to trace dealloc time
        std::vector<std::vector<size_t>>().swap(
            groups);  // force groups vector to dealloc
        entList.clear();
        entList.reserve(0);  // try to force dealloc of hashmap
        // Force second_level_hash_maps vector to dealloc. This should
        // deallocate the hashmaps as well.
        std::vector<UNORD_MAP_CONTAINER<
            size_t, size_t, joinHashFcts::SecondLevelHashHashJoinTable,
            joinHashFcts::SecondLevelKeyEqualHashJoinTable>>()
            .swap(second_level_hash_maps);

        // Delete the hashes
        delete[] hashes_left;
        delete[] hashes_right;
        if (short_nonequal_key_hashes != nullptr) {
            delete[] short_nonequal_key_hashes;
        }
        ev_clear_map.finalize();
        // In replicated case, we put the long rows in distributed output
        if (long_table_outer && long_miss_needs_reduction) {
            MPI_Allreduce_bool_or(V_long_map);
            int pos = 0;
            for (size_t i_long = 0; i_long < long_table_rows; i_long++) {
                bool bit = GetBit(V_long_map.data(), i_long);
                // The replicated input table is dispatched over the rows in the
                // distributed output.
                if (!bit) {
                    int node = pos % n_pes;
                    if (node == myrank) ListPairWrite.emplace_back(-1, i_long);
                    pos++;
                }
            }
        }
        ev_tuples.add_attribute("output_nrows", ListPairWrite.size());
        ev_tuples.finalize();
        // TODO if we are tight on memory for the next phase and ListPairWrite
        // capacity is much greater than its size, we could do a realloc+resize
        // here
        std::vector<array_info*> out_arrs;
        // Computing the last time at which a column is used.
        // This is for the call to decref_array.
        // There are many cases to cover, so it is better to preprocess
        // first to determine last usage before creating the return
        // table.
        // last_col_use_left/right can be assigned many different values.
        // Here are there meanings.
        // 0 - Default value. This means the column is not used in the output at
        // all. 1 - Column is the output of converting an index to the output
        // data columns. 2 - Column is a key column with the same name in the
        // left and right tables. 3 - Column is in the left table and is not a
        // matching key with the right table. 4 - Column is in the right table,
        // is not a matching key with the left table,
        //        and the operation is not a join in Pandas.
        std::vector<uint8_t> last_col_use_left(n_tot_left, 0);
        std::vector<uint8_t> last_col_use_right(n_tot_right, 0);
        if (extra_data_col) {
            size_t i = 0;
            last_col_use_left[i] = 1;
            last_col_use_right[i] = 1;
        }
        for (size_t i = 0; i < n_tot_left; i++) {
            if (i < n_key && vect_same_key[i] == 1) {
                // Key is shared by both tables
                last_col_use_left[i] = 2;
                last_col_use_right[i] = 2;
            } else {
                // Column is only in the left table.
                last_col_use_left[i] = 3;
            }
        }
        for (size_t i = 0; i < n_tot_right; i++) {
            // There are two cases where we put the column in output:
            // ---It is a right key column with different name from the left.
            // ---It is a right data column
            if (i >= n_key ||
                (i < n_key && vect_same_key[i < n_key ? i : 0] == 0 &&
                 !is_join)) {
                // TODO: Why do we check !is_join?
                // Column is only in the right table.
                last_col_use_right[i] = 4;
            }
        }
        // If the arrays aren't used in the output decref immediately.
        for (size_t i = 0; i < n_tot_left; i++) {
            if (last_col_use_left[i] == 0) {
                array_info* left_arr = work_left_table->columns[i];
                decref_array(left_arr);
            }
        }
        for (size_t i = 0; i < n_tot_right; i++) {
            if (last_col_use_right[i] == 0) {
                array_info* right_arr = work_right_table->columns[i];
                decref_array(right_arr);
            }
        }
        // Determine the number of rows in your local chunk of the output.
        // This is passed to Python in case all columns are dead.
        uint64_t num_rows = ListPairWrite.size();
        *num_rows_ptr = num_rows;

        // Construct the output tables. This merges the results in the left and
        // right tables. We resume using work_left_table and work_right_table to
        // ensure we match the expected column order (as opposed to short/long).

        // Inserting the optional column in the case of merging on column and
        // index.
        if (extra_data_col) {
            tracing::Event ev_fill_optional("fill_extra_data_col",
                                            parallel_trace);
            size_t i = 0;
            bool map_integer_type = false;
            array_info* left_arr = work_left_table->columns[i];
            array_info* right_arr = work_right_table->columns[i];
            if (ChoiceOpt == 0) {
                out_arrs.push_back(RetrieveArray_TwoColumns(
                    left_arr, right_arr, ListPairWrite, 2, map_integer_type));
            } else {
                out_arrs.push_back(RetrieveArray_TwoColumns(
                    right_arr, left_arr, ListPairWrite, 2, map_integer_type));
            }
            // After adding the optional collect decref the original values
            if (last_col_use_left[i] == 1) {
                decref_array(left_arr);
            }
            if (last_col_use_right[i] == 1) {
                decref_array(right_arr);
            }
        }

        // Where to check in key_in_output if we have a key or
        // a general merge condition.
        offset_t key_in_output_idx = 0;

        // Inserting the Left side of the table
        tracing::Event ev_fill_left("fill_left", parallel_trace);
        int idx = 0;
        for (size_t i = 0; i < n_tot_left; i++) {
            if (i < n_key && vect_same_key[i] == 1) {
                // We are in the case of a key that has the same name on left
                // and right. This means that additional NaNs cannot happen.
                array_info* left_arr = work_left_table->columns[i];
                array_info* right_arr = work_right_table->columns[i];
                if (key_in_output[key_in_output_idx]) {
                    bool map_integer_type = false;
                    if (ChoiceOpt == 0) {
                        out_arrs.emplace_back(RetrieveArray_TwoColumns(
                            left_arr, right_arr, ListPairWrite, 2,
                            map_integer_type));
                    } else {
                        out_arrs.emplace_back(RetrieveArray_TwoColumns(
                            right_arr, left_arr, ListPairWrite, 2,
                            map_integer_type));
                    }
                    idx++;
                }
                // Decref columns that are no longer used
                if (last_col_use_left[i] == 2) {
                    decref_array(left_arr);
                }
                if (last_col_use_right[i] == 2) {
                    decref_array(right_arr);
                }
                key_in_output_idx++;
            } else {
                // Check if a column may be eliminated from the output
                bool check_key_in_output =
                    (i < n_key) || (left_cond_func_cols_set.count(i));
                array_info* left_arr = work_left_table->columns[i];
                // We are in data case or in the case of a key that is taken
                // only from one side. Therefore we have to plan for the
                // possibility of additional NaN.
                if (!check_key_in_output || key_in_output[key_in_output_idx]) {
                    bool map_integer_type = vect_need_typechange[idx];
                    array_info* right_arr = nullptr;
                    if (ChoiceOpt == 0) {
                        out_arrs.emplace_back(RetrieveArray_TwoColumns(
                            left_arr, right_arr, ListPairWrite, 0,
                            map_integer_type));
                    } else {
                        out_arrs.emplace_back(RetrieveArray_TwoColumns(
                            right_arr, left_arr, ListPairWrite, 1,
                            map_integer_type));
                    }
                    idx++;
                }
                // Decref columns that are no longer used
                if (last_col_use_left[i] == 3) {
                    decref_array(left_arr);
                }
                if (check_key_in_output) {
                    key_in_output_idx++;
                }
            }
        }
        // Clear to free memory
        left_cond_func_cols_set.clear();
        ev_fill_left.finalize();
        tracing::Event ev_fill_right("fill_right", parallel_trace);
        // Inserting the right side of the table.
        for (size_t i = 0; i < n_tot_right; i++) {
            // There are two cases where we put the column in output:
            // ---It is a right key column with different name from the left.
            // ---It is a right data column
            bool is_new_key =
                i < n_key && vect_same_key[i < n_key ? i : 0] == 0 && !is_join;
            if (i >= n_key || is_new_key) {
                array_info* right_arr = work_right_table->columns[i];
                // Check if a column may be eliminated from the output.
                bool check_key_in_output =
                    is_new_key || (right_cond_func_cols_set.count(i));
                if (!check_key_in_output || key_in_output[key_in_output_idx]) {
                    bool map_integer_type = vect_need_typechange[idx];
                    array_info* left_arr = nullptr;
                    if (ChoiceOpt == 0) {
                        out_arrs.emplace_back(RetrieveArray_TwoColumns(
                            left_arr, right_arr, ListPairWrite, 1,
                            map_integer_type));
                    } else {
                        out_arrs.emplace_back(RetrieveArray_TwoColumns(
                            right_arr, left_arr, ListPairWrite, 0,
                            map_integer_type));
                    }
                    idx++;
                }
                // Decref columns that are no longer used
                if (last_col_use_right[i] == 4) {
                    decref_array(right_arr);
                }
                if (check_key_in_output) {
                    key_in_output_idx++;
                }
            }
        }
        // Clear to free memory
        right_cond_func_cols_set.clear();
        ev_fill_right.finalize();
        // Create indicator column if indicator=True
        if (indicator) {
            tracing::Event ev_indicator("create_indicator", parallel_trace);
            array_info* indicator_col =
                alloc_array(num_rows, -1, -1, bodo_array_type::CATEGORICAL,
                            Bodo_CTypes::INT8, 0, 3);
            for (size_t rownum = 0; rownum < num_rows; rownum++) {
                // Determine the source of each row. At most 1 value can be -1.
                // Whichever value is -1, other table is the source of the row.
                auto table_nulls = ListPairWrite[rownum];
                // If the first is -1 if ChoiceOpt == 0, left is null (value is
                // 1) if ChoiceOpt != 0, right is null (value is 0)
                if (table_nulls.first == -1) {
                    indicator_col->data1[rownum] = ChoiceOpt == 0;
                    // If the first is -1 if ChoiceOpt != 0, left is null (value
                    // is 1) if ChoiceOpt != 0, right is null (value is 0)
                } else if (table_nulls.second == -1) {
                    indicator_col->data1[rownum] = ChoiceOpt != 0;
                    // If neither is -1 the output is both (2)
                } else {
                    indicator_col->data1[rownum] = 2;
                }
            }
            out_arrs.emplace_back(indicator_col);
        }

        // TODO if we see significant tracing gap at the end of hash_join_table
        // or right after it, we need to trace the "freeing" portion of this
        // function, and manually deallocate ListPairWrite, V_long_map, etc.
        // to make sure deallocations are captured by the trace event
        if (free_work_left) {
            delete_table(work_left_table);
        }
        if (free_work_right) {
            delete_table(work_right_table);
        }
        // Only return a table if there is at least 1
        // output column.
        if (out_arrs.size() > 0) {
            return new table_info(out_arrs);
        } else {
            return nullptr;
        }
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

/**
 * @brief handle dict-encoded columns of input tables to cross join.
 * Dictionaries need to be global since we use broadcast.
 *
 * @param left_table left input table to cross join
 * @param right_table right input table to cross join
 * @param left_parallel left table is parallel
 * @param right_parallel right table is parallel
 */
void cross_join_handle_dict_encoded(table_info* left_table,
                                    table_info* right_table, bool left_parallel,
                                    bool right_parallel) {
    // Set has_global_dictionary for replicated data just in case.
    // Should not be necessary since has_global_dictionary
    // should be set by the compiler automatically for replicated data
    if (!left_parallel) {
        for (array_info* a : left_table->columns) {
            if (a->arr_type == bodo_array_type::DICT) {
                a->has_global_dictionary = true;
            }
        }
    }
    if (!right_parallel) {
        for (array_info* a : right_table->columns) {
            if (a->arr_type == bodo_array_type::DICT) {
                a->has_global_dictionary = true;
            }
        }
    }
    // make all dictionaries global (necessary for broadcast and potentially
    // other operations)
    for (array_info* a : left_table->columns) {
        if (a->arr_type == bodo_array_type::DICT) {
            make_dictionary_global_and_unique(a, left_parallel);
        }
    }
    for (array_info* a : right_table->columns) {
        if (a->arr_type == bodo_array_type::DICT) {
            make_dictionary_global_and_unique(a, right_parallel);
        }
    }
}

/**
 * @brief cross join two tables locally with a simple nested loop join.
 * steals references (decrefs) both inputs since it calls RetrieveTable.
 *
 * @param left_table left input table
 * @param right_table right input table
 * @param parallel_trace parallel flag to pass to tracing calls
 * @return table_info* cross join output table
 */
table_info* cross_join_table_local(table_info* left_table,
                                   table_info* right_table,
                                   bool parallel_trace) {
    tracing::Event ev("cross_join_table_local", parallel_trace);
    size_t n_rows_left = left_table->nrows();
    size_t n_rows_right = right_table->nrows();
    size_t n_rows_out = n_rows_left * n_rows_right;
    std::vector<int64_t> left_idxs;
    std::vector<int64_t> right_idxs;
    left_idxs.reserve(n_rows_out);
    right_idxs.reserve(n_rows_out);

    // set 500K block size to make sure block data of all cores fits in L3 cache
    int64_t block_size_bytes = 500 * 1024;
    char* block_size = std::getenv("BODO_CROSS_JOIN_BLOCK_SIZE");
    if (block_size) {
        block_size_bytes = std::stoi(block_size);
    }
    if (block_size_bytes < 0) {
        throw std::runtime_error(
            "cross_join_table_local: block_size_bytes < 0");
    }

    int64_t n_left_blocks = (int64_t)std::ceil(
        table_local_memory_size(left_table) / (double)block_size_bytes);
    int64_t n_right_blocks = (int64_t)std::ceil(
        table_local_memory_size(right_table) / (double)block_size_bytes);

    int64_t left_block_n_rows =
        (int64_t)std::ceil(n_rows_left / (double)n_left_blocks);
    int64_t right_block_n_rows =
        (int64_t)std::ceil(n_rows_right / (double)n_right_blocks);

    for (int64_t b_left = 0; b_left < n_left_blocks; b_left++) {
        for (int64_t b_right = 0; b_right < n_right_blocks; b_right++) {
            int64_t left_block_start = b_left * left_block_n_rows;
            int64_t right_block_start = b_right * right_block_n_rows;
            for (int64_t i = left_block_start;
                 i < left_block_start + left_block_n_rows; i++) {
                for (int64_t j = right_block_start;
                     j < right_block_start + right_block_n_rows; j++) {
                    if ((i < (int64_t)n_rows_left) &&
                        (j < (int64_t)n_rows_right)) {
                        left_idxs.emplace_back(i);
                        right_idxs.emplace_back(j);
                    }
                }
            }
        }
    }

    table_info* out_left = RetrieveTable(left_table, left_idxs, -1);
    table_info* out_right = RetrieveTable(right_table, right_idxs, -1);
    std::vector<array_info*> out_arrs(out_left->columns);
    out_arrs.insert(out_arrs.end(), (out_right->columns).begin(),
                    (out_right->columns).end());
    return new table_info(out_arrs);
}

// design overview:
// https://bodo.atlassian.net/l/cp/Av2ijf9A
table_info* cross_join_table(table_info* left_table, table_info* right_table,
                             bool left_parallel, bool right_parallel,
                             uint64_t* num_rows_ptr) {
    cross_join_handle_dict_encoded(left_table, right_table, left_parallel,
                                   right_parallel);

    try {
        bool parallel_trace = (left_parallel || right_parallel);
        tracing::Event ev("cross_join_table", parallel_trace);
        table_info* out_table;

        // handle parallel cross join by broadcasting table chunks from every
        // rank
        if (left_parallel && right_parallel) {
            int n_pes, myrank;
            MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
            MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
            std::vector<table_info*> out_table_chunks;
            out_table_chunks.reserve(n_pes);

            // broadcast the smaller table to reduce overall communication
            int64_t left_table_size = table_global_memory_size(left_table);
            int64_t right_table_size = table_global_memory_size(right_table);
            bool left_table_bcast = left_table_size < right_table_size;
            table_info* bcast_table = left_table;
            table_info* other_table = right_table;
            if (!left_table_bcast) {
                bcast_table = right_table;
                other_table = left_table;
            }

            for (int p = 0; p < n_pes; p++) {
                // NOTE: broadcast_table steals a reference from bcast_table on
                // root rank. since all ranks become root in this loop
                // eventually, bcast_table is decrefed everywhere.
                table_info* bcast_table_chunk =
                    broadcast_table(bcast_table, bcast_table,
                                    bcast_table->ncols(), parallel_trace, p);

                // incref other table since needed in next iterations and
                // cross_join_table_local decrefs
                incref_table_arrays(other_table);
                table_info* out_table_chunk =
                    left_table_bcast
                        ? cross_join_table_local(bcast_table_chunk, other_table,
                                                 parallel_trace)
                        : cross_join_table_local(other_table, bcast_table_chunk,
                                                 parallel_trace);
                delete bcast_table_chunk;
                out_table_chunks.emplace_back(out_table_chunk);
            }
            decref_table_arrays(other_table);

            out_table = concat_tables(out_table_chunks);
        }
        // If either table is already replicated then broadcasting
        // isn't necessary (output's distribution will match the other input as
        // intended)
        else {
            out_table =
                cross_join_table_local(left_table, right_table, parallel_trace);
        }
        // NOTE: no need to delete table pointers since done in generated Python
        // code in join.py

        // number of local output rows is passed to Python in case all output
        // columns are dead.
        *num_rows_ptr = out_table->nrows();
        return out_table;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}
