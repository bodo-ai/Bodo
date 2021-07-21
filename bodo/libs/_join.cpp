// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include "_array_hash.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_distributed.h"
#include "_shuffle.h"

// If the test below is selected then many symbolic information are printed out.
// But it will not flood the logs even if you have millions of rows
#undef DEBUG_JOIN_SYMBOL
// This set of print statement can definitely flood your logs.
#undef DEBUG_JOIN_FULL

#ifdef DEBUG_JOIN_SYMBOL
#include <chrono>
#endif

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

table_info* hash_join_table(table_info* left_table, table_info* right_table,
                            bool left_parallel, bool right_parallel,
                            int64_t n_key_t, int64_t n_data_left_t,
                            int64_t n_data_right_t, int64_t* vect_same_key,
                            int64_t* vect_need_typechange, bool is_left,
                            bool is_right, bool is_join, bool optional_col,
                            bool indicator, bool is_na_equal) {
    try {
        // Reading the MPI settings
        int n_pes, myrank;
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        // Doing checks and basic assignments.
#ifdef DEBUG_JOIN_SYMBOL
        std::chrono::time_point<std::chrono::system_clock> time1 =
            std::chrono::system_clock::now();
        std::cout << "---------------------------------------------------------"
                     "--------------\n";
        std::cout << "hash_join_table. LEFT_TABLE):\n";
        DEBUG_PrintRefct(std::cout, left_table->columns);
#ifdef DEBUG_JOIN_FULL
        DEBUG_PrintSetOfColumn(std::cout, left_table->columns);
#endif
        std::cout << "hash_join_table. RIGHT_TABLE):\n";
        DEBUG_PrintRefct(std::cout, right_table->columns);
#ifdef DEBUG_JOIN_FULL
        DEBUG_PrintSetOfColumn(std::cout, right_table->columns);
#endif
#endif
        size_t n_key = size_t(n_key_t);
        size_t n_data_left = size_t(n_data_left_t);
        size_t n_data_right = size_t(n_data_right_t);
        size_t n_tot_left = n_key + n_data_left;
        size_t n_tot_right = n_key + n_data_right;
        for (size_t iKey = 0; iKey < n_key; iKey++)
            CheckEqualityArrayType(left_table->columns[iKey],
                                   right_table->columns[iKey]);
#ifdef DEBUG_JOIN_SYMBOL
        if (size_t(left_table->ncols()) != n_tot_left) {
            throw std::runtime_error(
                "Error in join.cpp::hash_join_table: incoherent dimensions for "
                "left tabče.");
        }
        if (size_t(right_table->ncols()) != n_tot_right) {
            throw std::runtime_error(
                "Error in join.cpp::hash_join_table: incoherent dimensions for "
                "right tabče.");
        }
        std::cout << "n_key_t=" << n_key_t << "\n";
        for (size_t iKey = 0; iKey < n_key; iKey++) {
            int64_t val = vect_same_key[iKey];
            std::cout << "iKey=" << iKey << "/" << n_key_t
                      << " vect_same_key[iKey]=" << val << "\n";
        }
        std::cout << "left_parallel=" << left_parallel
                  << " right_parallel=" << right_parallel << "\n";
        std::cout << "n_data_left_t=" << n_data_left_t
                  << " n_data_right_t=" << n_data_right_t << "\n";
        std::cout << "is_left=" << is_left << " is_right=" << is_right << "\n";
        std::cout << "optional_col=" << optional_col << "\n";
#endif
        // in the case of merging on index and one column, it can only be one
        // column
        if (n_key_t > 1 && optional_col) {
            throw std::runtime_error(
                "Error in join.cpp::hash_join_table: if optional_col=true then "
                "we must have n_key_t=1.");
        }
        //
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
        // Both algorithm requires the construction of a hash map for the keys.
        //
        table_info *work_left_table, *work_right_table;
        bool free_work_left = false, free_work_right = false;
        bool left_replicated, right_replicated;
        if (left_parallel && right_parallel) {
            // Only if both tables are parallel is something needed
            int64_t left_total_memory = table_global_memory_size(left_table);
            int64_t right_total_memory = table_global_memory_size(right_table);
            int CritMemorySize = 10 * 1024 * 1024;  // in bytes
            char* bcast_threshold = std::getenv("BODO_BCAST_JOIN_THRESHOLD");
            if (bcast_threshold) CritMemorySize = std::stoi(bcast_threshold);
            if (CritMemorySize < 0)
                throw std::runtime_error("hash_join: CritMemorySize < 0");
            bool all_gather = true;
            if (left_total_memory < right_total_memory &&
                left_total_memory < CritMemorySize) {
                work_left_table = gather_table(left_table, -1, all_gather);
                free_work_left = true;
                work_right_table = right_table;
                left_replicated = true;
                right_replicated = false;
            } else if (right_total_memory <= left_total_memory &&
                       right_total_memory < CritMemorySize) {
                work_left_table = left_table;
                work_right_table = gather_table(right_table, -1, all_gather);
                free_work_right = true;
                left_replicated = false;
                right_replicated = true;
            } else {
                work_left_table =
                    coherent_shuffle_table(left_table, right_table, n_key);
                work_right_table =
                    coherent_shuffle_table(right_table, left_table, n_key);
                free_work_left = true;
                free_work_right = true;
                left_replicated = false;
                right_replicated = false;
            }
        } else {
            work_left_table = left_table;
            work_right_table = right_table;
            left_replicated = !left_parallel;
            right_replicated = !right_parallel;
        }
#ifdef DEBUG_JOIN_SYMBOL
        std::cout << "left_replicated=" << left_replicated
                  << " right_replicated=" << right_replicated << "\n";
#endif
        size_t n_rows_left = work_left_table->nrows();
        size_t n_rows_right = work_right_table->nrows();
#ifdef DEBUG_JOIN_SYMBOL
        std::cout << "hash_join_table. WORK LEFT_TABLE):\n";
        DEBUG_PrintRefct(std::cout, work_left_table->columns);
#ifdef DEBUG_JOIN_FULL
        DEBUG_PrintSetOfColumn(std::cout, work_left_table->columns);
#endif
        std::cout << "hash_join_table. WORK_RIGHT_TABLE):\n";
        DEBUG_PrintRefct(std::cout, work_right_table->columns);
#ifdef DEBUG_JOIN_FULL
        DEBUG_PrintSetOfColumn(std::cout, work_right_table->columns);
#endif
#endif
        //
        // Now computing the hashes that will be used in the hash map.
        //
        uint32_t* hashes_left =
            hash_keys_table(work_left_table, n_key, SEED_HASH_JOIN);
        uint32_t* hashes_right =
            hash_keys_table(work_right_table, n_key, SEED_HASH_JOIN);
        double quot1 = double(n_rows_left) / double(n_rows_right);
        double quot2 = double(n_rows_right) / double(n_rows_left);
#ifdef DEBUG_JOIN_SYMBOL
        std::cout << "n_rows_left=" << n_rows_left
                  << " n_rows_right=" << n_rows_right << "\n";
        std::cout << " quot1=" << quot1 << " quot2=" << quot2 << "\n";
#endif
        // In the computation of the join we are building some hash map on one
        // of the tables. Which one we choose influence the running time of
        // course.
        // --- Selecting the smaller table is a good heuristic for the
        // construction, the smaller the table, the smaller the hash map.
        // --- Selecting one table for going into the hash-map while its values
        // show up in the output for example via an outer merge means that we
        // need to keep track of which keys have been used or not. This
        // complexifies the data structure.
        //
        // Thus we use what we have to make the best choice about the
        // computational technique.
        int MethodChoice;  // 1: by number of rows.
                           // 2: by the is_left / is_right values
        if (is_left == is_right) {
            // In that case we are doing either inner or outer merge
            // This means that the is_left / is_right argument do not apply.
            // Thus we simply have to use the number of rows for the choice
            MethodChoice = 1;
        } else {
            // In the case of is_left <> is_right we have a conflict regarding
            // how to choose We decide to set up an heuristic about this for
            // making the decision
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
        // We have chosen our method of choice and now we implement it.
        // Both choice method are considered in sequence.
        int ChoiceOpt;
        if (MethodChoice == 1) {
            // We choose by the number of rows.
            if (n_rows_left < n_rows_right) {
                ChoiceOpt = 0;
            } else {
                ChoiceOpt = 1;
            }
        } else {
            // This case may happen only if is_left <> is_right
            // Thus we can make a choice such that
            // short_table_work = false which led to a simpler hash map.
            if (is_right) {  // Thus is_left = false
                ChoiceOpt = 0;
            } else {
                ChoiceOpt = 1;
            }
        }
        // Now setting up pointers and values accordingly
#ifdef DEBUG_JOIN_SYMBOL
        std::cout << "MethodChoice=" << MethodChoice
                  << " ChoiceOpt=" << ChoiceOpt << "\n";
#endif
        // Now that we have made a choice opt, we set what is the "short" table
        // and the "long" one. For the short table we construct a hash map, for
        // the long table, we simply iterate over the rows and see if the keys
        // are in the hash map.
        size_t short_table_rows, long_table_rows;  // the number of rows
        uint32_t *short_table_hashes, *long_table_hashes;
        // This corresponds to is_left/is_right
        bool short_table_work, long_table_work;
        table_info *short_table, *long_table;
        bool short_replicated, long_replicated;
        if (ChoiceOpt == 0) {
            // short = left and long = right
            short_table_work = is_left;
            long_table_work = is_right;
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
            short_table_work = is_right;
            long_table_work = is_left;
            short_table = work_right_table;
            long_table = work_left_table;
            short_table_rows = n_rows_right;
            long_table_rows = n_rows_left;
            short_table_hashes = hashes_right;
            long_table_hashes = hashes_left;
            short_replicated = right_replicated;
            long_replicated = left_replicated;
        }
        // If one table (say left one) is replicated and the other is
        // distributed and is_left=T then we need to keep track of how the rows
        // of the left table are matched.
        bool parallel_track_miss_short =
            short_table_work && short_replicated && !long_replicated;
        bool parallel_track_miss_long =
            long_table_work && long_replicated && !short_replicated;
#ifdef DEBUG_JOIN_SYMBOL
        std::cout << "parallel_track_miss_short=" << parallel_track_miss_short
                  << "\n";
        std::cout << "parallel_track_miss_long=" << parallel_track_miss_long
                  << "\n";
        std::cout << "short_table_rows=" << short_table_rows
                  << " long_table_rows=" << long_table_rows << "\n";
#endif
        /* This is a function for comparing the rows.
         * This is the first lambda used as argument for the unordered map
         * container.
         *
         * rows can be in the left or the right tables.
         * If iRow < short_table_rows then it is in the first table.
         * If iRow >= short_table_rows then it is in the second table.
         *
         * Note that the hash is size_t (so 8 bytes on x86-64) while
         * the hashes array are int32_t (so 4 bytes)
         *
         * @param iRow is the first row index for the comparison
         * @return true/false depending on the case.
         */
        std::function<size_t(size_t)> hash_fct = [&](size_t iRow) -> size_t {
            if (iRow < short_table_rows)
                return short_table_hashes[iRow];
            else
                return long_table_hashes[iRow - short_table_rows];
        };
        /* This is a function for testing equality of rows.
         * This is used as second argument for the unordered map container.
         *
         * rows can be in the left or the right tables.
         * If iRow < short_table_rows then it is in the first table.
         * If iRow >= short_table_rows then it is in the second table.
         *
         * @param iRowA is the first row index for the comparison
         * @param iRowB is the second row index for the comparison
         * @return true/false depending on equality or not.
         */
        std::function<bool(size_t, size_t)> equal_fct =
            [&](size_t iRowA, size_t iRowB) -> bool {
            size_t jRowA, jRowB;
            table_info *table_A, *table_B;
            if (iRowA < short_table_rows) {
                table_A = short_table;
                jRowA = iRowA;
            } else {
                table_A = long_table;
                jRowA = iRowA - short_table_rows;
            }
            if (iRowB < short_table_rows) {
                table_B = short_table;
                jRowB = iRowB;
            } else {
                table_B = long_table;
                jRowB = iRowB - short_table_rows;
            }
            bool test = TestEqualJoin(table_A, table_B, jRowA, jRowB, n_key,
                                      is_na_equal);
            return test;
        };
        // The entList contains the identical keys with the corresponding rows.
        // We address the entry by the row index. We store all the rows which
        // are identical in the std::vector. If we need also to do
        // short_table_work then we store an index in the first position of the
        // std::vector
        UNORD_MAP_CONTAINER<size_t, std::vector<size_t>,
                            std::function<size_t(size_t)>,
                            std::function<bool(size_t, size_t)>>
            entList({}, hash_fct, equal_fct);
        //
        // ListPairWrite is the table used for the output
        // It precises the index used for the writing of the output table.
        std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> ListPairWrite;
        // Now running according to the short_table_work. The choice depends on
        // the initial choices that have been made.
        size_t n_bytes_long;
        if (parallel_track_miss_long) {
            n_bytes_long = (long_table_rows + 7) >> 3;
        } else {
            n_bytes_long = 0;
        }
        // V_long_map and V_short_map takes similar roles.
        // They indicate if an entry in the short or long table
        // has been matched to the other side.
        // This is needed only if said table in replicated which occurs
        // in the case of the broadcast join for example.
        std::vector<uint8_t> V_long_map(n_bytes_long, 255);
        if (short_table_work) {
#ifdef DEBUG_JOIN_SYMBOL
            std::cout << "SECOND SCHEME when we have short entries=true\n";
#endif
            // The loop over the short table.
            // entries are stored one by one and all of them are put even if
            // identical in value.
            // The first entry is going to be the index for the boolean array.
            // This code path will be selected whenever we have an OUTER merge.
            size_t pos_idx = 0;
            for (size_t i_short = 0; i_short < short_table_rows; i_short++) {
                std::vector<size_t>& group = entList[i_short];
                if (group.size() == 0) {
                    group.emplace_back(pos_idx);
                    pos_idx++;
                }
                group.emplace_back(i_short);
            }
#ifdef DEBUG_JOIN_SYMBOL
            size_t nEnt = entList.size();
            double uniq_frac = double(nEnt) / double(short_table_rows);
            std::cout << "nEnt=" << nEnt << " uniq_frac=" << uniq_frac << "\n";
            if (nEnt != pos_idx) {
                std::cout << "nEnt=" << nEnt << " pos_idx=" << pos_idx << "\n";
                std::cout << "Error between nEnt and pos_idx\n";
            }
#endif
            //
            // Now iterating and determining how many entries we have to do.
            //
            size_t n_bytes = (pos_idx + 7) >> 3;
            std::vector<uint8_t> V_short_map(n_bytes, 0);
            // We now iterate over all entries of the long table in order to get
            // the entries in the ListPairWrite.
#ifdef DEBUG_JOIN_SYMBOL
            size_t nb_from_long = 0, nb_short_long = 0;
#endif
            for (size_t i_long = 0; i_long < long_table_rows; i_long++) {
                size_t i_long_shift = i_long + short_table_rows;
                auto iter = entList.find(i_long_shift);
                if (iter == entList.end()) {
                    if (long_table_work) {
#ifdef DEBUG_JOIN_SYMBOL
                        nb_from_long++;
#endif
                        if (parallel_track_miss_long) {
                            SetBitTo(V_long_map.data(), i_long, false);
                        } else {
                            ListPairWrite.push_back({-1, i_long});
                        }
                    }
                } else {
                    // If the short table entry are present in output as well,
                    // then we need to keep track whether they are used or not
                    // by the long table.
                    size_t pos = iter->second[0];
                    SetBitTo(V_short_map.data(), pos, true);
                    for (size_t idx = 1; idx < iter->second.size(); idx++) {
                        size_t j_short = iter->second[idx];
                        ListPairWrite.push_back({j_short, i_long});
#ifdef DEBUG_JOIN_SYMBOL
                        nb_short_long++;
#endif
                    }
                }
            }
            // if short_table is in output then we need to check
            // if they are used by the long table and if so use them on output.
#ifdef DEBUG_JOIN_SYMBOL
            size_t nb_from_short = 0;
#endif
            if (parallel_track_miss_short) MPI_Allreduce_bool_or(V_short_map);
            int pos_short_disp = 0;
            for (auto& group : entList) {
                size_t pos = group.second[0];
                bool bit = GetBit(V_short_map.data(), pos);
                if (!bit) {
                    for (size_t idx = 1; idx < group.second.size(); idx++) {
                        size_t j_short = group.second[idx];
                        // For parallel_track_miss_short=T the output table is
                        // distributed. Since the table in input is replicated,
                        // we dispatch it by rank.
                        if (parallel_track_miss_short) {
                            int node = pos_short_disp % n_pes;
                            if (node == myrank)
                                ListPairWrite.push_back({j_short, -1});
                            pos_short_disp++;
                        } else {
                            ListPairWrite.push_back({j_short, -1});
                        }
#ifdef DEBUG_JOIN_SYMBOL
                        nb_from_short++;
#endif
                    }
                }
            }
#ifdef DEBUG_JOIN_SYMBOL
            std::cout << "nb_from_short=" << nb_from_short
                      << " nb_from_long=" << nb_from_long
                      << " nb_short_long=" << nb_short_long << "\n";
#endif
        } else {
#ifdef DEBUG_JOIN_SYMBOL
            std::cout << "FIRST SCHEME when we have short entries=false\n";
#endif
            // The loop over the short table.
            // entries are stored one by one and all of them are put even if
            // identical in value.
            // No need to keep track of the usage of the short table.
            // This code path is selected whenever INNER is true.
            for (size_t i_short = 0; i_short < short_table_rows; i_short++) {
                std::vector<size_t>& group = entList[i_short];
                group.emplace_back(i_short);
            }
#ifdef DEBUG_JOIN_SYMBOL
            size_t nEnt = entList.size();
            double uniq_frac = double(nEnt) / double(short_table_rows);
            std::cout << "nEnt=" << nEnt << " uniq_frac=" << uniq_frac << "\n";
#endif
            //
            // Now iterating and determining how many entries we have to do.
            //

            // We now iterate over all entries of the long table in order to get
            // the entries in the ListPairWrite.
#ifdef DEBUG_JOIN_SYMBOL
            size_t nb_from_long = 0, nb_short_long = 0;
#endif
            for (size_t i_long = 0; i_long < long_table_rows; i_long++) {
                size_t i_long_shift = i_long + short_table_rows;
                auto iter = entList.find(i_long_shift);
                if (iter == entList.end()) {
                    if (long_table_work) {
#ifdef DEBUG_JOIN_SYMBOL
                        nb_from_long++;
#endif
                        if (parallel_track_miss_long) {
                            SetBitTo(V_long_map.data(), i_long, false);
                        } else {
                            ListPairWrite.push_back({-1, i_long});
                        }
                    }
                } else {
                    // If the short table entry are present in output as well,
                    // then we need to keep track whether they are used or not
                    // by the long table.
                    for (auto& j_short : iter->second) {
                        ListPairWrite.push_back({j_short, i_long});
#ifdef DEBUG_JOIN_SYMBOL
                        nb_short_long++;
#endif
                    }
                }
            }
#ifdef DEBUG_JOIN_SYMBOL
            std::cout << "nb_from_long=" << nb_from_long
                      << " nb_short_long=" << nb_short_long << "\n";
#endif
        }
        // In replicated case, we put the long rows in distributed output
        if (long_table_work && parallel_track_miss_long) {
            MPI_Allreduce_bool_or(V_long_map);
            int pos = 0;
            for (size_t i_long = 0; i_long < long_table_rows; i_long++) {
                bool bit = GetBit(V_long_map.data(), i_long);
                // The replicated input table is dispatched over the rows in the
                // distributed output.
                if (!bit) {
                    int node = pos % n_pes;
                    if (node == myrank) ListPairWrite.push_back({-1, i_long});
                    pos++;
                }
            }
        }
        std::vector<array_info*> out_arrs;
        // Computing the last time at which a column is used.
        // This is for the call to decref_array.
        // There are many cases to cover, so it is better to preprocess
        // first to determine last usage before creating the return
        // table.
        std::vector<uint8_t> last_col_use_left(n_tot_left, 0);
        std::vector<uint8_t> last_col_use_right(n_tot_right, 0);
        if (optional_col) {
            size_t i = 0;
            last_col_use_left[i] = 1;
            last_col_use_right[i] = 1;
        }
        for (size_t i = 0; i < n_tot_left; i++) {
            if (i < n_key && vect_same_key[i < n_key ? i : 0] == 1) {
                last_col_use_left[i] = 2;
                last_col_use_right[i] = 2;
            } else {
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
                last_col_use_right[i] = 4;
            }
        }
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

        // Inserting the optional column in the case of merging on column and
        // index.
        if (optional_col) {
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
            if (last_col_use_left[i] == 1) {
                decref_array(left_arr);
            }
            if (last_col_use_right[i] == 1) {
                decref_array(right_arr);
            }
        }
#ifdef DEBUG_JOIN_SYMBOL
        std::cout << "After optional_col construction optional_col="
                  << optional_col << "\n";
#endif

        // Inserting the Left side of the table
        int idx = 0;
        for (size_t i = 0; i < n_tot_left; i++) {
            if (i < n_key && vect_same_key[i < n_key ? i : 0] == 1) {
                // We are in the case of a key that has the same name on left
                // and right. This means that additional NaNs cannot happen.
                bool map_integer_type = false;
                array_info* left_arr = work_left_table->columns[i];
                array_info* right_arr = work_right_table->columns[i];
                if (ChoiceOpt == 0) {
                    out_arrs.emplace_back(RetrieveArray_TwoColumns(
                        left_arr, right_arr, ListPairWrite, 2,
                        map_integer_type));
                } else {
                    out_arrs.emplace_back(RetrieveArray_TwoColumns(
                        right_arr, left_arr, ListPairWrite, 2,
                        map_integer_type));
                }
                if (last_col_use_left[i] == 2) {
                    decref_array(left_arr);
                }
                if (last_col_use_right[i] == 2) {
                    decref_array(right_arr);
                }
            } else {
                // We are in data case or in the case of a key that is taken
                // only from one side. Therefore we have to plan for the
                // possibility of additional NaN.
                bool map_integer_type = vect_need_typechange[idx];
                array_info* left_arr = work_left_table->columns[i];
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
                if (last_col_use_left[i] == 3) {
                    decref_array(left_arr);
                }
            }
            idx++;
        }
        // Inserting the right side of the table.
        for (size_t i = 0; i < n_tot_right; i++) {
            // There are two cases where we put the column in output:
            // ---It is a right key column with different name from the left.
            // ---It is a right data column
            if (i >= n_key ||
                (i < n_key && vect_same_key[i < n_key ? i : 0] == 0 &&
                 !is_join)) {
                bool map_integer_type = vect_need_typechange[idx];
                array_info* left_arr = nullptr;
                array_info* right_arr = work_right_table->columns[i];
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
                if (last_col_use_right[i] == 4) {
                    decref_array(right_arr);
                }
            }
        }
        // Create indicator column if indicator=True
        size_t num_rows = ListPairWrite.size();
        if (indicator) {
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

        if (free_work_left) {
            delete_table(work_left_table);
        }
        if (free_work_right) {
            delete_table(work_right_table);
        }
#ifdef DEBUG_JOIN_SYMBOL
        std::cout << "hash_join_table, output information. OUT_ARRS:\n";
        DEBUG_PrintRefct(std::cout, out_arrs);
#ifdef DEBUG_JOIN_FULL
        DEBUG_PrintSetOfColumn(std::cout, out_arrs);
#endif
        std::cout << "Finally leaving\n";
        std::chrono::time_point<std::chrono::system_clock> time2 =
            std::chrono::system_clock::now();
        std::cout << "Join C++ : time2 - time1="
                  << std::chrono::duration_cast<std::chrono::milliseconds>(
                         time2 - time1)
                         .count()
                  << "\n";
        std::cout << "---------------------------------------------------------"
                     "-----\n";
#endif
        //
        delete[] hashes_left;
        delete[] hashes_right;
        return new table_info(out_arrs);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}
