// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include "_array_hash.h"
#include "_array_utils.h"
#include "_bodo_common.h"

// If the test below is selected then many symbolic information are printed out.
// But it will not flood the logs even if you have millions of rows
#undef DEBUG_JOIN_SYMBOL
// This set of print statement can definitely flood your logs.
#undef DEBUG_JOIN_FULL

#ifdef DEBUG_JOIN_SYMBOL
#include <chrono>
#endif

table_info* hash_join_table(table_info* in_table, int64_t n_key_t,
                            int64_t n_data_left_t, int64_t n_data_right_t,
                            int64_t* vect_same_key,
                            int64_t* vect_need_typechange, bool is_left,
                            bool is_right, bool is_join, bool optional_col) {
#ifdef DEBUG_JOIN_SYMBOL
    std::chrono::time_point<std::chrono::system_clock> time1 =
        std::chrono::system_clock::now();
    std::cout << "IN_TABLE (hash_join_table):\n";
#ifdef DEBUG_JOIN_FULL
    DEBUG_PrintSetOfColumn(std::cout, in_table->columns);
#endif
    DEBUG_PrintRefct(std::cout, in_table->columns);
#endif
    size_t n_key = size_t(n_key_t);
    size_t n_data_left = size_t(n_data_left_t);
    size_t n_data_right = size_t(n_data_right_t);
    size_t n_tot_left = n_key + n_data_left;
    size_t n_tot_right = n_key + n_data_right;
    size_t sum_dim = 2 * n_key + n_data_left + n_data_right;
    size_t n_col = in_table->ncols();
    for (size_t iKey = 0; iKey < n_key; iKey++)
        CheckEqualityArrayType(in_table->columns[iKey],
                               in_table->columns[n_tot_left + iKey]);
    if (n_col != sum_dim) {
        Bodo_PyErr_SetString(PyExc_RuntimeError, "incoherent dimensions");
        return NULL;
    }
#ifdef DEBUG_JOIN_SYMBOL
    std::cout << "n_key_t=" << n_key_t << "\n";
    for (size_t iKey = 0; iKey < n_key; iKey++) {
        int64_t val = vect_same_key[iKey];
        std::cout << "iKey=" << iKey << "/" << n_key_t
                  << " vect_same_key[iKey]=" << val << "\n";
    }
    std::cout << "n_data_left_t=" << n_data_left_t
              << " n_data_right_t=" << n_data_right_t << "\n";
    std::cout << "is_left=" << is_left << " is_right=" << is_right << "\n";
    std::cout << "optional_col=" << optional_col << "\n";
#endif
    // in the case of merging on index and one column, it can only be one column
    if (n_key_t > 1 && optional_col) {
        Bodo_PyErr_SetString(
            PyExc_RuntimeError,
            "if optional_col=true then we must have n_key_t=1");
        return NULL;
    }
    // This is a hack because we may access vect_same_key_b above n_key
    // even if that is irrelevant to the computation.
    //
    size_t n_rows_left = (size_t)in_table->columns[0]->length;
    size_t n_rows_right = (size_t)in_table->columns[n_tot_left]->length;
    //
    std::vector<array_info*> key_arrs_left = std::vector<array_info*>(
        in_table->columns.begin(), in_table->columns.begin() + n_key);
    uint32_t seed = SEED_HASH_JOIN;
    uint32_t* hashes_left = hash_keys(key_arrs_left, seed);

    std::vector<array_info*> key_arrs_right = std::vector<array_info*>(
        in_table->columns.begin() + n_tot_left,
        in_table->columns.begin() + n_tot_left + n_key);
    uint32_t* hashes_right = hash_keys(key_arrs_right, seed);
#ifdef DEBUG_JOIN_FULL
    for (size_t i = 0; i < n_rows_left; i++)
        std::cout << "i=" << i << " hashes_left=" << hashes_left[i] << "\n";
    for (size_t i = 0; i < n_rows_right; i++)
        std::cout << "i=" << i << " hashes_right=" << hashes_right[i] << "\n";
#endif
    int ChoiceOpt;
    bool short_table_work,
        long_table_work;  // This corresponds to is_left/is_right
    size_t short_table_shift,
        long_table_shift;  // This corresponds to the shift for left and right.
    size_t short_table_rows, long_table_rows;  // the number of rows
    uint32_t *short_table_hashes, *long_table_hashes;
    double n_rows_left_d = double(n_rows_left);
    double n_rows_right_d = double(n_rows_right);
    double quot1 = n_rows_left_d / n_rows_right_d;
    double quot2 = n_rows_right_d / n_rows_left_d;
#ifdef DEBUG_JOIN_SYMBOL
    std::cout << "n_rows_left=" << n_rows_left
              << " n_rows_right=" << n_rows_right << "\n";
    std::cout << " quot1=" << quot1 << " quot2=" << quot2 << "\n";
#endif
    // In the computation of the join we are building some hash map on one of the tables.
    // Which one we choose influence the running time of course.
    // --- Selecting the smaller table is a good heuristic for the construction, the smaller the
    // table, the smaller the hash map.
    // --- Selecting one table for going into the hash-map while its values show up in the output
    // for example via an outer merge means that we need to keep track of which keys have been used
    // or not. This complexifies the data structure.
    //
    // Thus we use what we have to make the best choice about the computational technique.
    int MethodChoice;  // 1: by number of rows.
                       // 2: by the is_left / is_right values
    if (is_left == is_right) {
        // In that case we are doing either inner or outer merge
        // This means that the is_left / is_right argument do not apply.
        // Thus we simply have to use the number of rows for the choice
        MethodChoice = 1;
    } else {
        // In the case of is_left <> is_right we have a conflict regarding how to choose
        // We decide to set up an heuristic about this for making the decision
        double CritValue = 6.0;
        if (quot2 < CritValue && quot1 < CritValue)
            // In that case the large table is not so large comparable to the short one
            // This means that we can use the is_left / is_right for making the choice
            MethodChoice = 2;
        else
            // In that case one table is much larger than the other, therefore the choice by
            // the number of rows is the best here.
            MethodChoice = 1;
    }
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
        // short_table_work = false which is better
        if (is_right) {  // Thus is_left = false
            ChoiceOpt = 0;
        } else {
            ChoiceOpt = 1;
        }
    }
    // Now setting up pointers and values accordingly
#ifdef DEBUG_JOIN_SYMBOL
    std::cout << "MethodChoice=" << MethodChoice << " ChoiceOpt=" << ChoiceOpt
              << "\n";
#endif

    if (ChoiceOpt == 0) {
        // short = left and long = right
        short_table_work = is_left;
        long_table_work = is_right;
        short_table_shift = 0;
        long_table_shift = n_tot_left;
        short_table_rows = n_rows_left;
        long_table_rows = n_rows_right;
        short_table_hashes = hashes_left;
        long_table_hashes = hashes_right;
    } else {
        // short = right and long = left
        short_table_work = is_right;
        long_table_work = is_left;
        short_table_shift = n_tot_left;
        long_table_shift = 0;
        short_table_rows = n_rows_right;
        long_table_rows = n_rows_left;
        short_table_hashes = hashes_right;
        long_table_hashes = hashes_left;
    }
#ifdef DEBUG_JOIN_SYMBOL
    std::cout << "short_table_rows=" << short_table_rows
              << " long_table_rows=" << long_table_rows << "\n";
    std::cout << "short_table_work=" << short_table_work
              << " long_table_work=" << long_table_work << "\n";
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
    std::function<bool(size_t, size_t)> equal_fct = [&](size_t iRowA,
                                                        size_t iRowB) -> bool {
        size_t jRowA, jRowB;
        size_t shift_A, shift_B;
#ifdef DEBUG_JOIN_FULL
        std::cout << "Considered entries : ";
#endif
        if (iRowA < short_table_rows) {
            shift_A = short_table_shift;
            jRowA = iRowA;
#ifdef DEBUG_JOIN_FULL
            std::cout << "Short jRowA=" << jRowA << " ";
#endif
        } else {
            shift_A = long_table_shift;
            jRowA = iRowA - short_table_rows;
#ifdef DEBUG_JOIN_FULL
            std::cout << "Long jRowA=" << jRowA << " ";
#endif
        }
        if (iRowB < short_table_rows) {
            shift_B = short_table_shift;
            jRowB = iRowB;
#ifdef DEBUG_JOIN_FULL
            std::cout << "Short jRowB=" << jRowB << " ";
#endif
        } else {
            shift_B = long_table_shift;
            jRowB = iRowB - short_table_rows;
#ifdef DEBUG_JOIN_FULL
            std::cout << "Long jRowB=" << jRowB << " ";
#endif
        }
#ifdef DEBUG_JOIN_FULL
        std::cout << "\n";
#endif
        bool test =
            TestEqual(in_table->columns, n_key, shift_A, jRowA, shift_B, jRowB);
#ifdef DEBUG_JOIN_FULL
        std::cout << "After TestEqual call test=" << test << "\n";
#endif
        return test;
    };
    // The entList contains the identical keys with the corresponding rows.
    // We address the entry by the row index. We store all the rows which are
    // identical in the std::vector.
    // If we need also to do short_table_work then we store an index in the
    // first position of the std::vector
    UNORD_MAP_CONTAINER<size_t, std::vector<size_t>,
                        std::function<size_t(size_t)>,
                        std::function<bool(size_t, size_t)>>
        entList({}, hash_fct, equal_fct);
    //
    // ListPairWrite is the table used for the output
    // It precises the index used for the writing of the output table.
    std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> ListPairWrite;
    // Now running according to the short_table_work. The choice depends on the initial
    // choices that have been made.
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
#ifdef DEBUG_JOIN_FULL
            std::cout << "i_short=" << i_short << "\n";
#endif
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
        std::vector<uint8_t> V(n_bytes, 0);
        // We now iterate over all entries of the long table in order to get
        // the entries in the ListPairWrite.
#ifdef DEBUG_JOIN_SYMBOL
        size_t nb_from_long = 0, nb_short_long = 0;
#endif
        for (size_t i_long = 0; i_long < long_table_rows; i_long++) {
#ifdef DEBUG_JOIN_FULL
            std::cout << "i_long=" << i_long << "\n";
#endif
            size_t i_long_shift = i_long + short_table_rows;
            auto iter = entList.find(i_long_shift);
            if (iter == entList.end()) {
                if (long_table_work) {
#ifdef DEBUG_JOIN_SYMBOL
                    nb_from_long++;
#endif
                    ListPairWrite.push_back({-1, i_long});
                }
            } else {
                // If the short table entry are present in output as well, then
                // we need to keep track whether they are used or not by the
                // long table.
                size_t pos = iter->second[0];
                SetBitTo(V.data(), pos, true);
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
        for (auto& group : entList) {
            size_t pos = group.second[0];
            bool bit = GetBit(V.data(), pos);
            if (!bit) {
                for (size_t idx = 1; idx < group.second.size(); idx++) {
                    size_t j_short = group.second[idx];
                    ListPairWrite.push_back({j_short, -1});
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
#ifdef DEBUG_JOIN_FULL
            std::cout << "i_long=" << i_long << "\n";
#endif
            size_t i_long_shift = i_long + short_table_rows;
            auto iter = entList.find(i_long_shift);
            if (iter == entList.end()) {
                if (long_table_work) {
#ifdef DEBUG_JOIN_SYMBOL
                    nb_from_long++;
#endif
                    ListPairWrite.push_back({-1, i_long});
                }
            } else {
                // If the short table entry are present in output as well, then
                // we need to keep track whether they are used or not by the
                // long table.
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
#ifdef DEBUG_JOIN_SYMBOL
    size_t nbPair = ListPairWrite.size();
    std::cout << "nbPair=" << nbPair << "\n";
#ifdef DEBUG_JOIN_FULL
    for (size_t iPair = 0; iPair < nbPair; iPair++)
        std::cout << "iPair=" << iPair
                  << " ePair=" << ListPairWrite[iPair].first << " , "
                  << ListPairWrite[iPair].second << "\n";
#endif
#endif
    std::vector<array_info*> out_arrs;
    // Inserting the optional column in the case of merging on column and index.
    if (optional_col) {
        size_t i = 0;
        bool map_integer_type = false;
        if (ChoiceOpt == 0) {
            out_arrs.push_back(RetrieveArray_TwoColumns(in_table, ListPairWrite,
                                                        i, n_tot_left + i, 2,
                                                        map_integer_type));
        } else {
            out_arrs.push_back(RetrieveArray_TwoColumns(in_table, ListPairWrite,
                                                        n_tot_left + i, i, 2,
                                                        map_integer_type));
        }
    }
#ifdef DEBUG_JOIN_SYMBOL
    std::cout << "After optional_col construction optional_col=" << optional_col
              << "\n";
#endif
    // Inserting the Left side of the table
    int idx = 0;
    for (size_t i = 0; i < n_tot_left; i++) {
        if (i < n_key && vect_same_key[i < n_key ? i : 0] == 1) {
            // We are in the case of a key that has the same name on left and
            // right. This means that additional NaNs cannot happen.
            bool map_integer_type = false;
            if (ChoiceOpt == 0) {
                out_arrs.emplace_back(RetrieveArray_TwoColumns(
                    in_table, ListPairWrite, i, n_tot_left + i, 2,
                    map_integer_type));
            } else {
                out_arrs.emplace_back(RetrieveArray_TwoColumns(
                    in_table, ListPairWrite, n_tot_left + i, i, 2,
                    map_integer_type));
            }
        } else {
            // We are in data case or in the case of a key that is taken only
            // from one side. Therefore we have to plan for the possibility of
            // additional NaN.
            bool map_integer_type = vect_need_typechange[idx];
            if (ChoiceOpt == 0) {
                out_arrs.emplace_back(RetrieveArray_TwoColumns(
                    in_table, ListPairWrite, i, -1, 0, map_integer_type));
            } else {
                out_arrs.emplace_back(RetrieveArray_TwoColumns(
                    in_table, ListPairWrite, -1, i, 1, map_integer_type));
            }
        }
        idx++;
    }
#ifdef DEBUG_JOIN_SYMBOL
    std::cout << "After left side construction\n";
#endif
    // Inserting the right side of the table.
    for (size_t i = 0; i < n_tot_right; i++) {
        // There are two cases where we put the column in output:
        // ---It is a right key column with different name from the left.
        // ---It is a right data column
        if (i >= n_key ||
            (i < n_key && vect_same_key[i < n_key ? i : 0] == 0 && !is_join)) {
            bool map_integer_type = vect_need_typechange[idx];
            if (ChoiceOpt == 0) {
                out_arrs.emplace_back(RetrieveArray_TwoColumns(
                    in_table, ListPairWrite, -1, n_tot_left + i, 1,
                    map_integer_type));
            } else {
                out_arrs.emplace_back(RetrieveArray_TwoColumns(
                    in_table, ListPairWrite, n_tot_left + i, -1, 0,
                    map_integer_type));
            }
            idx++;
        }
    }
#ifdef DEBUG_JOIN_SYMBOL
    std::cout << "hash_join_table, output information\n";
#ifdef DEBUG_JOIN_FULL
    DEBUG_PrintSetOfColumn(std::cout, out_arrs);
#endif
    std::cout << "hash_join_table, output information\n";
    DEBUG_PrintRefct(std::cout, out_arrs);
    std::cout << "Finally leaving\n";
    std::chrono::time_point<std::chrono::system_clock> time2 =
        std::chrono::system_clock::now();
    std::cout << "Join C++ : time2 - time1="
              << std::chrono::duration_cast<std::chrono::milliseconds>(time2 -
                                                                       time1)
                     .count()
              << "\n";
#endif
    //
    delete[] hashes_left;
    delete[] hashes_right;
    return new table_info(out_arrs);
}
