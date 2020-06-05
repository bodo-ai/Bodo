// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include "_array_hash.h"
#include "_array_utils.h"
#include "_bodo_common.h"

#undef DEBUG_JOIN

table_info* hash_join_table(table_info* in_table, int64_t n_key_t,
                            int64_t n_data_left_t, int64_t n_data_right_t,
                            int64_t* vect_same_key, int64_t* vect_need_typechange,
                            bool is_left, bool is_right,
                            bool is_join, bool optional_col) {
#ifdef DEBUG_JOIN
    std::cout << "IN_TABLE (hash_join_table):\n";
    DEBUG_PrintSetOfColumn(std::cout, in_table->columns);
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
#ifdef DEBUG_JOIN
    std::cout << "n_key_t=" << n_key_t << "\n";
    for (size_t iKey = 0; iKey < n_key; iKey++) {
        int64_t val = vect_same_key[iKey];
        std::cout << "iKey=" << iKey << "/" << n_key_t << " vect_same_key[iKey]=" << val << "\n";
    }
    std::cout << "n_data_left_t=" << n_data_left_t
              << " n_data_right_t=" << n_data_right_t << "\n";
    std::cout << "is_left=" << is_left << " is_right=" << is_right << "\n";
    std::cout << "optional_col=" << optional_col << "\n";
#endif
    // in the case of merging on index and one column, it can only be one column
    if (n_key_t > 1 && optional_col) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
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
    uint32_t seed = 0xb0d01288;
    uint32_t* hashes_left = hash_keys(key_arrs_left, seed);

    std::vector<array_info*> key_arrs_right = std::vector<array_info*>(
        in_table->columns.begin() + n_tot_left,
        in_table->columns.begin() + n_tot_left + n_key);
    uint32_t* hashes_right = hash_keys(key_arrs_right, seed);
#ifdef DEBUG_JOIN
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
#ifdef DEBUG_JOIN
    std::cout << "n_rows_left=" << n_rows_left
              << " n_rows_right=" << n_rows_right << "\n";
#endif
    if (n_rows_left < n_rows_right) {
        ChoiceOpt = 0;
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
        ChoiceOpt = 1;
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
#ifdef DEBUG_JOIN
    std::cout << "ChoiceOpt=" << ChoiceOpt << "\n";
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
    std::function<bool(size_t, size_t)> equal_fct = [&](size_t iRowA,
                                                        size_t iRowB) -> bool {
        size_t jRowA, jRowB;
        size_t shift_A, shift_B;
        if (iRowA < short_table_rows) {
            shift_A = short_table_shift;
            jRowA = iRowA;
        } else {
            shift_A = long_table_shift;
            jRowA = iRowA - short_table_rows;
        }
        if (iRowB < short_table_rows) {
            shift_B = short_table_shift;
            jRowB = iRowB;
        } else {
            shift_B = long_table_shift;
            jRowB = iRowB - short_table_rows;
        }
        return TestEqual(in_table->columns, n_key, shift_A, jRowA, shift_B,
                         jRowB);
    };
    // The entList contains the hash of the short table.
    // We address the entry by the row index. We store all the rows which are
    // identical in the std::vector.
    MAP_CONTAINER<size_t, std::vector<size_t>, std::function<size_t(size_t)>,
                  std::function<bool(size_t, size_t)>>
        entList({}, hash_fct, equal_fct);
    // The loop over the short table.
    // entries are stored one by one and all of them are put even if identical
    // in value.
    for (size_t i_short = 0; i_short < short_table_rows; i_short++) {
#ifdef DEBUG_JOIN
        std::cout << "i_short=" << i_short << "\n";
#endif
        std::vector<size_t>& group = entList[i_short];
        group.emplace_back(i_short);
    }
    size_t nEnt = entList.size();
#ifdef DEBUG_JOIN
    std::cout << "nEnt=" << nEnt << "\n";
#endif
    //
    // Now iterating and determining how many entries we have to do.
    //

    //
    // ListPairWrite is the table used for the output
    // It precises the index used for the writing of the output table.
    std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> ListPairWrite;
    // This precise whether a short table entry has been used or not.
    std::vector<int> ListStatus(nEnt, 0);
    // We now iterate over all entries of the long table in order to get
    // the entries in the ListPairWrite.
    for (size_t i_long = 0; i_long < long_table_rows; i_long++) {
        size_t i_long_shift = i_long + short_table_rows;
        auto iter = entList.find(i_long_shift);
        if (iter == entList.end()) {
            if (long_table_work) ListPairWrite.push_back({-1, i_long});
        } else {
            // If the short table entry are present in output as well, then
            // we need to keep track whether they are used or not by the long
            // table.
            if (short_table_work) {
                auto index = std::distance(entList.begin(), iter);
                ListStatus[index] = 1;
            }
            for (auto& j_short : iter->second)
                ListPairWrite.push_back({j_short, i_long});
        }
    }
    // if short_table is in output then we need to check
    // if they are used by the long table and if so use them on output.
    if (short_table_work) {
        auto iter = entList.begin();
        size_t iter_s = 0;
        while (iter != entList.end()) {
            if (ListStatus[iter_s] == 0) {
                for (auto& j_short : iter->second) {
                    ListPairWrite.push_back({j_short, -1});
                }
            }
            iter++;
            iter_s++;
        }
#ifdef DEBUG_JOIN
        std::cout << "AFTER : iter_s=" << iter_s << "\n";
#endif
    }
#ifdef DEBUG_JOIN
    size_t nbPair = ListPairWrite.size();
    for (size_t iPair = 0; iPair < nbPair; iPair++)
        std::cout << "iPair=" << iPair
                  << " ePair=" << ListPairWrite[iPair].first << " , "
                  << ListPairWrite[iPair].second << "\n";
#endif
    std::vector<array_info*> out_arrs;
    // Inserting the optional column in the case of merging on column and index.
    if (optional_col) {
        size_t i = 0;
        bool map_integer_type = false;
        if (ChoiceOpt == 0) {
            out_arrs.push_back(
                RetrieveArray(in_table, ListPairWrite, i, n_tot_left + i, 2, map_integer_type));
        } else {
            out_arrs.push_back(
                RetrieveArray(in_table, ListPairWrite, n_tot_left + i, i, 2, map_integer_type));
        }
    }
#ifdef DEBUG_JOIN
    std::cout << "After optional_col construction optional_col=" << optional_col << "\n";
#endif
    // Inserting the Left side of the table
    int idx=0;
    for (size_t i = 0; i < n_tot_left; i++) {
        if (i < n_key && vect_same_key[i < n_key ? i : 0] == 1) {
            // We are in the case of a key that has the same name on left and right.
            // This means that additional NaNs cannot happen.
            bool map_integer_type=false;
            if (ChoiceOpt == 0) {
                out_arrs.emplace_back(
                    RetrieveArray(in_table, ListPairWrite, i, n_tot_left + i, 2, map_integer_type));
            } else {
                out_arrs.emplace_back(
                    RetrieveArray(in_table, ListPairWrite, n_tot_left + i, i, 2, map_integer_type));
            }
        } else {
            // We are in data case or in the case of a key that is taken only from one side.
            // Therefore we have to plan for the possibility of additional NaN.
            bool map_integer_type = vect_need_typechange[idx];
            if (ChoiceOpt == 0) {
                out_arrs.emplace_back(
                    RetrieveArray(in_table, ListPairWrite, i, -1, 0, map_integer_type));
            } else {
                out_arrs.emplace_back(
                    RetrieveArray(in_table, ListPairWrite, -1, i, 1, map_integer_type));
            }
        }
        idx++;
    }
#ifdef DEBUG_JOIN
    std::cout << "After left side construction\n";
#endif
    // Inserting the right side of the table.
    for (size_t i = 0; i < n_tot_right; i++) {
        // There are two cases where we put the column in output:
        // ---It is a right key column with different name from the left.
        // ---It is a right data column
        if (i >= n_key || (i < n_key && vect_same_key[i < n_key ? i : 0] == 0 && !is_join)) {
            bool map_integer_type = vect_need_typechange[idx];
            if (ChoiceOpt == 0) {
                out_arrs.emplace_back(
                    RetrieveArray(in_table, ListPairWrite, -1, n_tot_left + i, 1, map_integer_type));
            } else {
                out_arrs.emplace_back(
                    RetrieveArray(in_table, ListPairWrite, n_tot_left + i, -1, 0, map_integer_type));
            }
            idx++;
        }
    }
#ifdef DEBUG_JOIN
    std::cout << "hash_join_table, output information\n";
    DEBUG_PrintSetOfColumn(std::cout, out_arrs);
    std::cout << "hash_join_table, output information\n";
    DEBUG_PrintRefct(std::cout, out_arrs);
    std::cout << "Finally leaving\n";
#endif
    //
    delete[] hashes_left;
    delete[] hashes_right;
    return new table_info(out_arrs);
}
