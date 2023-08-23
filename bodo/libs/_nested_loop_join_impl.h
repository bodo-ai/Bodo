#pragma once

#include "_distributed.h"
#include "_memory.h"

/**
 * @brief Find unmatched outer join rows (using reduction over bit map if
 * necessary) and add them to list of output row indices.
 *
 * @param bit_map bitmap of matched rows
 * @param n_rows number of rows in input table
 * @param table_idxs indices in input table used for output generation
 * @param other_table_idxs indices in the other table used for output generation
 * @param needs_reduction : whether the bitmap needs a reduction (the
 * corresponding table is replicated, but the other table is distributed).
 * @param offset number of bits from the start of bit_map that belongs to
 * previous chunks. Default is 0
 */
template <typename BitMapAllocator>
void add_unmatched_rows(bodo::vector<uint8_t, BitMapAllocator>& bit_map,
                        size_t n_rows, bodo::vector<int64_t>& table_idxs,
                        bodo::vector<int64_t>& other_table_idxs,
                        bool needs_reduction, int64_t offset) {
    if (needs_reduction) {
        int n_pes, myrank;
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        MPI_Allreduce_bool_or(bit_map);
        int pos = 0;
        for (size_t i = 0; i < n_rows; i++) {
            bool bit = GetBit(bit_map.data(), i + offset);
            // distribute the replicated input table rows across ranks
            // to load balance the output
            if (!bit) {
                int node = pos % n_pes;
                if (node == myrank) {
                    table_idxs.emplace_back(i);
                    other_table_idxs.emplace_back(-1);
                }
                pos++;
            }
        }
    } else {
        for (size_t i = 0; i < n_rows; i++) {
            bool bit = GetBit(bit_map.data(), i + offset);
            if (!bit) {
                table_idxs.emplace_back(i);
                other_table_idxs.emplace_back(-1);
            }
        }
    }
}
