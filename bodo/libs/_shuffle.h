// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include <mpi.h>
#include "_bodo_common.h"

struct mpi_comm_info {
    int n_pes;
    std::vector<array_info*> arrays;
    size_t n_rows;
    bool has_nulls;
    // generally required MPI counts
    std::vector<int> send_count;
    std::vector<int> recv_count;
    std::vector<int> send_disp;
    std::vector<int> recv_disp;
    // counts required for string arrays
    std::vector<std::vector<int>> send_count_char;
    std::vector<std::vector<int>> recv_count_char;
    std::vector<std::vector<int>> send_disp_char;
    std::vector<std::vector<int>> recv_disp_char;
    // counts for arrays with null bitmask
    std::vector<int> send_count_null;
    std::vector<int> recv_count_null;
    std::vector<int> send_disp_null;
    std::vector<int> recv_disp_null;
    size_t n_null_bytes;

    explicit mpi_comm_info(int _n_pes, std::vector<array_info*>& _arrays);

    void set_counts(uint32_t* hashes);
};

table_info* shuffle_table(table_info* in_table, int64_t n_keys);

table_info* shuffle_table_kernel(table_info* in_table, uint32_t* hashes,
                                 int n_pes, mpi_comm_info const& comm_info);

/** Getting the computing node on which a row belongs to
 *
 * The template paramter is T.
 * @param in_table: the input table
 * @param n_keys  : the number of keys to be used for the hash
 * @param n_pes   : the number of processor considered
 * @return the table containing a single column with the nodes
 */
table_info* compute_node_partition_by_hash(table_info* in_table, int64_t n_keys,
                                           int64_t n_pes);
