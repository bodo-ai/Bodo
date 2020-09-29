// Copyright (C) 2019 Bodo Inc. All rights reserved.
#ifndef _SHUFFLE_H_INCLUDED
#define _SHUFFLE_H_INCLUDED

#include <mpi.h>
#include "_bodo_common.h"

#define A2AV_LARGE_DTYPE_SIZE 1024
extern MPI_Datatype a2av_large_dtype;

/**
  * This is a wrapper around MPI_Alltoallv that supports int64 counts and
  * displacements. The API is practically the same as MPI_Alltoallv.
  * If any count or displacement value is greater than INT_MAX, it will do a
  * manually implemented version of alltoallv that will first send most of the
  * data using a custom large-sized MPI type and then send the remainder.
  */
void bodo_alltoallv(const void *sendbuf,
                    const std::vector<int64_t> &send_counts,
                    const std::vector<int64_t> &send_disp,
                    MPI_Datatype sendtype,
                    void *recvbuf,
                    const std::vector<int64_t> &recv_counts,
                    const std::vector<int64_t> &recv_disp,
                    MPI_Datatype recvtype, MPI_Comm comm);

struct mpi_comm_info {
    int n_pes;
    std::vector<array_info*> arrays;
    size_t n_rows;
    bool has_nulls;
    // generally required MPI counts
    std::vector<int64_t> send_count;
    std::vector<int64_t> recv_count;
    std::vector<int64_t> send_disp;
    std::vector<int64_t> recv_disp;
    // counts required for string arrays
    std::vector<std::vector<int64_t>> send_count_sub;
    std::vector<std::vector<int64_t>> recv_count_sub;
    std::vector<std::vector<int64_t>> send_disp_sub;
    std::vector<std::vector<int64_t>> recv_disp_sub;
    // counts required for string list arrays
    std::vector<std::vector<int64_t>> send_count_sub_sub;
    std::vector<std::vector<int64_t>> recv_count_sub_sub;
    std::vector<std::vector<int64_t>> send_disp_sub_sub;
    std::vector<std::vector<int64_t>> recv_disp_sub_sub;
    // counts for arrays with null bitmask
    std::vector<int64_t> send_count_null;
    std::vector<int64_t> recv_count_null;
    std::vector<int64_t> send_disp_null;
    std::vector<int64_t> recv_disp_null;
    size_t n_null_bytes;

    explicit mpi_comm_info(std::vector<array_info*>& _arrays);

    void set_counts(uint32_t* hashes);
};

/** Shuffle a table so that same keys are on the same process.
 *  Note: Steals a reference from the input table.
 *
 * @param in_table : the input table.
 * @param n_keys   : the number of keys for comparison.
 * @return the new table after the shuffling-
 */
table_info* shuffle_table(table_info* in_table, int64_t n_keys);

/** Shuffling a table from all nodes to all the other nodes.
 *
 * @param in_table     : the input table.
 * @param hashes       : the array containing the values to be
 *                       i_node = hashes[i_row] % n_pes
 *                       for the nodes of rank not equal to 0.
 * @param comm_info    : the array for the communication.
 * @return the new table after the shuffling.
 */
table_info* shuffle_table_kernel(table_info* in_table, uint32_t* hashes,
                                 mpi_comm_info const& comm_info);

/** Reverse shuffling a table from all nodes to all the other nodes.
 *
 * @param in_table  : the input table.
 * @param hashes    : the hash of the reversed table
 * @param comm_info : the array for the communication.
 * @return the new table after the shuffling-
 */
table_info* reverse_shuffle_table_kernel(table_info* in_table, uint32_t* hashes,
                                         mpi_comm_info const& comm_info);

/** Broadcasting a table.
 * The table in the nodes of rank 0 is broadcast to all nodes.
 *
 * @param ref_table : the reference table used for the types
 * @param in_table  : the input table. It can be a NULL
 *                    for the nodes of rank not equal to 0 since it is not
 *                    read for those nodes.
 * @param n_cols    : the number of columns of the keys.
 * @return the table equal to in_table but available on all the nodes.
 */
table_info* broadcast_table(table_info* ref_table, table_info* in_table,
                            size_t n_cols);

/** Gather a table.
 *
 * @param in_table     : the input table.
 * @param n_cols       : the number of columns of the keys.
 * @param all_gather   : Whether to do all_gather or not.
 * @return the table obtained by concatenating the tables
 *         on the node 0.
 */
table_info* gather_table(table_info* in_table, size_t n_cols, bool all_gather);

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

/** Compute whether we need to do a reshuffling or not for performance reasons.
    The dilemna is following:
    ---If the workload is not well partitioned then the code becomes serialized
   and slower.
    ---Reshuffling itself is an expensive operation.

    The heuristic idea is following:
    ---What slows down the running is if one or 2 processors have a much higher
   load than other because it serializes the computation.
    ---If 1 or 2 processors have little load then that is not so bad. It just
   decreases the number of effective processors used.
    ---Thus the metric to consider or not a reshuffling is
        (max nb_row) / (avg nb_row)
    ---If the value is larger than 2 then reshuffling is interesting

    @param in_table : the input partitioned table
    @param crit_fraction : the critical fraction
    @return the boolean saying whether or not we need to do reshuffling.
*/
bool need_reshuffling(table_info* in_table, double crit_fraction);

/* Apply a renormalization shuffling
   After the operation, all nodes will have a standard size.

   @param in_table : the input partitioned table
   @return the reshuffled table
 */
table_info* shuffle_renormalization(table_info* in_table);

/* This function is used for the reverse shuffling of numpy data.
 *
 * It takes the rows after the MPI_alltoall and put them at their right position
 */
template <class T>
inline void fill_recv_data_inner(T* recv_buff, T* data, uint32_t* hashes,
                                 std::vector<int64_t> const& send_disp, int n_pes,
                                 size_t n_rows) {
    std::vector<int64_t> tmp_offset(send_disp);
    for (size_t i = 0; i < n_rows; i++) {
        size_t node = (size_t)hashes[i] % (size_t)n_pes;
        int64_t ind = tmp_offset[node];
        data[i] = recv_buff[ind];
        tmp_offset[node]++;
    }
}

#endif  // _SHUFFLE_H_INCLUDED
