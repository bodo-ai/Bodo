// Copyright (C) 2019 Bodo Inc. All rights reserved.
#ifndef _SHUFFLE_H_INCLUDED
#define _SHUFFLE_H_INCLUDED


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
    std::vector<std::vector<int>> send_count_sub;
    std::vector<std::vector<int>> recv_count_sub;
    std::vector<std::vector<int>> send_disp_sub;
    std::vector<std::vector<int>> recv_disp_sub;
    // counts required for string list arrays
    std::vector<std::vector<int>> send_count_sub_sub;
    std::vector<std::vector<int>> recv_count_sub_sub;
    std::vector<std::vector<int>> send_disp_sub_sub;
    std::vector<std::vector<int>> recv_disp_sub_sub;
    // counts for arrays with null bitmask
    std::vector<int> send_count_null;
    std::vector<int> recv_count_null;
    std::vector<int> send_disp_null;
    std::vector<int> recv_disp_null;
    size_t n_null_bytes;

    explicit mpi_comm_info(int _n_pes, std::vector<array_info*>& _arrays);

    void set_counts(uint32_t* hashes);
};

/** Shuffling a table from all nodes to all the other nodes.
 *  obtained by hashes.
 *
 * @param in_table : the input table.
 * @param n_keys   : the number of keys for comparison.
 * @return the new table after the shuffling-
 */
table_info* shuffle_table(table_info* in_table, int64_t n_keys);

/** Shuffling a table from all nodes to all the other nodes.
 *
 * @param in_table  : the input table.
 * @param hashes    : the array containing the values to be
 *                    i_node = hashes[i_row] % n_pes
 *                    for the nodes of rank not equal to 0.
 * @param n_pes     : the number of processors.
 * @param comm_info : the array for the communication.
 * @return the new table after the shuffling-
 */
table_info* shuffle_table_kernel(table_info* in_table, uint32_t* hashes,
                                 int n_pes, mpi_comm_info const& comm_info);

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
table_info* broadcast_table(table_info* ref_table, table_info* in_table, size_t n_cols);

/** Gather a table.
 *
 * @param in_table : the input table.
 * @param n_cols   : the number of columns of the keys.
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
    ---If the workload is not well partitioned then the code becomes serialized and slower.
    ---Reshuffling itself is an expensive operation.

    The heuristic idea is following:
    ---What slows down the running is if one or 2 processors have a much higher load
       than other because it serializes the computation.
    ---If 1 or 2 processors have little load then that is not so bad. It just decreases
       the number of effective processors used.
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

#endif // _SHUFFLE_H_INCLUDED
