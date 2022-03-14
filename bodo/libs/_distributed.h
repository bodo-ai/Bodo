// Copyright (C) 2019 Bodo Inc. All rights reserved.
#ifndef _DISTRIBUTED_H_INCLUDED
#define _DISTRIBUTED_H_INCLUDED

#include <Python.h>
#include <mpi.h>
#include <stdbool.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

#include "_bodo_common.h"

#define ROOT_PE 0

// XXX same as distributed_api.py:Reduce_Type
struct HPAT_ReduceOps {
    enum ReduceOpsEnum {
        SUM = 0,
        PROD = 1,
        MIN = 2,
        MAX = 3,
        ARGMIN = 4,
        ARGMAX = 5,
        OR = 6
    };
};

// data type for Decimal128 values (2 64-bit ints)
// initialized in dist/array tools C extensions
// NOTE: needs to be initialized in all C extensions that use it
extern MPI_Datatype decimal_mpi_type;

static int dist_get_rank() __UNUSED__;
static int dist_get_size() __UNUSED__;
static int dist_get_node_count() __UNUSED__;
static int64_t dist_get_start(int64_t total, int num_pes,
                              int node_id) __UNUSED__;
static int64_t dist_get_end(int64_t total, int num_pes, int node_id) __UNUSED__;
static int64_t dist_get_node_portion(int64_t total, int num_pes,
                                     int node_id) __UNUSED__;
static double dist_get_time() __UNUSED__;
static double get_time() __UNUSED__;
static int barrier() __UNUSED__;
static MPI_Datatype get_MPI_typ(int typ_enum) __UNUSED__;
static MPI_Datatype get_val_rank_MPI_typ(int typ_enum) __UNUSED__;
static MPI_Op get_MPI_op(int op_enum) __UNUSED__;
static int get_elem_size(int type_enum) __UNUSED__;
static void dist_reduce(char* in_ptr, char* out_ptr, int op,
                        int type_enum) __UNUSED__;
static void MPI_Allreduce_bool_or(std::vector<uint8_t>& V) __UNUSED__;
static void dist_exscan(char* in_ptr, char* out_ptr, int op,
                        int type_enum) __UNUSED__;

static void dist_arr_reduce(void* out, int64_t total_size, int op_enum,
                            int type_enum) __UNUSED__;
static MPI_Request dist_irecv(void* out, int size, int type_enum, int pe,
                              int tag, bool cond) __UNUSED__;
static MPI_Request dist_isend(void* out, int size, int type_enum, int pe,
                              int tag, bool cond) __UNUSED__;
static void dist_recv(void* out, int size, int type_enum, int pe,
                      int tag) __UNUSED__;
static void dist_send(void* out, int size, int type_enum, int pe,
                      int tag) __UNUSED__;
static void dist_wait(MPI_Request req, bool cond) __UNUSED__;
static void dist_waitall(int size, MPI_Request* req) __UNUSED__;

static void c_gather_scalar(void* send_data, void* recv_data, int typ_enum,
                            bool allgather, int root) __UNUSED__;
static void c_gatherv(void* send_data, int sendcount, void* recv_data,
                      int* recv_counts, int* displs, int typ_enum,
                      bool allgather, int root) __UNUSED__;
static void c_scatterv(void* send_data, int* sendcounts, int* displs,
                       void* recv_data, int recv_count,
                       int typ_enum) __UNUSED__;
static void c_allgatherv(void* send_data, int sendcount, void* recv_data,
                         int* recv_counts, int* displs,
                         int typ_enum) __UNUSED__;
static void c_bcast(void* send_data, int sendcount, int typ_enum,
                    int* comm_ranks, int nranks, int root) __UNUSED__;

static void c_alltoallv(void* send_data, void* recv_data, int* send_counts,
                        int* recv_counts, int* send_disp, int* recv_disp,
                        int typ_enum) __UNUSED__;
static void c_alltoall(void* send_data, void* recv_data, int count,
                       int typ_enum) __UNUSED__;
static void c_comm_create(const int* comm_ranks, int n,
                          MPI_Comm* comm) __UNUSED__;
static int64_t dist_get_item_pointer(int64_t ind, int64_t start,
                                     int64_t count) __UNUSED__;
static void allgather(void* out_data, int size, void* in_data,
                      int type_enum) __UNUSED__;
static MPI_Request* comm_req_alloc(int size) __UNUSED__;
static void comm_req_dealloc(MPI_Request* req_arr) __UNUSED__;
static void req_array_setitem(MPI_Request* req_arr, int64_t ind,
                              MPI_Request req) __UNUSED__;

/**
 * This is a wrapper around MPI_Alltoallv that supports int64 counts and
 * displacements. The API is practically the same as MPI_Alltoallv.
 * If any count or displacement value is greater than INT_MAX, it will do a
 * manually implemented version of alltoallv that will first send most of the
 * data using a custom large-sized MPI type and then send the remainder.
 */
static void bodo_alltoallv(const void* sendbuf,
                           const std::vector<int64_t>& send_counts,
                           const std::vector<int64_t>& send_disp,
                           MPI_Datatype sendtype, void* recvbuf,
                           const std::vector<int64_t>& recv_counts,
                           const std::vector<int64_t>& recv_disp,
                           MPI_Datatype recvtype, MPI_Comm comm);

static void oneD_reshape_shuffle(char* output, char* input,
                                 int64_t new_dim0_global_len,
                                 int64_t old_dim0_local_len,
                                 int64_t out_lower_dims_size,
                                 int64_t in_lower_dims_size, int n_dest_ranks,
                                 int* dest_ranks) __UNUSED__;

static void permutation_int(int64_t* output, int n) __UNUSED__;
static void permutation_array_index(unsigned char* lhs, int64_t len,
                                    int64_t elem_size, unsigned char* rhs,
                                    int64_t n_elems_rhs, int64_t* p,
                                    int64_t p_len) __UNUSED__;
static int finalize() __UNUSED__;
static int hpat_dummy_ptr[64] __UNUSED__;

/* *********************************************************************
************************************************************************/

static void* get_dummy_ptr() __UNUSED__;
static void* get_dummy_ptr() { return hpat_dummy_ptr; }

static size_t get_mpi_req_num_bytes() __UNUSED__;
static size_t get_mpi_req_num_bytes() { return sizeof(MPI_Request); }

/*
 * Returns how many nodes in the cluster
 * Creates subcommunicators based on shared memory
 * split type and then counts how many rank0
 * the cluster has
 */
static int dist_get_node_count() {
    int is_initialized;
    MPI_Initialized(&is_initialized);
    if (!is_initialized) MPI_Init(NULL, NULL);

    int rank, is_rank0, nodes;
    MPI_Comm shmcomm;

    // Split comm, into comms that has same shared memory
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &shmcomm);
    MPI_Comm_rank(shmcomm, &rank);

    // Identify rank 0 in each node
    is_rank0 = (rank == 0) ? 1 : 0;

    // Sum how many rank0 found
    MPI_Allreduce(&is_rank0, &nodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    MPI_Comm_free(&shmcomm);
    return nodes;
}

static int dist_get_rank() {
    int is_initialized;
    MPI_Initialized(&is_initialized);
    if (!is_initialized) MPI_Init(NULL, NULL);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // printf("my_rank:%d\n", rank);
    return rank;
}

static int dist_get_size() {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // printf("r size:%d\n", sizeof(MPI_Request));
    // printf("mpi_size:%d\n", size);
    return size;
}

static int64_t dist_get_start(int64_t total, int num_pes, int node_id) {
    int64_t res = total % num_pes;
    int64_t blk_size = (total - res) / num_pes;
    return node_id * blk_size + std::min(int64_t(node_id), res);
}

static int64_t dist_get_end(int64_t total, int num_pes, int node_id) {
    return dist_get_start(total, num_pes, node_id + 1);
}

static int64_t dist_get_node_portion(int64_t total, int num_pes, int node_id) {
    return dist_get_end(total, num_pes, node_id) -
           dist_get_start(total, num_pes, node_id);
}

/**
 * @param total: global number of elements
 * @param num_pes: number of ranks among to divide the elements
 * @param index: element index
 * @return rank that should have the element specified by index
 */
static int64_t index_rank(int64_t total, int num_pes, int64_t index) {
    int64_t res = total % num_pes;
    int64_t blk_size = (total - res) / num_pes;
    // In the part 0:crit_index the size of the blocks is blk_size+1.
    // In the range crit_index:total the size of the blocks is blk_size.
    int64_t crit_index = (blk_size + 1) * res;
    if (index < crit_index) {
        return index / (blk_size + 1);
    } else {
        return res + (index - crit_index) / blk_size;
    }
}

static double dist_get_time() {
    double wtime;
    MPI_Barrier(MPI_COMM_WORLD);
    wtime = MPI_Wtime();
    return wtime;
}

static double get_time() {
    double wtime;
    wtime = MPI_Wtime();
    return wtime;
}

static int barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
    return 0;
}

static void dist_reduce(char* in_ptr, char* out_ptr, int op_enum,
                        int type_enum) {
    // printf("reduce value: %d\n", value);
    MPI_Datatype mpi_typ = get_MPI_typ(type_enum);
    MPI_Op mpi_op = get_MPI_op(op_enum);

    // argmax and argmin need special handling
    if (mpi_op == MPI_MAXLOC || mpi_op == MPI_MINLOC) {
        // since MPI's indexed types use 32 bit integers, we workaround by
        // using rank as index, then broadcasting the actual values from the
        // target rank.
        // TODO: generate user-defined reduce operation to avoid this workaround
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        // allreduce struct is value + integer
        int value_size;
        MPI_Type_size(mpi_typ, &value_size);
        // TODO: support int64_int value on Windows
        MPI_Datatype val_rank_mpi_typ = get_val_rank_MPI_typ(type_enum);
        // copy input index_value to output
        memcpy(out_ptr, in_ptr, value_size + sizeof(int64_t));
        // printf("rank:%d index:%lld value:%lf value_size:%d\n", rank,
        //     *(int64_t*)in_ptr, *(double*)(in_ptr+sizeof(int64_t)),
        //     value_size);

        // Determine the size of the pointer to allocate.
        // argmin/argmax in MPI communicates a struct of 2 values:
        // the actual value and a 32-bit index.
        int val_idx_struct_size;
        MPI_Type_size(val_rank_mpi_typ, &val_idx_struct_size);

        // format: value + int (input format is int64+value)
        char* in_val_rank = (char*)malloc(val_idx_struct_size);
        if (in_val_rank == NULL) return;
        char* out_val_rank = (char*)malloc(val_idx_struct_size);
        if (out_val_rank == NULL) {
            free(in_val_rank);
            return;
        }

        char* in_val_ptr = in_ptr + sizeof(int64_t);

        // MPI doesn't support values smaller than int and unsigned values, so
        // cast values when they don't fit originally.
        // TODO: Add support value_size > struct_val_size (int64 and long on
        // Windows) and equal size but unsigned uint64 and long on Linux
        int struct_val_size = val_idx_struct_size - sizeof(MPI_INT);
        if (struct_val_size > value_size) {
            // Case 1: uint32 and long on Linux
            if (struct_val_size == sizeof(int64_t)) {
                uint32_t orig_val = *((uint32_t*)in_val_ptr);
                int64_t value = (int64_t)orig_val;
                memcpy(in_val_rank, (char*)&value, struct_val_size);
                // Case 2: Values smaller than int32_t (int8, uint8, int16,
                // uint16)
                // TODO: Can int be smaller on Windows?
            } else if (struct_val_size == sizeof(int32_t)) {
                int32_t value = 0;
                switch (type_enum) {
                    case Bodo_CTypes::INT8:
                        value = (int32_t) * ((int8_t*)in_val_ptr);
                        break;
                    case Bodo_CTypes::UINT8:
                        value = (int32_t) * ((uint8_t*)in_val_ptr);
                        break;
                    case Bodo_CTypes::INT16:
                        value = (int32_t) * ((int16_t*)in_val_ptr);
                        break;
                    case Bodo_CTypes::UINT16:
                        value = (int32_t) * ((uint16_t*)in_val_ptr);
                        break;
                }
                memcpy(in_val_rank, &value, struct_val_size);
            }
        } else {
            memcpy(in_val_rank, in_val_ptr, struct_val_size);
        }
        memcpy(in_val_rank + struct_val_size, &rank, sizeof(int));
        MPI_Allreduce(in_val_rank, out_val_rank, 1, val_rank_mpi_typ, mpi_op,
                      MPI_COMM_WORLD);

        int target_rank = *((int*)(out_val_rank + struct_val_size));
        // printf("rank:%d allreduce rank:%d val:%lf\n", rank, target_rank,
        // *(double*)out_val_rank);
        MPI_Bcast(out_ptr, value_size + sizeof(int64_t), MPI_BYTE, target_rank,
                  MPI_COMM_WORLD);
        free(in_val_rank);
        free(out_val_rank);
        return;
    }

    MPI_Allreduce(in_ptr, out_ptr, 1, mpi_typ, mpi_op, MPI_COMM_WORLD);
    return;
}

static void MPI_Allreduce_bool_or(std::vector<uint8_t>& V) {
    int len = V.size();
    MPI_Datatype mpi_typ8 = get_MPI_typ(Bodo_CTypes::UINT8);
    MPI_Allreduce(MPI_IN_PLACE, V.data(), len, mpi_typ8, MPI_BOR,
                  MPI_COMM_WORLD);
}

static void dist_arr_reduce(void* out, int64_t total_size, int op_enum,
                            int type_enum) {
    MPI_Datatype mpi_typ = get_MPI_typ(type_enum);
    MPI_Op mpi_op = get_MPI_op(op_enum);
    int elem_size = get_elem_size(type_enum);
    void* res_buf = malloc(total_size * elem_size);
    MPI_Allreduce(out, res_buf, total_size, mpi_typ, mpi_op, MPI_COMM_WORLD);
    memcpy(out, res_buf, total_size * elem_size);
    free(res_buf);
    return;
}

static void dist_exscan(char* in_ptr, char* out_ptr, int op_enum,
                        int type_enum) {
    MPI_Datatype mpi_typ = get_MPI_typ(type_enum);
    MPI_Op mpi_op = get_MPI_op(op_enum);
    MPI_Exscan(in_ptr, out_ptr, 1, mpi_typ, mpi_op, MPI_COMM_WORLD);
    return;
}

static void dist_recv(void* out, int size, int type_enum, int pe, int tag) {
    MPI_Datatype mpi_typ = get_MPI_typ(type_enum);
    MPI_Recv(out, size, mpi_typ, pe, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

static void dist_send(void* out, int size, int type_enum, int pe, int tag) {
    MPI_Datatype mpi_typ = get_MPI_typ(type_enum);
    MPI_Send(out, size, mpi_typ, pe, tag, MPI_COMM_WORLD);
}

static MPI_Request dist_irecv(void* out, int size, int type_enum, int pe,
                              int tag, bool cond) {
    MPI_Request mpi_req_recv(MPI_REQUEST_NULL);
    // printf("irecv size:%d pe:%d tag:%d, cond:%d\n", size, pe, tag, cond);
    // fflush(stdout);
    if (cond) {
        MPI_Datatype mpi_typ = get_MPI_typ(type_enum);
        MPI_Irecv(out, size, mpi_typ, pe, tag, MPI_COMM_WORLD, &mpi_req_recv);
    }
    // printf("after irecv size:%d pe:%d tag:%d, cond:%d\n", size, pe, tag,
    // cond);
    // fflush(stdout);
    return mpi_req_recv;
}

static MPI_Request dist_isend(void* out, int size, int type_enum, int pe,
                              int tag, bool cond) {
    MPI_Request mpi_req_recv(MPI_REQUEST_NULL);
    // printf("isend size:%d pe:%d tag:%d, cond:%d\n", size, pe, tag, cond);
    // fflush(stdout);
    if (cond) {
        MPI_Datatype mpi_typ = get_MPI_typ(type_enum);
        MPI_Isend(out, size, mpi_typ, pe, tag, MPI_COMM_WORLD, &mpi_req_recv);
    }
    // printf("after isend size:%d pe:%d tag:%d, cond:%d\n", size, pe, tag,
    // cond);
    // fflush(stdout);
    return mpi_req_recv;
}

static void dist_wait(MPI_Request req, bool cond) {
    if (cond) MPI_Wait(&req, MPI_STATUS_IGNORE);
}

static void allgather(void* out_data, int size, void* in_data, int type_enum) {
    MPI_Datatype mpi_typ = get_MPI_typ(type_enum);
    MPI_Allgather(in_data, size, mpi_typ, out_data, size, mpi_typ,
                  MPI_COMM_WORLD);
    return;
}

static void req_array_setitem(MPI_Request* req_arr, int64_t ind,
                              MPI_Request req) {
    req_arr[ind] = req;
    return;
}

static void dist_waitall(int size, MPI_Request* req_arr) {
    MPI_Waitall(size, req_arr, MPI_STATUSES_IGNORE);
    return;
}

static MPI_Datatype get_MPI_typ(int typ_enum) {
    switch (typ_enum) {
        case Bodo_CTypes::_BOOL:
            return MPI_UNSIGNED_CHAR;  // MPI_C_BOOL doesn't support operations
                                       // like min
        case Bodo_CTypes::INT8:
            return MPI_CHAR;
        case Bodo_CTypes::UINT8:
            return MPI_UNSIGNED_CHAR;
        case Bodo_CTypes::INT32:
            return MPI_INT;
        case Bodo_CTypes::UINT32:
            return MPI_UNSIGNED;
        case Bodo_CTypes::INT64:
        case Bodo_CTypes::DATE:
        case Bodo_CTypes::DATETIME:
        case Bodo_CTypes::TIMEDELTA:
            return MPI_LONG_LONG_INT;
        case Bodo_CTypes::UINT64:
            return MPI_UNSIGNED_LONG_LONG;
        case Bodo_CTypes::FLOAT32:
            return MPI_FLOAT;
        case Bodo_CTypes::FLOAT64:
            return MPI_DOUBLE;
        case Bodo_CTypes::INT16:
            // TODO: use MPI_INT16_T?
            return MPI_SHORT;
        case Bodo_CTypes::UINT16:
            return MPI_UNSIGNED_SHORT;
        case Bodo_CTypes::INT128:
        case Bodo_CTypes::DECIMAL:
            return decimal_mpi_type;
        default:
            std::cerr << "Invalid MPI_Type " << typ_enum << "\n";
    }
    // dummy value in case of error
    // TODO: raise error properly
    return MPI_LONG_LONG_INT;
}

static MPI_Datatype get_val_rank_MPI_typ(int typ_enum) {
    // printf("h5 type enum:%d\n", typ_enum);
    // XXX: LONG is used for int64, which doesn't work on Windows
    // XXX: LONG is used for uint64
    // XXX: INT is used for sizes <= int32_t. The data is cast to an
    // int type at runtime
    if (typ_enum == Bodo_CTypes::DATE || typ_enum == Bodo_CTypes::DATETIME ||
        typ_enum == Bodo_CTypes::TIMEDELTA)
        // treat date 64-bit values as int64
        typ_enum = Bodo_CTypes::INT64;
    if (typ_enum == Bodo_CTypes::_BOOL) {
        typ_enum = Bodo_CTypes::INT8;
    }
    if (typ_enum < 0 || typ_enum > 9) {
        std::cerr << "Invalid MPI_Type"
                  << "\n";
        return MPI_DATATYPE_NULL;
    }
    MPI_Datatype types_list[] = {
        MPI_2INT,      MPI_2INT,       MPI_2INT,     MPI_2INT, MPI_LONG_INT,
        MPI_FLOAT_INT, MPI_DOUBLE_INT, MPI_LONG_INT, MPI_2INT, MPI_2INT};
    return types_list[typ_enum];
}

// from distributed_api Reduce_Type
static MPI_Op get_MPI_op(int op_enum) {
    // printf("op type enum:%d\n", op_enum);
    if (op_enum < 0 || op_enum > 6) {
        std::cerr << "Invalid MPI_Op"
                  << "\n";
        return MPI_SUM;
    }
    MPI_Op ops_list[] = {MPI_SUM,    MPI_PROD,   MPI_MIN, MPI_MAX,
                         MPI_MINLOC, MPI_MAXLOC, MPI_BOR};

    return ops_list[op_enum];
}

static int get_elem_size(int type_enum) {
    if (type_enum < 0 || type_enum > 7) {
        std::cerr << "Invalid MPI_Type"
                  << "\n";
        return 8;
    }
    int types_sizes[] = {1, 1, 4, 4, 8, 4, 8, 8};
    return types_sizes[type_enum];
}

static int64_t dist_get_item_pointer(int64_t ind, int64_t start,
                                     int64_t count) {
    // printf("ind:%lld start:%lld count:%lld\n", ind, start, count);
    if (ind >= start && ind < start + count) return ind - start;
    return -1;
}

static void c_gather_scalar(void* send_data, void* recv_data, int typ_enum,
                            bool allgather, int root) {
    MPI_Datatype mpi_typ = get_MPI_typ(typ_enum);
    if (allgather)
        MPI_Allgather(send_data, 1, mpi_typ, recv_data, 1, mpi_typ,
                      MPI_COMM_WORLD);
    else
        MPI_Gather(send_data, 1, mpi_typ, recv_data, 1, mpi_typ, root,
                   MPI_COMM_WORLD);
    return;
}

static void c_gatherv(void* send_data, int sendcount, void* recv_data,
                      int* recv_counts, int* displs, int typ_enum,
                      bool allgather, int root) {
    MPI_Datatype mpi_typ = get_MPI_typ(typ_enum);
    if (allgather)
        MPI_Allgatherv(send_data, sendcount, mpi_typ, recv_data, recv_counts,
                       displs, mpi_typ, MPI_COMM_WORLD);
    else
        MPI_Gatherv(send_data, sendcount, mpi_typ, recv_data, recv_counts,
                    displs, mpi_typ, root, MPI_COMM_WORLD);
    return;
}

static void c_allgatherv(void* send_data, int sendcount, void* recv_data,
                         int* recv_counts, int* displs, int typ_enum) {
    MPI_Datatype mpi_typ = get_MPI_typ(typ_enum);
    MPI_Allgatherv(send_data, sendcount, mpi_typ, recv_data, recv_counts,
                   displs, mpi_typ, MPI_COMM_WORLD);
    return;
}

static void c_scatterv(void* send_data, int* sendcounts, int* displs,
                       void* recv_data, int recv_count, int typ_enum) {
    MPI_Datatype mpi_typ = get_MPI_typ(typ_enum);
    MPI_Scatterv(send_data, sendcounts, displs, mpi_typ, recv_data, recv_count,
                 mpi_typ, ROOT_PE, MPI_COMM_WORLD);
}

/**
 * Create a sub communicator with specific ranks from MPI_COMM_WORLD
 *
 * @param comm_ranks pointer to ranks integer array
 * @param comm new communicator handle
 */
static void c_comm_create(const int* comm_ranks, int n, MPI_Comm* comm) {
    MPI_Group new_group;
    MPI_Group world_group;
    int err = MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    assert(err == MPI_SUCCESS);
    err = MPI_Group_incl(world_group, n, comm_ranks, &new_group);
    assert(err == MPI_SUCCESS);
    err = MPI_Comm_create(MPI_COMM_WORLD, new_group, comm);
    assert(err == MPI_SUCCESS);
}

/**
 * MPI_Bcast for all ranks or subset of them
 *
 * @param send_data pointer to data buffer to broadcast
 * @param sendcount number of elements in the data buffer
 * @param typ_enum datatype of buffer
 * @param comm_ranks pointer to ranks integer array. ([-1] for MPI_COMM_WORLD)
 * @param n number of elements in ranks array (0 for MPI_COMM_WORLD)
 * @param root rank to broadcast.
 */
static void c_bcast(void* send_data, int sendcount, int typ_enum,
                    int* comm_ranks, int n, int root) {
    MPI_Datatype mpi_typ = get_MPI_typ(typ_enum);
    if (n == 0) {
        MPI_Bcast(send_data, sendcount, mpi_typ, root, MPI_COMM_WORLD);
    } else {
        MPI_Comm comm;
        c_comm_create(comm_ranks, n, &comm);
        if (MPI_COMM_NULL != comm) {
            MPI_Bcast(send_data, sendcount, mpi_typ, root, comm);
        }
    }
    return;
}

static void c_alltoallv(void* send_data, void* recv_data, int* send_counts,
                        int* recv_counts, int* send_disp, int* recv_disp,
                        int typ_enum) {
    MPI_Datatype mpi_typ = get_MPI_typ(typ_enum);
    MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    int err_code =
        MPI_Alltoallv(send_data, send_counts, send_disp, mpi_typ, recv_data,
                      recv_counts, recv_disp, mpi_typ, MPI_COMM_WORLD);

    // TODO: create a macro for this and add to all MPI calls
    if (err_code != MPI_SUCCESS) {
        char err_string[MPI_MAX_ERROR_STRING];
        err_string[MPI_MAX_ERROR_STRING - 1] = '\0';
        int err_len, err_class, my_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        MPI_Error_class(err_code, &err_class);
        MPI_Error_string(err_class, err_string, &err_len);
        fprintf(stderr, "%d: %s\n", my_rank, err_string);
        MPI_Error_string(err_code, err_string, &err_len);
        fprintf(stderr, "%d: %s\n", my_rank, err_string);
        MPI_Abort(MPI_COMM_WORLD, err_code);
    }
}

static void c_alltoall(void* send_data, void* recv_data, int count,
                       int typ_enum) {
    MPI_Datatype mpi_typ = get_MPI_typ(typ_enum);
    MPI_Alltoall(send_data, count, mpi_typ, recv_data, count, mpi_typ,
                 MPI_COMM_WORLD);
}

MPI_Request* comm_req_alloc(int size) {
    // printf("req alloc %d\n", size);
    return new MPI_Request[size];
}

static void comm_req_dealloc(MPI_Request* req_arr) { delete[] req_arr; }

static int finalize() {
    int is_initialized;
    MPI_Initialized(&is_initialized);
    if (!is_initialized) {
        return 0;
    }
    int is_finalized;
    MPI_Finalized(&is_finalized);
    if (!is_finalized) {
        // printf("finalizing\n");
        MPI_Finalize();
    }
    return 0;
}

static void permutation_int(int64_t* output, int n) {
    MPI_Bcast(output, n, MPI_INT64_T, 0, MPI_COMM_WORLD);
}

// Given the permutation index |p| and |rank|, and the number of ranks
// |num_ranks|, finds the destination ranks of indices of the |rank|.  For
// example, if |rank| is 1, |num_ranks| is 3, |p_len| is 12, and |p| is the
// following array [ 9, 8, 6, 4, 11, 7, 2, 3, 5, 0, 1, 10], the function returns
// [0, 2, 0, 1].
static std::vector<int64_t> find_dest_ranks(int64_t rank, int64_t n_elems_local,
                                            int64_t num_ranks, int64_t* p,
                                            int64_t p_len) {
    // find global start offset of my current chunk of data
    int64_t my_chunk_start = 0;
    // get current chunk sizes of all ranks
    std::vector<int64_t> AllSizes(num_ranks);
    MPI_Allgather(&n_elems_local, 1, MPI_INT64_T, AllSizes.data(), 1,
                  MPI_INT64_T, MPI_COMM_WORLD);
    for (int i = 0; i < rank; i++) my_chunk_start += AllSizes[i];

    std::vector<int64_t> dest_ranks(n_elems_local);
    // find destination of every element in my chunk based on the permutation
    for (int64_t i = 0; i < n_elems_local; i++) {
        dest_ranks[i] = index_rank(p_len, num_ranks, p[my_chunk_start + i]);
    }
    return dest_ranks;
}

static std::vector<int> find_send_counts(const std::vector<int64_t>& dest_ranks,
                                         int64_t num_ranks, int64_t elem_size) {
    std::vector<int> send_counts(num_ranks);
    for (auto dest : dest_ranks) ++send_counts[dest];
    return send_counts;
}

static std::vector<int> find_disps(const std::vector<int>& counts) {
    std::vector<int> disps(counts.size(), 0);
    for (size_t i = 1; i < disps.size(); ++i)
        disps[i] = disps[i - 1] + counts[i - 1];
    return disps;
}

static std::vector<int> find_recv_counts(int64_t num_ranks,
                                         const std::vector<int>& send_counts) {
    std::vector<int> recv_counts(num_ranks);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
                 MPI_COMM_WORLD);
    return recv_counts;
}

// Returns an |index_array| which would sort the array |v| of size |len| when
// applied to it.  Identical to numpy.argsort.
template <class T>
static std::vector<size_t> arg_sort(T* v, int64_t len) {
    std::vector<size_t> index_array(len);
    std::iota(index_array.begin(), index_array.end(), 0);
    std::sort(index_array.begin(), index_array.end(),
              [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
    return index_array;
}

// |v| is an array of elements of size |elem_size|.  This function swaps
// elements located at indices |i1| and |i2|.
static void elem_swap(unsigned char* v, int64_t elem_size, size_t i1,
                      size_t i2) {
    std::vector<unsigned char> tmp(elem_size);
    auto i1_offset = v + i1 * elem_size;
    auto i2_offset = v + i2 * elem_size;
    std::copy(i1_offset, i1_offset + elem_size, tmp.data());
    std::copy(i2_offset, i2_offset + elem_size, i1_offset);
    std::copy(std::begin(tmp), std::end(tmp), i2_offset);
}

// Applies the permutation represented by |p| to the array |v| whose elements
// are of size |elem_size| using O(1) space.  See the following URL for the
// details: https://blogs.msdn.microsoft.com/oldnewthing/20170102-00/?p=95095.
static void apply_permutation(unsigned char* v, int64_t elem_size,
                              std::vector<size_t>& p) {
    for (size_t i = 0; i < p.size(); ++i) {
        auto current = i;
        while (i != p[current]) {
            auto next = p[current];
            elem_swap(v, elem_size, next, current);
            p[current] = current;
            current = next;
        }
        p[current] = current;
    }
}

// Applies the permutation represented by |p| of size |p_len| to the array |rhs|
// of elements of size |elem_size| and stores the result in |lhs|.
static void permutation_array_index(unsigned char* lhs, int64_t len,
                                    int64_t elem_size, unsigned char* rhs,
                                    int64_t n_elems_rhs, int64_t* p,
                                    int64_t p_len) {
    try {
        if (len != p_len) {
            throw std::runtime_error(
                "_distributed.h::permutation_array_index: Array length and "
                "permutation index length should match!");
        }

        MPI_Datatype element_t;
        MPI_Type_contiguous(elem_size, MPI_UNSIGNED_CHAR, &element_t);
        MPI_Type_commit(&element_t);

        auto num_ranks = dist_get_size();
        auto rank = dist_get_rank();
        // dest_ranks contains the destination rank for each element i in rhs
        auto dest_ranks =
            find_dest_ranks(rank, n_elems_rhs, num_ranks, p, p_len);
        auto send_counts = find_send_counts(dest_ranks, num_ranks, elem_size);
        auto send_disps = find_disps(send_counts);
        auto recv_counts = find_recv_counts(num_ranks, send_counts);
        auto recv_disps = find_disps(recv_counts);

        auto offsets = send_disps;
        std::vector<unsigned char> send_buf(dest_ranks.size() * elem_size);
        for (size_t i = 0; i < dest_ranks.size(); ++i) {
            auto send_buf_offset = offsets[dest_ranks[i]]++ * elem_size;
            auto* send_buf_begin = send_buf.data() + send_buf_offset;
            auto* rhs_begin = rhs + i * elem_size;
            std::copy(rhs_begin, rhs_begin + elem_size, send_buf_begin);
        }

        MPI_Alltoallv(send_buf.data(), send_counts.data(), send_disps.data(),
                      element_t, lhs, recv_counts.data(), recv_disps.data(),
                      element_t, MPI_COMM_WORLD);

        // Let us assume that the global data array is [a b c d e f g h] and the
        // permutation array that we would like to apply to it is [2 7 5 6 4 3 1
        // 0]. Hence, the resultant permutation is [c h f g e d b a].  Assuming
        // that there are two ranks, each receiving 4 data items, and we are
        // rank 0, after MPI_Alltoallv returns, we receive the chunk [c f g h]
        // that corresponds to the sorted chunk of our permutation, which is [2
        // 5 6 7]. In order to recover the positions of [c f g h] in the target
        // permutation we first argsort our chunk of permutation array:
        auto begin = p + dist_get_start(p_len, num_ranks, rank);
        int64_t size_after_shuffle =
            dist_get_node_portion(p_len, num_ranks, rank);
        auto p1 = arg_sort(begin, size_after_shuffle);

        // The result of the argsort, stored in p1, is [0 2 3 1].  This tells us
        // how the chunk we have received is different from the target
        // permutation we want to achieve.  Hence, to achieve the target
        // permutation, we need to sort our data chunk based on p1.  One way of
        // sorting array A based on the values of array B, is to argsort array B
        // and apply the permutation to array A.  Therefore, we argsort p1:
        auto p2 = arg_sort(p1.data(), size_after_shuffle);

        // which gives us [0 3 1 2], and apply the resultant permutation to our
        // data chunk to obtain the target permutation.
        apply_permutation(lhs, elem_size, p2);

        MPI_Type_free(&element_t);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return;
    }
}

static void bodo_alltoallv(const void* sendbuf,
                           const std::vector<int64_t>& send_counts,
                           const std::vector<int64_t>& send_disp,
                           MPI_Datatype sendtype, void* recvbuf,
                           const std::vector<int64_t>& recv_counts,
                           const std::vector<int64_t>& recv_disp,
                           MPI_Datatype recvtype, MPI_Comm comm) {
    tracing::Event ev("bodo_alltoallv");
    const int A2AV_LARGE_DTYPE_SIZE = 1024;
    int send_typ_size;
    int recv_typ_size;
    int n_pes;
    MPI_Comm_size(comm, &n_pes);
    MPI_Type_size(sendtype, &send_typ_size);
    MPI_Type_size(recvtype, &recv_typ_size);
    int big_shuffle = 0;
    for (int i = 0; i < n_pes; i++) {
        if (big_shuffle > 1) break;  // error
        // if any count or displacement doesn't fit in int we have to do big
        // shuffle
        if (send_counts[i] >= (int64_t)INT_MAX ||
            recv_counts[i] >= (int64_t)INT_MAX ||
            send_disp[i] >= (int64_t)INT_MAX ||
            recv_disp[i] >= (int64_t)INT_MAX) {
            big_shuffle = 1;
        }
        if (big_shuffle == 1) {
            if (send_counts[i] * send_typ_size / A2AV_LARGE_DTYPE_SIZE >=
                INT_MAX)
                big_shuffle = 2;
            if (recv_counts[i] * recv_typ_size / A2AV_LARGE_DTYPE_SIZE >=
                INT_MAX)
                big_shuffle = 2;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &big_shuffle, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    if (big_shuffle == 2)
        // very improbable but not impossible
        throw std::runtime_error("Data is too big to shuffle");

    if (big_shuffle == 0) {
        ev.add_attribute("g_big_shuffle", 0);
        std::vector<int> send_counts_int(send_counts.begin(),
                                         send_counts.end());
        std::vector<int> recv_counts_int(recv_counts.begin(),
                                         recv_counts.end());
        std::vector<int> send_disp_int(send_disp.begin(), send_disp.end());
        std::vector<int> recv_disp_int(recv_disp.begin(), recv_disp.end());
        MPI_Alltoallv(sendbuf, send_counts_int.data(), send_disp_int.data(),
                      sendtype, recvbuf, recv_counts_int.data(),
                      recv_disp_int.data(), recvtype, comm);
    } else {
        ev.add_attribute("g_big_shuffle", 1);
        const int TAG = 11;  // arbitrary
        int rank;
        MPI_Comm_rank(comm, &rank);
        MPI_Datatype large_dtype;
        MPI_Type_contiguous(A2AV_LARGE_DTYPE_SIZE, MPI_CHAR, &large_dtype);
        MPI_Type_commit(&large_dtype);
        for (int i = 0; i < n_pes; i++) {
            int dest = (rank + i + n_pes) % n_pes;
            int src = (rank - i + n_pes) % n_pes;
            char* send_ptr = (char*)sendbuf + send_disp[dest] * send_typ_size;
            char* recv_ptr = (char*)recvbuf + recv_disp[src] * recv_typ_size;
            int send_count =
                send_counts[dest] * send_typ_size / A2AV_LARGE_DTYPE_SIZE;
            int recv_count =
                recv_counts[src] * recv_typ_size / A2AV_LARGE_DTYPE_SIZE;
            MPI_Sendrecv(send_ptr, send_count, large_dtype, dest, TAG, recv_ptr,
                         recv_count, large_dtype, src, TAG, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            // send leftover
            MPI_Sendrecv(
                send_ptr + int64_t(send_count) * A2AV_LARGE_DTYPE_SIZE,
                send_counts[dest] * send_typ_size % A2AV_LARGE_DTYPE_SIZE,
                MPI_CHAR, dest, TAG + 1,
                recv_ptr + int64_t(recv_count) * A2AV_LARGE_DTYPE_SIZE,
                recv_counts[src] * recv_typ_size % A2AV_LARGE_DTYPE_SIZE,
                MPI_CHAR, src, TAG + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        MPI_Type_free(&large_dtype);
    }
}

/**
 * @brief Shuffles the data to fill the output array properly for reshape.
 * Finds global byte offsets for data in each rank and calls alltoallv.
 *
 * @param output data pointer to output array
 * @param input data pointer to input array
 * @param new_dim0_global_len the new global size for the 1st array dimension
 * @param old_dim0_local_len local size of 1st array dimension in input
 * @param out_lower_dims_size total size of lower output dimensions in bytes
 * @param in_lower_dims_size total size of lower input dimensions in bytes
 * @param n_dest_ranks: number of destination ranks (if <=0 data is distributed
 *                      across all ranks, otherwise only to specified ranks)
 * @param dest_ranks: destination ranks
 */
static void oneD_reshape_shuffle(char* output, char* input,
                                 int64_t new_dim0_global_len,
                                 int64_t old_dim0_local_len,
                                 int64_t out_lower_dims_size,
                                 int64_t in_lower_dims_size, int n_dest_ranks,
                                 int* dest_ranks) {
    try {
        int num_pes = dist_get_size();
        int rank = dist_get_rank();

        // local sizes on all ranks
        std::vector<int64_t> all_old_dim0_local_sizes(num_pes);
        MPI_Allgather(&old_dim0_local_len, 1, MPI_INT64_T,
                      all_old_dim0_local_sizes.data(), 1, MPI_INT64_T,
                      MPI_COMM_WORLD);
        // dim0 start offset (not byte offset) on all pes
        std::vector<int64_t> all_old_starts(num_pes);
        all_old_starts[0] = 0;
        for (size_t i = 1; i < (size_t)num_pes; i++)
            all_old_starts[i] =
                all_old_starts[i - 1] + all_old_dim0_local_sizes[i - 1];

        // map rank in COMM_WORLD to rank in destination group
        std::vector<int> group_rank(num_pes, -1);
        if (n_dest_ranks <= 0) {
            // using all ranks in COMM_WORLD
            n_dest_ranks = num_pes;
            for (int i = 0; i < num_pes; i++) group_rank[i] = i;
        } else {
            for (int i = 0; i < n_dest_ranks; i++)
                group_rank[dest_ranks[i]] = i;
        }

        // get my old and new data interval and convert to byte offsets
        int64_t my_old_start = all_old_starts[rank] * in_lower_dims_size;
        int64_t my_new_start = 0;
        if (group_rank[rank] >= 0)
            my_new_start = dist_get_start(new_dim0_global_len, n_dest_ranks,
                                          group_rank[rank]) *
                           out_lower_dims_size;
        int64_t my_old_end =
            (all_old_starts[rank] + old_dim0_local_len) * in_lower_dims_size;
        int64_t my_new_end = 0;
        if (group_rank[rank] >= 0)
            my_new_end = dist_get_end(new_dim0_global_len, n_dest_ranks,
                                      group_rank[rank]) *
                         out_lower_dims_size;

        std::vector<int64_t> send_counts(num_pes);
        std::vector<int64_t> recv_counts(num_pes);
        std::vector<int64_t> send_disp(num_pes);
        std::vector<int64_t> recv_disp(num_pes);

        int64_t curr_send_offset = 0;
        int64_t curr_recv_offset = 0;

        for (int i = 0; i < num_pes; i++) {
            send_disp[i] = curr_send_offset;
            recv_disp[i] = curr_recv_offset;

            // get pe's old and new data interval and convert to byte offsets
            int64_t pe_old_start = all_old_starts[i] * in_lower_dims_size;
            int64_t pe_new_start = 0;
            if (group_rank[i] >= 0)
                pe_new_start = dist_get_start(new_dim0_global_len, n_dest_ranks,
                                              group_rank[i]) *
                               out_lower_dims_size;
            int64_t pe_old_end =
                (all_old_starts[i] + all_old_dim0_local_sizes[i]) *
                in_lower_dims_size;
            int64_t pe_new_end = 0;
            if (group_rank[i] >= 0)
                pe_new_end = dist_get_end(new_dim0_global_len, n_dest_ranks,
                                          group_rank[i]) *
                             out_lower_dims_size;

            send_counts[i] = 0;
            recv_counts[i] = 0;

            // if sending to processor (interval overlap)
            if (pe_new_end > my_old_start && pe_new_start < my_old_end) {
                send_counts[i] = std::min(my_old_end, pe_new_end) -
                                 std::max(my_old_start, pe_new_start);
                curr_send_offset += send_counts[i];
            }

            // if receiving from processor (interval overlap)
            if (my_new_end > pe_old_start && my_new_start < pe_old_end) {
                recv_counts[i] = std::min(pe_old_end, my_new_end) -
                                 std::max(pe_old_start, my_new_start);
                curr_recv_offset += recv_counts[i];
            }
        }

        bodo_alltoallv(input, send_counts, send_disp, MPI_CHAR, output,
                       recv_counts, recv_disp, MPI_CHAR, MPI_COMM_WORLD);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return;
    }
}

#endif  // _DISTRIBUTED_H_INCLUDED
