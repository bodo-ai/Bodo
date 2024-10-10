// Copyright (C) 2019 Bodo Inc. All rights reserved.
#pragma once

#include <Python.h>
#include <mpi.h>
#include <stdbool.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <vector>

#include "_bodo_common.h"
#include "_decimal_ext.h"

#define ROOT_PE 0

// Helper macro to make an MPI call that returns an error code. In case of an
// error, this raises a runtime_error (with MPI error details).
#define HANDLE_MPI_ERROR(CALL, USER_ERR_MSG_PREFIX)                           \
    {                                                                         \
        int err = CALL;                                                       \
        if (err) {                                                            \
            char err_msg[MPI_MAX_ERROR_STRING + 1];                           \
            int err_msg_len = 0;                                              \
            MPI_Error_string(err, err_msg, &err_msg_len);                     \
            throw std::runtime_error(USER_ERR_MSG_PREFIX + std::string(" ") + \
                                     std::string(err_msg, err_msg_len));      \
        }                                                                     \
    }

/**
 * @brief Print (to std::cerr) and raise a detailed error for MPI_Testall with
 * details about the top-level error as well as the individual status-es.
 *
 * @param err Top-level error code.
 * @param req_status_arr List of status-es for each of the requests passed to
 * MPI_Testall.
 * @param user_err_prefix String to use as a prefix for the overall error
 * message.
 */
void print_and_raise_detailed_mpi_test_all_err(
    int err, const std::vector<MPI_Status>& req_status_arr,
    const std::string_view user_err_prefix);

// Helper macro for MPI_Testall. In case of an error, this prints and raises a
// detailed error message about each of the individual status-es.
#define CHECK_MPI_TEST_ALL(requests, flag, USER_ERR_MSG_PREFIX)         \
    {                                                                   \
        std::vector<MPI_Status> req_status_arr(requests.size());        \
        int err = MPI_Testall(requests.size(), requests.data(), &flag,  \
                              req_status_arr.data());                   \
        if (err) {                                                      \
            print_and_raise_detailed_mpi_test_all_err(                  \
                err, req_status_arr, std::string(USER_ERR_MSG_PREFIX)); \
        }                                                               \
    }

// XXX same as distributed_api.py:Reduce_Type
struct BODO_ReduceOps {
    enum ReduceOpsEnum {
        SUM = 0,
        PROD = 1,
        MIN = 2,
        MAX = 3,
        ARGMIN = 4,
        ARGMAX = 5,
        BIT_OR = 6,
        BIT_AND = 7,
        BIT_XOR = 8,
        LOGICAL_OR = 9,
        LOGICAL_AND = 10,
        LOGICAL_XOR = 11
    };
};

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

template <int typ_enum>
static MPI_Datatype get_MPI_typ() __UNUSED__;
static MPI_Datatype get_MPI_typ(int typ_enum) __UNUSED__;
static MPI_Datatype get_val_rank_MPI_typ(int typ_enum) __UNUSED__;
static MPI_Op get_MPI_op(int op_enum) __UNUSED__;
static int get_elem_size(int type_enum) __UNUSED__;
static void timestamptz_reduce(int64_t in_timestamp, int64_t in_offset,
                               int64_t* out_timestamp, int64_t* out_offset,
                               bool is_max) __UNUSED__;
static void dist_reduce(char* in_ptr, char* out_ptr, int op,
                        int type_enum) __UNUSED__;
static void decimal_reduce(int64_t index, uint64_t* in_ptr, char* out_ptr,
                           int op, int type_enum) __UNUSED__;
template <typename Alloc>
static void MPI_Allreduce_bool_or(std::vector<uint8_t, Alloc>& V) __UNUSED__;
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
                    int root) __UNUSED__;

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

/**
 * @brief Performs the communication step of distributed 2D array transpose
 * (alltoallv)
 *
 * @param output output buffer of alltoallv
 * @param input input data buffer with data of target ranks laid out in
 * contiguous chunks
 * @param typ_enum type of data elements (e.g. Bodo_CTypes::FLOAT32)
 * @param n_loc_rows number of local rows in input array
 * @param n_cols number of global columns in input array
 */
void _dist_transpose_comm(char* output, char* input, int typ_enum,
                          int64_t n_loc_rows, int64_t n_cols) __UNUSED__;

static void oneD_reshape_shuffle(char* output, char* input,
                                 int64_t new_dim0_global_len,
                                 int64_t old_dim0_local_len,
                                 int64_t out_lower_dims_size,
                                 int64_t in_lower_dims_size, int n_dest_ranks,
                                 int* dest_ranks) __UNUSED__;

static void permutation_int(int64_t* output, int n) __UNUSED__;
static void permutation_array_index(unsigned char* lhs, uint64_t len,
                                    uint64_t elem_size, unsigned char* rhs,
                                    uint64_t n_elems_rhs, int64_t* p,
                                    uint64_t p_len,
                                    uint64_t n_samples) __UNUSED__;
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
    if (!is_initialized)
        MPI_Init(NULL, NULL);

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
    if (!is_initialized)
        MPI_Init(NULL, NULL);
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

/**
 * Compute the min or the max of TimestampTZ values as an MPI reduction.
 *
 * @param in input value
 * @param out currently reduced value, and the destination for the new reduced
 * value
 * @param len number of elements
 * @param type MPI datatype
 * @param is_max if true, compute the max, otherwise compute the min
 */
static void min_max_timestamptz(void* in, void* out, int* len,
                                MPI_Datatype* type, bool is_max) {
    int64_t* in64 = (int64_t*)in;
    int64_t* out64 = (int64_t*)out;

    int64_t in_value = *in64;
    int32_t in_offset = *(int32_t*)(in64 + 1);

    int64_t out_value = *out64;

    bool in_is_greater = in_value > out_value;

    if (in_is_greater == is_max) {
        *out64 = in_value;
        *(int32_t*)(out64 + 1) = in_offset;
    }
}

// see min_max_timestamptz
static void max_timestamptz(void* in, void* out, int* len, MPI_Datatype* type) {
    min_max_timestamptz(in, out, len, type, true);
}

// see min_max_timestamptz
static void min_timestamptz(void* in, void* out, int* len, MPI_Datatype* type) {
    min_max_timestamptz(in, out, len, type, false);
}

// Distributed reduction of TimestampTZ scalar values - note that only min and
// max are supported.
static void timestamptz_reduce(int64_t in_timestamp, int64_t in_offset,
                               int64_t* out_timestamp, int64_t* out_offset,
                               bool is_max) {
    MPI_Op cmp_ttz;
    MPI_Op_create(is_max ? max_timestamptz : min_timestamptz, 1, &cmp_ttz);

    // If we pack the value and the offset together we can create a stable
    // value for reducing - while only the value is needed for correctness, it
    // can be quite confusing for users if the offset isn't preserved.
    int64_t value = in_timestamp;
    int16_t offset = in_offset;
    MPI_Datatype mpi_typ = MPI_LONG_INT;

    // Determine the size of the pointer to allocate - note that MPI does
    // not use any padding
    constexpr int struct_size = sizeof(int64_t) + sizeof(int32_t);
    int mpi_struct_size;
    MPI_Type_size(mpi_typ, &mpi_struct_size);
    assert(mpi_struct_size == struct_size);

    char in_val[struct_size];
    char out_val[struct_size];

    memcpy(in_val, &value, sizeof(int64_t));
    // Cast offset to i32
    int32_t offset32 = offset;
    memcpy(in_val + sizeof(int64_t), &offset32, sizeof(int32_t));

    MPI_Allreduce(in_val, out_val, 1, mpi_typ, cmp_ttz, MPI_COMM_WORLD);
    MPI_Op_free(&cmp_ttz);

    // Extract timestamp and offset from the reduced value
    *out_timestamp = *((int64_t*)out_val);
    int32_t out_offset32 = *((int32_t*)(out_val + sizeof(int64_t)));
    *out_offset = out_offset32;
}

/**
 * @brief Custom reducer for argmax/argmin on decimal types. Note that for
 * min/max we don't actually care abput the scale if everything is the same so
 * just operates directly on the underlying int128's.
 *
 * @param in The ptr to the current element (For argmax this is a an array of 3
 * uint64's: 1 for the index and 2 to represent an int128/decimal value).
 * @param out Currently reduced value, and the destination for the new reduced
 * value.
 * @param len number of elements.
 * @param type the MPI datatype.
 * @param is_max Whether we are doing an argmax or argmin.
 */
static void argmin_argmax_decimal(void* in, void* out, int* len,
                                  MPI_Datatype* type, bool is_max) {
    uint64_t* in64 = (uint64_t*)in;
    uint64_t* out64 = (uint64_t*)out;

    uint64_t in_index = in64[0];
    uint64_t in_lo = in64[1];
    uint64_t in_hi = in64[2];

    uint64_t out_index = out64[0];
    uint64_t out_lo = out64[1];
    uint64_t out_hi = out64[2];

    __int128 in_val = static_cast<__int128>(in_hi) << 64 | in_lo;
    __int128 out_val = static_cast<__int128>(out_hi) << 64 | out_lo;

    if (in_val == out_val) {
        if (in_index < out_index)
            out64[0] = in_index;
        return;
    }

    bool in_is_greater = in_val > out_val;
    if (in_is_greater == is_max) {
        out64[0] = in_index;
        out64[1] = in_lo;
        out64[2] = in_hi;
    }
}

/**
 * @brief Custom reducer for min/max on decimal types. Note that for
 * min/max we don't actually care abput the scale if everything is the same so
 * just operates directly on the underlying int128's.
 *
 * @param in The ptr to the current element (For min/max this is a an array of 2
 * uint64's to represent an int128/decimal value).
 * @param out Currently reduced value, and the destination for the new reduced
 * value.
 * @param len number of elements.
 * @param type the MPI datatype.
 * @param is_max Whether we are doing an argmax or argmin.
 */
static void min_max_decimal(void* in, void* out, int* len, MPI_Datatype* type,
                            bool is_max) {
    uint64_t* in64 = (uint64_t*)in;
    uint64_t* out64 = (uint64_t*)out;

    uint64_t in_lo = in64[0];
    uint64_t in_hi = in64[1];

    uint64_t out_lo = out64[0];
    uint64_t out_hi = out64[1];

    __int128 in_val = static_cast<__int128>(in_hi) << 64 | in_lo;
    __int128 out_val = static_cast<__int128>(out_hi) << 64 | out_lo;

    bool in_is_greater = in_val > out_val;
    if (in_is_greater == is_max) {
        out64[0] = in_lo;
        out64[1] = in_hi;
    }
}

static void argmin_decimal(void* in, void* out, int* len, MPI_Datatype* type) {
    argmin_argmax_decimal(in, out, len, type, false);
}

static void argmax_decimal(void* in, void* out, int* len, MPI_Datatype* type) {
    argmin_argmax_decimal(in, out, len, type, true);
}

static void min_decimal(void* in, void* out, int* len, MPI_Datatype* type) {
    min_max_decimal(in, out, len, type, false);
}

static void max_decimal(void* in, void* out, int* len, MPI_Datatype* type) {
    min_max_decimal(in, out, len, type, true);
}

static void decimal_reduce(int64_t index, uint64_t* in_ptr, char* out_ptr,
                           int op_enum, int type_enum) {
    try {
        // only supports argmin, argmax, min and min for now
        MPI_Op mpi_op = get_MPI_op(op_enum);
        assert(type_enum == Bodo_CTypes::DECIMAL);
        assert(mpi_op == MPI_MAXLOC || mpi_op == MPI_MINLOC ||
               mpi_op == MPI_MIN || mpi_op == MPI_MAX);

        if (mpi_op == MPI_MIN || mpi_op == MPI_MAX) {
            // create MPI OP corresponding to min/max for decimals
            MPI_Op cmp_decimal;
            MPI_Op_create(mpi_op == MPI_MAX ? max_decimal : min_decimal, 1,
                          &cmp_decimal);

            // type consisting of a decimal (represented as 2 int64's)
            MPI_Datatype decimal_type;
            MPI_Type_contiguous(2, MPI_LONG_LONG_INT, &decimal_type);
            MPI_Type_commit(&decimal_type);

            constexpr int struct_size = 2 * sizeof(uint64_t);
            int mpi_struct_size;
            MPI_Type_size(decimal_type, &mpi_struct_size);
            assert(mpi_struct_size == struct_size);

            char out_val[struct_size];

            uint64_t lo = in_ptr[0];
            uint64_t hi = in_ptr[1];
            __int128 val = static_cast<__int128>(hi) << 64 | lo;

            HANDLE_MPI_ERROR(MPI_Allreduce(&val, out_val, 1, decimal_type,
                                           cmp_decimal, MPI_COMM_WORLD),
                             "_distributed.h::decimal_reduce");

            MPI_Op_free(&cmp_decimal);
            MPI_Type_free(&decimal_type);

            // copy over the min/max decimal into the result
            memcpy(out_ptr, out_val, sizeof(__int128_t));
        } else {
            // create MPI OP corresponding to argmax/argmin for decimals
            MPI_Op argcmp_decimal;
            MPI_Op_create(
                mpi_op == MPI_MAXLOC ? argmax_decimal : argmin_decimal, 1,
                &argcmp_decimal);

            // type consisting of an int64 index and a decimal (represented as 2
            // int64's)
            MPI_Datatype index_decimal_type;
            MPI_Type_contiguous(3, MPI_LONG_LONG_INT, &index_decimal_type);
            MPI_Type_commit(&index_decimal_type);

            constexpr int struct_size = 3 * sizeof(uint64_t);
            int mpi_struct_size;
            MPI_Type_size(index_decimal_type, &mpi_struct_size);
            assert(mpi_struct_size == struct_size);

            char in_val[struct_size];
            char out_val[struct_size];

            uint64_t lo = in_ptr[0];
            uint64_t hi = in_ptr[1];
            __int128 val = static_cast<__int128>(hi) << 64 | lo;

            // remove padding by representing the index, decimal struct as 3
            // int64's
            memcpy(in_val, &index, sizeof(uint64_t));
            memcpy(in_val + sizeof(uint64_t), &val, sizeof(__int128));
            HANDLE_MPI_ERROR(
                MPI_Allreduce(&in_val, out_val, 1, index_decimal_type,
                              argcmp_decimal, MPI_COMM_WORLD),
                "_distributed.h::decimal_reduce");

            MPI_Op_free(&argcmp_decimal);
            MPI_Type_free(&index_decimal_type);

            // copy over the index into the result
            memcpy(out_ptr, out_val, sizeof(uint64_t));
        }
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return;
    }
}

static void dist_reduce(char* in_ptr, char* out_ptr, int op_enum,
                        int type_enum) {
    // printf("reduce value: %d\n", value);
    MPI_Datatype mpi_typ = get_MPI_typ(type_enum);
    MPI_Op mpi_op = get_MPI_op(op_enum);

    // This is a hack because TimestampTZ's scalar representation isn't the same
    // as it's array representation.

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
        if (in_val_rank == NULL)
            return;
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
                    case Bodo_CTypes::_BOOL:
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

    try {
        HANDLE_MPI_ERROR(
            MPI_Allreduce(in_ptr, out_ptr, 1, mpi_typ, mpi_op, MPI_COMM_WORLD),
            "_distributed.h::dist_reduce:");
        return;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return;
    }
}

template <typename Alloc>
static void MPI_Allreduce_bool_or(std::vector<uint8_t, Alloc>& V) {
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
    if (cond)
        MPI_Wait(&req, MPI_STATUS_IGNORE);
}

static void allgather(void* out_data, int size, void* in_data, int type_enum) {
    MPI_Datatype mpi_typ = get_MPI_typ(type_enum);
    MPI_Allgather(in_data, size, mpi_typ, out_data, size, mpi_typ,
                  MPI_COMM_WORLD);
    return;
}

template <int typ_enum>
static MPI_Datatype get_MPI_typ() {
    switch (typ_enum) {
        case Bodo_CTypes::_BOOL:
            return MPI_UNSIGNED_CHAR;  // MPI_C_BOOL doesn't support operations
                                       // like min
        case Bodo_CTypes::INT8:
            return MPI_CHAR;
        case Bodo_CTypes::UINT8:
            return MPI_UNSIGNED_CHAR;
        case Bodo_CTypes::INT32:
        case Bodo_CTypes::DATE:
            return MPI_INT;
        case Bodo_CTypes::UINT32:
            return MPI_UNSIGNED;
        case Bodo_CTypes::INT64:
        case Bodo_CTypes::DATETIME:
        case Bodo_CTypes::TIMEDELTA:
        case Bodo_CTypes::TIMESTAMPTZ:
        // TODO: [BE-4106] Split Time into Time32 and Time64
        case Bodo_CTypes::TIME:
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
            // data type for Decimal128 values (2 64-bit ints)
            static MPI_Datatype decimal_mpi_type = MPI_DATATYPE_NULL;
            // initialize decimal_mpi_type
            // TODO: free when program exits
            if (decimal_mpi_type == MPI_DATATYPE_NULL) {
                MPI_Type_contiguous(2, MPI_LONG_LONG_INT, &decimal_mpi_type);
                MPI_Type_commit(&decimal_mpi_type);
            }
            return decimal_mpi_type;
        case Bodo_CTypes::COMPLEX128:
            return MPI_C_DOUBLE_COMPLEX;
        case Bodo_CTypes::COMPLEX64:
            return MPI_C_FLOAT_COMPLEX;

        default:
            std::cerr << "Invalid MPI_Type " << typ_enum << "\n";
    }
    // dummy value in case of error
    // TODO: raise error properly
    return MPI_LONG_LONG_INT;
}

static MPI_Datatype get_MPI_typ(int typ_enum) {
    switch (typ_enum) {
        case Bodo_CTypes::_BOOL:
            return get_MPI_typ<Bodo_CTypes::_BOOL>();
        case Bodo_CTypes::INT8:
            return get_MPI_typ<Bodo_CTypes::INT8>();
        case Bodo_CTypes::UINT8:
            return get_MPI_typ<Bodo_CTypes::UINT8>();
        case Bodo_CTypes::INT32:
            return get_MPI_typ<Bodo_CTypes::INT32>();
        case Bodo_CTypes::DATE:
            return get_MPI_typ<Bodo_CTypes::DATE>();
        case Bodo_CTypes::UINT32:
            return get_MPI_typ<Bodo_CTypes::UINT32>();
        case Bodo_CTypes::INT64:
            return get_MPI_typ<Bodo_CTypes::INT64>();
        case Bodo_CTypes::DATETIME:
            return get_MPI_typ<Bodo_CTypes::DATETIME>();
        case Bodo_CTypes::TIMEDELTA:
            return get_MPI_typ<Bodo_CTypes::TIMEDELTA>();
        case Bodo_CTypes::TIMESTAMPTZ:
            return get_MPI_typ<Bodo_CTypes::TIMESTAMPTZ>();
        // TODO: [BE-4106] Split Time into Time32 and Time64
        case Bodo_CTypes::TIME:
            return get_MPI_typ<Bodo_CTypes::TIME>();
        case Bodo_CTypes::UINT64:
            return get_MPI_typ<Bodo_CTypes::UINT64>();
        case Bodo_CTypes::FLOAT32:
            return get_MPI_typ<Bodo_CTypes::FLOAT32>();
        case Bodo_CTypes::FLOAT64:
            return get_MPI_typ<Bodo_CTypes::FLOAT64>();
        case Bodo_CTypes::INT16:
            return get_MPI_typ<Bodo_CTypes::INT16>();
        case Bodo_CTypes::UINT16:
            return get_MPI_typ<Bodo_CTypes::UINT16>();
        case Bodo_CTypes::INT128:
            return get_MPI_typ<Bodo_CTypes::INT128>();
        case Bodo_CTypes::DECIMAL:
            return get_MPI_typ<Bodo_CTypes::DECIMAL>();
        case Bodo_CTypes::COMPLEX128:
            return get_MPI_typ<Bodo_CTypes::COMPLEX128>();
        case Bodo_CTypes::COMPLEX64:
            return get_MPI_typ<Bodo_CTypes::COMPLEX64>();
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
    if (typ_enum == Bodo_CTypes::DATETIME || typ_enum == Bodo_CTypes::TIMEDELTA)
        typ_enum = Bodo_CTypes::INT64;
    if (typ_enum == Bodo_CTypes::_BOOL) {
        typ_enum = Bodo_CTypes::INT8;
    }
    if (typ_enum == Bodo_CTypes::DATE) {
        typ_enum = Bodo_CTypes::INT32;
    }
    if (typ_enum == Bodo_CTypes::DECIMAL) {
        static MPI_Datatype decimal_index_mpi_type = MPI_DATATYPE_NULL;
        if (decimal_index_mpi_type == MPI_DATATYPE_NULL) {
            // two int64's representing the decimal and one representing the
            // index
            MPI_Type_contiguous(3, MPI_LONG_LONG_INT, &decimal_index_mpi_type);
            MPI_Type_commit(&decimal_index_mpi_type);
        }
        return decimal_index_mpi_type;
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
    if (op_enum < 0 || op_enum > 11) {
        std::cerr << "Invalid MPI_Op"
                  << "\n";
        return MPI_SUM;
    }
    MPI_Op ops_list[] = {MPI_SUM,    MPI_PROD,   MPI_MIN,  MPI_MAX,
                         MPI_MINLOC, MPI_MAXLOC, MPI_BOR,  MPI_BAND,
                         MPI_BXOR,   MPI_LOR,    MPI_LAND, MPI_LXOR};

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
    if (ind >= start && ind < start + count)
        return ind - start;
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
    if (err != MPI_SUCCESS) {
        PyErr_SetString(PyExc_RuntimeError, "MPI_Comm_group failed");
        return;
    }
    err = MPI_Group_incl(world_group, n, comm_ranks, &new_group);
    if (err != MPI_SUCCESS) {
        PyErr_SetString(PyExc_RuntimeError, "MPI_Group_incl failed");
        return;
    }
    err = MPI_Comm_create(MPI_COMM_WORLD, new_group, comm);
    if (err != MPI_SUCCESS) {
        PyErr_SetString(PyExc_RuntimeError, "MPI_Comm_create failed");
        return;
    }
}

/**
 * MPI_Bcast for all ranks or subset of them
 *
 * @param send_data pointer to data buffer to broadcast
 * @param sendcount number of elements in the data buffer
 * @param typ_enum datatype of buffer
 * @param root rank to broadcast.
 */
static void c_bcast(void* send_data, int sendcount, int typ_enum, int root) {
    MPI_Datatype mpi_typ = get_MPI_typ(typ_enum);
    MPI_Bcast(send_data, sendcount, mpi_typ, root, MPI_COMM_WORLD);
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

static int finalize() {
    int is_initialized;
    MPI_Initialized(&is_initialized);
    if (!is_initialized) {
        return 0;
    }

    // Free user-defined decimal MPI type to avoid leaks
    MPI_Datatype decimal_mpi_type = get_MPI_typ(Bodo_CTypes::DECIMAL);
    MPI_Type_free(&decimal_mpi_type);

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
// following array [9, 8, 6, 4, 11, 7, 2, 3, 5, 0, 1, 10], the function returns
// [2, 1, 0, 0] which are the ranks corresponding to [11, 7, 2, 3] because
// indices [0, 1, 2, 3] go to rank 0, [4, 5, 6, 7] go to rank 1, and
// [8, 9, 10, 11] go to rank 2.
//
// When |n_samples| != |p_len|, only the first |n_samples| elements of the
// permuted output are considered, and the remaining samples are dropped.
// The number of local samples per rank is |n_samples // num_ranks|, and
// the function returns -1 for dropped samples with no destination rank.
// For example, if |n_samples| is 9 using the same example as above, then
// the function returns [-1, 2, 0, 1] because indices [0, 1, 2] go to rank 0,
// [3, 4, 5] go to rank 1, and [6, 7, 8] go to rank 2.
static bodo::vector<int64_t> find_dest_ranks(int64_t rank,
                                             int64_t n_elems_local,
                                             int64_t num_ranks, int64_t* p,
                                             int64_t p_len, int64_t n_samples) {
    // find global start offset of my current chunk of data
    int64_t my_chunk_start = 0;
    // get current chunk sizes of all ranks
    std::vector<int64_t> AllSizes(num_ranks);
    MPI_Allgather(&n_elems_local, 1, MPI_INT64_T, AllSizes.data(), 1,
                  MPI_INT64_T, MPI_COMM_WORLD);
    for (int i = 0; i < rank; i++)
        my_chunk_start += AllSizes[i];

    bodo::vector<int64_t> dest_ranks(n_elems_local);
    // find destination of every element in my chunk based on the permutation,
    // or -1 when there is no destination because p[my_chunk_start + i] >=
    // n_samples
    for (int64_t i = 0; i < n_elems_local; i++) {
        if (p[my_chunk_start + i] >= n_samples) {
            dest_ranks[i] = -1;
        } else {
            dest_ranks[i] =
                index_rank(n_samples, num_ranks, p[my_chunk_start + i]);
        }
    }
    return dest_ranks;
}

static std::vector<int> find_send_counts(
    const std::span<const int64_t> dest_ranks, int64_t num_ranks,
    int64_t elem_size) {
    std::vector<int> send_counts(num_ranks);
    for (auto dest : dest_ranks) {
        if (dest != -1) {
            ++send_counts[dest];
        }
    }
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
// Only take the first |n_samples| elements of the output.
static void permutation_array_index(unsigned char* lhs, uint64_t len,
                                    uint64_t elem_size, unsigned char* rhs,
                                    uint64_t n_elems_rhs, int64_t* p,
                                    uint64_t p_len, uint64_t n_samples) {
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
        // dest_ranks contains the destination rank for each element i in rhs,
        // or -1 for elements with no destination because num_samples < p_len
        // (for cases such as calling random_shuffle with n_samples)
        auto dest_ranks =
            find_dest_ranks(rank, n_elems_rhs, num_ranks, p, p_len, n_samples);
        auto send_counts = find_send_counts(dest_ranks, num_ranks, elem_size);
        auto send_disps = find_disps(send_counts);
        auto recv_counts = find_recv_counts(num_ranks, send_counts);
        auto recv_disps = find_disps(recv_counts);

        auto offsets = send_disps;
        std::vector<unsigned char> send_buf(dest_ranks.size() * elem_size);
        for (size_t i = 0; i < dest_ranks.size(); ++i) {
            if (dest_ranks[i] == -1)
                continue;
            auto send_buf_offset = offsets[dest_ranks[i]]++ * elem_size;
            auto* send_buf_begin = send_buf.data() + send_buf_offset;
            auto* rhs_begin = rhs + i * elem_size;
            std::copy(rhs_begin, rhs_begin + elem_size, send_buf_begin);
        }

        MPI_Alltoallv(send_buf.data(), send_counts.data(), send_disps.data(),
                      element_t, lhs, recv_counts.data(), recv_disps.data(),
                      element_t, MPI_COMM_WORLD);

        // Let us assume that the global data array is [a b c d e f g h] and the
        // permutation array that we want to apply is [2 7 5 6 4 3 1 0]. Then,
        // the target output is [h g a f e c d b], where 'a' goes to the 2nd
        // position, 'b' goes to the 7th position, and so on.
        //
        // Assuming that there are two ranks, each receiving 4 data items, and
        // we are rank 1, after MPI_Alltoallv returns, we receive our chunk
        // [b c d e] of the target output, where the elements are in the same
        // order as the input.
        //
        // To transform [b c d e] into [e c d b], we filter the permutation
        // array for elements in our chunk, then subtract the starting index
        // of our chunk, to get a local permutation with values between 0 and
        // chunk_size: [ 2 (7) (5) (6) (4) 3 1 0 ] => [7 5 6 4] => [3 1 2 0].
        std::vector<size_t> my_p;
        int64_t chunk_size = dist_get_node_portion(n_samples, num_ranks, rank);
        int64_t chunk_start = dist_get_start(n_samples, num_ranks, rank);
        my_p.reserve(chunk_size);

        for (size_t i = 0; i < p_len; ++i) {
            if (chunk_start <= p[i] && p[i] < chunk_start + chunk_size) {
                my_p.push_back(p[i] - chunk_start);
            }
        }

        // We then apply the local permutation [3 1 2 0] to our data chunk
        // [b c d e] to obtain the target output [e c d b].
        apply_permutation(lhs, elem_size, my_p);

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
        if (big_shuffle > 1)
            break;  // error
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
            for (int i = 0; i < num_pes; i++)
                group_rank[i] = i;
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

template <class T>
static void calc_disp(std::vector<T>& disps, std::vector<T> const& counts) {
    size_t n = counts.size();
    disps[0] = 0;
    for (size_t i = 1; i < n; i++) {
        disps[i] = disps[i - 1] + counts[i - 1];
    }
}
