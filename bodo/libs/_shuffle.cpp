// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include "_shuffle.h"
#include <arrow/api.h>
#include <numeric>
#include "_array_hash.h"
#include "_array_utils.h"
#include "_distributed.h"

#undef DEBUG_SHUFFLE
#undef DEBUG_REVERSE_SHUFFLE
#undef DEBUG_GATHER
#undef DEBUG_BROADCAST
#undef DEBUG_BOUND_INFO
#undef DEBUG_COMP_HASH
#undef DEBUG_ARROW_SHUFFLE

void bodo_alltoallv(const void* sendbuf,
                    const std::vector<int64_t>& send_counts,
                    const std::vector<int64_t>& send_disp,
                    MPI_Datatype sendtype, void* recvbuf,
                    const std::vector<int64_t>& recv_counts,
                    const std::vector<int64_t>& recv_disp,
                    MPI_Datatype recvtype, MPI_Comm comm) {
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
                send_ptr + send_count * A2AV_LARGE_DTYPE_SIZE,
                send_counts[dest] * send_typ_size % A2AV_LARGE_DTYPE_SIZE,
                MPI_CHAR, dest, TAG + 1,
                recv_ptr + recv_count * A2AV_LARGE_DTYPE_SIZE,
                recv_counts[src] * recv_typ_size % A2AV_LARGE_DTYPE_SIZE,
                MPI_CHAR, src, TAG + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        MPI_Type_free(&large_dtype);
    }
}

mpi_comm_info::mpi_comm_info(std::vector<array_info*>& _arrays)
    : arrays(_arrays) {
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    n_rows = arrays[0]->length;
    has_nulls = false;
    for (array_info* arr_info : arrays) {
        if (arr_info->arr_type == bodo_array_type::STRING ||
            arr_info->arr_type == bodo_array_type::LIST_STRING ||
            arr_info->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
            has_nulls = true;
            break;
        }
    }
    n_null_bytes = 0;
    // init counts
    send_count = std::vector<int64_t>(n_pes, 0);
    recv_count = std::vector<int64_t>(n_pes);
    send_disp = std::vector<int64_t>(n_pes);
    recv_disp = std::vector<int64_t>(n_pes);
    // init counts for string arrays
    for (array_info* arr_info : arrays) {
        if (arr_info->arr_type == bodo_array_type::STRING ||
            arr_info->arr_type == bodo_array_type::LIST_STRING) {
            send_count_sub.emplace_back(std::vector<int64_t>(n_pes, 0));
            recv_count_sub.emplace_back(std::vector<int64_t>(n_pes));
            send_disp_sub.emplace_back(std::vector<int64_t>(n_pes));
            recv_disp_sub.emplace_back(std::vector<int64_t>(n_pes));
        } else {
            send_count_sub.emplace_back(std::vector<int64_t>());
            recv_count_sub.emplace_back(std::vector<int64_t>());
            send_disp_sub.emplace_back(std::vector<int64_t>());
            recv_disp_sub.emplace_back(std::vector<int64_t>());
        }
        if (arr_info->arr_type == bodo_array_type::LIST_STRING) {
            send_count_sub_sub.emplace_back(std::vector<int64_t>(n_pes, 0));
            recv_count_sub_sub.emplace_back(std::vector<int64_t>(n_pes));
            send_disp_sub_sub.emplace_back(std::vector<int64_t>(n_pes));
            recv_disp_sub_sub.emplace_back(std::vector<int64_t>(n_pes));
        } else {
            send_count_sub_sub.emplace_back(std::vector<int64_t>());
            recv_count_sub_sub.emplace_back(std::vector<int64_t>());
            send_disp_sub_sub.emplace_back(std::vector<int64_t>());
            recv_disp_sub_sub.emplace_back(std::vector<int64_t>());
        }
    }
    if (has_nulls) {
        send_count_null = std::vector<int64_t>(n_pes);
        recv_count_null = std::vector<int64_t>(n_pes);
        send_disp_null = std::vector<int64_t>(n_pes);
        recv_disp_null = std::vector<int64_t>(n_pes);
    }
}

template <class T>
static void calc_disp(std::vector<T>& disps, std::vector<T> const& counts) {
    size_t n = counts.size();
    disps[0] = 0;
    for (size_t i = 1; i < n; i++) disps[i] = disps[i - 1] + counts[i - 1];
}

/*
  Computation of
  ---send_count/recv_count arrays
  ---send_count_sub / recv_count_sub
  ---send_count_sub_sub / recv_count_sub_sub
  Those are used for the shuffling of data1/data2/data3 and their sizes.
 */
void mpi_comm_info::set_counts(uint32_t* hashes) {
    // get send count
    for (size_t i = 0; i < n_rows; i++) {
        size_t node = (size_t)hashes[i] % (size_t)n_pes;
        send_count[node]++;
    }
    // get recv count
    MPI_Alltoall(send_count.data(), 1, MPI_INT64_T, recv_count.data(), 1,
                 MPI_INT64_T, MPI_COMM_WORLD);

    // get displacements
    calc_disp(send_disp, send_count);
    calc_disp(recv_disp, recv_count);

    // counts for string arrays
    for (size_t i = 0; i < arrays.size(); i++) {
        array_info* arr_info = arrays[i];
        if (arr_info->arr_type == bodo_array_type::STRING) {
            // send counts
            std::vector<int64_t>& sub_counts = send_count_sub[i];
            uint32_t* offsets = (uint32_t*)arr_info->data2;
            for (size_t i = 0; i < n_rows; i++) {
                uint32_t str_len = offsets[i + 1] - offsets[i];
                size_t node = (size_t)hashes[i] % (size_t)n_pes;
                sub_counts[node] += str_len;
            }
            // get recv count
            MPI_Alltoall(sub_counts.data(), 1, MPI_INT64_T,
                         recv_count_sub[i].data(), 1, MPI_INT64_T,
                         MPI_COMM_WORLD);
            // get displacements
            calc_disp(send_disp_sub[i], sub_counts);
            calc_disp(recv_disp_sub[i], recv_count_sub[i]);
        }
        if (arr_info->arr_type == bodo_array_type::LIST_STRING) {
            // send counts
            std::vector<int64_t>& sub_counts = send_count_sub[i];
            std::vector<int64_t>& sub_sub_counts = send_count_sub_sub[i];
            uint32_t* index_offsets = (uint32_t*)arr_info->data3;
            uint32_t* data_offsets = (uint32_t*)arr_info->data2;
            for (size_t i = 0; i < n_rows; i++) {
                size_t node = (size_t)hashes[i] % (size_t)n_pes;
                uint32_t len_sub = index_offsets[i + 1] - index_offsets[i];
                uint32_t len_sub_sub = data_offsets[index_offsets[i + 1]] -
                                       data_offsets[index_offsets[i]];
                sub_counts[node] += len_sub;
                sub_sub_counts[node] += len_sub_sub;
            }
            // get recv count_sub
            MPI_Alltoall(sub_counts.data(), 1, MPI_INT64_T,
                         recv_count_sub[i].data(), 1, MPI_INT64_T,
                         MPI_COMM_WORLD);
            // get recv count_sub_sub
            MPI_Alltoall(sub_sub_counts.data(), 1, MPI_INT64_T,
                         recv_count_sub_sub[i].data(), 1, MPI_INT64_T,
                         MPI_COMM_WORLD);
            // get displacements
            calc_disp(send_disp_sub[i], sub_counts);
            calc_disp(recv_disp_sub[i], recv_count_sub[i]);
            calc_disp(send_disp_sub_sub[i], sub_sub_counts);
            calc_disp(recv_disp_sub_sub[i], recv_count_sub_sub[i]);
        }
    }
    if (has_nulls) {
        for (size_t i = 0; i < size_t(n_pes); i++) {
            send_count_null[i] = (send_count[i] + 7) >> 3;
            recv_count_null[i] = (recv_count[i] + 7) >> 3;
        }
        calc_disp(send_disp_null, send_count_null);
        calc_disp(recv_disp_null, recv_count_null);
        n_null_bytes =
            std::accumulate(recv_count_null.begin(), recv_count_null.end(), 0);
    }
    return;
}

template <class T>
static void fill_send_array_inner(T* send_buff, const T* data,
                                  const uint32_t* hashes,
                                  std::vector<int64_t> const& send_disp,
                                  int n_pes, size_t n_rows) {
    std::vector<int64_t> tmp_offset(send_disp);
    for (size_t i = 0; i < n_rows; i++) {
        size_t node = (size_t)hashes[i] % (size_t)n_pes;
        int64_t ind = tmp_offset[node];
        send_buff[ind] = data[i];
        tmp_offset[node]++;
    }
}

static void fill_send_array_inner_decimal(uint8_t* send_buff, uint8_t* data,
                                          uint32_t* hashes,
                                          std::vector<int64_t> const& send_disp,
                                          int n_pes, size_t n_rows) {
    std::vector<int64_t> tmp_offset(send_disp);
    for (size_t i = 0; i < n_rows; i++) {
        size_t node = (size_t)hashes[i] % (size_t)n_pes;
        int64_t ind = tmp_offset[node];
        // send_buff[ind] = data[i];
        memcpy(send_buff + ind * BYTES_PER_DECIMAL,
               data + i * BYTES_PER_DECIMAL, BYTES_PER_DECIMAL);
        tmp_offset[node]++;
    }
}

/*
  This is a function for assigning the output arrays.
  @param send_data_buff   : output array of data
  @param send_length_buff : output array of length (not offsets)
  @param arr_data         : input array of data
  @param arr_offsets      : input array of offsets
  @param hashes           : the hashes of the rows
  @param send_disp        : the sending array of displacements
  @param send_disp_sub    : the sending array of sub displacements
  @param n_pes            : the number of processors
  @param n_rows           : the number of rows.
 */
static void fill_send_array_string_inner(
    char* send_data_buff, uint32_t* send_length_buff, char* arr_data,
    uint32_t* arr_offsets, uint32_t* hashes,
    std::vector<int64_t> const& send_disp,
    std::vector<int64_t> const& send_disp_sub, int n_pes, size_t n_rows) {
    std::vector<int64_t> tmp_offset(send_disp);
    std::vector<int64_t> tmp_offset_sub(send_disp_sub);
    for (size_t i = 0; i < n_rows; i++) {
        size_t node = (size_t)hashes[i] % (size_t)n_pes;
        // write length
        int64_t ind = tmp_offset[node];
        uint32_t str_len = arr_offsets[i + 1] - arr_offsets[i];
        send_length_buff[ind] = str_len;
        tmp_offset[node]++;
        // write data
        int64_t c_ind = tmp_offset_sub[node];
        memcpy(&send_data_buff[c_ind], &arr_data[arr_offsets[i]], str_len);
        tmp_offset_sub[node] += str_len;
    }
}

/*
  The function for setting up the sending arrays in list_string_array case.
  Data should be ordered by processor for being sent and received by the other
  side.
  @param send_data_buff    : the data to be sent.
  @param send_length_data  : the data lengths
  @param send_length_index : the data indexes
  @param arr_data          : the original stored data
  @param arr_data_offsets  : the original stored data offsets
  @param arr_index_offsets : the original stored index offsets
  @param hashes            : the hashes of the rows
  @param send_disp         : the displacements of the index length
  @param send_disp_sub     : the displacements of the data length
  @param send_disp_sub_sub : the displacements of the data
  @param n_pes             : the number of processors
  @param n_rows            : the number of rows.
 */
static void fill_send_array_list_string_inner(
    char* send_data_buff, uint32_t* send_length_data,
    uint32_t* send_length_index, char* arr_data, uint32_t* arr_data_offsets,
    uint32_t* arr_index_offsets, uint32_t* hashes,
    std::vector<int64_t> const& send_disp,
    std::vector<int64_t> const& send_disp_sub,
    std::vector<int64_t> const& send_disp_sub_sub, int n_pes, size_t n_rows) {
    std::vector<int64_t> tmp_offset(send_disp);
    std::vector<int64_t> tmp_offset_sub(send_disp_sub);
    std::vector<int64_t> tmp_offset_sub_sub(send_disp_sub_sub);
    for (size_t i = 0; i < n_rows; i++) {
        size_t node = (size_t)hashes[i] % (size_t)n_pes;
        // Compute the number of strings and the number of characters that will
        // have to be sent.
        int64_t ind = tmp_offset[node];
        uint32_t len_sub = arr_index_offsets[i + 1] - arr_index_offsets[i];
        uint32_t len_sub_sub = arr_data_offsets[arr_index_offsets[i + 1]] -
                               arr_data_offsets[arr_index_offsets[i]];
        // Assigning the number of strings to be sent (len_sub is the number of
        // strings)
        send_length_index[ind] = len_sub;
        tmp_offset[node]++;
        // write the lengths of the strings that will be sent from this
        // processor to others.
        int64_t ind_sub = tmp_offset_sub[node];
        for (uint32_t u = 0; u < len_sub; u++) {
            // size_data is the length of the strings to be sent.
            uint32_t size_data =
                arr_data_offsets[arr_index_offsets[i] + u + 1] -
                arr_data_offsets[arr_index_offsets[i] + u];
            send_length_data[ind_sub + u] = size_data;
        }
        tmp_offset_sub[node] += len_sub;
        // write the set of characters that corresponds to this entry.
        int64_t ind_sub_sub = tmp_offset_sub_sub[node];
        memcpy(&send_data_buff[ind_sub_sub],
               &arr_data[arr_data_offsets[arr_index_offsets[i]]], len_sub_sub);
        tmp_offset_sub_sub[node] += len_sub_sub;
    }
}

static void fill_send_array_null_inner(
    uint8_t* send_null_bitmask, uint8_t* array_null_bitmask, uint32_t* hashes,
    std::vector<int64_t> const& send_disp_null, int n_pes, size_t n_rows) {
    std::vector<int64_t> tmp_offset(n_pes, 0);
    for (size_t i = 0; i < n_rows; i++) {
        size_t node = (size_t)hashes[i] % (size_t)n_pes;
        int64_t ind = tmp_offset[node];
        // write null bit
        bool bit = GetBit(array_null_bitmask, i);
        uint8_t* out_bitmap = &send_null_bitmask[send_disp_null[node]];
        SetBitTo(out_bitmap, ind, bit);
        tmp_offset[node]++;
    }
    return;
}

static void fill_send_array(array_info* send_arr, array_info* in_arr,
                            uint32_t* hashes,
                            std::vector<int64_t> const& send_disp,
                            std::vector<int64_t> const& send_disp_sub,
                            std::vector<int64_t> const& send_disp_sub_sub,
                            std::vector<int64_t> const& send_disp_null,
                            int n_pes) {
    size_t n_rows = (size_t)in_arr->length;
    // dispatch to proper function
    // TODO: general dispatcher
    if (in_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL)
        fill_send_array_null_inner((uint8_t*)send_arr->null_bitmask,
                                   (uint8_t*)in_arr->null_bitmask, hashes,
                                   send_disp_null, n_pes, n_rows);
    if (in_arr->dtype == Bodo_CTypes::_BOOL)
        return fill_send_array_inner<bool>((bool*)send_arr->data1,
                                           (bool*)in_arr->data1, hashes,
                                           send_disp, n_pes, n_rows);
    if (in_arr->dtype == Bodo_CTypes::INT8)
        return fill_send_array_inner<int8_t>((int8_t*)send_arr->data1,
                                             (int8_t*)in_arr->data1, hashes,
                                             send_disp, n_pes, n_rows);
    if (in_arr->dtype == Bodo_CTypes::UINT8)
        return fill_send_array_inner<uint8_t>((uint8_t*)send_arr->data1,
                                              (uint8_t*)in_arr->data1, hashes,
                                              send_disp, n_pes, n_rows);
    if (in_arr->dtype == Bodo_CTypes::INT16)
        return fill_send_array_inner<int16_t>((int16_t*)send_arr->data1,
                                              (int16_t*)in_arr->data1, hashes,
                                              send_disp, n_pes, n_rows);
    if (in_arr->dtype == Bodo_CTypes::UINT16)
        return fill_send_array_inner<uint16_t>((uint16_t*)send_arr->data1,
                                               (uint16_t*)in_arr->data1, hashes,
                                               send_disp, n_pes, n_rows);
    if (in_arr->dtype == Bodo_CTypes::INT32)
        return fill_send_array_inner<int32_t>((int32_t*)send_arr->data1,
                                              (int32_t*)in_arr->data1, hashes,
                                              send_disp, n_pes, n_rows);
    if (in_arr->dtype == Bodo_CTypes::UINT32)
        return fill_send_array_inner<uint32_t>((uint32_t*)send_arr->data1,
                                               (uint32_t*)in_arr->data1, hashes,
                                               send_disp, n_pes, n_rows);
    if (in_arr->dtype == Bodo_CTypes::INT64)
        return fill_send_array_inner<int64_t>((int64_t*)send_arr->data1,
                                              (int64_t*)in_arr->data1, hashes,
                                              send_disp, n_pes, n_rows);
    if (in_arr->dtype == Bodo_CTypes::UINT64)
        return fill_send_array_inner<uint64_t>((uint64_t*)send_arr->data1,
                                               (uint64_t*)in_arr->data1, hashes,
                                               send_disp, n_pes, n_rows);
    if (in_arr->dtype == Bodo_CTypes::DATE ||
        in_arr->dtype == Bodo_CTypes::DATETIME ||
        in_arr->dtype == Bodo_CTypes::TIMEDELTA)
        return fill_send_array_inner<int64_t>((int64_t*)send_arr->data1,
                                              (int64_t*)in_arr->data1, hashes,
                                              send_disp, n_pes, n_rows);
    if (in_arr->dtype == Bodo_CTypes::FLOAT32)
        return fill_send_array_inner<float>((float*)send_arr->data1,
                                            (float*)in_arr->data1, hashes,
                                            send_disp, n_pes, n_rows);
    if (in_arr->dtype == Bodo_CTypes::FLOAT64)
        return fill_send_array_inner<double>((double*)send_arr->data1,
                                             (double*)in_arr->data1, hashes,
                                             send_disp, n_pes, n_rows);
    if (in_arr->dtype == Bodo_CTypes::DECIMAL)
        return fill_send_array_inner_decimal((uint8_t*)send_arr->data1,
                                             (uint8_t*)in_arr->data1, hashes,
                                             send_disp, n_pes, n_rows);
    if (in_arr->arr_type == bodo_array_type::STRING) {
        fill_send_array_string_inner(
            (char*)send_arr->data1, (uint32_t*)send_arr->data2,
            (char*)in_arr->data1, (uint32_t*)in_arr->data2, hashes, send_disp,
            send_disp_sub, n_pes, n_rows);
        fill_send_array_null_inner((uint8_t*)send_arr->null_bitmask,
                                   (uint8_t*)in_arr->null_bitmask, hashes,
                                   send_disp_null, n_pes, n_rows);
        return;
    }
    if (in_arr->arr_type == bodo_array_type::LIST_STRING) {
        fill_send_array_list_string_inner(
            (char*)send_arr->data1, (uint32_t*)send_arr->data2,
            (uint32_t*)send_arr->data3, (char*)in_arr->data1,
            (uint32_t*)in_arr->data2, (uint32_t*)in_arr->data3, hashes,
            send_disp, send_disp_sub, send_disp_sub_sub, n_pes, n_rows);
        fill_send_array_null_inner((uint8_t*)send_arr->null_bitmask,
                                   (uint8_t*)in_arr->null_bitmask, hashes,
                                   send_disp_null, n_pes, n_rows);
        return;
    }
    Bodo_PyErr_SetString(PyExc_RuntimeError, "Invalid data type for send fill");
}

/* Internal function. Convert counts to displacements
 */
static void convert_len_arr_to_offset(uint32_t* offsets,
                                      size_t const& num_strs) {
    uint32_t curr_offset = 0;
    for (size_t i = 0; i < num_strs; i++) {
        uint32_t val = offsets[i];
        offsets[i] = curr_offset;
        curr_offset += val;
    }
    offsets[num_strs] = curr_offset;
}

template <class T>
static void copy_gathered_null_bytes(uint8_t* null_bitmask,
                                     std::vector<uint8_t> const& tmp_null_bytes,
                                     std::vector<T> const& recv_count_null,
                                     std::vector<T> const& recv_count) {
    size_t curr_tmp_byte = 0;  // current location in buffer with all data
    size_t curr_str = 0;       // current string in output bitmap
    // for each chunk
    for (size_t i = 0; i < recv_count.size(); i++) {
        size_t n_strs = recv_count[i];
        size_t n_bytes = recv_count_null[i];
        const uint8_t* chunk_bytes = &tmp_null_bytes[curr_tmp_byte];
        // for each string in chunk
        for (size_t j = 0; j < n_strs; j++) {
            SetBitTo(null_bitmask, curr_str, GetBit(chunk_bytes, j));
            curr_str += 1;
        }
        curr_tmp_byte += n_bytes;
    }
}

// shuffle_array
void shuffle_list_string_null_bitmask(array_info* in_arr, array_info* out_arr,
                                      uint32_t* hashes,
                                      mpi_comm_info const& comm_info) {
    int64_t n_rows = in_arr->length;
    int n_pes = comm_info.n_pes;
    std::vector<int64_t> send_count(n_pes), recv_count(n_pes);
    uint32_t* index_offset = (uint32_t*)in_arr->data3;
    for (int64_t i_row = 0; i_row < n_rows; i_row++) {
        size_t node = (size_t)hashes[i_row] % (size_t)n_pes;
        uint32_t len = index_offset[i_row + 1] - index_offset[i_row];
        send_count[node] += len;
    }
    MPI_Alltoall(send_count.data(), 1, MPI_INT64_T, recv_count.data(), 1,
                 MPI_INT64_T, MPI_COMM_WORLD);
    std::vector<int64_t> send_disp(n_pes), recv_disp(n_pes);
    calc_disp(send_disp, send_count);
    calc_disp(recv_disp, recv_count);
    std::vector<int64_t> send_count_null(n_pes), recv_count_null(n_pes);
    for (int i_p = 0; i_p < n_pes; i_p++) {
        send_count_null[i_p] = (send_count[i_p] + 7) >> 3;
        recv_count_null[i_p] = (recv_count[i_p] + 7) >> 3;
    }
    std::vector<int64_t> send_disp_null(n_pes), recv_disp_null(n_pes);
    calc_disp(send_disp_null, send_count_null);
    calc_disp(recv_disp_null, recv_count_null);
    int64_t send_count_sum =
        std::accumulate(send_count_null.begin(), send_count_null.end(), 0);
    int64_t recv_count_sum =
        std::accumulate(recv_count_null.begin(), recv_count_null.end(), 0);
    std::vector<uint8_t> Vsend(send_count_sum);
    std::vector<uint8_t> Vrecv(recv_count_sum);
    uint32_t pos_index = 0;
    uint8_t* sub_null_bitmap = (uint8_t*)in_arr->sub_null_bitmask;
    std::vector<int64_t> shift(n_pes, 0);
    for (int64_t i_row = 0; i_row < n_rows; i_row++) {
        size_t node = (size_t)hashes[i_row] % (size_t)n_pes;
        uint32_t len = index_offset[i_row + 1] - index_offset[i_row];
        for (uint32_t u = 0; u < len; u++) {
            bool bit = GetBit(sub_null_bitmap, pos_index + u);
            SetBitTo(Vsend.data(), 8 * send_disp_null[node] + shift[node], bit);
            shift[node]++;
        }
        pos_index += len;
    }
    MPI_Datatype mpi_typ8 = get_MPI_typ(Bodo_CTypes::UINT8);
    bodo_alltoallv(Vsend.data(), send_count_null, send_disp_null, mpi_typ8,
                   Vrecv.data(), recv_count_null, recv_disp_null, mpi_typ8,
                   MPI_COMM_WORLD);
    copy_gathered_null_bytes((uint8_t*)out_arr->sub_null_bitmask, Vrecv,
                             recv_count_null, recv_count);
}

static void shuffle_array(array_info* send_arr, array_info* out_arr,
                          std::vector<int64_t> const& send_count,
                          std::vector<int64_t> const& recv_count,
                          std::vector<int64_t> const& send_disp,
                          std::vector<int64_t> const& recv_disp,
                          std::vector<int64_t> const& send_count_sub,
                          std::vector<int64_t> const& recv_count_sub,
                          std::vector<int64_t> const& send_disp_sub,
                          std::vector<int64_t> const& recv_disp_sub,
                          std::vector<int64_t> const& send_count_sub_sub,
                          std::vector<int64_t> const& recv_count_sub_sub,
                          std::vector<int64_t> const& send_disp_sub_sub,
                          std::vector<int64_t> const& recv_disp_sub_sub,
                          std::vector<int64_t> const& send_count_null,
                          std::vector<int64_t> const& recv_count_null,
                          std::vector<int64_t> const& send_disp_null,
                          std::vector<int64_t> const& recv_disp_null,
                          std::vector<uint8_t>& tmp_null_bytes) {
    if (send_arr->arr_type == bodo_array_type::LIST_STRING) {
        // index_offsets
        MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT32);
        bodo_alltoallv(send_arr->data3, send_count, send_disp, mpi_typ,
                       out_arr->data3, recv_count, recv_disp, mpi_typ,
                       MPI_COMM_WORLD);
        convert_len_arr_to_offset((uint32_t*)out_arr->data3,
                                  (size_t)out_arr->length);
        // data_offsets
        bodo_alltoallv(send_arr->data2, send_count_sub, send_disp_sub, mpi_typ,
                       out_arr->data2, recv_count_sub, recv_disp_sub, mpi_typ,
                       MPI_COMM_WORLD);
        uint32_t* data3_uint32 = (uint32_t*)out_arr->data3;
        size_t len = data3_uint32[out_arr->length];
        convert_len_arr_to_offset((uint32_t*)out_arr->data2, len);
        // data
        mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
        bodo_alltoallv(send_arr->data1, send_count_sub_sub, send_disp_sub_sub,
                       mpi_typ, out_arr->data1, recv_count_sub_sub,
                       recv_disp_sub_sub, mpi_typ, MPI_COMM_WORLD);
    }
    if (send_arr->arr_type == bodo_array_type::STRING) {
        // string lengths
        MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT32);
        bodo_alltoallv(send_arr->data2, send_count, send_disp, mpi_typ,
                       out_arr->data2, recv_count, recv_disp, mpi_typ,
                       MPI_COMM_WORLD);
        convert_len_arr_to_offset((uint32_t*)out_arr->data2,
                                  (size_t)out_arr->length);
        // string data
        mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
        bodo_alltoallv(send_arr->data1, send_count_sub, send_disp_sub, mpi_typ,
                       out_arr->data1, recv_count_sub, recv_disp_sub, mpi_typ,
                       MPI_COMM_WORLD);
    }
    if (send_arr->arr_type == bodo_array_type::NUMPY ||
        send_arr->arr_type == bodo_array_type::CATEGORICAL ||
        send_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        MPI_Datatype mpi_typ = get_MPI_typ(send_arr->dtype);
        bodo_alltoallv(send_arr->data1, send_count, send_disp, mpi_typ,
                       out_arr->data1, recv_count, recv_disp, mpi_typ,
                       MPI_COMM_WORLD);
    }
    if (send_arr->arr_type == bodo_array_type::STRING ||
        send_arr->arr_type == bodo_array_type::LIST_STRING ||
        send_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        // nulls
        MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
        bodo_alltoallv(send_arr->null_bitmask, send_count_null, send_disp_null,
                       mpi_typ, tmp_null_bytes.data(), recv_count_null,
                       recv_disp_null, mpi_typ, MPI_COMM_WORLD);
        copy_gathered_null_bytes((uint8_t*)out_arr->null_bitmask,
                                 tmp_null_bytes, recv_count_null, recv_count);
    }
}

/*
  We need a separate function since we depend on the use of polymorphism
  for the computation (not all classes have null_bitmap)
 */
template <typename T>
std::shared_ptr<arrow::Buffer> shuffle_arrow_bitmap_buffer(
    std::vector<int64_t> const& send_count,
    std::vector<int64_t> const& recv_count, int const& n_pes, uint32_t* hashes,
    T const& input_array) {
    size_t n_rows = std::accumulate(send_count.begin(), send_count.end(), 0);
    size_t n_rows_out =
        std::accumulate(recv_count.begin(), recv_count.end(), 0);
    // Computing the null_bitmap.
    // The array support null_bitmap but we cannot test in output if it is
    // indeed nullptr. So, we have to put a null_bitmap no matter what.
    std::vector<int64_t> send_count_null(n_pes);
    std::vector<int64_t> recv_count_null(n_pes);
    for (size_t i = 0; i < size_t(n_pes); i++) {
        send_count_null[i] = (send_count[i] + 7) >> 3;
        recv_count_null[i] = (recv_count[i] + 7) >> 3;
    }
    std::vector<int64_t> send_disp_null(n_pes);
    std::vector<int64_t> recv_disp_null(n_pes);
    calc_disp(send_disp_null, send_count_null);
    calc_disp(recv_disp_null, recv_count_null);
    MPI_Datatype mpi_typ_null = get_MPI_typ(Bodo_CTypes::UINT8);
    std::vector<uint8_t> send_null_bitmask((n_rows + 7) >> 3, 0);
    for (size_t i_row = 0; i_row < n_rows; i_row++)
        SetBitTo(send_null_bitmask.data(), i_row, !input_array->IsNull(i_row));
    int64_t n_row_send_null =
        std::accumulate(send_count_null.begin(), send_count_null.end(), 0);
    int64_t n_row_recv_null =
        std::accumulate(recv_count_null.begin(), recv_count_null.end(), 0);
    std::vector<uint8_t> send_array_null_bitmask(n_row_send_null, 0);
    std::vector<uint8_t> recv_array_null_bitmask(n_row_recv_null, 0);
    fill_send_array_null_inner(send_array_null_bitmask.data(),
                               send_null_bitmask.data(), hashes, send_disp_null,
                               n_pes, n_rows);
    bodo_alltoallv(send_array_null_bitmask.data(), send_count_null,
                   send_disp_null, mpi_typ_null, recv_array_null_bitmask.data(),
                   recv_count_null, recv_disp_null, mpi_typ_null,
                   MPI_COMM_WORLD);
    size_t siz_out = (n_rows_out + 7) >> 3;
    arrow::Result<std::unique_ptr<arrow::Buffer>> maybe_buffer =
        arrow::AllocateBuffer(siz_out);
    if (!maybe_buffer.ok()) {
        Bodo_PyErr_SetString(PyExc_RuntimeError, "allocation error");
        return nullptr;
    }
    std::shared_ptr<arrow::Buffer> buffer = *std::move(maybe_buffer);
    uint8_t* null_bitmask_out_ptr = buffer->mutable_data();
    copy_gathered_null_bytes(null_bitmask_out_ptr, recv_array_null_bitmask,
                             recv_count_null, recv_count);
    return buffer;
}

/*
  We need a separate function since we depends on the use of polymorphism
  for the computation (Not all classes have null_bitmap)
 */
template <typename T>
std::shared_ptr<arrow::Buffer> shuffle_arrow_offset_buffer(
    std::vector<int64_t> const& send_count,
    std::vector<int64_t> const& recv_count, uint32_t* hashes, int const& n_pes,
    T const& input_array) {
    size_t n_rows = std::accumulate(send_count.begin(), send_count.end(), 0);
    size_t n_rows_out =
        std::accumulate(recv_count.begin(), recv_count.end(), 0);
    std::vector<int64_t> send_disp(n_pes);
    std::vector<int64_t> recv_disp(n_pes);
    calc_disp(send_disp, send_count);
    calc_disp(recv_disp, recv_count);
    std::vector<int64_t> send_len(n_rows);
    std::vector<int64_t> list_shift = send_disp;
    for (size_t i_row = 0; i_row < n_rows; i_row++) {
        size_t node = (size_t)hashes[i_row] % (size_t)n_pes;
        int64_t off1 = input_array->value_offset(i_row);
        int64_t off2 = input_array->value_offset(i_row + 1);
        int32_t e_len = off2 - off1;
        send_len[list_shift[node]] = e_len;
        list_shift[node]++;
    }
    std::vector<int64_t> recv_len(n_rows_out);
    MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::INT64);
    bodo_alltoallv(send_len.data(), send_count, send_disp, mpi_typ,
                   recv_len.data(), recv_count, recv_disp, mpi_typ,
                   MPI_COMM_WORLD);
    size_t siz_out = sizeof(int32_t) * (n_rows_out + 1);
    arrow::Result<std::unique_ptr<arrow::Buffer>> maybe_buffer =
        arrow::AllocateBuffer(siz_out);
    if (!maybe_buffer.ok()) {
        Bodo_PyErr_SetString(PyExc_RuntimeError, "allocation error");
        return nullptr;
    }
    std::shared_ptr<arrow::Buffer> buffer = *std::move(maybe_buffer);
    uint8_t* list_offsets_ptr_ui8 = buffer->mutable_data();
    int32_t* list_offsets_ptr = (int32_t*)list_offsets_ptr_ui8;
    int32_t pos = 0;
    list_offsets_ptr[0] = 0;
    for (size_t i_row = 0; i_row < n_rows_out; i_row++) {
        pos += recv_len[i_row];
        list_offsets_ptr[i_row + 1] = pos;
    }
    return buffer;
}

template <typename T>
std::vector<uint32_t> map_hashes_array(std::vector<int64_t> const& send_count,
                                       std::vector<int64_t> const& recv_count,
                                       uint32_t* hashes, int const& n_pes,
                                       T const& input_array) {
    size_t n_rows = std::accumulate(send_count.begin(), send_count.end(), 0);
    size_t n_ent = input_array->value_offset(n_rows);
    std::vector<uint32_t> hashes_out(n_ent);
    for (size_t i_row = 0; i_row < n_rows; i_row++) {
        uint32_t e_hash = hashes[i_row];
        int64_t off1 = input_array->value_offset(i_row);
        int64_t off2 = input_array->value_offset(i_row + 1);
        for (int64_t idx = off1; idx < off2; idx++) hashes_out[idx] = e_hash;
    }
    return hashes_out;
}

std::shared_ptr<arrow::Buffer> shuffle_arrow_primitive_buffer(
    std::vector<int64_t> const& send_count,
    std::vector<int64_t> const& recv_count, uint32_t* hashes, int const& n_pes,
    std::shared_ptr<arrow::PrimitiveArray> const& input_array) {
    // Typing stuff
    arrow::Type::type typ = input_array->type()->id();
    Bodo_CTypes::CTypeEnum dtype = arrow_to_bodo_type(typ);
    uint64_t siztype = numpy_item_size[dtype];
    MPI_Datatype mpi_typ = get_MPI_typ(dtype);
    // Setting up the arrays
    size_t n_rows = std::accumulate(send_count.begin(), send_count.end(), 0);
    size_t n_rows_out =
        std::accumulate(recv_count.begin(), recv_count.end(), 0);
    std::vector<int64_t> send_disp(n_pes);
    std::vector<int64_t> recv_disp(n_pes);
    calc_disp(send_disp, send_count);
    calc_disp(recv_disp, recv_count);
    std::vector<char> send_arr(n_rows * siztype);
    char* values = (char*)input_array->values()->data();
    std::vector<int64_t> tmp_offset(send_disp);
    for (size_t i = 0; i < n_rows; i++) {
        size_t node = (size_t)hashes[i] % (size_t)n_pes;
        int64_t ind = tmp_offset[node];
        memcpy(send_arr.data() + ind * siztype, values + i * siztype, siztype);
        tmp_offset[node]++;
    }
    // Allocating returning arrays
    size_t siz_out = siztype * n_rows_out;
    arrow::Result<std::unique_ptr<arrow::Buffer>> maybe_buffer =
        arrow::AllocateBuffer(siz_out);
    if (!maybe_buffer.ok()) {
        Bodo_PyErr_SetString(PyExc_RuntimeError, "allocation error");
        return nullptr;
    }
    std::shared_ptr<arrow::Buffer> buffer = *std::move(maybe_buffer);
    // Doing the exchanges
    char* data_ptr = (char*)buffer->mutable_data();
    bodo_alltoallv(send_arr.data(), send_count, send_disp, mpi_typ, data_ptr,
                   recv_count, recv_disp, mpi_typ, MPI_COMM_WORLD);
    return buffer;
}

std::shared_ptr<arrow::Buffer> shuffle_string_buffer(
    std::vector<int64_t> const& send_count,
    std::vector<int64_t> const& recv_count, uint32_t* hashes, int const& n_pes,
    std::shared_ptr<arrow::StringArray> const& string_array) {
    size_t n_rows = std::accumulate(send_count.begin(), send_count.end(), 0);
    std::vector<int64_t> send_count_char(n_pes, 0);
    std::vector<int64_t> recv_count_char(n_pes);
    for (size_t i_row = 0; i_row < n_rows; i_row++) {
        size_t node = (size_t)hashes[i_row] % (size_t)n_pes;
        std::string e_str = string_array->GetString(i_row);
        int64_t n_char = e_str.size();
        send_count_char[node] += n_char;
    }
    MPI_Alltoall(send_count_char.data(), 1, MPI_INT64_T, recv_count_char.data(),
                 1, MPI_INT64_T, MPI_COMM_WORLD);
    std::vector<int64_t> send_disp_char(n_pes);
    std::vector<int64_t> recv_disp_char(n_pes);
    calc_disp(send_disp_char, send_count_char);
    calc_disp(recv_disp_char, recv_count_char);
    int64_t n_chars_send_tot =
        std::accumulate(send_count_char.begin(), send_count_char.end(), 0);
    int64_t n_chars_recv_tot =
        std::accumulate(recv_count_char.begin(), recv_count_char.end(), 0);
    char* send_char = new char[n_chars_send_tot];
    size_t siz_out = sizeof(char) * n_chars_recv_tot;
    arrow::Result<std::unique_ptr<arrow::Buffer>> maybe_buffer =
        arrow::AllocateBuffer(siz_out);
    if (!maybe_buffer.ok()) {
        Bodo_PyErr_SetString(PyExc_RuntimeError, "allocation error");
        return nullptr;
    }
    std::shared_ptr<arrow::Buffer> buffer = *std::move(maybe_buffer);
    char* recv_char = (char*)buffer->mutable_data();
    std::vector<int64_t> list_shift = send_disp_char;
    for (size_t i_row = 0; i_row < n_rows; i_row++) {
        size_t node = (size_t)hashes[i_row] % (size_t)n_pes;
        std::string e_str = string_array->GetString(i_row);
        int64_t n_char = e_str.size();
        for (int64_t i_char = 0; i_char < n_char; i_char++) {
            send_char[list_shift[node] + i_char] = e_str.data()[i_char];
        }
        list_shift[node] += n_char;
    }
    MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
    bodo_alltoallv(send_char, send_count_char, send_disp_char, mpi_typ,
                   recv_char, recv_count_char, recv_disp_char, mpi_typ,
                   MPI_COMM_WORLD);
    delete[] send_char;
    return buffer;
}

/*
  We need to exchange arrays between nodes.
  We use dynamic allocation since the scheme is recursive and reusability of
  arrays appear difficult to do.
  ---
  The hashes specify how the nodes will be sent.
 */
std::shared_ptr<arrow::Array> shuffle_arrow_array(
    std::shared_ptr<arrow::Array> input_array, uint32_t* hashes, int n_pes) {
    // Computing total number of rows on output
    // Note that the number of rows, counts and hashes varies according
    // to the array in the recursive structure
    size_t n_rows = (size_t)input_array->length();
    std::vector<int64_t> send_count(n_pes, 0);
    std::vector<int64_t> recv_count(n_pes);
    for (size_t i_row = 0; i_row < n_rows; i_row++) {
        size_t node = (size_t)hashes[i_row] % (size_t)n_pes;
        send_count[node]++;
    }
    MPI_Alltoall(send_count.data(), 1, MPI_INT64_T, recv_count.data(), 1,
                 MPI_INT64_T, MPI_COMM_WORLD);
    int64_t n_rows_out =
        std::accumulate(recv_count.begin(), recv_count.end(), 0);
    //
    //
    if (input_array->type_id() == arrow::Type::LIST) {
        // Converting the pointer
        std::shared_ptr<arrow::ListArray> list_array =
            std::dynamic_pointer_cast<arrow::ListArray>(input_array);
        // Computing the offsets, hashes
        std::shared_ptr<arrow::Buffer> list_offsets =
            shuffle_arrow_offset_buffer(send_count, recv_count, hashes, n_pes,
                                        list_array);
        std::vector<uint32_t> hashes_out =
            map_hashes_array(send_count, recv_count, hashes, n_pes, list_array);
        // Now computing the bitmap
        std::shared_ptr<arrow::Buffer> null_bitmap_out =
            shuffle_arrow_bitmap_buffer(send_count, recv_count, n_pes, hashes,
                                        list_array);
        // Computing the child_array
        std::shared_ptr<arrow::Array> child_array =
            shuffle_arrow_array(list_array->values(), hashes_out.data(), n_pes);
        // Now returning the shuffled array
        return std::make_shared<arrow::ListArray>(list_array->type(),
                                                  n_rows_out, list_offsets,
                                                  child_array, null_bitmap_out);

    } else if (input_array->type_id() == arrow::Type::STRUCT) {
        // Converting to the right pointer type.
        auto struct_array =
            std::dynamic_pointer_cast<arrow::StructArray>(input_array);
        auto struct_type =
            std::dynamic_pointer_cast<arrow::StructType>(struct_array->type());
        // Now computing the children arrays
        std::vector<std::shared_ptr<arrow::Array>> children;
        for (int64_t i = 0; i < struct_type->num_fields(); i++)
            children.push_back(
                shuffle_arrow_array(struct_array->field(i), hashes, n_pes));
        // Now computing the bitmap
        std::shared_ptr<arrow::Buffer> null_bitmap_out =
            shuffle_arrow_bitmap_buffer(send_count, recv_count, n_pes, hashes,
                                        struct_array);
        // Now returning the arrays
        return std::make_shared<arrow::StructArray>(
            struct_array->type(), n_rows_out, children, null_bitmap_out);
    } else if (input_array->type_id() == arrow::Type::STRING) {
        // Converting pointers
        auto string_array =
            std::dynamic_pointer_cast<arrow::StringArray>(input_array);
        // Now computing the offsets
        std::shared_ptr<arrow::Buffer> list_offsets =
            shuffle_arrow_offset_buffer(send_count, recv_count, hashes, n_pes,
                                        string_array);
        // Now computing the bitmap
        std::shared_ptr<arrow::Buffer> null_bitmap_out =
            shuffle_arrow_bitmap_buffer(send_count, recv_count, n_pes, hashes,
                                        string_array);
        // Now computing the characters
        std::shared_ptr<arrow::Buffer> data = shuffle_string_buffer(
            send_count, recv_count, hashes, n_pes, string_array);
        // Now returning the array
        return std::make_shared<arrow::StringArray>(n_rows_out, list_offsets,
                                                    data, null_bitmap_out);
    } else {
        // Converting pointers
        auto primitive_array =
            std::dynamic_pointer_cast<arrow::PrimitiveArray>(input_array);
        // Now computing the data
        std::shared_ptr<arrow::Buffer> data = shuffle_arrow_primitive_buffer(
            send_count, recv_count, hashes, n_pes, primitive_array);
        // Now computing the bitmap
        std::shared_ptr<arrow::Buffer> null_bitmap_out =
            shuffle_arrow_bitmap_buffer(send_count, recv_count, n_pes, hashes,
                                        primitive_array);
        return std::make_shared<arrow::PrimitiveArray>(
            primitive_array->type(), n_rows_out, data, null_bitmap_out);
    }
}

/*
  A prerequisite for the function to run correctly is that the set_counts
  method has been used and set correctly. It determines which sizes go to
  each processor. It performs a determination of the the sending and receiving
  arrays.
  ---
  1) The first step is to accumulate the sizes from each processors.
  2) Then for each row a number of operations are done:
  2a) First set up the send_arr by aligning the data from the input columns
  accordingly. 2b) Second do the shuffling of data between all processors.
 */
table_info* shuffle_table_kernel(table_info* in_table, uint32_t* hashes,
                                 mpi_comm_info const& comm_info) {
    int n_pes = comm_info.n_pes;
    int64_t total_recv = std::accumulate(comm_info.recv_count.begin(),
                                         comm_info.recv_count.end(), 0);
    size_t n_cols = (size_t)in_table->ncols();
    std::vector<int64_t> n_sub_recvs(n_cols);
    for (size_t i = 0; i < n_cols; i++)
        n_sub_recvs[i] = std::accumulate(comm_info.recv_count_sub[i].begin(),
                                         comm_info.recv_count_sub[i].end(), 0);
    std::vector<int64_t> n_sub_sub_recvs(n_cols);
    for (size_t i = 0; i < n_cols; i++)
        n_sub_sub_recvs[i] =
            std::accumulate(comm_info.recv_count_sub_sub[i].begin(),
                            comm_info.recv_count_sub_sub[i].end(), 0);

    // fill send buffer and send
    std::vector<array_info*> out_arrs;
    std::vector<uint8_t> tmp_null_bytes(comm_info.n_null_bytes);
    for (size_t i = 0; i < n_cols; i++) {
        array_info* in_arr = in_table->columns[i];
        array_info* out_arr;
        if (in_arr->arr_type != bodo_array_type::ARROW) {
            array_info* send_arr =
                alloc_array(in_table->nrows(), in_arr->n_sub_elems,
                            in_arr->n_sub_sub_elems, in_arr->arr_type,
                            in_arr->dtype, 2 * n_pes, in_arr->num_categories);
            out_arr = alloc_array(total_recv, n_sub_recvs[i],
                                  n_sub_sub_recvs[i], in_arr->arr_type,
                                  in_arr->dtype, 0, in_arr->num_categories);
            fill_send_array(send_arr, in_arr, hashes, comm_info.send_disp,
                            comm_info.send_disp_sub[i],
                            comm_info.send_disp_sub_sub[i],
                            comm_info.send_disp_null, n_pes);

            shuffle_array(
                send_arr, out_arr, comm_info.send_count, comm_info.recv_count,
                comm_info.send_disp, comm_info.recv_disp,
                comm_info.send_count_sub[i], comm_info.recv_count_sub[i],
                comm_info.send_disp_sub[i], comm_info.recv_disp_sub[i],
                comm_info.send_count_sub_sub[i],
                comm_info.recv_count_sub_sub[i], comm_info.send_disp_sub_sub[i],
                comm_info.recv_disp_sub_sub[i], comm_info.send_count_null,
                comm_info.recv_count_null, comm_info.send_disp_null,
                comm_info.recv_disp_null, tmp_null_bytes);
            if (in_arr->arr_type == bodo_array_type::LIST_STRING)
                shuffle_list_string_null_bitmask(in_arr, out_arr, hashes,
                                                 comm_info);
            delete_info_decref_array(send_arr);
        } else {
            std::shared_ptr<arrow::Array> out_array =
                shuffle_arrow_array(in_arr->array, hashes, n_pes);
            uint64_t n_items =
                out_array
                    ->length();  // Should get the value from the output array
            NRT_MemInfo* meminfo = NULL;
            out_arr = new array_info(
                bodo_array_type::ARROW, Bodo_CTypes::INT8 /*dummy*/, n_items,
                -1, -1, NULL, NULL, NULL, NULL, NULL, meminfo, NULL, out_array);
        }
        // release reference of input array
        // This a a steal reference case. The idea is to release memory as soon
        // as possible. If this release is not wished (which is a rare case)
        // then the incref before operation is needed. Using an optional
        // argument (like decref_input) to the input is a false good idea since
        // it changes the semantics to something different from Python.
        decref_array(in_arr);
        out_arrs.push_back(out_arr);
    }

    return new table_info(out_arrs);
}

array_info* reverse_shuffle_numpy_array(array_info* in_arr, uint32_t* hashes,
                                        mpi_comm_info const& comm_info) {
    uint64_t siztype = numpy_item_size[in_arr->dtype];
    MPI_Datatype mpi_typ = get_MPI_typ(in_arr->dtype);
    size_t n_rows_ret = std::accumulate(comm_info.send_count.begin(),
                                        comm_info.send_count.end(), 0);
    array_info* out_arr = alloc_array(n_rows_ret, 0, 0, in_arr->arr_type,
                                      in_arr->dtype, 0, in_arr->num_categories);
    char* data1_i = in_arr->data1;
    char* data1_o = out_arr->data1;
    std::vector<char> tmp_recv(out_arr->length * siztype);
    bodo_alltoallv(data1_i, comm_info.recv_count, comm_info.recv_disp, mpi_typ,
                   tmp_recv.data(), comm_info.send_count, comm_info.send_disp,
                   mpi_typ, MPI_COMM_WORLD);
    std::vector<int64_t> tmp_offset(comm_info.send_disp);
    for (size_t i = 0; i < n_rows_ret; i++) {
        size_t node = (size_t)hashes[i] % (size_t)comm_info.n_pes;
        int64_t ind = tmp_offset[node];
        memcpy(data1_o + siztype * i, tmp_recv.data() + siztype * ind, siztype);
        tmp_offset[node]++;
    }
    return out_arr;
}

array_info* reverse_shuffle_string_array(array_info* in_arr, uint32_t* hashes,
                                         mpi_comm_info const& comm_info) {
    // 1: computing the recv_count_sub and related
    uint32_t* in_offset = (uint32_t*)in_arr->data2;
    int n_pes = comm_info.n_pes;
    std::vector<int64_t> recv_count_sub(n_pes),
        recv_disp_sub(n_pes);  // we continue using here the recv/send
    for (int i = 0; i < n_pes; i++)
        recv_count_sub[i] =
            in_offset[comm_info.recv_disp[i] + comm_info.recv_count[i]] -
            in_offset[comm_info.recv_disp[i]];
    std::vector<int64_t> send_count_sub(n_pes), send_disp_sub(n_pes);
  
    MPI_Alltoall(recv_count_sub.data(), 1, MPI_INT64_T, send_count_sub.data(), 1,
                 MPI_INT64_T, MPI_COMM_WORLD);

    calc_disp(send_disp_sub, send_count_sub);
    calc_disp(recv_disp_sub, recv_count_sub);
    // 2: allocating the array
    int64_t n_count_sub = send_disp_sub[n_pes - 1] + send_count_sub[n_pes - 1];
    int64_t n_rows_ret = std::accumulate(comm_info.send_count.begin(),
                                         comm_info.send_count.end(), 0);
    array_info* out_arr =
        alloc_array(n_rows_ret, n_count_sub, 0, in_arr->arr_type, in_arr->dtype,
                    0, in_arr->num_categories);
    int64_t in_len = in_arr->length;
    int64_t out_len = out_arr->length;
    // 3: the offsets
    std::vector<uint32_t> list_len_send(in_len);
    uint32_t* out_offset = (uint32_t*)out_arr->data2;
    for (int64_t i = 0; i < in_len; i++)
        list_len_send[i] = in_offset[i + 1] - in_offset[i];
    MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT32);
    std::vector<uint32_t> list_len_recv(out_len);
    bodo_alltoallv(list_len_send.data(), comm_info.recv_count,
                   comm_info.recv_disp, mpi_typ, list_len_recv.data(),
                   comm_info.send_count, comm_info.send_disp, mpi_typ,
                   MPI_COMM_WORLD);
    fill_recv_data_inner(list_len_recv.data(), out_offset, hashes,
                         comm_info.send_disp, comm_info.n_pes, out_len);
    convert_len_arr_to_offset(out_offset, out_len);
    // 4: the characters themselves
    int64_t tot_char =
        std::accumulate(send_count_sub.begin(), send_count_sub.end(), 0);
    std::vector<uint8_t> tmp_recv(tot_char);
    mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
    bodo_alltoallv(in_arr->data1, recv_count_sub, recv_disp_sub, mpi_typ,
                   tmp_recv.data(), send_count_sub, send_disp_sub, mpi_typ,
                   MPI_COMM_WORLD);
    std::vector<int64_t> tmp_offset_sub(send_disp_sub);
    for (int64_t i = 0; i < out_len; i++) {
        size_t node = (size_t)hashes[i] % (size_t)n_pes;
        uint32_t str_len = out_offset[i + 1] - out_offset[i];
        int64_t c_ind = tmp_offset_sub[node];
        char* out_ptr = out_arr->data1 + out_offset[i];
        char* in_ptr = (char*)tmp_recv.data() + c_ind;
        memcpy(out_ptr, in_ptr, str_len);
        tmp_offset_sub[node] += str_len;
    }
    return out_arr;
}

void reverse_shuffle_null_bitmap_array(array_info* in_arr, array_info* out_arr,
                                       uint32_t* hashes,
                                       mpi_comm_info const& comm_info) {
    int n_pes = comm_info.n_pes;
    std::vector<int64_t> send_count_null(n_pes), recv_count_null(n_pes);
    for (int i = 0; i < n_pes; i++) {
        send_count_null[i] = (comm_info.send_count[i] + 7) >> 3;
        recv_count_null[i] = (comm_info.recv_count[i] + 7) >> 3;
    }
    int64_t n_send_null_tot =
        std::accumulate(send_count_null.begin(), send_count_null.end(), 0);
    int64_t n_recv_null_tot =
        std::accumulate(recv_count_null.begin(), recv_count_null.end(), 0);
    std::vector<int64_t> send_disp_null(n_pes), recv_disp_null(n_pes);
    calc_disp(send_disp_null, send_count_null);
    calc_disp(recv_disp_null, recv_count_null);
    std::vector<uint8_t> mask_send(n_recv_null_tot);
    uint8_t* null_bitmask_in = (uint8_t*)in_arr->null_bitmask;
    uint8_t* null_bitmask_out = (uint8_t*)out_arr->null_bitmask;
    int64_t pos = 0;
    for (int i = 0; i < n_pes; i++) {
        for (int64_t i_row = 0; i_row < comm_info.recv_count[i]; i_row++) {
            bool bit = GetBit(null_bitmask_in, pos);
            SetBitTo(mask_send.data(), 8 * recv_disp_null[i] + i_row, bit);
            pos++;
        }
    }
    std::vector<uint8_t> mask_recv(n_send_null_tot);
    MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
    bodo_alltoallv(mask_send.data(), recv_count_null, recv_disp_null, mpi_typ,
                   mask_recv.data(), send_count_null, send_disp_null, mpi_typ,
                   MPI_COMM_WORLD);
    std::vector<int64_t> tmp_offset(n_pes, 0);
    for (int64_t i_row = 0; i_row < out_arr->length; i_row++) {
        size_t node = (size_t)hashes[i_row] % (size_t)n_pes;
        uint8_t* out_bitmap = &(mask_recv.data())[send_disp_null[node]];
        bool bit = GetBit(out_bitmap, tmp_offset[node]);
        SetBitTo(null_bitmask_out, i_row, bit);
        tmp_offset[node]++;
    }
}

array_info* reverse_shuffle_list_string_array(array_info* in_arr,
                                              uint32_t* hashes,
                                              mpi_comm_info const& comm_info) {
    // 1: computing the recv_count_sub and related
    int n_pes = comm_info.n_pes;
    uint32_t* in_str_offset = (uint32_t*)in_arr->data3;
    std::vector<int64_t> recv_count_sub(n_pes),
        recv_disp_sub(n_pes);  // we continue using here the recv/send
    for (int i = 0; i < n_pes; i++)
        recv_count_sub[i] =
            in_str_offset[comm_info.recv_disp[i] + comm_info.recv_count[i]] -
            in_str_offset[comm_info.recv_disp[i]];
    std::vector<int64_t> send_count_sub(n_pes), send_disp_sub(n_pes);
    MPI_Alltoall(recv_count_sub.data(), 1, MPI_INT64_T, send_count_sub.data(),
                 1, MPI_INT64_T, MPI_COMM_WORLD);
    calc_disp(send_disp_sub, send_count_sub);
    calc_disp(recv_disp_sub, recv_count_sub);
    // 2: computing the recv_count_sub_sub and related
    uint32_t* in_data_offset = (uint32_t*)in_arr->data2;
    std::vector<int64_t> recv_count_sub_sub(n_pes),
        recv_disp_sub_sub(n_pes);  // we continue using here the recv/send
    for (int i = 0; i < n_pes; i++)
        recv_count_sub_sub[i] =
            in_data_offset[recv_disp_sub[i] + recv_count_sub[i]] -
            in_data_offset[recv_disp_sub[i]];
    std::vector<int64_t> send_count_sub_sub(n_pes), send_disp_sub_sub(n_pes);
    MPI_Alltoall(recv_count_sub_sub.data(), 1, MPI_INT64_T,
                 send_count_sub_sub.data(), 1, MPI_INT64_T, MPI_COMM_WORLD);
    calc_disp(send_disp_sub_sub, send_count_sub_sub);
    calc_disp(recv_disp_sub_sub, recv_count_sub_sub);
    // 3: Now allocating
    int64_t n_rows_ret = std::accumulate(comm_info.send_count.begin(),
                                         comm_info.send_count.end(), 0);
    int64_t n_count_sub = send_disp_sub[n_pes - 1] + send_count_sub[n_pes - 1];
    int64_t n_count_sub_sub =
        send_disp_sub_sub[n_pes - 1] + send_count_sub_sub[n_pes - 1];
    array_info* out_arr =
        alloc_array(n_rows_ret, n_count_sub, n_count_sub_sub, in_arr->arr_type,
                    in_arr->dtype, 0, in_arr->num_categories);
    int64_t in_len = in_arr->length;
    int64_t out_len = out_arr->length;
    // 4: the string offsets
    std::vector<uint32_t> list_str_len_send(in_len);
    uint32_t* out_str_offset = (uint32_t*)out_arr->data3;
    for (int64_t i = 0; i < in_len; i++)
        list_str_len_send[i] = in_str_offset[i + 1] - in_str_offset[i];
    MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT32);
    std::vector<uint32_t> list_str_len_recv(out_len);
    bodo_alltoallv(list_str_len_send.data(), comm_info.recv_count,
                   comm_info.recv_disp, mpi_typ, list_str_len_recv.data(),
                   comm_info.send_count, comm_info.send_disp, mpi_typ,
                   MPI_COMM_WORLD);
    fill_recv_data_inner(list_str_len_recv.data(), out_str_offset, hashes,
                         comm_info.send_disp, comm_info.n_pes, out_len);
    convert_len_arr_to_offset(out_str_offset, out_len);
    // 5: the character offsets
    int32_t in_sub_len = in_str_offset[in_len];
    int32_t out_sub_len = out_str_offset[out_len];
    std::vector<uint32_t> list_char_len_send(in_sub_len);
    uint32_t* out_data_offset = (uint32_t*)out_arr->data2;
    for (int64_t i = 0; i < in_sub_len; i++)
        list_char_len_send[i] = in_data_offset[i + 1] - in_data_offset[i];
    std::vector<uint32_t> list_char_len_recv(out_sub_len);
    bodo_alltoallv(list_char_len_send.data(), recv_count_sub, recv_disp_sub,
                   mpi_typ, list_char_len_recv.data(), send_count_sub,
                   send_disp_sub, mpi_typ, MPI_COMM_WORLD);
    std::vector<int64_t> tmp_offset_sub(send_disp_sub);
    for (int64_t i = 0; i < out_len; i++) {
        size_t node = (size_t)hashes[i] % (size_t)n_pes;
        uint32_t nb_str = out_str_offset[i + 1] - out_str_offset[i];
        int64_t c_ind = tmp_offset_sub[node];
        uint32_t* out_ptr = out_data_offset + out_str_offset[i];
        uint32_t* in_ptr = list_char_len_recv.data() + c_ind;
        memcpy((char*)out_ptr, (char*)in_ptr, sizeof(uint32_t) * nb_str);
        tmp_offset_sub[node] += nb_str;
    }
    convert_len_arr_to_offset(out_data_offset, out_sub_len);
    // 6: the characters themselves
    int32_t out_sub_sub_len = out_data_offset[out_sub_len];
    char* in_char = in_arr->data1;
    char* out_char = out_arr->data1;
    MPI_Datatype mpi_typ8 = get_MPI_typ(Bodo_CTypes::UINT8);
    std::vector<char> list_char_recv(out_sub_sub_len);
    bodo_alltoallv(in_char, recv_count_sub_sub, recv_disp_sub_sub, mpi_typ8,
                   list_char_recv.data(), send_count_sub_sub, send_disp_sub_sub,
                   mpi_typ8, MPI_COMM_WORLD);
    std::vector<int64_t> tmp_offset_sub_sub(send_disp_sub_sub);
    for (int64_t i = 0; i < out_len; i++) {
        size_t node = (size_t)hashes[i] % (size_t)n_pes;
        uint32_t nb_char = out_data_offset[out_str_offset[i + 1]] -
                           out_data_offset[out_str_offset[i]];
        int64_t c_ind = tmp_offset_sub_sub[node];
        char* out_ptr = out_char + out_data_offset[out_str_offset[i]];
        char* in_ptr = list_char_recv.data() + c_ind;
        memcpy(out_ptr, in_ptr, nb_char);
        tmp_offset_sub_sub[node] += nb_char;
    }
    //
    // Now doing the mask
    //
    std::vector<int64_t> send_count_sub_null(n_pes), recv_count_sub_null(n_pes);
    for (int i = 0; i < n_pes; i++) {
        send_count_sub_null[i] = (send_count_sub[i] + 7) >> 3;
        recv_count_sub_null[i] = (recv_count_sub[i] + 7) >> 3;
    }
    int64_t n_send_sub_null_tot = std::accumulate(send_count_sub_null.begin(),
                                                  send_count_sub_null.end(), 0);
    int64_t n_recv_sub_null_tot = std::accumulate(recv_count_sub_null.begin(),
                                                  recv_count_sub_null.end(), 0);
    std::vector<int64_t> send_disp_sub_null(n_pes), recv_disp_sub_null(n_pes);
    calc_disp(send_disp_sub_null, send_count_sub_null);
    calc_disp(recv_disp_sub_null, recv_count_sub_null);
    std::vector<uint8_t> mask_send(n_recv_sub_null_tot);
    uint8_t* sub_null_bitmask_in = (uint8_t*)in_arr->sub_null_bitmask;
    uint8_t* sub_null_bitmask_out = (uint8_t*)out_arr->sub_null_bitmask;
    for (int i_p = 0; i_p < n_pes; i_p++) {
        for (int64_t i_str = 0; i_str < recv_count_sub[i_p]; i_str++) {
            int64_t pos = i_str + recv_disp_sub[i_p];
            bool bit = GetBit(sub_null_bitmask_in, pos);
            SetBitTo(mask_send.data(), 8 * recv_disp_sub_null[i_p] + i_str,
                     bit);
        }
    }
    std::vector<uint8_t> mask_recv(n_send_sub_null_tot);
    bodo_alltoallv(mask_send.data(), recv_count_sub_null, recv_disp_sub_null,
                   mpi_typ8, mask_recv.data(), send_count_sub_null,
                   send_disp_sub_null, mpi_typ8, MPI_COMM_WORLD);
    std::vector<int64_t> tmp_offset(n_pes, 0);
    int64_t pos_si = 0;
    for (int64_t i_row = 0; i_row < out_arr->length; i_row++) {
        size_t node = (size_t)hashes[i_row] % (size_t)n_pes;
        uint8_t* sub_out_bitmap = &(mask_recv.data())[send_disp_sub_null[node]];
        int64_t n_str = out_str_offset[i_row + 1] - out_str_offset[i_row];
        for (int64_t i_str = 0; i_str < n_str; i_str++) {
            bool bit = GetBit(sub_out_bitmap, tmp_offset[node] + i_str);
            SetBitTo(sub_null_bitmask_out, pos_si, bit);
            pos_si++;
        }
        tmp_offset[node] += n_str;
    }
    return out_arr;
}

/** When we do the shuffle operation the end result is to have
    ROWS FROM 0 / ROWS FROM 1 / ROWS FROM 2 ... / ROWS FROM (n_pes-1)
    Thus the data is in consecutive blocks.
    As a consequence of that, we cannot use the existing infrastructure
    for making the reverse shuffle.
    There is no way we can build a uint32_t* compute_reverse_shuffle(uint32_t*
   hashes)
    ---
    @param in_table  : The shuffled input table
    @param hashes    : the hashes (of the original table)
    @param comm_info : The comm_info (computed from the original table)
    @return the reshuffled table
 */
table_info* reverse_shuffle_table_kernel(table_info* in_table, uint32_t* hashes,
                                         mpi_comm_info const& comm_info) {
#ifdef DEBUG_REVERSE_SHUFFLE
    std::cout << "Beginning of reverse_shuffle_table_kernel. in_table:\n";
    DEBUG_PrintSetOfColumn(std::cout, in_table->columns);
    DEBUG_PrintRefct(std::cout, in_table->columns);
#endif
    size_t n_cols = in_table->ncols();
    std::vector<array_info*> out_arrs;
    for (size_t i = 0; i < n_cols; i++) {
        array_info* in_arr = in_table->columns[i];
        bodo_array_type::arr_type_enum arr_type = in_arr->arr_type;
        array_info* out_arr = nullptr;
        if (in_arr->arr_type == bodo_array_type::ARROW) {
            Bodo_PyErr_SetString(
                PyExc_RuntimeError,
                "Reverse shuffle for arrow data not yet supported");
            return nullptr;
        } else {
            if (arr_type == bodo_array_type::NUMPY ||
                arr_type == bodo_array_type::CATEGORICAL ||
                arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
                out_arr =
                    reverse_shuffle_numpy_array(in_arr, hashes, comm_info);
            }
            if (arr_type == bodo_array_type::STRING) {
                out_arr =
                    reverse_shuffle_string_array(in_arr, hashes, comm_info);
            }
            if (arr_type == bodo_array_type::LIST_STRING) {
                out_arr = reverse_shuffle_list_string_array(in_arr, hashes,
                                                            comm_info);
            }
            if (arr_type == bodo_array_type::STRING ||
                arr_type == bodo_array_type::LIST_STRING ||
                arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
                reverse_shuffle_null_bitmap_array(in_arr, out_arr, hashes,
                                                  comm_info);
            }
        }
        // Reference stealing see shuffle_table_kernel for discussion
        decref_array(in_arr);
        out_arrs.push_back(out_arr);
    }
#ifdef DEBUG_REVERSE_SHUFFLE
    std::cout << "Ending of reverse_shuffle_table_kernel. out_arrs:\n";
    DEBUG_PrintSetOfColumn(std::cout, out_arrs);
    DEBUG_PrintRefct(std::cout, out_arrs);
#endif
    return new table_info(out_arrs);
}

// Note: Steals a reference from the input table.
table_info* shuffle_table(table_info* in_table, int64_t n_keys) {
    // error checking
    if (in_table->ncols() <= 0 || n_keys <= 0) {
        Bodo_PyErr_SetString(PyExc_RuntimeError, "Invalid input shuffle table");
        return NULL;
    }
#ifdef DEBUG_REVERSE_SHUFFLE
    std::cout << "IN_TABLE (SHUFFLE):\n";
    DEBUG_PrintSetOfColumn(std::cout, in_table->columns);
    DEBUG_PrintRefct(std::cout, in_table->columns);
#endif
    mpi_comm_info comm_info(in_table->columns);
    // computing the hash data structure
    uint32_t* hashes = hash_keys_table(in_table, n_keys, SEED_HASH_PARTITION);
    comm_info.set_counts(hashes);

    table_info* table = shuffle_table_kernel(in_table, hashes, comm_info);
    delete[] hashes;
#ifdef DEBUG_SHUFFLE
    std::cout << "RET_TABLE (SHUFFLE):\n";
    DEBUG_PrintSetOfColumn(std::cout, table->columns);
    DEBUG_PrintRefct(std::cout, table->columns);
    std::cout << "After the print statements\n";
#endif
    return table;
}

table_info* coherent_shuffle_table(table_info* in_table, table_info* ref_table,
                                   int64_t n_keys) {
    // error checking
    if (in_table->ncols() <= 0 || n_keys <= 0) {
        Bodo_PyErr_SetString(PyExc_RuntimeError, "Invalid input shuffle table");
        return NULL;
    }
#ifdef DEBUG_SHUFFLE
    std::cout << "IN_TABLE (COHERENT SHUFFLE):\n";
    DEBUG_PrintSetOfColumn(std::cout, in_table->columns);
    DEBUG_PrintRefct(std::cout, in_table->columns);
    std::cout << "REF_TABLE (COHERENT SHUFFLE):\n";
    DEBUG_PrintRefct(std::cout, ref_table->columns);
#endif
    mpi_comm_info comm_info(in_table->columns);
    // computing the hash data structure
    uint32_t* hashes = coherent_hash_keys_table(in_table, ref_table, n_keys,
                                                SEED_HASH_PARTITION);
    comm_info.set_counts(hashes);
    table_info* table = shuffle_table_kernel(in_table, hashes, comm_info);
    delete[] hashes;
#ifdef DEBUG_REVERSE_SHUFFLE
    std::cout << "RET_TABLE (SHUFFLE):\n";
    DEBUG_PrintSetOfColumn(std::cout, table->columns);
    DEBUG_PrintRefct(std::cout, table->columns);
#endif
    return table;
}

template <typename T>
std::shared_ptr<arrow::Buffer> broadcast_arrow_bitmap_buffer(int64_t n_rows,
                                                             T const& arr) {
    int myrank, mpi_root = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    int n_bytes = (n_rows + 7) >> 3;
    size_t siz_out = n_bytes;
    arrow::Result<std::unique_ptr<arrow::Buffer>> maybe_buffer =
        arrow::AllocateBuffer(siz_out);
    if (!maybe_buffer.ok()) {
        Bodo_PyErr_SetString(PyExc_RuntimeError, "allocation error");
        return nullptr;
    }
    std::shared_ptr<arrow::Buffer> buffer = *std::move(maybe_buffer);
    uint8_t* null_bitmask_out_ptr = buffer->mutable_data();
    if (myrank == mpi_root) {
        for (int i_row = 0; i_row < n_rows; i_row++)
            SetBitTo(null_bitmask_out_ptr, i_row, !arr->IsNull(i_row));
    }
    MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
    MPI_Bcast(null_bitmask_out_ptr, n_bytes, mpi_typ, mpi_root, MPI_COMM_WORLD);
    return buffer;
}

template <typename T>
std::shared_ptr<arrow::Buffer> broadcast_arrow_offsets_buffer(int64_t n_rows,
                                                              T const& arr) {
    int myrank, mpi_root = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    size_t siz_out = sizeof(int32_t) * (n_rows + 1);
    arrow::Result<std::unique_ptr<arrow::Buffer>> maybe_buffer =
        arrow::AllocateBuffer(siz_out);
    if (!maybe_buffer.ok()) {
        Bodo_PyErr_SetString(PyExc_RuntimeError, "allocation error");
        return nullptr;
    }
    std::shared_ptr<arrow::Buffer> buffer = *std::move(maybe_buffer);
    uint32_t* offset_out_ptr = (uint32_t*)buffer->mutable_data();
    if (myrank == mpi_root) {
        for (int i_row = 0; i_row <= n_rows; i_row++)
            offset_out_ptr[i_row] = arr->value_offset(i_row);
    }
    MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::INT32);
    MPI_Bcast(offset_out_ptr, n_rows + 1, mpi_typ, mpi_root, MPI_COMM_WORLD);
    return buffer;
}

std::shared_ptr<arrow::Buffer> broadcast_arrow_primitive_buffer(
    int64_t n_rows, std::shared_ptr<arrow::PrimitiveArray> const& arr) {
    int myrank, mpi_root = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    // broadcasting type size
    int64_t siz_typ = 0;
    if (myrank == mpi_root) {
        arrow::Type::type typ = arr->type()->id();
        Bodo_CTypes::CTypeEnum bodo_typ = arrow_to_bodo_type(typ);
        siz_typ = numpy_item_size[bodo_typ];
    }
    MPI_Bcast(&siz_typ, 1, MPI_LONG_LONG_INT, mpi_root, MPI_COMM_WORLD);
    //
    size_t siz_out = siz_typ * n_rows;
    arrow::Result<std::unique_ptr<arrow::Buffer>> maybe_buffer =
        arrow::AllocateBuffer(siz_out);
    if (!maybe_buffer.ok()) {
        Bodo_PyErr_SetString(PyExc_RuntimeError, "allocation error");
        return nullptr;
    }
    std::shared_ptr<arrow::Buffer> buffer = *std::move(maybe_buffer);
    uint8_t* data_out_ptr = buffer->mutable_data();
    if (myrank == mpi_root) {
        uint8_t* data_in_ptr = (uint8_t*)arr->values()->data();
        memcpy(data_out_ptr, data_in_ptr, siz_typ * n_rows);
    }
    MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
    MPI_Bcast(data_out_ptr, siz_typ * n_rows, mpi_typ, mpi_root,
              MPI_COMM_WORLD);
    return buffer;
}

std::shared_ptr<arrow::Buffer> broadcast_arrow_string_buffer(
    int64_t n_rows, std::shared_ptr<arrow::StringArray> const& string_array) {
    int myrank, mpi_root = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    // broadcasting number of characters
    int64_t arr_typ[1];
    if (myrank == mpi_root) arr_typ[0] = string_array->value_offset(n_rows);
    MPI_Bcast(arr_typ, 1, MPI_LONG_LONG_INT, mpi_root, MPI_COMM_WORLD);
    int64_t n_chars = arr_typ[0];
    //
    size_t siz_out = n_chars;
    arrow::Result<std::unique_ptr<arrow::Buffer>> maybe_buffer =
        arrow::AllocateBuffer(siz_out);
    if (!maybe_buffer.ok()) {
        Bodo_PyErr_SetString(PyExc_RuntimeError, "allocation error");
        return nullptr;
    }
    std::shared_ptr<arrow::Buffer> buffer = *std::move(maybe_buffer);
    uint8_t* data_out_ptr = buffer->mutable_data();
    if (myrank == mpi_root) {
        std::shared_ptr<arrow::Buffer> data_buff = string_array->value_data();
        uint8_t* data_in_ptr = (uint8_t*)data_buff->data();
        memcpy(data_out_ptr, data_in_ptr, n_chars);
    }
    MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
    MPI_Bcast(data_out_ptr, n_chars, mpi_typ, mpi_root, MPI_COMM_WORLD);
    return buffer;
}

/* Broadcasting the arrow array to other nodes.

   @param ref_arr : the arrow array used for the datatype
   @param arr : the arrow array put in input
   @return the arrow array put in all the nodes
*/
std::shared_ptr<arrow::Array> broadcast_arrow_array(
    std::shared_ptr<arrow::Array> const& ref_arr,
    std::shared_ptr<arrow::Array> const& arr) {
    int myrank, mpi_root = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    int64_t arr_bcast[2];
    if (myrank == mpi_root) {
        arr_bcast[0] = arr->length();
        int64_t typ = arr->type_id();
        arr_bcast[1] = typ;
    }
    MPI_Bcast(arr_bcast, 2, MPI_LONG_LONG_INT, mpi_root, MPI_COMM_WORLD);
    int64_t n_rows = arr_bcast[0];
    arrow::Type::type typ_arrow = arrow::Type::type(arr_bcast[1]);
    //
    if (typ_arrow == arrow::Type::LIST) {
        std::shared_ptr<arrow::ListArray> list_arr = nullptr;
        std::shared_ptr<arrow::ListArray> ref_list_arr =
            std::dynamic_pointer_cast<arrow::ListArray>(ref_arr);
        std::shared_ptr<arrow::Array> arr_values = nullptr;
        if (myrank == mpi_root) {
            list_arr = std::dynamic_pointer_cast<arrow::ListArray>(arr);
            arr_values = list_arr->values();
        }
        std::shared_ptr<arrow::Buffer> list_offsets =
            broadcast_arrow_offsets_buffer(n_rows, list_arr);
        std::shared_ptr<arrow::Buffer> null_bitmap =
            broadcast_arrow_bitmap_buffer(n_rows, list_arr);
        std::shared_ptr<arrow::Array> child_array =
            broadcast_arrow_array(ref_list_arr->values(), arr_values);
        // Now returning the broadcast array
        return std::make_shared<arrow::ListArray>(ref_list_arr->type(), n_rows,
                                                  list_offsets, child_array,
                                                  null_bitmap);
    } else if (typ_arrow == arrow::Type::STRUCT) {
        std::shared_ptr<arrow::StructArray> struct_arr = nullptr;
        std::shared_ptr<arrow::StructArray> ref_struct_arr =
            std::dynamic_pointer_cast<arrow::StructArray>(ref_arr);
        int64_t num_fields = 0;
        if (myrank == mpi_root) {
            struct_arr = std::dynamic_pointer_cast<arrow::StructArray>(arr);
            auto struct_type = std::dynamic_pointer_cast<arrow::StructType>(
                struct_arr->type());
            num_fields = struct_type->num_fields();
        }
        MPI_Bcast(&num_fields, 1, MPI_LONG_LONG_INT, mpi_root, MPI_COMM_WORLD);
        std::vector<std::shared_ptr<arrow::Array>> children;
        for (int i_field = 0; i_field < int(num_fields); i_field++) {
            std::shared_ptr<arrow::Array> child = nullptr;
            std::shared_ptr<arrow::Array> ref_child =
                ref_struct_arr->field(i_field);
            if (myrank == mpi_root) child = struct_arr->field(i_field);
            children.push_back(broadcast_arrow_array(ref_child, child));
        }
        std::shared_ptr<arrow::Buffer> null_bitmap =
            broadcast_arrow_bitmap_buffer(n_rows, struct_arr);
        return std::make_shared<arrow::StructArray>(
            ref_struct_arr->type(), n_rows, children, null_bitmap);
    } else if (typ_arrow == arrow::Type::STRING) {
        std::shared_ptr<arrow::StringArray> string_array = nullptr;
        if (myrank == mpi_root) {
            string_array = std::dynamic_pointer_cast<arrow::StringArray>(arr);
        }
        std::shared_ptr<arrow::Buffer> list_offsets =
            broadcast_arrow_offsets_buffer(n_rows, string_array);
        std::shared_ptr<arrow::Buffer> null_bitmap =
            broadcast_arrow_bitmap_buffer(n_rows, string_array);
        std::shared_ptr<arrow::Buffer> data =
            broadcast_arrow_string_buffer(n_rows, string_array);
        return std::make_shared<arrow::StringArray>(n_rows, list_offsets, data,
                                                    null_bitmap);
    } else {
        std::shared_ptr<arrow::PrimitiveArray> primitive_arr = nullptr;
        std::shared_ptr<arrow::PrimitiveArray> ref_primitive_arr =
            std::dynamic_pointer_cast<arrow::PrimitiveArray>(ref_arr);
        if (myrank == mpi_root)
            primitive_arr =
                std::dynamic_pointer_cast<arrow::PrimitiveArray>(arr);
        std::shared_ptr<arrow::Buffer> data =
            broadcast_arrow_primitive_buffer(n_rows, primitive_arr);
        std::shared_ptr<arrow::Buffer> null_bitmap =
            broadcast_arrow_bitmap_buffer(n_rows, primitive_arr);
        return std::make_shared<arrow::PrimitiveArray>(
            ref_primitive_arr->type(), n_rows, data, null_bitmap);
    }
}

/* Broadcasting the first n_cols of in_table to the other nodes.
   The ref_table contains only the type information. In order to eliminate it,
   we would need to have a broadcast_datatype function

   @param ref_table : the reference table used for the datatype
   @param in_table : the table that is broadcasted.
   @param n_cols : the number of columns in output
   @return the table put in all the nodes
*/
table_info* broadcast_table(table_info* ref_table, table_info* in_table,
                            size_t n_cols) {
#ifdef DEBUG_BROADCAST
    std::cout << "INPUT of broadcast_table. ref_table=\n";
    DEBUG_PrintRefct(std::cout, ref_table->columns);
    DEBUG_PrintSetOfColumn(std::cout, ref_table->columns);
    if (myrank == mpi_root) {
        std::cout << "in_table=\n";
        DEBUG_PrintRefct(std::cout, in_table->columns);
        DEBUG_PrintSetOfColumn(std::cout, in_table->columns);
    }
#endif
    int n_pes, myrank;
    int mpi_root = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    std::vector<array_info*> out_arrs;
    for (size_t i_col = 0; i_col < n_cols; i_col++) {
        int64_t arr_bcast[6];
        array_info* in_arr = nullptr;
        if (myrank == mpi_root) {
            in_arr = in_table->columns[i_col];
            arr_bcast[0] = in_arr->length;
            arr_bcast[1] = in_arr->dtype;
            arr_bcast[2] = in_arr->arr_type;
            arr_bcast[3] = in_arr->n_sub_elems;
            arr_bcast[4] = in_arr->n_sub_sub_elems;
            arr_bcast[5] = in_arr->num_categories;
        }
        MPI_Bcast(arr_bcast, 6, MPI_LONG_LONG_INT, mpi_root, MPI_COMM_WORLD);
        int64_t n_rows = arr_bcast[0];
        Bodo_CTypes::CTypeEnum dtype = Bodo_CTypes::CTypeEnum(arr_bcast[1]);
        bodo_array_type::arr_type_enum arr_type =
            bodo_array_type::arr_type_enum(arr_bcast[2]);
        int64_t n_sub_elems = arr_bcast[3];
        int64_t n_sub_sub_elems = arr_bcast[4];
        int64_t num_categories = arr_bcast[5];
        //
        array_info* out_arr;
        if (arr_type == bodo_array_type::ARROW) {
            std::shared_ptr<arrow::Array> ref_array =
                ref_table->columns[i_col]->array;
            std::shared_ptr<arrow::Array> in_array = nullptr;
            if (myrank == mpi_root) in_array = in_arr->array;
            std::shared_ptr<arrow::Array> array =
                broadcast_arrow_array(ref_array, in_array);
            uint64_t n_rows = array->length();
            NRT_MemInfo* meminfo = NULL;
            out_arr = new array_info(
                bodo_array_type::ARROW, Bodo_CTypes::INT8 /*dummy*/, n_rows, -1,
                -1, NULL, NULL, NULL, NULL, NULL, meminfo, NULL, array);
        }
        if (arr_type == bodo_array_type::NUMPY ||
            arr_type == bodo_array_type::CATEGORICAL ||
            arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
            MPI_Datatype mpi_typ = get_MPI_typ(dtype);
            if (myrank == mpi_root)
                out_arr = copy_array(in_arr);
            else
                out_arr = alloc_array(n_rows, -1, -1, arr_type, dtype, 0, 0);
            MPI_Bcast(out_arr->data1, n_rows, mpi_typ, mpi_root,
                      MPI_COMM_WORLD);
        }
        if (arr_type == bodo_array_type::STRING) {
            MPI_Datatype mpi_typ32 = get_MPI_typ(Bodo_CTypes::UINT32);
            MPI_Datatype mpi_typ8 = get_MPI_typ(Bodo_CTypes::UINT8);
            if (myrank == mpi_root)
                out_arr = copy_array(in_arr);
            else
                out_arr = alloc_array(n_rows, n_sub_elems, n_sub_sub_elems,
                                      arr_type, dtype, 0, num_categories);
            MPI_Bcast(out_arr->data1, n_sub_elems, mpi_typ8, mpi_root,
                      MPI_COMM_WORLD);
            MPI_Bcast(out_arr->data2, n_rows, mpi_typ32, mpi_root,
                      MPI_COMM_WORLD);
        }
        if (arr_type == bodo_array_type::LIST_STRING) {
            MPI_Datatype mpi_typ32 = get_MPI_typ(Bodo_CTypes::UINT32);
            MPI_Datatype mpi_typ8 = get_MPI_typ(Bodo_CTypes::UINT8);
            if (myrank == mpi_root)
                out_arr = copy_array(in_arr);
            else
                out_arr = alloc_array(n_rows, n_sub_elems, n_sub_sub_elems,
                                      arr_type, dtype, 0, num_categories);
            MPI_Bcast(out_arr->data1, n_sub_sub_elems, mpi_typ8, mpi_root,
                      MPI_COMM_WORLD);
            MPI_Bcast(out_arr->data2, n_sub_elems, mpi_typ32, mpi_root,
                      MPI_COMM_WORLD);
            MPI_Bcast(out_arr->data3, n_rows, mpi_typ32, mpi_root,
                      MPI_COMM_WORLD);
            MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
            int n_sub_bytes = (n_sub_elems + 7) >> 3;
            MPI_Bcast(out_arr->sub_null_bitmask, n_sub_bytes, mpi_typ, mpi_root,
                      MPI_COMM_WORLD);
        }
        if (arr_type == bodo_array_type::NULLABLE_INT_BOOL ||
            arr_type == bodo_array_type::STRING ||
            arr_type == bodo_array_type::LIST_STRING) {
            MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
            int n_bytes = (n_rows + 7) >> 3;
            MPI_Bcast(out_arr->null_bitmask, n_bytes, mpi_typ, mpi_root,
                      MPI_COMM_WORLD);
        }
        out_arrs.push_back(out_arr);
        // Standard stealing of reference. See shuffle_table_kernel for
        // discussion. The table is not-null only on mpi_root.
        if (myrank == mpi_root) decref_array(in_arr);
    }
#ifdef DEBUG_BROADCAST
    std::cout << "OUTPUT of broadcast_table. out_arrs=\n";
    DEBUG_PrintRefct(std::cout, out_arrs);
    DEBUG_PrintSetOfColumn(std::cout, out_arrs);
#endif
    return new table_info(out_arrs);
}

int MPI_Gengather(void* sendbuf, int sendcount, MPI_Datatype sendtype,
                  void* recvbuf, int recvcount, MPI_Datatype recvtype,
                  int root_pe, MPI_Comm comm, bool all_gather) {
    if (all_gather) {
        return MPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                             recvtype, comm);
    } else {
        return MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                          recvtype, root_pe, comm);
    }
}

int MPI_Gengatherv(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                   void* recvbuf, const int* recvcounts, const int* displs,
                   MPI_Datatype recvtype, int root_pe, MPI_Comm comm,
                   bool all_gather) {
    if (all_gather) {
        return MPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts,
                              displs, recvtype, comm);
    } else {
        return MPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts,
                           displs, recvtype, root_pe, comm);
    }
}

template <typename T>
std::shared_ptr<arrow::Buffer> gather_arrow_offset_buffer(T const& arr,
                                                          bool all_gather) {
    int n_pes, myrank, mpi_root = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    std::vector<int> rows_count, rows_disps;
    if (myrank == mpi_root || all_gather) {
        rows_count.resize(n_pes);
        rows_disps.resize(n_pes);
    }
    int n_rows = arr->length();
    MPI_Gengather(&n_rows, 1, MPI_INT, rows_count.data(), 1, MPI_INT, mpi_root,
                  MPI_COMM_WORLD, all_gather);
    int64_t n_rows_tot =
        std::accumulate(rows_count.begin(), rows_count.end(), 0);
    if (myrank == mpi_root || all_gather) {
        int64_t rows_pos = 0;
        for (int i_p = 0; i_p < n_pes; i_p++) {
            rows_disps[i_p] = rows_pos;
            rows_pos += rows_count[i_p];
        }
    }
    std::vector<int32_t> list_siz_loc(n_rows);
    for (int32_t i = 0; i < n_rows; i++)
        list_siz_loc[i] = arr->value_offset(i + 1) - arr->value_offset(i);
    MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::INT32);
    std::vector<int32_t> list_siz_tot(n_rows_tot);
    MPI_Gengatherv(list_siz_loc.data(), n_rows, mpi_typ, list_siz_tot.data(),
                   rows_count.data(), rows_disps.data(), mpi_typ, mpi_root,
                   MPI_COMM_WORLD, all_gather);
    if (myrank == mpi_root || all_gather) {
        size_t siz_out = sizeof(int32_t) * (n_rows_tot + 1);
        arrow::Result<std::unique_ptr<arrow::Buffer>> maybe_buffer =
            arrow::AllocateBuffer(siz_out);
        if (!maybe_buffer.ok()) {
            Bodo_PyErr_SetString(PyExc_RuntimeError, "allocation error");
            return nullptr;
        }
        std::shared_ptr<arrow::Buffer> buffer = *std::move(maybe_buffer);
        uint32_t* data_out_ptr = (uint32_t*)buffer->mutable_data();
        uint32_t pos = 0;
        for (int64_t i = 0; i <= n_rows_tot; i++) {
            data_out_ptr[i] = pos;
            pos += list_siz_tot[i];
        }
        return buffer;
    }
    return nullptr;
}

template <typename T>
std::shared_ptr<arrow::Buffer> gather_arrow_bitmap_buffer(T const& arr,
                                                          bool all_gather) {
    int n_pes, myrank, mpi_root = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    std::vector<int> rows_count, recv_count_null, recv_disp_null;
    if (myrank == mpi_root || all_gather) {
        rows_count.resize(n_pes);
        recv_count_null.resize(n_pes);
        recv_disp_null.resize(n_pes);
    }
    int n_rows = arr->length();
    MPI_Gengather(&n_rows, 1, MPI_INT, rows_count.data(), 1, MPI_INT, mpi_root,
                  MPI_COMM_WORLD, all_gather);
    int n_rows_tot = std::accumulate(rows_count.begin(), rows_count.end(), 0);
    if (myrank == mpi_root || all_gather) {
        for (size_t i = 0; i < size_t(n_pes); i++)
            recv_count_null[i] = (rows_count[i] + 7) >> 3;
        calc_disp(recv_disp_null, recv_count_null);
    }
    int n_bytes = (n_rows + 7) >> 3;
    std::vector<uint8_t> send_null_bitmask(n_bytes, 0);
    for (int i_row = 0; i_row < n_rows; i_row++)
        SetBitTo(send_null_bitmask.data(), i_row, !arr->IsNull(i_row));

    int n_null_bytes =
        std::accumulate(recv_count_null.begin(), recv_count_null.end(), 0);
    std::vector<uint8_t> tmp_null_bytes(n_null_bytes, 0);
    MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
    MPI_Gengatherv(send_null_bitmask.data(), n_bytes, mpi_typ,
                   tmp_null_bytes.data(), recv_count_null.data(),
                   recv_disp_null.data(), mpi_typ, mpi_root, MPI_COMM_WORLD,
                   all_gather);
    if (myrank == mpi_root || all_gather) {
        size_t siz_out = (n_rows_tot + 7) >> 3;
        arrow::Result<std::unique_ptr<arrow::Buffer>> maybe_buffer =
            arrow::AllocateBuffer(siz_out);
        if (!maybe_buffer.ok()) {
            Bodo_PyErr_SetString(PyExc_RuntimeError, "allocation error");
            return nullptr;
        }
        std::shared_ptr<arrow::Buffer> buffer = *std::move(maybe_buffer);
        uint8_t* data_out_ptr = buffer->mutable_data();
        copy_gathered_null_bytes(data_out_ptr, tmp_null_bytes, recv_count_null,
                                 rows_count);
        return buffer;
    }
    return nullptr;
}

std::shared_ptr<arrow::Buffer> gather_arrow_string_buffer(
    std::shared_ptr<arrow::StringArray> const& arr, bool all_gather) {
    int n_pes, myrank, mpi_root = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    std::vector<int> char_count, char_disps;
    if (myrank == mpi_root || all_gather) {
        char_count.resize(n_pes);
        char_disps.resize(n_pes);
    }
    int n_string = arr->length();
    int n_char = arr->value_offset(n_string);
    MPI_Gengather(&n_char, 1, MPI_INT, char_count.data(), 1, MPI_INT, mpi_root,
                  MPI_COMM_WORLD, all_gather);
    int n_char_tot = std::accumulate(char_count.begin(), char_count.end(), 0);
    if (myrank == mpi_root || all_gather) calc_disp(char_disps, char_count);
    std::shared_ptr<arrow::Buffer> buffer = nullptr;
    uint8_t* data_out_ptr = nullptr;
    if (myrank == mpi_root || all_gather) {
        size_t siz_out = n_char_tot;
        arrow::Result<std::unique_ptr<arrow::Buffer>> maybe_buffer =
            arrow::AllocateBuffer(siz_out);
        if (!maybe_buffer.ok()) {
            Bodo_PyErr_SetString(PyExc_RuntimeError, "allocation error");
            return nullptr;
        }
        buffer = *std::move(maybe_buffer);
        data_out_ptr = buffer->mutable_data();
    }
    MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
    std::shared_ptr<arrow::Buffer> data_buff = arr->value_data();
    uint8_t* data_in_ptr = (uint8_t*)data_buff->data();
    MPI_Gengatherv(data_in_ptr, n_char, mpi_typ, data_out_ptr,
                   char_count.data(), char_disps.data(), mpi_typ, mpi_root,
                   MPI_COMM_WORLD, all_gather);
    return buffer;
}

std::shared_ptr<arrow::Buffer> gather_arrow_primitive_buffer(
    int64_t n_rows, std::shared_ptr<arrow::PrimitiveArray> const& arr,
    bool all_gather) {
    int n_pes, myrank, mpi_root = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    arrow::Type::type typ = arr->type()->id();
    Bodo_CTypes::CTypeEnum bodo_typ = arrow_to_bodo_type(typ);
    int64_t siz_typ = numpy_item_size[bodo_typ];
    // Determination of sizes
    std::vector<int> char_count, char_disps;
    if (myrank == mpi_root || all_gather) {
        char_count.resize(n_pes);
        char_disps.resize(n_pes);
    }
    int n_char = n_rows * siz_typ;
    MPI_Gengather(&n_char, 1, MPI_INT, char_count.data(), 1, MPI_INT, mpi_root,
                  MPI_COMM_WORLD, all_gather);
    int n_char_tot = std::accumulate(char_count.begin(), char_count.end(), 0);
    if (myrank == mpi_root || all_gather) calc_disp(char_disps, char_count);
    std::shared_ptr<arrow::Buffer> buffer = nullptr;
    uint8_t* data_out_ptr = nullptr;
    if (myrank == mpi_root || all_gather) {
        size_t siz_out = n_char_tot;
        arrow::Result<std::unique_ptr<arrow::Buffer>> maybe_buffer =
            arrow::AllocateBuffer(siz_out);
        if (!maybe_buffer.ok()) {
            Bodo_PyErr_SetString(PyExc_RuntimeError, "allocation error");
            return nullptr;
        }
        buffer = *std::move(maybe_buffer);
        data_out_ptr = buffer->mutable_data();
    }
    uint8_t* send_data = (uint8_t*)arr->values()->data();
    MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
    MPI_Gengatherv(send_data, n_char, mpi_typ, data_out_ptr, char_count.data(),
                   char_disps.data(), mpi_typ, mpi_root, MPI_COMM_WORLD,
                   all_gather);
    return buffer;
}

std::shared_ptr<arrow::Array> gather_arrow_array(
    std::shared_ptr<arrow::Array> const& arr, bool all_gather) {
    int n_pes, myrank, mpi_root = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    int n_rows_tot = 0, n_rows = arr->length();
    if (all_gather)
        MPI_Allreduce(&n_rows, &n_rows_tot, 1, MPI_INT, MPI_SUM,
                      MPI_COMM_WORLD);
    else
        MPI_Reduce(&n_rows, &n_rows_tot, 1, MPI_INT, MPI_SUM, mpi_root,
                   MPI_COMM_WORLD);
    if (arr->type_id() == arrow::Type::LIST) {
        std::shared_ptr<arrow::ListArray> list_arr =
            std::dynamic_pointer_cast<arrow::ListArray>(arr);
        std::shared_ptr<arrow::Buffer> list_offsets =
            gather_arrow_offset_buffer(list_arr, all_gather);
        std::shared_ptr<arrow::Buffer> null_bitmap_out =
            gather_arrow_bitmap_buffer(list_arr, all_gather);
        std::shared_ptr<arrow::Array> child_array =
            gather_arrow_array(list_arr->values(), all_gather);
        if (myrank == mpi_root || all_gather) {
            return std::make_shared<arrow::ListArray>(
                list_arr->type(), n_rows_tot, list_offsets, child_array,
                null_bitmap_out);
        }
        return nullptr;
    } else if (arr->type_id() == arrow::Type::STRUCT) {
        std::shared_ptr<arrow::StructArray> struct_arr =
            std::dynamic_pointer_cast<arrow::StructArray>(arr);
        auto struct_type =
            std::dynamic_pointer_cast<arrow::StructType>(struct_arr->type());
        std::vector<std::shared_ptr<arrow::Array>> children;
        for (int i_field = 0; i_field < struct_type->num_fields(); i_field++)
            children.push_back(
                gather_arrow_array(struct_arr->field(i_field), all_gather));
        std::shared_ptr<arrow::Buffer> null_bitmap_out =
            gather_arrow_bitmap_buffer(struct_arr, all_gather);
        if (myrank == mpi_root || all_gather) {
            return std::make_shared<arrow::StructArray>(
                struct_arr->type(), n_rows_tot, children, null_bitmap_out);
        }
        return nullptr;
    } else if (arr->type_id() == arrow::Type::STRING) {
        std::shared_ptr<arrow::StringArray> string_arr =
            std::dynamic_pointer_cast<arrow::StringArray>(arr);
        std::shared_ptr<arrow::Buffer> list_offsets =
            gather_arrow_offset_buffer(string_arr, all_gather);
        std::shared_ptr<arrow::Buffer> null_bitmap_out =
            gather_arrow_bitmap_buffer(string_arr, all_gather);
        std::shared_ptr<arrow::Buffer> data =
            gather_arrow_string_buffer(string_arr, all_gather);
        if (myrank == mpi_root || all_gather) {
            return std::make_shared<arrow::StringArray>(
                n_rows_tot, list_offsets, data, null_bitmap_out);
        }
        return nullptr;
    } else {
        std::shared_ptr<arrow::PrimitiveArray> primitive_arr =
            std::dynamic_pointer_cast<arrow::PrimitiveArray>(arr);
        std::shared_ptr<arrow::Buffer> null_bitmap_out =
            gather_arrow_bitmap_buffer(primitive_arr, all_gather);
        std::shared_ptr<arrow::Buffer> data =
            gather_arrow_primitive_buffer(n_rows, primitive_arr, all_gather);
        if (myrank == mpi_root || all_gather) {
            return std::make_shared<arrow::PrimitiveArray>(
                primitive_arr->type(), n_rows_tot, data, null_bitmap_out);
        }
        return nullptr;
    }
}

table_info* gather_table(table_info* in_table, int64_t n_cols_i, bool all_gather) {
#ifdef DEBUG_GATHER
    std::cout << "INPUT of gather_table. in_table=\n";
    DEBUG_PrintSetOfColumn(std::cout, in_table->columns);
    DEBUG_PrintRefct(std::cout, in_table->columns);
#endif
    int n_pes, myrank;
    int mpi_root = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    std::vector<array_info*> out_arrs;
    size_t n_cols;
    if (n_cols_i == -1)
        n_cols = in_table->ncols();
    else
        n_cols = n_cols_i;
    for (size_t i_col = 0; i_col < n_cols; i_col++) {
        int64_t arr_gath_s[3];
        array_info* in_arr = in_table->columns[i_col];
        int64_t n_rows = in_arr->length;
        int64_t n_sub_elems = in_arr->n_sub_elems;
        int64_t n_sub_sub_elems = in_arr->n_sub_sub_elems;
        arr_gath_s[0] = n_rows;
        arr_gath_s[1] = n_sub_elems;
        arr_gath_s[2] = n_sub_sub_elems;
        std::vector<int64_t> arr_gath_r(3 * n_pes, 0);
        MPI_Gengather(arr_gath_s, 3, MPI_LONG_LONG_INT, arr_gath_r.data(), 3,
                      MPI_LONG_LONG_INT, mpi_root, MPI_COMM_WORLD, all_gather);
        Bodo_CTypes::CTypeEnum dtype = in_arr->dtype;
        bodo_array_type::arr_type_enum arr_type = in_arr->arr_type;
        int64_t num_categories = in_arr->num_categories;
        //
        std::vector<int> rows_disps(n_pes), rows_counts(n_pes);
        int rows_pos = 0;
        for (int i_p = 0; i_p < n_pes; i_p++) {
            int siz = arr_gath_r[3 * i_p];
            rows_counts[i_p] = siz;
            rows_disps[i_p] = rows_pos;
            rows_pos += siz;
        }
        //
        array_info* out_arr = NULL;
        if (arr_type == bodo_array_type::ARROW) {
            std::shared_ptr<arrow::Array> array = in_arr->array;
            std::shared_ptr<arrow::Array> out_array =
                gather_arrow_array(array, all_gather);
            uint64_t n_rows_tot = 0;
            NRT_MemInfo* meminfo = NULL;
            if (myrank == mpi_root || all_gather)
                n_rows_tot = out_array->length();
            out_arr = new array_info(
                bodo_array_type::ARROW, Bodo_CTypes::INT8 /*dummy*/, n_rows_tot,
                -1, -1, NULL, NULL, NULL, NULL, NULL, meminfo, NULL, out_array);
        }
        if (arr_type == bodo_array_type::NUMPY ||
            arr_type == bodo_array_type::CATEGORICAL ||
            arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
            MPI_Datatype mpi_typ = get_MPI_typ(dtype);
            // Computing the total number of rows.
            // On mpi_root, all rows, on others just 1 row for consistency.
            int64_t n_rows_tot = 0;
            for (int i_p = 0; i_p < n_pes; i_p++)
                n_rows_tot += arr_gath_r[3 * i_p];
            char* data1_ptr = NULL;
            if (myrank == mpi_root || all_gather) {
                out_arr = alloc_array(n_rows_tot, -1, -1, arr_type, dtype, 0,
                                      num_categories);
                data1_ptr = out_arr->data1;
            }
            MPI_Gengatherv(in_arr->data1, n_rows, mpi_typ, data1_ptr,
                           rows_counts.data(), rows_disps.data(), mpi_typ,
                           mpi_root, MPI_COMM_WORLD, all_gather);
        }
        if (arr_type == bodo_array_type::STRING) {
            MPI_Datatype mpi_typ32 = get_MPI_typ(Bodo_CTypes::UINT32);
            MPI_Datatype mpi_typ8 = get_MPI_typ(Bodo_CTypes::UINT8);
            // Computing indexing data in characters and rows.
            int64_t n_rows_tot = 0;
            int64_t n_chars_tot = 0;
            for (int i_p = 0; i_p < n_pes; i_p++) {
                n_rows_tot += arr_gath_r[3 * i_p];
                n_chars_tot += arr_gath_r[3 * i_p + 1];
            }
            // Doing the characters
            char* data1_ptr = NULL;
            if (myrank == mpi_root || all_gather) {
                out_arr = alloc_array(n_rows_tot, n_chars_tot, -1, arr_type,
                                      dtype, 0, num_categories);
                data1_ptr = out_arr->data1;
            }
            std::vector<int> char_disps(n_pes), char_counts(n_pes);
            int pos = 0;
            for (int i_p = 0; i_p < n_pes; i_p++) {
                int siz = arr_gath_r[3 * i_p + 1];
                char_disps[i_p] = pos;
                char_counts[i_p] = siz;
                pos += siz;
            }
            MPI_Gengatherv(in_arr->data1, n_sub_elems, mpi_typ8, data1_ptr,
                           char_counts.data(), char_disps.data(), mpi_typ8,
                           mpi_root, MPI_COMM_WORLD, all_gather);
            // Collecting the offsets data
            std::vector<uint32_t> list_count_loc(n_rows);
            uint32_t* offsets_i = (uint32_t*)in_arr->data2;
            uint32_t curr_offset = 0;
            for (int64_t pos = 0; pos < n_rows; pos++) {
                uint32_t new_offset = offsets_i[pos + 1];
                list_count_loc[pos] = new_offset - curr_offset;
                curr_offset = new_offset;
            }
            std::vector<uint32_t> list_count_tot(n_rows_tot);
            MPI_Gengatherv(list_count_loc.data(), n_rows, mpi_typ32,
                           list_count_tot.data(), rows_counts.data(),
                           rows_disps.data(), mpi_typ32, mpi_root,
                           MPI_COMM_WORLD, all_gather);
            if (myrank == mpi_root || all_gather) {
                uint32_t* offsets_o = (uint32_t*)out_arr->data2;
                offsets_o[0] = 0;
                for (int64_t pos = 0; pos < n_rows_tot; pos++)
                    offsets_o[pos + 1] = offsets_o[pos] + list_count_tot[pos];
            }
        }
        if (arr_type == bodo_array_type::LIST_STRING) {
            MPI_Datatype mpi_typ32 = get_MPI_typ(Bodo_CTypes::UINT32);
            MPI_Datatype mpi_typ8 = get_MPI_typ(Bodo_CTypes::UINT8);
            // Computing indexing data in characters and rows.
            int64_t n_rows_tot = 0;
            int64_t n_sub_elems_tot = 0;
            int64_t n_sub_sub_elems_tot = 0;
            for (int i_p = 0; i_p < n_pes; i_p++) {
                n_rows_tot += arr_gath_r[3 * i_p];
                n_sub_elems_tot += arr_gath_r[3 * i_p + 1];
                n_sub_sub_elems_tot += arr_gath_r[3 * i_p + 2];
            }
            std::vector<int> n_sub_bytes_count, n_sub_bytes_disp, string_count;
            if (myrank == mpi_root || all_gather) {
                n_sub_bytes_count.resize(n_pes);
                n_sub_bytes_disp.resize(n_pes);
                string_count.resize(n_pes);
                for (int i_p = 0; i_p < n_pes; i_p++) {
                    string_count[i_p] = arr_gath_r[3 * i_p + 1];
                    n_sub_bytes_count[i_p] = (string_count[i_p] + 7) >> 3;
                }
                calc_disp(n_sub_bytes_disp, n_sub_bytes_count);
            }
            int64_t n_sub_bytes_tot = std::accumulate(
                n_sub_bytes_count.begin(), n_sub_bytes_count.end(), 0);
            std::vector<uint8_t> V(n_sub_bytes_tot, 0);
            uint8_t* sub_null_bitmask_i = (uint8_t*)in_arr->sub_null_bitmask;
            int n_bytes = (n_sub_elems + 7) >> 3;
            MPI_Gengatherv(sub_null_bitmask_i, n_bytes, mpi_typ8, V.data(),
                           n_sub_bytes_count.data(), n_sub_bytes_disp.data(),
                           mpi_typ8, mpi_root, MPI_COMM_WORLD, all_gather);
            char* data1_ptr = NULL;
            if (myrank == mpi_root || all_gather) {
                out_arr = alloc_array(n_rows_tot, n_sub_elems_tot,
                                      n_sub_sub_elems_tot, arr_type, dtype, 0,
                                      num_categories);
                data1_ptr = out_arr->data1;
                uint8_t* sub_null_bitmask_o =
                    (uint8_t*)out_arr->sub_null_bitmask;
                copy_gathered_null_bytes(sub_null_bitmask_o, V,
                                         n_sub_bytes_count, string_count);
            }
            std::vector<int> char_disps(n_pes), char_counts(n_pes);
            int pos = 0;
            for (int i_p = 0; i_p < n_pes; i_p++) {
                int siz = arr_gath_r[3 * i_p + 2];
                char_disps[i_p] = pos;
                char_counts[i_p] = siz;
                pos += siz;
            }
            MPI_Gengatherv(in_arr->data1, n_sub_sub_elems, mpi_typ8, data1_ptr,
                           char_counts.data(), char_disps.data(), mpi_typ8,
                           mpi_root, MPI_COMM_WORLD, all_gather);
            // Sending of the data_offsets
            std::vector<int> data_offsets_disps(n_pes),
                data_offsets_counts(n_pes);
            int pos_data = 0;
            for (int i_p = 0; i_p < n_pes; i_p++) {
                int siz = arr_gath_r[3 * i_p + 1];
                data_offsets_disps[i_p] = pos_data;
                data_offsets_counts[i_p] = siz;
                pos_data += siz;
            }
            std::vector<uint32_t> len_strings_loc(n_sub_elems);
            uint32_t* data_offsets_i = (uint32_t*)in_arr->data2;
            uint32_t curr_data_offset = 0;
            for (int64_t pos = 0; pos < n_sub_elems; pos++) {
                uint32_t new_data_offset = data_offsets_i[pos + 1];
                uint32_t len_str = new_data_offset - curr_data_offset;
                len_strings_loc[pos] = len_str;
                curr_data_offset = new_data_offset;
            }
            std::vector<uint32_t> len_strings_tot(n_sub_elems_tot);
            MPI_Gengatherv(len_strings_loc.data(), n_sub_elems, mpi_typ32,
                           len_strings_tot.data(), data_offsets_counts.data(),
                           data_offsets_disps.data(), mpi_typ32, mpi_root,
                           MPI_COMM_WORLD, all_gather);
            if (myrank == mpi_root || all_gather) {
                uint32_t* data_offsets_o = (uint32_t*)out_arr->data2;
                data_offsets_o[0] = 0;
                for (int64_t pos = 0; pos < n_sub_elems_tot; pos++)
                    data_offsets_o[pos + 1] =
                        data_offsets_o[pos] + len_strings_tot[pos];
            }
            // index_offsets
            std::vector<int> index_offsets_disps(n_pes),
                index_offsets_counts(n_pes);
            int pos_index = 0;
            for (int i_p = 0; i_p < n_pes; i_p++) {
                int siz = arr_gath_r[3 * i_p];
                index_offsets_disps[i_p] = pos_index;
                index_offsets_counts[i_p] = siz;
                pos_index += siz;
            }
            std::vector<uint32_t> n_strings_loc(n_rows);
            uint32_t* index_offsets_i = (uint32_t*)in_arr->data3;
            uint32_t curr_index_offset = 0;
            for (int64_t pos = 0; pos < n_rows; pos++) {
                uint32_t new_index_offset = index_offsets_i[pos + 1];
                uint32_t n_str = new_index_offset - curr_index_offset;
                n_strings_loc[pos] = n_str;
                curr_index_offset = new_index_offset;
            }
            std::vector<uint32_t> n_strings_tot(n_rows_tot, 405);
            MPI_Gengatherv(n_strings_loc.data(), n_rows, mpi_typ32,
                           n_strings_tot.data(), index_offsets_counts.data(),
                           index_offsets_disps.data(), mpi_typ32, mpi_root,
                           MPI_COMM_WORLD, all_gather);
            if (myrank == mpi_root || all_gather) {
                uint32_t* index_offsets_o = (uint32_t*)out_arr->data3;
                index_offsets_o[0] = 0;
                for (int64_t pos = 0; pos < n_rows_tot; pos++)
                    index_offsets_o[pos + 1] =
                        index_offsets_o[pos] + n_strings_tot[pos];
            }
        }
        if (arr_type == bodo_array_type::STRING ||
            arr_type == bodo_array_type::LIST_STRING ||
            arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
            char* null_bitmask_i = in_arr->null_bitmask;
            std::vector<int> recv_count_null(n_pes), recv_disp_null(n_pes);
            for (int i_p = 0; i_p < n_pes; i_p++)
                recv_count_null[i_p] = (rows_counts[i_p] + 7) >> 3;
            calc_disp(recv_disp_null, recv_count_null);
            size_t n_null_bytes = std::accumulate(recv_count_null.begin(),
                                                  recv_count_null.end(), 0);
            std::vector<uint8_t> tmp_null_bytes(n_null_bytes, 0);
            MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
            int n_bytes = (n_rows + 7) >> 3;
            MPI_Gengatherv(null_bitmask_i, n_bytes, mpi_typ,
                           tmp_null_bytes.data(), recv_count_null.data(),
                           recv_disp_null.data(), mpi_typ, mpi_root,
                           MPI_COMM_WORLD, all_gather);
            if (myrank == mpi_root || all_gather) {
                char* null_bitmask_o = out_arr->null_bitmask;
                copy_gathered_null_bytes((uint8_t*)null_bitmask_o,
                                         tmp_null_bytes, recv_count_null,
                                         rows_counts);
            }
        }
        out_arrs.push_back(out_arr);
        // Reference stealing. See shuffle_table_kernel for discussion.
        decref_array(in_arr);
    }
#ifdef DEBUG_GATHER
    std::cout << "OUTPUT of gather_table. out_arrs=\n";
    DEBUG_PrintRefct(std::cout, out_arrs);
    DEBUG_PrintSetOfColumn(std::cout, out_arrs);
#endif
    return new table_info(out_arrs);
}

table_info* compute_node_partition_by_hash(table_info* in_table, int64_t n_keys,
                                           int64_t n_pes) {
    int64_t n_rows = in_table->nrows();
    uint32_t* hashes = hash_keys_table(in_table, n_keys, SEED_HASH_PARTITION);

    std::vector<array_info*> out_arrs;
    array_info* out_arr = alloc_array(n_rows, -1, -1, bodo_array_type::NUMPY,
                                      Bodo_CTypes::INT32, 0, 0);
    for (int64_t i_row = 0; i_row < n_rows; i_row++) {
        int32_t node_id = hashes[i_row] % n_pes;
        out_arr->at<int32_t>(i_row) = node_id;
    }
    out_arrs.push_back(out_arr);
    return new table_info(out_arrs);
}

/* Whether or not a reshuffling is needed.
   The idea is following:
   ---What slows down the running is if one or 2 processors have a much higher
   load than other because it serializes the computation.
   ---If 1 or 2 processors have little load then that is not so bad. It just
   decreases the number of effective processors used.
   ---Thus the metric to consider or not a reshuffling is
          (max nb_row) / (avg nb_row)
   ---If the value is larger than 2 then reshuffling is interesting
 */
bool need_reshuffling(table_info* in_table, double crit_fraction) {
    int64_t n_rows = in_table->nrows(), sum_n_rows, max_n_rows;
    int n_pes;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    if (n_pes == 1) return false;
    MPI_Allreduce(&n_rows, &sum_n_rows, 1, MPI_LONG_LONG_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&n_rows, &max_n_rows, 1, MPI_LONG_LONG_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    double avg_n_rows = ceil(double(sum_n_rows) / double(n_pes));
    double objective_measure = double(max_n_rows) / avg_n_rows;
    bool result = objective_measure > crit_fraction;
#ifdef DEBUG_BOUND_INFO
    std::cout << "n_rows=" << n_rows << "\n";
    std::cout << "avg_n_rows=" << avg_n_rows << " max_n_rows=" << max_n_rows
              << "\n";
    std::cout << "objective_measure=" << objective_measure
              << " result=" << result << "\n";
#endif
    return result;
}

/* Apply a renormalization shuffling
   After the operation, all nodes will have a standard size
 */
table_info* shuffle_renormalization(table_info* in_table) {
    int64_t n_rows = in_table->nrows();
    int n_pes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    std::vector<int64_t> AllSizes(n_pes);
    MPI_Allgather(&n_rows, 1, MPI_LONG_LONG_INT, AllSizes.data(), 1,
                  MPI_LONG_LONG_INT, MPI_COMM_WORLD);
    int64_t n_rows_tot = std::accumulate(AllSizes.begin(), AllSizes.end(), 0);
    int64_t shift = 0;
    for (int i_p = 0; i_p < myrank; i_p++) shift += AllSizes[i_p];
    // We use the word "hashes" as they are used all over the shuffle code.
    // However, in that case, it does not mean literally "hash". What it means
    // is the index to which the row is going to be sent.
    std::vector<uint32_t> hashes(n_rows);
    for (int64_t i_row = 0; i_row < n_rows; i_row++) {
        int64_t glob_row = shift + i_row;
        int64_t rnk = index_rank(n_rows_tot, n_pes, glob_row);
        hashes[i_row] = rnk;
    }
    //
    mpi_comm_info comm_info(in_table->columns);
    comm_info.set_counts(hashes.data());
    table_info* ret_table =
        shuffle_table_kernel(in_table, hashes.data(), comm_info);
#ifdef DEBUG_SHUFFLE
    std::cout << "RET_TABLE (shuffle_renormalization):\n";
    DEBUG_PrintRefct(std::cout, ret_table->columns);
    DEBUG_PrintSetOfColumn(std::cout, ret_table->columns);
    std::cout << "Leaving now the shuffle_renormalization\n";
#endif
    return ret_table;
}
