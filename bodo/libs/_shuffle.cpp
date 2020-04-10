// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include "_shuffle.h"
#include <numeric>
#include "_array_hash.h"
#include "_array_utils.h"
#include "_distributed.h"

#undef DEBUG_SHUFFLE

mpi_comm_info::mpi_comm_info(int _n_pes, std::vector<array_info*>& _arrays)
    : n_pes(_n_pes), arrays(_arrays) {
    n_rows = arrays[0]->length;
    has_nulls = false;
    for (array_info* arr_info : arrays) {
        if (arr_info->arr_type == bodo_array_type::STRING ||
            arr_info->arr_type == bodo_array_type::LIST_STRING ||
            arr_info->arr_type == bodo_array_type::NULLABLE_INT_BOOL)
            has_nulls = true;
    }
    n_null_bytes = 0;
    // init counts
    send_count = std::vector<int>(n_pes, 0);
    recv_count = std::vector<int>(n_pes);
    send_disp = std::vector<int>(n_pes);
    recv_disp = std::vector<int>(n_pes);
    // init counts for string arrays
    for (array_info* arr_info : arrays) {
        if (arr_info->arr_type == bodo_array_type::STRING ||
            arr_info->arr_type == bodo_array_type::LIST_STRING) {
            send_count_sub.emplace_back(std::vector<int>(n_pes, 0));
            recv_count_sub.emplace_back(std::vector<int>(n_pes));
            send_disp_sub.emplace_back(std::vector<int>(n_pes));
            recv_disp_sub.emplace_back(std::vector<int>(n_pes));
        } else {
            send_count_sub.emplace_back(std::vector<int>());
            recv_count_sub.emplace_back(std::vector<int>());
            send_disp_sub.emplace_back(std::vector<int>());
            recv_disp_sub.emplace_back(std::vector<int>());
        }
        if (arr_info->arr_type == bodo_array_type::LIST_STRING) {
            send_count_sub_sub.emplace_back(std::vector<int>(n_pes, 0));
            recv_count_sub_sub.emplace_back(std::vector<int>(n_pes));
            send_disp_sub_sub.emplace_back(std::vector<int>(n_pes));
            recv_disp_sub_sub.emplace_back(std::vector<int>(n_pes));
        } else {
            send_count_sub_sub.emplace_back(std::vector<int>());
            recv_count_sub_sub.emplace_back(std::vector<int>());
            send_disp_sub_sub.emplace_back(std::vector<int>());
            recv_disp_sub_sub.emplace_back(std::vector<int>());
        }
    }
    if (has_nulls) {
        send_count_null = std::vector<int>(n_pes);
        recv_count_null = std::vector<int>(n_pes);
        send_disp_null = std::vector<int>(n_pes);
        recv_disp_null = std::vector<int>(n_pes);
    }
}

static void calc_disp(std::vector<int>& disps, std::vector<int> const& counts) {
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
#ifdef DEBUG_SHUFFLE
    for (int i_node=0; i_node<n_pes; i_node++) {
      std::cout << "i_node=" << i_node << " send_count=" << send_count[i_node] << "\n";
    }
#endif
    // get recv count
    MPI_Alltoall(send_count.data(), 1, MPI_INT, recv_count.data(), 1, MPI_INT,
                 MPI_COMM_WORLD);
#ifdef DEBUG_SHUFFLE
    for (int i_node=0; i_node<n_pes; i_node++) {
      std::cout << "i_node=" << i_node << " recv_count=" << recv_count[i_node] << "\n";
    }
#endif

    // get displacements
    calc_disp(send_disp, send_count);
    calc_disp(recv_disp, recv_count);

    // counts for string arrays
    for (size_t i = 0; i < arrays.size(); i++) {
        array_info* arr_info = arrays[i];
        if (arr_info->arr_type == bodo_array_type::STRING) {
            // send counts
            std::vector<int>& sub_counts = send_count_sub[i];
            uint32_t* offsets = (uint32_t*)arr_info->data2;
            for (size_t i = 0; i < n_rows; i++) {
                int str_len = offsets[i + 1] - offsets[i];
                size_t node = (size_t)hashes[i] % (size_t)n_pes;
                sub_counts[node] += str_len;
            }
            // get recv count
            MPI_Alltoall(sub_counts.data(), 1, MPI_INT,
                         recv_count_sub[i].data(), 1, MPI_INT, MPI_COMM_WORLD);
            // get displacements
            calc_disp(send_disp_sub[i], sub_counts);
            calc_disp(recv_disp_sub[i], recv_count_sub[i]);
        }
        if (arr_info->arr_type == bodo_array_type::LIST_STRING) {
            // send counts
            std::vector<int>& sub_counts = send_count_sub[i];
            std::vector<int>& sub_sub_counts = send_count_sub_sub[i];
            uint32_t* index_offsets = (uint32_t*)arr_info->data3;
            uint32_t* data_offsets = (uint32_t*)arr_info->data2;
            for (size_t i = 0; i < n_rows; i++) {
                size_t node = (size_t)hashes[i] % (size_t)n_pes;
                int len_sub = index_offsets[i + 1] - index_offsets[i];
                int len_sub_sub = data_offsets[index_offsets[i + 1]] - data_offsets[index_offsets[i]];
                sub_counts[node] += len_sub;
                sub_sub_counts[node] += len_sub_sub;
            }
            // get recv count_sub
            MPI_Alltoall(sub_counts.data(), 1, MPI_INT,
                         recv_count_sub[i].data(), 1, MPI_INT, MPI_COMM_WORLD);
            // get recv count_sub_sub
            MPI_Alltoall(sub_sub_counts.data(), 1, MPI_INT,
                         recv_count_sub_sub[i].data(), 1, MPI_INT, MPI_COMM_WORLD);
#ifdef DEBUG_SHUFFLE
            for (int i_node=0; i_node<n_pes; i_node++)
              std::cout << "i_node=" << i_node << " send_count_sub[i]=" << send_count_sub[i][i_node] << "\n";
            for (int i_node=0; i_node<n_pes; i_node++)
              std::cout << "i_node=" << i_node << " recv_count_sub[i]=" << recv_count_sub[i][i_node] << "\n";
            for (int i_node=0; i_node<n_pes; i_node++)
              std::cout << "i_node=" << i_node << " send_count_sub_sub[i]=" << send_count_sub_sub[i][i_node] << "\n";
            for (int i_node=0; i_node<n_pes; i_node++)
              std::cout << "i_node=" << i_node << " recv_count_sub_sub[i]=" << recv_count_sub_sub[i][i_node] << "\n";
#endif
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
static void fill_send_array_inner(T* send_buff, T* data, uint32_t* hashes,
                                  std::vector<int> const& send_disp, int n_pes,
                                  size_t n_rows) {
    std::vector<int> tmp_offset(send_disp);
    for (size_t i = 0; i < n_rows; i++) {
        size_t node = (size_t)hashes[i] % (size_t)n_pes;
        int ind = tmp_offset[node];
        send_buff[ind] = data[i];
        tmp_offset[node]++;
    }
}

static void fill_send_array_inner_decimal(uint8_t* send_buff, uint8_t* data,
                                          uint32_t* hashes,
                                          std::vector<int> const& send_disp,
                                          int n_pes, size_t n_rows) {
    std::vector<int> tmp_offset(send_disp);
    for (size_t i = 0; i < n_rows; i++) {
        size_t node = (size_t)hashes[i] % (size_t)n_pes;
        int ind = tmp_offset[node];
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
    char* send_data_buff, uint32_t* send_length_buff,
    char* arr_data, uint32_t* arr_offsets,
    uint32_t* hashes,
    std::vector<int> const& send_disp, std::vector<int> const& send_disp_sub,
    int n_pes, size_t n_rows) {
    std::vector<int> tmp_offset(send_disp);
    std::vector<int> tmp_offset_sub(send_disp_sub);
    for (size_t i = 0; i < n_rows; i++) {
        size_t node = (size_t)hashes[i] % (size_t)n_pes;
        // write length
        int ind = tmp_offset[node];
        uint32_t str_len = arr_offsets[i + 1] - arr_offsets[i];
        send_length_buff[ind] = str_len;
        tmp_offset[node]++;
        // write data
        int c_ind = tmp_offset_sub[node];
        memcpy(&send_data_buff[c_ind], &arr_data[arr_offsets[i]], str_len);
        tmp_offset_sub[node] += str_len;
    }
}


/*
  The function for setting up the sending arrays in list_string_array case.
  Data should be ordered by processor for being sent and received by the other side.
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
    char* send_data_buff, uint32_t* send_length_data, uint32_t* send_length_index,
    char* arr_data, uint32_t* arr_data_offsets, uint32_t* arr_index_offsets,
    uint32_t* hashes,
    std::vector<int> const& send_disp, std::vector<int> const& send_disp_sub,std::vector<int> const& send_disp_sub_sub,
    int n_pes, size_t n_rows) {
    std::vector<int> tmp_offset(send_disp);
    std::vector<int> tmp_offset_sub(send_disp_sub);
    std::vector<int> tmp_offset_sub_sub(send_disp_sub_sub);
    for (size_t i = 0; i < n_rows; i++) {
        size_t node = (size_t)hashes[i] % (size_t)n_pes;
        // Compute the number of strings and the number of characters that will have to be sent.
        int ind = tmp_offset[node];
        uint32_t len_sub = arr_index_offsets[i + 1] - arr_index_offsets[i];
        uint32_t len_sub_sub = arr_data_offsets[arr_index_offsets[i+1]] - arr_data_offsets[arr_index_offsets[i]];
        // Assigning the number of strings to be sent (len_sub is the number of strings)
        send_length_index[ind] = len_sub;
        tmp_offset[node]++;
        // write the lengths of the strings that will be sent from this processor to others.
        int ind_sub = tmp_offset_sub[node];
        for (uint32_t u=0; u<len_sub; u++) {
            // size_data is the length of the strings to be sent.
            uint32_t size_data = arr_data_offsets[arr_index_offsets[i] + u + 1] - arr_data_offsets[arr_index_offsets[i]+u];
            send_length_data[ind_sub + u] = size_data;
        }
        tmp_offset_sub[node] += len_sub;
        // write the set of characters that corresponds to this entry.
        int ind_sub_sub = tmp_offset_sub_sub[node];
        memcpy(&send_data_buff[ind_sub_sub], &arr_data[arr_data_offsets[arr_index_offsets[i]]], len_sub_sub);
        tmp_offset_sub_sub[node] += len_sub_sub;
    }
}






static void fill_send_array_null_inner(uint8_t* send_null_bitmask,
                                       uint8_t* array_null_bitmask,
                                       uint32_t* hashes,
                                       std::vector<int> const& send_disp_null,
                                       int n_pes, size_t n_rows) {
    std::vector<int> tmp_offset(n_pes, 0);
    for (size_t i = 0; i < n_rows; i++) {
        size_t node = (size_t)hashes[i] % (size_t)n_pes;
        int ind = tmp_offset[node];
        // write null bit
        bool bit = GetBit(array_null_bitmask, i);
        uint8_t* out_bitmap = &send_null_bitmask[send_disp_null[node]];
        SetBitTo(out_bitmap, ind, bit);
        tmp_offset[node]++;
    }
    return;
}

static void fill_send_array(array_info* send_arr, array_info* in_arr,
                            uint32_t* hashes, std::vector<int> const& send_disp,
                            std::vector<int> const& send_disp_sub,
                            std::vector<int> const& send_disp_sub_sub,
                            std::vector<int> const& send_disp_null, int n_pes) {
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
            (char*)send_arr->data1, (uint32_t*)send_arr->data2, (uint32_t*)send_arr->data3,
            (char*)in_arr->data1, (uint32_t*)in_arr->data2, (uint32_t*)in_arr->data3,
            hashes, send_disp, send_disp_sub, send_disp_sub_sub, n_pes, n_rows);
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

static void copy_gathered_null_bytes(uint8_t* null_bitmask,
                                     std::vector<uint8_t> const& tmp_null_bytes,
                                     std::vector<int> const& recv_count_null,
                                     std::vector<int> const& recv_count) {
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

static void shuffle_array(array_info* send_arr, array_info* out_arr,
                          std::vector<int> const& send_count,
                          std::vector<int> const& recv_count,
                          std::vector<int> const& send_disp,
                          std::vector<int> const& recv_disp,
                          std::vector<int> const& send_count_sub,
                          std::vector<int> const& recv_count_sub,
                          std::vector<int> const& send_disp_sub,
                          std::vector<int> const& recv_disp_sub,
                          std::vector<int> const& send_count_sub_sub,
                          std::vector<int> const& recv_count_sub_sub,
                          std::vector<int> const& send_disp_sub_sub,
                          std::vector<int> const& recv_disp_sub_sub,
                          std::vector<int> const& send_count_null,
                          std::vector<int> const& recv_count_null,
                          std::vector<int> const& send_disp_null,
                          std::vector<int> const& recv_disp_null,
                          std::vector<uint8_t>& tmp_null_bytes) {
    if (send_arr->arr_type == bodo_array_type::LIST_STRING) {
        // index_offsets
        MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT32);
        MPI_Alltoallv(send_arr->data3, send_count.data(), send_disp.data(), mpi_typ,
                      out_arr->data3, recv_count.data(), recv_disp.data(), mpi_typ,
                      MPI_COMM_WORLD);
        convert_len_arr_to_offset((uint32_t*)out_arr->data3,
                                  (size_t)out_arr->length);
        // data_offsets
        MPI_Alltoallv(send_arr->data2, send_count_sub.data(), send_disp_sub.data(), mpi_typ,
                      out_arr->data2, recv_count_sub.data(), recv_disp_sub.data(), mpi_typ,
                      MPI_COMM_WORLD);
        uint32_t* data3_uint32 = (uint32_t*)out_arr->data3;
        size_t len = data3_uint32[out_arr->length];
        convert_len_arr_to_offset((uint32_t*)out_arr->data2, len);
        // data
        mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
        MPI_Alltoallv(send_arr->data1, send_count_sub_sub.data(), send_disp_sub_sub.data(), mpi_typ,
                      out_arr->data1, recv_count_sub_sub.data(), recv_disp_sub_sub.data(), mpi_typ,
                      MPI_COMM_WORLD);
    }
    if (send_arr->arr_type == bodo_array_type::STRING) {
        // string lengths
        MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT32);
        MPI_Alltoallv(send_arr->data2, send_count.data(), send_disp.data(), mpi_typ,
                      out_arr->data2, recv_count.data(), recv_disp.data(), mpi_typ,
                      MPI_COMM_WORLD);
        convert_len_arr_to_offset((uint32_t*)out_arr->data2,
                                  (size_t)out_arr->length);
        // string data
        mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
        MPI_Alltoallv(send_arr->data1, send_count_sub.data(), send_disp_sub.data(), mpi_typ,
                      out_arr->data1, recv_count_sub.data(), recv_disp_sub.data(), mpi_typ,
                      MPI_COMM_WORLD);
    }
    if (send_arr->arr_type == bodo_array_type::NUMPY ||
        send_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        MPI_Datatype mpi_typ = get_MPI_typ(send_arr->dtype);
        MPI_Alltoallv(send_arr->data1, send_count.data(), send_disp.data(), mpi_typ,
                      out_arr->data1, recv_count.data(), recv_disp.data(), mpi_typ,
                      MPI_COMM_WORLD);
    }
    if (send_arr->arr_type == bodo_array_type::STRING ||
        send_arr->arr_type == bodo_array_type::LIST_STRING ||
        send_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        // nulls
        MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
        MPI_Alltoallv(send_arr->null_bitmask, send_count_null.data(), send_disp_null.data(), mpi_typ,
                      tmp_null_bytes.data(), recv_count_null.data(), recv_disp_null.data(), mpi_typ,
                      MPI_COMM_WORLD);
        copy_gathered_null_bytes((uint8_t*)out_arr->null_bitmask,
                                 tmp_null_bytes, recv_count_null, recv_count);
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
  2a) First set up the send_arr by aligning the data from the input columns accordingly.
  2b) Second do the shuffling of data between all processors.
 */
table_info* shuffle_table_kernel(table_info* in_table, uint32_t* hashes,
                                 int n_pes, mpi_comm_info const& comm_info) {
    int total_recv = std::accumulate(comm_info.recv_count.begin(),
                                     comm_info.recv_count.end(), 0);
    size_t n_cols = in_table->ncols();
    std::vector<int> n_sub_recvs(n_cols);
    for (size_t i = 0; i < n_cols; i++)
        n_sub_recvs[i] =
            std::accumulate(comm_info.recv_count_sub[i].begin(),
                            comm_info.recv_count_sub[i].end(), 0);
    std::vector<int> n_sub_sub_recvs(n_cols);
    for (size_t i = 0; i < n_cols; i++)
        n_sub_sub_recvs[i] =
            std::accumulate(comm_info.recv_count_sub_sub[i].begin(),
                            comm_info.recv_count_sub_sub[i].end(), 0);

    // fill send buffer and send
    std::vector<array_info*> out_arrs;
    size_t n_rows = (size_t)in_table->nrows();
    std::vector<uint8_t> tmp_null_bytes(comm_info.n_null_bytes);
    for (size_t i = 0; i < n_cols; i++) {
        array_info* in_arr = in_table->columns[i];
        array_info* send_arr =
            alloc_array(n_rows, in_arr->n_sub_elems, in_arr->n_sub_sub_elems, in_arr->arr_type,
                        in_arr->dtype, 2 * n_pes);
        array_info* out_arr = alloc_array(total_recv, n_sub_recvs[i], n_sub_sub_recvs[i],
                                          in_arr->arr_type, in_arr->dtype, 0);
        fill_send_array(send_arr, in_arr, hashes,
                        comm_info.send_disp, comm_info.send_disp_sub[i],
                        comm_info.send_disp_sub_sub[i], comm_info.send_disp_null,
                        n_pes);

        shuffle_array(send_arr, out_arr,
                      comm_info.send_count, comm_info.recv_count,
                      comm_info.send_disp, comm_info.recv_disp,
                      comm_info.send_count_sub[i], comm_info.recv_count_sub[i],
                      comm_info.send_disp_sub[i], comm_info.recv_disp_sub[i],
                      comm_info.send_count_sub_sub[i], comm_info.recv_count_sub_sub[i],
                      comm_info.send_disp_sub_sub[i], comm_info.recv_disp_sub_sub[i],
                      comm_info.send_count_null, comm_info.recv_count_null,
                      comm_info.send_disp_null, comm_info.recv_disp_null,
                      tmp_null_bytes);

        out_arrs.push_back(out_arr);
        free_array(send_arr);
        delete send_arr;
    }

    return new table_info(out_arrs);
}

table_info* shuffle_table(table_info* in_table, int64_t n_keys) {
    // error checking
    if (in_table->ncols() <= 0 || n_keys <= 0) {
        Bodo_PyErr_SetString(PyExc_RuntimeError, "Invalid input shuffle table");
        return NULL;
    }
    int n_pes;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
#ifdef DEBUG_SHUFFLE
    std::cout << "IN_TABLE (SHUFFLE):\n";
    DEBUG_PrintSetOfColumn(std::cout, in_table->columns);
    DEBUG_PrintRefct(std::cout, in_table->columns);
    std::cout << "mpi_comm_size n_pes=" << n_pes << "\n";
#endif
    mpi_comm_info comm_info(n_pes, in_table->columns);
    // computing the hash data structure
    std::vector<array_info*> key_arrs = std::vector<array_info*>(
        in_table->columns.begin(), in_table->columns.begin() + n_keys);
    uint32_t seed = SEED_HASH_PARTITION;
    uint32_t* hashes = hash_keys(key_arrs, seed);

    comm_info.set_counts(hashes);

    table_info* table =
        shuffle_table_kernel(in_table, hashes, n_pes, comm_info);
    delete[] hashes;
#ifdef DEBUG_SHUFFLE
    std::cout << "RET_TABLE (SHUFFLE):\n";
    DEBUG_PrintSetOfColumn(std::cout, table->columns);
    DEBUG_PrintRefct(std::cout, table->columns);
#endif
    return table;
}

table_info* broadcast_table(table_info* in_table, size_t n_cols) {
    int n_pes, myrank;
    int mpi_root = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    std::vector<array_info*> out_arrs;
    for (size_t i_col = 0; i_col < n_cols; i_col++) {
        int64_t arr_bcast[5];
        if (myrank == mpi_root) {
            arr_bcast[0] = in_table->columns[i_col]->length;
            arr_bcast[1] = in_table->columns[i_col]->dtype;
            arr_bcast[2] = in_table->columns[i_col]->arr_type;
            arr_bcast[3] = in_table->columns[i_col]->n_sub_elems;
            arr_bcast[4] = in_table->columns[i_col]->n_sub_sub_elems;
        }
        MPI_Bcast(arr_bcast, 5, MPI_LONG_LONG_INT, mpi_root, MPI_COMM_WORLD);
        int64_t n_rows = arr_bcast[0];
        Bodo_CTypes::CTypeEnum dtype = Bodo_CTypes::CTypeEnum(arr_bcast[1]);
        bodo_array_type::arr_type_enum arr_type =
            bodo_array_type::arr_type_enum(arr_bcast[2]);
        int64_t n_sub_elems = arr_bcast[3];
        int64_t n_sub_sub_elems = arr_bcast[4];
        //
        array_info* out_arr;
        if (arr_type == bodo_array_type::NUMPY ||
            arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
            MPI_Datatype mpi_typ = get_MPI_typ(dtype);
            if (myrank == mpi_root)
                out_arr = copy_array(in_table->columns[i_col]);
            else
                out_arr = alloc_array(n_rows, -1, -1, arr_type, dtype, 0);
            MPI_Bcast(out_arr->data1, n_rows, mpi_typ, mpi_root,
                      MPI_COMM_WORLD);
        }
        if (arr_type == bodo_array_type::STRING) {
            MPI_Datatype mpi_typ32 = get_MPI_typ(Bodo_CTypes::UINT32);
            MPI_Datatype mpi_typ8 = get_MPI_typ(Bodo_CTypes::UINT8);
            if (myrank == mpi_root)
                out_arr = copy_array(in_table->columns[i_col]);
            else
                out_arr = alloc_array(n_rows, n_sub_elems, n_sub_sub_elems, arr_type, dtype, 0);
            MPI_Bcast(out_arr->data1, n_sub_elems, mpi_typ8, mpi_root,
                      MPI_COMM_WORLD);
            MPI_Bcast(out_arr->data2, n_rows, mpi_typ32, mpi_root,
                      MPI_COMM_WORLD);
        }
        if (arr_type == bodo_array_type::LIST_STRING) {
            MPI_Datatype mpi_typ32 = get_MPI_typ(Bodo_CTypes::UINT32);
            MPI_Datatype mpi_typ8 = get_MPI_typ(Bodo_CTypes::UINT8);
            if (myrank == mpi_root)
                out_arr = copy_array(in_table->columns[i_col]);
            else
                out_arr = alloc_array(n_rows, n_sub_elems, n_sub_sub_elems, arr_type, dtype, 0);
            MPI_Bcast(out_arr->data1, n_sub_sub_elems, mpi_typ8, mpi_root,
                      MPI_COMM_WORLD);
            MPI_Bcast(out_arr->data2, n_sub_elems, mpi_typ32, mpi_root,
                      MPI_COMM_WORLD);
            MPI_Bcast(out_arr->data3, n_rows, mpi_typ32, mpi_root,
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
    }
    return new table_info(out_arrs);
}

table_info* gather_table(table_info* in_table, size_t n_cols) {
#undef DEBUG_GATHER
    int n_pes, myrank;
    int mpi_root = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    std::vector<array_info*> out_arrs;
#ifdef DEBUG_GATHER
    std::cout << "GATHER : n_cols=" << n_cols << "\n";
#endif
    for (size_t i_col = 0; i_col < n_cols; i_col++) {
        int64_t arr_gath_s[3];
        int64_t n_rows = in_table->columns[i_col]->length;
        int64_t n_sub_elems = in_table->columns[i_col]->n_sub_elems;
        int64_t n_sub_sub_elems = in_table->columns[i_col]->n_sub_sub_elems;
#ifdef DEBUG_GATHER
        std::cout << " n_rows=" << n_rows
                  << " n_sub_elems=" << n_sub_elems
                  << " n_sub_sub_elems=" << n_sub_sub_elems
                  << "\n";
#endif
        arr_gath_s[0] = n_rows;
        arr_gath_s[1] = n_sub_elems;
        arr_gath_s[2] = n_sub_sub_elems;
        std::vector<int64_t> arr_gath_r(3 * n_pes, 0);
        MPI_Gather(arr_gath_s, 3, MPI_LONG_LONG_INT, arr_gath_r.data(), 3,
                   MPI_LONG_LONG_INT, mpi_root, MPI_COMM_WORLD);
        Bodo_CTypes::CTypeEnum dtype = in_table->columns[i_col]->dtype;
        bodo_array_type::arr_type_enum arr_type =
            in_table->columns[i_col]->arr_type;
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
        if (arr_type == bodo_array_type::NUMPY ||
            arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
            MPI_Datatype mpi_typ = get_MPI_typ(dtype);
#ifdef DEBUG_GATHER
            std::cout << "dtype=" << dtype << " mpi_typ=" << mpi_typ << "\n";
#endif
            // Computing the total number of rows.
            // On mpi_root, all rows, on others just 1 row for consistency.
            int64_t n_rows_tot = 0;
            for (int i_p = 0; i_p < n_pes; i_p++)
                n_rows_tot += arr_gath_r[3 * i_p];
#ifdef DEBUG_GATHER
            std::cout << "n_rows_tot=" << n_rows_tot << "\n";
#endif
            char* data1_ptr = NULL;
            if (myrank == mpi_root) {
                out_arr = alloc_array(n_rows_tot, -1, -1, arr_type, dtype, 0);
                data1_ptr = out_arr->data1;
            }
            MPI_Gatherv(in_table->columns[i_col]->data1, n_rows, mpi_typ,
                        data1_ptr, rows_counts.data(), rows_disps.data(),
                        mpi_typ, mpi_root, MPI_COMM_WORLD);
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
            if (myrank == mpi_root) {
                out_arr = alloc_array(n_rows_tot, n_chars_tot, -1, arr_type, dtype, 0);
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
            MPI_Gatherv(in_table->columns[i_col]->data1, n_sub_elems, mpi_typ8,
                        data1_ptr, char_counts.data(), char_disps.data(),
                        mpi_typ8, mpi_root, MPI_COMM_WORLD);
            // Collecting the offsets data
            std::vector<uint32_t> list_count_loc(n_rows);
            uint32_t* offsets_i = (uint32_t*)in_table->columns[i_col]->data2;
            uint32_t curr_offset = 0;
            for (int64_t pos = 0; pos < n_rows; pos++) {
                uint32_t new_offset = offsets_i[pos + 1];
                list_count_loc[pos] = new_offset - curr_offset;
                curr_offset = new_offset;
            }
            std::vector<uint32_t> list_count_tot(n_rows_tot);
            MPI_Gatherv(list_count_loc.data(), n_rows, mpi_typ32,
                        list_count_tot.data(), rows_counts.data(),
                        rows_disps.data(), mpi_typ32, mpi_root, MPI_COMM_WORLD);
            if (myrank == mpi_root) {
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
#ifdef DEBUG_GATHER
            std::cout << " n_rows_tot=" << n_rows_tot << " n_sub_elems_tot=" << n_sub_elems_tot << " n_sub_sub_elems_tot=" << n_sub_sub_elems_tot << "\n";
#endif
            char* data1_ptr = NULL;
            if (myrank == mpi_root) {
                out_arr = alloc_array(n_rows_tot, n_sub_elems_tot, n_sub_sub_elems_tot, arr_type, dtype, 0);
                data1_ptr = out_arr->data1;
            }
            std::vector<int> char_disps(n_pes), char_counts(n_pes);
            int pos = 0;
            for (int i_p = 0; i_p < n_pes; i_p++) {
                int siz = arr_gath_r[3 * i_p + 2];
                char_disps[i_p] = pos;
                char_counts[i_p] = siz;
                pos += siz;
            }
            MPI_Gatherv(in_table->columns[i_col]->data1, n_sub_sub_elems, mpi_typ8,
                        data1_ptr, char_counts.data(), char_disps.data(),
                        mpi_typ8, mpi_root, MPI_COMM_WORLD);
            // Sending of the data_offsets
            std::vector<int> data_offsets_disps(n_pes), data_offsets_counts(n_pes);
            int pos_data = 0;
            for (int i_p = 0; i_p < n_pes; i_p++) {
                int siz = arr_gath_r[3 * i_p + 1];
                data_offsets_disps[i_p] = pos_data;
                data_offsets_counts[i_p] = siz;
                pos_data += siz;
            }
            std::vector<uint32_t> len_strings_loc(n_sub_elems);
            uint32_t* data_offsets_i = (uint32_t*)in_table->columns[i_col]->data2;
            uint32_t curr_data_offset = 0;
            for (int64_t pos = 0; pos < n_sub_elems; pos++) {
                uint32_t new_data_offset = data_offsets_i[pos + 1];
                uint32_t len_str = new_data_offset - curr_data_offset;
                len_strings_loc[pos] = len_str;
                curr_data_offset = new_data_offset;
            }
            std::vector<uint32_t> len_strings_tot(n_sub_elems_tot);
            MPI_Gatherv(len_strings_loc.data(), n_sub_elems, mpi_typ32,
                        len_strings_tot.data(), data_offsets_counts.data(),
                        data_offsets_disps.data(), mpi_typ32, mpi_root, MPI_COMM_WORLD);
            if (myrank == mpi_root) {
                uint32_t* data_offsets_o = (uint32_t*)out_arr->data2;
                data_offsets_o[0] = 0;
                for (int64_t pos = 0; pos < n_sub_elems_tot; pos++)
                    data_offsets_o[pos + 1] = data_offsets_o[pos] + len_strings_tot[pos];
            }
            // index_offsets
            std::vector<int> index_offsets_disps(n_pes), index_offsets_counts(n_pes);
            int pos_index = 0;
            for (int i_p = 0; i_p < n_pes; i_p++) {
                int siz = arr_gath_r[3 * i_p];
                index_offsets_disps[i_p] = pos_index;
                index_offsets_counts[i_p] = siz;
                pos_index += siz;
            }
            std::vector<uint32_t> n_strings_loc(n_rows);
            uint32_t* index_offsets_i = (uint32_t*)in_table->columns[i_col]->data3;
            uint32_t curr_index_offset = 0;
            for (int64_t pos = 0; pos < n_rows; pos++) {
                uint32_t new_index_offset = index_offsets_i[pos + 1];
                uint32_t n_str = new_index_offset - curr_index_offset;
                n_strings_loc[pos] = n_str;
                curr_index_offset = new_index_offset;
            }
            std::vector<uint32_t> n_strings_tot(n_rows_tot,405);
            MPI_Gatherv(n_strings_loc.data(), n_rows, mpi_typ32,
                        n_strings_tot.data(), index_offsets_counts.data(),
                        index_offsets_disps.data(), mpi_typ32, mpi_root, MPI_COMM_WORLD);
            if (myrank == mpi_root) {
                uint32_t* index_offsets_o = (uint32_t*)out_arr->data3;
                index_offsets_o[0] = 0;
                for (int64_t pos = 0; pos < n_rows_tot; pos++)
                    index_offsets_o[pos + 1] = index_offsets_o[pos] + n_strings_tot[pos];
            }
        }
        if (arr_type == bodo_array_type::STRING ||
            arr_type == bodo_array_type::LIST_STRING ||
            arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
            char* null_bitmask_i = in_table->columns[i_col]->null_bitmask;
            std::vector<int> recv_count_null(n_pes), recv_disp_null(n_pes);
            for (int i_p = 0; i_p < n_pes; i_p++)
                recv_count_null[i_p] = (rows_counts[i_p] + 7) >> 3;
            calc_disp(recv_disp_null, recv_count_null);
            size_t n_null_bytes = std::accumulate(recv_count_null.begin(),
                                                  recv_count_null.end(), 0);
            std::vector<uint8_t> tmp_null_bytes(n_null_bytes, 0);
            MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
            int n_bytes = (n_rows + 7) >> 3;
            MPI_Gatherv(null_bitmask_i, n_bytes, mpi_typ, tmp_null_bytes.data(),
                        recv_count_null.data(), recv_disp_null.data(), mpi_typ,
                        mpi_root, MPI_COMM_WORLD);
            if (myrank == mpi_root) {
                char* null_bitmask_o = out_arr->null_bitmask;
                copy_gathered_null_bytes((uint8_t*)null_bitmask_o,
                                         tmp_null_bytes, recv_count_null,
                                         rows_counts);
            }
        }
        out_arrs.push_back(out_arr);
    }
    return new table_info(out_arrs);
}

table_info* compute_node_partition_by_hash(table_info* in_table, int64_t n_keys,
                                           int64_t n_pes) {
#undef DEBUG_COMP_HASH
    int64_t n_rows = in_table->nrows();
    std::vector<array_info*> key_arrs = std::vector<array_info*>(
        in_table->columns.begin(), in_table->columns.begin() + n_keys);
    uint32_t seed = SEED_HASH_PARTITION;
    uint32_t* hashes = hash_keys(key_arrs, seed);

    std::vector<array_info*> out_arrs;
    array_info* out_arr =
        alloc_array(n_rows, -1, -1, bodo_array_type::NUMPY, Bodo_CTypes::INT32, 0);
#ifdef DEBUG_COMP_HASH
    std::cout << "COMPUTE_HASH\n";
#endif
    for (int64_t i_row = 0; i_row < n_rows; i_row++) {
        int32_t node_id = hashes[i_row] % n_pes;
        out_arr->at<int32_t>(i_row) = node_id;
#ifdef DEBUG_COMP_HASH
        std::cout << "i_row=" << i_row << " node_id=" << node_id << "\n";
#endif
    }
    out_arrs.push_back(out_arr);
    return new table_info(out_arrs);
}
