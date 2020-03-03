// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include "_shuffle.h"
#include <numeric>
#include "_array_hash.h"
#include "_distributed.h"

mpi_comm_info::mpi_comm_info(int _n_pes, std::vector<array_info*>& _arrays)
    : n_pes(_n_pes), arrays(_arrays) {
    n_rows = arrays[0]->length;
    has_nulls = false;
    for (array_info* arr_info : arrays) {
        if (arr_info->arr_type == bodo_array_type::STRING ||
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
        if (arr_info->arr_type == bodo_array_type::STRING) {
            send_count_char.emplace_back(std::vector<int>(n_pes, 0));
            recv_count_char.emplace_back(std::vector<int>(n_pes));
            send_disp_char.emplace_back(std::vector<int>(n_pes));
            recv_disp_char.emplace_back(std::vector<int>(n_pes));
        } else {
            send_count_char.emplace_back(std::vector<int>());
            recv_count_char.emplace_back(std::vector<int>());
            send_disp_char.emplace_back(std::vector<int>());
            recv_disp_char.emplace_back(std::vector<int>());
        }
    }
    if (has_nulls) {
        send_count_null = std::vector<int>(n_pes);
        recv_count_null = std::vector<int>(n_pes);
        send_disp_null = std::vector<int>(n_pes);
        recv_disp_null = std::vector<int>(n_pes);
    }
}

static void calc_disp(std::vector<int>& disps, std::vector<int>& counts) {
    size_t n = counts.size();
    disps[0] = 0;
    for (size_t i = 1; i < n; i++) disps[i] = disps[i - 1] + counts[i - 1];
    return;
}

void mpi_comm_info::set_counts(uint32_t* hashes) {
    // get send count
    for (size_t i = 0; i < n_rows; i++) {
        size_t node = (size_t)hashes[i] % (size_t)n_pes;
        send_count[node]++;
    }

    // get recv count
    MPI_Alltoall(send_count.data(), 1, MPI_INT, recv_count.data(), 1, MPI_INT,
                 MPI_COMM_WORLD);

    // get displacements
    calc_disp(send_disp, send_count);
    calc_disp(recv_disp, recv_count);

    // counts for string arrays
    for (size_t i = 0; i < arrays.size(); i++) {
        array_info* arr_info = arrays[i];
        if (arr_info->arr_type == bodo_array_type::STRING) {
            // send counts
            std::vector<int>& char_counts = send_count_char[i];
            uint32_t* offsets = (uint32_t*)arr_info->data2;
            for (size_t i = 0; i < n_rows; i++) {
                int str_len = offsets[i + 1] - offsets[i];
                size_t node = (size_t)hashes[i] % (size_t)n_pes;
                char_counts[node] += str_len;
            }
            // get recv count
            MPI_Alltoall(char_counts.data(), 1, MPI_INT,
                         recv_count_char[i].data(), 1, MPI_INT, MPI_COMM_WORLD);
            // get displacements
            calc_disp(send_disp_char[i], char_counts);
            calc_disp(recv_disp_char[i], recv_count_char[i]);
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

static void fill_send_array_string_inner(
    char* send_data_buff, uint32_t* send_length_buff, char* arr_data,
    uint32_t* arr_offsets, uint32_t* hashes, std::vector<int> const& send_disp,
    std::vector<int> const& send_disp_char, int n_pes, size_t n_rows) {
    std::vector<int> tmp_offset(send_disp);
    std::vector<int> tmp_offset_char(send_disp_char);
    for (size_t i = 0; i < n_rows; i++) {
        size_t node = (size_t)hashes[i] % (size_t)n_pes;
        // write length
        int ind = tmp_offset[node];
        uint32_t str_len = arr_offsets[i + 1] - arr_offsets[i];
        send_length_buff[ind] = str_len;
        tmp_offset[node]++;
        // write data
        int c_ind = tmp_offset_char[node];
        memcpy(&send_data_buff[c_ind], &arr_data[arr_offsets[i]], str_len);
        tmp_offset_char[node] += str_len;
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

static void fill_send_array(array_info* send_arr, array_info* array,
                            uint32_t* hashes, std::vector<int> const& send_disp,
                            std::vector<int> const& send_disp_char,
                            std::vector<int> const& send_disp_null, int n_pes) {
    size_t n_rows = (size_t)array->length;
    // dispatch to proper function
    // TODO: general dispatcher
    if (array->arr_type == bodo_array_type::NULLABLE_INT_BOOL)
        fill_send_array_null_inner((uint8_t*)send_arr->null_bitmask,
                                   (uint8_t*)array->null_bitmask, hashes,
                                   send_disp_null, n_pes, n_rows);
    if (array->dtype == Bodo_CTypes::_BOOL)
        return fill_send_array_inner<bool>((bool*)send_arr->data1,
                                           (bool*)array->data1, hashes,
                                           send_disp, n_pes, n_rows);
    if (array->dtype == Bodo_CTypes::INT8)
        return fill_send_array_inner<int8_t>((int8_t*)send_arr->data1,
                                             (int8_t*)array->data1, hashes,
                                             send_disp, n_pes, n_rows);
    if (array->dtype == Bodo_CTypes::UINT8)
        return fill_send_array_inner<uint8_t>((uint8_t*)send_arr->data1,
                                              (uint8_t*)array->data1, hashes,
                                              send_disp, n_pes, n_rows);
    if (array->dtype == Bodo_CTypes::INT16)
        return fill_send_array_inner<int16_t>((int16_t*)send_arr->data1,
                                              (int16_t*)array->data1, hashes,
                                              send_disp, n_pes, n_rows);
    if (array->dtype == Bodo_CTypes::UINT16)
        return fill_send_array_inner<uint16_t>((uint16_t*)send_arr->data1,
                                               (uint16_t*)array->data1, hashes,
                                               send_disp, n_pes, n_rows);
    if (array->dtype == Bodo_CTypes::INT32)
        return fill_send_array_inner<int32_t>((int32_t*)send_arr->data1,
                                              (int32_t*)array->data1, hashes,
                                              send_disp, n_pes, n_rows);
    if (array->dtype == Bodo_CTypes::UINT32)
        return fill_send_array_inner<uint32_t>((uint32_t*)send_arr->data1,
                                               (uint32_t*)array->data1, hashes,
                                               send_disp, n_pes, n_rows);
    if (array->dtype == Bodo_CTypes::INT64)
        return fill_send_array_inner<int64_t>((int64_t*)send_arr->data1,
                                              (int64_t*)array->data1, hashes,
                                              send_disp, n_pes, n_rows);
    if (array->dtype == Bodo_CTypes::UINT64)
        return fill_send_array_inner<uint64_t>((uint64_t*)send_arr->data1,
                                               (uint64_t*)array->data1, hashes,
                                               send_disp, n_pes, n_rows);
    if (array->dtype == Bodo_CTypes::FLOAT32)
        return fill_send_array_inner<float>((float*)send_arr->data1,
                                            (float*)array->data1, hashes,
                                            send_disp, n_pes, n_rows);
    if (array->dtype == Bodo_CTypes::FLOAT64)
        return fill_send_array_inner<double>((double*)send_arr->data1,
                                             (double*)array->data1, hashes,
                                             send_disp, n_pes, n_rows);
    if (array->dtype == Bodo_CTypes::DECIMAL)
        return fill_send_array_inner_decimal((uint8_t*)send_arr->data1,
                                             (uint8_t*)array->data1, hashes,
                                             send_disp, n_pes, n_rows);
    if (array->arr_type == bodo_array_type::STRING)
        fill_send_array_string_inner(
            (char*)send_arr->data1, (uint32_t*)send_arr->data2,
            (char*)array->data1, (uint32_t*)array->data2, hashes, send_disp,
            send_disp_char, n_pes, n_rows);
    fill_send_array_null_inner((uint8_t*)send_arr->null_bitmask,
                               (uint8_t*)array->null_bitmask, hashes,
                               send_disp_null, n_pes, n_rows);
    return;
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
    return;
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
    return;
}

static void shuffle_array(array_info* send_arr, array_info* out_arr,
                          std::vector<int> const& send_count,
                          std::vector<int> const& recv_count,
                          std::vector<int> const& send_disp,
                          std::vector<int> const& recv_disp,
                          std::vector<int> const& send_count_char,
                          std::vector<int> const& recv_count_char,
                          std::vector<int> const& send_disp_char,
                          std::vector<int> const& recv_disp_char,
                          std::vector<int> const& send_count_null,
                          std::vector<int> const& recv_count_null,
                          std::vector<int> const& send_disp_null,
                          std::vector<int> const& recv_disp_null,
                          std::vector<uint8_t>& tmp_null_bytes) {
    // strings need data and length comm
    if (send_arr->arr_type == bodo_array_type::STRING) {
        // string lengths
        MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT32);
        MPI_Alltoallv(send_arr->data2, send_count.data(), send_disp.data(),
                      mpi_typ, out_arr->data2, recv_count.data(),
                      recv_disp.data(), mpi_typ, MPI_COMM_WORLD);
        convert_len_arr_to_offset((uint32_t*)out_arr->data2,
                                  (size_t)out_arr->length);
        // string data
        mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
        MPI_Alltoallv(send_arr->data1, send_count_char.data(),
                      send_disp_char.data(), mpi_typ, out_arr->data1,
                      recv_count_char.data(), recv_disp_char.data(), mpi_typ,
                      MPI_COMM_WORLD);

    } else {  // Numpy/nullable arrays
        MPI_Datatype mpi_typ = get_MPI_typ(send_arr->dtype);
        MPI_Alltoallv(send_arr->data1, send_count.data(), send_disp.data(),
                      mpi_typ, out_arr->data1, recv_count.data(),
                      recv_disp.data(), mpi_typ, MPI_COMM_WORLD);
    }
    if (send_arr->arr_type == bodo_array_type::STRING ||
        send_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        // nulls
        MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
        MPI_Alltoallv(send_arr->null_bitmask, send_count_null.data(),
                      send_disp_null.data(), mpi_typ, tmp_null_bytes.data(),
                      recv_count_null.data(), recv_disp_null.data(), mpi_typ,
                      MPI_COMM_WORLD);
        copy_gathered_null_bytes((uint8_t*)out_arr->null_bitmask,
                                 tmp_null_bytes, recv_count_null, recv_count);
    }
    return;
}

table_info* shuffle_table_kernel(table_info* in_table, uint32_t* hashes,
                                 int n_pes, mpi_comm_info const& comm_info) {
    int total_recv = std::accumulate(comm_info.recv_count.begin(),
                                     comm_info.recv_count.end(), 0);
    size_t n_cols = in_table->ncols();
    std::vector<int> n_char_recvs(n_cols);
    for (size_t i = 0; i < n_cols; i++)
        n_char_recvs[i] =
            std::accumulate(comm_info.recv_count_char[i].begin(),
                            comm_info.recv_count_char[i].end(), 0);

    // fill send buffer and send
    std::vector<array_info*> out_arrs;
    size_t n_rows = (size_t)in_table->nrows();
    std::vector<uint8_t> tmp_null_bytes(comm_info.n_null_bytes);
    for (size_t i = 0; i < n_cols; i++) {
        array_info* in_arr = in_table->columns[i];
        array_info* send_arr =
            alloc_array(n_rows, in_arr->n_sub_elems, in_arr->arr_type,
                        in_arr->dtype, 2 * n_pes);
        array_info* out_arr = alloc_array(total_recv, n_char_recvs[i],
                                          in_arr->arr_type, in_arr->dtype, 0);
        fill_send_array(send_arr, in_arr, hashes, comm_info.send_disp,
                        comm_info.send_disp_char[i], comm_info.send_disp_null,
                        n_pes);

        shuffle_array(send_arr, out_arr, comm_info.send_count,
                      comm_info.recv_count, comm_info.send_disp,
                      comm_info.recv_disp, comm_info.send_count_char[i],
                      comm_info.recv_count_char[i], comm_info.send_disp_char[i],
                      comm_info.recv_disp_char[i], comm_info.send_count_null,
                      comm_info.recv_count_null, comm_info.send_disp_null,
                      comm_info.recv_disp_null, tmp_null_bytes);

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
    return table;
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
        alloc_array(n_rows, -1, bodo_array_type::NUMPY, Bodo_CTypes::INT32, 0);
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
