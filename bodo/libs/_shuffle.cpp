// Copyright (C) 2023 Bodo Inc. All rights reserved.
#include "_shuffle.h"
#include <arrow/api.h>
#include <numeric>
#include <span>
#include "_array_hash.h"
#include "_array_operations.h"
#include "_array_utils.h"
#include "_bodo_to_arrow.h"
#include "_distributed.h"

#ifndef SHUFFLE_THRESHOLD
#define SHUFFLE_THRESHOLD 16000
#endif
#ifndef SHUFFLE_SYNC_ITER
#define SHUFFLE_SYNC_ITER 1
#endif

mpi_comm_info::mpi_comm_info(std::vector<std::shared_ptr<array_info>>& _arrays)
    : arrays(_arrays) {
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    n_rows = arrays[0]->length;
    has_nulls = false;
    for (std::shared_ptr<array_info> arr_info : arrays) {
        if (arr_info->arr_type == bodo_array_type::STRING ||
            arr_info->arr_type == bodo_array_type::DICT ||
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
    for (std::shared_ptr<array_info> arr_info : arrays) {
        // Note that for dictionary arrays the dictionary is handled separately
        // and so we only shuffle the indices, which is a nullable int array
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

/**
 * @brief Template used to handle the unmatchable rows in
 * mpi_comm_info::set_counts. This is used in two cases:
 * - A bloom filter is provided for additional filtering and we
 * need to decide what to do with the misses.
 * - A null bitmask is provided (only provided when nulls don't match with each
 * other, e.g. SQL joins), and we need to decide what to do with the nulls.
 *
 * @param row_dest Vector keeping track of the destination for each row.
 * This will be updated in place.
 * @param send_count Vector keeping track of the number of rows to send
 * to each rank. This will be updated in place.
 * @param pos The row position to keep on current rank.
 * @param myrank Rank of the current process.
 *
 * @tparam keep_unmatchable_rows_local Whether to keep the unmatchable rows on
 * the current rank (e.g. in case they are on an outer side of a join), or drop
 * them altogether (e.g. in case they are on an inner side of a join).
 */
template <bool keep_unmatchable_rows_local>
void handle_unmatchable_rows(bodo::vector<int>& row_dest,
                             std::vector<int64_t>& send_count, size_t pos,
                             int myrank);

/**
 * @brief Specialization of keep_unmatchable_rows_local for the case where
 * we decide to keep these rows locally. This is useful for rows on an outer
 * side of a join.
 */
template <>
void handle_unmatchable_rows<true>(bodo::vector<int>& row_dest,
                                   std::vector<int64_t>& send_count, size_t pos,
                                   int myrank) {
    row_dest[pos] = myrank;
    send_count[myrank]++;
}

/**
 * @brief Specialization of keep_unmatchable_rows_local for the case where
 * we decide to drop these rows altogether. This is useful for rows on an
 * inner side of a join.
 */
template <>
void handle_unmatchable_rows<false>(bodo::vector<int>& row_dest,
                                    std::vector<int64_t>& send_count,
                                    size_t pos, int myrank) {
    // This function does nothing.
    // row_dest[pos] will stay as default initialized value of
    // -1 and hence get dropped.
}

template <bool keep_nulls_and_filter_misses>
void mpi_comm_info::set_counts(
    const std::shared_ptr<uint32_t[]>& hashes, bool is_parallel,
    SimdBlockFilterFixed<::hashing::SimpleMixSplit>* filter,
    const uint8_t* null_bitmask) {
    tracing::Event ev("set_counts", is_parallel);
    ev.add_attribute("n_rows", n_rows);
    ev.add_attribute("g_using_filter", filter != nullptr);
    ev.add_attribute("g_keep_filter_misses", keep_nulls_and_filter_misses);
    ev.add_attribute("g_using_hash_null_bitmask", null_bitmask != nullptr);
    ev.add_attribute("g_keep_nulls_local", ((null_bitmask != nullptr) &&
                                            keep_nulls_and_filter_misses));
    ev.add_attribute("g_drop_nulls", ((null_bitmask != nullptr) &&
                                      !keep_nulls_and_filter_misses));

    // get send count
    // -1 indicates that a row is dropped (not sent anywhere)
    row_dest.resize(n_rows, -1);
    if (filter == nullptr) {
        if (null_bitmask == nullptr) {
            for (size_t i = 0; i < n_rows; i++) {
                int node = hash_to_rank(hashes[i], n_pes);
                row_dest[i] = node;
                send_count[node]++;
            }
        } else {
            for (size_t i = 0; i < n_rows; i++) {
                if (!GetBit(null_bitmask, i)) {
                    // if null: keep the row on this rank if
                    // keep_nulls_and_filter_misses=true, or drop entirely
                    // otherwise (i.e. leave row_dest[i] as -1).
                    handle_unmatchable_rows<keep_nulls_and_filter_misses>(
                        row_dest, send_count, i, this->myrank);
                } else {
                    int node = hash_to_rank(hashes[i], n_pes);
                    row_dest[i] = node;
                    send_count[node]++;
                }
            }
        }
    } else {
        if (null_bitmask == nullptr) {
            for (size_t i = 0; i < n_rows; i++) {
                const uint32_t& hash = hashes[i];
                if (!filter->Find(static_cast<uint64_t>(hash))) {
                    // if not in filter: keep the row on this rank if
                    // keep_nulls_and_filter_misses=true, or drop entirely
                    // otherwise (i.e. leave row_dest[i] as -1).
                    handle_unmatchable_rows<keep_nulls_and_filter_misses>(
                        row_dest, send_count, i, this->myrank);
                } else {
                    int node = hash_to_rank(hash, n_pes);
                    row_dest[i] = node;
                    send_count[node]++;
                }
            }
        } else {
            for (size_t i = 0; i < n_rows; i++) {
                if (!GetBit(null_bitmask, i)) {
                    // if null: keep the row on this rank if
                    // keep_nulls_and_filter_misses=true, or drop entirely
                    // otherwise (i.e. leave row_dest[i] as -1).
                    handle_unmatchable_rows<keep_nulls_and_filter_misses>(
                        row_dest, send_count, i, this->myrank);
                } else {
                    const uint32_t& hash = hashes[i];
                    if (!filter->Find(static_cast<uint64_t>(hash))) {
                        // if not in filter: keep the row on this rank if
                        // keep_nulls_and_filter_misses=true, or drop entirely
                        // otherwise (i.e. leave row_dest[i] as -1).
                        handle_unmatchable_rows<keep_nulls_and_filter_misses>(
                            row_dest, send_count, i, this->myrank);
                    } else {
                        int node = hash_to_rank(hash, n_pes);
                        row_dest[i] = node;
                        send_count[node]++;
                    }
                }
            }
        }
    }
    // Rows can get filtered if either a bloom-filter or a null filter is
    // provided
    this->filtered = (null_bitmask != nullptr) || (filter != nullptr);

    if (ev.is_tracing()) {
        int64_t n_rows_send =
            std::accumulate(send_count.begin(), send_count.end(), int64_t(0));
        ev.add_attribute("nrows_filtered",
                         static_cast<size_t>(n_rows - n_rows_send));
        ev.add_attribute("filtered", filtered);
    }
    // get recv count
    MPI_Alltoall(send_count.data(), 1, MPI_INT64_T, recv_count.data(), 1,
                 MPI_INT64_T, MPI_COMM_WORLD);

    // get displacements
    calc_disp(send_disp, send_count);
    calc_disp(recv_disp, recv_count);

    // counts for string arrays
    for (size_t i = 0; i < arrays.size(); i++) {
        std::shared_ptr<array_info> arr_info = arrays[i];
        if (arr_info->arr_type == bodo_array_type::STRING) {
            // send counts
            std::vector<int64_t>& sub_counts = send_count_sub[i];
            offset_t const* const offsets = (offset_t*)arr_info->data2();
            for (size_t i = 0; i < n_rows; i++) {
                offset_t str_len = offsets[i + 1] - offsets[i];
                if (row_dest[i] == -1)
                    continue;
                sub_counts[row_dest[i]] += str_len;
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
            offset_t const* const index_offsets = (offset_t*)arr_info->data3();
            offset_t const* const data_offsets = (offset_t*)arr_info->data2();
            for (size_t i = 0; i < n_rows; i++) {
                const int node = row_dest[i];
                if (node == -1)
                    continue;
                offset_t len_sub = index_offsets[i + 1] - index_offsets[i];
                offset_t len_sub_sub = data_offsets[index_offsets[i + 1]] -
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
        n_null_bytes = std::accumulate(recv_count_null.begin(),
                                       recv_count_null.end(), size_t(0));
    }
    return;
}

/**
 * @param is_parallel: Used to indicate whether tracing should be parallel
 * or not
 */
template <class T>
static void fill_send_array_inner(T* send_buff, const T* data,
                                  std::vector<int64_t> const& send_disp,
                                  const size_t n_rows,
                                  const std::span<const int> row_dest,
                                  bool filter, bool is_parallel) {
    tracing::Event ev("fill_send_array_inner", is_parallel);
    std::vector<int64_t> tmp_offset(send_disp);
    if (!filter) {
        for (size_t i = 0; i < n_rows; i++) {
            int64_t& ind = tmp_offset[row_dest[i]];
            send_buff[ind++] = data[i];
        }
    } else {
        for (size_t i = 0; i < n_rows; i++) {
            const int& dest = row_dest[i];
            if (dest == -1) {
                continue;
            }
            int64_t& ind = tmp_offset[dest];
            send_buff[ind++] = data[i];
        }
    }
}

/**
 * @param is_parallel: Used to indicate whether tracing should be parallel
 * or not
 */
static void fill_send_array_inner_decimal(uint8_t* send_buff, uint8_t* data,
                                          std::vector<int64_t> const& send_disp,
                                          const size_t n_rows,
                                          const std::span<const int> row_dest,
                                          bool is_parallel) {
    tracing::Event ev("fill_send_array_inner_decimal", is_parallel);
    std::vector<int64_t> tmp_offset(send_disp);
    for (size_t i = 0; i < n_rows; i++) {
        if (row_dest[i] == -1) {
            continue;
        }
        int64_t& ind = tmp_offset[row_dest[i]];
        // send_buff[ind] = data[i];
        memcpy(send_buff + ind * BYTES_PER_DECIMAL,
               data + i * BYTES_PER_DECIMAL, BYTES_PER_DECIMAL);
        ind++;
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
  @param n_rows           : the number of rows.
  @param is_parallel: Used to indicate whether tracing should be parallel or
  not
 */
static void fill_send_array_string_inner(
    // XXX send_length_buff was allocated as offset_t but treating as uint32
    char* send_data_buff, uint32_t* send_length_buff, char* arr_data,
    offset_t* arr_offsets, std::vector<int64_t> const& send_disp,
    std::vector<int64_t> const& send_disp_sub, const size_t n_rows,
    const std::span<const int> row_dest, bool is_parallel) {
    tracing::Event ev("fill_send_array_string_inner", is_parallel);
    std::vector<int64_t> tmp_offset(send_disp);
    std::vector<int64_t> tmp_offset_sub(send_disp_sub);
    for (size_t i = 0; i < n_rows; i++) {
        // write length
        const int node = row_dest[i];
        if (node == -1)
            continue;
        int64_t& ind = tmp_offset[node];
        const uint32_t str_len = arr_offsets[i + 1] - arr_offsets[i];
        send_length_buff[ind++] = str_len;
        // write data
        int64_t& c_ind = tmp_offset_sub[node];
        memcpy(&send_data_buff[c_ind], &arr_data[arr_offsets[i]], str_len);
        c_ind += str_len;
    }
}

/*
  The function for setting up the sending arrays in list_string_array case.
  Data should be ordered by processor for being sent and received by the
  other side.
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
    // XXX send_length_xxx was allocated as offset_t but treating as uint32
    char* send_data_buff, uint32_t* send_length_data,
    uint32_t* send_length_index, char* arr_data, offset_t* arr_data_offsets,
    offset_t* arr_index_offsets, std::vector<int64_t> const& send_disp,
    std::vector<int64_t> const& send_disp_sub,
    std::vector<int64_t> const& send_disp_sub_sub, int n_pes, size_t n_rows,
    const std::span<const int> row_dest) {
    std::vector<int64_t> tmp_offset(send_disp);
    std::vector<int64_t> tmp_offset_sub(send_disp_sub);
    std::vector<int64_t> tmp_offset_sub_sub(send_disp_sub_sub);
    for (size_t i = 0; i < n_rows; i++) {
        // Compute the number of strings and the number of characters that
        // will have to be sent.
        const int node = row_dest[i];
        if (node == -1)
            continue;
        int64_t ind = tmp_offset[node];
        uint32_t len_sub = arr_index_offsets[i + 1] - arr_index_offsets[i];
        uint32_t len_sub_sub = arr_data_offsets[arr_index_offsets[i + 1]] -
                               arr_data_offsets[arr_index_offsets[i]];
        // Assigning the number of strings to be sent (len_sub is the number
        // of strings)
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
    uint8_t* send_null_bitmask, uint8_t* array_null_bitmask,
    std::vector<int64_t> const& send_disp_null, int n_pes, size_t n_rows,
    const std::span<const int> row_dest) {
    std::vector<int64_t> tmp_offset(n_pes, 0);
    for (size_t i = 0; i < n_rows; i++) {
        int node = row_dest[i];
        if (node == -1) {
            continue;
        }
        int64_t& ind = tmp_offset[node];
        // write null bit
        bool bit = GetBit(array_null_bitmask, i);
        uint8_t* out_bitmap = &send_null_bitmask[send_disp_null[node]];
        SetBitTo(out_bitmap, ind++, bit);
    }
    return;
}

/**
 * @param is_parallel: Used to indicate whether tracing should be parallel
 * or not
 */
static void fill_send_array(std::shared_ptr<array_info> send_arr,
                            std::shared_ptr<array_info> in_arr,
                            std::vector<int64_t> const& send_disp,
                            std::vector<int64_t> const& send_disp_sub,
                            std::vector<int64_t> const& send_disp_sub_sub,
                            std::vector<int64_t> const& send_disp_null,
                            int n_pes, const std::span<const int> row_dest,
                            bool filter, bool is_parallel) {
    tracing::Event ev("fill_send_array", is_parallel);
    const size_t n_rows = (size_t)in_arr->length;
    // dispatch to proper function
    // TODO: general dispatcher
    if (in_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        fill_send_array_null_inner((uint8_t*)send_arr->null_bitmask(),
                                   (uint8_t*)in_arr->null_bitmask(),
                                   send_disp_null, n_pes, n_rows, row_dest);
        if (in_arr->dtype == Bodo_CTypes::_BOOL) {
            // Nullable boolean uses 1 bit per boolean so we can reuse the
            // null_bitmap function
            return fill_send_array_null_inner(
                (uint8_t*)send_arr->data1(), (uint8_t*)in_arr->data1(),
                send_disp_null, n_pes, n_rows, row_dest);
        }
    }
    switch (in_arr->dtype) {
        case Bodo_CTypes::_BOOL:
            return fill_send_array_inner<bool>(
                (bool*)send_arr->data1(), (bool*)in_arr->data1(), send_disp,
                n_rows, row_dest, filter, is_parallel);
        case Bodo_CTypes::INT8:
            return fill_send_array_inner<int8_t>(
                (int8_t*)send_arr->data1(), (int8_t*)in_arr->data1(), send_disp,
                n_rows, row_dest, filter, is_parallel);
        case Bodo_CTypes::UINT8:
            return fill_send_array_inner<uint8_t>(
                (uint8_t*)send_arr->data1(), (uint8_t*)in_arr->data1(),
                send_disp, n_rows, row_dest, filter, is_parallel);
        case Bodo_CTypes::INT16:
            return fill_send_array_inner<int16_t>(
                (int16_t*)send_arr->data1(), (int16_t*)in_arr->data1(),
                send_disp, n_rows, row_dest, filter, is_parallel);
        case Bodo_CTypes::UINT16:
            return fill_send_array_inner<uint16_t>(
                (uint16_t*)send_arr->data1(), (uint16_t*)in_arr->data1(),
                send_disp, n_rows, row_dest, filter, is_parallel);
        case Bodo_CTypes::DATE:
        case Bodo_CTypes::INT32:
            return fill_send_array_inner<int32_t>(
                (int32_t*)send_arr->data1(), (int32_t*)in_arr->data1(),
                send_disp, n_rows, row_dest, filter, is_parallel);
        case Bodo_CTypes::UINT32:
            return fill_send_array_inner<uint32_t>(
                (uint32_t*)send_arr->data1(), (uint32_t*)in_arr->data1(),
                send_disp, n_rows, row_dest, filter, is_parallel);
        case Bodo_CTypes::INT64:
        case Bodo_CTypes::DATETIME:
        case Bodo_CTypes::TIME:
        case Bodo_CTypes::TIMEDELTA:
            return fill_send_array_inner<int64_t>(
                (int64_t*)send_arr->data1(), (int64_t*)in_arr->data1(),
                send_disp, n_rows, row_dest, filter, is_parallel);
        case Bodo_CTypes::UINT64:
            return fill_send_array_inner<uint64_t>(
                (uint64_t*)send_arr->data1(), (uint64_t*)in_arr->data1(),
                send_disp, n_rows, row_dest, filter, is_parallel);
        case Bodo_CTypes::FLOAT32:
            return fill_send_array_inner<float>(
                (float*)send_arr->data1(), (float*)in_arr->data1(), send_disp,
                n_rows, row_dest, filter, is_parallel);
        case Bodo_CTypes::FLOAT64:
            return fill_send_array_inner<double>(
                (double*)send_arr->data1(), (double*)in_arr->data1(), send_disp,
                n_rows, row_dest, filter, is_parallel);
        case Bodo_CTypes::DECIMAL:
            return fill_send_array_inner_decimal(
                (uint8_t*)send_arr->data1(), (uint8_t*)in_arr->data1(),
                send_disp, n_rows, row_dest, is_parallel);
        default:
            if (in_arr->arr_type == bodo_array_type::STRING) {
                fill_send_array_string_inner(
                    /// XXX casting data2 offset_t to uint32
                    (char*)send_arr->data1(), (uint32_t*)send_arr->data2(),
                    (char*)in_arr->data1(), (offset_t*)in_arr->data2(),
                    send_disp, send_disp_sub, n_rows, row_dest, is_parallel);
                fill_send_array_null_inner((uint8_t*)send_arr->null_bitmask(),
                                           (uint8_t*)in_arr->null_bitmask(),
                                           send_disp_null, n_pes, n_rows,
                                           row_dest);
                return;
            } else if (in_arr->arr_type == bodo_array_type::LIST_STRING) {
                fill_send_array_list_string_inner(
                    /// XXX casting data2 and data3 offset_t to uint32
                    (char*)send_arr->data1(), (uint32_t*)send_arr->data2(),
                    (uint32_t*)send_arr->data3(), (char*)in_arr->data1(),
                    (offset_t*)in_arr->data2(), (offset_t*)in_arr->data3(),
                    send_disp, send_disp_sub, send_disp_sub_sub, n_pes, n_rows,
                    row_dest);
                fill_send_array_null_inner((uint8_t*)send_arr->null_bitmask(),
                                           (uint8_t*)in_arr->null_bitmask(),
                                           send_disp_null, n_pes, n_rows,
                                           row_dest);
                return;
            } else {
                throw std::runtime_error(
                    "Invalid data type for fill_send_array: " +
                    GetDtype_as_string(in_arr->dtype));
            }
    }
}

/* Internal function. Convert counts to displacements
 */
#if OFFSET_BITWIDTH == 32
static void convert_len_arr_to_offset32(uint32_t* offsets,
                                        size_t const& num_strs) {
    uint32_t curr_offset = 0;
    for (size_t i = 0; i < num_strs; i++) {
        uint32_t val = offsets[i];
        offsets[i] = curr_offset;
        curr_offset += val;
    }
    offsets[num_strs] = curr_offset;
}
#endif

static void convert_len_arr_to_offset(uint32_t* lens, offset_t* offsets,
                                      size_t num_strs) {
    offset_t curr_offset = 0;
    for (size_t i = 0; i < num_strs; i++) {
        uint32_t length = lens[i];
        offsets[i] = curr_offset;
        curr_offset += length;
    }
    offsets[num_strs] = curr_offset;
}

template <class T>
static void copy_gathered_null_bytes(
    uint8_t* null_bitmask, const std::span<const uint8_t> tmp_null_bytes,
    std::vector<T> const& recv_count_null, std::vector<T> const& recv_count) {
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

/**
 * @brief Updates a dictionary array to drop any duplicates from its
 * dictionary. If we gather_global_dict then we first unify the dictionaries
 * on all ranks to become consistent. We also drop the nulls in the
 * dictionary. The null bit of the elements in the indices array that were
 * pointing to nulls in the dictionary, are set to false to indicate null
 * instead.
 *
 * @param dict_array Dictionary array to update.
 * @param gather_global_dict Should we gather the dictionaries across all
 * ranks?
 * @param sort_dictionary Should we sort the dictionary?
 */
void update_local_dictionary_remove_duplicates(
    std::shared_ptr<array_info> dict_array, bool gather_global_dict,
    bool sort_dictionary) {
    std::shared_ptr<array_info> local_dictionary = dict_array->child_arrays[0];
    std::shared_ptr<array_info> global_dictionary;
    // If we don't have the global dictionary with unique values
    // then we need to drop duplicates on the local data and then
    // gather. If we already the global dictionary but there are duplicates
    // then we can just skip the gather step. Here we assume that
    // has_global_dictionary will be synchronized across all ranks.

    // table containing a single column with the dictionary values (not
    // indices/codes)
    std::shared_ptr<table_info> in_dictionary_table =
        std::make_shared<table_info>();
    in_dictionary_table->columns.push_back(local_dictionary);
    // get distributed global dictionary

    // We always want to drop the NAs from the dictionary itself. We will
    // handle the indices pointing to the null during the re-assignment.
    std::shared_ptr<table_info> dist_dictionary_table =
        drop_duplicates_table(in_dictionary_table, gather_global_dict, 1, 0,
                              /*dropna=*/true, false);
    in_dictionary_table.reset();

    // replicate the global dictionary on all ranks
    // allgather
    std::shared_ptr<table_info> global_dictionary_table;
    if (gather_global_dict) {
        global_dictionary_table =
            gather_table(dist_dictionary_table, 1, true, gather_global_dict);
        dist_dictionary_table.reset();
    } else {
        global_dictionary_table = dist_dictionary_table;
    }
    global_dictionary = global_dictionary_table->columns[0];
    global_dictionary_table.reset();

    // Sort dictionary locally
    // XXX Should we always sort?
    if (sort_dictionary) {
        // sort_values_array_local decrefs the input array_info (but doesn't
        // delete it). It returns a new array_info.
        std::shared_ptr<array_info> new_global_dictionary =
            sort_values_array_local(global_dictionary, false, 1, 1);

        global_dictionary = new_global_dictionary;
        dict_array->has_sorted_dictionary = true;
    }

    // XXX this doesn't propagate to Python
    dict_array->child_arrays[0] = global_dictionary;
    if (gather_global_dict) {
        // If we didn't gather data, then the dictionary can remain
        // global if it was already global.
        dict_array->has_global_dictionary = true;
    }
    dict_array->has_deduped_local_dictionary = true;

    // -------------
    // calculate mapping from old (local) indices to global ones
    const uint32_t hash_seed = SEED_HASH_JOIN;
    const size_t local_dict_len = static_cast<size_t>(local_dictionary->length);
    const size_t global_dict_len =
        static_cast<size_t>(global_dictionary->length);
    std::unique_ptr<uint32_t[]> hashes_local_dict =
        std::make_unique<uint32_t[]>(local_dict_len);
    std::unique_ptr<uint32_t[]> hashes_global_dict =
        std::make_unique<uint32_t[]>(global_dict_len);
    hash_array(hashes_local_dict, local_dictionary, local_dict_len, hash_seed,
               false, /*global_dict_needed=*/false);
    hash_array(hashes_global_dict, global_dictionary, global_dict_len,
               hash_seed, false, /*global_dict_needed=*/false);

    HashDict hash_fct{global_dict_len, hashes_global_dict, hashes_local_dict};
    KeyEqualDict equal_fct{global_dict_len, global_dictionary,
                           local_dictionary /*, is_na_equal*/};
    // dict_value_to_global_index will map a dictionary value (string) to
    // its index in the global dictionary array. We don't want strings as
    // keys of the hash map because that would be inefficient in terms of
    // storage and string copies. Instead, we will use the global and local
    // dictionary indices as keys, and use these indices to refer to the
    // strings. Because the index space of global and local dictionaries
    // overlap, to have the hash map distinguish between them, indices
    // referring to the local dictionary are incremented by
    // 'global_dict_len' before accessing the map. For example: global
    // dictionary: ["ABC", "CC", "D"]. Keys that refer to these strings: [0,
    // 1, 2] local dictionary: ["CC", "D"]. Keys that refer to these
    // strings: [3, 4] Also see HashDict and KeyEqualDict to see how keys
    // are mapped to get the hashes and to compare values

    bodo::unord_map_container<size_t, dict_indices_t, HashDict, KeyEqualDict>
        dict_value_to_global_index({}, hash_fct, equal_fct);
    dict_value_to_global_index.reserve(global_dict_len);

    dict_indices_t next_index = 1;
    for (size_t i = 0; i < global_dict_len; i++) {
        dict_indices_t& index = dict_value_to_global_index[i];
        if (index == 0) {
            index = next_index++;
        }
    }

    bodo::vector<dict_indices_t> local_to_global_index(local_dict_len);
    for (size_t i = 0; i < local_dict_len; i++) {
        // if val is not in new_map, inserts it and returns next code
        dict_indices_t index = dict_value_to_global_index[i + global_dict_len];
        local_to_global_index[i] = index - 1;
    }
    dict_value_to_global_index.clear();
    dict_value_to_global_index.reserve(0);  // try to force dealloc of hashmap
    hashes_local_dict.reset();
    hashes_global_dict.reset();
    local_dictionary.reset();

    // --------------
    // remap old (local) indices to global ones

    // TODO? if there is only one reference to dict_array remaining, I can
    // modify indices in place, otherwise I have to allocate a new array if
    // I am not changing the dictionary of the input array in Python side
    bool inplace =
        (dict_array->child_arrays[1]->buffers[0]->getMeminfo()->refct == 1);
    if (!inplace) {
        std::shared_ptr<array_info> dict_indices =
            copy_array(dict_array->child_arrays[1]);
        dict_array->child_arrays[1] = dict_indices;
    }

    uint8_t* null_bitmask = (uint8_t*)dict_array->null_bitmask();
    for (size_t i = 0; i < dict_array->child_arrays[1]->length; i++) {
        if (GetBit(null_bitmask, i)) {
            dict_indices_t& index =
                dict_array->child_arrays[1]->at<dict_indices_t>(i);
            if (local_to_global_index[index] < 0) {
                // This has to be an NA since all values in local dictionary
                // (except NA) _must_ be in the global/deduplicated
                // dictionary, and therefore have an index >= 0.
                SetBitTo(null_bitmask, i, false);
                // Set index to 0 to avoid any indexing issues later on.
                index = 0;
            } else {
                index = local_to_global_index[index];
            }
        }
    }
    // Update the dict id since dictionary values changed
    int64_t dict_id = generate_dict_id(global_dictionary->length);
    dict_array->dict_id = dict_id;
}

void drop_duplicates_local_dictionary(std::shared_ptr<array_info> dict_array,
                                      bool sort_dictionary_if_modified) {
    if (dict_array->has_deduped_local_dictionary) {
        return;
    }
    update_local_dictionary_remove_duplicates(std::move(dict_array), false,
                                              sort_dictionary_if_modified);
}

array_info* drop_duplicates_local_dictionary_py_entry(
    array_info* dict_array, bool sort_dictionary_if_modified) {
    try {
        std::shared_ptr<array_info> dict_arr =
            std::shared_ptr<array_info>(dict_array);
        drop_duplicates_local_dictionary(dict_arr, sort_dictionary_if_modified);
        // return a new array_info* to Python to own
        return new array_info(*dict_arr);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

void convert_local_dictionary_to_global(std::shared_ptr<array_info> dict_array,
                                        bool is_parallel,
                                        bool sort_dictionary_if_modified) {
    if (!is_parallel || dict_array->has_global_dictionary) {
        // If data is replicated we just return and rely on other kernels
        // to avoid checking for global. This is because some C++ functions
        // use is_parallel=False to implement certain steps of the
        // is_parallel=True implementation.
        return;
    }
    update_local_dictionary_remove_duplicates(std::move(dict_array), true,
                                              sort_dictionary_if_modified);
}

void make_dictionary_global_and_unique(std::shared_ptr<array_info> dict_array,
                                       bool is_parallel,
                                       bool sort_dictionary_if_modified) {
    convert_local_dictionary_to_global(dict_array, is_parallel,
                                       sort_dictionary_if_modified);
    drop_duplicates_local_dictionary(std::move(dict_array),
                                     sort_dictionary_if_modified);
}

// shuffle_array
void shuffle_list_string_null_bitmask(std::shared_ptr<array_info> in_arr,
                                      std::shared_ptr<array_info> out_arr,
                                      mpi_comm_info const& comm_info,
                                      const std::span<const int> row_dest) {
    int64_t n_rows = in_arr->length;
    int n_pes = comm_info.n_pes;
    std::vector<int64_t> send_count(n_pes), recv_count(n_pes);
    offset_t* index_offset = (offset_t*)in_arr->data3();
    for (int64_t i_row = 0; i_row < n_rows; i_row++) {
        int node = row_dest[i_row];
        if (node == -1)
            continue;
        offset_t len = index_offset[i_row + 1] - index_offset[i_row];
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
    int64_t send_count_sum = std::accumulate(send_count_null.begin(),
                                             send_count_null.end(), int64_t(0));
    int64_t recv_count_sum = std::accumulate(recv_count_null.begin(),
                                             recv_count_null.end(), int64_t(0));
    bodo::vector<uint8_t> Vsend(send_count_sum);
    bodo::vector<uint8_t> Vrecv(recv_count_sum);
    offset_t pos_index = 0;
    uint8_t* sub_null_bitmap = (uint8_t*)in_arr->sub_null_bitmask();
    std::vector<int64_t> shift(n_pes, 0);
    for (int64_t i_row = 0; i_row < n_rows; i_row++) {
        int node = row_dest[i_row];
        if (node == -1)
            continue;
        offset_t len = index_offset[i_row + 1] - index_offset[i_row];
        for (offset_t u = 0; u < len; u++) {
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
    copy_gathered_null_bytes((uint8_t*)out_arr->sub_null_bitmask(), Vrecv,
                             recv_count_null, recv_count);
}

/**
 * @param is_parallel: Used to indicate whether tracing should be parallel
 * or not
 */
static void shuffle_array(std::shared_ptr<array_info> send_arr,
                          std::shared_ptr<array_info> out_arr,
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
                          bodo::vector<uint8_t>& tmp_null_bytes,
                          bool is_parallel) {
    tracing::Event ev("shuffle_array", is_parallel);

    switch (send_arr->arr_type) {
        case bodo_array_type::LIST_STRING: {
            // index_offsets
            MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT32);
#if OFFSET_BITWIDTH == 32
            bodo_alltoallv(send_arr->data3(), send_count, send_disp, mpi_typ,
                           out_arr->data3(), recv_count, recv_disp, mpi_typ,
                           MPI_COMM_WORLD);
            convert_len_arr_to_offset32((uint32_t*)out_arr->data3(),
                                        (size_t)out_arr->length);
            // data_offsets
            bodo_alltoallv(send_arr->data2(), send_count_sub, send_disp_sub,
                           mpi_typ, out_arr->data2(), recv_count_sub,
                           recv_disp_sub, mpi_typ, MPI_COMM_WORLD);
            offset_t* data3_offset_t = (offset_t*)out_arr->data3();
            size_t len = data3_offset_t[out_arr->length];
            convert_len_arr_to_offset32((offset_t*)out_arr->data2(), len);
#else
            bodo::vector<uint32_t> lens(out_arr->length);
            bodo_alltoallv(send_arr->data3(), send_count, send_disp, mpi_typ,
                           lens.data(), recv_count, recv_disp, mpi_typ,
                           MPI_COMM_WORLD);
            convert_len_arr_to_offset(lens.data(), (offset_t*)out_arr->data3(),
                                      (size_t)out_arr->length);
            lens.resize(out_arr->n_sub_elems());
            // data_offsets
            bodo_alltoallv(send_arr->data2(), send_count_sub, send_disp_sub,
                           mpi_typ, lens.data(), recv_count_sub, recv_disp_sub,
                           mpi_typ, MPI_COMM_WORLD);
            offset_t* data3_offset_t = (offset_t*)out_arr->data3();
            size_t len = data3_offset_t[out_arr->length];
            convert_len_arr_to_offset(lens.data(), (offset_t*)out_arr->data2(),
                                      len);
#endif
            // data
            mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
            bodo_alltoallv(send_arr->data1(), send_count_sub_sub,
                           send_disp_sub_sub, mpi_typ, out_arr->data1(),
                           recv_count_sub_sub, recv_disp_sub_sub, mpi_typ,
                           MPI_COMM_WORLD);

            // nulls
            mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
            bodo_alltoallv(send_arr->null_bitmask(), send_count_null,
                           send_disp_null, mpi_typ, tmp_null_bytes.data(),
                           recv_count_null, recv_disp_null, mpi_typ,
                           MPI_COMM_WORLD);
            copy_gathered_null_bytes((uint8_t*)out_arr->null_bitmask(),
                                     tmp_null_bytes, recv_count_null,
                                     recv_count);
            break;
        }
        case bodo_array_type::STRING: {
            // string lengths
            MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT32);
#if OFFSET_BITWIDTH == 32
            bodo_alltoallv(send_arr->data2(), send_count, send_disp, mpi_typ,
                           out_arr->data2(), recv_count, recv_disp, mpi_typ,
                           MPI_COMM_WORLD);
            convert_len_arr_to_offset32((uint32_t*)out_arr->data2(),
                                        (size_t)out_arr->length);
#else
            std::vector<uint32_t> lens(out_arr->length);
            bodo_alltoallv(send_arr->data2(), send_count, send_disp, mpi_typ,
                           lens.data(), recv_count, recv_disp, mpi_typ,
                           MPI_COMM_WORLD);
            convert_len_arr_to_offset(lens.data(), (offset_t*)out_arr->data2(),
                                      (size_t)out_arr->length);
#endif
            // string data
            mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
            bodo_alltoallv(send_arr->data1(), send_count_sub, send_disp_sub,
                           mpi_typ, out_arr->data1(), recv_count_sub,
                           recv_disp_sub, mpi_typ, MPI_COMM_WORLD);

            // Nulls
            mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
            bodo_alltoallv(send_arr->null_bitmask(), send_count_null,
                           send_disp_null, mpi_typ, tmp_null_bytes.data(),
                           recv_count_null, recv_disp_null, mpi_typ,
                           MPI_COMM_WORLD);
            copy_gathered_null_bytes((uint8_t*)out_arr->null_bitmask(),
                                     tmp_null_bytes, recv_count_null,
                                     recv_count);
            break;
        }
        case bodo_array_type::NULLABLE_INT_BOOL: {
            // data
            MPI_Datatype mpi_typ = get_MPI_typ(send_arr->dtype);
            if (send_arr->dtype == Bodo_CTypes::_BOOL) {
                // Nullable booleans use 1 bit per boolean so we have to use
                // an intermediate array and copy the same as the null bitmap.
                // We can reuse tmp_null_bytes for both.
                // Note: Boolean always uses UINT8 MPI type.
                mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
                bodo_alltoallv(send_arr->data1(), send_count_null,
                               send_disp_null, mpi_typ, tmp_null_bytes.data(),
                               recv_count_null, recv_disp_null, mpi_typ,
                               MPI_COMM_WORLD);
                copy_gathered_null_bytes((uint8_t*)out_arr->data1(),
                                         tmp_null_bytes, recv_count_null,
                                         recv_count);
            } else {
                bodo_alltoallv(send_arr->data1(), send_count, send_disp,
                               mpi_typ, out_arr->data1(), recv_count, recv_disp,
                               mpi_typ, MPI_COMM_WORLD);
            }

            // nulls
            mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
            bodo_alltoallv(send_arr->null_bitmask(), send_count_null,
                           send_disp_null, mpi_typ, tmp_null_bytes.data(),
                           recv_count_null, recv_disp_null, mpi_typ,
                           MPI_COMM_WORLD);
            copy_gathered_null_bytes((uint8_t*)out_arr->null_bitmask(),
                                     tmp_null_bytes, recv_count_null,
                                     recv_count);
            break;
        }
        case bodo_array_type::NUMPY:
        case bodo_array_type::CATEGORICAL: {
            // data
            MPI_Datatype mpi_typ = get_MPI_typ(send_arr->dtype);
            bodo_alltoallv(send_arr->data1(), send_count, send_disp, mpi_typ,
                           out_arr->data1(), recv_count, recv_disp, mpi_typ,
                           MPI_COMM_WORLD);
            break;
        }
        default:
            throw std::runtime_error(
                "Unsupported array type for shuffle_array: " +
                GetArrType_as_string(send_arr->arr_type));
    }
}

/*
  We need a separate function since we depend on the use of polymorphism
  for the computation (not all classes have null_bitmap)
 */
template <typename T>
std::shared_ptr<arrow::Buffer> shuffle_arrow_bitmap_buffer(
    std::vector<int64_t> const& send_count,
    std::vector<int64_t> const& recv_count, int const& n_pes,
    T const& input_array, const std::span<const int> row_dest) {
    size_t n_rows_in = static_cast<size_t>(input_array->length());
    size_t n_rows_send =
        std::accumulate(send_count.begin(), send_count.end(), size_t(0));
    size_t n_rows_out =
        std::accumulate(recv_count.begin(), recv_count.end(), size_t(0));
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
    bodo::vector<uint8_t> send_null_bitmask((n_rows_send + 7) >> 3, 0);
    bodo::vector<int> row_dest_send(n_rows_send);
    for (size_t i_row = 0, s_row = 0; i_row < n_rows_in; i_row++) {
        if (row_dest[i_row] == -1) {
            continue;
        }
        SetBitTo(send_null_bitmask.data(), s_row, !input_array->IsNull(i_row));
        row_dest_send[s_row++] = row_dest[i_row];
    }
    int64_t n_row_send_null = std::accumulate(
        send_count_null.begin(), send_count_null.end(), int64_t(0));
    int64_t n_row_recv_null = std::accumulate(
        recv_count_null.begin(), recv_count_null.end(), int64_t(0));
    bodo::vector<uint8_t> send_array_null_bitmask(n_row_send_null, 0);
    bodo::vector<uint8_t> recv_array_null_bitmask(n_row_recv_null, 0);
    fill_send_array_null_inner(send_array_null_bitmask.data(),
                               send_null_bitmask.data(), send_disp_null, n_pes,
                               n_rows_send, row_dest_send);
    bodo_alltoallv(send_array_null_bitmask.data(), send_count_null,
                   send_disp_null, mpi_typ_null, recv_array_null_bitmask.data(),
                   recv_count_null, recv_disp_null, mpi_typ_null,
                   MPI_COMM_WORLD);
    size_t siz_out = (n_rows_out + 7) >> 3;
    arrow::Result<std::unique_ptr<arrow::Buffer>> maybe_buffer =
        arrow::AllocateBuffer(siz_out, bodo::BufferPool::DefaultPtr());
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
    std::vector<int64_t> const& recv_count, int const& n_pes,
    T const& input_array, const std::span<const int> row_dest) {
    size_t n_rows = static_cast<size_t>(input_array->length());
    size_t n_rows_out =
        std::accumulate(recv_count.begin(), recv_count.end(), size_t(0));
    std::vector<int64_t> send_disp(n_pes);
    std::vector<int64_t> recv_disp(n_pes);
    calc_disp(send_disp, send_count);
    calc_disp(recv_disp, recv_count);
    bodo::vector<int64_t> send_len(n_rows, 0);
    std::vector<int64_t> list_shift = send_disp;
    for (size_t i_row = 0; i_row < n_rows; i_row++) {
        int node = row_dest[i_row];
        if (node == -1)
            continue;
        int64_t off1 = input_array->value_offset(i_row);
        int64_t off2 = input_array->value_offset(i_row + 1);
        offset_t e_len = off2 - off1;
        send_len[list_shift[node]] = e_len;
        list_shift[node]++;
    }
    bodo::vector<int64_t> recv_len(n_rows_out);
    MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::INT64);
    bodo_alltoallv(send_len.data(), send_count, send_disp, mpi_typ,
                   recv_len.data(), recv_count, recv_disp, mpi_typ,
                   MPI_COMM_WORLD);
    size_t siz_out = sizeof(offset_t) * (n_rows_out + 1);
    arrow::Result<std::unique_ptr<arrow::Buffer>> maybe_buffer =
        arrow::AllocateBuffer(siz_out, bodo::BufferPool::DefaultPtr());
    if (!maybe_buffer.ok()) {
        Bodo_PyErr_SetString(PyExc_RuntimeError, "allocation error");
        return nullptr;
    }
    std::shared_ptr<arrow::Buffer> buffer = *std::move(maybe_buffer);
    uint8_t* list_offsets_ptr_ui8 = buffer->mutable_data();
    offset_t* list_offsets_ptr = (offset_t*)list_offsets_ptr_ui8;
    offset_t pos = 0;
    list_offsets_ptr[0] = 0;
    for (size_t i_row = 0; i_row < n_rows_out; i_row++) {
        pos += recv_len[i_row];
        list_offsets_ptr[i_row + 1] = pos;
    }
    return buffer;
}

template <typename T>
bodo::vector<int> map_hashes_array(std::vector<int64_t> const& send_count,
                                   std::vector<int64_t> const& recv_count,
                                   const std::span<const int> row_dest,
                                   int const& n_pes, T const& input_array) {
    size_t n_rows = static_cast<size_t>(input_array->length());
    size_t n_ent = input_array->value_offset(n_rows);
    bodo::vector<int> row_dest_out(n_ent, -1);
    for (size_t i_row = 0; i_row < n_rows; i_row++) {
        int node = row_dest[i_row];
        if (node == -1)
            continue;
        int64_t off1 = input_array->value_offset(i_row);
        int64_t off2 = input_array->value_offset(i_row + 1);
        for (int64_t idx = off1; idx < off2; idx++) {
            row_dest_out[idx] = node;
        }
    }
    return row_dest_out;
}

std::shared_ptr<arrow::Buffer> shuffle_arrow_primitive_buffer(
    std::vector<int64_t> const& send_count,
    std::vector<int64_t> const& recv_count, int const& n_pes,
    std::shared_ptr<arrow::PrimitiveArray> const& input_array,
    const std::span<const int> row_dest) {
    // Typing stuff
    auto typ = input_array->type();
    Bodo_CTypes::CTypeEnum dtype = arrow_to_bodo_type(typ->id());
    uint64_t siztype = numpy_item_size[dtype];
    MPI_Datatype mpi_typ = get_MPI_typ(dtype);
    // Setting up the arrays
    size_t n_rows_send =
        std::accumulate(send_count.begin(), send_count.end(), size_t(0));
    size_t n_rows = static_cast<size_t>(input_array->length());
    size_t n_rows_out =
        std::accumulate(recv_count.begin(), recv_count.end(), size_t(0));
    std::vector<int64_t> send_disp(n_pes);
    std::vector<int64_t> recv_disp(n_pes);
    if (typ->id() == arrow::Type::type::BOOL) {
        // Booleans store 1 bit per Boolean. To make the code simpler to
        // follow we create two code paths.
        std::vector<int64_t> send_count_bytes(n_pes);
        std::vector<int64_t> recv_count_bytes(n_pes);
        for (size_t i = 0; i < size_t(n_pes); i++) {
            send_count_bytes[i] = (send_count[i] + 7) >> 3;
            recv_count_bytes[i] = (recv_count[i] + 7) >> 3;
        }
        calc_disp(send_disp, send_count_bytes);
        calc_disp(recv_disp, recv_count_bytes);
        MPI_Datatype mpi_typ_null = get_MPI_typ(Bodo_CTypes::UINT8);
        int64_t n_bytes = (n_rows_send + 7) >> 3;
        bodo::vector<uint8_t> send_arr(n_bytes, 0);
        bodo::vector<int> row_dest_send(n_rows_send);
        for (size_t i_row = 0, s_row = 0; i_row < n_rows; i_row++) {
            if (row_dest[i_row] == -1) {
                continue;
            }
            bool bit =
                ::arrow::bit_util::GetBit(input_array->values()->data(), i_row);
            SetBitTo(send_arr.data(), s_row, bit);
            row_dest_send[s_row++] = row_dest[i_row];
        }
        int64_t n_row_send_bytes = std::accumulate(
            send_count_bytes.begin(), send_count_bytes.end(), int64_t(0));
        int64_t n_row_recv_bytes = std::accumulate(
            recv_count_bytes.begin(), recv_count_bytes.end(), int64_t(0));
        bodo::vector<uint8_t> send_array_tmp(n_row_send_bytes, 0);
        bodo::vector<uint8_t> recv_array_tmp(n_row_recv_bytes, 0);
        // We reuse null bitmap functions because the boolean array is also a
        // bitmap.
        fill_send_array_null_inner(send_array_tmp.data(), send_arr.data(),
                                   send_disp, n_pes, n_rows_send,
                                   row_dest_send);
        bodo_alltoallv(send_array_tmp.data(), send_count_bytes, send_disp,
                       mpi_typ_null, recv_array_tmp.data(), recv_count_bytes,
                       recv_disp, mpi_typ_null, MPI_COMM_WORLD);
        size_t siz_out = (n_rows_out + 7) >> 3;
        arrow::Result<std::unique_ptr<arrow::Buffer>> maybe_buffer =
            arrow::AllocateBuffer(siz_out, bodo::BufferPool::DefaultPtr());
        if (!maybe_buffer.ok()) {
            Bodo_PyErr_SetString(PyExc_RuntimeError, "allocation error");
            return nullptr;
        }
        std::shared_ptr<arrow::Buffer> buffer = *std::move(maybe_buffer);
        uint8_t* buffer_data = buffer->mutable_data();
        copy_gathered_null_bytes(buffer_data, recv_array_tmp, recv_count_bytes,
                                 recv_count);
        return buffer;

    } else {
        calc_disp(send_disp, send_count);
        calc_disp(recv_disp, recv_count);
        bodo::vector<char> send_arr(n_rows_send * siztype);
        char* values = (char*)input_array->values()->data();
        std::vector<int64_t> tmp_offset(send_disp);
        for (size_t i = 0; i < n_rows; i++) {
            int node = row_dest[i];
            if (node == -1)
                continue;
            int64_t ind = tmp_offset[node];
            memcpy(send_arr.data() + ind * siztype, values + i * siztype,
                   siztype);
            tmp_offset[node]++;
        }
        // Allocating returning arrays
        size_t siz_out = siztype * n_rows_out;
        arrow::Result<std::unique_ptr<arrow::Buffer>> maybe_buffer =
            arrow::AllocateBuffer(siz_out, bodo::BufferPool::DefaultPtr());
        if (!maybe_buffer.ok()) {
            Bodo_PyErr_SetString(PyExc_RuntimeError, "allocation error");
            return nullptr;
        }
        std::shared_ptr<arrow::Buffer> buffer = *std::move(maybe_buffer);
        // Doing the exchanges
        char* data_ptr = (char*)buffer->mutable_data();
        bodo_alltoallv(send_arr.data(), send_count, send_disp, mpi_typ,
                       data_ptr, recv_count, recv_disp, mpi_typ,
                       MPI_COMM_WORLD);
        return buffer;
    }
}

std::shared_ptr<arrow::Buffer> shuffle_string_buffer(
    std::vector<int64_t> const& send_count,
    std::vector<int64_t> const& recv_count, std::span<const int> row_dest,
    int const& n_pes,
#if OFFSET_BITWIDTH == 32
    std::shared_ptr<arrow::StringArray> const& string_array) {
#else
    std::shared_ptr<arrow::LargeStringArray> const& string_array) {
#endif
    size_t n_rows = static_cast<size_t>(string_array->length());
    std::vector<int64_t> send_count_char(n_pes, 0);
    std::vector<int64_t> recv_count_char(n_pes);
    for (size_t i_row = 0; i_row < n_rows; i_row++) {
        int node = row_dest[i_row];
        if (node == -1)
            continue;
        int64_t n_char = string_array->value_length(i_row);
        send_count_char[node] += n_char;
    }
    MPI_Alltoall(send_count_char.data(), 1, MPI_INT64_T, recv_count_char.data(),
                 1, MPI_INT64_T, MPI_COMM_WORLD);
    std::vector<int64_t> send_disp_char(n_pes);
    std::vector<int64_t> recv_disp_char(n_pes);
    calc_disp(send_disp_char, send_count_char);
    calc_disp(recv_disp_char, recv_count_char);
    int64_t n_chars_send_tot = std::accumulate(
        send_count_char.begin(), send_count_char.end(), int64_t(0));
    int64_t n_chars_recv_tot = std::accumulate(
        recv_count_char.begin(), recv_count_char.end(), int64_t(0));
    char* send_char = new char[n_chars_send_tot];
    size_t siz_out = sizeof(char) * n_chars_recv_tot;
    arrow::Result<std::unique_ptr<arrow::Buffer>> maybe_buffer =
        arrow::AllocateBuffer(siz_out, bodo::BufferPool::DefaultPtr());
    if (!maybe_buffer.ok()) {
        Bodo_PyErr_SetString(PyExc_RuntimeError, "allocation error");
        return nullptr;
    }
    std::shared_ptr<arrow::Buffer> buffer = *std::move(maybe_buffer);
    char* recv_char = (char*)buffer->mutable_data();
    std::vector<int64_t> list_shift = send_disp_char;
    for (size_t i_row = 0; i_row < n_rows; i_row++) {
        int node = row_dest[i_row];
        if (node == -1)
            continue;
        std::string_view e_str = ArrowStrArrGetView(string_array, i_row);
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
    std::shared_ptr<arrow::Array> input_array, int n_pes,
    const std::span<const int> row_dest) {
    // Computing total number of rows on output
    // Note that the number of rows, counts and hashes varies according
    // to the array in the recursive structure
    size_t n_rows = (size_t)input_array->length();
    std::vector<int64_t> send_count(n_pes, 0);
    std::vector<int64_t> recv_count(n_pes);
    for (size_t i_row = 0; i_row < n_rows; i_row++) {
        int node = row_dest[i_row];
        if (node == -1) {
            continue;
        }
        send_count[node]++;
    }
    MPI_Alltoall(send_count.data(), 1, MPI_INT64_T, recv_count.data(), 1,
                 MPI_INT64_T, MPI_COMM_WORLD);
    int64_t n_rows_out =
        std::accumulate(recv_count.begin(), recv_count.end(), int64_t(0));
    //
    //
#if OFFSET_BITWIDTH == 32
    if (input_array->type_id() == arrow::Type::LIST) {
        std::shared_ptr<arrow::ListArray> list_array =
            std::dynamic_pointer_cast<arrow::ListArray>(input_array);
#else
    if (input_array->type_id() == arrow::Type::LARGE_LIST) {
        std::shared_ptr<arrow::LargeListArray> list_array =
            std::dynamic_pointer_cast<arrow::LargeListArray>(input_array);
#endif
        // Computing the offsets, hashes
        std::shared_ptr<arrow::Buffer> list_offsets =
            shuffle_arrow_offset_buffer(send_count, recv_count, n_pes,
                                        list_array, row_dest);
        bodo::vector<int> row_dest_out = map_hashes_array(
            send_count, recv_count, row_dest, n_pes, list_array);
        // Now computing the bitmap
        std::shared_ptr<arrow::Buffer> null_bitmap_out =
            shuffle_arrow_bitmap_buffer(send_count, recv_count, n_pes,
                                        list_array, row_dest);
        // Computing the child_array
        std::shared_ptr<arrow::Array> child_array =
            shuffle_arrow_array(list_array->values(), n_pes, row_dest_out);
        // Now returning the shuffled array
#if OFFSET_BITWIDTH == 32
        return std::make_shared<arrow::ListArray>(list_array->type(),
#else
        return std::make_shared<arrow::LargeListArray>(
            list_array->type(),
#endif
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
                shuffle_arrow_array(struct_array->field(i), n_pes, row_dest));
        // Now computing the bitmap
        std::shared_ptr<arrow::Buffer> null_bitmap_out =
            shuffle_arrow_bitmap_buffer(send_count, recv_count, n_pes,
                                        struct_array, row_dest);
        // Now returning the arrays
        return std::make_shared<arrow::StructArray>(
            struct_array->type(), n_rows_out, children, null_bitmap_out);
#if OFFSET_BITWIDTH == 32
    } else if (input_array->type_id() == arrow::Type::STRING) {
        auto string_array =
            std::dynamic_pointer_cast<arrow::StringArray>(input_array);
#else
    } else if (input_array->type_id() == arrow::Type::LARGE_STRING) {
        auto string_array =
            std::dynamic_pointer_cast<arrow::LargeStringArray>(input_array);
#endif
        // Now computing the offsets
        std::shared_ptr<arrow::Buffer> list_offsets =
            shuffle_arrow_offset_buffer(send_count, recv_count, n_pes,
                                        string_array, row_dest);
        // Now computing the bitmap
        std::shared_ptr<arrow::Buffer> null_bitmap_out =
            shuffle_arrow_bitmap_buffer(send_count, recv_count, n_pes,
                                        string_array, row_dest);
        // Now computing the characters
        std::shared_ptr<arrow::Buffer> data = shuffle_string_buffer(
            send_count, recv_count, row_dest, n_pes, string_array);
        // Now returning the array
#if OFFSET_BITWIDTH == 32
        return std::make_shared<arrow::StringArray>(n_rows_out, list_offsets,
#else
        return std::make_shared<arrow::LargeStringArray>(n_rows_out,
                                                         list_offsets,
#endif
                                                    data, null_bitmap_out);
    } else {
        // Converting pointers
        auto primitive_array =
            std::dynamic_pointer_cast<arrow::PrimitiveArray>(input_array);
        // Now computing the data
        std::shared_ptr<arrow::Buffer> data = shuffle_arrow_primitive_buffer(
            send_count, recv_count, n_pes, primitive_array, row_dest);
        // Now computing the bitmap
        std::shared_ptr<arrow::Buffer> null_bitmap_out =
            shuffle_arrow_bitmap_buffer(send_count, recv_count, n_pes,
                                        primitive_array, row_dest);
        return std::make_shared<arrow::PrimitiveArray>(
            primitive_array->type(), n_rows_out, data, null_bitmap_out);
    }
}

/*
  A prerequisite for the function to run correctly is that the set_counts
  method has been used and set correctly. It determines which sizes go to
  each processor. It performs a determination of the the sending and
  receiving arrays.
  ---
  1) The first step is to accumulate the sizes from each processors.
  2) Then for each row a number of operations are done:
  2a) First set up the send_arr by aligning the data from the input columns
  accordingly. 2b) Second do the shuffling of data between all processors.
 */
std::shared_ptr<table_info> shuffle_table_kernel(
    std::shared_ptr<table_info> in_table, std::shared_ptr<uint32_t[]>& hashes,
    mpi_comm_info const& comm_info, bool is_parallel) {
    tracing::Event ev("shuffle_table_kernel", is_parallel);
    if (ev.is_tracing()) {
        ev.add_attribute("table_nrows_before",
                         static_cast<size_t>(in_table->nrows()));
        ev.add_attribute("filtered", comm_info.filtered);
        size_t global_table_nbytes = table_global_memory_size(in_table);
        ev.add_attribute("g_table_nbytes", global_table_nbytes);
    }
    int n_pes = comm_info.n_pes;
    int64_t total_recv = std::accumulate(
        comm_info.recv_count.begin(), comm_info.recv_count.end(), int64_t(0));
    size_t n_cols = (size_t)in_table->ncols();
    std::vector<int64_t> n_sub_recvs(n_cols);
    for (size_t i = 0; i < n_cols; i++) {
        n_sub_recvs[i] =
            std::accumulate(comm_info.recv_count_sub[i].begin(),
                            comm_info.recv_count_sub[i].end(), int64_t(0));
    }
    std::vector<int64_t> n_sub_sub_recvs(n_cols);
    for (size_t i = 0; i < n_cols; i++) {
        n_sub_sub_recvs[i] =
            std::accumulate(comm_info.recv_count_sub_sub[i].begin(),
                            comm_info.recv_count_sub_sub[i].end(), int64_t(0));
    }

    // fill send buffer and send
    std::vector<std::shared_ptr<array_info>> out_arrs;
    bodo::vector<uint8_t> tmp_null_bytes(comm_info.n_null_bytes);
    const int64_t n_rows_send = std::accumulate(
        comm_info.send_count.begin(), comm_info.send_count.end(), int64_t(0));
    for (size_t i = 0; i < n_cols; i++) {
        std::shared_ptr<array_info> in_arr = in_table->columns[i];
        if (in_arr->arr_type == bodo_array_type::DICT) {
            if (!in_arr->has_global_dictionary)
                throw std::runtime_error(
                    "shuffle_array: input dictionary array doesn't have a "
                    "global dictionary");
            // in_arr <- indices array, to simplify code below
            in_arr = in_arr->child_arrays[1];
        }
        std::shared_ptr<array_info> out_arr;
        if (in_arr->arr_type != bodo_array_type::STRUCT &&
            in_arr->arr_type != bodo_array_type::ARRAY_ITEM) {
            const std::vector<int64_t>& send_count_sub =
                comm_info.send_count_sub[i];
            const std::vector<int64_t>& send_count_sub_sub =
                comm_info.send_count_sub_sub[i];
            int64_t n_sub_elems = -1;
            int64_t n_sub_sub_elems = -1;
            if (send_count_sub.size() > 0)
                n_sub_elems = std::accumulate(send_count_sub.begin(),
                                              send_count_sub.end(), int64_t(0));
            if (send_count_sub_sub.size() > 0)
                n_sub_sub_elems =
                    std::accumulate(send_count_sub_sub.begin(),
                                    send_count_sub_sub.end(), int64_t(0));
            std::shared_ptr<array_info> send_arr = alloc_array(
                n_rows_send, n_sub_elems, n_sub_sub_elems, in_arr->arr_type,
                in_arr->dtype, 2 * n_pes, in_arr->num_categories);
            out_arr = alloc_array(total_recv, n_sub_recvs[i],
                                  n_sub_sub_recvs[i], in_arr->arr_type,
                                  in_arr->dtype, 0, in_arr->num_categories);
            fill_send_array(send_arr, in_arr, comm_info.send_disp,
                            comm_info.send_disp_sub[i],
                            comm_info.send_disp_sub_sub[i],
                            comm_info.send_disp_null, n_pes, comm_info.row_dest,
                            comm_info.filtered, is_parallel);

            shuffle_array(
                std::move(send_arr), out_arr, comm_info.send_count,
                comm_info.recv_count, comm_info.send_disp, comm_info.recv_disp,
                comm_info.send_count_sub[i], comm_info.recv_count_sub[i],
                comm_info.send_disp_sub[i], comm_info.recv_disp_sub[i],
                comm_info.send_count_sub_sub[i],
                comm_info.recv_count_sub_sub[i], comm_info.send_disp_sub_sub[i],
                comm_info.recv_disp_sub_sub[i], comm_info.send_count_null,
                comm_info.recv_count_null, comm_info.send_disp_null,
                comm_info.recv_disp_null, tmp_null_bytes, is_parallel);
            if (in_arr->arr_type == bodo_array_type::LIST_STRING) {
                shuffle_list_string_null_bitmask(in_arr, out_arr, comm_info,
                                                 comm_info.row_dest);
            }
        } else {
            std::shared_ptr<arrow::Array> out_array = shuffle_arrow_array(
                to_arrow(in_arr), n_pes, std::span{comm_info.row_dest});
            out_arr = arrow_array_to_bodo(out_array);
        }
        // release reference of input array
        // This is a steal reference case. The idea is to release memory as
        // soon as possible. If this release is not wished (which is a rare
        // case) then the incref before operation is needed. Using an
        // optional argument (like decref_input) to the input is a false
        // good idea since it changes the semantics to something different
        // from Python.
        if (in_table->columns[i]->arr_type == bodo_array_type::DICT) {
            in_arr = in_table->columns[i];
            std::shared_ptr<array_info> out_dict_arr = create_dict_string_array(
                in_arr->child_arrays[0], out_arr, in_arr->has_global_dictionary,
                in_arr->has_deduped_local_dictionary,
                in_arr->has_sorted_dictionary, in_arr->dict_id);
            out_arr = out_dict_arr;
        }
        in_arr.reset();
        // Release reference (and memory) early if possible.
        reset_col_if_last_table_ref(in_table, i);
        out_arrs.push_back(out_arr);
    }

    ev.add_attribute("table_nrows_after",
                     static_cast<size_t>(out_arrs[0]->length));
    return std::make_shared<table_info>(out_arrs);
}

// Shuffle is basically to send data to other processes for operations like
// drop_duplicates, etc. Usually though you want to know the indices in the
// original DF (usually the first occurring ones). Reverse shuffle is
// basically transferring the shuffled data back to the original DF. Useful
// for things like cumulative operations, array_isin, etc.

void reverse_shuffle_preallocated_data_array(
    std::shared_ptr<array_info> in_arr, std::shared_ptr<array_info> out_arr,
    std::shared_ptr<uint32_t[]>& hashes, mpi_comm_info const& comm_info) {
    tracing::Event ev("reverse_shuffle_preallocated_data_array");
    if (in_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
        in_arr->dtype == Bodo_CTypes::_BOOL) {
        // Nullable boolean arrays store 1 bit per boolean so we need a
        // specialized shuffle and loop

        // Update the counts
        int n_pes = comm_info.n_pes;
        std::vector<int64_t> send_count_bytes(n_pes), recv_count_bytes(n_pes);
        for (int i = 0; i < n_pes; i++) {
            send_count_bytes[i] = (comm_info.send_count[i] + 7) >> 3;
            recv_count_bytes[i] = (comm_info.recv_count[i] + 7) >> 3;
        }
        int64_t n_send_bytes_tot = std::accumulate(
            send_count_bytes.begin(), send_count_bytes.end(), int64_t(0));
        int64_t n_recv_bytes_tot = std::accumulate(
            recv_count_bytes.begin(), recv_count_bytes.end(), int64_t(0));
        std::vector<int64_t> send_disp_bytes(n_pes), recv_disp_bytes(n_pes);
        calc_disp(send_disp_bytes, send_count_bytes);
        calc_disp(recv_disp_bytes, recv_count_bytes);

        bodo::vector<uint8_t> temp_send(n_recv_bytes_tot);
        uint8_t* data1_i = (uint8_t*)in_arr->data1();
        uint8_t* data1_o = (uint8_t*)out_arr->data1();
        int64_t pos = 0;
        for (int i = 0; i < n_pes; i++) {
            for (int64_t i_row = 0; i_row < comm_info.recv_count[i]; i_row++) {
                bool bit = GetBit(data1_i, pos);
                SetBitTo(temp_send.data(), 8 * recv_disp_bytes[i] + i_row, bit);
                pos++;
            }
        }
        bodo::vector<uint8_t> temp_recv(n_send_bytes_tot);
        MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
        bodo_alltoallv(temp_send.data(), recv_count_bytes, recv_disp_bytes,
                       mpi_typ, temp_recv.data(), send_count_bytes,
                       send_disp_bytes, mpi_typ, MPI_COMM_WORLD);
        std::vector<int64_t> tmp_offset(n_pes, 0);
        for (size_t i_row = 0; i_row < out_arr->length; i_row++) {
            size_t node = hash_to_rank(hashes[i_row], n_pes);
            uint8_t* out_bitmap = &(temp_recv.data())[send_disp_bytes[node]];
            bool bit = GetBit(out_bitmap, tmp_offset[node]);
            SetBitTo(data1_o, i_row, bit);
            tmp_offset[node]++;
        }

    } else {
        const uint64_t siztype = numpy_item_size[in_arr->dtype];
        MPI_Datatype mpi_typ = get_MPI_typ(in_arr->dtype);
        char* data1_i = in_arr->data1();
        char* data1_o = out_arr->data1();
        bodo::vector<char> tmp_recv(out_arr->length * siztype);
        bodo_alltoallv(data1_i, comm_info.recv_count, comm_info.recv_disp,
                       mpi_typ, tmp_recv.data(), comm_info.send_count,
                       comm_info.send_disp, mpi_typ, MPI_COMM_WORLD);
        std::vector<int64_t> tmp_offset(comm_info.send_disp);
        const bodo::vector<int>& row_dest = comm_info.row_dest;
        // Nullable boolean arrays store 1 bit per boolean so
        // we need a specialized setitem.
        for (size_t i = 0; i < out_arr->length; i++) {
            int64_t& ind = tmp_offset[row_dest[i]];
            memcpy(data1_o + siztype * i, tmp_recv.data() + siztype * ind++,
                   siztype);
        }
    }
}

std::shared_ptr<array_info> reverse_shuffle_data_array(
    std::shared_ptr<array_info> in_arr, std::shared_ptr<uint32_t[]>& hashes,
    mpi_comm_info const& comm_info) {
    tracing::Event ev("reverse_shuffle_data_array");
    size_t n_rows_ret = std::accumulate(comm_info.send_count.begin(),
                                        comm_info.send_count.end(), size_t(0));
    std::shared_ptr<array_info> out_arr =
        alloc_array(n_rows_ret, 0, 0, in_arr->arr_type, in_arr->dtype, 0,
                    in_arr->num_categories);
    reverse_shuffle_preallocated_data_array(in_arr, out_arr, hashes, comm_info);
    return out_arr;
}

std::shared_ptr<array_info> reverse_shuffle_string_array(
    std::shared_ptr<array_info> in_arr, std::shared_ptr<uint32_t[]>& hashes,
    mpi_comm_info const& comm_info) {
    tracing::Event ev("reverse_shuffle_string_array");
    // 1: computing the recv_count_sub and related
    offset_t* in_offset = (offset_t*)in_arr->data2();
    int n_pes = comm_info.n_pes;
    std::vector<int64_t> recv_count_sub(n_pes),
        recv_disp_sub(n_pes);  // we continue using here the recv/send
    for (int i = 0; i < n_pes; i++)
        recv_count_sub[i] =
            in_offset[comm_info.recv_disp[i] + comm_info.recv_count[i]] -
            in_offset[comm_info.recv_disp[i]];
    std::vector<int64_t> send_count_sub(n_pes), send_disp_sub(n_pes);

    MPI_Alltoall(recv_count_sub.data(), 1, MPI_INT64_T, send_count_sub.data(),
                 1, MPI_INT64_T, MPI_COMM_WORLD);

    calc_disp(send_disp_sub, send_count_sub);
    calc_disp(recv_disp_sub, recv_count_sub);
    // 2: allocating the array
    int64_t n_count_sub = send_disp_sub[n_pes - 1] + send_count_sub[n_pes - 1];
    int64_t n_rows_ret = std::accumulate(
        comm_info.send_count.begin(), comm_info.send_count.end(), int64_t(0));
    std::shared_ptr<array_info> out_arr =
        alloc_array(n_rows_ret, n_count_sub, 0, in_arr->arr_type, in_arr->dtype,
                    0, in_arr->num_categories);
    int64_t in_len = in_arr->length;
    int64_t out_len = out_arr->length;
    // 3: the offsets
    bodo::vector<uint32_t> list_len_send(in_len);
    offset_t* out_offset = (offset_t*)out_arr->data2();
    for (int64_t i = 0; i < in_len; i++)
        list_len_send[i] = in_offset[i + 1] - in_offset[i];
    MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT32);
    bodo::vector<uint32_t> list_len_recv(out_len);
    bodo_alltoallv(list_len_send.data(), comm_info.recv_count,
                   comm_info.recv_disp, mpi_typ, list_len_recv.data(),
                   comm_info.send_count, comm_info.send_disp, mpi_typ,
                   MPI_COMM_WORLD);
#if OFFSET_BITWIDTH == 32
    fill_recv_data_inner(list_len_recv.data(), out_offset, hashes,
                         comm_info.send_disp, comm_info.n_pes, out_len);
    convert_len_arr_to_offset32(out_offset, out_len);
#else
    bodo::vector<uint32_t> out_lens(out_arr->length);
    fill_recv_data_inner(list_len_recv.data(), out_lens.data(), hashes,
                         comm_info.send_disp, comm_info.n_pes, out_len);
    convert_len_arr_to_offset(out_lens.data(), out_offset, out_len);
#endif
    // 4: the characters themselves
    int64_t tot_char = std::accumulate(send_count_sub.begin(),
                                       send_count_sub.end(), int64_t(0));
    bodo::vector<uint8_t> tmp_recv(tot_char);
    mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
    bodo_alltoallv(in_arr->data1(), recv_count_sub, recv_disp_sub, mpi_typ,
                   tmp_recv.data(), send_count_sub, send_disp_sub, mpi_typ,
                   MPI_COMM_WORLD);
    std::vector<int64_t> tmp_offset_sub(send_disp_sub);
    for (int64_t i = 0; i < out_len; i++) {
        size_t node = hash_to_rank(hashes[i], n_pes);
        offset_t str_len = out_offset[i + 1] - out_offset[i];
        int64_t c_ind = tmp_offset_sub[node];
        char* out_ptr = out_arr->data1() + out_offset[i];
        char* in_ptr = (char*)tmp_recv.data() + c_ind;
        memcpy(out_ptr, in_ptr, str_len);
        tmp_offset_sub[node] += str_len;
    }
    return out_arr;
}

void reverse_shuffle_null_bitmap_array(std::shared_ptr<array_info> in_arr,
                                       std::shared_ptr<array_info> out_arr,
                                       std::shared_ptr<uint32_t[]>& hashes,
                                       mpi_comm_info const& comm_info) {
    tracing::Event ev("reverse_shuffle_null_bitmap_array");
    int n_pes = comm_info.n_pes;
    std::vector<int64_t> send_count_null(n_pes), recv_count_null(n_pes);
    for (int i = 0; i < n_pes; i++) {
        send_count_null[i] = (comm_info.send_count[i] + 7) >> 3;
        recv_count_null[i] = (comm_info.recv_count[i] + 7) >> 3;
    }
    int64_t n_send_null_tot = std::accumulate(
        send_count_null.begin(), send_count_null.end(), int64_t(0));
    int64_t n_recv_null_tot = std::accumulate(
        recv_count_null.begin(), recv_count_null.end(), int64_t(0));
    std::vector<int64_t> send_disp_null(n_pes), recv_disp_null(n_pes);
    calc_disp(send_disp_null, send_count_null);
    calc_disp(recv_disp_null, recv_count_null);
    bodo::vector<uint8_t> mask_send(n_recv_null_tot);
    uint8_t* null_bitmask_in = (uint8_t*)in_arr->null_bitmask();
    uint8_t* null_bitmask_out = (uint8_t*)out_arr->null_bitmask();
    int64_t pos = 0;
    for (int i = 0; i < n_pes; i++) {
        for (int64_t i_row = 0; i_row < comm_info.recv_count[i]; i_row++) {
            bool bit = GetBit(null_bitmask_in, pos);
            SetBitTo(mask_send.data(), 8 * recv_disp_null[i] + i_row, bit);
            pos++;
        }
    }
    bodo::vector<uint8_t> mask_recv(n_send_null_tot);
    MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
    bodo_alltoallv(mask_send.data(), recv_count_null, recv_disp_null, mpi_typ,
                   mask_recv.data(), send_count_null, send_disp_null, mpi_typ,
                   MPI_COMM_WORLD);
    std::vector<int64_t> tmp_offset(n_pes, 0);
    for (size_t i_row = 0; i_row < out_arr->length; i_row++) {
        size_t node = hash_to_rank(hashes[i_row], n_pes);
        uint8_t* out_bitmap = &(mask_recv.data())[send_disp_null[node]];
        bool bit = GetBit(out_bitmap, tmp_offset[node]);
        SetBitTo(null_bitmask_out, i_row, bit);
        tmp_offset[node]++;
    }
}

std::shared_ptr<array_info> reverse_shuffle_list_string_array(
    std::shared_ptr<array_info> in_arr, std::shared_ptr<uint32_t[]>& hashes,
    mpi_comm_info const& comm_info) {
    tracing::Event ev("reverse_shuffle_list_string_array");
    // 1: computing the recv_count_sub and related
    int n_pes = comm_info.n_pes;
    offset_t* in_str_offset = (offset_t*)in_arr->data3();
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
    offset_t* in_data_offset = (offset_t*)in_arr->data2();
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
    int64_t n_rows_ret = std::accumulate(
        comm_info.send_count.begin(), comm_info.send_count.end(), int64_t(0));
    int64_t n_count_sub = send_disp_sub[n_pes - 1] + send_count_sub[n_pes - 1];
    int64_t n_count_sub_sub =
        send_disp_sub_sub[n_pes - 1] + send_count_sub_sub[n_pes - 1];
    std::shared_ptr<array_info> out_arr =
        alloc_array(n_rows_ret, n_count_sub, n_count_sub_sub, in_arr->arr_type,
                    in_arr->dtype, 0, in_arr->num_categories);
    int64_t in_len = in_arr->length;
    int64_t out_len = out_arr->length;
    // 4: the string offsets
    bodo::vector<uint32_t> list_str_len_send(in_len);
    offset_t* out_str_offset = (offset_t*)out_arr->data3();
    for (int64_t i = 0; i < in_len; i++)
        list_str_len_send[i] = in_str_offset[i + 1] - in_str_offset[i];
    MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT32);
    bodo::vector<uint32_t> list_str_len_recv(out_len);
    bodo_alltoallv(list_str_len_send.data(), comm_info.recv_count,
                   comm_info.recv_disp, mpi_typ, list_str_len_recv.data(),
                   comm_info.send_count, comm_info.send_disp, mpi_typ,
                   MPI_COMM_WORLD);
#if OFFSET_BITWIDTH == 32
    fill_recv_data_inner(list_str_len_recv.data(), out_str_offset, hashes,
                         comm_info.send_disp, comm_info.n_pes, out_len);
    convert_len_arr_to_offset32(out_str_offset, out_len);
#else
    bodo::vector<uint32_t> out_lens(out_len);
    fill_recv_data_inner(list_str_len_recv.data(), out_lens.data(), hashes,
                         comm_info.send_disp, comm_info.n_pes, out_len);
    convert_len_arr_to_offset(out_lens.data(), out_str_offset, out_len);
#endif
    // 5: the character offsets
    int32_t in_sub_len = in_str_offset[in_len];
    int32_t out_sub_len = out_str_offset[out_len];
    bodo::vector<uint32_t> list_char_len_send(in_sub_len);
    offset_t* out_data_offset = (offset_t*)out_arr->data2();
    for (int64_t i = 0; i < in_sub_len; i++)
        list_char_len_send[i] = in_data_offset[i + 1] - in_data_offset[i];
    bodo::vector<uint32_t> list_char_len_recv(out_sub_len);
    bodo_alltoallv(list_char_len_send.data(), recv_count_sub, recv_disp_sub,
                   mpi_typ, list_char_len_recv.data(), send_count_sub,
                   send_disp_sub, mpi_typ, MPI_COMM_WORLD);
    std::vector<int64_t> tmp_offset_sub(send_disp_sub);
#if OFFSET_BITWIDTH != 32
    out_lens.resize(out_sub_len);
#endif
    for (int64_t i = 0; i < out_len; i++) {
        size_t node = hash_to_rank(hashes[i], n_pes);
        uint32_t nb_str = out_str_offset[i + 1] - out_str_offset[i];
        int64_t c_ind = tmp_offset_sub[node];
#if OFFSET_BITWIDTH == 32
        uint32_t* out_ptr = out_data_offset + out_str_offset[i];
#else
        uint32_t* out_ptr = out_lens.data() + out_str_offset[i];
#endif
        uint32_t* in_ptr = list_char_len_recv.data() + c_ind;
        memcpy((char*)out_ptr, (char*)in_ptr, sizeof(uint32_t) * nb_str);
        tmp_offset_sub[node] += nb_str;
    }
#if OFFSET_BITWIDTH == 32
    convert_len_arr_to_offset32(out_data_offset, out_sub_len);
#else
    convert_len_arr_to_offset(out_lens.data(), out_data_offset, out_sub_len);
#endif
    // 6: the characters themselves
    int32_t out_sub_sub_len = out_data_offset[out_sub_len];
    char* in_char = in_arr->data1();
    char* out_char = out_arr->data1();
    MPI_Datatype mpi_typ8 = get_MPI_typ(Bodo_CTypes::UINT8);
    bodo::vector<char> list_char_recv(out_sub_sub_len);
    bodo_alltoallv(in_char, recv_count_sub_sub, recv_disp_sub_sub, mpi_typ8,
                   list_char_recv.data(), send_count_sub_sub, send_disp_sub_sub,
                   mpi_typ8, MPI_COMM_WORLD);
    std::vector<int64_t> tmp_offset_sub_sub(send_disp_sub_sub);
    for (int64_t i = 0; i < out_len; i++) {
        size_t node = hash_to_rank(hashes[i], n_pes);
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
    int64_t n_send_sub_null_tot = std::accumulate(
        send_count_sub_null.begin(), send_count_sub_null.end(), int64_t(0));
    int64_t n_recv_sub_null_tot = std::accumulate(
        recv_count_sub_null.begin(), recv_count_sub_null.end(), int64_t(0));
    std::vector<int64_t> send_disp_sub_null(n_pes), recv_disp_sub_null(n_pes);
    calc_disp(send_disp_sub_null, send_count_sub_null);
    calc_disp(recv_disp_sub_null, recv_count_sub_null);
    bodo::vector<uint8_t> mask_send(n_recv_sub_null_tot);
    uint8_t* sub_null_bitmask_in = (uint8_t*)in_arr->sub_null_bitmask();
    uint8_t* sub_null_bitmask_out = (uint8_t*)out_arr->sub_null_bitmask();
    for (int i_p = 0; i_p < n_pes; i_p++) {
        for (int64_t i_str = 0; i_str < recv_count_sub[i_p]; i_str++) {
            int64_t pos = i_str + recv_disp_sub[i_p];
            bool bit = GetBit(sub_null_bitmask_in, pos);
            SetBitTo(mask_send.data(), 8 * recv_disp_sub_null[i_p] + i_str,
                     bit);
        }
    }
    bodo::vector<uint8_t> mask_recv(n_send_sub_null_tot);
    bodo_alltoallv(mask_send.data(), recv_count_sub_null, recv_disp_sub_null,
                   mpi_typ8, mask_recv.data(), send_count_sub_null,
                   send_disp_sub_null, mpi_typ8, MPI_COMM_WORLD);
    std::vector<int64_t> tmp_offset(n_pes, 0);
    int64_t pos_si = 0;
    for (size_t i_row = 0; i_row < out_arr->length; i_row++) {
        size_t node = hash_to_rank(hashes[i_row], n_pes);
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
    There is no way we can build a uint32_t*
   compute_reverse_shuffle(std::shared_ptr<uint32_t[]>& hashes)
    ---
    @param in_table  : The shuffled input table
    @param hashes    : the hashes (of the original table)
    @param comm_info : The comm_info (computed from the original table)
    @return the reshuffled table
 */
std::shared_ptr<table_info> reverse_shuffle_table_kernel(
    std::shared_ptr<table_info> in_table, std::shared_ptr<uint32_t[]>& hashes,
    mpi_comm_info const& comm_info) {
    tracing::Event ev("reverse_shuffle_table_kernel");
    uint64_t n_cols = in_table->ncols();
    std::vector<std::shared_ptr<array_info>> out_arrs;
    for (uint64_t i = 0; i < n_cols; i++) {
        std::shared_ptr<array_info> in_arr = in_table->columns[i];
        bodo_array_type::arr_type_enum arr_type = in_arr->arr_type;
        std::shared_ptr<array_info> out_arr = nullptr;
        if (in_arr->arr_type == bodo_array_type::STRUCT ||
            in_arr->arr_type == bodo_array_type::ARRAY_ITEM) {
            Bodo_PyErr_SetString(
                PyExc_RuntimeError,
                "Reverse shuffle for nested data not yet supported");
            return nullptr;

        } else if (arr_type == bodo_array_type::DICT) {
            if (!in_arr->has_global_dictionary) {
                throw std::runtime_error(
                    "reverse_shuffle_table: input dictionary array doesn't "
                    "have a "
                    "global dictionary");
            }
            // in_arr <- indices array, to simplify code below
            in_arr = in_arr->child_arrays[1];
            arr_type = in_arr->arr_type;
        }
        {
            if (arr_type == bodo_array_type::NUMPY ||
                arr_type == bodo_array_type::CATEGORICAL ||
                arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
                out_arr = reverse_shuffle_data_array(in_arr, hashes, comm_info);
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
        if (in_table->columns[i]->arr_type == bodo_array_type::DICT) {
            in_arr = in_table->columns[i];
            std::shared_ptr<array_info> out_dict_arr = create_dict_string_array(
                in_arr->child_arrays[0], out_arr, in_arr->has_global_dictionary,
                in_arr->has_deduped_local_dictionary,
                in_arr->has_sorted_dictionary, in_arr->dict_id);
            out_arr = out_dict_arr;
        }
        in_arr.reset();
        reset_col_if_last_table_ref(in_table, i);
        out_arrs.push_back(out_arr);
    }
    return std::make_shared<table_info>(out_arrs);
}

// NOTE: Steals a reference from the input table.
std::shared_ptr<table_info> shuffle_table(std::shared_ptr<table_info> in_table,
                                          int64_t n_keys, bool is_parallel,
                                          int32_t keep_comm_info,
                                          std::shared_ptr<uint32_t[]> hashes) {
    tracing::Event ev("shuffle_table", is_parallel);
    // error checking
    if (in_table->ncols() <= 0 || n_keys <= 0) {
        Bodo_PyErr_SetString(PyExc_RuntimeError, "Invalid input shuffle table");
        return NULL;
    }

    const bool delete_hashes = bool(hashes);
    std::shared_ptr<mpi_comm_info> comm_info =
        std::make_shared<mpi_comm_info>(in_table->columns);

    // For any dictionary arrays that have local dictionaries, convert their
    // dictionaries to global now (since it's needed for the shuffle) and if
    // any of them are key columns, this will allow to simply hash the
    // indices
    // TODO maybe it's better to do this in hash_keys_table to avoid
    // repeating this code in other operations?
    for (std::shared_ptr<array_info> a : in_table->columns) {
        if (a->arr_type == bodo_array_type::DICT) {
            // XXX is the dictionary replaced in Python input array and
            // correctly replaced? Can it be done easily? (this includes
            // has_global_dictionary attribute). We need dictionaries
            // to be global and unique for hashing.
            make_dictionary_global_and_unique(a, is_parallel);
        }
    }

    // computing the hash data structure
    if (!hashes) {
        hashes =
            hash_keys_table(in_table, n_keys, SEED_HASH_PARTITION, is_parallel);
    }
    comm_info->set_counts(hashes, is_parallel);

    std::shared_ptr<table_info> table = shuffle_table_kernel(
        std::move(in_table), hashes, *comm_info, is_parallel);
    if (keep_comm_info) {
        table->comm_info = comm_info;
        table->hashes = hashes;
    } else {
        if (delete_hashes) {
            hashes.reset();
        }
    }

    return table;
}

table_info* shuffle_table_py_entrypt(table_info* in_table, int64_t n_keys,
                                     bool is_parallel, int32_t keep_comm_info) {
    try {
        std::shared_ptr<table_info> out =
            shuffle_table(std::shared_ptr<table_info>(in_table), n_keys,
                          is_parallel, keep_comm_info);
        return new table_info(*out);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

/**
 * @brief get shuffle info from table struct
 * Using raw pointers since called from Python.
 *
 * @param table input table
 * @return shuffle_info* shuffle info of input table
 */
shuffle_info* get_shuffle_info(table_info* table) {
    return new shuffle_info(table->comm_info, table->hashes);
}

/**
 * @brief free allocated data of shuffle info
 * Called from Python.
 *
 * @param sh_info input shuffle info
 */
void delete_shuffle_info(shuffle_info* sh_info) { delete sh_info; }

// Note: Steals a reference from the input table.
/**
 * @brief reverse a previous shuffle of input table
 * Using raw pointers since called from Python.
 *
 * @param in_table input table
 * @param sh_info shuffle info
 * @return table_info* reverse shuffled output table
 */
table_info* reverse_shuffle_table(table_info* in_table, shuffle_info* sh_info) {
    // TODO: move outside shuffle function.
    // Ideally the convert should happen at the function that
    // calls the reverse.
    for (std::shared_ptr<array_info> a : in_table->columns) {
        if (a->arr_type == bodo_array_type::DICT) {
            // NOTE: This can never be called on replicated data?
            // We need dictionaries to be global and unique for hashing.
            make_dictionary_global_and_unique(a, true);
        }
    }
    std::shared_ptr<table_info> revshuf_table =
        reverse_shuffle_table_kernel(std::shared_ptr<table_info>(in_table),
                                     sh_info->hashes, *sh_info->comm_info);

    return new table_info(*revshuf_table);
}

std::shared_ptr<table_info> coherent_shuffle_table(
    std::shared_ptr<table_info> in_table, std::shared_ptr<table_info> ref_table,
    int64_t n_keys, std::shared_ptr<uint32_t[]> hashes,
    SimdBlockFilterFixed<::hashing::SimpleMixSplit>* filter,
    const uint8_t* null_bitmask, const bool keep_nulls_and_filter_misses) {
    tracing::Event ev("coherent_shuffle_table", true);
    // error checking
    if (in_table->ncols() <= 0 || n_keys <= 0) {
        Bodo_PyErr_SetString(PyExc_RuntimeError, "Invalid input shuffle table");
        return NULL;
    }
    const bool delete_hashes = bool(hashes);
    mpi_comm_info comm_info(in_table->columns);

    // For any dictionary arrays that have local dictionaries, convert their
    // dictionaries to global now (since it's needed for the shuffle) and if
    // any of them are key columns, this will allow to simply hash the
    // indices
    for (std::shared_ptr<array_info> a : in_table->columns) {
        if (a->arr_type == bodo_array_type::DICT) {
            // coherent_shuffle_table only called in join with parallel
            // options. is_parallel = true We need dictionaries to be global
            // and unique for hashing.
            make_dictionary_global_and_unique(a, true);
        }
    }

    // computing the hash data structure
    if (!hashes) {
        hashes = coherent_hash_keys_table(in_table, std::move(ref_table),
                                          n_keys, SEED_HASH_PARTITION, true);
    } else {
        ref_table.reset();
    }
    // coherent_shuffle_table only called in join with parallel options.
    // is_parallel = true
    // Prereq to calling shuffle_table_kernel
    if (keep_nulls_and_filter_misses) {
        comm_info.set_counts<true>(hashes, true, filter, null_bitmask);
    } else {
        comm_info.set_counts<false>(hashes, true, filter, null_bitmask);
    }
    std::shared_ptr<table_info> table =
        shuffle_table_kernel(std::move(in_table), hashes, comm_info, true);
    if (delete_hashes) {
        hashes.reset();
    }
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
        arrow::AllocateBuffer(siz_out, bodo::BufferPool::DefaultPtr());
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
    size_t siz_out = sizeof(offset_t) * (n_rows + 1);
    arrow::Result<std::unique_ptr<arrow::Buffer>> maybe_buffer =
        arrow::AllocateBuffer(siz_out, bodo::BufferPool::DefaultPtr());
    if (!maybe_buffer.ok()) {
        Bodo_PyErr_SetString(PyExc_RuntimeError, "allocation error");
        return nullptr;
    }
    std::shared_ptr<arrow::Buffer> buffer = *std::move(maybe_buffer);
    offset_t* offset_out_ptr = (offset_t*)buffer->mutable_data();
    if (myrank == mpi_root) {
        for (int i_row = 0; i_row <= n_rows; i_row++)
            offset_out_ptr[i_row] = arr->value_offset(i_row);
    }
    MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CType_offset);
    MPI_Bcast(offset_out_ptr, n_rows + 1, mpi_typ, mpi_root, MPI_COMM_WORLD);
    return buffer;
}

std::shared_ptr<arrow::Buffer> broadcast_arrow_primitive_buffer(
    int64_t n_rows, std::shared_ptr<arrow::PrimitiveArray> const& arr) {
    int myrank, mpi_root = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    // broadcasting type id. This is an enum so we just store
    // this in an unsigned integer and cast back.
    uint64_t arrow_type_int = 0;
    if (myrank == mpi_root) {
        arrow_type_int = arr->type()->id();
    }
    MPI_Bcast(&arrow_type_int, 1, MPI_LONG_LONG_INT, mpi_root, MPI_COMM_WORLD);
    arrow::Type::type arrow_type = (arrow::Type::type)arrow_type_int;
    Bodo_CTypes::CTypeEnum bodo_typ = arrow_to_bodo_type(arrow_type);
    int64_t siz_typ = numpy_item_size[bodo_typ];
    size_t siz_out;
    if (arrow_type == arrow::Type::type::BOOL) {
        // Boolean arrays store 1 bit per boolean
        siz_out = (n_rows + 7) >> 3;
    } else {
        siz_out = siz_typ * n_rows;
    }
    arrow::Result<std::unique_ptr<arrow::Buffer>> maybe_buffer =
        arrow::AllocateBuffer(siz_out, bodo::BufferPool::DefaultPtr());
    if (!maybe_buffer.ok()) {
        Bodo_PyErr_SetString(PyExc_RuntimeError, "allocation error");
        return nullptr;
    }
    std::shared_ptr<arrow::Buffer> buffer = *std::move(maybe_buffer);
    uint8_t* data_out_ptr = buffer->mutable_data();
    if (myrank == mpi_root) {
        uint8_t* data_in_ptr = (uint8_t*)arr->values()->data();
        memcpy(data_out_ptr, data_in_ptr, siz_out);
    }
    MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
    MPI_Bcast(data_out_ptr, siz_typ * n_rows, mpi_typ, mpi_root,
              MPI_COMM_WORLD);
    return buffer;
}

std::shared_ptr<arrow::Buffer> broadcast_arrow_string_buffer(
#if OFFSET_BITWIDTH == 32
    int64_t n_rows, std::shared_ptr<arrow::StringArray> const& string_array) {
#else
    int64_t n_rows,
    std::shared_ptr<arrow::LargeStringArray> const& string_array) {
#endif
    int myrank, mpi_root = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    // broadcasting number of characters
    int64_t arr_typ[1];
    if (myrank == mpi_root)
        arr_typ[0] = string_array->value_offset(n_rows);
    MPI_Bcast(arr_typ, 1, MPI_LONG_LONG_INT, mpi_root, MPI_COMM_WORLD);
    int64_t n_chars = arr_typ[0];
    //
    size_t siz_out = n_chars;
    arrow::Result<std::unique_ptr<arrow::Buffer>> maybe_buffer =
        arrow::AllocateBuffer(siz_out, bodo::BufferPool::DefaultPtr());
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
#if OFFSET_BITWIDTH == 32
    if (typ_arrow == arrow::Type::LIST) {
        std::shared_ptr<arrow::ListArray> list_arr = nullptr;
        std::shared_ptr<arrow::ListArray> ref_list_arr =
            std::dynamic_pointer_cast<arrow::ListArray>(ref_arr);
#else
    if (typ_arrow == arrow::Type::LARGE_LIST) {
        std::shared_ptr<arrow::LargeListArray> list_arr = nullptr;
        std::shared_ptr<arrow::LargeListArray> ref_list_arr =
            std::dynamic_pointer_cast<arrow::LargeListArray>(ref_arr);
#endif
        std::shared_ptr<arrow::Array> arr_values = nullptr;
        if (myrank == mpi_root) {
#if OFFSET_BITWIDTH == 32
            list_arr = std::dynamic_pointer_cast<arrow::ListArray>(arr);
#else
            list_arr = std::dynamic_pointer_cast<arrow::LargeListArray>(arr);
#endif
            arr_values = list_arr->values();
        }
        std::shared_ptr<arrow::Buffer> list_offsets =
            broadcast_arrow_offsets_buffer(n_rows, list_arr);
        std::shared_ptr<arrow::Buffer> null_bitmap =
            broadcast_arrow_bitmap_buffer(n_rows, list_arr);
        std::shared_ptr<arrow::Array> child_array =
            broadcast_arrow_array(ref_list_arr->values(), arr_values);
        // Now returning the broadcast array
#if OFFSET_BITWIDTH == 32
        return std::make_shared<arrow::ListArray>(ref_list_arr->type(), n_rows,
#else
        return std::make_shared<arrow::LargeListArray>(
            ref_list_arr->type(), n_rows,
#endif
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
            if (myrank == mpi_root)
                child = struct_arr->field(i_field);
            children.push_back(broadcast_arrow_array(ref_child, child));
        }
        std::shared_ptr<arrow::Buffer> null_bitmap =
            broadcast_arrow_bitmap_buffer(n_rows, struct_arr);
        return std::make_shared<arrow::StructArray>(
            ref_struct_arr->type(), n_rows, children, null_bitmap);
#if OFFSET_BITWIDTH == 32
    } else if (typ_arrow == arrow::Type::STRING) {
        std::shared_ptr<arrow::StringArray> string_array = nullptr;
#else
    } else if (typ_arrow == arrow::Type::LARGE_STRING) {
        std::shared_ptr<arrow::LargeStringArray> string_array = nullptr;
#endif
        if (myrank == mpi_root) {
#if OFFSET_BITWIDTH == 32
            string_array = std::dynamic_pointer_cast<arrow::StringArray>(arr);
#else
            string_array =
                std::dynamic_pointer_cast<arrow::LargeStringArray>(arr);
#endif
        }
        std::shared_ptr<arrow::Buffer> list_offsets =
            broadcast_arrow_offsets_buffer(n_rows, string_array);
        std::shared_ptr<arrow::Buffer> null_bitmap =
            broadcast_arrow_bitmap_buffer(n_rows, string_array);
        std::shared_ptr<arrow::Buffer> data =
            broadcast_arrow_string_buffer(n_rows, string_array);
#if OFFSET_BITWIDTH == 32
        return std::make_shared<arrow::StringArray>(n_rows, list_offsets, data,
#else
        return std::make_shared<arrow::LargeStringArray>(n_rows, list_offsets,
                                                         data,
#endif
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

/* Broadcast the first n_cols of in_table to the other nodes.
   The ref_table contains only the type information. In order to eliminate
   it, we would need to have a broadcast_datatype function.

   For dict columns, we use the global dictionary from columns of
   ref_table (since columns in in_table are null on ranks != 0).
   Ensure that ref_table and in_table share the same dictionary
   and that it is global.

   @param ref_table : the reference table used for the datatype.
   @param in_table : the table that is broadcasted.
   @param n_cols : the number of columns in output
   @param is_parallel: Used to indicate whether tracing should be parallel
   or not
   @return the table put in all the nodes
*/
std::shared_ptr<table_info> broadcast_table(
    std::shared_ptr<table_info> ref_table, std::shared_ptr<table_info> in_table,
    size_t n_cols, bool is_parallel, int mpi_root) {
    tracing::Event ev("broadcast_table", is_parallel);
    int n_pes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    std::vector<std::shared_ptr<array_info>> out_arrs;
    for (size_t i_col = 0; i_col < n_cols; i_col++) {
        int64_t arr_bcast[7];
        std::shared_ptr<array_info> in_arr = nullptr;
        if (myrank == mpi_root) {
            in_arr = in_table->columns[i_col];
            if (in_arr->arr_type == bodo_array_type::DICT) {
                if (!in_arr->has_global_dictionary) {
                    throw std::runtime_error(
                        "broadcast_table: Not supported for DICT arrays "
                        "without global dictionary.");
                }

                // In case of dictionary encoded string arrays,
                // we assume that the dictionary is global.
                // Which means we just need to broadcast the indices
                // array, which is the same logic as NULLABLE_INT_BOOL.
                in_arr = in_arr->child_arrays[1];
            }
            arr_bcast[0] = in_arr->length;
            arr_bcast[1] = in_arr->dtype;
            arr_bcast[2] = in_arr->arr_type;
            arr_bcast[3] = in_arr->n_sub_elems();
            arr_bcast[4] = in_arr->n_sub_sub_elems();
            arr_bcast[5] = in_arr->num_categories;
            arr_bcast[6] = (int64_t)in_arr->precision;
        }
        MPI_Bcast(arr_bcast, 7, MPI_LONG_LONG_INT, mpi_root, MPI_COMM_WORLD);
        int64_t n_rows = arr_bcast[0];
        Bodo_CTypes::CTypeEnum dtype = Bodo_CTypes::CTypeEnum(arr_bcast[1]);
        bodo_array_type::arr_type_enum arr_type =
            bodo_array_type::arr_type_enum(arr_bcast[2]);
        int64_t n_sub_elems = arr_bcast[3];
        int64_t n_sub_sub_elems = arr_bcast[4];
        int64_t num_categories = arr_bcast[5];
        int32_t precision = (int32_t)arr_bcast[6];
        //
        std::shared_ptr<array_info> out_arr = nullptr;
        if (arr_type == bodo_array_type::STRUCT ||
            arr_type == bodo_array_type::ARRAY_ITEM) {
            std::shared_ptr<arrow::Array> ref_array =
                to_arrow(ref_table->columns[i_col]);
            std::shared_ptr<arrow::Array> in_array = nullptr;
            if (myrank == mpi_root) {
                in_array = to_arrow(in_arr);
            }
            std::shared_ptr<arrow::Array> array =
                broadcast_arrow_array(ref_array, in_array);
            out_arr = arrow_array_to_bodo(array);
        }
        if (arr_type == bodo_array_type::NUMPY ||
            arr_type == bodo_array_type::CATEGORICAL ||
            arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
            MPI_Datatype mpi_typ = get_MPI_typ(dtype);
            if (myrank == mpi_root) {
                out_arr = copy_array(in_arr);
            } else {
                out_arr = alloc_array(n_rows, -1, -1, arr_type, dtype, 0, 0);
            }
            out_arr->precision = precision;
            uint64_t bcast_size;
            if (arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
                dtype == Bodo_CTypes::_BOOL) {
                // Nullable booleans store 1 bit per boolean.
                mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
                bcast_size = (n_rows + 7) >> 3;
            } else {
                bcast_size = n_rows;
            }
            MPI_Bcast(out_arr->data1(), bcast_size, mpi_typ, mpi_root,
                      MPI_COMM_WORLD);
        }
        if (arr_type == bodo_array_type::INTERVAL) {
            MPI_Datatype mpi_typ = get_MPI_typ(dtype);
            if (myrank == mpi_root) {
                out_arr = copy_array(in_arr);
            } else {
                out_arr = alloc_array(n_rows, -1, -1, arr_type, dtype, 0, 0);
            }
            MPI_Bcast(out_arr->data1(), n_rows, mpi_typ, mpi_root,
                      MPI_COMM_WORLD);
            MPI_Bcast(out_arr->data2(), n_rows, mpi_typ, mpi_root,
                      MPI_COMM_WORLD);
        }
        if (arr_type == bodo_array_type::STRING) {
            MPI_Datatype mpi_typ_offset = get_MPI_typ(Bodo_CType_offset);
            MPI_Datatype mpi_typ8 = get_MPI_typ(Bodo_CTypes::UINT8);
            if (myrank == mpi_root)
                out_arr = copy_array(in_arr);
            else
                out_arr = alloc_array(n_rows, n_sub_elems, n_sub_sub_elems,
                                      arr_type, dtype, 0, num_categories);
            MPI_Bcast(out_arr->data1(), n_sub_elems, mpi_typ8, mpi_root,
                      MPI_COMM_WORLD);
            MPI_Bcast(out_arr->data2(), n_rows, mpi_typ_offset, mpi_root,
                      MPI_COMM_WORLD);
        }
        if (arr_type == bodo_array_type::LIST_STRING) {
            MPI_Datatype mpi_typ_offset = get_MPI_typ(Bodo_CType_offset);
            MPI_Datatype mpi_typ8 = get_MPI_typ(Bodo_CTypes::UINT8);
            if (myrank == mpi_root)
                out_arr = copy_array(in_arr);
            else
                out_arr = alloc_array(n_rows, n_sub_elems, n_sub_sub_elems,
                                      arr_type, dtype, 0, num_categories);
            MPI_Bcast(out_arr->data1(), n_sub_sub_elems, mpi_typ8, mpi_root,
                      MPI_COMM_WORLD);
            MPI_Bcast(out_arr->data2(), n_sub_elems, mpi_typ_offset, mpi_root,
                      MPI_COMM_WORLD);
            MPI_Bcast(out_arr->data3(), n_rows, mpi_typ_offset, mpi_root,
                      MPI_COMM_WORLD);
            MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
            int n_sub_bytes = (n_sub_elems + 7) >> 3;
            MPI_Bcast(out_arr->sub_null_bitmask(), n_sub_bytes, mpi_typ,
                      mpi_root, MPI_COMM_WORLD);
        }
        if (arr_type == bodo_array_type::NULLABLE_INT_BOOL ||
            arr_type == bodo_array_type::STRING ||
            arr_type == bodo_array_type::LIST_STRING) {
            MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
            int n_bytes = (n_rows + 7) >> 3;
            MPI_Bcast(out_arr->null_bitmask(), n_bytes, mpi_typ, mpi_root,
                      MPI_COMM_WORLD);
        }

        // Handle the case that input arr is DICT (we changed reference of
        // in_arr to the indices portion earlier).
        // At this point out_arr is a NULLABLE_INT_BOOL array and contains
        // the indices.
        // We assume the ref_table has the correct global
        // dictionary that can be used for the final dictionary output
        // array.
        if (ref_table->columns[i_col]->arr_type == bodo_array_type::DICT) {
            std::shared_ptr<array_info> ref_arr = ref_table->columns[i_col];
            std::shared_ptr<array_info> dict_arr = ref_arr->child_arrays[0];
            // Create a DICT out_arr
            out_arr = create_dict_string_array(
                dict_arr, out_arr, ref_arr->has_global_dictionary,
                ref_arr->has_deduped_local_dictionary,
                ref_arr->has_sorted_dictionary, ref_arr->dict_id);
        }
        in_arr.reset();
        out_arrs.push_back(out_arr);
    }

    return std::make_shared<table_info>(out_arrs);
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
    if (arr->length() >= INT_MAX)
        throw std::runtime_error(
            "gather_arrow_offset_buffer: exceeded size limit");
    int n_rows = static_cast<int>(arr->length());
    MPI_Gengather(&n_rows, 1, MPI_INT, rows_count.data(), 1, MPI_INT, mpi_root,
                  MPI_COMM_WORLD, all_gather);
    int64_t n_rows_tot =
        std::accumulate(rows_count.begin(), rows_count.end(), int64_t(0));
    if (myrank == mpi_root || all_gather) {
        int64_t rows_pos = 0;
        for (int i_p = 0; i_p < n_pes; i_p++) {
            rows_disps[i_p] = rows_pos;
            rows_pos += rows_count[i_p];
        }
        if (rows_pos >= INT_MAX)
            throw std::runtime_error(
                "gather_arrow_offset_buffer: exceeded size limit");
    }
    bodo::vector<offset_t> list_siz_loc(n_rows);
    for (size_t i = 0; i < size_t(n_rows); i++)
        list_siz_loc[i] = arr->value_offset(i + 1) - arr->value_offset(i);
    MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CType_offset);
    bodo::vector<offset_t> list_siz_tot(n_rows_tot);
    MPI_Gengatherv(list_siz_loc.data(), n_rows, mpi_typ, list_siz_tot.data(),
                   rows_count.data(), rows_disps.data(), mpi_typ, mpi_root,
                   MPI_COMM_WORLD, all_gather);
    if (myrank == mpi_root || all_gather) {
        size_t siz_out = sizeof(offset_t) * (n_rows_tot + 1);
        arrow::Result<std::unique_ptr<arrow::Buffer>> maybe_buffer =
            arrow::AllocateBuffer(siz_out, bodo::BufferPool::DefaultPtr());
        if (!maybe_buffer.ok()) {
            throw std::runtime_error(
                "gather_arrow_offset_buffer: allocation error");
        }
        std::shared_ptr<arrow::Buffer> buffer = *std::move(maybe_buffer);
        offset_t* data_out_ptr = (offset_t*)buffer->mutable_data();
        offset_t pos = 0;
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
        for (size_t i = 0; i < size_t(n_pes); i++) {
            recv_count_null[i] = (rows_count[i] + 7) >> 3;
        }
        calc_disp(recv_disp_null, recv_count_null);
    }
    int n_bytes = (n_rows + 7) >> 3;
    bodo::vector<uint8_t> send_null_bitmask(n_bytes, 0);
    for (int i_row = 0; i_row < n_rows; i_row++) {
        SetBitTo(send_null_bitmask.data(), i_row, !arr->IsNull(i_row));
    }

    int n_null_bytes =
        std::accumulate(recv_count_null.begin(), recv_count_null.end(), 0);
    bodo::vector<uint8_t> tmp_null_bytes(n_null_bytes, 0);
    MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
    MPI_Gengatherv(send_null_bitmask.data(), n_bytes, mpi_typ,
                   tmp_null_bytes.data(), recv_count_null.data(),
                   recv_disp_null.data(), mpi_typ, mpi_root, MPI_COMM_WORLD,
                   all_gather);
    if (myrank == mpi_root || all_gather) {
        size_t siz_out = (n_rows_tot + 7) >> 3;
        arrow::Result<std::unique_ptr<arrow::Buffer>> maybe_buffer =
            arrow::AllocateBuffer(siz_out, bodo::BufferPool::DefaultPtr());
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
#if OFFSET_BITWIDTH == 32
    std::shared_ptr<arrow::StringArray> const& arr, bool all_gather) {
#else
    std::shared_ptr<arrow::LargeStringArray> const& arr, bool all_gather) {
#endif
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
    if (myrank == mpi_root || all_gather)
        calc_disp(char_disps, char_count);
    std::shared_ptr<arrow::Buffer> buffer = nullptr;
    uint8_t* data_out_ptr = nullptr;
    if (myrank == mpi_root || all_gather) {
        size_t siz_out = n_char_tot;
        arrow::Result<std::unique_ptr<arrow::Buffer>> maybe_buffer =
            arrow::AllocateBuffer(siz_out, bodo::BufferPool::DefaultPtr());
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
    auto typ = arr->type();
    Bodo_CTypes::CTypeEnum bodo_typ = arrow_to_bodo_type(typ->id());
    int64_t siz_typ = numpy_item_size[bodo_typ];
    // Determination of sizes
    std::vector<int> char_count, char_count_bytes, char_disps;
    if (myrank == mpi_root || all_gather) {
        char_count.resize(n_pes);
        char_disps.resize(n_pes);
        if (typ->id() == arrow::Type::BOOL) {
            char_count_bytes.resize(n_pes);
        }
    }
    int n_char = n_rows;
    // Boolean arrays store 1 bit per boolean so we will
    // have some custom code paths.
    if (typ->id() != arrow::Type::BOOL) {
        n_char = n_rows * siz_typ;
    }
    MPI_Gengather(&n_char, 1, MPI_INT, char_count.data(), 1, MPI_INT, mpi_root,
                  MPI_COMM_WORLD, all_gather);
    int n_char_tot = std::accumulate(char_count.begin(), char_count.end(), 0);
    if (myrank == mpi_root || all_gather) {
        if (typ->id() == arrow::Type::BOOL) {
            for (size_t i = 0; i < size_t(n_pes); i++) {
                char_count_bytes[i] = (char_count[i] + 7) >> 3;
            }
            calc_disp(char_disps, char_count_bytes);
        } else {
            calc_disp(char_disps, char_count);
        }
    }
    std::shared_ptr<arrow::Buffer> buffer = nullptr;
    if (typ->id() == arrow::Type::BOOL) {
        // Boolean requires a very different send path because we use
        // 1 bit per boolean.
        int n_bytes = (n_rows + 7) >> 3;
        int64_t n_total_bytes =
            std::accumulate(char_disps.begin(), char_disps.end(), 0);
        bodo::vector<uint8_t> tmp_data_bytes(n_total_bytes, 0);
        MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
        MPI_Gengatherv((uint8_t*)arr->values()->data(), n_bytes, mpi_typ,
                       tmp_data_bytes.data(), char_count_bytes.data(),
                       char_disps.data(), mpi_typ, mpi_root, MPI_COMM_WORLD,
                       all_gather);
        if (myrank == mpi_root || all_gather) {
            size_t siz_out = (n_char_tot + 7) >> 3;
            arrow::Result<std::unique_ptr<arrow::Buffer>> maybe_buffer =
                arrow::AllocateBuffer(siz_out, bodo::BufferPool::DefaultPtr());
            if (!maybe_buffer.ok()) {
                Bodo_PyErr_SetString(PyExc_RuntimeError, "allocation error");
                return nullptr;
            }
            buffer = *std::move(maybe_buffer);
            uint8_t* data_out_ptr = buffer->mutable_data();
            copy_gathered_null_bytes(data_out_ptr, tmp_data_bytes,
                                     char_count_bytes, char_count);
        }
    } else {
        uint8_t* data_out_ptr = nullptr;
        if (myrank == mpi_root || all_gather) {
            size_t siz_out = n_char_tot;
            arrow::Result<std::unique_ptr<arrow::Buffer>> maybe_buffer =
                arrow::AllocateBuffer(siz_out, bodo::BufferPool::DefaultPtr());
            if (!maybe_buffer.ok()) {
                Bodo_PyErr_SetString(PyExc_RuntimeError, "allocation error");
                return nullptr;
            }
            buffer = *std::move(maybe_buffer);
            data_out_ptr = buffer->mutable_data();
        }
        uint8_t* send_data = (uint8_t*)arr->values()->data();
        MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
        MPI_Gengatherv(send_data, n_char, mpi_typ, data_out_ptr,
                       char_count.data(), char_disps.data(), mpi_typ, mpi_root,
                       MPI_COMM_WORLD, all_gather);
    }
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
#if OFFSET_BITWIDTH == 32
    if (arr->type_id() == arrow::Type::LIST) {
        std::shared_ptr<arrow::ListArray> list_arr =
            std::dynamic_pointer_cast<arrow::ListArray>(arr);
#else
    if (arr->type_id() == arrow::Type::LARGE_LIST) {
        std::shared_ptr<arrow::LargeListArray> list_arr =
            std::dynamic_pointer_cast<arrow::LargeListArray>(arr);
#endif
        std::shared_ptr<arrow::Buffer> list_offsets =
            gather_arrow_offset_buffer(list_arr, all_gather);
        std::shared_ptr<arrow::Buffer> null_bitmap_out =
            gather_arrow_bitmap_buffer(list_arr, all_gather);
        std::shared_ptr<arrow::Array> child_array =
            gather_arrow_array(list_arr->values(), all_gather);
        if (myrank == mpi_root || all_gather) {
#if OFFSET_BITWIDTH == 32
            return std::make_shared<arrow::ListArray>(
#else
            return std::make_shared<arrow::LargeListArray>(
#endif
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
#if OFFSET_BITWIDTH == 32
    } else if (arr->type_id() == arrow::Type::STRING) {
        std::shared_ptr<arrow::StringArray> string_arr =
            std::dynamic_pointer_cast<arrow::StringArray>(arr);
#else
    } else if (arr->type_id() == arrow::Type::LARGE_STRING) {
        std::shared_ptr<arrow::LargeStringArray> string_arr =
            std::dynamic_pointer_cast<arrow::LargeStringArray>(arr);
#endif
        std::shared_ptr<arrow::Buffer> list_offsets =
            gather_arrow_offset_buffer(string_arr, all_gather);
        std::shared_ptr<arrow::Buffer> null_bitmap_out =
            gather_arrow_bitmap_buffer(string_arr, all_gather);
        std::shared_ptr<arrow::Buffer> data =
            gather_arrow_string_buffer(string_arr, all_gather);
        if (myrank == mpi_root || all_gather) {
#if OFFSET_BITWIDTH == 32
            return std::make_shared<arrow::StringArray>(
#else
            return std::make_shared<arrow::LargeStringArray>(
#endif
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

/**
 * @param is_parallel: Used to indicate whether tracing should be parallel
 * or not
 */
std::shared_ptr<table_info> gather_table(std::shared_ptr<table_info> in_table,
                                         int64_t n_cols_i, bool all_gather,
                                         bool is_parallel = true) {
    tracing::Event ev("gather_table", is_parallel);
    int n_pes, myrank;
    int mpi_root = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    std::vector<std::shared_ptr<array_info>> out_arrs;
    size_t n_cols;
    if (n_cols_i == -1) {
        n_cols = in_table->ncols();
    } else {
        n_cols = n_cols_i;
    }
    for (size_t i_col = 0; i_col < n_cols; i_col++) {
        int64_t arr_gath_s[3];
        std::shared_ptr<array_info> in_arr = in_table->columns[i_col];
        if (in_arr->arr_type == bodo_array_type::DICT) {
            // Note: We need to revisit if gather_table should be
            // a no-op if is_parallel=False
            make_dictionary_global_and_unique(in_arr, true);
            in_arr = in_arr->child_arrays[1];
        }
        int64_t n_rows = in_arr->length;
        int64_t n_sub_elems = in_arr->n_sub_elems();
        int64_t n_sub_sub_elems = in_arr->n_sub_sub_elems();
        arr_gath_s[0] = n_rows;
        arr_gath_s[1] = n_sub_elems;
        arr_gath_s[2] = n_sub_sub_elems;
        bodo::vector<int64_t> arr_gath_r(3 * n_pes, 0);
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
        std::shared_ptr<array_info> out_arr = NULL;
        if (arr_type == bodo_array_type::STRUCT ||
            arr_type == bodo_array_type::ARRAY_ITEM) {
            std::shared_ptr<arrow::Array> array = to_arrow(in_arr);
            std::shared_ptr<arrow::Array> out_array =
                gather_arrow_array(array, all_gather);
            out_arr = arrow_array_to_bodo(out_array);
        }
        if (arr_type == bodo_array_type::NUMPY ||
            arr_type == bodo_array_type::CATEGORICAL ||
            arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
            // Computing the total number of rows.
            // On mpi_root, all rows, on others just 1 row for consistency.
            int64_t n_rows_tot = 0;
            for (int i_p = 0; i_p < n_pes; i_p++) {
                n_rows_tot += rows_counts[i_p];
            }
            if (arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
                dtype == Bodo_CTypes::_BOOL) {
                // Nullable boolean arrays store 1 bit per boolean. As
                // a result we need a separate code path to handle
                // fusing the bits
                char* data_arr_i = in_arr->data1();
                std::vector<int> recv_count_bytes(n_pes),
                    recv_disp_bytes(n_pes);
                for (int i_p = 0; i_p < n_pes; i_p++) {
                    recv_count_bytes[i_p] = (rows_counts[i_p] + 7) >> 3;
                }
                calc_disp(recv_disp_bytes, recv_count_bytes);
                size_t n_data_bytes =
                    std::accumulate(recv_count_bytes.begin(),
                                    recv_count_bytes.end(), size_t(0));
                bodo::vector<uint8_t> tmp_data_bytes(n_data_bytes, 0);
                // Boolean arrays always store data as UINT8
                MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
                int n_bytes = (n_rows + 7) >> 3;
                MPI_Gengatherv(data_arr_i, n_bytes, mpi_typ,
                               tmp_data_bytes.data(), recv_count_bytes.data(),
                               recv_disp_bytes.data(), mpi_typ, mpi_root,
                               MPI_COMM_WORLD, all_gather);
                if (myrank == mpi_root || all_gather) {
                    out_arr = alloc_array(n_rows_tot, -1, -1, arr_type, dtype,
                                          0, num_categories);
                    uint8_t* data_arr_o = (uint8_t*)out_arr->data1();
                    copy_gathered_null_bytes(data_arr_o, tmp_data_bytes,
                                             recv_count_bytes, rows_counts);
                }
            } else {
                MPI_Datatype mpi_typ = get_MPI_typ(dtype);
                char* data1_ptr = NULL;
                if (myrank == mpi_root || all_gather) {
                    out_arr = alloc_array(n_rows_tot, -1, -1, arr_type, dtype,
                                          0, num_categories);
                    data1_ptr = out_arr->data1();
                }
                MPI_Gengatherv(in_arr->data1(), n_rows, mpi_typ, data1_ptr,
                               rows_counts.data(), rows_disps.data(), mpi_typ,
                               mpi_root, MPI_COMM_WORLD, all_gather);
            }
        }
        if (arr_type == bodo_array_type::INTERVAL) {
            MPI_Datatype mpi_typ = get_MPI_typ(dtype);
            // Computing the total number of rows.
            // On mpi_root, all rows, on others just 1 row for consistency.
            int64_t n_rows_tot = 0;
            for (int i_p = 0; i_p < n_pes; i_p++)
                n_rows_tot += arr_gath_r[3 * i_p];
            char* data1_ptr = NULL;
            char* data2_ptr = NULL;
            if (myrank == mpi_root || all_gather) {
                out_arr = alloc_array(n_rows_tot, -1, -1, arr_type, dtype, 0,
                                      num_categories);
                data1_ptr = out_arr->data1();
                data2_ptr = out_arr->data2();
            }
            MPI_Gengatherv(in_arr->data1(), n_rows, mpi_typ, data1_ptr,
                           rows_counts.data(), rows_disps.data(), mpi_typ,
                           mpi_root, MPI_COMM_WORLD, all_gather);
            MPI_Gengatherv(in_arr->data2(), n_rows, mpi_typ, data2_ptr,
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
                data1_ptr = out_arr->data1();
            }
            std::vector<int> char_disps(n_pes), char_counts(n_pes);
            int pos = 0;
            for (int i_p = 0; i_p < n_pes; i_p++) {
                int siz = arr_gath_r[3 * i_p + 1];
                char_disps[i_p] = pos;
                char_counts[i_p] = siz;
                pos += siz;
            }
            MPI_Gengatherv(in_arr->data1(), n_sub_elems, mpi_typ8, data1_ptr,
                           char_counts.data(), char_disps.data(), mpi_typ8,
                           mpi_root, MPI_COMM_WORLD, all_gather);
            // Collecting the offsets data
            bodo::vector<uint32_t> list_count_loc(n_rows);
            offset_t* offsets_i = (offset_t*)in_arr->data2();
            offset_t curr_offset = 0;
            for (int64_t pos = 0; pos < n_rows; pos++) {
                offset_t new_offset = offsets_i[pos + 1];
                list_count_loc[pos] = new_offset - curr_offset;
                curr_offset = new_offset;
            }
            bodo::vector<uint32_t> list_count_tot(n_rows_tot);
            MPI_Gengatherv(list_count_loc.data(), n_rows, mpi_typ32,
                           list_count_tot.data(), rows_counts.data(),
                           rows_disps.data(), mpi_typ32, mpi_root,
                           MPI_COMM_WORLD, all_gather);
            if (myrank == mpi_root || all_gather) {
                offset_t* offsets_o = (offset_t*)out_arr->data2();
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
                n_sub_bytes_count.begin(), n_sub_bytes_count.end(), int64_t(0));
            bodo::vector<uint8_t> V(n_sub_bytes_tot, 0);
            uint8_t* sub_null_bitmask_i = (uint8_t*)in_arr->sub_null_bitmask();
            int n_bytes = (n_sub_elems + 7) >> 3;
            MPI_Gengatherv(sub_null_bitmask_i, n_bytes, mpi_typ8, V.data(),
                           n_sub_bytes_count.data(), n_sub_bytes_disp.data(),
                           mpi_typ8, mpi_root, MPI_COMM_WORLD, all_gather);
            char* data1_ptr = NULL;
            if (myrank == mpi_root || all_gather) {
                out_arr = alloc_array(n_rows_tot, n_sub_elems_tot,
                                      n_sub_sub_elems_tot, arr_type, dtype, 0,
                                      num_categories);
                data1_ptr = out_arr->data1();
                uint8_t* sub_null_bitmask_o =
                    (uint8_t*)out_arr->sub_null_bitmask();
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
            MPI_Gengatherv(in_arr->data1(), n_sub_sub_elems, mpi_typ8,
                           data1_ptr, char_counts.data(), char_disps.data(),
                           mpi_typ8, mpi_root, MPI_COMM_WORLD, all_gather);
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
            bodo::vector<uint32_t> len_strings_loc(n_sub_elems);
            offset_t* data_offsets_i = (offset_t*)in_arr->data2();
            offset_t curr_data_offset = 0;
            for (int64_t pos = 0; pos < n_sub_elems; pos++) {
                offset_t new_data_offset = data_offsets_i[pos + 1];
                uint32_t len_str = new_data_offset - curr_data_offset;
                len_strings_loc[pos] = len_str;
                curr_data_offset = new_data_offset;
            }
            bodo::vector<uint32_t> len_strings_tot(n_sub_elems_tot);
            MPI_Gengatherv(len_strings_loc.data(), n_sub_elems, mpi_typ32,
                           len_strings_tot.data(), data_offsets_counts.data(),
                           data_offsets_disps.data(), mpi_typ32, mpi_root,
                           MPI_COMM_WORLD, all_gather);
            if (myrank == mpi_root || all_gather) {
                offset_t* data_offsets_o = (offset_t*)out_arr->data2();
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
            bodo::vector<uint32_t> n_strings_loc(n_rows);
            offset_t* index_offsets_i = (offset_t*)in_arr->data3();
            offset_t curr_index_offset = 0;
            for (int64_t pos = 0; pos < n_rows; pos++) {
                offset_t new_index_offset = index_offsets_i[pos + 1];
                uint32_t n_str = new_index_offset - curr_index_offset;
                n_strings_loc[pos] = n_str;
                curr_index_offset = new_index_offset;
            }
            bodo::vector<uint32_t> n_strings_tot(n_rows_tot, 405);
            MPI_Gengatherv(n_strings_loc.data(), n_rows, mpi_typ32,
                           n_strings_tot.data(), index_offsets_counts.data(),
                           index_offsets_disps.data(), mpi_typ32, mpi_root,
                           MPI_COMM_WORLD, all_gather);
            if (myrank == mpi_root || all_gather) {
                offset_t* index_offsets_o = (offset_t*)out_arr->data3();
                index_offsets_o[0] = 0;
                for (int64_t pos = 0; pos < n_rows_tot; pos++)
                    index_offsets_o[pos + 1] =
                        index_offsets_o[pos] + n_strings_tot[pos];
            }
        }
        if (arr_type == bodo_array_type::STRING ||
            arr_type == bodo_array_type::LIST_STRING ||
            arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
            char* null_bitmask_i = in_arr->null_bitmask();
            std::vector<int> recv_count_null(n_pes), recv_disp_null(n_pes);
            for (int i_p = 0; i_p < n_pes; i_p++) {
                recv_count_null[i_p] = (rows_counts[i_p] + 7) >> 3;
            }
            calc_disp(recv_disp_null, recv_count_null);
            size_t n_null_bytes = std::accumulate(
                recv_count_null.begin(), recv_count_null.end(), size_t(0));
            bodo::vector<uint8_t> tmp_null_bytes(n_null_bytes, 0);
            MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
            int n_bytes = (n_rows + 7) >> 3;
            MPI_Gengatherv(null_bitmask_i, n_bytes, mpi_typ,
                           tmp_null_bytes.data(), recv_count_null.data(),
                           recv_disp_null.data(), mpi_typ, mpi_root,
                           MPI_COMM_WORLD, all_gather);
            if (myrank == mpi_root || all_gather) {
                char* null_bitmask_o = out_arr->null_bitmask();
                copy_gathered_null_bytes((uint8_t*)null_bitmask_o,
                                         tmp_null_bytes, recv_count_null,
                                         rows_counts);
            }
        }
        if (in_table->columns[i_col]->arr_type == bodo_array_type::DICT) {
            in_arr = in_table->columns[i_col];
            if (all_gather || myrank == mpi_root) {
                out_arr = create_dict_string_array(
                    in_arr->child_arrays[0], out_arr,
                    in_arr->has_global_dictionary,
                    in_arr->has_deduped_local_dictionary,
                    in_arr->has_sorted_dictionary, in_arr->dict_id);
            }  // else out_arr is already NULL, so doesn't need to be
               // handled
        }
        in_arr.reset();
        out_arrs.push_back(out_arr);
    }
    return std::make_shared<table_info>(out_arrs);
}

/* Whether or not a reshuffling is needed.
   The idea is following:
   ---What slows down the running is if one or 2 processors have a much
   higher load than other because it serializes the computation.
   ---If 1 or 2 processors have little load then that is not so bad. It just
   decreases the number of effective processors used.
   ---Thus the metric to consider or not a reshuffling is
          (max nb_row) / (avg nb_row)
   ---If the value is larger than 2 then reshuffling is interesting
 */
bool need_reshuffling(std::shared_ptr<table_info> in_table,
                      double crit_fraction) {
    int64_t n_rows = in_table->nrows(), sum_n_rows, max_n_rows;
    int n_pes;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    if (n_pes == 1)
        return false;
    MPI_Allreduce(&n_rows, &sum_n_rows, 1, MPI_LONG_LONG_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&n_rows, &max_n_rows, 1, MPI_LONG_LONG_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    double avg_n_rows = ceil(double(sum_n_rows) / double(n_pes));
    double objective_measure = double(max_n_rows) / avg_n_rows;
    bool result = objective_measure > crit_fraction;
    return result;
}

std::shared_ptr<table_info> shuffle_renormalization_group(
    std::shared_ptr<table_info> in_table, const int random, int64_t random_seed,
    bool parallel, int64_t n_dest_ranks, int* dest_ranks) {
    tracing::Event ev("shuffle_renormalization_group", parallel);
    if (!parallel && !random) {
        return in_table;
    }
    ev.add_attribute("n_dest_ranks", n_dest_ranks);
    ev.add_attribute("random", random);
    int64_t n_rows = in_table->nrows();
    int n_src_pes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_src_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    int64_t n_rows_tot;
    int64_t shift = 0;
    if (parallel) {
        std::vector<int64_t> AllSizes(n_src_pes);
        MPI_Allgather(&n_rows, 1, MPI_INT64_T, AllSizes.data(), 1, MPI_INT64_T,
                      MPI_COMM_WORLD);
        n_rows_tot =
            std::accumulate(AllSizes.begin(), AllSizes.end(), int64_t(0));
        for (int i_p = 0; i_p < myrank; i_p++) {
            shift += AllSizes[i_p];
        }
    } else {
        n_rows_tot = n_rows;
    }
    ev.add_attribute("nrows", n_rows);
    ev.add_attribute("g_nrows", n_rows_tot);

    bodo::vector<int64_t> random_order;
    std::mt19937 g;  // rng
    if (random) {
        if (random == 1) {  // seed not provided
            if (dist_get_rank() == 0) {
                std::random_device rd;
                random_seed = rd();
            }
            MPI_Bcast(&random_seed, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
        }
        random_order.resize(n_rows_tot);
        for (int64_t i = 0; i < n_rows_tot; i++) {
            random_order[i] = i;
        }
        g.seed(random_seed);
        std::shuffle(random_order.begin(), random_order.end(), g);
        if (!parallel) {
            return RetrieveTable(std::move(in_table), random_order, -1);
        }
    }

    // We use the word "hashes" as they are used all over the shuffle code.
    // However, in that case, it does not mean literally "hash". What it
    // means is the global rank to which the row is going to be sent.
    std::shared_ptr<uint32_t[]> hashes = std::make_unique<uint32_t[]>(n_rows);
    if (n_dest_ranks > 0) {
        // take data from all ranks and distribute to a subset of ranks
        for (int64_t i_row = 0; i_row < n_rows; i_row++) {
            int64_t global_row = shift + i_row;
            if (random) {
                global_row = random_order[global_row];
            }
            int64_t rank =
                dest_ranks[index_rank(n_rows_tot, n_dest_ranks, global_row)];
            hashes[i_row] = rank;
        }
    } else {
        // distributed data among all ranks
        n_dest_ranks = n_src_pes;
        for (int64_t i_row = 0; i_row < n_rows; i_row++) {
            int64_t global_row = shift + i_row;
            if (random) {
                global_row = random_order[global_row];
            }
            int64_t rank = index_rank(n_rows_tot, n_dest_ranks, global_row);
            hashes[i_row] = rank;
        }
    }
    //
    mpi_comm_info comm_info(in_table->columns);
    comm_info.set_counts(hashes, parallel);
    std::shared_ptr<table_info> ret_table =
        shuffle_table_kernel(std::move(in_table), hashes, comm_info, parallel);
    if (random) {
        // data arrives ordered by source and for each source in its
        // original (not random) order, so we need to do a local random
        // shuffle
        n_rows = ret_table->nrows();
        random_order.resize(n_rows);
        for (int64_t i = 0; i < n_rows; i++) {
            random_order[i] = i;
        }
        std::shuffle(random_order.begin(), random_order.end(), g);
        std::shared_ptr<table_info> shuffled_table =
            RetrieveTable(std::move(ret_table), random_order, -1);
        ret_table = shuffled_table;
    }
    ev.add_attribute("ret_table_nrows", ret_table->nrows());
    return ret_table;
}

table_info* shuffle_renormalization_group_py_entrypt(
    table_info* in_table, int random, int64_t random_seed, bool parallel,
    int64_t n_dest_ranks, int* dest_ranks) {
    try {
        std::shared_ptr<table_info> out_table = shuffle_renormalization_group(
            std::shared_ptr<table_info>(in_table), random, random_seed,
            parallel, n_dest_ranks, dest_ranks);
        return new table_info(*out_table);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

/* Apply a renormalization shuffling
   After the operation, all nodes will have a standard size
 */
std::shared_ptr<table_info> shuffle_renormalization(
    std::shared_ptr<table_info> in_table, int random, int64_t random_seed,
    bool parallel) {
    return shuffle_renormalization_group(std::move(in_table), random,
                                         random_seed, parallel, 0, nullptr);
}

table_info* shuffle_renormalization_py_entrypt(table_info* in_table, int random,
                                               int64_t random_seed,
                                               bool parallel) {
    try {
        std::shared_ptr<table_info> out_table =
            shuffle_renormalization(std::shared_ptr<table_info>(in_table),
                                    random, random_seed, parallel);
        return new table_info(*out_table);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

bool shuffle_this_iter(const bool parallel, const bool is_last,
                       const std::shared_ptr<table_info>& shuffle_table,
                       const uint64_t iter) {
    if (!parallel) {
        return false;
    }
    if (is_last) {
        return true;
    }

    if (iter % SHUFFLE_SYNC_ITER == 0) {
        uint64_t local_shuffle_buffer_nrows = shuffle_table->nrows();
        uint64_t reduced_shuffle_buffer_nrows;
        MPI_Allreduce(&local_shuffle_buffer_nrows,
                      &reduced_shuffle_buffer_nrows, 1, MPI_UINT64_T, MPI_MAX,
                      MPI_COMM_WORLD);
        return reduced_shuffle_buffer_nrows >= SHUFFLE_THRESHOLD;
    }
    return false;
}

#undef SHUFFLE_THRESHOLD
#undef SHUFFLE_SYNC_ITER
