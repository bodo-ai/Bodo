#include "_shuffle.h"

#include <numeric>
#include <random>
#include <span>

#include <arrow/api.h>

#include "_array_hash.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_dict_builder.h"
#include "_distributed.h"
#include "_mpi.h"

/**
 * @brief Template used to handle the unmatchable rows in mpi_comm_info. This is
 * used in two cases:
 * - A bloom filter is provided for additional filtering and we need to decide
 * what to do with the misses.
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

/**
 * @brief Generate inner array's row_dest vector from the row_dest of the parent
 * ARRAY_ITEM array
 *
 * @param parent_arr The parent ARRAY_ITEM array
 * @param row_dest Vector keeping track of the destination for each row
 */
bodo::vector<int> get_inner_array_row_dest(
    const std::shared_ptr<array_info>& parent_arr,
    const bodo::vector<int>& row_dest) {
    if (parent_arr->arr_type != bodo_array_type::ARRAY_ITEM) {
        throw std::runtime_error(
            "get_inner_array_row_dest: parent array type must be ARRAY_ITEM, "
            "not " +
            GetArrType_as_string(parent_arr->arr_type) + "!");
    }
    bodo::vector<int> row_dest_inner;
    offset_t* offsets =
        (offset_t*)parent_arr->data1<bodo_array_type::ARRAY_ITEM>();
    row_dest_inner.reserve(offsets[parent_arr->length]);
    for (size_t i = 0; i < parent_arr->length; i++) {
        row_dest_inner.insert(row_dest_inner.end(), offsets[i + 1] - offsets[i],
                              row_dest[i]);
    }
    return row_dest_inner;
}

template <bool keep_filter_misses>
void mpi_comm_info::set_send_count(
    const std::shared_ptr<uint32_t[]>& hashes,
    const SimdBlockFilterFixed<::hashing::SimpleMixSplit>*& filter,
    const uint8_t* keep_row_bitmask, const uint64_t& n_rows) {
    if (filter == nullptr) {
        if (keep_row_bitmask == nullptr) {
            for (size_t i = 0; i < n_rows; i++) {
                int node = hash_to_rank(hashes[i], n_pes);
                row_dest[i] = node;
                send_count[node]++;
            }
        } else {
            for (size_t i = 0; i < n_rows; i++) {
                if (!GetBit(keep_row_bitmask, i)) {
                    // if null: keep the row on this rank if
                    // keep_filter_misses=true, or drop entirely
                    // otherwise (i.e. leave row_dest[i] as -1).
                    handle_unmatchable_rows<keep_filter_misses>(
                        row_dest, send_count, i, this->myrank);
                } else {
                    int node = hash_to_rank(hashes[i], n_pes);
                    row_dest[i] = node;
                    send_count[node]++;
                }
            }
        }
    } else {
        if (keep_row_bitmask == nullptr) {
            for (size_t i = 0; i < n_rows; i++) {
                const uint32_t& hash = hashes[i];
                if (!filter->Find(static_cast<uint64_t>(hash))) {
                    // if not in filter: keep the row on this rank if
                    // keep_filter_misses=true, or drop entirely
                    // otherwise (i.e. leave row_dest[i] as -1).
                    handle_unmatchable_rows<keep_filter_misses>(
                        row_dest, send_count, i, this->myrank);
                } else {
                    int node = hash_to_rank(hash, n_pes);
                    row_dest[i] = node;
                    send_count[node]++;
                }
            }
        } else {
            for (size_t i = 0; i < n_rows; i++) {
                if (!GetBit(keep_row_bitmask, i)) {
                    // if null: keep the row on this rank if
                    // keep_filter_misses=true, or drop entirely
                    // otherwise (i.e. leave row_dest[i] as -1).
                    handle_unmatchable_rows<keep_filter_misses>(
                        row_dest, send_count, i, this->myrank);
                } else {
                    const uint32_t& hash = hashes[i];
                    if (!filter->Find(static_cast<uint64_t>(hash))) {
                        // if not in filter: keep the row on this rank if
                        // keep_filter_misses=true, or drop
                        // entirely otherwise (i.e. leave row_dest[i] as
                        // -1).
                        handle_unmatchable_rows<keep_filter_misses>(
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
}

mpi_comm_info::mpi_comm_info(
    const std::vector<std::shared_ptr<array_info>>& arrays,
    const std::shared_ptr<uint32_t[]>& hashes, bool is_parallel,
    const SimdBlockFilterFixed<::hashing::SimpleMixSplit>* filter,
    const uint8_t* keep_row_bitmask, bool keep_filter_misses, bool send_only,
    const std::vector<int>& dest_ranks)
    : has_nulls(false), row_dest(arrays[0]->length, -1) {
    tracing::Event ev("mpi_comm_info", is_parallel);
    const uint64_t& n_rows = arrays[0]->length;
    ev.add_attribute("n_rows", n_rows);
    ev.add_attribute("g_using_filter", filter != nullptr);
    ev.add_attribute("g_keep_filter_misses", keep_filter_misses);
    ev.add_attribute("g_using_keep_row_bitmask", keep_row_bitmask != nullptr);
    ev.add_attribute("g_keep_nulls_local",
                     ((keep_row_bitmask != nullptr) && keep_filter_misses));
    ev.add_attribute("g_drop_nulls",
                     ((keep_row_bitmask != nullptr) && !keep_filter_misses));

    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    for (const std::shared_ptr<array_info>& arr_info : arrays) {
        if (arr_info->null_bitmask() != nullptr) {
            has_nulls = true;
            break;
        }
    }

    // init counts
    send_count.resize(n_pes);
    recv_count.resize(n_pes);
    send_disp.resize(n_pes);
    recv_disp.resize(n_pes);

    // Shuffle to specific ranks only
    if (!dest_ranks.empty()) {
        for (size_t i = 0; i < n_rows; i++) {
            int node = dest_ranks[hash_to_rank(hashes[i], dest_ranks.size())];
            row_dest[i] = node;
            send_count[node]++;
        }
    } else {
        // General Shuffle to all ranks
        if (keep_filter_misses) {
            this->set_send_count<true>(hashes, filter, keep_row_bitmask,
                                       n_rows);
        } else {
            this->set_send_count<false>(hashes, filter, keep_row_bitmask,
                                        n_rows);
        }
    }

    // Rows can get filtered if either a bloom-filter or a null filter is
    // provided
    this->filtered = (keep_row_bitmask != nullptr) || (filter != nullptr);
    // NOTE: avoiding alltoall collective for async shuffle cases
    if (!send_only) {
        // get recv count
        CHECK_MPI(
            MPI_Alltoall(send_count.data(), 1, MPI_INT64_T, recv_count.data(),
                         1, MPI_INT64_T, MPI_COMM_WORLD),
            "mpi_comm_info::mpi_comm_info: MPI error on MPI_Alltoall:");
    }

    n_rows_send =
        std::accumulate(send_count.begin(), send_count.end(), int64_t(0));
    n_rows_recv =
        std::accumulate(recv_count.begin(), recv_count.end(), int64_t(0));

    if (ev.is_tracing()) {
        ev.add_attribute("nrows_filtered", n_rows - n_rows_send);
        ev.add_attribute("filtered", filtered);
    }

    // get displacements
    calc_disp(send_disp, send_count);
    calc_disp(recv_disp, recv_count);

    if (has_nulls) {
        send_count_null.reserve(n_pes);
        recv_count_null.reserve(n_pes);
        send_disp_null.resize(n_pes);
        recv_disp_null.resize(n_pes);
        for (int i = 0; i < n_pes; i++) {
            send_count_null.push_back((send_count[i] + 7) >> 3);
            recv_count_null.push_back((recv_count[i] + 7) >> 3);
        }
        calc_disp(send_disp_null, send_count_null);
        calc_disp(recv_disp_null, recv_count_null);
        n_null_bytes = std::accumulate(recv_count_null.begin(),
                                       recv_count_null.end(), size_t(0));
    }
}

mpi_comm_info::mpi_comm_info(const std::shared_ptr<array_info>& parent_arr,
                             const mpi_comm_info& parent_comm_info,
                             bool _has_nulls, bool send_only)
    : myrank(parent_comm_info.myrank),
      n_pes(parent_comm_info.n_pes),
      has_nulls(_has_nulls),
      send_count(n_pes),
      recv_count(n_pes),
      send_disp(n_pes),
      recv_disp(n_pes),
      n_null_bytes(parent_comm_info.n_null_bytes),
      // NOTE: get_inner_array_row_dest implicitly verifies that the parent
      // array is an array-item array.
      row_dest(get_inner_array_row_dest(parent_arr, parent_comm_info.row_dest)),
      filtered(parent_comm_info.filtered) {
    // send counts
    for (int i : row_dest) {
        if (i != -1) {
            send_count[i]++;
        }
    }
    // NOTE: avoiding alltoall collective for async shuffle cases
    if (!send_only) {
        // get recv count
        CHECK_MPI(
            MPI_Alltoall(send_count.data(), 1, MPI_INT64_T, recv_count.data(),
                         1, MPI_INT64_T, MPI_COMM_WORLD),
            "mpi_comm_info::mpi_comm_info: MPI error on MPI_Alltoall:");
    }
    // get displacements
    calc_disp(send_disp, send_count);
    calc_disp(recv_disp, recv_count);

    n_rows_send =
        std::accumulate(send_count.begin(), send_count.end(), int64_t(0));
    n_rows_recv =
        std::accumulate(recv_count.begin(), recv_count.end(), int64_t(0));

    if (has_nulls) {
        send_count_null.reserve(n_pes);
        recv_count_null.reserve(n_pes);
        send_disp_null.resize(n_pes);
        recv_disp_null.resize(n_pes);
        for (int i = 0; i < n_pes; i++) {
            send_count_null.push_back((send_count[i] + 7) >> 3);
            recv_count_null.push_back((recv_count[i] + 7) >> 3);
        }
        calc_disp(send_disp_null, send_count_null);
        calc_disp(recv_disp_null, recv_count_null);
        // inner array's n_null_bytes should always be great than or equal
        // to parent array's since we don't what to resize down the null
        // bytes vector
        n_null_bytes =
            std::max(std::accumulate(recv_count_null.begin(),
                                     recv_count_null.end(), size_t(0)),
                     n_null_bytes);
    }
}

mpi_str_comm_info::mpi_str_comm_info(
    const std::shared_ptr<array_info>& arr_info, const mpi_comm_info& comm_info,
    bool send_only) {
    if (arr_info->arr_type == bodo_array_type::STRING) {
        // initialization
        send_count_sub.resize(comm_info.n_pes);
        recv_count_sub.resize(comm_info.n_pes);
        send_disp_sub.resize(comm_info.n_pes);
        recv_disp_sub.resize(comm_info.n_pes);
        // send counts
        offset_t const* const offsets = (offset_t*)arr_info->data2();
        for (size_t i = 0; i < arr_info->length; i++) {
            if (comm_info.row_dest[i] != -1) {
                send_count_sub[comm_info.row_dest[i]] +=
                    (int64_t)(offsets[i + 1] - offsets[i]);
            }
        }
        // NOTE: avoiding alltoall collective for async shuffle cases
        if (!send_only) {
            // get recv count
            CHECK_MPI(MPI_Alltoall(send_count_sub.data(), 1, MPI_INT64_T,
                                   recv_count_sub.data(), 1, MPI_INT64_T,
                                   MPI_COMM_WORLD),
                      "mpi_str_comm_info::mpi_str_comm_info: MPI error "
                      "on MPI_Alltoall:");
        }
        // get displacements
        calc_disp(send_disp_sub, send_count_sub);
        calc_disp(recv_disp_sub, recv_count_sub);
        n_sub_send = std::accumulate(send_count_sub.begin(),
                                     send_count_sub.end(), int64_t(0));
        n_sub_recv = std::accumulate(recv_count_sub.begin(),
                                     recv_count_sub.end(), int64_t(0));
    }
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

/**
 * @brief Fill output send_length_buff array for array item array, with the
 * lengths calculated from arr_offsets
 *
 * @param[out] send_length_buff Output array of length (not offsets)
 * @param[in] arr_offsets Input array of offsets
 * @param[in] send_disp The sending array of displacements
 * @param[in] n_rows The number of rows
 * @param[in] row_dest Vector keeping track of the destination for each row.
 * @param[in] is_parallel: Used to indicate whether tracing should be parallel
 * or not
 */
static void fill_send_array_inner_array_item(
    uint32_t* send_length_buff, const offset_t* arr_offsets,
    std::vector<int64_t> const& send_disp, const size_t n_rows,
    const std::span<const int> row_dest, bool is_parallel) {
    tracing::Event ev("fill_send_array_inner_array_item", is_parallel);
    std::vector<int64_t> tmp_offset(send_disp);
    for (size_t i = 0; i < n_rows; i++) {
        if (row_dest[i] == -1) {
            continue;
        }
        int64_t& ind = tmp_offset[row_dest[i]];
        send_length_buff[ind++] = arr_offsets[i + 1] - arr_offsets[i];
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
        if (node == -1) {
            continue;
        }
        int64_t& ind = tmp_offset[node];
        const uint32_t str_len = arr_offsets[i + 1] - arr_offsets[i];
        send_length_buff[ind++] = str_len;
        // write data
        int64_t& c_ind = tmp_offset_sub[node];
        memcpy(&send_data_buff[c_ind], &arr_data[arr_offsets[i]], str_len);
        c_ind += str_len;
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

void fill_send_array(std::shared_ptr<array_info> send_arr,
                     std::shared_ptr<array_info> in_arr,
                     const mpi_comm_info& comm_info,
                     const mpi_str_comm_info& str_comm_info, bool is_parallel) {
    tracing::Event ev("fill_send_array", is_parallel);
    const size_t n_rows = (size_t)in_arr->length;
    // dispatch to proper function
    // TODO: general dispatcher
    if (in_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        fill_send_array_null_inner(
            (uint8_t*)send_arr->null_bitmask(),
            (uint8_t*)
                in_arr->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>(),
            comm_info.send_disp_null, comm_info.n_pes, n_rows,
            comm_info.row_dest);
        if (in_arr->dtype == Bodo_CTypes::_BOOL) {
            // Nullable boolean uses 1 bit per boolean so we can reuse the
            // null_bitmap function
            return fill_send_array_null_inner(
                (uint8_t*)send_arr->data1(),
                (uint8_t*)in_arr->data1<bodo_array_type::NULLABLE_INT_BOOL>(),
                comm_info.send_disp_null, comm_info.n_pes, n_rows,
                comm_info.row_dest);
        }
    }
    switch (in_arr->dtype) {
        case Bodo_CTypes::_BOOL:
            return fill_send_array_inner<bool>(
                (bool*)send_arr->data1(), (bool*)in_arr->data1(),
                comm_info.send_disp, n_rows, comm_info.row_dest,
                comm_info.filtered, is_parallel);
        case Bodo_CTypes::INT8:
            return fill_send_array_inner<int8_t>(
                (int8_t*)send_arr->data1(), (int8_t*)in_arr->data1(),
                comm_info.send_disp, n_rows, comm_info.row_dest,
                comm_info.filtered, is_parallel);
        case Bodo_CTypes::UINT8:
            return fill_send_array_inner<uint8_t>(
                (uint8_t*)send_arr->data1(), (uint8_t*)in_arr->data1(),
                comm_info.send_disp, n_rows, comm_info.row_dest,
                comm_info.filtered, is_parallel);
        case Bodo_CTypes::INT16:
            return fill_send_array_inner<int16_t>(
                (int16_t*)send_arr->data1(), (int16_t*)in_arr->data1(),
                comm_info.send_disp, n_rows, comm_info.row_dest,
                comm_info.filtered, is_parallel);
        case Bodo_CTypes::UINT16:
            return fill_send_array_inner<uint16_t>(
                (uint16_t*)send_arr->data1(), (uint16_t*)in_arr->data1(),
                comm_info.send_disp, n_rows, comm_info.row_dest,
                comm_info.filtered, is_parallel);
        case Bodo_CTypes::DATE:
        case Bodo_CTypes::INT32:
            return fill_send_array_inner<int32_t>(
                (int32_t*)send_arr->data1(), (int32_t*)in_arr->data1(),
                comm_info.send_disp, n_rows, comm_info.row_dest,
                comm_info.filtered, is_parallel);
        case Bodo_CTypes::UINT32:
            return fill_send_array_inner<uint32_t>(
                (uint32_t*)send_arr->data1(), (uint32_t*)in_arr->data1(),
                comm_info.send_disp, n_rows, comm_info.row_dest,
                comm_info.filtered, is_parallel);
        case Bodo_CTypes::INT64:
        case Bodo_CTypes::DATETIME:
        case Bodo_CTypes::TIME:
        case Bodo_CTypes::TIMEDELTA:
            return fill_send_array_inner<int64_t>(
                (int64_t*)send_arr->data1(), (int64_t*)in_arr->data1(),
                comm_info.send_disp, n_rows, comm_info.row_dest,
                comm_info.filtered, is_parallel);
        case Bodo_CTypes::UINT64:
            return fill_send_array_inner<uint64_t>(
                (uint64_t*)send_arr->data1(), (uint64_t*)in_arr->data1(),
                comm_info.send_disp, n_rows, comm_info.row_dest,
                comm_info.filtered, is_parallel);
        case Bodo_CTypes::FLOAT32:
            return fill_send_array_inner<float>(
                (float*)send_arr->data1(), (float*)in_arr->data1(),
                comm_info.send_disp, n_rows, comm_info.row_dest,
                comm_info.filtered, is_parallel);
        case Bodo_CTypes::FLOAT64:
            return fill_send_array_inner<double>(
                (double*)send_arr->data1(), (double*)in_arr->data1(),
                comm_info.send_disp, n_rows, comm_info.row_dest,
                comm_info.filtered, is_parallel);
        case Bodo_CTypes::DECIMAL:
            return fill_send_array_inner_decimal(
                (uint8_t*)send_arr->data1(), (uint8_t*)in_arr->data1(),
                comm_info.send_disp, n_rows, comm_info.row_dest, is_parallel);
        default:
            if (in_arr->arr_type == bodo_array_type::STRING) {
                fill_send_array_string_inner(
                    /// XXX casting data2 offset_t to uint32
                    (char*)send_arr->data1(), (uint32_t*)send_arr->data2(),
                    (char*)in_arr->data1(), (offset_t*)in_arr->data2(),
                    comm_info.send_disp, str_comm_info.send_disp_sub, n_rows,
                    comm_info.row_dest, is_parallel);
                fill_send_array_null_inner(
                    (uint8_t*)send_arr->null_bitmask(),
                    (uint8_t*)in_arr->null_bitmask(), comm_info.send_disp_null,
                    comm_info.n_pes, n_rows, comm_info.row_dest);
            } else if (in_arr->arr_type == bodo_array_type::ARRAY_ITEM) {
                // Fill the offset buffer
                fill_send_array_inner_array_item(
                    (uint32_t*)send_arr->data1(), (offset_t*)in_arr->data1(),
                    comm_info.send_disp, n_rows, comm_info.row_dest,
                    is_parallel);
                // Fill the null bitmask
                fill_send_array_null_inner(
                    (uint8_t*)send_arr->null_bitmask(),
                    (uint8_t*)in_arr->null_bitmask(), comm_info.send_disp_null,
                    comm_info.n_pes, n_rows, comm_info.row_dest);
            } else if (in_arr->arr_type == bodo_array_type::STRUCT) {
                // Fill the null bitmask
                fill_send_array_null_inner(
                    (uint8_t*)send_arr->null_bitmask(),
                    (uint8_t*)in_arr->null_bitmask(), comm_info.send_disp_null,
                    comm_info.n_pes, n_rows, comm_info.row_dest);

            } else if (in_arr->arr_type == bodo_array_type::MAP) {
                fill_send_array(send_arr->child_arrays[0],
                                in_arr->child_arrays[0], comm_info,
                                str_comm_info, is_parallel);
            } else if (in_arr->arr_type == bodo_array_type::TIMESTAMPTZ) {
                // Send Timestamp section
                fill_send_array_inner<int64_t>(
                    (int64_t*)send_arr->data1(), (int64_t*)in_arr->data1(),
                    comm_info.send_disp, n_rows, comm_info.row_dest,
                    comm_info.filtered, is_parallel);
                // Send offsets
                fill_send_array_inner<int16_t>(
                    (int16_t*)send_arr->data2(), (int16_t*)in_arr->data2(),
                    comm_info.send_disp, n_rows, comm_info.row_dest,
                    comm_info.filtered, is_parallel);
                // Send nulls
                fill_send_array_null_inner(
                    (uint8_t*)send_arr->null_bitmask(),
                    (uint8_t*)in_arr->null_bitmask(), comm_info.send_disp_null,
                    comm_info.n_pes, n_rows, comm_info.row_dest);
            } else if (in_arr->arr_type != bodo_array_type::DICT) {
                throw std::runtime_error(
                    "Invalid data type for fill_send_array: " +
                    GetDtype_as_string(in_arr->dtype));
            }
    }
}

/* Internal function. Convert counts to displacements
 */
#if OFFSET_BITWIDTH == 32
void convert_len_arr_to_offset32(uint32_t* offsets, size_t const& num_strs) {
    uint32_t curr_offset = 0;
    for (size_t i = 0; i < num_strs; i++) {
        uint32_t val = offsets[i];
        offsets[i] = curr_offset;
        curr_offset += val;
    }
    offsets[num_strs] = curr_offset;
}
#endif

void convert_len_arr_to_offset(uint32_t* lens, offset_t* offsets,
                               size_t num_strs) {
    static_assert(sizeof(offset_t) * 8 == OFFSET_BITWIDTH,
                  "offset_t must be 8 bytes");
    offset_t curr_offset = 0;
    for (size_t i = 0; i < num_strs; i++) {
        uint32_t length = lens[i];
        offsets[i] = curr_offset;
        curr_offset += length;
    }
    offsets[num_strs] = curr_offset;
}

/**
 * @brief Shuffle in_arr based on comm_info
 *
 * @param in_arr Input array
 * @param comm_info Information needed for shuffling
 * @param tmp_null_bytes Temperary buffer used to store intermediate result for
 * shuffling null bitmask
 * @param is_parallel Used to indicate whether tracing should be parallel or not
 * @return The shuffled array
 */
std::shared_ptr<array_info> shuffle_array(std::shared_ptr<array_info> in_arr,
                                          const mpi_comm_info& comm_info,
                                          bodo::vector<uint8_t>& tmp_null_bytes,
                                          bool is_parallel) {
    tracing::Event ev("shuffle_array", is_parallel);

    mpi_str_comm_info str_comm_info(in_arr, comm_info);

    // NOTE: we pass extra_null_bytes to account for padding in null buffer
    // for process boundaries (bits of two different processes cannot be packed
    // in the same byte).
    std::shared_ptr<array_info> send_arr = alloc_array_top_level(
        comm_info.n_rows_send, str_comm_info.n_sub_send, 0, in_arr->arr_type,
        in_arr->dtype, -1, 2 * comm_info.n_pes, in_arr->num_categories);
    fill_send_array(send_arr, in_arr, comm_info, str_comm_info, is_parallel);

    std::shared_ptr<array_info> out_arr = alloc_array_top_level(
        comm_info.n_rows_recv, str_comm_info.n_sub_recv, 0, in_arr->arr_type,
        in_arr->dtype, -1, 0, in_arr->num_categories);
    out_arr->precision = in_arr->precision;
    out_arr->scale = in_arr->scale;

    switch (send_arr->arr_type) {
        case bodo_array_type::STRING: {
            // string lengths
            MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT32);
#if OFFSET_BITWIDTH == 32
            bodo_alltoallv(send_arr->data2(), comm_info.send_count,
                           comm_info.send_disp, mpi_typ, out_arr->data2(),
                           comm_info.recv_count, comm_info.recv_disp, mpi_typ,
                           MPI_COMM_WORLD);
            convert_len_arr_to_offset32((uint32_t*)out_arr->data2(),
                                        (size_t)out_arr->length);
#else
            std::vector<uint32_t> lens(out_arr->length);
            bodo_alltoallv(send_arr->data2(), comm_info.send_count,
                           comm_info.send_disp, mpi_typ, lens.data(),
                           comm_info.recv_count, comm_info.recv_disp, mpi_typ,
                           MPI_COMM_WORLD);
            convert_len_arr_to_offset(lens.data(), (offset_t*)out_arr->data2(),
                                      (size_t)out_arr->length);
#endif
            // string data
            mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
            bodo_alltoallv(send_arr->data1(), str_comm_info.send_count_sub,
                           str_comm_info.send_disp_sub, mpi_typ,
                           out_arr->data1(), str_comm_info.recv_count_sub,
                           str_comm_info.recv_disp_sub, mpi_typ,
                           MPI_COMM_WORLD);
            // nulls
            mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
            bodo_alltoallv(send_arr->null_bitmask(), comm_info.send_count_null,
                           comm_info.send_disp_null, mpi_typ,
                           tmp_null_bytes.data(), comm_info.recv_count_null,
                           comm_info.recv_disp_null, mpi_typ, MPI_COMM_WORLD);
            copy_gathered_null_bytes((uint8_t*)out_arr->null_bitmask(),
                                     tmp_null_bytes, comm_info.recv_count_null,
                                     comm_info.recv_count);
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
                bodo_alltoallv(send_arr->data1(), comm_info.send_count_null,
                               comm_info.send_disp_null, mpi_typ,
                               tmp_null_bytes.data(), comm_info.recv_count_null,
                               comm_info.recv_disp_null, mpi_typ,
                               MPI_COMM_WORLD);
                copy_gathered_null_bytes(
                    (uint8_t*)out_arr->data1(), tmp_null_bytes,
                    comm_info.recv_count_null, comm_info.recv_count);
            } else {
                bodo_alltoallv(send_arr->data1(), comm_info.send_count,
                               comm_info.send_disp, mpi_typ, out_arr->data1(),
                               comm_info.recv_count, comm_info.recv_disp,
                               mpi_typ, MPI_COMM_WORLD);
            }
            // nulls
            mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
            bodo_alltoallv(send_arr->null_bitmask(), comm_info.send_count_null,
                           comm_info.send_disp_null, mpi_typ,
                           tmp_null_bytes.data(), comm_info.recv_count_null,
                           comm_info.recv_disp_null, mpi_typ, MPI_COMM_WORLD);
            copy_gathered_null_bytes((uint8_t*)out_arr->null_bitmask(),
                                     tmp_null_bytes, comm_info.recv_count_null,
                                     comm_info.recv_count);
            break;
        }
        case bodo_array_type::NUMPY:
        case bodo_array_type::CATEGORICAL: {
            // data
            MPI_Datatype mpi_typ = get_MPI_typ(send_arr->dtype);
            bodo_alltoallv(send_arr->data1(), comm_info.send_count,
                           comm_info.send_disp, mpi_typ, out_arr->data1(),
                           comm_info.recv_count, comm_info.recv_disp, mpi_typ,
                           MPI_COMM_WORLD);
            break;
        }
        case bodo_array_type::DICT: {
            if (!in_arr->child_arrays[0]->is_globally_replicated) {
                make_dictionary_global_and_unique(in_arr, is_parallel);
            }
            out_arr = create_dict_string_array(
                in_arr->child_arrays[0],
                shuffle_array(in_arr->child_arrays[1], comm_info,
                              tmp_null_bytes, is_parallel));
            break;
        }
        case bodo_array_type::ARRAY_ITEM: {
            // offsets
            // NOTE: While the offset could be 64 or 32 bit integer, length is
            // always 32 bit as we expect length to be smaller in general.
            MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT32);
#if OFFSET_BITWIDTH == 32
            bodo_alltoallv(send_arr->data1(), comm_info.send_count,
                           comm_info.send_disp, mpi_typ, out_arr->data1(),
                           comm_info.recv_count, comm_info.recv_disp, mpi_typ,
                           MPI_COMM_WORLD);
            convert_len_arr_to_offset32((uint32_t*)out_arr->data1(),
                                        (size_t)out_arr->length);
#else
            std::vector<uint32_t> lens(out_arr->length);
            bodo_alltoallv(send_arr->data1(), comm_info.send_count,
                           comm_info.send_disp, mpi_typ, lens.data(),
                           comm_info.recv_count, comm_info.recv_disp, mpi_typ,
                           MPI_COMM_WORLD);
            convert_len_arr_to_offset(lens.data(), (offset_t*)out_arr->data1(),
                                      (size_t)out_arr->length);
#endif
            // nulls
            mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
            bodo_alltoallv(send_arr->null_bitmask(), comm_info.send_count_null,
                           comm_info.send_disp_null, mpi_typ,
                           tmp_null_bytes.data(), comm_info.recv_count_null,
                           comm_info.recv_disp_null, mpi_typ, MPI_COMM_WORLD);
            copy_gathered_null_bytes((uint8_t*)out_arr->null_bitmask(),
                                     tmp_null_bytes, comm_info.recv_count_null,
                                     comm_info.recv_count);
            // inner array
            mpi_comm_info comm_info_inner(
                in_arr, comm_info,
                in_arr->child_arrays[0]->null_bitmask() != nullptr);
            tmp_null_bytes.resize(comm_info_inner.n_null_bytes);
            out_arr->child_arrays[0] =
                shuffle_array(in_arr->child_arrays[0], comm_info_inner,
                              tmp_null_bytes, is_parallel);
            tmp_null_bytes.resize(comm_info.n_null_bytes);
            break;
        }
        case bodo_array_type::STRUCT: {
            // nulls
            MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
            bodo_alltoallv(send_arr->null_bitmask(), comm_info.send_count_null,
                           comm_info.send_disp_null, mpi_typ,
                           tmp_null_bytes.data(), comm_info.recv_count_null,
                           comm_info.recv_disp_null, mpi_typ, MPI_COMM_WORLD);
            copy_gathered_null_bytes((uint8_t*)out_arr->null_bitmask(),
                                     tmp_null_bytes, comm_info.recv_count_null,
                                     comm_info.recv_count);
            // child arrays
            for (size_t i = 0; i < in_arr->child_arrays.size(); i++) {
                out_arr->child_arrays.push_back(
                    shuffle_array(in_arr->child_arrays[i], comm_info,
                                  tmp_null_bytes, is_parallel));
            }
            break;
        }
        case bodo_array_type::MAP: {
            out_arr->child_arrays[0] =
                shuffle_array(in_arr->child_arrays[0], comm_info,
                              tmp_null_bytes, is_parallel);
            break;
        }
        case bodo_array_type::TIMESTAMPTZ: {
            // timestamp_data
            MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::TIMESTAMPTZ);
            bodo_alltoallv(send_arr->data1(), comm_info.send_count,
                           comm_info.send_disp, mpi_typ, out_arr->data1(),
                           comm_info.recv_count, comm_info.recv_disp, mpi_typ,
                           MPI_COMM_WORLD);
            // offset data
            mpi_typ = get_MPI_typ(Bodo_CTypes::INT16);
            bodo_alltoallv(send_arr->data2(), comm_info.send_count,
                           comm_info.send_disp, mpi_typ, out_arr->data2(),
                           comm_info.recv_count, comm_info.recv_disp, mpi_typ,
                           MPI_COMM_WORLD);
            // nulls
            mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
            bodo_alltoallv(send_arr->null_bitmask(), comm_info.send_count_null,
                           comm_info.send_disp_null, mpi_typ,
                           tmp_null_bytes.data(), comm_info.recv_count_null,
                           comm_info.recv_disp_null, mpi_typ, MPI_COMM_WORLD);
            copy_gathered_null_bytes((uint8_t*)out_arr->null_bitmask(),
                                     tmp_null_bytes, comm_info.recv_count_null,
                                     comm_info.recv_count);
            break;
        }
        default:
            throw std::runtime_error(
                "Unsupported array type for shuffle_array: " +
                GetArrType_as_string(in_arr->arr_type));
    }

    return out_arr;
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
        if (node == -1) {
            continue;
        }
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
        if (node == -1) {
            continue;
        }
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
            if (node == -1) {
                continue;
            }
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
        if (node == -1) {
            continue;
        }
        int64_t n_char = string_array->value_length(i_row);
        send_count_char[node] += n_char;
    }
    CHECK_MPI(
        MPI_Alltoall(send_count_char.data(), 1, MPI_INT64_T,
                     recv_count_char.data(), 1, MPI_INT64_T, MPI_COMM_WORLD),
        "shuffle_string_buffer: MPI error on MPI_Alltoall:");
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
        if (node == -1) {
            continue;
        }
        std::string_view e_str = string_array->GetView(i_row);
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
  A prerequisite for the function to run correctly is that the comm_info is
  set correctly. It determines which sizes go to each processor. It performs
  a determination of the the sending and receiving arrays.
  ---
  1) The first step is to accumulate the sizes from each processors.
  2) Then for each row a number of operations are done:
  2a) First set up the send_arr by aligning the data from the input columns
  accordingly. 2b) Second do the shuffling of data between all processors.
 */
std::shared_ptr<table_info> shuffle_table_kernel(
    std::shared_ptr<table_info> in_table,
    const std::shared_ptr<uint32_t[]>& hashes, const mpi_comm_info& comm_info,
    bool is_parallel) {
    tracing::Event ev("shuffle_table_kernel", is_parallel);
    if (ev.is_tracing()) {
        ev.add_attribute("table_nrows_before", in_table->nrows());
        ev.add_attribute("filtered", comm_info.filtered);
        ev.add_attribute("g_table_nbytes", table_global_memory_size(in_table));
    }
    // fill send buffer and send
    std::vector<std::shared_ptr<array_info>> out_arrs;
    bodo::vector<uint8_t> tmp_null_bytes(comm_info.n_null_bytes);
    for (uint64_t i = 0; i < in_table->ncols(); i++) {
        out_arrs.push_back(shuffle_array(in_table->columns[i], comm_info,
                                         tmp_null_bytes, is_parallel));
        // Release reference (and memory) early if possible.
        reset_col_if_last_table_ref(in_table, i);
    }
    ev.add_attribute("table_nrows_after", out_arrs[0]->length);
    return std::make_shared<table_info>(out_arrs);
}
/**
 * @brief Perform a reverse shuffle for data
 *
 * @param[in] in_data_arr: The input data array to reverse-shuffle.
 * @param[out] out_data_arr: The output data array to fill.
 * @param[in] out_data_len: size of out array
 * @param[in] siztype: sizeof(in_data_arr)
 * @param[in] mpi_typ: MPI_Datatype
 * @param[in] comm_info The communication information from the original shuffle.
 */
void reverse_shuffle_data(char* in_data_arr, char* out_data_arr,
                          uint64_t out_data_len, uint64_t siztype,
                          MPI_Datatype mpi_typ,
                          mpi_comm_info const& comm_info) {
    bodo::vector<char> tmp_recv(out_data_len * siztype);
    bodo_alltoallv(in_data_arr, comm_info.recv_count, comm_info.recv_disp,
                   mpi_typ, tmp_recv.data(), comm_info.send_count,
                   comm_info.send_disp, mpi_typ, MPI_COMM_WORLD);
    std::vector<int64_t> tmp_offset(comm_info.send_disp);
    const bodo::vector<int>& row_dest = comm_info.row_dest;
    for (size_t i = 0; i < out_data_len; i++) {
        int64_t& ind = tmp_offset[row_dest[i]];
        memcpy(out_data_arr + siztype * i, tmp_recv.data() + siztype * ind++,
               siztype);
    }
}

// Shuffle is basically to send data to other processes for operations like
// drop_duplicates, etc. Usually though you want to know the indices in the
// original DF (usually the first occurring ones). Reverse shuffle is
// basically transferring the shuffled data back to the original DF. Useful
// for things like cumulative operations, array_isin, etc.

void reverse_shuffle_preallocated_data_array(
    std::shared_ptr<array_info> in_arr, std::shared_ptr<array_info> out_arr,
    mpi_comm_info const& comm_info) {
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
        const bodo::vector<int>& row_dest = comm_info.row_dest;
        for (size_t i_row = 0; i_row < out_arr->length; i_row++) {
            size_t node = row_dest[i_row];
            uint8_t* out_bitmap = &(temp_recv.data())[send_disp_bytes[node]];
            bool bit = GetBit(out_bitmap, tmp_offset[node]);
            SetBitTo(data1_o, i_row, bit);
            tmp_offset[node]++;
        }

    } else if (in_arr->arr_type == bodo_array_type::TIMESTAMPTZ) {
        // data1
        uint64_t siztype = numpy_item_size[in_arr->dtype];
        MPI_Datatype mpi_typ = get_MPI_typ(in_arr->dtype);
        char* data1_i = in_arr->data1();
        char* data1_o = out_arr->data1();
        reverse_shuffle_data(data1_i, data1_o, out_arr->length, siztype,
                             mpi_typ, comm_info);
        // data2
        mpi_typ = get_MPI_typ(Bodo_CTypes::INT16);
        char* data2_i = in_arr->data2();
        char* data2_o = out_arr->data2();
        reverse_shuffle_data(data2_i, data2_o, out_arr->length, sizeof(int16_t),
                             mpi_typ, comm_info);
    } else {
        const uint64_t siztype = numpy_item_size[in_arr->dtype];
        MPI_Datatype mpi_typ = get_MPI_typ(in_arr->dtype);
        char* data1_i = in_arr->data1();
        char* data1_o = out_arr->data1();
        reverse_shuffle_data(data1_i, data1_o, out_arr->length, siztype,
                             mpi_typ, comm_info);
    }
}

std::shared_ptr<array_info> reverse_shuffle_data_array(
    std::shared_ptr<array_info> in_arr, mpi_comm_info const& comm_info) {
    tracing::Event ev("reverse_shuffle_data_array");
    size_t n_rows_ret = std::accumulate(comm_info.send_count.begin(),
                                        comm_info.send_count.end(), size_t(0));
    std::shared_ptr<array_info> out_arr =
        alloc_array_top_level(n_rows_ret, 0, 0, in_arr->arr_type, in_arr->dtype,
                              -1, 0, in_arr->num_categories);
    reverse_shuffle_preallocated_data_array(in_arr, out_arr, comm_info);
    return out_arr;
}

std::shared_ptr<array_info> reverse_shuffle_string_array(
    std::shared_ptr<array_info> in_arr, mpi_comm_info const& comm_info) {
    tracing::Event ev("reverse_shuffle_string_array");
    // 1: computing the recv_count_sub and related
    offset_t* in_offset = (offset_t*)in_arr->data2();
    int n_pes = comm_info.n_pes;
    std::vector<int64_t> recv_count_sub(n_pes),
        recv_disp_sub(n_pes);  // we continue using here the recv/send
    for (int i = 0; i < n_pes; i++) {
        recv_count_sub[i] =
            in_offset[comm_info.recv_disp[i] + comm_info.recv_count[i]] -
            in_offset[comm_info.recv_disp[i]];
    }
    std::vector<int64_t> send_count_sub(n_pes), send_disp_sub(n_pes);

    CHECK_MPI(
        MPI_Alltoall(recv_count_sub.data(), 1, MPI_INT64_T,
                     send_count_sub.data(), 1, MPI_INT64_T, MPI_COMM_WORLD),
        "reverse_shuffle_string_array: MPI error on MPI_Alltoall:");

    calc_disp(send_disp_sub, send_count_sub);
    calc_disp(recv_disp_sub, recv_count_sub);
    // 2: allocating the array
    int64_t n_count_sub = send_disp_sub[n_pes - 1] + send_count_sub[n_pes - 1];
    int64_t n_rows_ret = std::accumulate(
        comm_info.send_count.begin(), comm_info.send_count.end(), int64_t(0));
    std::shared_ptr<array_info> out_arr =
        alloc_array_top_level(n_rows_ret, n_count_sub, 0, in_arr->arr_type,
                              in_arr->dtype, -1, 0, in_arr->num_categories);
    int64_t in_len = in_arr->length;
    int64_t out_len = out_arr->length;
    // 3: the offsets
    bodo::vector<uint32_t> list_len_send(in_len);
    offset_t* out_offset = (offset_t*)out_arr->data2();
    for (int64_t i = 0; i < in_len; i++) {
        list_len_send[i] = in_offset[i + 1] - in_offset[i];
    }
    MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT32);
    bodo::vector<uint32_t> list_len_recv(out_len);
    bodo_alltoallv(list_len_send.data(), comm_info.recv_count,
                   comm_info.recv_disp, mpi_typ, list_len_recv.data(),
                   comm_info.send_count, comm_info.send_disp, mpi_typ,
                   MPI_COMM_WORLD);
#if OFFSET_BITWIDTH == 32
    fill_recv_data_inner(list_len_recv.data(), out_offset, comm_info.row_dest,
                         comm_info.send_disp, out_len);
    convert_len_arr_to_offset32(out_offset, out_len);
#else
    bodo::vector<uint32_t> out_lens(out_arr->length);
    fill_recv_data_inner(list_len_recv.data(), out_lens.data(),
                         comm_info.row_dest, comm_info.send_disp, out_len);
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
    const bodo::vector<int>& row_dest = comm_info.row_dest;
    char* out_arr_data1 = out_arr->data1();
    for (int64_t i = 0; i < out_len; i++) {
        size_t node = row_dest[i];
        offset_t str_len = out_offset[i + 1] - out_offset[i];
        int64_t c_ind = tmp_offset_sub[node];
        char* out_ptr = out_arr_data1 + out_offset[i];
        char* in_ptr = (char*)tmp_recv.data() + c_ind;
        memcpy(out_ptr, in_ptr, str_len);
        tmp_offset_sub[node] += str_len;
    }
    return out_arr;
}

void reverse_shuffle_null_bitmap_array(std::shared_ptr<array_info> in_arr,
                                       std::shared_ptr<array_info> out_arr,
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
    const bodo::vector<int>& row_dest = comm_info.row_dest;
    for (size_t i_row = 0; i_row < out_arr->length; i_row++) {
        size_t node = row_dest[i_row];
        uint8_t* out_bitmap = &(mask_recv.data())[send_disp_null[node]];
        bool bit = GetBit(out_bitmap, tmp_offset[node]);
        SetBitTo(null_bitmask_out, i_row, bit);
        tmp_offset[node]++;
    }
}

/**
 * @brief Reverse shuffle offsets of array(item) array
 *
 * @param in_offset input array offsets (shuffled array)
 * @param out_offset output array offsets (reverse shuffled array)
 * @param in_len input array length
 * @param out_len output array length
 * @param comm_info mpi comm info of shuffle
 */
void reverse_shuffle_offsets(offset_t* in_offset, offset_t* out_offset,
                             int64_t in_len, int64_t out_len,
                             mpi_comm_info const& comm_info) {
    bodo::vector<uint32_t> send_arr_lens(in_len);
    for (int64_t i = 0; i < in_len; i++) {
        send_arr_lens[i] = in_offset[i + 1] - in_offset[i];
    }
    MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT32);
    bodo::vector<uint32_t> recv_arr_lens(out_len);
    bodo_alltoallv(send_arr_lens.data(), comm_info.recv_count,
                   comm_info.recv_disp, mpi_typ, recv_arr_lens.data(),
                   comm_info.send_count, comm_info.send_disp, mpi_typ,
                   MPI_COMM_WORLD);
    static_assert(OFFSET_BITWIDTH == 64);
    bodo::vector<uint32_t> out_lens(out_len);
    fill_recv_data_inner(recv_arr_lens.data(), out_lens.data(),
                         comm_info.row_dest, comm_info.send_disp, out_len);
    convert_len_arr_to_offset(out_lens.data(), out_offset, out_len);
}

/**
 * @brief reverse shuffle an array(item) array
 *
 * @param in_arr previously shuffled array(item) array
 * @param comm_info mpi comm info of shuffle
 * @return std::shared_ptr<array_info> array before shuffle
 */
std::shared_ptr<array_info> reverse_shuffle_array_item(
    std::shared_ptr<array_info> in_arr, mpi_comm_info const& comm_info) {
    // allocate output array
    int64_t out_len = std::accumulate(comm_info.send_count.begin(),
                                      comm_info.send_count.end(), int64_t(0));
    std::shared_ptr<array_info> out_arr = alloc_array_item(out_len, nullptr);
    int64_t in_len = in_arr->length;

    // reverse shuffle offsets
    offset_t* in_offset = (offset_t*)in_arr->data1();
    offset_t* out_offset = (offset_t*)out_arr->data1();
    reverse_shuffle_offsets(in_offset, out_offset, in_len, out_len, comm_info);

    // reverse shuffle child array
    mpi_comm_info comm_info_inner(
        out_arr, comm_info, in_arr->child_arrays[0]->null_bitmask() != nullptr);
    out_arr->child_arrays[0] =
        reverse_shuffle_array(in_arr->child_arrays[0], comm_info_inner);
    return out_arr;
}

/**
 * @brief Reverse shuffle a previously shuffled array
 *
 * @param in_arr input array (previously shuffled)
 * @param comm_info comm_info of the original shuffle
 * @return std::shared_ptr<array_info> the original shuffled array
 */
std::shared_ptr<array_info> reverse_shuffle_array(
    std::shared_ptr<array_info> in_arr, mpi_comm_info const& comm_info) {
    // keep input array since in_arr is changed to indices for DICT below
    std::shared_ptr<array_info> in_arr_original = in_arr;
    bodo_array_type::arr_type_enum arr_type = in_arr->arr_type;
    std::shared_ptr<array_info> out_arr = nullptr;
    if (arr_type == bodo_array_type::ARRAY_ITEM) {
        out_arr = reverse_shuffle_array_item(in_arr, comm_info);
    } else if (in_arr->arr_type == bodo_array_type::STRUCT) {
        // reverse shuffle child arrays
        std::vector<std::shared_ptr<array_info>> child_arrays;
        for (const std::shared_ptr<array_info>& child_arr :
             in_arr->child_arrays) {
            child_arrays.push_back(reverse_shuffle_array(child_arr, comm_info));
        }
        int64_t length = child_arrays.size() == 0 ? 0 : child_arrays[0]->length;
        out_arr = alloc_struct(length, std::move(child_arrays));
    } else if (arr_type == bodo_array_type::MAP) {
        std::shared_ptr<array_info> child_arr =
            reverse_shuffle_array(in_arr->child_arrays[0], comm_info);
        out_arr = alloc_map(child_arr->length, child_arr);
    } else if (arr_type == bodo_array_type::DICT) {
        if (!in_arr->child_arrays[0]->is_globally_replicated) {
            make_dictionary_global_and_unique(in_arr, true);
        }
        // in_arr <- indices array, to simplify code below
        in_arr = in_arr->child_arrays[1];
        arr_type = in_arr->arr_type;
    }
    if (arr_type == bodo_array_type::NUMPY ||
        arr_type == bodo_array_type::CATEGORICAL ||
        arr_type == bodo_array_type::TIMESTAMPTZ ||
        arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        out_arr = reverse_shuffle_data_array(in_arr, comm_info);
    }
    if (arr_type == bodo_array_type::STRING) {
        out_arr = reverse_shuffle_string_array(in_arr, comm_info);
    }
    if (arr_type == bodo_array_type::STRING ||
        arr_type == bodo_array_type::NULLABLE_INT_BOOL ||
        arr_type == bodo_array_type::ARRAY_ITEM ||
        arr_type == bodo_array_type::TIMESTAMPTZ ||
        arr_type == bodo_array_type::STRUCT) {
        reverse_shuffle_null_bitmap_array(in_arr, out_arr, comm_info);
    }
    if (in_arr_original->arr_type == bodo_array_type::DICT) {
        in_arr = in_arr_original;
        std::shared_ptr<array_info> out_dict_arr =
            create_dict_string_array(in_arr->child_arrays[0], out_arr);
        out_arr = out_dict_arr;
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
    @param comm_info : The comm_info (computed from the original table)
    @return the reshuffled table
 */
std::shared_ptr<table_info> reverse_shuffle_table_kernel(
    std::shared_ptr<table_info> in_table, mpi_comm_info const& comm_info) {
    tracing::Event ev("reverse_shuffle_table_kernel");
    uint64_t n_cols = in_table->ncols();
    std::vector<std::shared_ptr<array_info>> out_arrs;
    for (uint64_t i = 0; i < n_cols; i++) {
        out_arrs.push_back(
            reverse_shuffle_array(in_table->columns[i], comm_info));
        reset_col_if_last_table_ref(in_table, i);
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
        return nullptr;
    }

    const bool delete_hashes = bool(hashes);

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
            // is_globally_replicated attribute of underlying string array.
            // We need dictionaries to be global and unique for hashing.
            make_dictionary_global_and_unique(a, is_parallel);
        }
    }

    // computing the hash data structure
    if (!hashes) {
        hashes =
            hash_keys_table(in_table, n_keys, SEED_HASH_PARTITION, is_parallel);
    }
    std::shared_ptr<mpi_comm_info> comm_info =
        std::make_shared<mpi_comm_info>(in_table->columns, hashes, is_parallel);

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
        return nullptr;
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
    std::shared_ptr<table_info> revshuf_table = reverse_shuffle_table_kernel(
        std::shared_ptr<table_info>(in_table), *sh_info->comm_info);

    return new table_info(*revshuf_table);
}

std::shared_ptr<table_info> coherent_shuffle_table(
    std::shared_ptr<table_info> in_table, std::shared_ptr<table_info> ref_table,
    int64_t n_keys, std::shared_ptr<uint32_t[]> hashes,
    SimdBlockFilterFixed<::hashing::SimpleMixSplit>* filter,
    const uint8_t* keep_row_bitmask, const bool keep_filter_misses) {
    tracing::Event ev("coherent_shuffle_table", true);
    // error checking
    if (in_table->ncols() <= 0 || n_keys <= 0) {
        Bodo_PyErr_SetString(PyExc_RuntimeError, "Invalid input shuffle table");
        return nullptr;
    }
    const bool delete_hashes = bool(hashes);

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
    mpi_comm_info comm_info(in_table->columns, hashes, true, filter,
                            keep_row_bitmask, keep_filter_misses);
    std::shared_ptr<table_info> table =
        shuffle_table_kernel(std::move(in_table), hashes, comm_info, true);
    if (delete_hashes) {
        hashes.reset();
    }
    return table;
}

std::shared_ptr<array_info> broadcast_array(
    std::shared_ptr<array_info> ref_arr, std::shared_ptr<array_info> in_arr,
    std::shared_ptr<std::vector<int>> comm_ranks, bool is_parallel, int root,
    int myrank, MPI_Comm* comm_ptr) {
    MPI_Comm comm = MPI_COMM_WORLD;
    bool is_sender = (myrank == root);
    // Use provided comm pointer if available (nullptr means not provided)
    if (comm_ptr) {
        comm = *comm_ptr;
        is_sender = (root == MPI_ROOT);
    }
    int64_t arr_bcast[6];
    if (is_sender) {
        arr_bcast[0] = in_arr->length;
        arr_bcast[1] = in_arr->dtype;
        arr_bcast[2] = in_arr->arr_type;
        arr_bcast[3] = in_arr->n_sub_elems();
        arr_bcast[4] = in_arr->num_categories;
        arr_bcast[5] = (int64_t)in_arr->precision;
    }
    CHECK_MPI(MPI_Bcast(arr_bcast, 6, MPI_LONG_LONG_INT, root, comm),
              "broadcast_array: MPI error on MPI_Bcast:");
    int64_t n_rows = arr_bcast[0];
    Bodo_CTypes::CTypeEnum dtype = Bodo_CTypes::CTypeEnum(arr_bcast[1]);
    bodo_array_type::arr_type_enum arr_type =
        bodo_array_type::arr_type_enum(arr_bcast[2]);
    int64_t n_sub_elems = arr_bcast[3];
    int64_t num_categories = arr_bcast[4];
    int32_t precision = (int32_t)arr_bcast[5];

    // Create new communicator if target ranks are specified by user
    if (comm_ranks && comm_ranks->size() > 0) {
        c_comm_create(comm_ranks->data(), comm_ranks->size(), &comm);
        if (comm == MPI_COMM_NULL) {
            return alloc_array_top_level(0, 0, 0, arr_type, dtype);
        }
    }

    std::shared_ptr<array_info> out_arr;
    if (arr_type == bodo_array_type::NUMPY ||
        arr_type == bodo_array_type::CATEGORICAL ||
        arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        MPI_Datatype mpi_typ = get_MPI_typ(dtype);
        if (is_sender) {
            out_arr = in_arr;
        } else {
            out_arr = alloc_array_top_level(n_rows, -1, -1, arr_type, dtype);
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
        CHECK_MPI(MPI_Bcast(out_arr->data1(), bcast_size, mpi_typ, root, comm),
                  "broadcast_array: MPI error on MPI_Bcast:");
    } else if (arr_type == bodo_array_type::INTERVAL) {
        MPI_Datatype mpi_typ = get_MPI_typ(dtype);
        if (is_sender) {
            out_arr = in_arr;
        } else {
            out_arr = alloc_array_top_level(n_rows, -1, -1, arr_type, dtype);
        }
        CHECK_MPI(MPI_Bcast(out_arr->data1(), n_rows, mpi_typ, root, comm),
                  "broadcast_array: MPI error on MPI_Bcast:");
        CHECK_MPI(MPI_Bcast(out_arr->data2(), n_rows, mpi_typ, root, comm),
                  "broadcast_array: MPI error on MPI_Bcast:");
    } else if (arr_type == bodo_array_type::TIMESTAMPTZ) {
        MPI_Datatype utc_mpi_typ = get_MPI_typ(dtype);
        MPI_Datatype offset_mpi_typ = get_MPI_typ(Bodo_CTypes::INT16);
        if (is_sender) {
            out_arr = in_arr;
        } else {
            out_arr = alloc_array_top_level(n_rows, -1, -1, arr_type, dtype);
        }
        CHECK_MPI(MPI_Bcast(out_arr->data1(), n_rows, utc_mpi_typ, root, comm),
                  "broadcast_array: MPI error on MPI_Bcast:");
        CHECK_MPI(
            MPI_Bcast(out_arr->data2(), n_rows, offset_mpi_typ, root, comm),
            "broadcast_array: MPI error on MPI_Bcast:");
    } else if (arr_type == bodo_array_type::STRING) {
        MPI_Datatype mpi_typ_offset = get_MPI_typ(Bodo_CType_offset);
        MPI_Datatype mpi_typ8 = get_MPI_typ(Bodo_CTypes::UINT8);
        if (is_sender) {
            out_arr = in_arr;
        } else {
            out_arr = alloc_array_top_level(n_rows, n_sub_elems, -1, arr_type,
                                            dtype, -1, 0, num_categories);
        }
        CHECK_MPI(
            MPI_Bcast(out_arr->data1(), n_sub_elems, mpi_typ8, root, comm),
            "broadcast_array: MPI error on MPI_Bcast:");
        CHECK_MPI(
            MPI_Bcast(out_arr->data2(), n_rows, mpi_typ_offset, root, comm),
            "broadcast_array: MPI error on MPI_Bcast:");
    } else if (arr_type == bodo_array_type::DICT) {
        if (is_sender && !in_arr->child_arrays[0]->is_globally_replicated) {
            throw std::runtime_error(
                "broadcast_array: not supported for DICT arrays without "
                "global dictionary.");
        }
        // Create a DICT out_arr
        std::shared_ptr<array_info> out_dict =
            ref_arr
                ? ref_arr->child_arrays[0]
                : broadcast_array(nullptr,
                                  is_sender ? in_arr->child_arrays[0] : nullptr,
                                  comm_ranks, is_parallel, root, myrank);
        out_arr = create_dict_string_array(
            out_dict,
            broadcast_array(ref_arr ? ref_arr->child_arrays[1] : nullptr,
                            is_sender ? in_arr->child_arrays[1] : nullptr,
                            comm_ranks, is_parallel, root, myrank));
    } else if (arr_type == bodo_array_type::ARRAY_ITEM) {
        MPI_Datatype mpi_typ_offset = get_MPI_typ(Bodo_CType_offset);
        if (is_sender) {
            out_arr = in_arr;
        } else {
            out_arr = alloc_array_item(n_rows, nullptr);
        }
        out_arr->child_arrays.front() =
            broadcast_array(ref_arr ? ref_arr->child_arrays.front() : nullptr,
                            out_arr->child_arrays.front(), comm_ranks,
                            is_parallel, root, myrank);
        CHECK_MPI(
            MPI_Bcast(out_arr->data1(), n_rows + 1, mpi_typ_offset, root, comm),
            "broadcast_array: MPI error on MPI_Bcast:");
    } else if (arr_type == bodo_array_type::STRUCT) {
        int32_t n_child_arrs;
        if (ref_arr) {
            n_child_arrs = ref_arr->child_arrays.size();
        } else {
            CHECK_MPI(MPI_Bcast(&n_child_arrs, 1, MPI_INT32_T, root, comm),
                      "broadcast_array: MPI error on MPI_Bcast:");
        }
        if (is_sender) {
            out_arr = in_arr;
        } else {
            out_arr = alloc_struct(
                n_rows, std::vector<std::shared_ptr<array_info>>(n_child_arrs));
        }
        for (size_t i = 0; i < out_arr->child_arrays.size(); ++i) {
            out_arr->child_arrays[i] =
                broadcast_array(ref_arr ? ref_arr->child_arrays[i] : nullptr,
                                out_arr->child_arrays[i], comm_ranks,
                                is_parallel, root, myrank);
        }
    } else if (arr_type == bodo_array_type::MAP) {
        if (is_sender) {
            out_arr = in_arr;
        } else {
            out_arr = alloc_map(n_rows, alloc_array_item(n_rows, nullptr));
        }
        out_arr->child_arrays[0] = broadcast_array(
            ref_arr ? ref_arr->child_arrays[0] : nullptr,
            out_arr->child_arrays[0], comm_ranks, is_parallel, root, myrank);
    } else {
        throw std::runtime_error(
            "Unsupported array type for broadcast_array: " +
            GetArrType_as_string(arr_type));
    }
    if (arr_type == bodo_array_type::NULLABLE_INT_BOOL ||
        arr_type == bodo_array_type::TIMESTAMPTZ ||
        arr_type == bodo_array_type::STRING ||
        arr_type == bodo_array_type::ARRAY_ITEM ||
        arr_type == bodo_array_type::STRUCT) {
        // broadcasting the null bitmask
        MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
        int n_bytes = (n_rows + 7) >> 3;
        CHECK_MPI(
            MPI_Bcast(out_arr->null_bitmask(), n_bytes, mpi_typ, root, comm),
            "broadcast_array: MPI error on MPI_Bcast:");
    }

    // Free communicator if created
    if (comm_ranks && comm_ranks->size() > 0) {
        MPI_Comm_free(&comm);
    }

    return out_arr;
}

array_info* broadcast_array_py_entry(array_info* in_arr,
                                     array_info* comm_ranks_in, int root,
                                     int64_t comm_ptr) {
    try {
        int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        bool is_sender = (myrank == root);
        if (comm_ptr != 0) {
            is_sender = (root == MPI_ROOT);
        }
        std::shared_ptr<array_info> input_array(in_arr);
        std::shared_ptr<array_info> comm_ranks_arr(comm_ranks_in);
        std::shared_ptr<std::vector<int>> comm_ranks =
            std::make_shared<std::vector<int>>();
        if (comm_ranks_arr && comm_ranks_arr->length) {
            int* data_ptr =
                comm_ranks_arr->data1<bodo_array_type::NUMPY, int>();
            comm_ranks->insert(comm_ranks->end(), data_ptr,
                               data_ptr + comm_ranks_arr->length);
        }
        std::shared_ptr<array_info> out_arr = broadcast_array(
            nullptr, is_sender ? input_array : nullptr, comm_ranks, true, root,
            myrank, reinterpret_cast<MPI_Comm*>(comm_ptr));
        return new array_info(*out_arr);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

std::shared_ptr<table_info> broadcast_table(
    std::shared_ptr<table_info> ref_table, std::shared_ptr<table_info> in_table,
    std::shared_ptr<std::vector<int>> comm_ranks, size_t n_cols,
    bool is_parallel, int root, MPI_Comm* comm_ptr) {
    tracing::Event ev("broadcast_table", is_parallel);
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    bool is_sender = (myrank == root);
    if (comm_ptr) {
        is_sender = (root == MPI_ROOT);
    }

    std::vector<std::shared_ptr<array_info>> out_arrs;
    out_arrs.reserve(n_cols);
    for (size_t i_col = 0; i_col < n_cols; i_col++) {
        // NOTE: in_table may not have columns in non-root ranks
        out_arrs.push_back(
            broadcast_array(ref_table ? ref_table->columns[i_col] : nullptr,
                            is_sender ? in_table->columns[i_col] : nullptr,
                            comm_ranks, is_parallel, root, myrank, comm_ptr));
    }

    return std::make_shared<table_info>(out_arrs);
}

table_info* broadcast_table_py_entry(table_info* in_table,
                                     array_info* comm_ranks_in, int root,
                                     int64_t comm_ptr) {
    try {
        std::shared_ptr<table_info> input_table(in_table);
        std::shared_ptr<array_info> comm_ranks_arr(comm_ranks_in);
        std::shared_ptr<std::vector<int>> comm_ranks =
            std::make_shared<std::vector<int>>();
        if (comm_ranks_arr && comm_ranks_arr->length) {
            int* data_ptr =
                comm_ranks_arr->data1<bodo_array_type::NUMPY, int>();
            comm_ranks->insert(comm_ranks->end(), data_ptr,
                               data_ptr + comm_ranks_arr->length);
        }
        std::shared_ptr<table_info> out_table = broadcast_table(
            nullptr, input_table, comm_ranks, input_table->ncols(), true, root,
            reinterpret_cast<MPI_Comm*>(comm_ptr));
        return new table_info(*out_table);
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
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
    if (n_pes == 1) {
        return false;
    }
    CHECK_MPI(MPI_Allreduce(&n_rows, &sum_n_rows, 1, MPI_LONG_LONG_INT, MPI_SUM,
                            MPI_COMM_WORLD),
              "need_reshuffling: MPI error on MPI_Allreduce:");
    CHECK_MPI(MPI_Allreduce(&n_rows, &max_n_rows, 1, MPI_LONG_LONG_INT, MPI_MAX,
                            MPI_COMM_WORLD),
              "need_reshuffling: MPI error on MPI_Allreduce:");
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
        CHECK_MPI(MPI_Allgather(&n_rows, 1, MPI_INT64_T, AllSizes.data(), 1,
                                MPI_INT64_T, MPI_COMM_WORLD),
                  "shuffle_renormalization_group: MPI error on MPI_Allgather:");
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
            CHECK_MPI(
                MPI_Bcast(&random_seed, 1, MPI_INT64_T, 0, MPI_COMM_WORLD),
                "shuffle_renormalization_group: MPI error on MPI_Bcast:");
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
    mpi_comm_info comm_info(in_table->columns, hashes, parallel);
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
        return nullptr;
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
        return nullptr;
    }
}
