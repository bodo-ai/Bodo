#include "_distributed.h"

#include <fmt/format.h>
#include <mpi.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <ctime>
#include <vector>

#include "_array_utils.h"
#include "_dict_builder.h"
#include "_shuffle.h"
#include "_table_builder_utils.h"
#include "streaming/_shuffle.h"

void print_and_raise_detailed_mpi_test_all_err(
    int err, const std::vector<MPI_Status> &req_status_arr,
    const std::string_view user_err_prefix) {
    char testall_mpi_err_msg_[MPI_MAX_ERROR_STRING + 1];
    int testall_mpi_err_msg_len = 0;
    MPI_Error_string(err, testall_mpi_err_msg_, &testall_mpi_err_msg_len);

    std::string overall_err_msg = fmt::format(
        "{}MPI_Error Code: {}. "
        "MPI_Error_string:\n\t{}\nIndividual Status Details:\n",
        user_err_prefix, err,
        std::string(testall_mpi_err_msg_, testall_mpi_err_msg_len));

    for (size_t i = 0; i < req_status_arr.size(); i++) {
        const MPI_Status &status = req_status_arr[i];
        std::string mpi_err_msg;
        if (status.MPI_ERROR == MPI_SUCCESS) {
            mpi_err_msg = "MPI_SUCCESS";
        } else if (status.MPI_ERROR == MPI_ERR_PENDING) {
            mpi_err_msg = "MPI_ERR_PENDING";
        } else {
            char mpi_err_msg_[MPI_MAX_ERROR_STRING + 1];
            int mpi_err_msg_len = 0;
            MPI_Error_string(status.MPI_ERROR, mpi_err_msg_, &mpi_err_msg_len);
            mpi_err_msg = std::string(mpi_err_msg_, mpi_err_msg_len);
        }
        overall_err_msg += fmt::format(
            "\tStatus #{}: MPI_SOURCE: {}, MPI_TAG: {}, MPI_ERROR Code: "
            "{}, MPI_Error_string:\n\t\t{}\n",
            i, status.MPI_SOURCE, status.MPI_TAG, status.MPI_ERROR,
            mpi_err_msg);
    }

    // Sometimes long messages aren't propagated properly all the way to Python,
    // so we also print the error message to be safe. A failure in MPI_Testall
    // implies a bug in Bodo (and potentially a hard to reproduce one), so it
    // should be rare.
    std::cerr << overall_err_msg << std::endl;
    throw std::runtime_error(overall_err_msg);
}

void _dist_transpose_comm(char *output, char *input, int typ_enum,
                          int64_t n_loc_rows, int64_t n_cols) {
    int myrank, n_pes;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);

    // Calculate send counts for each target rank, which is number of local rows
    // multiplied by number of columns that will become the target rank's rows
    // after transpose
    std::vector<int64_t> send_counts(n_pes);
    for (int i = 0; i < n_pes; i++) {
        int64_t n_target_cols = dist_get_node_portion(n_cols, n_pes, i);
        send_counts[i] = n_loc_rows * n_target_cols;
    }

    std::vector<int64_t> recv_counts(n_pes);
    CHECK_MPI(MPI_Alltoall(send_counts.data(), 1, MPI_INT64_T,
                           recv_counts.data(), 1, MPI_INT64_T, MPI_COMM_WORLD),
              "_dist_transpose_comm: MPI error on MPI_Alltoall:");

    std::vector<int64_t> send_disp(n_pes);
    std::vector<int64_t> recv_disp(n_pes);
    calc_disp(send_disp, send_counts);
    calc_disp(recv_disp, recv_counts);

    MPI_Datatype mpi_typ = get_MPI_typ(typ_enum);
    bodo_alltoallv(input, send_counts, send_disp, mpi_typ, output, recv_counts,
                   recv_disp, mpi_typ, MPI_COMM_WORLD);
}

/**
 * @brief Create a new IsLastState and return to Python
 *
 */
IsLastState *init_is_last_state() { return new IsLastState(); }

/**
 * @brief Delete IsLastState object (called from Python)
 *
 */
void delete_is_last_state(IsLastState *state) { delete state; }

/**
 * @brief Performs non-blocking synchronization of is_last flag
 *
 * @param state non-blocking synchronization state
 * @param local_is_last local is_last flag that needs synchronized
 * @return 1 if is_last is true on all ranks else 0
 */
int32_t sync_is_last_non_blocking(IsLastState *state, int32_t local_is_last) {
    if (state->global_is_last) {
        return 1;
    }

    if (local_is_last) {
        if (!state->is_last_barrier_started) {
            CHECK_MPI(
                MPI_Ibarrier(state->is_last_comm, &state->is_last_request),
                "sync_is_last_non_blocking: MPI error on MPI_Ibarrier:");
            state->is_last_barrier_started = true;
            return 0;
        } else {
            int flag = 0;
            CHECK_MPI(
                MPI_Test(&state->is_last_request, &flag, MPI_STATUS_IGNORE),
                "sync_is_last_non_blocking: MPI error on MPI_Test:");
            if (flag) {
                state->global_is_last = true;
            }
            return flag;
        }
    } else {
        return 0;
    }
}

int MPI_Gengather(void *sendbuf, int sendcount, MPI_Datatype sendtype,
                  void *recvbuf, int recvcount, MPI_Datatype recvtype,
                  int root_pe, MPI_Comm comm, bool all_gather) {
    if (all_gather) {
        return MPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                             recvtype, comm);
    } else {
        return MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount,
                          recvtype, root_pe, comm);
    }
}

int MPI_Gengatherv(const void *sendbuf, int64_t sendcount,
                   MPI_Datatype sendtype, void *recvbuf,
                   const int64_t *recvcounts, const int64_t *displs,
                   MPI_Datatype recvtype, int root_pe, MPI_Comm comm,
                   bool all_gather) {
    const MPI_Aint *mpi_displs = reinterpret_cast<const MPI_Aint *>(displs);
    const MPI_Count *mpi_recvcounts =
        reinterpret_cast<const MPI_Count *>(recvcounts);
    if (all_gather) {
        return MPI_Allgatherv_c(sendbuf, sendcount, sendtype, recvbuf,
                                mpi_recvcounts, mpi_displs, recvtype, comm);
    } else {
        return MPI_Gatherv_c(sendbuf, sendcount, sendtype, recvbuf,
                             mpi_recvcounts, mpi_displs, recvtype, root_pe,
                             comm);
    }
}

std::shared_ptr<array_info> gather_array(std::shared_ptr<array_info> in_arr,
                                         bool all_gather, bool is_parallel,
                                         int mpi_root, int n_pes, int myrank,
                                         MPI_Comm *comm_ptr) {
    int64_t n_rows = in_arr->length;
    int64_t n_sub_elems = in_arr->n_sub_elems();

    MPI_Comm comm = MPI_COMM_WORLD;
    bool is_receiver = (myrank == mpi_root);
    bool is_intercomm = false;
    if (comm_ptr != nullptr) {
        comm = *comm_ptr;
        is_intercomm = true;
        is_receiver = (mpi_root == MPI_ROOT);
        if (is_receiver) {
            CHECK_MPI(MPI_Comm_remote_size(*comm_ptr, &n_pes),
                      "gather_array: MPI error on MPI_Comm_remote_size:");
            n_rows = 0;
            n_sub_elems = 0;
        }
    }

    int64_t arr_gath_s[2] = {n_rows, n_sub_elems};
    bodo::vector<int64_t> arr_gath_r(2 * n_pes);
    CHECK_MPI(MPI_Gengather(arr_gath_s, 2, MPI_LONG_LONG_INT, arr_gath_r.data(),
                            2, MPI_LONG_LONG_INT, mpi_root, comm, all_gather),
              "_distributed.cpp::gather_array: MPI error on MPI_Gengather:");

    Bodo_CTypes::CTypeEnum dtype = in_arr->dtype;
    bodo_array_type::arr_type_enum arr_type = in_arr->arr_type;
    int64_t num_categories = in_arr->num_categories;

    std::vector<int64_t> rows_disps(n_pes), rows_counts(n_pes);
    int64_t rows_pos = 0;
    for (int i_p = 0; i_p < n_pes; i_p++) {
        int64_t siz = arr_gath_r[2 * i_p];
        rows_counts[i_p] = siz;
        rows_disps[i_p] = rows_pos;
        rows_pos += siz;
    }

    std::shared_ptr<array_info> out_arr;
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
            char *data_arr_i = in_arr->data1();
            std::vector<int64_t> recv_count_bytes(n_pes),
                recv_disp_bytes(n_pes);
            for (int i_p = 0; i_p < n_pes; i_p++) {
                recv_count_bytes[i_p] = (rows_counts[i_p] + 7) >> 3;
            }
            calc_disp(recv_disp_bytes, recv_count_bytes);
            size_t n_data_bytes = std::accumulate(
                recv_count_bytes.begin(), recv_count_bytes.end(), size_t(0));
            bodo::vector<uint8_t> tmp_data_bytes(n_data_bytes, 0);
            // Boolean arrays always store data as UINT8
            MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
            int64_t n_bytes = (n_rows + 7) >> 3;
            CHECK_MPI(
                MPI_Gengatherv(data_arr_i, n_bytes, mpi_typ,
                               tmp_data_bytes.data(), recv_count_bytes.data(),
                               recv_disp_bytes.data(), mpi_typ, mpi_root, comm,
                               all_gather),
                "_distributed.cpp::gather_array: MPI error on MPI_Gengatherv:");
            if (is_receiver || all_gather) {
                out_arr = alloc_array_top_level(n_rows_tot, -1, -1, arr_type,
                                                dtype, -1, 0, num_categories);
                uint8_t *data_arr_o = (uint8_t *)out_arr->data1();
                copy_gathered_null_bytes(data_arr_o, tmp_data_bytes,
                                         recv_count_bytes, rows_counts);
            }
        } else {
            MPI_Datatype mpi_typ = get_MPI_typ(dtype);
            char *data1_ptr = nullptr;
            if (is_receiver || all_gather) {
                out_arr = alloc_array_top_level(n_rows_tot, -1, -1, arr_type,
                                                dtype, -1, 0, num_categories);
                data1_ptr = out_arr->data1();
            }
            CHECK_MPI(
                MPI_Gengatherv(in_arr->data1(), n_rows, mpi_typ, data1_ptr,
                               rows_counts.data(), rows_disps.data(), mpi_typ,
                               mpi_root, comm, all_gather),
                "_distributed.cpp::gather_array: MPI error on MPI_Gengatherv:");
        }
    } else if (arr_type == bodo_array_type::INTERVAL) {
        MPI_Datatype mpi_typ = get_MPI_typ(dtype);
        // Computing the total number of rows.
        // On mpi_root, all rows, on others just 1 row for consistency.
        int64_t n_rows_tot = 0;
        for (int i_p = 0; i_p < n_pes; i_p++) {
            n_rows_tot += arr_gath_r[2 * i_p];
        }
        char *data1_ptr = nullptr;
        char *data2_ptr = nullptr;
        if (is_receiver || all_gather) {
            out_arr = alloc_array_top_level(n_rows_tot, -1, -1, arr_type, dtype,
                                            -1, 0, num_categories);
            data1_ptr = out_arr->data1();
            data2_ptr = out_arr->data2();
        }
        CHECK_MPI(
            MPI_Gengatherv(in_arr->data1(), n_rows, mpi_typ, data1_ptr,
                           rows_counts.data(), rows_disps.data(), mpi_typ,
                           mpi_root, comm, all_gather),
            "_distributed.cpp::gather_array: MPI error on MPI_Gengatherv:");
        CHECK_MPI(
            MPI_Gengatherv(in_arr->data2(), n_rows, mpi_typ, data2_ptr,
                           rows_counts.data(), rows_disps.data(), mpi_typ,
                           mpi_root, comm, all_gather),
            "_distributed.cpp::gather_array: MPI error on MPI_Gengatherv:");
    } else if (arr_type == bodo_array_type::STRING) {
        MPI_Datatype mpi_typ32 = get_MPI_typ(Bodo_CTypes::UINT32);
        MPI_Datatype mpi_typ8 = get_MPI_typ(Bodo_CTypes::UINT8);
        // Computing indexing data in characters and rows.
        int64_t n_rows_tot = 0;
        int64_t n_chars_tot = 0;
        for (int i_p = 0; i_p < n_pes; i_p++) {
            n_rows_tot += arr_gath_r[2 * i_p];
            n_chars_tot += arr_gath_r[2 * i_p + 1];
        }
        // Doing the characters
        char *data1_ptr = nullptr;
        if (is_receiver || all_gather) {
            out_arr =
                alloc_array_top_level(n_rows_tot, n_chars_tot, -1, arr_type,
                                      dtype, -1, 0, num_categories);
            data1_ptr = out_arr->data1();
        }
        std::vector<int64_t> char_disps(n_pes), char_counts(n_pes);
        int64_t pos = 0;
        for (int i_p = 0; i_p < n_pes; i_p++) {
            int64_t siz = arr_gath_r[2 * i_p + 1];
            char_disps[i_p] = pos;
            char_counts[i_p] = siz;
            pos += siz;
        }
        CHECK_MPI(
            MPI_Gengatherv(in_arr->data1(), n_sub_elems, mpi_typ8, data1_ptr,
                           char_counts.data(), char_disps.data(), mpi_typ8,
                           mpi_root, comm, all_gather),
            "_distributed.cpp::gather_array: MPI error on MPI_Gengatherv:");
        // Collecting the offsets data
        bodo::vector<uint32_t> list_count_loc(n_rows);
        offset_t *offsets_i = (offset_t *)in_arr->data2();
        offset_t curr_offset = 0;
        for (int64_t pos = 0; pos < n_rows; pos++) {
            offset_t new_offset = offsets_i[pos + 1];
            list_count_loc[pos] = new_offset - curr_offset;
            curr_offset = new_offset;
        }
        bodo::vector<uint32_t> list_count_tot(n_rows_tot);
        CHECK_MPI(
            MPI_Gengatherv(list_count_loc.data(), n_rows, mpi_typ32,
                           list_count_tot.data(), rows_counts.data(),
                           rows_disps.data(), mpi_typ32, mpi_root, comm,
                           all_gather),
            "_distributed.cpp::gather_array: MPI error on MPI_Gengatherv:");
        if (is_receiver || all_gather) {
            offset_t *offsets_o = (offset_t *)out_arr->data2();
            offsets_o[0] = 0;
            for (int64_t pos = 0; pos < n_rows_tot; pos++) {
                offsets_o[pos + 1] = offsets_o[pos] + list_count_tot[pos];
            }
        }
    } else if (arr_type == bodo_array_type::DICT) {
        // Note: We need to revisit if gather_table should be a no-op if
        // is_parallel=False
        if (!(is_receiver && is_intercomm)) {
            make_dictionary_global_and_unique(in_arr, true);
        }
        std::shared_ptr<array_info> dict_arr = in_arr->child_arrays[0];

        // Workers need to send dictionary to receiver in the intercomm case
        if (is_intercomm) {
            std::shared_ptr<array_info> dict_to_send = dict_arr;

            // Use gather to send dictionary slices to receiver
            // TODO(ehsan): use point-to-point array communication when
            // available
            if (!is_receiver) {
                int64_t start = dist_get_start(dict_arr->length, n_pes, myrank);
                int64_t end = dist_get_end(dict_arr->length, n_pes, myrank);
                int64_t size = end - start;
                std::vector<int64_t> slice_inds(size);
                std::iota(slice_inds.begin(), slice_inds.end(), start);
                dict_to_send =
                    RetrieveArray_SingleColumn(dict_arr, slice_inds, false);
            }
            dict_arr = gather_array(dict_to_send, false, is_parallel, mpi_root,
                                    n_pes, myrank, comm_ptr);
        }
        out_arr = gather_array(in_arr->child_arrays[1], all_gather, is_parallel,
                               mpi_root, n_pes, myrank, comm_ptr);
        if (all_gather || is_receiver) {
            out_arr = create_dict_string_array(dict_arr, out_arr);
        }
    } else if (arr_type == bodo_array_type::TIMESTAMPTZ) {
        // Computing the total number of rows.
        // On mpi_root, all rows, on others just 1 row for consistency.
        // The null bitmask is handled at the bottom of the function.
        int64_t n_rows_tot = 0;
        for (int i_p = 0; i_p < n_pes; i_p++) {
            n_rows_tot += rows_counts[i_p];
        }
        MPI_Datatype utc_mpi_typ = get_MPI_typ(dtype);
        MPI_Datatype offset_mpi_typ = get_MPI_typ(Bodo_CTypes::INT16);
        // Copy the UTC timestamp and offset minutes buffers
        char *data1_ptr = nullptr;
        char *data2_ptr = nullptr;
        if (is_receiver || all_gather) {
            out_arr = alloc_array_top_level(n_rows_tot, -1, -1, arr_type, dtype,
                                            -1, 0, num_categories);
            data1_ptr = out_arr->data1();
            data2_ptr = out_arr->data2();
        }
        CHECK_MPI(
            MPI_Gengatherv(in_arr->data1(), n_rows, utc_mpi_typ, data1_ptr,
                           rows_counts.data(), rows_disps.data(), utc_mpi_typ,
                           mpi_root, comm, all_gather),
            "_distributed.cpp::gather_array: MPI error on MPI_Gengatherv:");
        CHECK_MPI(
            MPI_Gengatherv(in_arr->data2(), n_rows, offset_mpi_typ, data2_ptr,
                           rows_counts.data(), rows_disps.data(),
                           offset_mpi_typ, mpi_root, comm, all_gather),
            "_distributed.cpp::gather_array: MPI error on MPI_Gengatherv:");
    } else if (arr_type == bodo_array_type::ARRAY_ITEM) {
        int64_t n_rows_tot = 0;
        for (int i_p = 0; i_p < n_pes; i_p++) {
            n_rows_tot += arr_gath_r[2 * i_p];
        }
        // Collecting the offsets data
        std::vector<offset_t> list_count_loc;
        list_count_loc.reserve(n_rows);
        offset_t *offsets_i = (offset_t *)in_arr->data1();
        for (int64_t pos = 0; pos < n_rows; pos++) {
            list_count_loc.push_back(offsets_i[pos + 1] - offsets_i[pos]);
        }
        std::vector<offset_t> list_count_tot(n_rows_tot);
        MPI_Datatype mpi_typ64 = get_MPI_typ(Bodo_CTypes::UINT64);
        CHECK_MPI(
            MPI_Gengatherv(list_count_loc.data(), n_rows, mpi_typ64,
                           list_count_tot.data(), rows_counts.data(),
                           rows_disps.data(), mpi_typ64, mpi_root, comm,
                           all_gather),
            "_distributed.cpp::gather_array: MPI error on MPI_Gengatherv:");
        // Gathering inner array
        out_arr = gather_array(in_arr->child_arrays.front(), all_gather,
                               is_parallel, mpi_root, n_pes, myrank, comm_ptr);
        if (is_receiver || all_gather) {
            out_arr = alloc_array_item(n_rows_tot, out_arr);
            offset_t *offsets_o = (offset_t *)out_arr->data1();
            offsets_o[0] = 0;
            for (int64_t pos = 0; pos < n_rows_tot; pos++) {
                offsets_o[pos + 1] = offsets_o[pos] + list_count_tot[pos];
            }
        }
    } else if (arr_type == bodo_array_type::STRUCT) {
        if (is_receiver || all_gather) {
            int64_t n_rows_tot = 0;
            for (int i_p = 0; i_p < n_pes; i_p++) {
                n_rows_tot += arr_gath_r[2 * i_p];
            }
            std::vector<std::shared_ptr<array_info>> child_arrays;
            child_arrays.reserve(in_arr->child_arrays.size());
            for (const auto &child_array : in_arr->child_arrays) {
                child_arrays.push_back(gather_array(child_array, all_gather,
                                                    is_parallel, mpi_root,
                                                    n_pes, myrank, comm_ptr));
            }
            out_arr = alloc_struct(n_rows_tot, std::move(child_arrays));
        } else {
            for (const auto &child_array : in_arr->child_arrays) {
                gather_array(child_array, all_gather, is_parallel, mpi_root,
                             n_pes, myrank, comm_ptr);
            }
        }
    } else if (arr_type == bodo_array_type::MAP) {
        std::shared_ptr<array_info> out_arr_item =
            gather_array(in_arr->child_arrays[0], all_gather, is_parallel,
                         mpi_root, n_pes, myrank, comm_ptr);
        if (is_receiver || all_gather) {
            out_arr = alloc_map(out_arr_item->length, out_arr_item);
        }
    } else {
        throw std::runtime_error("Unexpected array type in gather_array: " +
                                 GetArrType_as_string(arr_type));
    }
    if (arr_type == bodo_array_type::STRING ||
        arr_type == bodo_array_type::NULLABLE_INT_BOOL ||
        arr_type == bodo_array_type::TIMESTAMPTZ ||
        arr_type == bodo_array_type::ARRAY_ITEM ||
        arr_type == bodo_array_type::STRUCT) {
        char *null_bitmask_i = in_arr->null_bitmask();
        std::vector<int64_t> recv_count_null(n_pes), recv_disp_null(n_pes);
        for (int i_p = 0; i_p < n_pes; i_p++) {
            recv_count_null[i_p] = (rows_counts[i_p] + 7) >> 3;
        }
        calc_disp(recv_disp_null, recv_count_null);
        size_t n_null_bytes = std::accumulate(recv_count_null.begin(),
                                              recv_count_null.end(), size_t(0));
        bodo::vector<uint8_t> tmp_null_bytes(n_null_bytes, 0);
        MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
        int64_t n_bytes = (n_rows + 7) >> 3;
        CHECK_MPI(
            MPI_Gengatherv(null_bitmask_i, n_bytes, mpi_typ,
                           tmp_null_bytes.data(), recv_count_null.data(),
                           recv_disp_null.data(), mpi_typ, mpi_root, comm,
                           all_gather),
            "_distributed.cpp::gather_array: MPI error on MPI_Gengatherv:");
        if (is_receiver || all_gather) {
            char *null_bitmask_o = out_arr->null_bitmask();
            copy_gathered_null_bytes((uint8_t *)null_bitmask_o, tmp_null_bytes,
                                     recv_count_null, rows_counts);
        }
    }

    return out_arr;
}

std::shared_ptr<table_info> gather_table(std::shared_ptr<table_info> in_table,
                                         int64_t n_cols, bool all_gather,
                                         bool is_parallel, int mpi_root,
                                         MPI_Comm *comm_ptr) {
    tracing::Event ev("gather_table", is_parallel);
    int n_pes, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    std::vector<std::shared_ptr<array_info>> out_arrs;
    if (n_cols == -1) {
        n_cols = in_table->ncols();
    }
    out_arrs.reserve(n_cols);
    for (int64_t i_col = 0; i_col < n_cols; i_col++) {
        out_arrs.push_back(gather_array(in_table->columns[i_col], all_gather,
                                        is_parallel, mpi_root, n_pes, myrank,
                                        comm_ptr));
    }
    return std::make_shared<table_info>(out_arrs);
}

table_info *gather_table_py_entry(table_info *in_table, bool all_gather,
                                  int mpi_root, int64_t comm_ptr) {
    try {
        int myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        bool is_receiver =
            comm_ptr != 0 ? (mpi_root == MPI_ROOT) : (myrank == mpi_root);

        std::shared_ptr<table_info> table(in_table);
        std::shared_ptr<table_info> output =
            gather_table(table, in_table->ncols(), all_gather, true, mpi_root,
                         reinterpret_cast<MPI_Comm *>(comm_ptr));
        if (!is_receiver && !all_gather) {
            output = alloc_table_like(table);
        }

        return new table_info(*output);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

array_info *gather_array_py_entry(array_info *in_array, bool all_gather,
                                  int mpi_root, int64_t comm_ptr) {
    try {
        int myrank, n_pes;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);

        MPI_Comm *mpi_comm_ptr = reinterpret_cast<MPI_Comm *>(comm_ptr);
        bool is_receiver = (myrank == mpi_root);
        if (mpi_comm_ptr != nullptr) {
            is_receiver = (mpi_root == MPI_ROOT);
            if (is_receiver) {
                CHECK_MPI(MPI_Comm_remote_size(*mpi_comm_ptr, &n_pes),
                          "gather_array_py_entry: MPI error on "
                          "MPI_Comm_remote_size:");
            }
        }

        std::shared_ptr<array_info> array(in_array);
        std::shared_ptr<array_info> output = gather_array(
            array, all_gather, true, mpi_root, n_pes, myrank, mpi_comm_ptr);

        if (!is_receiver && !all_gather) {
            output = alloc_array_like(array);
        }
        return new array_info(*output);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

// Corresponds to Numba's IndexValueType, used for MPI ArgMin/ArgMax.
// https://github.com/numba/numba/blob/bced43d44b405a9db443b70171eb0125216770b6/numba/core/typing/new_builtins.py#L1152
#pragma pack(1)
template <typename T>
struct IndexValuePair {
    int64_t index;
    T value;
};

/**
 * @brief Create a pair MPI datatype (index & value) for the given MPI value
 * type
 *
 * @param mpi_typ value type
 * @return MPI_Datatype output pair MPI data type
 */
MPI_Datatype createPairDatatype(MPI_Datatype mpi_typ) {
    MPI_Datatype newType;

    // Each field is length 1 (1 index & 1 value)
    int blocklengths[2] = {1, 1};

    // The offsets for index/value members in IndexValuePair
    MPI_Aint offsets[2] = {0, sizeof(int64_t)};

    // Index/Value MPI data types
    MPI_Datatype types[2] = {MPI_INT64_T, mpi_typ};

    // Create and commit the custom datatype.
    CHECK_MPI(MPI_Type_create_struct(2, blocklengths, offsets, types, &newType),
              "dist_reduce: MPI error on MPI_Type_create_struct:");
    CHECK_MPI(MPI_Type_commit(&newType),
              "dist_reduce: MPI error on MPI_Type_commit:");

    return newType;
}

/**
 * @brief Custom MPI_Op function for ArgMin
 *
 * @tparam T value type
 * @param invec input index/value
 * @param inoutvec input/output index/value
 * @param len number of elements (1 in our case but the op has to be general)
 * @param datatype MPI data type of input (not used since templated)
 */
template <typename T>
void ArgMinFunction(void *invec, void *inoutvec, int *len,
                    MPI_Datatype *datatype) {
    IndexValuePair<T> *in = static_cast<IndexValuePair<T> *>(invec);
    IndexValuePair<T> *inout = static_cast<IndexValuePair<T> *>(inoutvec);

    for (int i = 0; i < *len; i++) {
        // If incoming value is smaller, take it
        if (in[i].value < inout[i].value) {
            inout[i].value = in[i].value;
            inout[i].index = in[i].index;
        }
        // If tie, pick the smaller index
        else if (in[i].value == inout[i].value) {
            if (in[i].index < inout[i].index) {
                inout[i].index = in[i].index;
            }
        }
    }
}

/**
 * @brief Custom MPI_Op function for ArgMax
 *
 * @tparam T value type
 * @param invec input index/value
 * @param inoutvec input/output index/value
 * @param len number of elements (1 in our case but the op has to be general)
 * @param datatype MPI data type of input (not used since templated)
 */
template <typename T>
void ArgMaxFunction(void *invec, void *inoutvec, int *len,
                    MPI_Datatype *datatype) {
    IndexValuePair<T> *in = static_cast<IndexValuePair<T> *>(invec);
    IndexValuePair<T> *inout = static_cast<IndexValuePair<T> *>(inoutvec);

    for (int i = 0; i < *len; i++) {
        // If incoming value is larger, take it
        if (in[i].value > inout[i].value) {
            inout[i].value = in[i].value;
            inout[i].index = in[i].index;
        }
        // If tie, pick the smaller index
        else if (in[i].value == inout[i].value) {
            if (in[i].index < inout[i].index) {
                inout[i].index = in[i].index;
            }
        }
    }
}

template <typename T>
MPI_Op createArgMinOp() {
    MPI_Op op;
    CHECK_MPI(MPI_Op_create(&ArgMinFunction<T>, /*commute=*/true, &op),
              "dist_reduce: MPI error on MPI_Op_create:");
    return op;
}

template <typename T>
MPI_Op createArgMaxOp() {
    MPI_Op op;
    CHECK_MPI(MPI_Op_create(&ArgMaxFunction<T>, /*commute=*/true, &op),
              "dist_reduce: MPI error on MPI_Op_create:");
    return op;
}

#define EXPAND_ARGMINMAX(mpi_dtype, ctype)  \
    if (mpi_typ == (mpi_dtype)) {           \
        if (mpi_op == MPI_MINLOC) {         \
            return createArgMinOp<ctype>(); \
        } else if (mpi_op == MPI_MAXLOC) {  \
            return createArgMaxOp<ctype>(); \
        }                                   \
    }

/**
 * @brief Create a custom MPI_Op for argmin/argmax for provided MPI data type
 *
 * @param mpi_typ value type
 * @param mpi_op MPI_MINLOC or MPI_MAXLOC, specifying which operation to use
 * @return MPI_Op custom MPI operator for reduction
 */
MPI_Op createArgMinMaxOp(MPI_Datatype mpi_typ, MPI_Op mpi_op) {
    // Instantiate templates for argmin/argmax data types (should match relevant
    // types in get_MPI_typ)
    EXPAND_ARGMINMAX(MPI_UNSIGNED_CHAR, unsigned char)
    EXPAND_ARGMINMAX(MPI_CHAR, char)
    EXPAND_ARGMINMAX(MPI_INT, int)
    EXPAND_ARGMINMAX(MPI_UNSIGNED, uint32_t)
    EXPAND_ARGMINMAX(MPI_LONG_LONG_INT, int64_t)
    EXPAND_ARGMINMAX(MPI_UNSIGNED_LONG_LONG, uint64_t)
    EXPAND_ARGMINMAX(MPI_FLOAT, float)
    EXPAND_ARGMINMAX(MPI_DOUBLE, double)
    EXPAND_ARGMINMAX(MPI_SHORT, int16_t)
    EXPAND_ARGMINMAX(MPI_UNSIGNED_SHORT, uint16_t)

    throw std::runtime_error("Unsupported MPI operation for ArgMin/ArgMax");
}

void dist_reduce(char *in_ptr, char *out_ptr, int op_enum, int type_enum,
                 int64_t comm_ptr) {
    try {
        MPI_Datatype mpi_typ = get_MPI_typ(type_enum);
        MPI_Op mpi_op = get_MPI_op(op_enum);
        MPI_Comm comm = MPI_COMM_WORLD;
        if (comm_ptr != 0) {
            comm = *(reinterpret_cast<MPI_Comm *>(comm_ptr));
        }

        // argmax and argmin need special handling
        if (mpi_op == MPI_MAXLOC || mpi_op == MPI_MINLOC) {
            MPI_Datatype pairMPIType = createPairDatatype(mpi_typ);
            MPI_Op argMPIOp = createArgMinMaxOp(mpi_typ, mpi_op);

            CHECK_MPI(
                MPI_Allreduce(in_ptr, out_ptr, 1, pairMPIType, argMPIOp, comm),
                "_distributed.h::dist_reduce: MPI error on MPI_Allreduce "
                "(argmin/argmax):");

            CHECK_MPI(MPI_Op_free(&argMPIOp),
                      "_distributed.h::dist_reduce: MPI error on MPI_Op_free:");
            CHECK_MPI(
                MPI_Type_free(&pairMPIType),
                "_distributed.h::dist_reduce: MPI error on MPI_Type_free:");
        } else {
            CHECK_MPI(MPI_Allreduce(in_ptr, out_ptr, 1, mpi_typ, mpi_op, comm),
                      "_distributed.h::dist_reduce:");
        }
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

// Only works on x86 Apple machines
#if defined(__APPLE__) && defined(__x86_64__)
#include <cpuid.h>

#define CPUID(INFO, LEAF, SUBLEAF) \
    __cpuid_count(LEAF, SUBLEAF, INFO[0], INFO[1], INFO[2], INFO[3])

#define GETCPU(CPU)                                     \
    {                                                   \
        uint32_t CPUInfo[4];                            \
        CPUID(CPUInfo, 1, 0);                           \
        /* CPUInfo[1] is EBX, bits 24-31 are APIC ID */ \
        if ((CPUInfo[3] & (1 << 9)) == 0) {             \
            CPU = -1; /* no APIC on chip */             \
        } else {                                        \
            CPU = (unsigned)CPUInfo[1] >> 24;           \
        }                                               \
        if (CPU < 0)                                    \
            CPU = 0;                                    \
    }
#endif

/**
 * @brief Get the ID of the CPU that this thread is running on. Returns -1 if we
 * cannot get the ID, e.g. if on Windows.
 *
 * @return int
 */
[[maybe_unused]] static int get_cpu_id() {
    int cpu_id = -1;
#ifdef __linux__
    cpu_id = sched_getcpu();
#elif defined(__APPLE__) && defined(__x86_64__)
    GETCPU(cpu_id);
#endif
    return cpu_id;
}

/**
 * @brief Wrapper around get_rank() to be called from Python (avoids Numba JIT
 overhead and makes compiler debugging easier by eliminating extra compilation)
 *
 */
static PyObject *get_rank_py_wrapper(PyObject *self, PyObject *args) {
    if (PyTuple_Size(args) != 0) {
        PyErr_SetString(PyExc_TypeError, "get_rank() does not take arguments");
        return nullptr;
    }
    PyObject *rank_obj = PyLong_FromLong(dist_get_rank());
    return rank_obj;
}

/**
 * @brief Wrapper around get_size() to be called from Python (avoids Numba JIT
 overhead and makes compiler debugging easier by eliminating extra compilation)
 *
 */
static PyObject *get_size_py_wrapper(PyObject *self, PyObject *args) {
    if (PyTuple_Size(args) != 0) {
        PyErr_SetString(PyExc_TypeError, "get_size() does not take arguments");
        return nullptr;
    }
    PyObject *size_obj = PyLong_FromLong(dist_get_size());
    return size_obj;
}

/**
 * @brief Wrapper around finalize() to be called from Python (avoids Numba JIT
 overhead and makes compiler debugging easier by eliminating extra compilation)
 *
 */
static PyObject *finalize_py_wrapper(PyObject *self, PyObject *args) {
    if (PyTuple_Size(args) != 0) {
        PyErr_SetString(PyExc_TypeError, "finalize() does not take arguments");
        return nullptr;
    }
    PyObject *ret_obj = PyLong_FromLong(finalize());
    return ret_obj;
}

static PyMethodDef ext_methods[] = {
#define declmethod(func) {#func, (PyCFunction)func, METH_VARARGS, NULL}
    declmethod(get_rank_py_wrapper),
    declmethod(get_size_py_wrapper),
    declmethod(finalize_py_wrapper),
    {nullptr},
#undef declmethod
};

PyMODINIT_FUNC PyInit_hdist(void) {
    PyObject *m;
    MOD_DEF(m, "hdist", "No docs", ext_methods);
    if (m == nullptr) {
        return nullptr;
    }

    // make sure MPI is initialized, assuming this will be called
    // on all processes
    int is_initialized;
    MPI_Initialized(&is_initialized);
    if (!is_initialized)
        CHECK_MPI(MPI_Init(nullptr, nullptr),
                  "PyInit_hdist: MPI error on MPI_Init:");

    int decimal_bytes;
    CHECK_MPI(MPI_Type_size(get_MPI_typ(Bodo_CTypes::DECIMAL), &decimal_bytes),
              "PyInit_hdist: MPI error on MPI_Type_size:");
    // decimal_value should be exactly 128 bits to match Python
    if (decimal_bytes != 16) {
        std::cerr << "invalid decimal mpi type size" << std::endl;
    }

    SetAttrStringFromVoidPtr(m, dist_get_rank);
    SetAttrStringFromVoidPtr(m, dist_get_size);
    SetAttrStringFromVoidPtr(m, dist_get_remote_size);
    SetAttrStringFromVoidPtr(m, dist_get_start);
    SetAttrStringFromVoidPtr(m, dist_get_end);
    SetAttrStringFromVoidPtr(m, dist_get_node_portion);
    SetAttrStringFromVoidPtr(m, get_time);
    SetAttrStringFromVoidPtr(m, barrier);

    SetAttrStringFromVoidPtr(m, dist_reduce);
    SetAttrStringFromVoidPtr(m, dist_exscan);
    SetAttrStringFromVoidPtr(m, dist_arr_reduce);
    SetAttrStringFromVoidPtr(m, dist_irecv);
    SetAttrStringFromVoidPtr(m, dist_isend);
    SetAttrStringFromVoidPtr(m, dist_recv);
    SetAttrStringFromVoidPtr(m, dist_send);
    SetAttrStringFromVoidPtr(m, dist_wait);
    SetAttrStringFromVoidPtr(m, dist_get_item_pointer);
    SetAttrStringFromVoidPtr(m, get_dummy_ptr);
    SetAttrStringFromVoidPtr(m, c_gather_scalar);
    SetAttrStringFromVoidPtr(m, c_gatherv);
    SetAttrStringFromVoidPtr(m, c_allgatherv);
    SetAttrStringFromVoidPtr(m, c_scatterv);
    SetAttrStringFromVoidPtr(m, c_bcast);
    SetAttrStringFromVoidPtr(m, broadcast_array_py_entry);
    SetAttrStringFromVoidPtr(m, broadcast_table_py_entry);
    SetAttrStringFromVoidPtr(m, allgather);
    SetAttrStringFromVoidPtr(m, oneD_reshape_shuffle);
    SetAttrStringFromVoidPtr(m, permutation_int);
    SetAttrStringFromVoidPtr(m, permutation_array_index);
    SetAttrStringFromVoidPtr(m, timestamptz_reduce);
    SetAttrStringFromVoidPtr(m, decimal_reduce);
    SetAttrStringFromVoidPtr(m, _dist_transpose_comm);
    SetAttrStringFromVoidPtr(m, init_is_last_state);
    SetAttrStringFromVoidPtr(m, delete_is_last_state);
    SetAttrStringFromVoidPtr(m, sync_is_last_non_blocking);
    SetAttrStringFromVoidPtr(m, get_cpu_id);

    SetAttrStringFromVoidPtr(m, gather_table_py_entry);
    SetAttrStringFromVoidPtr(m, gather_array_py_entry);

    // add actual int value to module
    PyObject_SetAttrString(m, "mpi_req_num_bytes",
                           PyLong_FromSize_t(get_mpi_req_num_bytes()));
    PyObject_SetAttrString(m, "ANY_SOURCE",
                           PyLong_FromLong((long)MPI_ANY_SOURCE));

    return m;
}
