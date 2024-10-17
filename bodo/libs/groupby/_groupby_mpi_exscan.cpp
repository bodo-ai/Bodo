// Copyright (C) 2023 Bodo Inc. All rights reserved.

#include "_groupby_mpi_exscan.h"
#include "../_array_hash.h"
#include "../_array_operations.h"
#include "../_array_utils.h"
#include "../_bodo_common.h"
#include "../_dict_builder.h"
#include "../_distributed.h"
#include "../_shuffle.h"
#include "_groupby_common.h"
#include "_groupby_ftypes.h"
#include "_groupby_hashing.h"

// Strategy for determining exscan

/**
 * This file implements the functions that are used to determine and utilize
 * the MPI_Exscan strategy for groupby. This strategy is used when we have
 * only cumulative operations to avoid shuffling the data.
 */

int determine_groupby_strategy(std::shared_ptr<table_info> in_table,
                               int64_t num_keys, int8_t* ncols_per_func,
                               int64_t num_funcs, int* ftypes,
                               bool input_has_index) {
    // First decision: If it is cumulative, then we can use the MPI_Exscan.
    // Otherwise no
    bool has_non_cumulative_op = false;
    bool has_cumulative_op = false;
    bool has_multi_col_input = false;
    int index_i = int(input_has_index);
    for (int i = 0; i < num_funcs; i++) {
        int ftype = ftypes[i];
        if (ftype == Bodo_FTypes::cumsum || ftype == Bodo_FTypes::cummin ||
            ftype == Bodo_FTypes::cumprod || ftype == Bodo_FTypes::cummax) {
            has_cumulative_op = true;
        } else {
            has_non_cumulative_op = true;
        }
        if (ncols_per_func[i] != 1) {
            has_multi_col_input = true;
        }
    }
    if (has_non_cumulative_op) {
        return 0;  // No choice, we have to use the classic hash scheme
    }
    if (!has_cumulative_op) {
        return 0;  // It does not make sense to use MPI_exscan here.
    }
    if (has_multi_col_input) {
        // All cumulative operations expect only a single function input.
        // The framework assumes we use exactly 1 input column per function
        // so if this has changed we must use the classic hash scheme.
        return 0;
    }
    // Second decision: Whether it is arithmetic or not. If arithmetic, we can
    // use MPI_Exscan. If not, we may make it work for cumsum of strings or list
    // of strings but that would be definitely quite complicated and use more
    // than just MPI_Exscan.
    bool has_non_arithmetic_type = false;
    for (uint64_t i = num_keys; i < in_table->ncols() - index_i; i++) {
        const std::shared_ptr<array_info>& oper_col = in_table->columns[i];
        if (oper_col->arr_type != bodo_array_type::NUMPY &&
            oper_col->arr_type != bodo_array_type::NULLABLE_INT_BOOL) {
            has_non_arithmetic_type = true;
        }
    }
    if (has_non_arithmetic_type) {
        return 0;  // No choice, we have to use the classic hash scheme
    }
    // Third decision: Whether we use categorical with just one key. Working
    // with other keys would require some preprocessing.
    if (num_keys > 1) {
        return 2;  // For more than 1 key column, use multikey mpi_exscan
    }
    bodo_array_type::arr_type_enum key_arr_type =
        in_table->columns[0]->arr_type;
    if (key_arr_type != bodo_array_type::CATEGORICAL) {
        return 2;  // For key column that are not categorical, use multikey
                   // mpi_exscan
    }
    if (in_table->columns[0]->num_categories >
        max_global_number_groups_exscan) {
        return 0;  // For too many categories the hash partition will be better
    }
    return 1;  // all conditions satisfied. Let's go for EXSCAN code
}

// Categorical index info

std::shared_ptr<array_info> compute_categorical_index(
    std::shared_ptr<table_info> in_table, int64_t num_keys, bool is_parallel,
    bool key_dropna) {
    tracing::Event ev("compute_categorical_index", is_parallel);
    for (int64_t i_key = 0; i_key < num_keys; i_key++) {
        const std::shared_ptr<array_info>& a = in_table->columns[i_key];
        if (a->arr_type == bodo_array_type::DICT) {
            make_dictionary_global_and_unique(a, is_parallel);
        }
    }
    std::shared_ptr<table_info> red_table =
        drop_duplicates_keys(in_table, num_keys, is_parallel, key_dropna);
    size_t n_rows_full, n_rows = red_table->nrows();
    if (is_parallel) {
        HANDLE_MPI_ERROR(
            MPI_Allreduce(&n_rows, &n_rows_full, 1, MPI_LONG_LONG_INT, MPI_SUM,
                          MPI_COMM_WORLD),
            "compute_categorical_index: MPI error on MPI_Allreduce:");
    } else {
        n_rows_full = n_rows;
    }
    // Two approaches for cumulative operations : shuffle (then reshuffle) or
    // use exscan. Preferable to do shuffle when we have too many unique values.
    // This is a heuristic to decide approach.
    if (n_rows_full > max_global_number_groups_exscan) {
        return nullptr;
    }
    // We are below threshold. Now doing an allgather for determining the keys.
    bool all_gather = true;
    std::shared_ptr<table_info> full_table;
    if (is_parallel) {
        full_table = gather_table(red_table, num_keys, all_gather, is_parallel);
    } else {
        full_table = red_table;
    }
    // Now building the map_container.
    std::shared_ptr<uint32_t[]> hashes_full =
        hash_keys_table(full_table, num_keys, SEED_HASH_MULTIKEY, is_parallel);
    std::shared_ptr<uint32_t[]> hashes_in_table =
        hash_keys_table(in_table, num_keys, SEED_HASH_MULTIKEY, is_parallel);
    std::vector<std::shared_ptr<array_info>> concat_column(
        full_table->columns.begin(), full_table->columns.begin() + num_keys);
    concat_column.insert(concat_column.end(), in_table->columns.begin(),
                         in_table->columns.begin() + num_keys);

    HashComputeCategoricalIndex hash_fct{hashes_full, hashes_in_table,
                                         n_rows_full};
    HashEqualComputeCategoricalIndex equal_fct{num_keys, n_rows_full,
                                               &concat_column};
    bodo::unord_map_container<size_t, size_t, HashComputeCategoricalIndex,
                              HashEqualComputeCategoricalIndex>
        entSet({}, hash_fct, equal_fct);
    for (size_t iRow = 0; iRow < size_t(n_rows_full); iRow++) {
        entSet[iRow] = iRow;
    }
    size_t n_rows_in = in_table->nrows();
    std::shared_ptr<array_info> out_arr =
        alloc_categorical(n_rows_in, Bodo_CTypes::INT32, n_rows_full);
    std::vector<std::shared_ptr<array_info>> key_cols(
        in_table->columns.begin(), in_table->columns.begin() + num_keys);
    bool has_nulls = does_keys_have_nulls(key_cols);
    for (size_t iRow = 0; iRow < n_rows_in; iRow++) {
        int32_t pos;
        if (has_nulls) {
            if (key_dropna && does_row_has_nulls(key_cols, iRow)) {
                pos = -1;
            } else {
                pos = entSet[iRow + n_rows_full];
            }
        } else {
            pos = entSet[iRow + n_rows_full];
        }
        out_arr->at<int32_t, bodo_array_type::CATEGORICAL>(iRow) = pos;
    }
    return out_arr;
}

// MPI_Exscan: https://www.mpich.org/static/docs/v3.1.x/www3/MPI_Exscan.html
// Useful for cumulative functions. Instead of doing shuffling, we compute the
// groups in advance without doing shuffling using MPI_Exscan. We do the
// cumulative operation first locally on each processor, and we use step
// functions on each processor (sum, min, etc.)

/**
 * @brief MPI exscan implementation on numpy arrays.
 *
 * @tparam Tkey The type of the key column
 * @tparam T The type of the operation column
 * @tparam dtype The dtype of the operation column.
 * @param out_arrs The output arrays
 * @param cat_column The categorical column
 * @param in_table The input table
 * @param num_keys The number of key columns
 * @param k The index of the operation column
 * @param ftypes The types of the functions
 * @param func_offsets The offsets of the functions
 * @param is_parallel Whether the computation is parallel
 * @param skip_na_data Whether to skip/drop na data
 */
template <typename Tkey, typename T, Bodo_CTypes::CTypeEnum DType>
void mpi_exscan_computation_numpy_T(
    std::vector<std::shared_ptr<array_info>>& out_arrs,
    std::shared_ptr<array_info> cat_column,
    std::shared_ptr<table_info> in_table, int64_t num_keys, int64_t k,
    int* ftypes, int* func_offsets, bool is_parallel, bool skip_na_data) {
    assert(cat_column->arr_type == bodo_array_type::CATEGORICAL);
    int64_t n_rows = in_table->nrows();
    int start = func_offsets[k];
    int end = func_offsets[k + 1];
    int n_oper = end - start;
    int64_t max_row_idx = cat_column->num_categories;
    bodo::vector<T> cumulative(max_row_idx * n_oper);
    for (int j = start; j != end; j++) {
        int ftype = ftypes[j];
        T value_init = -1;  // Dummy value set to avoid a compiler warning
        if (ftype == Bodo_FTypes::cumsum) {
            value_init = 0;
        } else if (ftype == Bodo_FTypes::cumprod) {
            value_init = 1;
        } else if (ftype == Bodo_FTypes::cummax) {
            value_init = std::numeric_limits<T>::min();
        } else if (ftype == Bodo_FTypes::cummin) {
            value_init = std::numeric_limits<T>::max();
        }
        for (int i_row = 0; i_row < max_row_idx; i_row++) {
            cumulative[i_row + max_row_idx * (j - start)] = value_init;
        }
    }
    bodo::vector<T> cumulative_recv = cumulative;
    std::shared_ptr<array_info> in_col = in_table->columns[k + num_keys];
    assert(in_col->arr_type == bodo_array_type::NUMPY);
    T nan_value = GetTentry<T>(RetrieveNaNentry(DType).data());
    Tkey miss_idx = -1;
    for (int j = start; j != end; j++) {
        const std::shared_ptr<array_info>& work_col = out_arrs[j];
        int ftype = ftypes[j];
        auto apply_oper = [&](auto const& oper) -> void {
            for (int64_t i_row = 0; i_row < n_rows; i_row++) {
                Tkey idx =
                    cat_column->at<Tkey, bodo_array_type::CATEGORICAL>(i_row);
                if (idx == miss_idx) {
                    work_col->at<T>(i_row) = nan_value;
                } else {
                    size_t pos = idx + max_row_idx * (j - start);
                    T val = in_col->at<T, bodo_array_type::NUMPY>(i_row);
                    if (skip_na_data && isnan_alltype<T, DType>(val)) {
                        work_col->at<T>(i_row) = val;
                    } else {
                        T new_val = oper(val, cumulative[pos]);
                        work_col->at<T>(i_row) = new_val;
                        cumulative[pos] = new_val;
                    }
                }
            }
        };
        if (ftype == Bodo_FTypes::cumsum) {
            apply_oper([](T val1, T val2) -> T { return val1 + val2; });
        } else if (ftype == Bodo_FTypes::cumprod) {
            apply_oper([](T val1, T val2) -> T { return val1 * val2; });
        } else if (ftype == Bodo_FTypes::cummax) {
            apply_oper(
                [](T val1, T val2) -> T { return std::max(val1, val2); });
        } else if (ftype == Bodo_FTypes::cummin) {
            apply_oper(
                [](T val1, T val2) -> T { return std::min(val1, val2); });
        }
    }
    if (!is_parallel) {
        return;
    }
    MPI_Datatype mpi_typ = get_MPI_typ((int)DType);
    for (int j = start; j != end; j++) {
        T* data_s = cumulative.data() + max_row_idx * (j - start);
        T* data_r = cumulative_recv.data() + max_row_idx * (j - start);
        int ftype = ftypes[j];
        if (ftype == Bodo_FTypes::cumsum) {
            HANDLE_MPI_ERROR(MPI_Exscan(data_s, data_r, max_row_idx, mpi_typ,
                                        MPI_SUM, MPI_COMM_WORLD),
                             "mpi_exscan_computation_numpy_T[cumsum]: MPI "
                             "error on MPI_Exscan:");
        } else if (ftype == Bodo_FTypes::cumprod) {
            HANDLE_MPI_ERROR(MPI_Exscan(data_s, data_r, max_row_idx, mpi_typ,
                                        MPI_PROD, MPI_COMM_WORLD),
                             "mpi_exscan_computation_numpy_T[cumprod]: MPI "
                             "error on MPI_Exscan:");
        } else if (ftype == Bodo_FTypes::cummax) {
            HANDLE_MPI_ERROR(MPI_Exscan(data_s, data_r, max_row_idx, mpi_typ,
                                        MPI_MAX, MPI_COMM_WORLD),
                             "mpi_exscan_computation_numpy_T[cummax]: MPI "
                             "error on MPI_Exscan:");
        } else if (ftype == Bodo_FTypes::cummin) {
            HANDLE_MPI_ERROR(MPI_Exscan(data_s, data_r, max_row_idx, mpi_typ,
                                        MPI_MIN, MPI_COMM_WORLD),
                             "mpi_exscan_computation_numpy_T[cummin]: MPI "
                             "error on MPI_Exscan:");
        }
    }
    for (int j = start; j != end; j++) {
        const std::shared_ptr<array_info>& work_col = out_arrs[j];
        int ftype = ftypes[j];
        // For skip_na_data:
        //   The cumulative is never a NaN. The sum therefore works
        //   correctly whether val is a NaN or not.
        // For !skip_na_data:
        //   the cumulative can be a NaN. The sum also works correctly.
        auto apply_oper = [&](auto const& oper) -> void {
            for (int64_t i_row = 0; i_row < n_rows; i_row++) {
                Tkey idx =
                    cat_column->at<Tkey, bodo_array_type::CATEGORICAL>(i_row);
                if (idx != miss_idx) {
                    size_t pos = idx + max_row_idx * (j - start);
                    T val = work_col->at<T>(i_row);
                    T new_val = oper(val, cumulative_recv[pos]);
                    work_col->at<T>(i_row) = new_val;
                }
            }
        };
        if (ftype == Bodo_FTypes::cumsum) {
            apply_oper([](T val1, T val2) -> T { return val1 + val2; });
        } else if (ftype == Bodo_FTypes::cumprod) {
            apply_oper([](T val1, T val2) -> T { return val1 * val2; });
        } else if (ftype == Bodo_FTypes::cummax) {
            apply_oper(
                [](T val1, T val2) -> T { return std::max(val1, val2); });
        } else if (ftype == Bodo_FTypes::cummin) {
            apply_oper(
                [](T val1, T val2) -> T { return std::min(val1, val2); });
        }
    }
}

/**
 * @brief MPI exscan implementation on nullable arrays.
 *
 * @tparam Tkey The type of the key column
 * @tparam T The type of the operation column
 * @tparam dtype The dtype of the operation column.
 * @param out_arrs The output arrays
 * @param cat_column The categorical column
 * @param in_table The input table
 * @param num_keys The number of key columns
 * @param k The index of the operation column
 * @param ftypes The types of the functions
 * @param func_offsets The offsets of the functions
 * @param is_parallel Whether the computation is parallel
 * @param skip_na_data Whether to skip dropna
 */
template <typename Tkey, typename T, int dtype>
void mpi_exscan_computation_nullable_T(
    std::vector<std::shared_ptr<array_info>>& out_arrs,
    std::shared_ptr<array_info> cat_column,
    std::shared_ptr<table_info> in_table, int64_t num_keys, int64_t k,
    int* ftypes, int* func_offsets, bool is_parallel, bool skip_na_data) {
    assert(cat_column->arr_type == bodo_array_type::CATEGORICAL);
    int64_t n_rows = in_table->nrows();
    int start = func_offsets[k];
    int end = func_offsets[k + 1];
    int n_oper = end - start;
    int64_t max_row_idx = cat_column->num_categories;
    bodo::vector<T> cumulative(max_row_idx * n_oper);
    for (int j = start; j != end; j++) {
        int ftype = ftypes[j];
        T value_init = -1;  // Dummy value set to avoid a compiler warning
        if (ftype == Bodo_FTypes::cumsum) {
            value_init = 0;
        } else if (ftype == Bodo_FTypes::cumprod) {
            value_init = 1;
        } else if (ftype == Bodo_FTypes::cummax) {
            value_init = std::numeric_limits<T>::min();
        } else if (ftype == Bodo_FTypes::cummin) {
            value_init = std::numeric_limits<T>::max();
        }
        for (int i_row = 0; i_row < max_row_idx; i_row++) {
            cumulative[i_row + max_row_idx * (j - start)] = value_init;
        }
    }
    bodo::vector<T> cumulative_recv = cumulative;
    bodo::vector<uint8_t> cumulative_mask, cumulative_mask_recv;
    // If we use skip_na_data then we do not need to keep track of
    // the previous values
    if (!skip_na_data) {
        cumulative_mask = bodo::vector<uint8_t>(max_row_idx * n_oper, 0);
        cumulative_mask_recv = bodo::vector<uint8_t>(max_row_idx * n_oper, 0);
    }
    std::shared_ptr<array_info> in_col = in_table->columns[k + num_keys];
    assert(in_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL);
    Tkey miss_idx = -1;
    for (int j = start; j != end; j++) {
        const std::shared_ptr<array_info>& work_col = out_arrs[j];
        int ftype = ftypes[j];
        auto apply_oper = [&](auto const& oper) -> void {
            const uint8_t* in_col_null_bitmask =
                (uint8_t*)
                    in_col->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>();
            uint8_t* work_col_null_bitmask = (uint8_t*)work_col->null_bitmask();
            for (int64_t i_row = 0; i_row < n_rows; i_row++) {
                Tkey idx =
                    cat_column->at<Tkey, bodo_array_type::CATEGORICAL>(i_row);
                if (idx == miss_idx) {
                    SetBitTo(work_col_null_bitmask, i_row, false);
                } else {
                    size_t pos = idx + max_row_idx * (j - start);
                    T val = in_col->at<T, bodo_array_type::NULLABLE_INT_BOOL>(
                        i_row);
                    bool bit_i = GetBit(in_col_null_bitmask, i_row);
                    T new_val = oper(val, cumulative[pos]);
                    bool bit_o = bit_i;
                    work_col->at<T>(i_row) = new_val;
                    if (skip_na_data) {
                        if (bit_i)
                            cumulative[pos] = new_val;
                    } else {
                        if (bit_i) {
                            if (cumulative_mask[pos] == 1) {
                                bit_o = false;
                            } else {
                                cumulative[pos] = new_val;
                            }
                        } else {
                            cumulative_mask[pos] = 1;
                        }
                    }
                    SetBitTo(work_col_null_bitmask, i_row, bit_o);
                }
            }
        };
        if (ftype == Bodo_FTypes::cumsum) {
            apply_oper([](T val1, T val2) -> T { return val1 + val2; });
        } else if (ftype == Bodo_FTypes::cumprod) {
            apply_oper([](T val1, T val2) -> T { return val1 * val2; });
        } else if (ftype == Bodo_FTypes::cummax) {
            apply_oper(
                [](T val1, T val2) -> T { return std::max(val1, val2); });
        } else if (ftype == Bodo_FTypes::cummin) {
            apply_oper(
                [](T val1, T val2) -> T { return std::min(val1, val2); });
        }
    }
    if (!is_parallel) {
        return;
    }
    MPI_Datatype mpi_typ = get_MPI_typ(dtype);
    for (int j = start; j != end; j++) {
        T* data_s = cumulative.data() + max_row_idx * (j - start);
        T* data_r = cumulative_recv.data() + max_row_idx * (j - start);
        int ftype = ftypes[j];
        if (ftype == Bodo_FTypes::cumsum) {
            HANDLE_MPI_ERROR(MPI_Exscan(data_s, data_r, max_row_idx, mpi_typ,
                                        MPI_SUM, MPI_COMM_WORLD),
                             "mpi_exscan_computation_nullable_T[cumsum]: MPI "
                             "error on MPI_Exscan:");
        } else if (ftype == Bodo_FTypes::cumprod) {
            HANDLE_MPI_ERROR(MPI_Exscan(data_s, data_r, max_row_idx, mpi_typ,
                                        MPI_PROD, MPI_COMM_WORLD),
                             "mpi_exscan_computation_nullable_T[cumprod]: MPI "
                             "error on MPI_Exscan:");
        } else if (ftype == Bodo_FTypes::cummax) {
            HANDLE_MPI_ERROR(MPI_Exscan(data_s, data_r, max_row_idx, mpi_typ,
                                        MPI_MAX, MPI_COMM_WORLD),
                             "mpi_exscan_computation_nullable_T[cummax]: MPI "
                             "error on MPI_Exscan:");
        } else if (ftype == Bodo_FTypes::cummin) {
            HANDLE_MPI_ERROR(MPI_Exscan(data_s, data_r, max_row_idx, mpi_typ,
                                        MPI_MIN, MPI_COMM_WORLD),
                             "mpi_exscan_computation_nullable_T[cummin]: MPI "
                             "error on MPI_Exscan:");
        }
    }
    if (!skip_na_data) {
        mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
        HANDLE_MPI_ERROR(
            MPI_Exscan(cumulative_mask.data(), cumulative_mask_recv.data(),
                       max_row_idx * n_oper, mpi_typ, MPI_MAX, MPI_COMM_WORLD),
            "mpi_exscan_computation_nullable_T: MPI error on MPI_Exscan:");
    }
    for (int j = start; j != end; j++) {
        std::shared_ptr<array_info> work_col = out_arrs[j];
        int ftype = ftypes[j];
        auto apply_oper = [&](auto const& oper) -> void {
            uint8_t* work_col_null_bitmask = (uint8_t*)work_col->null_bitmask();
            for (int64_t i_row = 0; i_row < n_rows; i_row++) {
                Tkey idx =
                    cat_column->at<Tkey, bodo_array_type::CATEGORICAL>(i_row);
                if (idx != miss_idx) {
                    size_t pos = idx + max_row_idx * (j - start);
                    T val = work_col->at<T>(i_row);
                    T new_val = oper(val, cumulative_recv[pos]);
                    work_col->at<T>(i_row) = new_val;
                    if (!skip_na_data && cumulative_mask_recv[pos] == 1) {
                        SetBitTo(work_col_null_bitmask, i_row, false);
                    }
                }
            }
        };
        if (ftype == Bodo_FTypes::cumsum) {
            apply_oper([](T val1, T val2) -> T { return val1 + val2; });
        } else if (ftype == Bodo_FTypes::cumprod) {
            apply_oper([](T val1, T val2) -> T { return val1 * val2; });
        } else if (ftype == Bodo_FTypes::cummax) {
            apply_oper(
                [](T val1, T val2) -> T { return std::max(val1, val2); });
        } else if (ftype == Bodo_FTypes::cummin) {
            apply_oper(
                [](T val1, T val2) -> T { return std::min(val1, val2); });
        }
    }
}

/**
 * @brief MPI exscan computation on all columns.
 *
 * @tparam Tkey The type of the key column
 * @param cat_column The categorical column
 * @param in_table The input table
 * @param num_keys The number of key columns
 * @param ftypes The types of the functions
 * @param func_offsets The offsets of the functions
 * @param is_parallel Whether the computation is parallel
 * @param skip_na_data Whether to skip dropna
 */
template <typename Tkey, typename T, Bodo_CTypes::CTypeEnum DType>
void mpi_exscan_computation_T(
    std::vector<std::shared_ptr<array_info>>& out_arrs,
    std::shared_ptr<array_info> cat_column,
    std::shared_ptr<table_info> in_table, int64_t num_keys, int64_t k,
    int* ftypes, int* func_offsets, bool is_parallel, bool skip_na_data) {
    const std::shared_ptr<array_info>& in_col = in_table->columns[k + num_keys];
    if (in_col->arr_type == bodo_array_type::NUMPY) {
        return mpi_exscan_computation_numpy_T<Tkey, T, DType>(
            out_arrs, std::move(cat_column), in_table, num_keys, k, ftypes,
            func_offsets, is_parallel, skip_na_data);
    } else {
        return mpi_exscan_computation_nullable_T<Tkey, T, DType>(
            out_arrs, std::move(cat_column), in_table, num_keys, k, ftypes,
            func_offsets, is_parallel, skip_na_data);
    }
}

/**
 * @brief MPI exscan implementation on a particular key type.
 *
 * @tparam Tkey The type of the key column
 * @param cat_column The categorical column
 * @param in_table The input table
 * @param num_keys The number of key columns
 * @param ftypes The types of the functions
 * @param func_offsets The offsets of the functions
 * @param is_parallel Whether the computation is parallel
 * @param skip_na_data Whether to skip dropna
 * @param return_key Whether to return the key column
 * @param return_index Whether to return the index column
 * @param use_sql_rules Whether to use SQL rules in allocation
 * @return std::shared_ptr<table_info> The output table
 */
template <typename Tkey>
std::shared_ptr<table_info> mpi_exscan_computation_Tkey(
    std::shared_ptr<array_info> cat_column,
    std::shared_ptr<table_info> in_table, int64_t num_keys, int* ftypes,
    int* func_offsets, bool is_parallel, bool skip_na_data, bool return_key,
    bool return_index, bool use_sql_rules) {
    std::vector<std::shared_ptr<array_info>> out_arrs;
    // We do not return the keys in output in the case of cumulative operations.
    int64_t n_rows = in_table->nrows();
    int return_index_i = return_index;
    int k = 0;
    for (uint64_t i = num_keys; i < in_table->ncols() - return_index_i;
         i++, k++) {
        const std::shared_ptr<array_info>& col = in_table->columns[i];
        int start = func_offsets[k];
        int end = func_offsets[k + 1];
        for (int j = start; j != end; j++) {
            std::shared_ptr<array_info> out_col =
                alloc_array_top_level(n_rows, 1, 1, col->arr_type, col->dtype,
                                      -1, 0, col->num_categories);
            int ftype = ftypes[j];
            aggfunc_output_initialize(out_col, ftype, use_sql_rules);
            out_arrs.push_back(std::move(out_col));
        }
    }
    // Since each column can have different data type and MPI_Exscan can only do
    // one type at a time. thus we have an iteration over the columns of the
    // input table. But we can consider the various cumsum / cumprod / cummax /
    // cummin in turn.
    k = 0;
    // macro to reduce code duplication
#ifndef MPI_EXSCAN_COMPUTATION_T_CALL
#define MPI_EXSCAN_COMPUTATION_T_CALL(CTYPE)                                \
    if (dtype == CTYPE) {                                                   \
        mpi_exscan_computation_T<Tkey, typename dtype_to_type<CTYPE>::type, \
                                 CTYPE>(out_arrs, cat_column, in_table,     \
                                        num_keys, k, ftypes, func_offsets,  \
                                        is_parallel, skip_na_data);         \
    }
#endif
    for (uint64_t i = num_keys; i < in_table->ncols() - return_index_i;
         i++, k++) {
        std::shared_ptr<array_info> col = in_table->columns[i];
        const Bodo_CTypes::CTypeEnum dtype = col->dtype;
        MPI_EXSCAN_COMPUTATION_T_CALL(Bodo_CTypes::INT8)
        MPI_EXSCAN_COMPUTATION_T_CALL(Bodo_CTypes::UINT8)
        MPI_EXSCAN_COMPUTATION_T_CALL(Bodo_CTypes::INT16)
        MPI_EXSCAN_COMPUTATION_T_CALL(Bodo_CTypes::UINT16)
        MPI_EXSCAN_COMPUTATION_T_CALL(Bodo_CTypes::INT32)
        MPI_EXSCAN_COMPUTATION_T_CALL(Bodo_CTypes::UINT32)
        MPI_EXSCAN_COMPUTATION_T_CALL(Bodo_CTypes::INT64)
        MPI_EXSCAN_COMPUTATION_T_CALL(Bodo_CTypes::UINT64)
        MPI_EXSCAN_COMPUTATION_T_CALL(Bodo_CTypes::FLOAT32)
        MPI_EXSCAN_COMPUTATION_T_CALL(Bodo_CTypes::FLOAT64)
    }
    if (return_index) {
        out_arrs.push_back(copy_array(in_table->columns.back()));
    }
#undef MPI_EXSCAN_COMPUTATION_T_CALL

    return std::make_shared<table_info>(out_arrs);
}

std::shared_ptr<table_info> mpi_exscan_computation(
    std::shared_ptr<array_info> cat_column,
    std::shared_ptr<table_info> in_table, int64_t num_keys, int* ftypes,
    int* func_offsets, bool is_parallel, bool skip_na_data, bool return_key,
    bool return_index, bool use_sql_rules) {
    tracing::Event ev("mpi_exscan_computation", is_parallel);
    const Bodo_CTypes::CTypeEnum dtype = cat_column->dtype;
    // macro to reduce code duplication
#ifndef MPI_EXSCAN_COMPUTATION_CALL
#define MPI_EXSCAN_COMPUTATION_CALL(CTYPE)                                   \
    if (dtype == CTYPE) {                                                    \
        return mpi_exscan_computation_Tkey<                                  \
            typename dtype_to_type<CTYPE>::type>(                            \
            std::move(cat_column), in_table, num_keys, ftypes, func_offsets, \
            is_parallel, skip_na_data, return_key, return_index,             \
            use_sql_rules);                                                  \
    }
#endif
    MPI_EXSCAN_COMPUTATION_CALL(Bodo_CTypes::INT8)
    MPI_EXSCAN_COMPUTATION_CALL(Bodo_CTypes::UINT8)
    MPI_EXSCAN_COMPUTATION_CALL(Bodo_CTypes::INT16)
    MPI_EXSCAN_COMPUTATION_CALL(Bodo_CTypes::UINT16)
    MPI_EXSCAN_COMPUTATION_CALL(Bodo_CTypes::INT32)
    MPI_EXSCAN_COMPUTATION_CALL(Bodo_CTypes::UINT32)
    MPI_EXSCAN_COMPUTATION_CALL(Bodo_CTypes::INT64)
    MPI_EXSCAN_COMPUTATION_CALL(Bodo_CTypes::UINT64)
    // If we haven't returned in the macro we didn't find a match.
    throw std::runtime_error(
        "MPI EXSCAN groupby implementation failed to find a matching "
        "dtype");
#undef MPI_EXSCAN_COMPUTATION_CALL
}
