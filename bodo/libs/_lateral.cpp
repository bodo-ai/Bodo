// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include "_lateral.h"
#include <Python.h>
#include "_array_utils.h"
#include "_bodo_common.h"

std::unique_ptr<table_info> lateral_flatten_array(
    const std::unique_ptr<table_info> &in_table, int64_t *n_rows,
    bool output_seq, bool output_key, bool output_path, bool output_index,
    bool output_value, bool output_this, bodo::IBufferPool *const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    // Create the table where we will place all of the out columns in.
    std::unique_ptr<table_info> out_table = std::make_unique<table_info>();

    // Find the column to be exploded and calculate the number of times each
    // row should be duplicated.
    std::shared_ptr<array_info> explode_arr = in_table->columns[0];
    size_t n_inner_arrs = explode_arr->length;
    offset_t *offset_buffer =
        (offset_t *)(explode_arr->buffers[0]->mutable_data() +
                     explode_arr->offset);
    size_t exploded_size = offset_buffer[n_inner_arrs] - offset_buffer[0];
    std::vector<int64_t> rows_to_copy(exploded_size);
    size_t write_idx = 0;
    for (size_t i = 0; i < n_inner_arrs; i++) {
        size_t n_rows = offset_buffer[i + 1] - offset_buffer[i];
        for (size_t j = 0; j < n_rows; j++) {
            rows_to_copy[write_idx] = i;
            write_idx++;
        }
    }

    // If we need to output the index, then create a column with the
    // indices within each inner array.
    if (output_index) {
        std::shared_ptr<array_info> idx_arr =
            alloc_numpy(exploded_size, Bodo_CTypes::INT64);
        offset_t *offset_buffer =
            (offset_t *)(explode_arr->buffers[0]->mutable_data() +
                         explode_arr->offset);
        for (size_t i = 0; i < explode_arr->length; i++) {
            offset_t start_offset = offset_buffer[i];
            offset_t end_offset = offset_buffer[i + 1];
            for (offset_t j = start_offset; j < end_offset; j++) {
                getv<int64_t>(idx_arr, j) = j - start_offset;
            }
        }
        out_table->columns.push_back(idx_arr);
    }

    // If we need to output the array, then append the inner array
    // from the exploded column
    if (output_value) {
        out_table->columns.push_back(explode_arr->child_arrays[0]);
    }

    // If we need to output the 'this' column, then repeat the replication
    // procedure on the input column
    if (output_this) {
        std::shared_ptr<array_info> old_col = in_table->columns[0];
        std::shared_ptr<array_info> new_col =
            RetrieveArray_SingleColumn(old_col, rows_to_copy, false, pool, mm);
        out_table->columns.push_back(new_col);
    }

    // For each column in the table (except the one to be exploded) create
    // a copy with the rows duplicated the specified number of times and
    // append it to the output table.
    for (size_t i = 1; i < in_table->columns.size(); i++) {
        std::shared_ptr<array_info> old_col = in_table->columns[i];
        std::shared_ptr<array_info> new_col =
            RetrieveArray_SingleColumn(old_col, rows_to_copy, false, pool, mm);
        out_table->columns.push_back(new_col);
    }

    *n_rows = exploded_size;
    return out_table;
}

std::unique_ptr<table_info> lateral_flatten_map(
    const std::unique_ptr<table_info> &in_table, int64_t *n_rows,
    bool output_seq, bool output_key, bool output_path, bool output_index,
    bool output_value, bool output_this, bodo::IBufferPool *const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    // Create the table where we will place all of the out columns in.
    std::unique_ptr<table_info> out_table = std::make_unique<table_info>();

    // Find the column to be exploded and calculate the number of times each
    // row should be duplicated.
    std::shared_ptr<array_info> explode_arr = in_table->columns[0];
    size_t n_inner_arrs = explode_arr->length;
    offset_t *offset_buffer =
        (offset_t *)(explode_arr->buffers[0]->mutable_data() +
                     explode_arr->offset);
    size_t exploded_size = offset_buffer[n_inner_arrs] - offset_buffer[0];
    std::vector<int64_t> rows_to_copy(exploded_size);
    size_t write_idx = 0;
    for (size_t i = 0; i < n_inner_arrs; i++) {
        size_t n_rows = offset_buffer[i + 1] - offset_buffer[i];
        for (size_t j = 0; j < n_rows; j++) {
            rows_to_copy[write_idx] = i;
            write_idx++;
        }
    }

    // If we need to output the key, then add the first column
    // of the inner struct array.
    if (output_key) {
        std::shared_ptr<array_info> key_arr =
            explode_arr->child_arrays[0]->child_arrays[0];
        out_table->columns.push_back(key_arr);
    }

    // If we need to output the key, then add the second column
    // of the inner struct array.
    if (output_value) {
        std::shared_ptr<array_info> value_arr =
            explode_arr->child_arrays[0]->child_arrays[1];
        out_table->columns.push_back(value_arr);
    }

    // If we need to output the 'this' column, then repeat the replication
    // procedure on the input column
    if (output_this) {
        std::shared_ptr<array_info> old_col = in_table->columns[0];
        std::shared_ptr<array_info> new_col =
            RetrieveArray_SingleColumn(old_col, rows_to_copy, false, pool, mm);
        out_table->columns.push_back(new_col);
    }

    // For each column in the table (except the one to be exploded) create
    // a copy with the rows duplicated the specified number of times and
    // append it to the output table.
    for (size_t i = 1; i < in_table->columns.size(); i++) {
        std::shared_ptr<array_info> old_col = in_table->columns[i];
        std::shared_ptr<array_info> new_col =
            RetrieveArray_SingleColumn(old_col, rows_to_copy, false, pool, mm);
        out_table->columns.push_back(new_col);
    }

    *n_rows = exploded_size;
    return out_table;
}

std::unique_ptr<table_info> lateral_flatten_struct(
    const std::unique_ptr<table_info> &in_table, int64_t *n_rows,
    bool output_seq, bool output_key, bool output_path, bool output_index,
    bool output_value, bool output_this, bodo::IBufferPool *const pool,
    std::shared_ptr<::arrow::MemoryManager> mm) {
    throw std::runtime_error("lateral_flatten_struct: currently unsupported");
}

// Python entrypoint for lateral_flatten
table_info *lateral_flatten_py_entrypt(table_info *in_table, int64_t *n_rows,
                                       bool output_seq, bool output_key,
                                       bool output_path, bool output_index,
                                       bool output_value, bool output_this,
                                       bool json_mode) {
    try {
        std::unique_ptr<table_info> tab(in_table);
        std::unique_ptr<table_info> result;
        if (json_mode) {
            switch (tab->columns[0]->arr_type) {
                case bodo_array_type::ARRAY_ITEM: {
                    if (tab->columns[0]->child_arrays[0]->arr_type !=
                        bodo_array_type::STRUCT) {
                        throw std::runtime_error(
                            "lateral flatten with json mode requires a map "
                            "array or struct array as an input");
                    }
                    result = lateral_flatten_map(
                        tab, n_rows, output_seq, output_key, output_path,
                        output_index, output_value, output_this);
                    break;
                }
                case bodo_array_type::STRUCT: {
                    result = lateral_flatten_struct(
                        tab, n_rows, output_seq, output_key, output_path,
                        output_index, output_value, output_this);
                    break;
                }
                default: {
                    throw std::runtime_error(
                        "lateral flatten with json mode requires a map array "
                        "or struct array as an input");
                }
            }
        } else {
            if (tab->columns[0]->arr_type != bodo_array_type::ARRAY_ITEM) {
                throw std::runtime_error(
                    "lateral flatten with array mode requires an array item "
                    "array as an input");
            }
            result = lateral_flatten_array(tab, n_rows, output_seq, output_key,
                                           output_path, output_index,
                                           output_value, output_this);
        }
        table_info *raw_result = result.release();
        return raw_result;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

PyMODINIT_FUNC PyInit_lateral(void) {
    PyObject *m;
    MOD_DEF(m, "lateral", "No docs", NULL);
    if (m == NULL) {
        return NULL;
    }

    bodo_common_init();

    SetAttrStringFromVoidPtr(m, lateral_flatten_py_entrypt);

    return m;
}
