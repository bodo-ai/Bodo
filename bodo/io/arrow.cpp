// Copyright (C) 2021 Bodo Inc. All rights reserved.

// Here we export parquet read and write functions of the
// bodo.io.arrow_cpp extension module to Python

#include <Python.h>
#include "../libs/_bodo_common.h"

// --------- functions defined in parquet_reader.cpp ---------
table_info* pq_read(PyObject* path, bool parallel, PyObject* dnf_filters,
                    PyObject* expr_filters, PyObject* storage_options,
                    int64_t tot_rows_to_read, int32_t* selected_cols,
                    int32_t num_selected_cols, int32_t* is_nullable,
                    int32_t* selected_part_cols, int32_t* part_cols_cat_dtype,
                    int32_t num_partition_cols, int32_t* str_as_dict_cols,
                    int32_t num_str_as_dict_cols, int64_t* total_rows_out,
                    bool input_file_name_col);

// --------- functions defined in iceberg_parquet_reader.cpp --------
table_info* iceberg_pq_read(const char* conn, const char* database_schema,
                            const char* table_name, bool parallel,
                            int64_t tot_rows_to_read, PyObject* dnf_filters,
                            PyObject* expr_filters, int32_t* selected_fields,
                            int32_t num_selected_fields, int32_t* is_nullable,
                            PyObject* pyarrow_table_schema,
                            int32_t* str_as_dict_cols,
                            int32_t num_str_as_dict_cols,
                            int64_t* total_rows_out);

// --------- functions defined in parquet_write.cpp ---------
int64_t pq_write_py_entry(const char* filename, const table_info* table,
                          const array_info* col_names, const array_info* index,
                          bool write_index, const char* metadata,
                          const char* compression, bool parallel,
                          bool write_rangeindex_to_metadata, const int start,
                          const int stop, const int step, const char* name,
                          const char* bucket_region, int64_t row_group_size,
                          const char* prefix, const char* tz);

void pq_write_partitioned(const char* _path_name, table_info* table,
                          const array_info* col_names_arr,
                          const array_info* col_names_arr_no_partitions,
                          table_info* categories_table, int* partition_cols_idx,
                          int num_partition_cols, const char* compression,
                          bool is_parallel, const char* bucket_region,
                          int64_t row_group_size, const char* prefix);

// ---------- functions defined in iceberg_parquet_write.cpp ----

PyObject* iceberg_pq_write_py_entry(
    const char* table_data_loc, table_info* table,
    const array_info* col_names_arr, PyObject* partition_spec,
    PyObject* sort_order, const char* compression, bool is_parallel,
    const char* bucket_region, int64_t row_group_size, char* iceberg_metadata);

// --------- function defined in snowflake_reader.cpp ---------
table_info* snowflake_read(const char* query, const char* conn, bool parallel,
                           int64_t n_fields, int32_t* is_nullable,
                           int32_t* str_as_dict_cols,
                           int32_t num_str_as_dict_cols, int64_t* total_nrows);

PyMODINIT_FUNC PyInit_arrow_cpp(void) {
    PyObject* m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "arrow_cpp", "No docs", -1, NULL,
    };
    m = PyModule_Create(&moduledef);
    if (m == NULL) return NULL;

    bodo_common_init();

    PyObject_SetAttrString(m, "pq_read", PyLong_FromVoidPtr((void*)(&pq_read)));
    PyObject_SetAttrString(m, "iceberg_pq_read",
                           PyLong_FromVoidPtr((void*)(&iceberg_pq_read)));
    PyObject_SetAttrString(m, "pq_write",
                           PyLong_FromVoidPtr((void*)(&pq_write_py_entry)));
    PyObject_SetAttrString(
        m, "iceberg_pq_write",
        PyLong_FromVoidPtr((void*)(&iceberg_pq_write_py_entry)));
    PyObject_SetAttrString(m, "pq_write_partitioned",
                           PyLong_FromVoidPtr((void*)(&pq_write_partitioned)));
    PyObject_SetAttrString(m, "snowflake_read",
                           PyLong_FromVoidPtr((void*)(&snowflake_read)));
    PyObject_SetAttrString(m, "get_stats_alloc",
                           PyLong_FromVoidPtr((void*)(&get_stats_alloc)));
    PyObject_SetAttrString(m, "get_stats_free",
                           PyLong_FromVoidPtr((void*)(&get_stats_free)));
    PyObject_SetAttrString(m, "get_stats_mi_alloc",
                           PyLong_FromVoidPtr((void*)(&get_stats_mi_alloc)));
    PyObject_SetAttrString(m, "get_stats_mi_free",
                           PyLong_FromVoidPtr((void*)(&get_stats_mi_free)));

    return m;
}
