// Copyright (C) 2021 Bodo Inc. All rights reserved.

// Here we export parquet read and write functions of the
// bodo.io.arrow_cpp extension module to Python

#include <Python.h>
#include "../libs/_bodo_common.h"

// --------- functions defined in parquet_reader.cpp ---------
table_info* pq_read_py_entry(
    PyObject* path, bool parallel, PyObject* dnf_filters,
    PyObject* expr_filters, PyObject* storage_options, PyObject* pyarrow_schema,
    int64_t tot_rows_to_read, int32_t* _selected_cols,
    int32_t num_selected_cols, int32_t* _is_nullable,
    int32_t* selected_part_cols, int32_t* part_cols_cat_dtype,
    int32_t num_partition_cols, int32_t* str_as_dict_cols,
    int32_t num_str_as_dict_cols, int64_t* total_rows_out,
    bool input_file_name_col, bool use_hive);

// --------- functions defined in iceberg_parquet_reader.cpp --------
table_info* iceberg_pq_read_py_entry(
    const char* conn, const char* database_schema, const char* table_name,
    bool parallel, int64_t tot_rows_to_read, PyObject* dnf_filters,
    PyObject* expr_filters, int32_t* _selected_fields,
    int32_t num_selected_fields, int32_t* _is_nullable,
    PyObject* pyarrow_schema, int32_t* str_as_dict_cols,
    int32_t num_str_as_dict_cols, bool is_merge_into_cow,
    int64_t* total_rows_out, PyObject** file_list_ptr,
    int64_t* snapshot_id_ptr);

// --------- function defined in snowflake_reader.cpp ---------
table_info* snowflake_read_py_entry(
    const char* query, const char* conn, bool parallel, bool is_independent,
    PyObject* arrow_schema, int64_t n_fields, int32_t* _is_nullable,
    int32_t* _str_as_dict_cols, int32_t num_str_as_dict_cols,
    int32_t* _allow_unsafe_dt_to_ts_cast_cols,
    int32_t num_allow_unsafe_dt_to_ts_cast_cols, int64_t* total_nrows,
    bool _only_length_query, bool _is_select_query);

// --------- functions defined in parquet_write.cpp ---------
int64_t pq_write_py_entry(const char* _path_name, table_info* table,
                          array_info* col_names_arr, array_info* index,
                          bool write_index, const char* metadata,
                          const char* compression, bool is_parallel,
                          bool write_rangeindex_to_metadata, const int ri_start,
                          const int ri_stop, const int ri_step,
                          const char* idx_name, const char* bucket_region,
                          int64_t row_group_size, const char* prefix,
                          bool convert_timedelta_to_int64, const char* tz,
                          bool downcast_time_ns_to_us);

void pq_write_partitioned_py_entry(
    const char* _path_name, table_info* in_table, array_info* in_col_names_arr,
    array_info* in_col_names_arr_no_partitions, table_info* in_categories_table,
    int* partition_cols_idx, int num_partition_cols, const char* compression,
    bool is_parallel, const char* bucket_region, int64_t row_group_size,
    const char* prefix, const char* tz);

// ---------- functions defined in iceberg_parquet_write.cpp ----
PyObject* iceberg_pq_write_py_entry(
    const char* table_data_loc, table_info* table, array_info* col_names_arr,
    PyObject* partition_spec, PyObject* sort_order, const char* compression,
    bool is_parallel, const char* bucket_region, int64_t row_group_size,
    char* iceberg_metadata, PyObject* iceberg_arrow_schema_py);

PyMODINIT_FUNC PyInit_arrow_cpp(void) {
    PyObject* m;
    MOD_DEF(m, "arrow_cpp", "No docs", NULL);
    if (m == NULL)
        return NULL;

    bodo_common_init();

    SetAttrStringFromVoidPtr(m, pq_read_py_entry);
    SetAttrStringFromVoidPtr(m, iceberg_pq_read_py_entry);
    SetAttrStringFromVoidPtr(m, pq_write_py_entry);
    SetAttrStringFromVoidPtr(m, iceberg_pq_write_py_entry);
    SetAttrStringFromVoidPtr(m, pq_write_partitioned_py_entry);
    SetAttrStringFromVoidPtr(m, snowflake_read_py_entry);

    return m;
}
