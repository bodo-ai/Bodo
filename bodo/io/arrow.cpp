
// Here we export parquet read and write functions of the
// bodo.io.arrow_cpp extension module to Python

#include <Python.h>
#include <arrow/filesystem/filesystem.h>
#include <object.h>
#include "../libs/_bodo_common.h"
#include "../libs/_theta_sketches.h"
#include "_s3_reader.h"
#include "arrow_reader.h"

table_info* arrow_reader_read_py_entry(ArrowReader* reader, bool* is_last_out,
                                       uint64_t* total_rows_out,
                                       bool produce_output);

void arrow_reader_del_py_entry(ArrowReader* reader);

PyObject* fetch_parquet_frags_metadata(PyObject* self, PyObject* const* args,
                                       Py_ssize_t nargs);

PyObject* fetch_parquet_frag_row_counts(PyObject* self, PyObject* const* args,
                                        Py_ssize_t nargs);

// --------- functions defined in parquet_reader.cpp ---------
table_info* pq_read_py_entry(PyObject* path, bool parallel,
                             PyObject* expr_filters, PyObject* storage_options,
                             PyObject* pyarrow_schema, int64_t tot_rows_to_read,
                             int32_t* _selected_cols, int32_t num_selected_cols,
                             int32_t* _is_nullable, int32_t* selected_part_cols,
                             int32_t* part_cols_cat_dtype,
                             int32_t num_partition_cols,
                             int32_t* str_as_dict_cols,
                             int32_t num_str_as_dict_cols,
                             int64_t* total_rows_out, bool input_file_name_col,
                             bool use_hive);

ArrowReader* pq_reader_init_py_entry(
    PyObject* path, bool parallel, PyObject* expr_filters,
    PyObject* storage_options, PyObject* pyarrow_schema,
    int64_t tot_rows_to_read, int32_t* _selected_fields,
    int32_t num_selected_fields, int32_t* _is_nullable,
    int32_t* selected_part_cols, int32_t* part_cols_cat_dtype,
    int32_t num_partition_cols, int32_t* str_as_dict_cols,
    int32_t num_str_as_dict_cols, bool input_file_name_col, int64_t batch_size,
    bool use_hive, int64_t op_id);

// --------- functions defined in iceberg_parquet_reader.cpp --------
table_info* iceberg_pq_read_py_entry(
    PyObject* catalog, const char* table_id, bool parallel,
    int64_t tot_rows_to_read, PyObject* iceberg_filter_str,
    const char* expr_filter_f_str_, PyObject* filter_scalars,
    int32_t* _selected_fields, int32_t num_selected_fields,
    int32_t* _is_nullable, PyObject* pyarrow_schema, int32_t* str_as_dict_cols,
    int32_t num_str_as_dict_cols, bool create_dict_from_string,
    bool is_merge_into_cow, int64_t* snapshot_id_ptr, int64_t* total_rows_out,
    PyObject** file_list_ptr);

ArrowReader* iceberg_pq_reader_init_py_entry(
    PyObject* catalog, const char* table_id, bool parallel,
    int64_t tot_rows_to_read, PyObject* iceberg_filter_str,
    const char* expr_filter_f_str, PyObject* filter_scalars,
    int32_t* _selected_fields, int32_t num_selected_fields,
    int32_t* _is_nullable, PyObject* pyarrow_schema, int32_t* _str_as_dict_cols,
    int32_t num_str_as_dict_cols, bool create_dict_from_string,
    int64_t batch_size, int64_t op_id);

// --------- function defined in snowflake_reader.cpp ---------
table_info* snowflake_read_py_entry(
    const char* query, const char* conn, bool parallel, bool is_independent,
    PyObject* arrow_schema, int64_t n_fields, int32_t* _is_nullable,
    int32_t* _str_as_dict_cols, int32_t num_str_as_dict_cols,
    int64_t* total_nrows, bool _only_length_query, bool _is_select_query,
    bool downcast_decimal_to_double);

ArrowReader* snowflake_reader_init_py_entry(
    const char* query, const char* conn, bool parallel, bool is_independent,
    PyObject* arrow_schema, int64_t n_fields, int32_t* _is_nullable,
    int32_t num_str_as_dict_cols, int32_t* _str_as_dict_cols,
    int64_t* total_nrows, bool _only_length_query, bool _is_select_query,
    bool downcast_decimal_to_double, int64_t batch_size, int64_t op_id);

// --------- functions defined in parquet_write.cpp ---------
int64_t pq_write_py_entry(const char* _path_name, table_info* table,
                          array_info* col_names_arr, const char* metadata,
                          const char* compression, bool is_parallel,
                          const char* bucket_region, int64_t row_group_size,
                          const char* prefix, bool convert_timedelta_to_int64,
                          const char* tz, bool downcast_time_ns_to_us,
                          bool create_dir);

void pq_write_create_dir_py_entry(const char* _path_name);

void pq_write_partitioned_py_entry(
    const char* _path_name, table_info* in_table, array_info* in_col_names_arr,
    array_info* in_col_names_arr_no_partitions, table_info* in_categories_table,
    int* partition_cols_idx, int num_partition_cols, const char* compression,
    bool is_parallel, const char* bucket_region, int64_t row_group_size,
    const char* prefix, const char* tz);

// ---------- functions defined in iceberg_parquet_write.cpp ----
PyObject* iceberg_pq_write_py_entry(
    const char* table_data_loc, table_info* in_table,
    array_info* in_col_names_arr, PyObject* partition_spec,
    PyObject* sort_order, const char* compression, bool is_parallel,
    const char* bucket_region, int64_t row_group_size,
    const char* iceberg_metadata, PyObject* iceberg_arrow_schema_py,
    PyObject* arrow_fs, void* sketches_ptr);

PyMethodDef fetch_frags_method_def = {"fetch_parquet_frags_metadata",
                                      (PyCFunction)fetch_parquet_frags_metadata,
                                      METH_FASTCALL, ""};
PyMethodDef fetch_row_count_method_def = {
    "fetch_parquet_frag_row_counts", (PyCFunction)fetch_parquet_frag_row_counts,
    METH_FASTCALL, ""};

PyMODINIT_FUNC PyInit_arrow_cpp(void) {
    PyObject* m;
    MOD_DEF(m, "arrow_cpp", "No docs", nullptr);
    if (m == nullptr) {
        return nullptr;
    }

    bodo_common_init();

    SetAttrStringFromVoidPtr(m, pq_read_py_entry);
    SetAttrStringFromVoidPtr(m, pq_reader_init_py_entry);

    SetAttrStringFromVoidPtr(m, iceberg_pq_read_py_entry);
    SetAttrStringFromVoidPtr(m, iceberg_pq_reader_init_py_entry);

    SetAttrStringFromVoidPtr(m, pq_write_py_entry);
    SetAttrStringFromVoidPtr(m, pq_write_create_dir_py_entry);
    SetAttrStringFromVoidPtr(m, iceberg_pq_write_py_entry);
    SetAttrStringFromVoidPtr(m, pq_write_partitioned_py_entry);
    SetAttrStringFromVoidPtr(m, snowflake_read_py_entry);
    SetAttrStringFromVoidPtr(m, snowflake_reader_init_py_entry);

    SetAttrStringFromVoidPtr(m, arrow_reader_read_py_entry);
    SetAttrStringFromVoidPtr(m, arrow_reader_del_py_entry);

    PyObject_SetAttrString(m, "fetch_parquet_frags_metadata",
                           PyCFunction_New(&fetch_frags_method_def, NULL));
    PyObject_SetAttrString(m, "fetch_parquet_frag_row_counts",
                           PyCFunction_New(&fetch_row_count_method_def, NULL));

    return m;
}
