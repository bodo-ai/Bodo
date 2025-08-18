#include <Python.h>

extern "C" {
PyMODINIT_FUNC PyInit_hdist(void);
PyMODINIT_FUNC PyInit_hstr_ext(void);
PyMODINIT_FUNC PyInit_decimal_ext(void);
PyMODINIT_FUNC PyInit_hdatetime_ext(void);
PyMODINIT_FUNC PyInit_hio(void);
PyMODINIT_FUNC PyInit_array_ext(void);
PyMODINIT_FUNC PyInit_s3_reader(void);
PyMODINIT_FUNC PyInit_hdfs_reader(void);
#ifndef NO_HDF5
PyMODINIT_FUNC PyInit__hdf5(void);
#endif
PyMODINIT_FUNC PyInit_arrow_cpp(void);
PyMODINIT_FUNC PyInit_csv_cpp(void);
PyMODINIT_FUNC PyInit_json_cpp(void);
PyMODINIT_FUNC PyInit_stream_join_cpp(void);
PyMODINIT_FUNC PyInit_stream_sort_cpp(void);
PyMODINIT_FUNC PyInit_memory_budget_cpp(void);
PyMODINIT_FUNC PyInit_stream_groupby_cpp(void);
PyMODINIT_FUNC PyInit_stream_window_cpp(void);
PyMODINIT_FUNC PyInit_stream_dict_encoding_cpp(void);
PyMODINIT_FUNC PyInit_table_builder_cpp(void);
PyMODINIT_FUNC PyInit_fft_cpp(void);
#ifdef BUILD_WITH_V8
PyMODINIT_FUNC PyInit_javascript_udf_cpp(void);
#endif
PyMODINIT_FUNC PyInit_query_profile_collector_cpp(void);
#ifdef IS_TESTING
PyMODINIT_FUNC PyInit_test_cpp(void);
#endif
PyMODINIT_FUNC PyInit_plan_optimizer(void);
}  // extern "C"
