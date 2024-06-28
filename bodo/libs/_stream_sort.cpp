#include "_bodo_common.h"
#include "_chunked_table_builder.h"
#include "_dict_builder.h"

struct StreamSortState {};

StreamSortState* stream_sort_state_init_py_entry(int8_t* arr_c_types,
                                                 int8_t* arr_array_types,
                                                 int n_arrs,
                                                 int64_t chunk_size) {
    return nullptr;
}

bool stream_sort_build_consume_batch_py_entry(StreamSortState* state,
                                              table_info* in_table,
                                              bool is_last) {
    return true;
}

table_info* stream_sort_product_output_batch_py_entry(StreamSortState* state,
                                                      bool produce_output,
                                                      bool* out_is_last) {
    *out_is_last = true;
    return nullptr;
}

void delete_stream_sort_state(StreamSortState* state) { delete state; }

PyMODINIT_FUNC PyInit_stream_sort_cpp(void) {
    PyObject* m;
    MOD_DEF(m, "stream_sort_cpp", "No docs", NULL);
    if (m == NULL) {
        return NULL;
    }

    bodo_common_init();

    SetAttrStringFromVoidPtr(m, stream_sort_state_init_py_entry);
    SetAttrStringFromVoidPtr(m, stream_sort_build_consume_batch_py_entry);
    SetAttrStringFromVoidPtr(m, stream_sort_product_output_batch_py_entry);
    SetAttrStringFromVoidPtr(m, delete_stream_sort_state);
    return m;
}
