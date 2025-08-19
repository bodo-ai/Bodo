#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <algorithm>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include <arrow/python/pyarrow.h>
#include <numpy/arrayobject.h>

#include "_bodo_common.h"
#include "_bodo_to_arrow.h"
#include "_stl.h"

#include <boost/lexical_cast.hpp>

#include "_str_decode.cpp"

extern "C" {

// Map of integers to hex values. Note we use an array because the keys are 0-15
static constexpr char hex_values[] = {'0', '1', '2', '3', '4', '5', '6', '7',
                                      '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};

// Copied _PyLong_DigitValue from CPython internals (seems not exported in
// Python 3.11, causing compilation errors).
// https://github.com/python/cpython/blob/869f177b5cd5c2a2a015e4658cbbb0e9566210f7/Objects/longobject.c#L2203
unsigned char _DigitValue[256] = {
    37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
    37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
    37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 0,  1,  2,  3,  4,  5,  6,  7,  8,
    9,  37, 37, 37, 37, 37, 37, 37, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 37, 37, 37,
    37, 37, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
    37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
    37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
    37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
    37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
    37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
    37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37,
    37, 37, 37, 37, 37, 37, 37, 37, 37,
};

void* init_string_const(char* in_str, int64_t size);
void dtor_str_arr_split_view(str_arr_split_view_payload* in_str_arr,
                             int64_t size, void* in);
void str_arr_split_view_alloc(str_arr_split_view_payload* out_view,
                              int64_t num_items, int64_t num_offsets);
void str_arr_split_view_impl(str_arr_split_view_payload* out_view,
                             int64_t n_strs, offset_t* offsets, char* data,
                             char* null_bitmap, char sep);
const char* get_c_str(std::string* s);

int64_t str_to_int64(char* data, int64_t length);
uint64_t str_to_uint64(char* data, int64_t length);
int64_t str_to_int64_base(char* data, int64_t length, int64_t base);
double str_to_float64(std::string* str);
float str_to_float32(std::string* str);
int64_t get_str_len(std::string* str);

void* pd_pyarrow_array_from_string_array(array_info* str_arr,
                                         const int is_bytes);

void setitem_string_array(offset_t* offsets, char* data, uint64_t n_bytes,
                          char* str, int64_t len, int kind, int is_ascii,
                          int64_t index);
void setitem_binary_array(offset_t* offsets, char* data, uint64_t n_bytes,
                          char* str, int64_t len, int64_t index);
int64_t get_utf8_size(char* str, int64_t len, int kind);

void set_string_array_range(offset_t* out_offsets, char* out_data,
                            offset_t* in_offsets, char* in_data,
                            int64_t start_str_ind, int64_t start_chars_ind,
                            int64_t num_strs, int64_t num_chars);
void convert_len_arr_to_offset32(uint32_t* offsets, int64_t num_strs);
void convert_len_arr_to_offset(uint32_t* lens, offset_t* offsets,
                               uint64_t num_strs);

int str_arr_to_int64(int64_t* out, offset_t* offsets, char* data,
                     int64_t index);
int str_arr_to_float64(double* out, offset_t* offsets, char* data,
                       int64_t index);

void str_from_float32(char* s, float in);
void str_from_float64(char* s, double in);
void inplace_int64_to_str(char* str, int64_t l, int64_t value);

void del_str(std::string* in_str);
npy_intp array_size(PyArrayObject* arr);
void* array_getptr1(PyArrayObject* arr, npy_intp ind);
void array_setitem(PyArrayObject* arr, char* p, PyObject* s);
void bool_arr_to_bitmap(uint8_t* bitmap_arr, uint8_t* bool_arr, int64_t n);
void mask_arr_to_bitmap(uint8_t* bitmap_arr, uint8_t* mask_arr, int64_t n);
void print_str_arr(uint64_t n, uint64_t n_chars, offset_t* offsets,
                   uint8_t* data);
void print_list_str_arr(uint64_t n, const char* data,
                        const offset_t* data_offsets,
                        const offset_t* index_offsets,
                        const uint8_t* null_bitmap);
void bytes_to_hex(char* output, char* data, int64_t data_len);
int64_t bytes_fromhex(unsigned char* output, unsigned char* data,
                      int64_t data_len);
void int_to_hex(char* output, int64_t output_len, uint64_t int_val);
array_info* str_to_dict_str_array(array_info* str_arr);
int64_t re_escape_length_kind1(char* pattern, int64_t length);
int64_t re_escape_length_kind2(char* pattern, int64_t length);
int64_t re_escape_length_kind4(char* pattern, int64_t length);
int64_t re_escape_length(char* pattern, int64_t length, int32_t kind);
void re_escape_with_output_kind1(char* pattern, int64_t length,
                                 char* out_pattern);
void re_escape_with_output_kind2(char* pattern, int64_t length,
                                 char* out_pattern);
void re_escape_with_output_kind4(char* pattern, int64_t length,
                                 char* out_pattern);
void re_escape_with_output(char* pattern, int64_t length, char* out_pattern,
                           int32_t kind);

PyMODINIT_FUNC PyInit_hstr_ext(void) {
    PyObject* m;
    MOD_DEF(m, "hstr_ext", "No docs", nullptr);
    if (m == nullptr) {
        return nullptr;
    }

    // init numpy
    import_array();

    bodo_common_init();

    // All C functions, so they don't throw exceptions.
    // In some cases, there are error conditions that we might want to handle
    // using exceptions, but most of those occur in box/unbox which we don't
    // know how to handle on the python side yet.

    SetAttrStringFromVoidPtr(m, init_string_const);
    SetAttrStringFromVoidPtr(m, dtor_str_arr_split_view);
    SetAttrStringFromVoidPtr(m, str_arr_split_view_alloc);
    SetAttrStringFromVoidPtr(m, str_arr_split_view_impl);
    SetAttrStringFromVoidPtr(m, get_c_str);

    SetAttrStringFromVoidPtr(m, str_to_int64);
    SetAttrStringFromVoidPtr(m, str_to_uint64);
    SetAttrStringFromVoidPtr(m, str_to_int64_base);
    SetAttrStringFromVoidPtr(m, str_to_float64);
    SetAttrStringFromVoidPtr(m, str_to_float32);
    SetAttrStringFromVoidPtr(m, get_str_len);
    SetAttrStringFromVoidPtr(m, pd_pyarrow_array_from_string_array);
    SetAttrStringFromVoidPtr(m, setitem_string_array);
    SetAttrStringFromVoidPtr(m, setitem_binary_array);

    SetAttrStringFromVoidPtr(m, set_string_array_range);
    SetAttrStringFromVoidPtr(m, convert_len_arr_to_offset32);
    SetAttrStringFromVoidPtr(m, convert_len_arr_to_offset);
    SetAttrStringFromVoidPtr(m, print_str_arr);
    SetAttrStringFromVoidPtr(m, str_arr_to_int64);
    SetAttrStringFromVoidPtr(m, str_arr_to_float64);
    SetAttrStringFromVoidPtr(m, str_from_float32);
    SetAttrStringFromVoidPtr(m, str_from_float64);
    SetAttrStringFromVoidPtr(m, inplace_int64_to_str);
    SetAttrStringFromVoidPtr(m, is_na);
    SetAttrStringFromVoidPtr(m, del_str);
    SetAttrStringFromVoidPtr(m, array_size);
    SetAttrStringFromVoidPtr(m, unicode_to_utf8);
    SetAttrStringFromVoidPtr(m, array_getptr1);
    SetAttrStringFromVoidPtr(m, array_setitem);
    SetAttrStringFromVoidPtr(m, get_utf8_size);
    SetAttrStringFromVoidPtr(m, bool_arr_to_bitmap);
    SetAttrStringFromVoidPtr(m, mask_arr_to_bitmap);
    SetAttrStringFromVoidPtr(m, memcmp);
    SetAttrStringFromVoidPtr(m, bytes_to_hex);
    SetAttrStringFromVoidPtr(m, bytes_fromhex);
    SetAttrStringFromVoidPtr(m, int_to_hex);
    SetAttrStringFromVoidPtr(m, str_to_dict_str_array);
    SetAttrStringFromVoidPtr(m, re_escape_length);
    SetAttrStringFromVoidPtr(m, re_escape_with_output);

    return m;
}

void* init_string_const(char* in_str, int64_t size) {
    // std::cout<<"init str: "<<in_str<<" "<<size<<std::endl;
    return new std::string(in_str, size);
}

void del_str(std::string* in_str) {
    delete in_str;
    return;
}

void dtor_str_arr_split_view(str_arr_split_view_payload* in_str_arr,
                             int64_t size, void* in) {
    // printf("str arr dtor size: %lld\n", in_str_arr->size);
    // printf("num chars: %d\n", in_str_arr->offsets[in_str_arr->size]);
    delete[] in_str_arr->index_offsets;
    delete[] in_str_arr->data_offsets;
    delete[] in_str_arr->null_bitmap;
    // if (in_str_arr->null_bitmap != nullptr)
    //     delete[] in_str_arr->null_bitmap;
}

void str_arr_split_view_alloc(str_arr_split_view_payload* out_view,
                              int64_t num_items, int64_t num_offsets) {
    out_view->index_offsets = new offset_t[num_items + 1];
    out_view->data_offsets = new offset_t[num_offsets];
    int64_t n_bytes = (num_items + 7) >> 3;
    out_view->null_bitmap = new uint8_t[n_bytes];
}

// example: ['AB,CC', 'C,ABB,D', 'G', '', 'g,f']
// offsets [0, 5, 12, 13, 13, 14, 17]
// data_offsets [-1, 2, 5,   4, 6, 10, 12,  11, 13,   12, 13,   12, 14, 16]
// index_offsets [0, 3, 7, 9, 11, 14]
void str_arr_split_view_impl(str_arr_split_view_payload* out_view,
                             int64_t n_strs, offset_t* offsets, char* data,
                             char* null_bitmap, char sep) {
    offset_t total_chars = offsets[n_strs];
    // printf("n_strs %d sep %c total chars:%d\n", n_strs, sep, total_chars);
    offset_t* index_offsets = new offset_t[n_strs + 1];
    bodo::vector<offset_t> data_offs;

    data_offs.push_back(-1);
    index_offsets[0] = 0;
    // uint32_t curr_data_off = 0;

    offset_t data_ind = offsets[0];
    int str_ind = 0;
    // while there are chars to consume, equal since the first if will consume
    // it
    while (data_ind <= total_chars) {
        // string has finished
        if (data_ind == offsets[str_ind + 1]) {
            data_offs.push_back(data_ind);
            index_offsets[str_ind + 1] = data_offs.size();
            str_ind++;
            if (str_ind == n_strs) {
                break;  // all finished
            }
            // start new string
            data_offs.push_back(data_ind - 1);
            continue;  // stay on same data_ind for start of next string
        }
        if (data[data_ind] == sep) {
            data_offs.push_back(data_ind);
        }
        data_ind++;
    }
    out_view->index_offsets = index_offsets;
    out_view->data_offsets = new offset_t[data_offs.size()];
    // TODO: avoid copy
    std::ranges::copy(data_offs, out_view->data_offsets);

    // copying the null_bitmap. Maybe we can avoid that
    // in some cases.
    int64_t n_bytes = (n_strs + 7) >> 3;
    out_view->null_bitmap = new uint8_t[n_bytes];
    memcpy(out_view->null_bitmap, null_bitmap, n_bytes);

    // printf("index_offsets: ");
    // for (int i=0; i<=n_strs; i++)
    //     printf("%d ", index_offsets[i]);
    // printf("\n");
    // printf("data_offsets: ");
    // for (int i=0; i<data_offs.size(); i++)
    //     printf("%d ", data_offs[i]);
    // printf("\n");
    return;
}

const char* get_c_str(std::string* s) {
    // printf("in get %s\n", s->c_str());
    return s->c_str();
}

double str_to_float64(std::string* str) {
    try {
        return std::stod(*str);
    } catch (const std::invalid_argument&) {
        PyErr_SetString(
            PyExc_RuntimeError,
            ("invalid string to float conversion: " + *str).c_str());
    } catch (const std::out_of_range&) {
        PyErr_SetString(PyExc_RuntimeError,
                        "out of range string to float conversion");
    }
    return nan("");
}

float str_to_float32(std::string* str) {
    try {
        return std::stof(*str);
    } catch (const std::invalid_argument&) {
        PyErr_SetString(PyExc_RuntimeError,
                        "invalid string to float conversion");
    } catch (const std::out_of_range&) {
        PyErr_SetString(PyExc_RuntimeError,
                        "out of range string to float conversion");
    }
    return nanf("");
}

int64_t get_str_len(std::string* str) {
    // std::cout << "str len called: " << *str << " " <<
    // str->length()<<std::endl;
    return str->length();
}

void setitem_string_array(offset_t* offsets, char* data, uint64_t n_bytes,
                          char* str, int64_t len, int kind, int is_ascii,
                          int64_t index) {
#undef CHECK
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        return;                        \
    }
    // std::cout << "setitem str: " << *str << " " << index << std::endl;
    if (index == 0) {
        offsets[index] = 0;
    }
    offset_t start = offsets[index];
    offset_t utf8_len = 0;
    // std::cout << "start " << start << " len " << len << std::endl;

    if (is_ascii == 1) {
        memcpy(&data[start], str, len);
        utf8_len = len;
    } else {
        utf8_len = unicode_to_utf8(&data[start], str, len, kind);
    }

    CHECK(utf8_len < std::numeric_limits<offset_t>::max(),
          "string array too large");
    CHECK(start + utf8_len <= n_bytes, "out of bounds string array setitem");
    offsets[index + 1] = start + utf8_len;
    return;
#undef CHECK
}

void setitem_binary_array(offset_t* offsets, char* data, uint64_t n_bytes,
                          char* str, int64_t len, int64_t index) {
#undef CHECK
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        return;                        \
    }
    offset_t utf8_len = (offset_t)len;

    if (index == 0) {
        offsets[index] = 0;
    }
    offset_t start = offsets[index];

    // Bytes objects in python are always just an array of chars,
    // so we should never need to do any decoding
    memcpy(&data[start], str, len);

    CHECK(utf8_len < std::numeric_limits<offset_t>::max(),
          "string array too large");
    CHECK(start + utf8_len <= n_bytes, "out of bounds string array setitem");
    offsets[index + 1] = start + utf8_len;
    return;
#undef CHECK
}

int64_t get_utf8_size(char* str, int64_t len, int kind) {
    return unicode_to_utf8(nullptr, str, len, kind);
}

void set_string_array_range(offset_t* out_offsets, char* out_data,
                            offset_t* in_offsets, char* in_data,
                            int64_t start_str_ind, int64_t start_chars_ind,
                            int64_t num_strs, int64_t num_chars) {
    // printf("%d %d\n", start_str_ind, start_chars_ind); fflush(stdout);
    offset_t curr_offset = 0;
    if (start_str_ind != 0) {
        curr_offset = out_offsets[start_str_ind];
    }

    // set offsets
    for (size_t i = 0; i < (size_t)num_strs; i++) {
        out_offsets[start_str_ind + i] = curr_offset;
        offset_t len = in_offsets[i + 1] - in_offsets[i];
        curr_offset += len;
    }
    out_offsets[start_str_ind + num_strs] = curr_offset;
    // copy all chars
    memcpy(out_data + start_chars_ind, in_data, num_chars);
    return;
}

void convert_len_arr_to_offset32(uint32_t* offsets, int64_t num_strs) {
    uint32_t curr_offset = 0;
    for (int64_t i = 0; i < num_strs; i++) {
        uint32_t val = offsets[i];
        offsets[i] = curr_offset;
        curr_offset += val;
    }
    offsets[num_strs] = curr_offset;
}

void convert_len_arr_to_offset(uint32_t* lens, uint64_t* offsets,
                               uint64_t num_strs) {
    uint64_t curr_offset = 0;
    for (uint64_t i = 0; i < num_strs; i++) {
        uint32_t length = lens[i];
        offsets[i] = curr_offset;
        curr_offset += length;
    }
    offsets[num_strs] = curr_offset;
}

int str_arr_to_int64(int64_t* out, offset_t* offsets, char* data,
                     int64_t index) {
    offset_t size = offsets[index + 1] - offsets[index];
    offset_t start = offsets[index];
    try {
        *out = boost::lexical_cast<int64_t>(data + start, (std::size_t)size);
        return 0;
    } catch (const boost::bad_lexical_cast&) {
        *out = 0;
        return -1;
    }
    return -1;
}

int str_arr_to_float64(double* out, offset_t* offsets, char* data,
                       int64_t index) {
    offset_t size = offsets[index + 1] - offsets[index];
    offset_t start = offsets[index];
    try {
        *out = boost::lexical_cast<double>(data + start, (std::size_t)size);
        return 0;
    } catch (const boost::bad_lexical_cast&) {
        *out = std::nan("");  // TODO: numpy NaN
        return -1;
    }
    return -1;
}

int64_t str_to_int64(char* data, int64_t length) {
    try {
        return boost::lexical_cast<int64_t>(data, (std::size_t)length);
    } catch (const boost::bad_lexical_cast&) {
        PyErr_SetString(PyExc_RuntimeError, "invalid string to int conversion");
        return -1;
    }
}

uint64_t str_to_uint64(char* data, int64_t length) {
    try {
        return boost::lexical_cast<uint64_t>(data, (std::size_t)length);
    } catch (const boost::bad_lexical_cast&) {
        PyErr_SetString(PyExc_RuntimeError, "invalid string to int conversion");
        return -1;
    }
}

int64_t str_to_int64_base(char* data, int64_t length, int64_t base) {
    /* Influenced by stack overflow:
       https://stackoverflow.com/questions/194465/how-to-parse-a-string-to-an-int-in-c
       Base is at most 36 per strtlon requirements.
       Called with base=10 by default if no base was provided by a user.
    */
    char* end;
    int64_t l;
    errno = 0;
    char* buffer = (char*)malloc(length + 1);
    if (!buffer) {
        PyErr_SetString(
            PyExc_RuntimeError,
            "Failed to allocate space for string to int conversion");
        return -1;
    }
    buffer[length] = '\0';
    strncpy(buffer, data, length);
    // This assumes data is null terminated. Is that safe?
    l = strtoll(buffer, &end, base);
    if (errno || *buffer == '\0' || *end != '\0') {
        free(buffer);
        PyErr_SetString(PyExc_RuntimeError, "invalid string to int conversion");
        return -1;
    }
    free(buffer);
    return l;
}

void str_from_float32(char* s, float in) {
    // Use 1024 as arbitrary buffer size to make compiler happy
    snprintf(s, 1024, "%f", in);
}

void str_from_float64(char* s, double in) {
    // Use 1024 as arbitrary buffer size to make compiler happy
    snprintf(s, 1024, "%f", in);
}

/**
 * @brief convert int64 value to string and write to string pointer
 *
 * @param str output string pointer to write to
 * @param l length of output string value
 * @param value input integer value to convert
 */
void inplace_int64_to_str(char* str, int64_t l, int64_t value) {
    if (value == 0) {
        str[0] = '0';
        return;
    }
    if (value < 0) {
        value = -value;
        str[0] = '-';
    }
    size_t i = 1;
    while (value != 0) {
        str[l - i] = '0' + (value % 10);
        value /= 10;
        i++;
    }
}

/**
 * @brief Create Pandas ArrowStringArray from Bodo's packed StringArray or
 * dict-encoded string array. Creates ArrowExtensionArray for binary arrays if
 * is_bytes is set.
 *
 * @param str_arr input string array, dict-encoded string array, or binary array
 * (deleted by function after use)
 * @param is_bytes 1 if input is binary array otherwise 0
 * @return void* Pandas ArrowStringArray or ArrowExtensionArray
 */
void* pd_pyarrow_array_from_string_array(array_info* str_arr,
                                         const int is_bytes) {
#undef CHECK
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        return NULL;                   \
    }

    // convert to Arrow array with copy (since passing to Pandas)
    // only str_arr and true arguments are relevant here
    arrow::TimeUnit::type time_unit = arrow::TimeUnit::NANO;
    auto arrow_arr = bodo_array_to_arrow(
        bodo::BufferPool::DefaultPtr(), std::shared_ptr<array_info>(str_arr),
        false /*convert_timedelta_to_int64*/, "", time_unit,
        false /*downcast_time_ns_to_us*/,
        bodo::default_buffer_memory_manager());

    // https://arrow.apache.org/docs/python/integration/extending.html
    CHECK(!arrow::py::import_pyarrow(), "importing pyarrow failed");

    // convert Arrow C++ to PyArrow
    PyObject* pyarrow_arr = arrow::py::wrap_array(arrow_arr);

    // call pd.arrays.ArrowStringArray(pyarrow_arr) which avoids copy
    PyObject* pd_mod = PyImport_ImportModule("pandas");
    CHECK(pd_mod, "importing pandas module failed");
    PyObject* pd_arrays_mod = PyObject_GetAttrString(pd_mod, "arrays");
    CHECK(pd_arrays_mod, "importing pandas.arrays module failed");

    PyObject* str_arr_obj = PyObject_CallMethod(
        pd_arrays_mod, is_bytes ? "ArrowExtensionArray" : "ArrowStringArray",
        "O", pyarrow_arr);

    Py_DECREF(pd_mod);
    Py_DECREF(pd_arrays_mod);
    Py_DECREF(pyarrow_arr);
    return str_arr_obj;
#undef CHECK
}

// helper functions for call Numpy APIs

npy_intp array_size(PyArrayObject* arr) { return PyArray_SIZE(arr); }

void* array_getptr1(PyArrayObject* arr, npy_intp ind) {
    return PyArray_GETPTR1(arr, ind);
}

void array_setitem(PyArrayObject* arr, char* p, PyObject* s) {
#undef CHECK
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        return;                        \
    }
    int err = PyArray_SETITEM(arr, p, s);
    CHECK(err == 0, "setting item in numpy array failed");
    return;
#undef CHECK
}

void bool_arr_to_bitmap(uint8_t* bitmap_arr, uint8_t* bool_arr, int64_t n) {
    for (int i = 0; i < n; i++) {
        bitmap_arr[i / 8] ^=
            static_cast<uint8_t>(-static_cast<uint8_t>(bool_arr[i] != 0) ^
                                 bitmap_arr[i / 8]) &
            kBitmask[i % 8];
    }
}

void mask_arr_to_bitmap(uint8_t* bitmap_arr, uint8_t* mask_arr, int64_t n) {
    for (int i = 0; i < n; i++) {
        bitmap_arr[i / 8] ^=
            static_cast<uint8_t>(-static_cast<uint8_t>(mask_arr[i] == 0) ^
                                 bitmap_arr[i / 8]) &
            kBitmask[i % 8];
    }
}

void print_str_arr(uint64_t n, uint64_t n_chars, offset_t* offsets,
                   uint8_t* data) {
    std::cout << "n: " << n << " n_chars: " << n_chars << "\n";
    for (uint64_t i = 0; i < n; i++) {
        std::cout << "offsets: " << offsets[i] << " " << offsets[i + 1]
                  << "  chars:";
        for (uint64_t j = offsets[i]; j < offsets[i + 1]; j++) {
            std::cout << data[j] << " ";
        }
        std::cout << "\n";
    }
}

void print_list_str_arr(uint64_t n, const char* data,
                        const offset_t* data_offsets,
                        const offset_t* index_offsets,
                        const uint8_t* null_bitmap) {
    uint64_t n_strs = index_offsets[n];
    uint64_t n_chars = data_offsets[n_strs];
    std::cout << "n: " << n << " n_strs: " << n_strs << " n_chars: " << n_chars
              << "\n";
    for (uint64_t i = 0; i < n; i++) {
        std::cout << "index_offsets: " << index_offsets[i] << " "
                  << index_offsets[i + 1] << "  lists:";
        for (uint64_t j = index_offsets[i]; j < index_offsets[i + 1]; j++) {
            for (uint64_t k = data_offsets[j]; k < data_offsets[j + 1]; k++) {
                std::cout << data[k] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

void bytes_to_hex(char* output, char* data, int64_t data_len) {
    /*
        Implementation of bytes.hex() which converts
        each the bytes to a string hex representation.

        This is handled in regular Python here:
        https://github.com/python/cpython/blob/bb3e0c240bc60fe08d332ff5955d54197f79751c/Objects/clinic/bytesobject.c.h#L807
        https://github.com/python/cpython/blob/bb3e0c240bc60fe08d332ff5955d54197f79751c/Objects/bytesobject.c#L2464
        https://github.com/python/cpython/blob/bb3e0c240bc60fe08d332ff5955d54197f79751c/Python/pystrhex.c#L164
        https://github.com/python/cpython/blob/bb3e0c240bc60fe08d332ff5955d54197f79751c/Python/pystrhex.c#L7

        Note: We ignore sep and bytes_per_sep_group because sep is always NULL
    */
    // Note: We assume output is allocated to be 2 * data_len character
    // and a null terminator at the end that is already set.
    for (int i = 0; i < data_len; i++) {
        char c = data[i];
        output[2 * i] = hex_values[(c >> 4) & 0x0f];
        output[(2 * i) + 1] = hex_values[c & 0x0f];
    }
}

int64_t bytes_fromhex(unsigned char* output, unsigned char* data,
                      int64_t data_len) {
    /*
        Implementation of bytes.hex() which converts
        a string hex representation to the bytes value.

        Returns the number of bytes that are allocated for truncated the output.

        This is handled in regular Python here:
        https://github.com/python/cpython/blob/1d08d85cbe49c0748a8ee03aec31f89ab8e81496/Objects/bytesobject.c#L2359
        We assume we have already error checked an allocated the data in Python.

    */
#undef CHECK
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        PyGILState_Release(gilstate);  \
        return -1;                     \
    }
    auto gilstate = PyGILState_Ensure();
    unsigned int top, bot;
    int64_t length = 0;
    // Note: We assume output is allocated to be data_len // 2.
    unsigned char* end = data + data_len;
    while (data < end) {
        // We will always break out of this loop because we assume
        // data is null terminated
        if (Py_ISSPACE(*data)) {
            do {
                data++;
            } while (Py_ISSPACE(*data));
            // This break is taken if we end with a space character
            if (data >= end) {
                break;
            }
        }
        CHECK((end - data) >= 2,
              "bytes.fromhex, must provide two hex values per byte");
        top = _DigitValue[*data++];
        bot = _DigitValue[*data++];
        CHECK(top < 16, "bytes.fromhex: unsupport character");
        CHECK(bot < 16, "bytes.fromhex: unsupport character");
        output[length++] = (unsigned char)((top << 4) + bot);
    }
    return length;
#undef CHECK
}

void int_to_hex(char* output, int64_t output_len, uint64_t int_val) {
    /*
        Implementation of hex(int) with a precomputed final length.
        I could not find the source Python implementation
    */
    // Append characters in reverse order.
    for (int i = output_len - 1; i >= 0; i--) {
        output[i] = hex_values[int_val & 0x0F];
        int_val = int_val >> 4;
    }
}

}  // extern "C"

/**
 * @brief Kernel to Convert String Arrays to Dictionary String Arrays
 * Most of the logic is shared with DictionaryEncodedFromStringBuilder
 * (TableBuilder::BuilderColumn subclass in arrow_reader.cpp)
 *
 * @param str_arr Input String Array (deleted by function after use)
 * @return array_info* Output Dictionary Array with Copied Contents
 */
array_info* str_to_dict_str_array(array_info* str_arr) {
    assert(str_arr->arr_type == bodo_array_type::STRING);
    const auto arr_len = str_arr->length;
    const auto num_null_bitmask_bytes = (arr_len + 7) >> 3;

    // Dictionary Indices Array
    std::shared_ptr<array_info> indices_arr =
        alloc_nullable_array(arr_len, Bodo_CTypes::INT32);
    memcpy(indices_arr->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>(),
           str_arr->null_bitmask<bodo_array_type::STRING>(),
           num_null_bitmask_bytes);

    // Map string to its new index in dictionary values array
    bodo::unord_map_container<std::string, std::pair<int32_t, uint64_t>,
                              string_hash, std::equal_to<>>
        str_to_ind;
    uint32_t num_dict_strs = 0;
    uint64_t total_dict_chars = 0;

    offset_t* offsets = (offset_t*)str_arr->data2<bodo_array_type::STRING>();
    for (uint64_t i = 0; i < arr_len; i++) {
        if (!str_arr->get_null_bit<bodo_array_type::STRING>(i)) {
            continue;
        }
        std::string_view elem(
            str_arr->data1<bodo_array_type::STRING>() + offsets[i],
            offsets[i + 1] - offsets[i]);

        int32_t elem_idx;
        if (!str_to_ind.contains(elem)) {
            std::pair<int32_t, uint64_t> ind_offset_len =
                std::make_pair(num_dict_strs, total_dict_chars);
            // TODO: remove std::string() after upgrade to C++23
            str_to_ind[std::string(elem)] = ind_offset_len;
            total_dict_chars += elem.length();
            elem_idx = num_dict_strs++;
        } else {
            elem_idx = str_to_ind.find(elem)->second.first;
        }

        indices_arr->at<int32_t, bodo_array_type::NULLABLE_INT_BOOL>(i) =
            elem_idx;
    }
    delete str_arr;

    // Create Dictionary String Values
    int64_t dict_id = generate_array_id(num_dict_strs);
    std::shared_ptr<array_info> values_arr = alloc_string_array(
        Bodo_CTypes::STRING, num_dict_strs, total_dict_chars, dict_id);
    int64_t n_null_bytes = (num_dict_strs + 7) >> 3;
    memset(values_arr->null_bitmask<bodo_array_type::STRING>(), 0xFF,
           n_null_bytes);  // No nulls

    offset_t* out_offsets =
        (offset_t*)values_arr->data2<bodo_array_type::STRING>();
    out_offsets[0] = 0;
    for (auto& it : str_to_ind) {
        memcpy(values_arr->data1<bodo_array_type::STRING>() + it.second.second,
               it.first.c_str(), it.first.size());
        out_offsets[it.second.first] = it.second.second;
    }
    out_offsets[num_dict_strs] = static_cast<offset_t>(total_dict_chars);

    // Python is responsible for deleting pointer
    return new array_info(bodo_array_type::DICT, Bodo_CTypes::CTypeEnum::STRING,
                          arr_len, {}, {values_arr, indices_arr});
}

// Inspired by the Cpython implementation
// https://github.com/python/cpython/blob/89442e18e1e17c0eb0eb06e5da489e1cb2d4219d/Lib/re/__init__.py#L251
// However Numba doesn't support translate so we implement an optimized kernel
// for the dictionary. This set is _special_chars_map from Cpython but we always
// map to `\\` so we don't need a dict
static const std::set<char> escapes{
    '(', ')',  '[', ']', '{', '}', '?', '*',  '+',  '-',  '|',  '^',
    '$', '\\', '.', '&', '~', '#', ' ', '\t', '\n', '\r', '\v', '\f'};

/**
 * @brief Get the string length of a 1 byte width string escaped by
 * re.escape.
 *
 * @param pattern The input pattern to escape
 * @param length The number of elements in the string.
 * @return int64_t Length of the escaped string.
 */
int64_t re_escape_length_kind1(char* pattern, int64_t length) {
    int64_t num_escapes = 0;
    for (int i = 0; i < length; i++) {
        // Count the number of escapes.
        num_escapes += int(escapes.contains(pattern[i]));
    }
    return length + num_escapes;
}

/**
 * @brief Get the string length of a 2 byte width string escaped by
 * re.escape. Note the number of bytes = 2 * length.
 *
 * @param pattern The input pattern to escape
 * @param length The number of elements in the string.
 * @return int64_t Length of the escaped string. This is not the number
 * of bytes.
 */
int64_t re_escape_length_kind2(char* pattern, int64_t length) {
    int64_t num_escapes = 0;
    for (int i = 0; i < length; i++) {
        int16_t char_val = *((int16_t*)(pattern) + i);
        // All the escaped characters fit in 1 byte.
        if (char_val < 0xff) {
            char to_check = (char)char_val;
            num_escapes += int(escapes.contains(to_check));
        }
    }
    return length + num_escapes;
}

/**
 * @brief Get the string length of a 4 byte width string escaped by
 * re.escape. Note the number of bytes = 4 * length.
 *
 * @param pattern The input pattern to escape
 * @param length The number of elements in the string.
 * @return int64_t Length of the escaped string. This is not the number
 * of bytes.
 */
int64_t re_escape_length_kind4(char* pattern, int64_t length) {
    int64_t num_escapes = 0;
    for (int i = 0; i < length; i++) {
        int32_t char_val = *((int32_t*)(pattern) + i);
        // All the escaped characters fit in 1 byte.
        if (char_val < 0xff) {
            // Count the number of escapes.
            char to_check = (char)char_val;
            num_escapes += int(escapes.contains(to_check));
        }
    }
    return length + num_escapes;
}

/**
 * @brief Get the string length of a string escaped by
 * re.escape. Note this is not the number of bytes!
 *
 * @param pattern The input pattern to escape
 * @param length The number of elements in the string.
 * @param kind The number of bytes per element in pattern.
 * @return int64_t Length of the escaped string. This is not the number
 * of bytes.
 */
int64_t re_escape_length(char* pattern, int64_t length, int32_t kind) {
    if (kind == PyUnicode_1BYTE_KIND) {
        return re_escape_length_kind1(pattern, length);
    } else if (kind == PyUnicode_2BYTE_KIND) {
        return re_escape_length_kind2(pattern, length);
    } else {
        return re_escape_length_kind4(pattern, length);
    }
}

/**
 * @brief Perform re.escape on pattern and store the output in
 * out_pattern. This is written for 1 byte per element strings.
 *
 * @param pattern The input pattern to escape.
 * @param length The number of elements in the input string.
 * @param[out] out_pattern The character array to store output characters.
 * It is already allocated with the correct length.
 */
void re_escape_with_output_kind1(char* pattern, int64_t length,
                                 char* out_pattern) {
    int j = 0;
    for (int i = 0; i < length; i++) {
        if (escapes.contains(pattern[i])) {
            out_pattern[j++] = '\\';
        }
        out_pattern[j++] = pattern[i];
    }
}

/**
 * @brief Perform re.escape on pattern and store the output in
 * out_pattern. This is written for 2 byte per element strings.
 *
 * @param pattern The input pattern to escape.
 * @param length The number of elements in the input string. Note this is
 * not the number of bytes.
 * @param[out] out_pattern The character array to store output characters.
 * It is already allocated with the correct length.
 */
void re_escape_with_output_kind2(char* pattern, int64_t length,
                                 char* out_pattern) {
    int j = 0;
    int16_t* in_cast = ((int16_t*)pattern);
    int16_t* out_cast = ((int16_t*)out_pattern);
    for (int i = 0; i < length; i++) {
        int16_t in_val = in_cast[i];
        // All the escaped characters fit in 1 byte.
        if (in_val < 0xff && escapes.contains(in_val)) {
            // \ is 0x5C
            out_cast[j++] = 0x5C;
        }
        out_cast[j++] = in_val;
    }
}

/**
 * @brief Perform re.escape on pattern and store the output in
 * out_pattern. This is written for 4 byte per element strings.
 *
 * @param pattern The input pattern to escape.
 * @param length The number of elements in the input string. Note this is
 * not the number of bytes.
 * @param[out] out_pattern The character array to store output characters.
 * It is already allocated with the correct length.
 */
void re_escape_with_output_kind4(char* pattern, int64_t length,
                                 char* out_pattern) {
    int j = 0;
    int32_t* in_cast = ((int32_t*)pattern);
    int32_t* out_cast = ((int32_t*)out_pattern);
    for (int i = 0; i < length; i++) {
        int32_t in_val = in_cast[i];
        // All the escaped characters fit in 1 byte.
        if (in_val < 0xff && escapes.contains(in_val)) {
            // \ is 0x5C
            out_cast[j++] = 0x5C;
        }
        out_cast[j++] = in_val;
    }
}

/**
 * @brief Perform re.escape on pattern and store the output in
 * out_pattern.
 *
 * @param pattern The input pattern to escape.
 * @param length The number of elements in the input string. Note this is
 * not the number of bytes.
 * @param[out] out_pattern The character array to store output characters.
 * It is already allocated with the correct length.
 * @param kind The number of bytes per element in pattern and out_pattern.
 */
void re_escape_with_output(char* pattern, int64_t length, char* out_pattern,
                           int32_t kind) {
    if (kind == PyUnicode_1BYTE_KIND) {
        re_escape_with_output_kind1(pattern, length, out_pattern);
    } else if (kind == PyUnicode_2BYTE_KIND) {
        re_escape_with_output_kind2(pattern, length, out_pattern);
    } else {
        re_escape_with_output_kind4(pattern, length, out_pattern);
    }
}
