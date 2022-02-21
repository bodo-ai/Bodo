// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <boost/algorithm/string/replace.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "_bodo_common.h"

#include "_str_decode.cpp"

#include <boost/lexical_cast.hpp>

extern "C" {

// taken from Arrow bin-util.h
// the bitwise complement version of kBitmask
static constexpr uint8_t kFlippedBitmask[] = {254, 253, 251, 247,
                                              239, 223, 191, 127};

// Map of integers to hex values. Note we use an array because the keys are 0-15
static constexpr char hex_values[] = {'0', '1', '2', '3', '4', '5', '6', '7',
                                      '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};

static inline void ClearBit(uint8_t* bits, int64_t i) {
    bits[i / 8] &= kFlippedBitmask[i % 8];
}

static inline void SetBit(uint8_t* bits, int64_t i) {
    bits[i / 8] |= kBitmask[i % 8];
}

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

void* np_array_from_string_array(int64_t no_strings,
                                 const offset_t* offset_table,
                                 const char* buffer, const uint8_t* null_bitmap,
                                 const int is_bytes);
void* pd_array_from_string_array(int64_t no_strings,
                                 const offset_t* offset_table,
                                 const char* buffer,
                                 const uint8_t* null_bitmap);

void setitem_string_array(offset_t* offsets, char* data, uint64_t n_bytes,
                          char* str, int64_t len, int kind, int is_ascii,
                          int64_t index);
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
void mask_arr_to_bitmap(uint8_t* bitmap_arr, uint8_t* mask_arr, int64_t n);
int is_bool_array(PyArrayObject* arr);
int is_pd_boolean_array(PyObject* arr);
void unbox_bool_array_obj(PyArrayObject* arr, uint8_t* data, uint8_t* bitmap,
                          int64_t n);
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

PyMODINIT_FUNC PyInit_hstr_ext(void) {
    PyObject* m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "hstr_ext", "No docs", -1, NULL,
    };
    m = PyModule_Create(&moduledef);
    if (m == NULL) return NULL;

    // init numpy
    import_array();

    bodo_common_init();

    // All C functions, so they don't throw exceptions.
    // In some cases, there are error conditions that we might want to handle
    // using exceptions, but most of those occur in box/unbox which we don't
    // know how to handle on the python side yet.

    PyObject_SetAttrString(m, "init_string_const",
                           PyLong_FromVoidPtr((void*)(&init_string_const)));
    PyObject_SetAttrString(
        m, "dtor_str_arr_split_view",
        PyLong_FromVoidPtr((void*)(&dtor_str_arr_split_view)));
    PyObject_SetAttrString(
        m, "str_arr_split_view_alloc",
        PyLong_FromVoidPtr((void*)(&str_arr_split_view_alloc)));
    PyObject_SetAttrString(
        m, "str_arr_split_view_impl",
        PyLong_FromVoidPtr((void*)(&str_arr_split_view_impl)));
    PyObject_SetAttrString(m, "get_c_str",
                           PyLong_FromVoidPtr((void*)(&get_c_str)));

    PyObject_SetAttrString(m, "str_to_int64",
                           PyLong_FromVoidPtr((void*)(&str_to_int64)));
    PyObject_SetAttrString(m, "str_to_uint64",
                           PyLong_FromVoidPtr((void*)(&str_to_uint64)));
    PyObject_SetAttrString(m, "str_to_int64_base",
                           PyLong_FromVoidPtr((void*)(&str_to_int64_base)));
    PyObject_SetAttrString(m, "str_to_float64",
                           PyLong_FromVoidPtr((void*)(&str_to_float64)));
    PyObject_SetAttrString(m, "str_to_float32",
                           PyLong_FromVoidPtr((void*)(&str_to_float32)));
    PyObject_SetAttrString(m, "get_str_len",
                           PyLong_FromVoidPtr((void*)(&get_str_len)));
    PyObject_SetAttrString(
        m, "pd_array_from_string_array",
        PyLong_FromVoidPtr((void*)(&pd_array_from_string_array)));
    PyObject_SetAttrString(
        m, "np_array_from_string_array",
        PyLong_FromVoidPtr((void*)(&np_array_from_string_array)));
    PyObject_SetAttrString(m, "setitem_string_array",
                           PyLong_FromVoidPtr((void*)(&setitem_string_array)));
    PyObject_SetAttrString(
        m, "set_string_array_range",
        PyLong_FromVoidPtr((void*)(&set_string_array_range)));
    PyObject_SetAttrString(
        m, "convert_len_arr_to_offset32",
        PyLong_FromVoidPtr((void*)(&convert_len_arr_to_offset32)));
    PyObject_SetAttrString(
        m, "convert_len_arr_to_offset",
        PyLong_FromVoidPtr((void*)(&convert_len_arr_to_offset)));
    PyObject_SetAttrString(m, "print_str_arr",
                           PyLong_FromVoidPtr((void*)(&print_str_arr)));
    PyObject_SetAttrString(m, "str_arr_to_int64",
                           PyLong_FromVoidPtr((void*)(&str_arr_to_int64)));
    PyObject_SetAttrString(m, "str_arr_to_float64",
                           PyLong_FromVoidPtr((void*)(&str_arr_to_float64)));
    PyObject_SetAttrString(m, "str_from_float32",
                           PyLong_FromVoidPtr((void*)(&str_from_float32)));
    PyObject_SetAttrString(m, "str_from_float64",
                           PyLong_FromVoidPtr((void*)(&str_from_float64)));
    PyObject_SetAttrString(m, "inplace_int64_to_str",
                           PyLong_FromVoidPtr((void*)(&inplace_int64_to_str)));
    PyObject_SetAttrString(m, "is_na", PyLong_FromVoidPtr((void*)(&is_na)));
    PyObject_SetAttrString(m, "del_str", PyLong_FromVoidPtr((void*)(&del_str)));
    PyObject_SetAttrString(m, "array_size",
                           PyLong_FromVoidPtr((void*)(&array_size)));
    PyObject_SetAttrString(m, "unicode_to_utf8",
                           PyLong_FromVoidPtr((void*)(&unicode_to_utf8)));
    PyObject_SetAttrString(m, "array_getptr1",
                           PyLong_FromVoidPtr((void*)(&array_getptr1)));
    PyObject_SetAttrString(m, "array_setitem",
                           PyLong_FromVoidPtr((void*)(&array_setitem)));
    PyObject_SetAttrString(m, "get_utf8_size",
                           PyLong_FromVoidPtr((void*)(&get_utf8_size)));
    PyObject_SetAttrString(m, "mask_arr_to_bitmap",
                           PyLong_FromVoidPtr((void*)(&mask_arr_to_bitmap)));
    PyObject_SetAttrString(m, "is_bool_array",
                           PyLong_FromVoidPtr((void*)(&is_bool_array)));
    PyObject_SetAttrString(m, "is_pd_boolean_array",
                           PyLong_FromVoidPtr((void*)(&is_pd_boolean_array)));
    PyObject_SetAttrString(m, "unbox_bool_array_obj",
                           PyLong_FromVoidPtr((void*)(&unbox_bool_array_obj)));
    PyObject_SetAttrString(m, "memcmp", PyLong_FromVoidPtr((void*)(&memcmp)));
    PyObject_SetAttrString(m, "bytes_to_hex",
                           PyLong_FromVoidPtr((void*)(&bytes_to_hex)));
    PyObject_SetAttrString(m, "bytes_fromhex",
                           PyLong_FromVoidPtr((void*)(&bytes_fromhex)));
    PyObject_SetAttrString(m, "int_to_hex",
                           PyLong_FromVoidPtr((void*)(&int_to_hex)));
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
    std::vector<offset_t> data_offs;

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
            if (str_ind == n_strs) break;  // all finished
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
    std::copy(data_offs.cbegin(), data_offs.cend(), out_view->data_offsets);

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
    } catch(const std::invalid_argument&) {
        PyErr_SetString(PyExc_RuntimeError, "invalid string to float conversion");
    } catch(const std::out_of_range&) {
        PyErr_SetString(PyExc_RuntimeError, "out of range string to float conversion");
    }
    return nan("");
}

float str_to_float32(std::string* str) {
    try {
        return std::stof(*str);
    } catch(const std::invalid_argument&) {
        PyErr_SetString(PyExc_RuntimeError, "invalid string to float conversion");
    } catch(const std::out_of_range&) {
        PyErr_SetString(PyExc_RuntimeError, "out of range string to float conversion");
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
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        return;                        \
    }
    // std::cout << "setitem str: " << *str << " " << index << std::endl;
    if (index == 0) offsets[index] = 0;
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

int64_t get_utf8_size(char* str, int64_t len, int kind) {
    return unicode_to_utf8(NULL, str, len, kind);
}

void set_string_array_range(offset_t* out_offsets, char* out_data,
                            offset_t* in_offsets, char* in_data,
                            int64_t start_str_ind, int64_t start_chars_ind,
                            int64_t num_strs, int64_t num_chars) {
    // printf("%d %d\n", start_str_ind, start_chars_ind); fflush(stdout);
    offset_t curr_offset = 0;
    if (start_str_ind != 0) curr_offset = out_offsets[start_str_ind];

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
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate space for string to int conversion");
        return -1;
    }
    buffer[length] = '\0';
    strncpy(buffer, data, length);
    // This assumes data is null terminated. Is that safe?
    l = strtol(buffer, &end, base);
    if (errno || *buffer == '\0' || *end != '\0') {
        free(buffer);
        PyErr_SetString(PyExc_RuntimeError, "invalid string to int conversion");
        return -1;
    }
    free(buffer);
    return l;
}

void str_from_float32(char* s, float in) { sprintf(s, "%f", in); }

void str_from_float64(char* s, double in) { sprintf(s, "%f", in); }

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

/// @brief  From a StringArray create a numpy array of string objects
/// @return numpy array of str objects
/// @param[in] no_strings number of strings found in buffer
/// @param[in] offset_table offsets for strings in buffer
/// @param[in] buffer with concatenated strings (from StringArray)
void* np_array_from_string_array(int64_t no_strings,
                                 const offset_t* offset_table,
                                 const char* buffer, const uint8_t* null_bitmap,
                                 const int is_bytes) {
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        PyGILState_Release(gilstate);  \
        return NULL;                   \
    }
    auto gilstate = PyGILState_Ensure();

    npy_intp dims[] = {no_strings};
    PyObject* ret = PyArray_SimpleNew(1, dims, NPY_OBJECT);
    CHECK(ret, "allocating numpy array failed");
    int err;
    PyObject* np_mod = PyImport_ImportModule("numpy");
    CHECK(np_mod, "importing numpy module failed");
    PyObject* nan_obj = PyObject_GetAttrString(np_mod, "nan");
    CHECK(nan_obj, "getting np.nan failed");

    for (int64_t i = 0; i < no_strings; ++i) {
        PyObject* s;
        if (is_bytes)
            s = PyBytes_FromStringAndSize(
                buffer + offset_table[i],
                offset_table[i + 1] - offset_table[i]);
        else
            s = PyUnicode_FromStringAndSize(
                buffer + offset_table[i],
                offset_table[i + 1] - offset_table[i]);
        CHECK(s, "creating Python string/unicode object failed");
        auto p = PyArray_GETPTR1((PyArrayObject*)ret, i);
        CHECK(p, "getting offset in numpy array failed");
        if (!is_na(null_bitmap, i))
            err = PyArray_SETITEM((PyArrayObject*)ret, (char*)p, s);
        else
            err = PyArray_SETITEM((PyArrayObject*)ret, (char*)p, nan_obj);
        CHECK(err == 0, "setting item in numpy array failed");
        Py_DECREF(s);
    }

    Py_DECREF(np_mod);
    Py_DECREF(nan_obj);
    PyGILState_Release(gilstate);
    return ret;
#undef CHECK
}

/// @brief  Create Pandas StringArray from Bodo's packed StringArray
/// @return Pandas StringArray of str objects
/// @param[in] no_strings number of strings found in buffer
/// @param[in] offset_table offsets for strings in buffer
/// @param[in] buffer with concatenated strings (from StringArray)
void* pd_array_from_string_array(int64_t no_strings,
                                 const offset_t* offset_table,
                                 const char* buffer,
                                 const uint8_t* null_bitmap) {
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        PyGILState_Release(gilstate);  \
        return NULL;                   \
    }
    auto gilstate = PyGILState_Ensure();

    npy_intp dims[] = {no_strings};
    PyObject* ret = PyArray_SimpleNew(1, dims, NPY_OBJECT);
    CHECK(ret, "allocating numpy array failed");
    int err;

    PyObject* pd_mod = PyImport_ImportModule("pandas");
    CHECK(pd_mod, "importing pandas module failed");
    PyObject* na_obj = PyObject_GetAttrString(pd_mod, "NA");
    CHECK(na_obj, "getting pd.NA failed");

    for (int64_t i = 0; i < no_strings; ++i) {
        PyObject* s = PyUnicode_FromStringAndSize(
            buffer + offset_table[i], offset_table[i + 1] - offset_table[i]);
        CHECK(s, "creating Python string/unicode object failed");
        auto p = PyArray_GETPTR1((PyArrayObject*)ret, i);
        CHECK(p, "getting offset in numpy array failed");
        if (!is_na(null_bitmap, i))
            err = PyArray_SETITEM((PyArrayObject*)ret, (char*)p, s);
        else
            err = PyArray_SETITEM((PyArrayObject*)ret, (char*)p, na_obj);
        CHECK(err == 0, "setting item in numpy array failed");
        Py_DECREF(s);
    }

    PyObject* str_arr_obj =
        PyObject_CallMethod(pd_mod, "array", "Osl", ret, "string", 0);

    Py_DECREF(pd_mod);
    Py_DECREF(na_obj);
    Py_DECREF(ret);
    PyGILState_Release(gilstate);
    return str_arr_obj;
#undef CHECK
}

// helper functions for call Numpy APIs
npy_intp array_size(PyArrayObject* arr) {
    // std::cout << "get size\n";
    return PyArray_SIZE(arr);
}

void* array_getptr1(PyArrayObject* arr, npy_intp ind) {
    // std::cout << "get array ptr " << ind << '\n';
    return PyArray_GETPTR1(arr, ind);
}

void array_setitem(PyArrayObject* arr, char* p, PyObject* s) {
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        return;                        \
    }
    // std::cout << "get array ptr " << ind << '\n';
    int err = PyArray_SETITEM(arr, p, s);
    CHECK(err == 0, "setting item in numpy array failed");
    return;
#undef CHECK
}

void mask_arr_to_bitmap(uint8_t* bitmap_arr, uint8_t* mask_arr, int64_t n) {
    for (int i = 0; i < n; i++)
        bitmap_arr[i / 8] ^=
            static_cast<uint8_t>(-static_cast<uint8_t>(mask_arr[i] == 0) ^
                                 bitmap_arr[i / 8]) &
            kBitmask[i % 8];
}

int is_bool_array(PyArrayObject* arr) {
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        PyGILState_Release(gilstate);  \
        return false;                  \
    }
    auto gilstate = PyGILState_Ensure();

    PyArray_Descr* dtype = PyArray_DTYPE(arr);
    CHECK(dtype, "getting dtype failed");
    // printf("dtype kind %c type %c\n", dtype->kind, dtype->type);

    // returning int instead of bool to avoid potential bool call convention
    // issues
    int res = dtype->kind == 'b';
    PyGILState_Release(gilstate);
    return res;
#undef CHECK
}

int is_pd_boolean_array(PyObject* arr) {
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        PyGILState_Release(gilstate);  \
        return false;                  \
    }

    auto gilstate = PyGILState_Ensure();
    // pd.arrays.BooleanArray
    PyObject* pd_mod = PyImport_ImportModule("pandas");
    CHECK(pd_mod, "importing pandas module failed");
    PyObject* pd_arrays_obj = PyObject_GetAttrString(pd_mod, "arrays");
    CHECK(pd_arrays_obj, "getting pd.arrays failed");
    PyObject* pd_arrays_bool_arr_obj =
        PyObject_GetAttrString(pd_arrays_obj, "BooleanArray");
    CHECK(pd_arrays_obj, "getting pd.arrays.BooleanArray failed");

    // isinstance(arr, BooleanArray)
    int ret = PyObject_IsInstance(arr, pd_arrays_bool_arr_obj);
    CHECK(ret >= 0, "isinstance fails");

    Py_DECREF(pd_mod);
    Py_DECREF(pd_arrays_obj);
    Py_DECREF(pd_arrays_bool_arr_obj);
    PyGILState_Release(gilstate);
    return ret;

#undef CHECK
}

void unbox_bool_array_obj(PyArrayObject* arr, uint8_t* data, uint8_t* bitmap,
                          int64_t n) {
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        std::cerr << msg << std::endl; \
        PyGILState_Release(gilstate);  \
        return;                        \
    }
    auto gilstate = PyGILState_Ensure();

    // get pd.NA object to check for new NA kind
    // simple equality check is enough since the object is a singleton
    // example:
    // https://github.com/pandas-dev/pandas/blob/fcadff30da9feb3edb3acda662ff6143b7cb2d9f/pandas/_libs/missing.pyx#L57
    PyObject* pd_mod = PyImport_ImportModule("pandas");
    CHECK(pd_mod, "importing pandas module failed");
    PyObject* C_NA = PyObject_GetAttrString(pd_mod, "NA");
    CHECK(C_NA, "getting pd.NA failed");

    for (uint64_t i = 0; i < uint64_t(n); ++i) {
        auto p = PyArray_GETPTR1((PyArrayObject*)arr, i);
        CHECK(p, "getting offset in numpy array failed");
        PyObject* s = PyArray_GETITEM(arr, (const char*)p);
        CHECK(s, "getting element failed");
        // Pandas stores NA as either None or nan
        if (s == Py_None ||
            (PyFloat_Check(s) && std::isnan(PyFloat_AsDouble(s))) ||
            s == C_NA) {
            // null bit
            ClearBit(bitmap, i);
            data[i] = 0;
        } else {
            SetBit(bitmap, i);
            int is_true = PyObject_IsTrue(s);
            CHECK(is_true != -1, "invalid bool element");
            data[i] = (uint8_t)is_true;
        }
        Py_DECREF(s);
    }

    Py_DECREF(pd_mod);
    Py_DECREF(C_NA);
    PyGILState_Release(gilstate);
#undef CHECK
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
            for (uint64_t k = data_offsets[j]; k < data_offsets[j + 1]; k++)
                std::cout << data[k] << " ";
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
        output[2 * i] = hex_values[c >> 4];
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
            if (data >= end) break;
        }
        CHECK((end - data) >= 2,
              "bytes.fromhex, must provide two hex values per byte");
        top = _PyLong_DigitValue[*data++];
        bot = _PyLong_DigitValue[*data++];
        CHECK(top < 16, "bytes.fromhex: unsupport character");
        CHECK(bot < 16, "bytes.fromhex: unsupport character");
        output[length++] = (unsigned char)((top << 4) + bot);
    }
    return length;
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
