// Copyright (C) 2019 Bodo Inc. All rights reserved.
#ifndef BODO_COMMON_H_INCLUDED_
#define BODO_COMMON_H_INCLUDED_

#if defined(__GNUC__)
#define __UNUSED__ __attribute__((unused))
#else
#define __UNUSED__
#endif

#include <Python.h>
#include <arrow/api.h>
#include <vector>
#include "_meminfo.h"

#define ALIGNMENT 64  // preferred alignment for AVX512

inline void Bodo_PyErr_SetString(PyObject* type, const char* message) {
    std::cerr << "BodoRuntimeCppError, setting PyErr_SetString to " << message
              << "\n";
    PyErr_SetString(type, message);
}

// get memory alloc/free info from _meminfo.h
size_t get_stats_alloc();
size_t get_stats_free();
size_t get_stats_mi_alloc();
size_t get_stats_mi_free();

struct Bodo_CTypes {
    enum CTypeEnum {
        INT8 = 0,
        UINT8 = 1,
        INT32 = 2,
        UINT32 = 3,
        INT64 = 4,
        UINT64 = 7,
        FLOAT32 = 5,
        FLOAT64 = 6,
        INT16 = 8,
        UINT16 = 9,
        STRING = 10,
        _BOOL = 11,
        DECIMAL = 12,
        DATE = 13,
        DATETIME = 14,
        TIMEDELTA = 15,
        INT128 = 16,
        LIST_STRING = 17,
        LIST = 18,    // for nested datastructures, maps to Arrow List
        STRUCT = 19,  // for nested datastructures, maps to Arrow Struct
        _numtypes
    };
};

/** Getting the expression of a T value as a vector of characters
 *
 * The template paramter is T.
 * @param val the value in the type T.
 * @return the vector of characters on output
 */
template <typename T>
inline std::vector<char> GetCharVector(T const& val) {
    const T* valptr = &val;
    const char* charptr = (char*)valptr;
    std::vector<char> V(sizeof(T));
    for (size_t u = 0; u < sizeof(T); u++) V[u] = charptr[u];
    return V;
}

#define BYTES_PER_DECIMAL 16

struct decimal_value_cpp {
    int64_t low;
    int64_t high;
};

/* The NaN entry used in the case a normal value is not available.
 *
 * The choice are done in following way:
 * ---int8_t / int16_t / int32_t / int64_t : return a -1 value
 * ---uint8_t / uint16_t / uint32_t / uint64_t : return a 0 value
 * ---float / double : return a NaN
 * This is obviously not perfect as -1 can be a legitimate value, but here goes.
 *
 * @param the dtype used.
 * @return the list of characters in output.
 */
inline std::vector<char> RetrieveNaNentry(Bodo_CTypes::CTypeEnum const& dtype) {
    if (dtype == Bodo_CTypes::_BOOL) return GetCharVector<bool>(false);
    if (dtype == Bodo_CTypes::INT8) return GetCharVector<int8_t>(-1);
    if (dtype == Bodo_CTypes::UINT8) return GetCharVector<uint8_t>(0);
    if (dtype == Bodo_CTypes::INT16) return GetCharVector<int16_t>(-1);
    if (dtype == Bodo_CTypes::UINT16) return GetCharVector<uint16_t>(0);
    if (dtype == Bodo_CTypes::INT32) return GetCharVector<int32_t>(-1);
    if (dtype == Bodo_CTypes::UINT32) return GetCharVector<uint32_t>(0);
    if (dtype == Bodo_CTypes::INT64) return GetCharVector<int64_t>(-1);
    if (dtype == Bodo_CTypes::UINT64) return GetCharVector<uint64_t>(0);
    if (dtype == Bodo_CTypes::DATE) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "In DATE case missing values are handled by "
                             "NULLABLE_INT_BOOL so this case is impossible");
    }
    if (dtype == Bodo_CTypes::DATETIME || dtype == Bodo_CTypes::TIMEDELTA)
        return GetCharVector<int64_t>(std::numeric_limits<int64_t>::min());
    if (dtype == Bodo_CTypes::FLOAT32)
        return GetCharVector<float>(std::nanf("1"));
    if (dtype == Bodo_CTypes::FLOAT64)
        return GetCharVector<double>(std::nan("1"));
    if (dtype == Bodo_CTypes::DECIMAL) {
        // Normally the null value of decimal_value should never show up
        // anywhere. A value is assigned for simplicity of the code
        decimal_value_cpp e_val{0, 0};
        return GetCharVector<decimal_value_cpp>(e_val);
    }
    return {};
}

// for numpy arrays, this maps dtype to sizeof(dtype)
extern std::vector<size_t> numpy_item_size;

/**
 * @brief enum for array types supported by Bodo
 *
 */
struct bodo_array_type {
    enum arr_type_enum {
        NUMPY = 0,
        STRING = 1,
        NULLABLE_INT_BOOL = 2,  // nullable int or bool
        LIST_STRING = 3,        // list_string_array_type
        ARROW = 4,              // Arrow Array
        CATEGORICAL = 5,
        ARRAY_ITEM = 6,
        // string_array_split_view_type, etc.
    };
};

/**
 * @brief generic struct that holds info of Bodo arrays to enable communication.
 *
 * The column of a dataframe.
 *
 * Case of NUMPY column:
 * --- Only the data1 array is used.
 * --- length is the number of rows.
 * Case of NULLABLE_INT_BOOL:
 * --- The data1 is used for the data
 * --- null_bitmask is the mask
 * --- length is the number of rows.
 * Case of STRING:
 * --- The length is the number of rows.
 * --- The n_sub_elems is the total number of characters
 * --- data1 is for the characters
 * --- data2 is for the index offsets
 * --- null_bitmask is for the missing entries
 * Case of LIST_STRING:
 * --- The length is the number of rows.
 * --- The n_sub_elems is the total number of strings.
 * --- The n_sub_sub_elems is the total number of characters
 * --- data1 is the characters
 * --- data2 is for the data_offsets
 * --- data3 is for the index_offsets
 * --- null_bitmask for whether the data is missing or not.
 */
struct array_info {
    bodo_array_type::arr_type_enum arr_type;
    Bodo_CTypes::CTypeEnum dtype;
    int64_t length;       // number of elements in the array (not bytes)
    int64_t n_sub_elems;  // number of sub-elements for variable length arrays,
                          // e.g. characters in string array
    int64_t n_sub_sub_elems;  // second level of subelements (e.g. for the
                              // list_string_array_type)
    // data1 is the main data pointer. some arrays have multiple data pointers
    // e.g. string offsets
    char* data1;
    char* data2;
    char* data3;
    char* null_bitmask;      // for nullable arrays like strings
    char* sub_null_bitmask;  // for second level nullable like list-string
    NRT_MemInfo* meminfo;
    NRT_MemInfo* meminfo_bitmask;
    std::shared_ptr<arrow::Array> array;
    int32_t precision;       // for array of decimals
    int32_t scale;           // for array of decimals
    int64_t num_categories;  // for categorical arrays
    // TODO: shape/stride for multi-dim arrays
    explicit array_info(bodo_array_type::arr_type_enum _arr_type,
                        Bodo_CTypes::CTypeEnum _dtype, int64_t _length,
                        int64_t _n_sub_elems, int64_t _n_sub_sub_elems,
                        char* _data1, char* _data2, char* _data3,
                        char* _null_bitmask, char* _sub_null_bitmask,
                        NRT_MemInfo* _meminfo, NRT_MemInfo* _meminfo_bitmask,
                        std::shared_ptr<arrow::Array> _array = nullptr,
                        int32_t _precision = 0, int32_t _scale = 0,
                        int64_t _num_categories = 0)
        : arr_type(_arr_type),
          dtype(_dtype),
          length(_length),
          n_sub_elems(_n_sub_elems),
          n_sub_sub_elems(_n_sub_sub_elems),
          data1(_data1),
          data2(_data2),
          data3(_data3),
          null_bitmask(_null_bitmask),
          sub_null_bitmask(_sub_null_bitmask),
          meminfo(_meminfo),
          meminfo_bitmask(_meminfo_bitmask),
          array(_array),
          precision(_precision),
          scale(_scale),
          num_categories(_num_categories) {}

    template <typename T>
    T& at(size_t idx) {
        return ((T*)data1)[idx];
    }

    array_info& operator=(
        array_info&& other) noexcept;  // move assignment operator
};

struct table_info {
    std::vector<array_info*> columns;
    // this is set and used by groupby to avoid putting additional info in
    // multi_col_key (which is only needed when key equality is checked but not
    // for hashing)
    // TODO consider passing 'num_keys' to the constructor
    int64_t num_keys;
    table_info() {}
    explicit table_info(std::vector<array_info*>& _columns)
        : columns(_columns) {}

    int64_t nrows() const { return columns[0]->length; }
    int64_t ncols() const { return columns.size(); }
    array_info* operator[](size_t idx) { return columns[idx]; }
    const array_info* operator[](size_t idx) const { return columns[idx]; }
};

/// Initialize numpy_item_size and verify size of dtypes
void bodo_common_init();

array_info* alloc_array(int64_t length, int64_t n_sub_elems,
                        int64_t n_sub_sub_elems,
                        bodo_array_type::arr_type_enum arr_type,
                        Bodo_CTypes::CTypeEnum dtype, int64_t extra_null_bytes,
                        int64_t num_categories);

array_info* alloc_numpy(int64_t length, Bodo_CTypes::CTypeEnum typ_enum);

array_info* alloc_array_item(int64_t n_arrays, int64_t n_total_items,
                             Bodo_CTypes::CTypeEnum dtype);
array_info* alloc_categorical(int64_t length, Bodo_CTypes::CTypeEnum typ_enum,
                              int64_t num_categories);

array_info* alloc_nullable_array(int64_t length,
                                 Bodo_CTypes::CTypeEnum typ_enum,
                                 int64_t extra_null_bytes);

array_info* alloc_string_array(int64_t length, int64_t n_chars,
                               int64_t extra_null_bytes);

array_info* alloc_list_string_array(int64_t n_lists, int64_t n_strings,
                                    int64_t n_chars, int64_t extra_null_bytes);

array_info* copy_array(array_info* arr);

/**
 * Free underlying array of array_info pointer and delete the pointer
 */
void delete_info_decref_array(array_info* arr);

/**
 * delete table pointer and its column array_info pointers (but not the arrays).
 */
void delete_table(table_info* table);

/**
 * Decref all arrays of a table and delete the table.
 */
void delete_table_decref_arrays(table_info* table);

/**
 * decref Bodo array and free all memory if refcount is zero.
 */
void decref_array(array_info* arr);

/**
 * incref Bodo array
 */
void incref_array(array_info* arr);

/**
 * Decref list of string array and free all memory in refcount is zero.
 */
void decref_list_string_array(NRT_MemInfo* meminfo);

extern "C" {

struct numpy_arr_payload {
    NRT_MemInfo* meminfo;
    PyObject* parent;
    int64_t nitems;
    int64_t itemsize;
    char* data;
    int64_t shape;
    int64_t strides;

    numpy_arr_payload(NRT_MemInfo* _meminfo, PyObject* _parent, int64_t _nitems,
                      int64_t _itemsize, char* _data, int64_t _shape,
                      int64_t _strides)
        : meminfo(_meminfo),
          parent(_parent),
          nitems(_nitems),
          itemsize(_itemsize),
          data(_data),
          shape(_shape),
          strides(_strides) {}
};

// XXX: equivalent to payload data model in str_arr_ext.py
struct str_arr_payload {
    int64_t num_strings;
    int32_t* offsets;
    char* data;
    uint8_t* null_bitmap;
};

struct array_item_arr_payload {
    int64_t n_arrays;
    // currently this is not general. this is specific for arrays whose model
    // is a meminfo (like StringArray)
    NRT_MemInfo* data;
    numpy_arr_payload offsets;
    numpy_arr_payload null_bitmap;
};

/**
 * @brief array(item) array payload with numpy data
 * TODO: create a general templated payload and avoid duplication
 *
 */
struct array_item_arr_numpy_payload {
    int64_t n_arrays;
    numpy_arr_payload data;
    numpy_arr_payload offsets;
    numpy_arr_payload null_bitmap;
};

// XXX: equivalent to payload data model in split_impl.py
struct str_arr_split_view_payload {
    uint32_t* index_offsets;
    uint32_t* data_offsets;
    uint8_t* null_bitmap;
};

numpy_arr_payload allocate_numpy_payload(int64_t length,
                                         Bodo_CTypes::CTypeEnum typ_enum);

void dtor_string_array(str_arr_payload* in_str_arr, int64_t size, void* in);
void dtor_array_item_array(array_item_arr_numpy_payload* payload, int64_t size, void* in);
NRT_MemInfo* alloc_array_item_arr_meminfo();


void allocate_list_string_array(int64_t n_lists, int64_t n_strings,
                                int64_t n_chars, int64_t extra_null_bytes,
                                array_item_arr_payload* payload,
                                str_arr_payload* sub_payload);

Bodo_CTypes::CTypeEnum arrow_to_bodo_type(arrow::Type::type type);

void nested_array_to_c(std::shared_ptr<arrow::Array> array, int64_t* lengths,
                       array_info** infos, int64_t& lengths_pos,
                       int64_t& infos_pos);

// copied from Arrow bit_util.h
// Bitmask selecting the k-th bit in a byte
static constexpr uint8_t kBitmask[] = {1, 2, 4, 8, 16, 32, 64, 128};

inline bool GetBit(const uint8_t* bits, uint64_t i) {
    return (bits[i >> 3] >> (i & 0x07)) & 1;
}

inline void SetBitTo(uint8_t* bits, int64_t i, bool bit_is_set) {
    bits[i / 8] ^=
        static_cast<uint8_t>(-static_cast<uint8_t>(bit_is_set) ^ bits[i / 8]) &
        kBitmask[i % 8];
}

inline void InitializeBitMask(uint8_t* bits, size_t length, bool val) {
    size_t n_bytes = (length + 7) >> 3;
    if (!val)
        memset(bits, 0, n_bytes);
    else
        memset(bits, 0xff, n_bytes);
}

inline bool is_na(const uint8_t* null_bitmap, int64_t i) {
    return (null_bitmap[i / 8] & kBitmask[i % 8]) == 0;
}
}

#endif /* BODO_COMMON_H_INCLUDED_ */
