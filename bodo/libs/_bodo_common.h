// Copyright (C) 2019 Bodo Inc. All rights reserved.
#ifndef BODO_COMMON_H_
#define BODO_COMMON_H_

#if defined(__GNUC__)
#define __UNUSED__ __attribute__((unused))
#else
#define __UNUSED__
#endif

#include <Python.h>
#include <vector>
#include "_meminfo.h"

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
        _numtypes
    };
};

#define BYTES_PER_DECIMAL 16

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
        // TODO: add all Bodo arrays list_string_array_type,
        // string_array_split_view_type, etc.
    };
};

/**
 * @brief generic struct that holds info of Bodo arrays to enable communication.
 */
struct array_info {
    bodo_array_type::arr_type_enum arr_type;
    Bodo_CTypes::CTypeEnum dtype;
    int64_t length;       // number of elements in the array (not bytes)
    int64_t n_sub_elems;  // number of sub-elements for variable length arrays,
                          // e.g. characters in string array
    // data1 is the main data pointer. some arrays have multiple data pointers
    // e.g. string offsets
    char* data1;
    char* data2;
    char* data3;
    char* null_bitmask;  // for nullable arrays like strings
    NRT_MemInfo* meminfo;
    NRT_MemInfo* meminfo_bitmask;
    int32_t precision;  // for array of decimals
    int32_t scale;      // for array of decimals
    // TODO: shape/stride for multi-dim arrays
    explicit array_info(bodo_array_type::arr_type_enum _arr_type,
                        Bodo_CTypes::CTypeEnum _dtype, int64_t _length,
                        int64_t _n_sub_elems, char* _data1, char* _data2,
                        char* _data3, char* _null_bitmask,
                        NRT_MemInfo* _meminfo, NRT_MemInfo* _meminfo_bitmask,
                        int32_t _precision = 0, int32_t _scale = 0)
        : arr_type(_arr_type),
          dtype(_dtype),
          length(_length),
          n_sub_elems(_n_sub_elems),
          data1(_data1),
          data2(_data2),
          data3(_data3),
          null_bitmask(_null_bitmask),
          meminfo(_meminfo),
          meminfo_bitmask(_meminfo_bitmask),
          precision(_precision),
          scale(_scale) {}

    template <typename T>
    T& at(size_t idx) {
        return ((T*)data1)[idx];
    }
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
                        bodo_array_type::arr_type_enum arr_type,
                        Bodo_CTypes::CTypeEnum dtype, int64_t extra_null_bytes);

array_info* alloc_numpy(int64_t length, Bodo_CTypes::CTypeEnum typ_enum);

array_info* alloc_nullable_array(int64_t length,
                                 Bodo_CTypes::CTypeEnum typ_enum,
                                 int64_t extra_null_bytes);

array_info* alloc_string_array(int64_t length, int64_t n_chars,
                               int64_t extra_null_bytes);

array_info* copy_array(array_info* arr);

void delete_table(table_info* table);

/**
 * Free all arrays of a table and delete the table.
 */
void delete_table_free_arrays(table_info* table);

/**
 * Free all memory of an array.
 */
void free_array(array_info* arr);

inline void Bodo_PyErr_SetString(PyObject* type, const char* message) {
    std::cerr << "BodoRuntimeCppError, setting PyErr_SetString to " << message
              << "\n";
    PyErr_SetString(type, message);
}

extern "C" {

// XXX: equivalent to payload data model in str_arr_ext.py
struct str_arr_payload {
    uint32_t* offsets;
    char* data;
    uint8_t* null_bitmap;
};

// XXX: equivalent to payload data model in list_str_arr_ext.py
struct list_str_arr_payload {
    char* data;
    uint32_t* data_offsets;
    uint32_t* index_offsets;
    uint8_t* null_bitmap;
};

// XXX: equivalent to payload data model in split_impl.py
struct str_arr_split_view_payload {
    uint32_t* index_offsets;
    uint32_t* data_offsets;
    // uint8_t* null_bitmap;
};

void dtor_string_array(str_arr_payload* in_str_arr, int64_t size, void* in);

void dtor_list_string_array(list_str_arr_payload* in_list_str_arr, int64_t size,
                            void* in);

void allocate_string_array(uint32_t** offsets, char** data,
                           uint8_t** null_bitmap, int64_t num_strings,
                           int64_t total_size, int64_t extra_null_bytes);

void allocate_list_string_array(char** data, uint32_t** data_offsets,
                                uint32_t** index_offsets, uint8_t** null_bitmap,
                                int64_t num_lists, int64_t num_strings,
                                int64_t num_chars, int64_t extra_null_bytes);

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

#endif /* BODO_COMMON_H_ */
