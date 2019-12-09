// Copyright (C) 2019 Bodo Inc. All rights reserved.
#ifndef BODO_COMMON_H_
#define BODO_COMMON_H_

#if defined(__GNUC__)
#define __UNUSED__ __attribute__((unused))
#else
#define __UNUSED__
#endif

#include <vector>
#include "_meminfo.h"

// class CTypeEnum(Enum):
//     Int8 = 0
//     UInt8 = 1
//     Int32 = 2
//     UInt32 = 3
//     Int64 = 4
//     UInt64 = 5
//     Float32 = 6
//     Float64 = 7

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
    };
};

#define BODO_NUMPY_ARRAY_NUM_DTYPES 10

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
    // TODO: shape/stride for multi-dim arrays
    explicit array_info(bodo_array_type::arr_type_enum _arr_type,
                        Bodo_CTypes::CTypeEnum _dtype, int64_t _length,
                        int64_t _n_sub_elems, char* _data1, char* _data2,
                        char* _data3, char* _null_bitmask,
                        NRT_MemInfo* _meminfo, NRT_MemInfo* _meminfo_bitmask)
        : arr_type(_arr_type),
          dtype(_dtype),
          length(_length),
          n_sub_elems(_n_sub_elems),
          data1(_data1),
          data2(_data2),
          data3(_data3),
          null_bitmask(_null_bitmask),
          meminfo(_meminfo),
          meminfo_bitmask(_meminfo_bitmask) {}

    template<typename T>
    T& at(size_t idx) { return ((T*)data1)[idx]; }
};

struct table_info {
    std::vector<array_info*> columns;
    // this is set and used by groupby to avoid putting additional info in multi_col_key
    // (which is only needed when key equality is checked but not for hashing)
    // TODO consider passing 'num_keys' to the constructor
    int64_t num_keys;
    explicit table_info(std::vector<array_info*> &_columns)
        : columns(_columns) {}

    int64_t nrows() const { return columns[0]->length; }
    int64_t ncols() const { return columns.size(); }
    array_info* operator[](size_t idx) { return columns[idx]; }
};

#define DEC_MOD_METHOD(func) \
    PyObject_SetAttrString(m, #func, PyLong_FromVoidPtr((void*)(&func)))

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

void dtor_string_array(str_arr_payload* in_str_arr, int64_t size, void* in) {
    // printf("str arr dtor size: %lld\n", in_str_arr->size);
    // printf("num chars: %d\n", in_str_arr->offsets[in_str_arr->size]);
    delete[] in_str_arr->offsets;
    delete[] in_str_arr->data;
    if (in_str_arr->null_bitmap != nullptr)
        delete[] in_str_arr->null_bitmap;
    return;
}


void dtor_list_string_array(list_str_arr_payload* in_list_str_arr, int64_t size, void* in) {
    delete[] in_list_str_arr->data;
    delete[] in_list_str_arr->data_offsets;
    delete[] in_list_str_arr->index_offsets;
    if (in_list_str_arr->null_bitmap != nullptr)
        delete[] in_list_str_arr->null_bitmap;
    return;
}


void allocate_string_array(uint32_t** offsets, char** data,
                           uint8_t** null_bitmap, int64_t num_strings,
                           int64_t total_size, int64_t extra_null_bytes) {
    // std::cout << "allocating string array: " << num_strings << " " <<
    //                                                 total_size << std::endl;
    *offsets = new uint32_t[num_strings + 1];
    *data = new char[total_size];
    (*offsets)[0] = 0;
    (*offsets)[num_strings] =
        (uint32_t)total_size;  // in case total chars is read from here
    // allocate nulls
    int64_t n_bytes = ((num_strings + 7) >> 3) + extra_null_bytes;
    *null_bitmap = new uint8_t[(size_t)n_bytes];
    // set all bits to 1 indicating non-null as default
    memset(*null_bitmap, 0xff, n_bytes);
    // *data = (char*) new std::string("gggg");
    return;
}


void allocate_list_string_array(char** data, uint32_t** data_offsets, uint32_t** index_offsets,
                           uint8_t** null_bitmap, int64_t num_lists, int64_t num_strings,
                           int64_t num_chars, int64_t extra_null_bytes) {
    // std::cout << "allocating list string array: " << num_lists << " "<< num_strings << " " <<
    //                                                 num_chars << std::endl;
    *data = new char[num_chars];

    *data_offsets = new uint32_t[num_strings + 1];
    (*data_offsets)[0] = 0;
    (*data_offsets)[num_strings] =
        (uint32_t)num_chars;  // in case total chars is read from here

    *index_offsets = new uint32_t[num_lists + 1];
    (*index_offsets)[0] = 0;
    (*index_offsets)[num_lists] =
        (uint32_t)num_strings;  // in case total strings is read from here

    // allocate nulls
    int64_t n_bytes = ((num_lists + 7) >> 3) + extra_null_bytes;
    *null_bitmap = new uint8_t[(size_t)n_bytes];
    // set all bits to 1 indicating non-null as default
    memset(*null_bitmap, 0xff, n_bytes);
    return;
}


// copied from Arrow bit_util.h
// Bitmask selecting the k-th bit in a byte
static constexpr uint8_t kBitmask[] = {1, 2, 4, 8, 16, 32, 64, 128};

static inline bool GetBit(const uint8_t* bits, uint64_t i) {
    return (bits[i >> 3] >> (i & 0x07)) & 1;
}

static inline void SetBitTo(uint8_t* bits, int64_t i, bool bit_is_set) {
    bits[i / 8] ^=
        static_cast<uint8_t>(-static_cast<uint8_t>(bit_is_set) ^ bits[i / 8]) &
        kBitmask[i % 8];
}

static inline void InitializeBitMask(uint8_t* bits, size_t length, bool val) {
    size_t n_bytes = (length + 7) >> 3;
    if (!val)
        memset(bits, 0, n_bytes);
    else
        memset(bits, 0xff, n_bytes);
}

}

#endif /* BODO_COMMON_H_ */
