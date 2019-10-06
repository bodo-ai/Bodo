#ifndef BODO_COMMON_H_
#define BODO_COMMON_H_

#if defined(__GNUC__)
#define __UNUSED__ __attribute__((unused))
#else
#define __UNUSED__
#endif


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


/**
 * @brief enum for array types supported by Bodo
 * 
 */
struct bodo_array_type {
    enum arr_type_enum {
        NUMPY = 0,
        STRING = 1,
        INT_NULLABLE = 2,
        BOOL_NULLABLE = 3,
        // TODO: add all Bodo arrays list_string_array_type, string_array_split_view_type, etc.
    };
};


/**
 * @brief generic struct that holds info of Bodo arrays to enable communication.
 */
struct array_info {
    bodo_array_type::arr_type_enum arr_type;
    Bodo_CTypes::CTypeEnum dtype;
    int64_t length; // number of elements in the array (not bytes)
    int64_t n_sub_elems; // number of sub-elements for variable length arrays, e.g. characters in string array
    // data1 is the main data pointer. some arrays have multiple data pointers e.g. string offsets
    char* data1;
    char* data2;
    char* data3;
    char* null_bitmask;  // for nullable arrays like strings
    // TODO: shape/stride for multi-dim arrays
    explicit array_info(bodo_array_type::arr_type_enum _arr_type, Bodo_CTypes::CTypeEnum _dtype,
        int64_t _length, int64_t _n_sub_elems, char* _data1, char* _data2, char* _data3, char* _null_bitmask): 
           arr_type(_arr_type), dtype(_dtype), length(_length), n_sub_elems(_n_sub_elems),
           data1(_data1), data2(_data2), data3(_data3), null_bitmask(_null_bitmask) {}
};


#define DEC_MOD_METHOD(func) PyObject_SetAttrString(m, #func, PyLong_FromVoidPtr((void*)(&func)))

#endif /* BODO_COMMON_H_ */
