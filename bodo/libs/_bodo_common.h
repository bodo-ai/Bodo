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

#undef DEBUG_ARRAY_ACCESS

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

// use 64-bit offsets for string and nested arrays
typedef uint64_t offset_t;
#define OFFSET_BITWIDTH 64
#define Bodo_CType_offset Bodo_CTypes::CTypeEnum::UINT64

inline bool is_unsigned_integer(Bodo_CTypes::CTypeEnum typ) {
    if (typ == Bodo_CTypes::UINT8) return true;
    if (typ == Bodo_CTypes::UINT16) return true;
    if (typ == Bodo_CTypes::UINT32) return true;
    if (typ == Bodo_CTypes::UINT64) return true;
    return false;
}

inline bool is_integer(Bodo_CTypes::CTypeEnum typ) {
    if (is_unsigned_integer(typ)) return true;
    if (typ == Bodo_CTypes::INT8) return true;
    if (typ == Bodo_CTypes::INT16) return true;
    if (typ == Bodo_CTypes::INT32) return true;
    if (typ == Bodo_CTypes::INT64) return true;
    return false;
}

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
#ifdef DEBUG_ARRAY_ACCESS
        std::cout << "     at access SINGLE for idx=" << idx << "\n";
#endif
        return ((T*)data1)[idx];
    }

    bool get_null_bit(size_t idx) const {
        return GetBit((uint8_t*)null_bitmask, idx);
    }

    /**
     * Return code in position `idx` of categorical array as int64
     */
    int64_t get_code_as_int64(size_t idx) {
        if (arr_type != bodo_array_type::CATEGORICAL)
            throw std::runtime_error("get_code: not a categorical array");
        switch (dtype) {
            case Bodo_CTypes::INT8:
                return (int64_t)(this->at<int8_t>(idx));
            case Bodo_CTypes::INT16:
                return (int64_t)(this->at<int16_t>(idx));
            case Bodo_CTypes::INT32:
                return (int64_t)(this->at<int32_t>(idx));
            case Bodo_CTypes::INT64:
                return this->at<int64_t>(idx);
            default:
                throw std::runtime_error("get_code: codes have unrecognized dtype");
        }
    }

    /**
     * Return string representation of value in position `idx` of this array.
     */
    std::string val_to_str(size_t idx) {
        switch (dtype) {
            case Bodo_CTypes::INT8:
                return std::to_string(this->at<int8_t>(idx));
            case Bodo_CTypes::UINT8:
                return std::to_string(this->at<uint8_t>(idx));
            case Bodo_CTypes::INT32:
                return std::to_string(this->at<int32_t>(idx));
            case Bodo_CTypes::UINT32:
                return std::to_string(this->at<uint32_t>(idx));
            case Bodo_CTypes::INT64:
                return std::to_string(this->at<int64_t>(idx));
            case Bodo_CTypes::UINT64:
                return std::to_string(this->at<uint64_t>(idx));
            case Bodo_CTypes::FLOAT32:
                return std::to_string(this->at<float>(idx));
            case Bodo_CTypes::FLOAT64:
                return std::to_string(this->at<double>(idx));
            case Bodo_CTypes::INT16:
                return std::to_string(this->at<int16_t>(idx));
            case Bodo_CTypes::UINT16:
                return std::to_string(this->at<uint16_t>(idx));
            case Bodo_CTypes::STRING: {
                offset_t* offsets = (offset_t*)data2;
                return std::string(data1 + offsets[idx], offsets[idx + 1] - offsets[idx]);
            }
            case Bodo_CTypes::DATE: {
                int64_t val = this->at<int64_t>(idx);
                int year = val >> 32;
                int month = (val >> 16) & 0xFFFF;
                int day = val & 0xFFFF;
                std::string date_str;
                date_str.reserve(10);
                date_str += std::to_string(year) + "-";
                if (month < 10) date_str += "0";
                date_str += std::to_string(month) + "-";
                if (day < 10) date_str += "0";
                date_str += std::to_string(day);
                return date_str;
            }
            case Bodo_CTypes::_BOOL:
                if (this->at<bool>(idx)) return "True";
                return "False";
            default: {
                std::vector<char> error_msg(100);
                sprintf(error_msg.data(), "val_to_str not implemented for dtype %d", dtype);
                throw std::runtime_error(error_msg.data());
            }
        }
    }

    void set_null_bit(size_t idx, bool bit) {
        SetBitTo((uint8_t*)null_bitmask, idx, bit);
    }

    array_info& operator=(
        array_info&& other) noexcept;  // move assignment operator
};

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

/* Store several array_info in one structure and assign them
   as required.
   This is for pivot_table and crosstab.
   The rows of the pivot_table / crosstab correspond to
   the index. But the columns force us to create a multiple_array_info
   with each column corresponding to an entry.
   -
   The vect_arr contains those columns.
   the vect_access contains whether we have accessed to the entry or not.
   It is important to keep track of this info since in contrast to the
   groupby, some info may be missing.
*/
struct multiple_array_info {
    bodo_array_type::arr_type_enum arr_type;
    Bodo_CTypes::CTypeEnum dtype;
    int64_t num_categories;  // for categorical arrays
    int64_t length;
    int64_t length_loc;
    int64_t n_pivot;
    std::vector<array_info*> vect_arr;
    std::vector<array_info*> vect_access;
    explicit multiple_array_info(std::vector<array_info*> _vect_arr)
        : vect_arr(_vect_arr) {
        n_pivot = vect_arr.size();
        length_loc = vect_arr[0]->length;
        length = length_loc * n_pivot;
        num_categories = vect_arr[0]->num_categories;
        arr_type = vect_arr[0]->arr_type;
        dtype = vect_arr[0]->dtype;
        int n_access = (n_pivot + 7) >> 3;
        for (int i_access = 0; i_access < n_access; i_access++) {
            array_info* arr_access =
                alloc_numpy(length_loc, Bodo_CTypes::UINT8);
            std::fill((uint8_t*)arr_access->data1,
                      (uint8_t*)arr_access->data1 + length_loc, 0);
            vect_access.push_back(arr_access);
        }
    }
    explicit multiple_array_info(std::vector<array_info*> _vect_arr,
                                 std::vector<array_info*> _vect_access)
        : vect_arr(_vect_arr), vect_access(_vect_access) {
        n_pivot = vect_arr.size();
        length_loc = vect_arr[0]->length;
        length = length_loc * n_pivot;
        num_categories = vect_arr[0]->num_categories;
        arr_type = vect_arr[0]->arr_type;
        dtype = vect_arr[0]->dtype;
    }
    template <typename T>
    T& at(size_t idx) {
        // index for the value itself
        size_t i_arr = idx % n_pivot;
        size_t idx_loc = idx / n_pivot;
        // setting up the access mask. If accessed the value stops to be missing
        int64_t i_arr_access = i_arr / 8;
        int64_t pos_arr_access = i_arr % 8;
        uint8_t* ptr = (uint8_t*)vect_access[i_arr_access]->data1 + idx_loc;
#ifdef DEBUG_ARRAY_ACCESS
        std::cout << "     at access MULT for idx=" << idx << " i_arr=" << i_arr
                  << " idx_loc=" << idx_loc << "\n";
#endif
        SetBitTo(ptr, pos_arr_access, true);
        return ((T*)vect_arr[i_arr]->data1)[idx_loc];
    }

    bool get_access_bit(size_t idx) {
        // index for the value itself
        size_t i_arr = idx % n_pivot;
        size_t idx_loc = idx / n_pivot;
        // setting up the access mask. If accessed the value stops to be missing
        int64_t i_arr_access = i_arr / 8;
        int64_t pos_arr_access = i_arr % 8;
        uint8_t* ptr = (uint8_t*)vect_access[i_arr_access]->data1 + idx_loc;
        return GetBit(ptr, pos_arr_access);
    }

    bool get_null_bit(size_t idx) const {
        size_t i_arr = idx % n_pivot;
        size_t idx_loc = idx / n_pivot;
        return GetBit((uint8_t*)vect_arr[i_arr]->null_bitmask, idx_loc);
    }

    void set_null_bit(size_t idx, bool bit) {
        size_t i_arr = idx % n_pivot;
        size_t idx_loc = idx / n_pivot;
        SetBitTo((uint8_t*)vect_arr[i_arr]->null_bitmask, idx_loc, bit);
    }

    multiple_array_info& operator=(
        multiple_array_info&& other) noexcept;  // move assignment operator
};

template <typename T>
struct is_multiple_array {
    static const bool value = false;
};

template <>
struct is_multiple_array<multiple_array_info> {
    static const bool value = true;
};

/* The "get-value" functionality for multiple_array_info and array_info.
   This is the equivalent of at functionality.
   We cannot use at(idx) statements.
 */
template <typename ARRAY, typename T>
inline typename std::enable_if<is_multiple_array<ARRAY>::value, T&>::type getv(
    ARRAY* arr, size_t idx) {
    size_t i_arr = idx % arr->n_pivot;
    size_t idx_loc = idx / arr->n_pivot;
    int64_t i_arr_access = i_arr / 8;
    int64_t pos_arr_access = i_arr % 8;
    uint8_t* ptr = (uint8_t*)arr->vect_access[i_arr_access]->data1 + idx_loc;
    SetBitTo(ptr, pos_arr_access, true);
    return ((T*)arr->vect_arr[i_arr]->data1)[idx_loc];
}

template <typename ARRAY, typename T>
inline typename std::enable_if<!is_multiple_array<ARRAY>::value, T&>::type getv(
    ARRAY* arr, size_t idx) {
    return ((T*)arr->data1)[idx];
}

struct mpi_comm_info {
    int n_pes;
    std::vector<array_info*> arrays;
    size_t n_rows;
    bool has_nulls;
    // generally required MPI counts
    std::vector<int64_t> send_count;
    std::vector<int64_t> recv_count;
    std::vector<int64_t> send_disp;
    std::vector<int64_t> recv_disp;
    // counts required for string arrays
    std::vector<std::vector<int64_t>> send_count_sub;
    std::vector<std::vector<int64_t>> recv_count_sub;
    std::vector<std::vector<int64_t>> send_disp_sub;
    std::vector<std::vector<int64_t>> recv_disp_sub;
    // counts required for string list arrays
    std::vector<std::vector<int64_t>> send_count_sub_sub;
    std::vector<std::vector<int64_t>> recv_count_sub_sub;
    std::vector<std::vector<int64_t>> send_disp_sub_sub;
    std::vector<std::vector<int64_t>> recv_disp_sub_sub;
    // counts for arrays with null bitmask
    std::vector<int64_t> send_count_null;
    std::vector<int64_t> recv_count_null;
    std::vector<int64_t> send_disp_null;
    std::vector<int64_t> recv_disp_null;
    size_t n_null_bytes;

    explicit mpi_comm_info(std::vector<array_info*>& _arrays);

    void set_counts(uint32_t* hashes);
};

struct table_info {
    std::vector<array_info*> columns;
    // this is set and used by groupby to avoid putting additional info in
    // multi_col_key (which is only needed when key equality is checked but not
    // for hashing)
    // TODO consider passing 'num_keys' to the constructor
    int64_t num_keys;
    // keep shuffle info to be able to reverse the shuffle if necessary
    // currently used in groupby apply
    // TODO: refactor out?
    mpi_comm_info* comm_info;
    uint32_t* hashes;
    int id;
    table_info() {}
    explicit table_info(std::vector<array_info*>& _columns)
        : columns(_columns) {}

    int64_t nrows() const { return columns[0]->length; }
    int64_t ncols() const { return columns.size(); }
    array_info* operator[](size_t idx) { return columns[idx]; }
    const array_info* operator[](size_t idx) const { return columns[idx]; }
};

/* Compute the total memory of the table accross all processors.
 *
 * @param table : The input table
 * @return the total size of the table all over the processors
 */
int64_t table_global_memory_size(table_info* table);

/// Initialize numpy_item_size and verify size of dtypes
void bodo_common_init();

array_info* copy_array(array_info* arr);

NRT_MemInfo* alloc_meminfo(int64_t length);

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

void decref_numpy_payload(numpy_arr_payload arr);

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
    offset_t* index_offsets;
    offset_t* data_offsets;
    uint8_t* null_bitmap;
};

numpy_arr_payload allocate_numpy_payload(int64_t length,
                                         Bodo_CTypes::CTypeEnum typ_enum);

void dtor_array_item_array(array_item_arr_numpy_payload* payload, int64_t size,
                           void* in);
NRT_MemInfo* alloc_array_item_arr_meminfo();

Bodo_CTypes::CTypeEnum arrow_to_bodo_type(arrow::Type::type type);

void nested_array_to_c(std::shared_ptr<arrow::Array> array, int64_t* lengths,
                       array_info** infos, int64_t& lengths_pos,
                       int64_t& infos_pos);

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
