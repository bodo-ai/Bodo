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
#include "simd-block-fixed-fpp.h"
#include "tracing.h"

#define ALIGNMENT 64  // preferred alignment for AVX512

// Convenience macros from
// https://github.com/numba/numba/blob/main/numba/_pymodule.h
#define MOD_DEF(ob, name, doc, methods)                                     \
    {                                                                       \
        static struct PyModuleDef moduledef = {PyModuleDef_HEAD_INIT, name, \
                                               doc, -1, methods};           \
        ob = PyModule_Create(&moduledef);                                   \
    }

#define SetAttrStringFromVoidPtr(m, name)                 \
    do {                                                  \
        PyObject* tmp = PyLong_FromVoidPtr((void*)&name); \
        PyObject_SetAttrString(m, #name, tmp);            \
        Py_DECREF(tmp);                                   \
    } while (0)

#define SetAttrStringFromPyInit(m, name)       \
    do {                                       \
        PyObject* mod = PyInit_##name();       \
        PyObject_SetAttrString(m, #name, mod); \
        Py_DECREF(mod);                        \
    } while (0)

inline void Bodo_PyErr_SetString(PyObject* type, const char* message) {
    std::cerr << "BodoRuntimeCppError, setting PyErr_SetString to " << message
              << "\n";
    PyErr_SetString(type, message);
}

class BodoBuffer;

// get memory alloc/free info from _meminfo.h
size_t get_stats_alloc();
size_t get_stats_free();
size_t get_stats_mi_alloc();
size_t get_stats_mi_free();

// NOTE: should match CTypeEnum in utils/utils.py
// NOTE: should match Bodo_CTypes_names in _bodo_common.cpp
struct Bodo_CTypes {
    enum CTypeEnum {
        INT8 = 0,
        UINT8 = 1,
        INT32 = 2,
        UINT32 = 3,
        INT64 = 4,
        FLOAT32 = 5,
        FLOAT64 = 6,
        UINT64 = 7,
        INT16 = 8,
        UINT16 = 9,
        STRING = 10,
        _BOOL = 11,
        DECIMAL = 12,
        DATE = 13,
        TIME = 14,
        DATETIME = 15,
        TIMEDELTA = 16,
        INT128 = 17,
        LIST_STRING = 18,
        LIST = 19,    // for nested datastructures, maps to Arrow List
        STRUCT = 20,  // for nested datastructures, maps to Arrow Struct
        BINARY = 21,
        _numtypes
    };
};

typedef int32_t dict_indices_t;

// use 64-bit offsets for string and nested arrays
typedef uint64_t offset_t;
#define OFFSET_BITWIDTH 64
#define Bodo_CType_offset Bodo_CTypes::CTypeEnum::UINT64

std::unique_ptr<BodoBuffer> AllocateBodoBuffer(const int64_t size);
std::unique_ptr<BodoBuffer> AllocateBodoBuffer(const int64_t length,
                                               Bodo_CTypes::CTypeEnum typ_enum);

inline bool is_unsigned_integer(Bodo_CTypes::CTypeEnum typ) {
    if (typ == Bodo_CTypes::UINT8)
        return true;
    if (typ == Bodo_CTypes::UINT16)
        return true;
    if (typ == Bodo_CTypes::UINT32)
        return true;
    if (typ == Bodo_CTypes::UINT64)
        return true;
    return false;
}

inline bool is_integer(Bodo_CTypes::CTypeEnum typ) {
    if (is_unsigned_integer(typ))
        return true;
    if (typ == Bodo_CTypes::INT8)
        return true;
    if (typ == Bodo_CTypes::INT16)
        return true;
    if (typ == Bodo_CTypes::INT32)
        return true;
    if (typ == Bodo_CTypes::INT64)
        return true;
    if (typ == Bodo_CTypes::INT128)
        return true;
    return false;
}

/**
 * @brief Check if typ is FLOAT32 or FLOAT64.
 */
inline bool is_float(Bodo_CTypes::CTypeEnum typ) {
    return ((typ == Bodo_CTypes::FLOAT32) || (typ == Bodo_CTypes::FLOAT64));
}

/**
 * @brief Check if typ is an integer, floating or decimal type.
 */
inline bool is_numerical(Bodo_CTypes::CTypeEnum typ) {
    return (is_integer(typ) || is_float(typ) || (typ == Bodo_CTypes::DECIMAL));
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
    for (size_t u = 0; u < sizeof(T); u++)
        V[u] = charptr[u];
    return V;
}

#define BYTES_PER_DECIMAL 16

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
    if (dtype == Bodo_CTypes::_BOOL)
        return GetCharVector<bool>(false);
    if (dtype == Bodo_CTypes::INT8)
        return GetCharVector<int8_t>(-1);
    if (dtype == Bodo_CTypes::UINT8)
        return GetCharVector<uint8_t>(0);
    if (dtype == Bodo_CTypes::INT16)
        return GetCharVector<int16_t>(-1);
    if (dtype == Bodo_CTypes::UINT16)
        return GetCharVector<uint16_t>(0);
    if (dtype == Bodo_CTypes::INT32)
        return GetCharVector<int32_t>(-1);
    if (dtype == Bodo_CTypes::UINT32)
        return GetCharVector<uint32_t>(0);
    if (dtype == Bodo_CTypes::INT64)
        return GetCharVector<int64_t>(-1);
    if (dtype == Bodo_CTypes::UINT64)
        return GetCharVector<uint64_t>(0);
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
        __int128 e_val = 0;
        return GetCharVector<__int128>(e_val);
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
        STRUCT = 4,
        CATEGORICAL = 5,
        ARRAY_ITEM = 6,
        INTERVAL = 7,
        DICT = 8,  // dictionary-encoded string array
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
 * --- data1 is for the characters
 * --- data2 is for the index offsets
 * --- null_bitmask is for the missing entries
 * Case of LIST_STRING:
 * --- The length is the number of rows.
 * --- data1 is the characters
 * --- data2 is for the data_offsets
 * --- data3 is for the index_offsets
 * --- null_bitmask for whether the data is missing or not.
 * case of DICT:
 * --- child_array[0] is the string array for the dictionary.
 * --- child_array[1] is a Int32 array for the indices. This array and
 *     the main info share a bitmap.
 * --- has_global_dictionary is true if the dictionary has the same
 *     values in the same order for all ranks.
 * --- has_deduped_local_dictionary is true if a dictionary doesn't have
 *     any duplicates on the current rank. This may be false if the values
 *     are unique but we couldn't safely determine this. There are no false
 *     positives.
 * --- has_sorted_dictionary is true if a dictionary is sorted on the current
 * rank.
 */
struct array_info {
    bodo_array_type::arr_type_enum arr_type;
    Bodo_CTypes::CTypeEnum dtype;
    uint64_t length;  // number of elements in the array (not bytes) For DICT
                      // arrays this is the length of indices array

    // Data buffer meminfos for this array, e.g. data/offsets/null_bitmap for
    // string array. Last element is always the null bitmap buffer.
    std::vector<std::shared_ptr<BodoBuffer>> buffers;

    // Child arrays for nested array cases (dict-encoded arrays only currently)
    std::vector<std::shared_ptr<array_info>> child_arrays;
    // name of each field in struct array (empty for other arrays)
    std::vector<std::string> field_names;

    int32_t precision;                  // for array of decimals and times
    int32_t scale;                      // for array of decimals
    uint64_t num_categories;            // for categorical arrays
    bool has_global_dictionary;         // for dict-encoded arrays
    bool has_deduped_local_dictionary;  // for dict-encoded arrays
    bool has_sorted_dictionary;         // for dict-encoded arrays

    // Starting point into the physical data buffer (in bytes, not logical
    // values) for NUMPY, NULLABLE_INT_BOOL, CATEGORICAL arrays. This is needed
    // since Numpy array slicing creates views on MemInfo buffers with an offset
    // from the MemInfo data pointer. For example, A[2:4] will have
    // an offset of 16 bytes for int64 arrays (and n_items=2).
    int64_t offset;

    array_info(bodo_array_type::arr_type_enum _arr_type,
               Bodo_CTypes::CTypeEnum _dtype, int64_t _length,
               std::vector<std::shared_ptr<BodoBuffer>> _buffers,
               std::vector<std::shared_ptr<array_info>> _child_arrays = {},
               int32_t _precision = 0, int32_t _scale = 0,
               int64_t _num_categories = 0, bool _has_global_dictionary = false,
               bool _has_deduped_local_dictionary = false,
               bool _has_sorted_dictionary = false, int64_t _offset = 0,
               std::vector<std::string> _field_names = {})
        : arr_type(_arr_type),
          dtype(_dtype),
          length(_length),
          precision(_precision),
          scale(_scale),
          num_categories(_num_categories),
          has_global_dictionary(_has_global_dictionary),
          has_deduped_local_dictionary(_has_deduped_local_dictionary),
          has_sorted_dictionary(_has_sorted_dictionary),
          offset(_offset) {
        this->buffers = std::move(_buffers);
        this->child_arrays = std::move(_child_arrays);
        this->field_names = std::move(_field_names);
    }

    /**
     * @brief returns the first data pointer for the array if any.
     *
     * @return char* data pointer
     */
    char* data1() const;

    /**
     * @brief returns the second data pointer for the array if any.
     *
     * @return char* data pointer
     */
    char* data2() const;

    /**
     * @brief returns the third data pointer for the array if any.
     *
     * @return char* data pointer
     */
    char* data3() const;

    /**
     * @brief returns the pointer to null bitmask buffer for the array if any.
     *
     * @return char* null bitmask pointer
     */
    char* null_bitmask() const;

    /**
     * @brief returns the pointer to null bitmask buffer for the nested array if
     * any.
     *
     * @return char* null bitmask pointer of nested array
     */
    char* sub_null_bitmask() const {
        switch (arr_type) {
            case bodo_array_type::LIST_STRING:
                return this->child_arrays[0]->null_bitmask();
            case bodo_array_type::STRING:
            case bodo_array_type::DICT:
            case bodo_array_type::NULLABLE_INT_BOOL:
            case bodo_array_type::INTERVAL:
            case bodo_array_type::ARRAY_ITEM:
            case bodo_array_type::STRUCT:
            case bodo_array_type::NUMPY:
            case bodo_array_type::CATEGORICAL:
            default:
                return nullptr;
        }
    }

    /**
     * @brief return number of sub-elements for nested arrays:
     * number of characters for strings arrays,
     * number of strings for list of string arrays,
     * number of sub-elements for array(item) arrays,
     * -1 for other arrays which are non-nested.
     *
     * @return int64_t number of sub-elements
     */
    int64_t n_sub_elems() {
        if (arr_type == bodo_array_type::STRING) {
            offset_t* offsets = (offset_t*)data2();
            return offsets[length];
        }
        if (arr_type == bodo_array_type::LIST_STRING) {
            offset_t* offsets = (offset_t*)data3();
            return offsets[length];
        }
        return -1;
    }

    /**
     * @brief return number of sub-sub-elements for two-level nested arrays
     * (only list of strings supported now): number of characters for list of
     * strings arrays, -1 for other arrays.
     *
     * @return int64_t number of sub-elements
     */
    int64_t n_sub_sub_elems() {
        if (arr_type == bodo_array_type::LIST_STRING) {
            int64_t n_strings = n_sub_elems();
            offset_t* offsets = (offset_t*)data2();
            return offsets[n_strings];
        }
        return -1;
    }

    template <typename T>
    T& at(size_t idx) {
        return ((T*)data1())[idx];
    }

    template <typename T>
    const T& at(size_t idx) const {
        return ((T*)data1())[idx];
    }

    bool get_null_bit(size_t idx) const {
        return GetBit((uint8_t*)null_bitmask(), idx);
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
                throw std::runtime_error(
                    "get_code: codes have unrecognized dtype");
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
                if (this->arr_type == bodo_array_type::DICT) {
                    // In case of dictionary encoded string array
                    // get the string value by indexing into the dictionary
                    return this->child_arrays[0]->val_to_str(
                        this->child_arrays[1]->at<int32_t>(idx));
                }
                offset_t* offsets = (offset_t*)data2();
                return std::string(data1() + offsets[idx],
                                   offsets[idx + 1] - offsets[idx]);
            }
            case Bodo_CTypes::DATE: {
                int64_t val = this->at<int64_t>(idx);
                int year = val >> 32;
                int month = (val >> 16) & 0xFFFF;
                int day = val & 0xFFFF;
                std::string date_str;
                date_str.reserve(10);
                date_str += std::to_string(year) + "-";
                if (month < 10)
                    date_str += "0";
                date_str += std::to_string(month) + "-";
                if (day < 10)
                    date_str += "0";
                date_str += std::to_string(day);
                return date_str;
            }
            case Bodo_CTypes::_BOOL:
                bool val;
                if (this->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
                    val = GetBit((uint8_t*)data1(), idx);
                } else {
                    val = this->at<bool>(idx);
                }
                if (val) {
                    return "True";
                } else {
                    return "False";
                }
            default: {
                std::vector<char> error_msg(100);
                snprintf(error_msg.data(), error_msg.size(),
                         "val_to_str not implemented for dtype %d", dtype);
                throw std::runtime_error(error_msg.data());
            }
        }
    }

    void set_null_bit(size_t idx, bool bit) {
        SetBitTo((uint8_t*)null_bitmask(), idx, bit);
    }
};

/**
 * @brief Convert array_info to equivalent Arrow array.
 *
 * @return std::shared_ptr<arrow::Array> equivalent Array array
 */
std::shared_ptr<arrow::Array> to_arrow(std::shared_ptr<array_info> info);

std::shared_ptr<array_info> alloc_array(int64_t length, int64_t n_sub_elems,
                                        int64_t n_sub_sub_elems,
                                        bodo_array_type::arr_type_enum arr_type,
                                        Bodo_CTypes::CTypeEnum dtype,
                                        int64_t extra_null_bytes,
                                        int64_t num_categories);

std::shared_ptr<array_info> alloc_numpy(int64_t length,
                                        Bodo_CTypes::CTypeEnum typ_enum);

std::shared_ptr<array_info> alloc_array_item(int64_t n_arrays,
                                             int64_t n_total_items,
                                             Bodo_CTypes::CTypeEnum dtype);
std::shared_ptr<array_info> alloc_categorical(int64_t length,
                                              Bodo_CTypes::CTypeEnum typ_enum,
                                              int64_t num_categories);

std::shared_ptr<array_info> alloc_nullable_array(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum,
    int64_t extra_null_bytes = 0);

std::shared_ptr<array_info> alloc_nullable_array_no_nulls(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum, int64_t extra_null_bytes);

std::shared_ptr<array_info> alloc_nullable_array_all_nulls(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum, int64_t extra_null_bytes);

std::shared_ptr<array_info> alloc_string_array(int64_t length, int64_t n_chars,
                                               int64_t extra_null_bytes = 0);

std::shared_ptr<array_info> alloc_list_string_array(
    int64_t n_lists, std::shared_ptr<array_info> string_arr,
    int64_t extra_null_bytes);

std::shared_ptr<array_info> alloc_list_string_array(int64_t n_lists,
                                                    int64_t n_strings,
                                                    int64_t n_chars,
                                                    int64_t extra_null_bytes);

std::shared_ptr<array_info> alloc_dict_string_array(
    int64_t length, int64_t n_keys, int64_t n_chars_keys,
    bool has_global_dictionary, bool has_deduped_local_dictionary);

/**
 * @brief Create a string array from
 * a null bitmap and a vector of strings.
 *
 * @param grp_info The group info.
 * @param null_bitmap The null bitmap for the array.
 * @param list_string The vector of strings.
 * @return std::shared_ptr<array_info> A string type array info constructed from
 * the vector of strings.
 */
std::shared_ptr<array_info> create_string_array(
    std::vector<uint8_t> const& null_bitmap,
    std::vector<std::string> const& list_string);

/**
 * @brief Create an array of list of strings,
 * a null bitmap, and a vector of list of strings.
 *
 * @param grp_info The group info.
 * @param null_bitmap The null bitmap for the array.
 * @param list_list_pair The vector of list of strings. The list of strings
 * is a pair of string and a boolean. The boolean is true if the list entry is
 * null.
 * @return std::shared_ptr<array_info> A list of strings type array info
 * constructed from the vector.
 */
std::shared_ptr<array_info> create_list_string_array(
    std::vector<uint8_t> const& null_bitmap,
    std::vector<std::vector<std::pair<std::string, bool>>> const&
        list_list_pair);

/**
 * @brief Create a dict string array object from the underlying data array and
 * indices array
 *
 * @param dict_arr: the underlying data array
 * @param indices_arr: the underlying indices array
 * @param has_global_dictionary: Is the dict_arr global?
 * @param has_deduped_local_dictionary Can dict_arr contain
 * duplicates on this rank?
 * @param has_sorted_dictionary Is dict_arr sorted on this rank?
 * @return std::shared_ptr<array_info> The dictionary array.
 */
std::shared_ptr<array_info> create_dict_string_array(
    std::shared_ptr<array_info> dict_arr,
    std::shared_ptr<array_info> indices_arr, bool has_global_dictionary = false,
    bool has_deduped_local_dictionary = false,
    bool has_sorted_dictionary = false);

/* The "get-value" functionality for array_info.
   This is the equivalent of at functionality.
   We cannot use at(idx) statements.
 */

template <typename T>
inline T& getv(std::shared_ptr<array_info> arr, size_t idx) {
    return ((T*)arr->data1())[idx];
}

struct mpi_comm_info {
    int myrank;
    int n_pes;
    std::vector<std::shared_ptr<array_info>> arrays;
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
    // Store input row to destination rank. This way we only need to do
    // hash_to_rank() once during shuffle. Also stores the result of filtering
    // rows with bloom filters (removed rows have dest == -1), so that we only
    // query the filter once.
    std::vector<int> row_dest;
    // true if using a bloom filter to discard rows before shuffling
    bool filtered = false;

    explicit mpi_comm_info(std::vector<std::shared_ptr<array_info>>& _arrays);

    /**
     * Computation of:
     * - send_count/recv_count arrays
     * - send_count_sub / recv_count_sub
     * - send_count_sub_sub / recv_count_sub_sub
     * Those are used for the shuffling of data1/data2/data3 and their sizes.
     * @param hashes : hashes of all the rows
     * @param is_parallel: true if data is distributed (used to indicate whether
     *                     tracing should be parallel or not)
     * @param filter : Bloom filter. Rows whose hash is not in the filter will
     * be discarded from shuffling. If no filter is provided no filtering will
     * happen.
     * @param null_bitmask : Null bitmask specifying if any of the keys are
     * null. In those cases, the rows will be handled based on the value of the
     * templated parameter keep_nulls_and_filter_misses. Note that this should
     * only be passed when nulls are not considered equal to each other (e.g.
     * SQL join).
     *
     * @tparam keep_nulls_and_filter_misses : In case a Bloom filter is provided
     * and a key is not present in the bloom filter, should we keep the value on
     * this rank (i.e. not discard it altogether).
     * Similarly, in case a null-bitmask is provided and a row
     * is determined to have a null in one of the keys (i.e. this row cannot
     * match with any other in case of SQL joins), this flag determines whether
     * to keep the row on this rank (i.e. not discard it altogether).
     * This is useful in the outer join cases. Defaults to false.
     */
    template <bool keep_nulls_and_filter_misses = false>
    void set_counts(
        const std::shared_ptr<uint32_t[]>& hashes, bool is_parallel,
        SimdBlockFilterFixed<::hashing::SimpleMixSplit>* filter = nullptr,
        const uint8_t* null_bitmask = nullptr);
};

struct table_info {
    std::vector<std::shared_ptr<array_info>> columns;
    // this is set and used by groupby to avoid putting additional info in
    // multi_col_key (which is only needed when key equality is checked but not
    // for hashing)
    // TODO consider passing 'num_keys' to the constructor
    int64_t num_keys;
    // keep shuffle info to be able to reverse the shuffle if necessary
    // currently used in groupby apply
    // TODO: refactor out?
    std::shared_ptr<mpi_comm_info> comm_info;
    std::shared_ptr<uint32_t[]> hashes;
    int id;
    table_info() {}
    explicit table_info(std::vector<std::shared_ptr<array_info>>& _columns)
        : columns(_columns) {}
    uint64_t nrows() const { return columns[0]->length; }
    uint64_t ncols() const { return columns.size(); }
    std::shared_ptr<array_info> operator[](size_t idx) { return columns[idx]; }
    const std::shared_ptr<array_info> operator[](size_t idx) const {
        return columns[idx];
    }
};

/* Compute the total memory of local chunk of the table on current rank
 *
 * @param table : The input table
 * @return the total size of the local chunk of the table
 */
int64_t table_local_memory_size(std::shared_ptr<table_info> table);

/* Compute the total memory of the table accross all processors.
 *
 * @param table : The input table
 * @return the total size of the table all over the processors
 */
int64_t table_global_memory_size(std::shared_ptr<table_info> table);

/// Initialize numpy_item_size and verify size of dtypes
void bodo_common_init();

std::shared_ptr<array_info> copy_array(std::shared_ptr<array_info> arr);

NRT_MemInfo* alloc_meminfo(int64_t length);

/**
 * Free underlying array of array_info pointer and delete the pointer
 */
void delete_info(array_info* arr);

/**
 * delete table pointer and its column array_info pointers (but not the arrays).
 */
void delete_table(table_info* table);

/**
 * decref meminfo refcount
 */
void decref_meminfo(MemInfo* meminfo);

/**
 * incref meminfo refcount
 */
void incref_meminfo(MemInfo* meminfo);

extern "C" {

struct numpy_arr_payload {
    NRT_MemInfo* meminfo;
    PyObject* parent;
    int64_t nitems;
    int64_t itemsize;
    char* data;
    int64_t shape;
    int64_t strides;
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

/// \brief Get binary value as a string_view
// A version of Arrow's GetView() that returns std::string_view directly.
// Arrow < 10.0 versions return arrow::util::string_view which is a vendored
// version that doesn't work in containers (see string_hash below).
// TODO: replace with Arrow's GetView after upgrade to Arrow 10
// https://github.com/apache/arrow/blob/a2881a124339d7d50088c5b9778c725316a7003e/cpp/src/arrow/array/array_binary.h#L70
///
/// \param i the value index
/// \return the view over the selected value
template <typename ARROW_ARRAY_TYPE>
std::string_view ArrowStrArrGetView(std::shared_ptr<ARROW_ARRAY_TYPE> str_arr,
                                    int64_t i) {
    auto raw_value_offsets = str_arr->raw_value_offsets();

    auto pos = raw_value_offsets[i];
    return std::string_view(
        reinterpret_cast<const char*>(str_arr->raw_data() + pos),
        raw_value_offsets[i + 1] - pos);
}

// C++20 magic to support "heterogeneous" access to unordered containers
// makes the key "transparent", allowing std::string_view to be used similar to
// std::string https://www.cppstories.com/2021/heterogeneous-access-cpp20/
struct string_hash {
    using is_transparent = void;
    [[nodiscard]] size_t operator()(const char* txt) const {
        return std::hash<std::string_view>{}(txt);
    }
    [[nodiscard]] size_t operator()(std::string_view txt) const {
        return std::hash<std::string_view>{}(txt);
    }
    [[nodiscard]] size_t operator()(const std::string& txt) const {
        return std::hash<std::string>{}(txt);
    }
};

#ifdef __cplusplus
// Define constructor outside of the struct to fix C linkage warning
inline struct numpy_arr_payload make_numpy_array_payload(
    NRT_MemInfo* _meminfo, PyObject* _parent, int64_t _nitems,
    int64_t _itemsize, char* _data, int64_t _shape, int64_t _strides) {
    struct numpy_arr_payload p;
    p.meminfo = _meminfo;
    p.parent = _parent;
    p.nitems = _nitems;
    p.itemsize = _itemsize;
    p.data = _data;
    p.shape = _shape;
    p.strides = _strides;
    return p;
}
#endif

extern "C" {
PyMODINIT_FUNC PyInit_hdist(void);
PyMODINIT_FUNC PyInit_hstr_ext(void);
PyMODINIT_FUNC PyInit_decimal_ext(void);
PyMODINIT_FUNC PyInit_quantile_alg(void);
PyMODINIT_FUNC PyInit_hdatetime_ext(void);
PyMODINIT_FUNC PyInit_hio(void);
PyMODINIT_FUNC PyInit_array_ext(void);
PyMODINIT_FUNC PyInit_s3_reader(void);
PyMODINIT_FUNC PyInit_fsspec_reader(void);
PyMODINIT_FUNC PyInit_hdfs_reader(void);
PyMODINIT_FUNC PyInit__hdf5(void);
PyMODINIT_FUNC PyInit_arrow_cpp(void);
PyMODINIT_FUNC PyInit_csv_cpp(void);
PyMODINIT_FUNC PyInit_json_cpp(void);
}  // extern "C"

#endif /* BODO_COMMON_H_INCLUDED_ */
