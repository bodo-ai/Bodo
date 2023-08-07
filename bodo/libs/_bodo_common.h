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
#include "_datetime_utils.h"
#include "_meminfo.h"
#include "simd-block-fixed-fpp.h"
#include "tracing.h"

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

// --------- MemInfo Helper Functions --------- //
NRT_MemInfo* alloc_meminfo(int64_t length);

/**
 * decref meminfo refcount
 */
void decref_meminfo(MemInfo* meminfo);

/**
 * incref meminfo refcount
 */
void incref_meminfo(MemInfo* meminfo);

// --------- BodoBuffer Definition --------- //

/**
 * @brief Arrow buffer that holds a reference to a Bodo meminfo and
 * decrefs/deallocates if necessary to Bodo's BufferPool.
 * Alternative to Arrow's PoolBuffer:
 * https://github.com/apache/arrow/blob/5b2fbade23eda9bc95b1e3854b19efff177cd0bd/cpp/src/arrow/memory_pool.cc#L838
 */
class BodoBuffer : public arrow::ResizableBuffer {
   public:
    /**
     * @brief Construct a new Bodo Buffer
     *
     * @param data Pointer to data buffer
     * @param size Size of the buffer
     * @param meminfo_ MemInfo object which manages the underlying data buffer
     * @param incref Whether to incref (default: true)
     * @param pool Pool that was used for allocating the data buffer. The same
     * pool should be used for resizing the buffer in the future.
     * @param mm Memory manager associated with the pool.
     */
    BodoBuffer(uint8_t* data, const int64_t size, NRT_MemInfo* meminfo_,
               bool incref = true,
               bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
               std::shared_ptr<::arrow::MemoryManager> mm =
                   bodo::default_buffer_memory_manager())
        : ResizableBuffer(data, size, std::move(mm)),
          meminfo(meminfo_),
          pool_(pool) {
        if (incref) {
            incref_meminfo(meminfo);
        }
    }

    ~BodoBuffer() override {
        // Adapted from:
        // https://github.com/apache/arrow/blob/5b2fbade23eda9bc95b1e3854b19efff177cd0bd/cpp/src/arrow/memory_pool.cc#L844
        uint8_t* ptr = mutable_data();
        // TODO(ehsan): add global_state.is_finalizing() check to match Arrow
        if (ptr) {
            decref_meminfo(meminfo);
        }
    }

    NRT_MemInfo* getMeminfo() { return meminfo; }

    /**
     * @brief Ensure that buffer can accommodate specified total number of bytes
     * without re-allocation.
     *
     * Copied from Arrow's PoolBuffer since it is not exposed to use as base
     * class:
     * https://github.com/apache/arrow/blob/apache-arrow-11.0.0/cpp/src/arrow/memory_pool.cc
     * Bodo change: alignment is not passed to alloc calls which defaults to
     * 64-byte. Meminfo attributes are also updated.
     *
     * @param capacity minimum total capacity required in bytes
     * @return arrow::Status
     */
    arrow::Status Reserve(const int64_t capacity) override {
        if (capacity < 0) {
            return arrow::Status::Invalid("Negative buffer capacity: ",
                                          capacity);
        }
        if (!data_ || capacity > capacity_) {
            int64_t new_capacity =
                arrow::bit_util::RoundUpToMultipleOf64(capacity);
            if (data_) {
                // NOTE: meminfo->data needs to be passed to buffer pool manager
                // since it stores swips
                RETURN_NOT_OK(pool_->Reallocate(capacity_, new_capacity,
                                                (uint8_t**)&(meminfo->data)));
            } else {
                RETURN_NOT_OK(
                    pool_->Allocate(new_capacity, (uint8_t**)&(meminfo->data)));
            }
            data_ = (uint8_t*)meminfo->data;
            capacity_ = new_capacity;
            // Updating size is necessary since used by the buffer pool manager
            // for freeing.
            meminfo->size = new_capacity;
        }
        return arrow::Status::OK();
    }

    /**
     * @brief Set the size of the buffer to 'new_size'.
     * If 'new_size' is greater than the buffer's
     * capacity, a CapacityError will be returned.
     *
     * @param new_size New size to set to.
     * @return arrow::Status
     */
    arrow::Status SetSize(const int64_t new_size) {
        if (ARROW_PREDICT_FALSE(new_size < 0)) {
            return arrow::Status::Invalid(
                "BodoBuffer::SetSize: Negative buffer resize: ", new_size);
        }
        if (new_size > this->capacity_) {
            return arrow::Status::CapacityError(
                "BodoBuffer::SetSize: new_size (" + std::to_string(new_size) +
                ") is greater than capacity (" +
                std::to_string(this->capacity_) + ")!");
        }
        this->size_ = new_size;
        return arrow::Status::OK();
    }

    /**
     * @brief Make sure there is enough capacity and resize the buffer.
     * If shrink_to_fit=true and new_size is smaller than existing size, updates
     * capacity to the nearest multiple of 64 bytes.
     *
     * Copied from Arrow's PoolBuffer since it is not exposed to use as base
     * class:
     * https://github.com/apache/arrow/blob/apache-arrow-11.0.0/cpp/src/arrow/memory_pool.cc
     * Bodo change: alignment is not passed to alloc calls which defaults to
     * 64-byte. Meminfo attributes are also updated.
     *
     * @param new_size New size of the buffer
     * @param shrink_to_fit if new size is smaller than the existing size,
     * reallocate to fit.
     * @return arrow::Status
     */
    arrow::Status Resize(const int64_t new_size,
                         bool shrink_to_fit = true) override {
        if (ARROW_PREDICT_FALSE(new_size < 0)) {
            return arrow::Status::Invalid("Negative buffer resize: ", new_size);
        }
        if (data_ && shrink_to_fit && new_size <= size_) {
            // Buffer is non-null and is not growing, so shrink to the requested
            // size without excess space.
            int64_t new_capacity =
                arrow::bit_util::RoundUpToMultipleOf64(new_size);
            if (capacity_ != new_capacity) {
                // Buffer hasn't got yet the requested size.
                RETURN_NOT_OK(pool_->Reallocate(capacity_, new_capacity,
                                                (uint8_t**)&(meminfo->data)));
                data_ = (uint8_t*)meminfo->data;
                capacity_ = new_capacity;
                // Updating size is necessary since used by the buffer pool
                // manager for freeing.
                meminfo->size = new_capacity;
            }
        } else {
            RETURN_NOT_OK(Reserve(new_size));
        }
        size_ = new_size;

        return arrow::Status::OK();
    }

    /// @brief Wrapper for Pinning Buffers
    inline void pin() {
        NRT_MemInfo_Pin(this->meminfo);
        // If the buffer was spilled, its data pointer could have changed upon
        // restoring it to memory. Therefore, we need to resynchronize the
        // `data_` pointer of the buffer with that of the MemInfo (which is what
        // the BufferPool changes in place to the new location).
        this->data_ = (uint8_t*)this->meminfo->data;
    }

    /// @brief Wrapper for Unpinning Buffers
    inline void unpin() { NRT_MemInfo_Unpin(this->meminfo); }

    template <typename X>
    friend class ::bodo::pin_guard;

   private:
    NRT_MemInfo* meminfo;
    arrow::MemoryPool* pool_;
};

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
        LIST = 19,    // for nested data structures, maps to Arrow List
        STRUCT = 20,  // for nested data structures, maps to Arrow Struct
        BINARY = 21,
        _numtypes
    };
};

typedef int32_t dict_indices_t;

// use 64-bit offsets for string and nested arrays
typedef uint64_t offset_t;
#define OFFSET_BITWIDTH 64
#define Bodo_CType_offset Bodo_CTypes::CTypeEnum::UINT64

std::unique_ptr<BodoBuffer> AllocateBodoBuffer(
    const int64_t size,
    bodo::IBufferPool* const = bodo::BufferPool::DefaultPtr(),
    const std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());
std::unique_ptr<BodoBuffer> AllocateBodoBuffer(
    const int64_t length, Bodo_CTypes::CTypeEnum typ_enum,
    bodo::IBufferPool* const = bodo::BufferPool::DefaultPtr(),
    const std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

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
 * The template parameter is T.
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
 * These are also defined in utils/utils.py and must match.
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

template <typename Alloc>
inline bool GetBit(const std::vector<uint8_t, Alloc>& V, uint64_t i) {
    return GetBit(V.data(), i);
}

inline void SetBitTo(uint8_t* bits, int64_t i, bool bit_is_set) {
    bits[i / 8] ^=
        static_cast<uint8_t>(-static_cast<uint8_t>(bit_is_set) ^ bits[i / 8]) &
        kBitmask[i % 8];
}

template <typename Alloc>
inline void SetBitTo(std::vector<uint8_t, Alloc>& V, int64_t i,
                     bool bit_is_set) {
    SetBitTo(V.data(), i, bit_is_set);
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
 * --- is_globally_replicated is true if the array
 *     is replicated on all ranks. This is used when
 *     the array is a dictionary for a DICT array.
 * --- is_locally_unique is true if the array
 *     is unique on the current rank. This is used when
 *     the array is a dictionary for a DICT array.
 * --- is_locally_sorted is true if the array
 *     is sorted on the current rank. This is used when
 *     the array is a dictionary for a DICT array.
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

    int32_t precision;        // for array of decimals and times
    int32_t scale;            // for array of decimals
    uint64_t num_categories;  // for categorical arrays
    // ID used to identify matching equivalent dictionaries.
    // Currently only used by string arrays that are the dictionaries
    // inside dictionary encoded arrays. It cannot be placed in the dictionary
    // array itself because dictionary builders just work with the string array
    // and we need data to be consistent.
    int64_t array_id;
    // Is this array globally shared on all ranks. This is currently only
    // used to describe string arrays that function as dictionaries in
    // dictionary encoded arrays.
    bool is_globally_replicated;
    // Is this array unique on the current rank. This is currently only
    // used to describe string arrays that function as dictionaries in
    // dictionary encoded arrays.
    bool is_locally_unique;
    // Is this array sorted on the current rank. This is currently only
    // used to describe string arrays that function as dictionaries in
    // dictionary encoded arrays.
    bool is_locally_sorted;

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
               int64_t _num_categories = 0, int64_t _array_id = -1,
               bool _is_globally_replicated = false,
               bool _is_locally_unique = false, bool _is_locally_sorted = false,
               int64_t _offset = 0, std::vector<std::string> _field_names = {})
        : arr_type(_arr_type),
          dtype(_dtype),
          length(_length),
          precision(_precision),
          scale(_scale),
          num_categories(_num_categories),
          array_id(_array_id),
          is_globally_replicated(_is_globally_replicated),
          is_locally_unique(_is_locally_unique),
          is_locally_sorted(_is_locally_sorted),
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
                int64_t day = this->at<int32_t>(idx);
                int64_t year = days_to_yearsdays(&day);
                int64_t month;
                get_month_day(year, day, &month, &day);
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

    void pin() {
        switch (arr_type) {
            case bodo_array_type::DICT:
            case bodo_array_type::LIST_STRING:
                for (auto& arr : this->child_arrays) {
                    arr->pin();
                }
                break;
            default:
                for (auto& buffer : this->buffers) {
                    buffer->pin();
                }
                break;
        }
    }

    /**
     * @param unpin_dict The flag that indicates if the dictionary for
     * dictionary-encoded string arrays will be unpinned. Dictionary are shared
     * among many dictionary-encoded arrays. Unpinning the dictionary can
     * potentially affect other pinned dictionary-encoded arrays, so we don't
     * want to unpin it by default.
     */
    void unpin(bool unpin_dict = false) {
        switch (arr_type) {
            case bodo_array_type::DICT:
                if (unpin_dict) {
                    this->child_arrays[0]->unpin();
                }
                this->child_arrays[1]->unpin();
                break;
            case bodo_array_type::LIST_STRING:
                for (auto& arr : this->child_arrays) {
                    arr->unpin();
                }
                break;
            default:
                for (auto& buffer : this->buffers) {
                    buffer->unpin();
                }
                break;
        }
    }

    template <typename X>
    friend class ::bodo::pin_guard;

    /**
     * @brief Can a given a array contain NA values.
     * Currently this decides solely based on type but
     * in the future may be used if a given array statically
     * has an empty null bitmap.
     *
     */
    bool can_contain_na() {
        switch (arr_type) {
            case bodo_array_type::LIST_STRING:
            case bodo_array_type::STRING:
            case bodo_array_type::DICT:
            case bodo_array_type::NULLABLE_INT_BOOL:
            case bodo_array_type::ARRAY_ITEM:
            case bodo_array_type::STRUCT:
            case bodo_array_type::CATEGORICAL:
                return true;
            case bodo_array_type::NUMPY:
                // TODO: Remove when TIMEDELTA moves to nullable arrays.
                return dtype == Bodo_CTypes::TIMEDELTA;
            default:
                return false;
        }
    }

    /**
     * @brief Get a boolean vector indicating if
     * each element is NOTNA
     *
     * @return bodo::vector<bool> bool vector of NOTNA.
     */
    bodo::vector<bool> get_notna_vector() {
        bodo::vector<bool> not_na(length, true);
        switch (arr_type) {
            case bodo_array_type::LIST_STRING:
            case bodo_array_type::STRING:
            case bodo_array_type::DICT:
            case bodo_array_type::NULLABLE_INT_BOOL:
            case bodo_array_type::ARRAY_ITEM:
            case bodo_array_type::STRUCT:
                for (size_t i = 0; i < length; i++) {
                    not_na[i] = get_null_bit(i);
                }
                break;
            case bodo_array_type::CATEGORICAL:
                for (size_t i = 0; i < length; i++) {
                    // TODO: Ensure templated for faster access.
                    not_na[i] = get_code_as_int64(i) != -1;
                }
                break;
            case bodo_array_type::NUMPY:
                if (dtype == Bodo_CTypes::TIMEDELTA) {
                    int64_t* data = (int64_t*)data1();
                    for (size_t i = 0; i < length; i++) {
                        // TIMEDELTA NaT is the min int64_t.
                        not_na[i] =
                            data[i] != std::numeric_limits<int64_t>::min();
                    }
                }
                break;
            default:
                break;
        }
        return not_na;
    }
};

/**
 * @brief Convert array_info to equivalent Arrow array.
 *
 * @return std::shared_ptr<arrow::Array> equivalent Array array
 */
std::shared_ptr<arrow::Array> to_arrow(const std::shared_ptr<array_info> info);

std::unique_ptr<array_info> alloc_array(
    int64_t length, int64_t n_sub_elems, int64_t n_sub_sub_elems,
    bodo_array_type::arr_type_enum arr_type, Bodo_CTypes::CTypeEnum dtype,
    int64_t array_id = -1, int64_t extra_null_bytes = 0,
    int64_t num_categories = 0, bool is_globally_replicated = false,
    bool is_locally_unique = false, bool is_locally_sorted = false,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

std::unique_ptr<array_info> alloc_numpy(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

std::unique_ptr<array_info> alloc_array_item(int64_t n_arrays,
                                             int64_t n_total_items,
                                             Bodo_CTypes::CTypeEnum dtype);
std::unique_ptr<array_info> alloc_categorical(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum, int64_t num_categories,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

std::unique_ptr<array_info> alloc_nullable_array(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum,
    int64_t extra_null_bytes = 0,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

std::unique_ptr<array_info> alloc_nullable_array_no_nulls(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum, int64_t extra_null_bytes);

std::unique_ptr<array_info> alloc_nullable_array_all_nulls(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum, int64_t extra_null_bytes);

std::unique_ptr<array_info> alloc_string_array(
    Bodo_CTypes::CTypeEnum typ_enum, int64_t length, int64_t n_chars,
    int64_t array_id = -1, int64_t extra_null_bytes = 0,
    bool is_globally_replicated = false, bool is_locally_unique = false,
    bool is_locally_sorted = false,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

std::unique_ptr<array_info> alloc_list_string_array(
    int64_t n_lists, std::shared_ptr<array_info> string_arr,
    int64_t extra_null_bytes,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

std::unique_ptr<array_info> alloc_list_string_array(
    int64_t n_lists, int64_t n_strings, int64_t n_chars,
    int64_t extra_null_bytes,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

std::unique_ptr<array_info> alloc_dict_string_array(
    int64_t length, int64_t n_keys, int64_t n_chars_keys,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

/**
 * @brief Create a string array from
 * a null bitmap and a vector of strings.
 *
 * @param typ_enum The dtype (String or Binary).
 * @param null_bitmap The null bitmap for the array.
 * @param list_string The vector of strings.
 * @param array_id The id for identifying this array. This is used when creating
 * dictionaries.
 * @return std::shared_ptr<array_info> A string type array info constructed from
 * the vector of strings.
 */
std::unique_ptr<array_info> create_string_array(
    Bodo_CTypes::CTypeEnum typ_enum, bodo::vector<uint8_t> const& null_bitmap,
    bodo::vector<std::string> const& list_string, int64_t array_id = -1);

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
std::unique_ptr<array_info> create_list_string_array(
    bodo::vector<uint8_t> const& null_bitmap,
    bodo::vector<bodo::vector<std::pair<std::string, bool>>> const&
        list_list_pair);

/**
 * @brief Create a dict string array object from the underlying data array and
 * indices array
 *
 * @param dict_arr: the underlying data array
 * @param indices_arr: the underlying indices array
 * @return std::shared_ptr<array_info> The dictionary array.
 */
std::unique_ptr<array_info> create_dict_string_array(
    std::shared_ptr<array_info> dict_arr,
    std::shared_ptr<array_info> indices_arr);

/* The "get-value" functionality for array_info.
   This is the equivalent of at functionality.
   We cannot use at(idx) statements.
 */

template <typename T>
inline T& getv(const std::shared_ptr<array_info>& arr, size_t idx) {
    return ((T*)arr->data1())[idx];
}

template <typename T>
inline T& getv(const std::unique_ptr<array_info>& arr, size_t idx) {
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
    bodo::vector<int> row_dest;
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
    // keep shuffle info to be able to reverse the shuffle if necessary
    // currently used in groupby apply
    // TODO: refactor out?
    std::shared_ptr<mpi_comm_info> comm_info;
    std::shared_ptr<uint32_t[]> hashes;
    int id;
    table_info() {}
    explicit table_info(std::vector<std::shared_ptr<array_info>>& _columns)
        : columns(_columns) {}
    uint64_t ncols() const { return columns.size(); }
    uint64_t nrows() const {
        return this->ncols() == 0 ? 0 : columns[0]->length;
    }
    std::shared_ptr<array_info> operator[](size_t idx) { return columns[idx]; }
    const std::shared_ptr<array_info> operator[](size_t idx) const {
        return columns[idx];
    }

    void pin() {
        for (auto& col : columns) {
            col->pin();
        }
    }

    void unpin() {
        for (auto& col : columns) {
            col->unpin();
        }
    }

    template <typename X>
    friend class ::bodo::pin_guard;
};

/**
 * @brief Helper function for early reference (and potentially memory) release
 * of a column in a table (to reduce peak memory usage wherever possible). This
 * is useful in performance and memory critical regions to release memory of
 * columns that are no longer needed. However, calling reset on the column is
 * only safe if this is the last reference to the table. We pass the shared_ptr
 * by reference to not incref the table_info during the function call.
 * NOTE: This function is only useful for performance reasons and should only be
 * used when you know what you are doing. See existing usages of this in our
 * code to understand the use cases.
 *
 * @param table Shared Pointer to the table, passed by reference.
 * @param col_idx Index of the column to reset.
 */
void reset_col_if_last_table_ref(std::shared_ptr<table_info> const& table,
                                 size_t col_idx);

/**
 * @brief Similar to reset_col_if_last_table_ref, but instead, we clear
 * all the shared_ptrs to the columns if this is the last reference to
 * a table.
 *
 * @param table Shared Pointer to the table, passed by reference.
 */
void clear_all_cols_if_last_table_ref(std::shared_ptr<table_info> const& table);

/* Compute the total memory of local chunk of the table on current rank
 *
 * @param table : The input table
 * @return the total size of the local chunk of the table
 */
int64_t table_local_memory_size(std::shared_ptr<table_info> table);

/* Compute the total memory of the table across all processors.
 *
 * @param table : The input table
 * @return the total size of the table all over the processors
 */
int64_t table_global_memory_size(std::shared_ptr<table_info> table);

/// Initialize numpy_item_size and verify size of dtypes
void bodo_common_init();

std::shared_ptr<array_info> copy_array(std::shared_ptr<array_info> arr);

/* Calculate the size of one row
 *
 * @param arr_c_types : the array of types for the row
 * @return the total size of the row
 */
size_t get_row_bytes(const std::vector<int8_t>& arr_array_types,
                     const std::vector<int8_t>& arr_c_types);

/**
 * Free underlying array of array_info pointer and delete the pointer
 */
void delete_info(array_info* arr);

/**
 * delete table pointer and its column array_info pointers (but not the arrays).
 */
void delete_table(table_info* table);

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

Bodo_CTypes::CTypeEnum arrow_to_bodo_type(arrow::Type::type type);

/**
 * @brief Generate a new local id for a dictionary. These
 * can be used to identify if dictionaries are "equivalent"
 * because they share an id. Other than ==, a particular
 * id has no significance. This is a wrapper around a static
 * function containing the id generation state.
 *
 * @param length The length of the dictionary being assigned
 * the id. All dictionaries of length 0 should get the same
 * id.
 * @return int64_t The new id that is generated.
 */
int64_t generate_array_id(int64_t length);

/**
 * @brief Determine if two dictionaries are equivalent. The two dictionaries are
 * equivalent if either they have the same exact child array or the dictionary
 * ids are the same. There is also support for a special case when the array_id
 * == 0, which means a dictionary is empty. Since there are no valid indices
 * every dictionary matches. This is mostly relevant for initialization when
 * "building" a dictionary.
 *
 * @param arr1 The first dictionary
 * @param arr2 The second dictionary
 * @return Do the dictionaries match (and therefore are the indices equivalent).
 */
static inline bool is_matching_dictionary(
    const std::shared_ptr<array_info>& arr1,
    const std::shared_ptr<array_info>& arr2) {
    bool arr1_valid_id = arr1->array_id >= 0;
    bool arr2_valid_id = arr2->array_id >= 0;
    return (arr1 == arr2) || (arr1_valid_id && arr2_valid_id &&
                              (arr1->array_id == 0 || arr2->array_id == 0 ||
                               (arr1->array_id == arr2->array_id)));
}

/**
 * @brief initialize bitmask for array
 *
 * @param bits bitmask pointer
 * @param length total length of array
 * @param val value to initialize (true or false)
 * @param start_row first row to start initializing from
 */
inline void InitializeBitMask(uint8_t* bits, size_t length, bool val,
                              int64_t start_row = 0) {
    // if start row isn't byte aligned for memset
    if ((start_row & 7) != 0) {
        for (size_t i = start_row; i < length; i++) {
            SetBitTo(bits, i, val);
        }
        return;
    }
    size_t n_bytes = (length - start_row + 7) >> 3;
    uint8_t* ptr = bits + (start_row >> 3);
    if (!val)
        memset(ptr, 0, n_bytes);
    else
        memset(ptr, 0xff, n_bytes);
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
PyMODINIT_FUNC PyInit_lead_lag(void);
PyMODINIT_FUNC PyInit_crypto_funcs(void);
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
PyMODINIT_FUNC PyInit_stream_join_cpp(void);
PyMODINIT_FUNC PyInit_listagg(void);
PyMODINIT_FUNC PyInit_stream_groupby_cpp(void);
PyMODINIT_FUNC PyInit_stream_dict_encoding_cpp(void);
PyMODINIT_FUNC PyInit_table_builder_cpp(void);
#ifdef IS_TESTING
PyMODINIT_FUNC PyInit_test_cpp(void);
#endif
}  // extern "C"

#endif /* BODO_COMMON_H_INCLUDED_ */
