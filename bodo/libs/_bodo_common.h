#pragma once

#include <complex>
#if defined(__GNUC__)
#define __UNUSED__ __attribute__((unused))
#else
#define __UNUSED__
#endif

#include <Python.h>
#include <arrow/type.h>
#include <utility>
#include <vector>

#include "_meminfo.h"
#include "vendored/simd-block-fixed-fpp.h"

// Macros for temporarily disabling compiler errors. Example usage:
//   [[deprecated]] int f() {
//     return 0;
//   }
//
//   int main() {
//     PUSH_IGNORED_COMPILER_ERROR("-Wdeprecated-declarations");
//     return f(); // would normally raise an error
//     POP_IGNORED_COMPILER_ERROR();
//   }
// See https://gcc.gnu.org/onlinedocs/gcc/Diagnostic-Pragmas.html
#if defined(__GNUC__) || defined(__clang__)
#define DO_PRAGMA(x) _Pragma(#x)
#define PUSH_IGNORED_COMPILER_ERROR(err)                                      \
    DO_PRAGMA(GCC diagnostic push);                                           \
    /* some compilers don't support -Wunknown-warning-option as used below */ \
    DO_PRAGMA(GCC diagnostic ignored "-Wpragmas");                            \
    /* some compilers might not implement the error class specified */        \
    DO_PRAGMA(GCC diagnostic ignored "-Wunknown-warning-option");             \
    /* Ignore the requested error class */                                    \
    DO_PRAGMA(GCC diagnostic ignored err);                                    \
    /* emit a compiler message so we don't lose track of ignored errors */    \
    DO_PRAGMA(message "Ignoring error " err " in " __FILE__)
// Every call to PUSH_IGNORED_COMPILER_ERROR  MUST  have a corresponding call to
// POP_IGNORED_COMPILER_ERROR. Otherwise the error will be disabled for the rest
// of compilation
#define POP_IGNORED_COMPILER_ERROR() DO_PRAGMA(GCC diagnostic pop)
#elif defined(_MSC_VER)
#define PUSH_IGNORED_COMPILER_ERROR(err) \
    __pragma(warning(push));             \
    __pragma(warning(disable : err))
#define POP_IGNORED_COMPILER_ERROR() __pragma(warning(pop))
#else
#define PUSH_IGNORED_COMPILER_ERROR(err) \
    static_assert(false, "Unsupported compiler: Cannot disable warnings")
#define POP_IGNORED_COMPILER_ERROR() \
    static_assert(false, "Unsupported compiler: Cannot pop ignored warnings")
#endif

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

void Bodo_PyErr_SetString(PyObject* type, const char* message);

// --------- Windows Compatibility ------------ //
#if defined(_WIN32)
#include <__msvc_int128.hpp>

template <typename T>
concept FloatOrDouble = std::is_same_v<T, float> || std::is_same_v<T, double>;

// Subclass std::_Signed128 to add missing C++ operators
// such as casting and conversion.
// Avoiding error checking and exceptions to behave like a native
// type as much as possible.
struct __int128_t : std::_Signed128 {
    __int128_t() : std::_Signed128() {}

    __int128_t(std::_Signed128 in_val) : std::_Signed128(in_val) {}

    template <std::integral T>
    constexpr __int128_t(T in_val) noexcept : std::_Signed128(in_val) {}

    template <FloatOrDouble T>
    __int128_t(T in_val);

    template <FloatOrDouble T>
    T int128_to_float() const;

    operator float() const { return int128_to_float<float>(); }

    operator double() const { return int128_to_float<double>(); }

    template <std::integral T>
    friend constexpr __int128_t operator<<(const __int128_t& _Left,
                                           const T& _Right) noexcept {
        return __int128_t(_Left << __int128_t(_Right));
    }

    template <std::integral T>
    friend constexpr __int128_t operator>>(const __int128_t& _Left,
                                           const T& _Right) noexcept {
        return __int128_t(_Left >> __int128_t(_Right));
    }

    template <std::integral T>
    friend constexpr bool operator==(const __int128_t& _Left,
                                     const T& _Right) noexcept {
        return (_Left == __int128_t(_Right));
    }

    template <std::integral T>
    friend constexpr __int128_t operator|(const __int128_t& _Left,
                                          const T& _Right) noexcept {
        return __int128_t(_Left | __int128_t(_Right));
    }
};
#endif

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
    void pin() {
        NRT_MemInfo_Pin(this->meminfo);
        // If the buffer was spilled, its data pointer could have changed upon
        // restoring it to memory. Therefore, we need to resynchronize the
        // `data_` pointer of the buffer with that of the MemInfo (which is what
        // the BufferPool changes in place to the new location).
        this->data_ = (uint8_t*)this->meminfo->data;
    }

    /// @brief Wrapper for Unpinning Buffers
    void unpin() { NRT_MemInfo_Unpin(this->meminfo); }

    template <typename X, typename... Args>
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
// NOTE: should match numpy_item_size in _bodo_common.cpp
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
        LIST = 18,    // for nested data structures, maps to Arrow List
        STRUCT = 19,  // for nested data structures, maps to Arrow Struct
        BINARY = 20,
        COMPLEX64 = 21,
        COMPLEX128 = 22,
        MAP = 23,
        TIMESTAMPTZ = 24,  // Used to raise errors in other code locations
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

constexpr inline bool is_unsigned_integer(Bodo_CTypes::CTypeEnum typ) {
    if (typ == Bodo_CTypes::UINT8) {
        return true;
    } else if (typ == Bodo_CTypes::UINT16) {
        return true;
    } else if (typ == Bodo_CTypes::UINT32) {
        return true;
    } else if (typ == Bodo_CTypes::UINT64) {
        return true;
    }
    return false;
}

constexpr inline bool is_integer(Bodo_CTypes::CTypeEnum typ) {
    if (is_unsigned_integer(typ)) {
        return true;
    } else if (typ == Bodo_CTypes::INT8) {
        return true;
    } else if (typ == Bodo_CTypes::INT16) {
        return true;
    } else if (typ == Bodo_CTypes::INT32) {
        return true;
    } else if (typ == Bodo_CTypes::INT64) {
        return true;
    } else if (typ == Bodo_CTypes::INT128) {
        return true;
    }
    return false;
}

/**
 * @brief Check if typ is FLOAT32 or FLOAT64.
 */
constexpr inline bool is_float(Bodo_CTypes::CTypeEnum typ) {
    return ((typ == Bodo_CTypes::FLOAT32) || (typ == Bodo_CTypes::FLOAT64));
}

/**
 * @brief Check if typ is COMPLEX64 or COMPLEX128.
 */
constexpr inline bool is_complex(Bodo_CTypes::CTypeEnum typ) {
    return ((typ == Bodo_CTypes::COMPLEX64) ||
            (typ == Bodo_CTypes::COMPLEX128));
}

/**
 * @brief Check if typ is an integer, floating, complex or decimal type.
 */
constexpr inline bool is_numerical(Bodo_CTypes::CTypeEnum typ) {
    return (is_integer(typ) || is_float(typ) || is_complex(typ) ||
            (typ == Bodo_CTypes::DECIMAL));
}

template <Bodo_CTypes::CTypeEnum DType>
    requires(is_complex(DType))
using complex_type =
    std::conditional_t<DType == Bodo_CTypes::COMPLEX128, std::complex<double>,
                       std::complex<float>>;
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
    for (size_t u = 0; u < sizeof(T); u++) {
        V[u] = charptr[u];
    }
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
    if (dtype == Bodo_CTypes::_BOOL) {
        return GetCharVector<bool>(false);
    } else if (dtype == Bodo_CTypes::INT8) {
        return GetCharVector<int8_t>(-1);
    } else if (dtype == Bodo_CTypes::UINT8) {
        return GetCharVector<uint8_t>(0);
    } else if (dtype == Bodo_CTypes::INT16) {
        return GetCharVector<int16_t>(-1);
    } else if (dtype == Bodo_CTypes::UINT16) {
        return GetCharVector<uint16_t>(0);
    } else if (dtype == Bodo_CTypes::INT32) {
        return GetCharVector<int32_t>(-1);
    } else if (dtype == Bodo_CTypes::UINT32) {
        return GetCharVector<uint32_t>(0);
    } else if (dtype == Bodo_CTypes::INT64) {
        return GetCharVector<int64_t>(-1);
    } else if (dtype == Bodo_CTypes::UINT64) {
        return GetCharVector<uint64_t>(0);
    } else if (dtype == Bodo_CTypes::DATE) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "In DATE case missing values are handled by "
                             "NULLABLE_INT_BOOL so this case is impossible");
    } else if (dtype == Bodo_CTypes::DATETIME ||
               dtype == Bodo_CTypes::TIMEDELTA ||
               dtype == Bodo_CTypes::TIMESTAMPTZ) {
        return GetCharVector<int64_t>(std::numeric_limits<int64_t>::min());
    } else if (dtype == Bodo_CTypes::FLOAT32) {
        return GetCharVector<float>(std::nanf("1"));
    } else if (dtype == Bodo_CTypes::FLOAT64) {
        return GetCharVector<double>(std::nan("1"));
    } else if (dtype == Bodo_CTypes::DECIMAL) {
        // Normally the null value of decimal_value should never show up
        // anywhere. A value is assigned for simplicity of the code
        __int128_t e_val = 0;
        return GetCharVector<__int128_t>(e_val);
    }
    return {};
}

// for numpy arrays, this maps dtype to sizeof(dtype)
extern const std::vector<size_t> numpy_item_size;

/**
 * @brief enum for array types supported by Bodo
 * These are also defined in utils/utils.py and must match.
 */
struct bodo_array_type {
    enum arr_type_enum {
        NUMPY = 0,
        STRING = 1,
        NULLABLE_INT_BOOL = 2,
        STRUCT = 3,
        CATEGORICAL = 4,
        ARRAY_ITEM = 5,
        INTERVAL = 6,
        DICT = 7,  // dictionary-encoded string array
        // string_array_split_view_type, etc.
        MAP = 8,
        TIMESTAMPTZ = 9,

        // Used to fallback to runtime type checks
        // for templated functions
        UNKNOWN = 15,
    };
};

/**
 * @brief Is typ a nested type (STRUCT, ARRAY_ITEM/LIST or MAP)?
 *
 */
inline bool is_nested_arr_type(bodo_array_type::arr_type_enum typ) {
    return ((typ == bodo_array_type::ARRAY_ITEM) ||
            (typ == bodo_array_type::STRUCT) || (typ == bodo_array_type::MAP));
}

/**
 * @brief Checks decimal scale and precision is in the valid range
 *
 */
inline bool is_valid_decimal128(int8_t precision, int8_t scale) {
    return (precision > 0 && scale >= 0 && precision <= 38 && scale < 38);
}

std::string GetDtype_as_string(Bodo_CTypes::CTypeEnum const& dtype);

inline std::string GetDtype_as_string(int8_t dtype) {
    return GetDtype_as_string(static_cast<Bodo_CTypes::CTypeEnum>(dtype));
}

std::string GetArrType_as_string(bodo_array_type::arr_type_enum arr_type);

inline std::string GetArrType_as_string(int8_t arr_type) {
    return GetArrType_as_string(
        static_cast<bodo_array_type::arr_type_enum>(arr_type));
}

namespace bodo {
/**
 * @brief Table metadata similar to Arrow's:
 * https://github.com/apache/arrow/blob/5e9fce493f21098d616f08034bc233fcc529b3ad/cpp/src/arrow/util/key_value_metadata.h#L36
 * Currently, the only key should be "pandas", which is used only to reconstruct
 * Index columns on the Python side (passed to Python with
 * table->schema()->ToArrowSchema()).
 */
struct TableMetadata {
    const std::vector<std::string> keys;
    const std::vector<std::string> values;

    /**
     * @brief Construct a new Table Metadata object by appending another
     * TableMetadata to this one.
     * @param other Other TableMetadata to append
     * @return TableMetadata New TableMetadata object
     */
    TableMetadata append(TableMetadata const& other) const;
};

/**
 * @brief Wrapper class for uniquely identifying the type of a Bodo Array.
 * Array types are a composition of a bodo_array_type (e.g. NUMPY, STRING, etc.)
 * and inner dtype (e.g. INT8, UINT8, etc.).
 */
struct DataType {
    const bodo_array_type::arr_type_enum array_type;
    const Bodo_CTypes::CTypeEnum c_type;
    const int8_t precision;
    const int8_t scale;
    const std::string timezone;  // for DATETIME types.

    /**
     * @brief Construct a new DataType from a bodo_array_type and CTypeEnum
     * @param array_type Type / Structure of the Array
     * @param c_type Type of the Array Elements
     * @param precision The precision (required for DECIMAL types)
     * @param scale The scale (required for DECIMAL types)
     * @param timezone The timezone (optional for DATETIME types)
     */
    DataType(bodo_array_type::arr_type_enum array_type,
             Bodo_CTypes::CTypeEnum c_type, int8_t precision = -1,
             int8_t scale = -1, std::string timezone = "")
        : array_type(array_type),
          c_type(c_type),
          precision(precision),
          scale(scale),
          timezone(std::move(timezone)) {
        // TODO: For decimal types, check if scale and precision are valid and
        // throw some exception (this will likely cause issues due to other
        // places where they are not being set properly.)
    }
    // Use copy instead
    DataType(const DataType& other) = delete;

    virtual ~DataType() = default;

    /// @brief Is the Array Primitive (i.e. not nested)
    bool is_primitive() {
        return array_type != bodo_array_type::STRUCT &&
               array_type != bodo_array_type::ARRAY_ITEM &&
               array_type != bodo_array_type::MAP &&
               array_type != bodo_array_type::DICT;
    }

    /// @brief Is the Array a Nested Array?
    bool is_array() const { return array_type == bodo_array_type::ARRAY_ITEM; }

    /// @brief If the Array a Struct Array?
    bool is_struct() const { return array_type == bodo_array_type::STRUCT; }

    /// @brief If the Array a Map Array?
    bool is_map() const { return array_type == bodo_array_type::MAP; }

    /// @brief Helper Function to Generate the Output of ToString()
    virtual void to_string_inner(std::string& out);

    /// @brief Convert the DataType to a single-line string
    std::string ToString() {
        std::string out;
        to_string_inner(out);
        return out;
    }
    /**
     * @brief Construct a bodo::DataType from a serialized DataType from Python.
     * The serialized DataType consists of a vector of bodo_array_types and
     * a vector of CTypes.
     *
     * @param arr_array_types First half of serialization, array types
     * @param arr_c_types Second half of serialization, content types
     * @return std::unique_ptr<DataType> Output DataType
     */
    static std::unique_ptr<DataType> Deserialize(
        const std::span<const int8_t> arr_array_types,
        const std::span<const int8_t> arr_c_types);

    ///@brief Serialize a bodo::Schema to a Python <-> C++ communication format
    virtual void Serialize(std::vector<int8_t>& arr_array_types,
                           std::vector<int8_t>& arr_c_types) const;

    /// @brief Convert to the equivalent Arrow field type
    virtual std::shared_ptr<::arrow::Field> ToArrowType(
        std::string& name) const;

    virtual std::shared_ptr<::arrow::DataType> ToArrowDataType() const;

    ///@brief Deep copy the Datatype, returns the proper child type if
    /// appropriate
    std::unique_ptr<DataType> copy() const;

    /// @brief Convert type to a nullable type (Numpy int/float/bool to
    /// nullable)
    std::unique_ptr<DataType> to_nullable_type() const;
};

/// @brief Wrapper class for Representing the Type of ArrayItem Arrays
struct ArrayType final : public DataType {
    const std::unique_ptr<DataType> value_type;

    /// @brief Construct a new ArrayType given the Inner Array DataType
    ArrayType(std::unique_ptr<DataType>&& _value_type)
        : DataType(bodo_array_type::ARRAY_ITEM, Bodo_CTypes::LIST),
          value_type(std::move(_value_type)) {}

    void to_string_inner(std::string& out) override;

    void Serialize(std::vector<int8_t>& arr_array_types,
                   std::vector<int8_t>& arr_c_types) const override;

    std::shared_ptr<::arrow::Field> ToArrowType(
        std::string& name) const override;
};

/// @brief Wrapper class for Representing the Type of Struct Arrays
struct StructType final : public DataType {
    std::vector<std::unique_ptr<DataType>> child_types;

    /// @brief Construct a new StructType given the Inner Array DataTypes
    StructType(std::vector<std::unique_ptr<DataType>>&& _child_types)
        : DataType(bodo_array_type::STRUCT, Bodo_CTypes::STRUCT),
          child_types(std::move(_child_types)) {}

    void to_string_inner(std::string& out) override;

    void Serialize(std::vector<int8_t>& arr_array_types,
                   std::vector<int8_t>& arr_c_types) const override;

    std::shared_ptr<::arrow::Field> ToArrowType(
        std::string& name) const override;
};

/// @brief Wrapper class for representing the type of Map Arrays
struct MapType final : public DataType {
    std::unique_ptr<DataType> key_type;
    std::unique_ptr<DataType> value_type;

    /// @brief Construct a new MapType given the key and value types
    MapType(std::unique_ptr<DataType>&& _key_type,
            std::unique_ptr<DataType>&& _value_type)
        : DataType(bodo_array_type::MAP, Bodo_CTypes::MAP),
          key_type(std::move(_key_type)),
          value_type(std::move(_value_type)) {}

    void to_string_inner(std::string& out) override;

    void Serialize(std::vector<int8_t>& arr_array_types,
                   std::vector<int8_t>& arr_c_types) const override;

    std::shared_ptr<::arrow::Field> ToArrowType(
        std::string& name) const override;
};

/**
 * @brief Wrapper Class for a Schema of a Bodo Table
 * Consisting of a vector of DataTypes for each column / array.
 */
struct Schema {
    std::vector<std::unique_ptr<DataType>> column_types;
    std::vector<std::string> column_names;
    std::shared_ptr<TableMetadata> metadata;
    Schema();
    Schema(const Schema& other);
    Schema(Schema&& other);
    Schema(std::vector<std::unique_ptr<bodo::DataType>>&& column_types_);
    Schema(std::vector<std::unique_ptr<bodo::DataType>>&& column_types_,
           std::vector<std::string> column_names);
    Schema(std::vector<std::unique_ptr<bodo::DataType>>&& column_types_,
           std::vector<std::string> column_names,
           std::shared_ptr<TableMetadata> metadata);
    /** @brief Return the number of columns in the schema
     *
     * @return void The number of columns in the schema
     */
    size_t ncols() const;

    /** @brief Insert a column to the schema
     * @param col The column to insert
     * @param idx The index to insert the column
     * @return void
     */
    void insert_column(std::unique_ptr<DataType>&& col, const size_t idx);

    /** @brief Insert a column to the schema
     * @param arr_array_type the array type of the column to insert
     * @param arr_c_type the c type of the column to insert
     * @param size_t the index to insert to
     * @return void
     */
    void insert_column(const int8_t arr_array_type, const int8_t arr_c_type,
                       const size_t idx);
    /** @brief Append a column to the schema
     * @param col The column to append
     * @return void
     */
    void append_column(std::unique_ptr<DataType>&& col);
    /** @brief Append a column to the schema
     * @param arr_array_type the array type of the column to append
     * @param arr_c_type the c type of the column to append
     * @return void
     */
    void append_column(const int8_t arr_array_type, const int8_t arr_c_type);

    /** @brief Append a schema to the current schema
     * @param other The schema to append
     * @return void
     */
    void append_schema(std::unique_ptr<Schema>&& other);

    /**
     * @brief Construct a bodo::Schema from a serialized schema from Python.
     * The serialized schema consists of a vector of bodo_array_types and
     * a vector of CTypes.
     *
     * @param arr_array_types First half of serialization, array types
     * @param arr_c_types Second half of serialization, content types
     * @return std::unique_ptr<Schema> Output Schema
     */
    static std::unique_ptr<Schema> Deserialize(
        const std::span<const int8_t> arr_array_types,
        const std::span<const int8_t> arr_c_types);

    /**
     * @brief Serialize a bodo::Schema to a Python <-> C++ communication
     * format. The serialized schema consists of a vector of bodo_array_types
     * and a vector of CTypes.
     *
     * @return (arr_array_types, arr_c_types) Serialization vectors
     */
    std::pair<std::vector<int8_t>, std::vector<int8_t>> Serialize() const;

    /**
     * @brief Get string representation of a Schema.
     *
     * @return std::string
     */
    std::string ToString(bool use_col_names = false);

    /**
     * @brief Return a new schema with only the first 'first_n' columns.
     *
     * @param first_n Number of columns to keep.
     * @return std::unique_ptr<Schema> New schema.
     */
    std::unique_ptr<Schema> Project(size_t first_n) const;

    /**
     * @brief Same as the previous, except it provides the column indices to
     * keep.
     *
     * @tparam T Integer type for indices
     * @param column_indices Column indices to keep in the new schema.
     * @return std::unique_ptr<Schema> New schema.
     */
    template <typename T>
        requires(std::integral<T> && !std::same_as<T, bool>)
    std::unique_ptr<Schema> Project(const std::vector<T>& column_indices) const;

    /// @brief Convert to an Arrow schema
    std::shared_ptr<::arrow::Schema> ToArrowSchema() const;

    /// @brief Convert from an Arrow schema to a Bodo schema
    static std::shared_ptr<Schema> FromArrowSchema(
        std::shared_ptr<::arrow::Schema> schema);

    // @brief Deep copy of the Schema
    std::unique_ptr<Schema> copy() const;
};

}  // namespace bodo

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
 * case of DICT:
 * --- child_arrays[0] is the string array for the dictionary.
 * --- child_arrays[1] is a Int32 array for the indices. This array and
 *     the main info share a bitmap.
 * case of ARRAY_ITEM:
 * --- The length is the number of rows.
 * --- data1 is for the offsets
 * --- null_bitmask is the mask
 * --- child_arrays[0] is the inner array
 * case of STRUCT:
 * --- The length is the number of rows.
 * --- null_bitmask is the mask
 * --- child_arrays is the arrays for all fields
 * --- field_names is the field names
 *
 * case of MAP:
 * MAP array is just a wrapper around list(struct) array that stores the
 * key/value pairs (the same as the Python side). The only used fields are
 * length and child_arrays which includes the list(struct) array.
 */
struct array_info {
    bodo_array_type::arr_type_enum arr_type;
    Bodo_CTypes::CTypeEnum dtype;
    uint64_t length;  // number of elements in the array (not bytes) For DICT
                      // arrays this is the length of indices array

    // Data buffer meminfos for this array, e.g. data/offsets/null_bitmap for
    // string array. Last element is always the null bitmap buffer.
    std::vector<std::shared_ptr<BodoBuffer>> buffers;

    // Child arrays for dict-encoded, nested and struct array
    std::vector<std::shared_ptr<array_info>> child_arrays;
    // name of each field in struct array (empty for other arrays)
    std::vector<std::string> field_names;

    int32_t precision;        // for array of decimals and times
    int32_t scale;            // for array of decimals
    uint64_t num_categories;  // for categorical arrays
    std::string timezone;     // timezone info for timestamp arrays
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
               int64_t _offset = 0, std::vector<std::string> _field_names = {},
               std::string _timezone_param = "")
        : arr_type(_arr_type),
          dtype(_dtype),
          length(_length),
          buffers(std::move(_buffers)),
          child_arrays(std::move(_child_arrays)),
          field_names(std::move(_field_names)),
          precision(_precision),
          scale(_scale),
          num_categories(_num_categories),
          timezone(std::move(_timezone_param)),
          array_id(_array_id),
          is_globally_replicated(_is_globally_replicated),
          is_locally_unique(_is_locally_unique),
          is_locally_sorted(_is_locally_sorted),
          offset(_offset) {}

    /**
     * @brief returns the first data pointer for the array if any.
     *
     * @return U* data pointer
     */
    template <
        bodo_array_type::arr_type_enum arr_type = bodo_array_type::UNKNOWN,
        typename U = char>
        requires(arr_type == bodo_array_type::UNKNOWN)
    U* data1() const {
        switch (this->arr_type) {
            case bodo_array_type::NULLABLE_INT_BOOL:
            case bodo_array_type::STRING:
            case bodo_array_type::NUMPY:
            case bodo_array_type::CATEGORICAL:
            case bodo_array_type::INTERVAL:
            case bodo_array_type::ARRAY_ITEM:
            case bodo_array_type::TIMESTAMPTZ:
                return reinterpret_cast<U*>(this->buffers[0]->mutable_data() +
                                            this->offset);
            case bodo_array_type::DICT:
            case bodo_array_type::STRUCT:
            case bodo_array_type::MAP:
            default:
                return nullptr;
        }
    }

    /**
     * @brief returns the first data pointer for the array if any.
     *
     * @return U* data pointer
     */
    template <bodo_array_type::arr_type_enum arr_type, typename U = char>
        requires(arr_type != bodo_array_type::UNKNOWN)
    U* data1() const {
        switch (arr_type) {
            case bodo_array_type::NULLABLE_INT_BOOL:
            case bodo_array_type::STRING:
            case bodo_array_type::NUMPY:
            case bodo_array_type::CATEGORICAL:
            case bodo_array_type::INTERVAL:
            case bodo_array_type::ARRAY_ITEM:
            case bodo_array_type::TIMESTAMPTZ:
                return reinterpret_cast<U*>(this->buffers[0]->mutable_data() +
                                            this->offset);
            case bodo_array_type::DICT:
            case bodo_array_type::STRUCT:
            case bodo_array_type::MAP:
            default:
                return nullptr;
        }
    }

    /**
     * @brief returns the second data pointer for the array if any.
     *
     * @return U* data pointer
     */
    template <
        bodo_array_type::arr_type_enum arr_type = bodo_array_type::UNKNOWN,
        typename U = char>
        requires(arr_type == bodo_array_type::UNKNOWN)
    U* data2() const {
        switch (this->arr_type) {
            case bodo_array_type::STRING:
            case bodo_array_type::INTERVAL:
            case bodo_array_type::TIMESTAMPTZ:
                return reinterpret_cast<U*>(this->buffers[1]->mutable_data());
            case bodo_array_type::DICT:
            case bodo_array_type::ARRAY_ITEM:
            case bodo_array_type::STRUCT:
            case bodo_array_type::MAP:
            case bodo_array_type::NULLABLE_INT_BOOL:
            case bodo_array_type::NUMPY:
            case bodo_array_type::CATEGORICAL:
            default:
                return nullptr;
        }
    }

    /**
     * @brief returns the second data pointer for the array if any.
     *
     * @return U* data pointer
     */
    template <
        bodo_array_type::arr_type_enum arr_type = bodo_array_type::UNKNOWN,
        typename U = char>
        requires(arr_type != bodo_array_type::UNKNOWN)
    U* data2() const {
        switch (arr_type) {
            case bodo_array_type::STRING:
            case bodo_array_type::INTERVAL:
            case bodo_array_type::TIMESTAMPTZ:
                return reinterpret_cast<U*>(this->buffers[1]->mutable_data());
            case bodo_array_type::DICT:
            case bodo_array_type::ARRAY_ITEM:
            case bodo_array_type::STRUCT:
            case bodo_array_type::MAP:
            case bodo_array_type::NULLABLE_INT_BOOL:
            case bodo_array_type::NUMPY:
            case bodo_array_type::CATEGORICAL:
            default:
                return nullptr;
        }
    }

    /**
     * @brief returns the pointer to null bitmask buffer for the array if any.
     *
     * @return char* null bitmask pointer
     */
    template <
        bodo_array_type::arr_type_enum arr_type = bodo_array_type::UNKNOWN>
        requires(arr_type == bodo_array_type::UNKNOWN)
    char* null_bitmask() const {
        switch (this->arr_type) {
            case bodo_array_type::NULLABLE_INT_BOOL:
            case bodo_array_type::ARRAY_ITEM:
                return (char*)this->buffers[1]->mutable_data();
            case bodo_array_type::STRING:
            case bodo_array_type::TIMESTAMPTZ:
                return (char*)this->buffers[2]->mutable_data();
            case bodo_array_type::DICT:
                return (char*)this->child_arrays[1]
                    ->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>();
            case bodo_array_type::STRUCT:
                return (char*)this->buffers[0]->mutable_data();
            case bodo_array_type::MAP:
                return (char*)this->child_arrays[0]
                    ->null_bitmask<bodo_array_type::ARRAY_ITEM>();
            case bodo_array_type::INTERVAL:
            case bodo_array_type::NUMPY:
            case bodo_array_type::CATEGORICAL:
            default:
                return nullptr;
        }
    }

    /**
     * @brief returns the pointer to null bitmask buffer for the array if any.
     *
     * @return char* null bitmask pointer
     */
    template <
        bodo_array_type::arr_type_enum arr_type = bodo_array_type::UNKNOWN>
        requires(arr_type != bodo_array_type::UNKNOWN)
    char* null_bitmask() const {
        switch (arr_type) {
            case bodo_array_type::NULLABLE_INT_BOOL:
            case bodo_array_type::ARRAY_ITEM:
                return (char*)this->buffers[1]->mutable_data();
            case bodo_array_type::STRING:
            case bodo_array_type::TIMESTAMPTZ:
                return (char*)this->buffers[2]->mutable_data();
            case bodo_array_type::DICT:
                return (char*)this->child_arrays[1]
                    ->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>();
            case bodo_array_type::STRUCT:
                return (char*)this->buffers[0]->mutable_data();
            case bodo_array_type::MAP:
                return (char*)this->child_arrays[0]
                    ->null_bitmask<bodo_array_type::ARRAY_ITEM>();
            case bodo_array_type::INTERVAL:
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
        return -1;
    }

    template <typename T, bodo_array_type::arr_type_enum arr_type =
                              bodo_array_type::UNKNOWN>
    T& at(size_t idx) {
        return ((T*)data1<arr_type>())[idx];
    }

    template <typename T, bodo_array_type::arr_type_enum arr_type =
                              bodo_array_type::UNKNOWN>
    const T& at(size_t idx) const {
        return ((T*)data1<arr_type>())[idx];
    }

    template <
        bodo_array_type::arr_type_enum arr_type = bodo_array_type::UNKNOWN>
    bool get_null_bit(size_t idx) const {
        return GetBit((uint8_t*)null_bitmask<arr_type>(), idx);
    }

    /**
     * Return code in position `idx` of categorical array as int64
     */
    int64_t get_code_as_int64(size_t idx) {
        if (this->arr_type != bodo_array_type::CATEGORICAL) {
            throw std::runtime_error("get_code: not a categorical array");
        }
        switch (dtype) {
            case Bodo_CTypes::INT8:
                return (int64_t)(this->at<int8_t, bodo_array_type::CATEGORICAL>(
                    idx));
            case Bodo_CTypes::INT16:
                return (
                    int64_t)(this->at<int16_t, bodo_array_type::CATEGORICAL>(
                    idx));
            case Bodo_CTypes::INT32:
                return (
                    int64_t)(this->at<int32_t, bodo_array_type::CATEGORICAL>(
                    idx));
            case Bodo_CTypes::INT64:
                return this->at<int64_t, bodo_array_type::CATEGORICAL>(idx);
            default:
                throw std::runtime_error(
                    "get_code: codes have unrecognized dtype");
        }
    }

    template <
        bodo_array_type::arr_type_enum arr_type = bodo_array_type::UNKNOWN>
    void set_null_bit(size_t idx, bool bit) {
        SetBitTo((uint8_t*)null_bitmask<arr_type>(), idx, bit);
    }

    void pin() {
        for (auto& buffer : this->buffers) {
            buffer->pin();
        }
        for (auto& arr : this->child_arrays) {
            arr->pin();
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
            default:
                for (auto& buffer : this->buffers) {
                    buffer->unpin();
                }
                for (auto& arr : this->child_arrays) {
                    arr->unpin();
                }
                break;
        }
    }

    template <typename X, typename... Args>
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
            case bodo_array_type::STRING:
            case bodo_array_type::DICT:
            case bodo_array_type::TIMESTAMPTZ:
            case bodo_array_type::NULLABLE_INT_BOOL:
            case bodo_array_type::ARRAY_ITEM:
            case bodo_array_type::STRUCT:
            case bodo_array_type::MAP:
            case bodo_array_type::CATEGORICAL:
                return true;
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
        switch (this->arr_type) {
            case bodo_array_type::STRING:
            case bodo_array_type::DICT:
            case bodo_array_type::TIMESTAMPTZ:
            case bodo_array_type::NULLABLE_INT_BOOL:
            case bodo_array_type::ARRAY_ITEM:
            case bodo_array_type::STRUCT:
            case bodo_array_type::MAP: {
                const uint8_t* _null_bitmask = (uint8_t*)this->null_bitmask();
                for (size_t i = 0; i < length; i++) {
                    not_na[i] = GetBit(_null_bitmask, i);
                }
            } break;

            case bodo_array_type::CATEGORICAL:
                for (size_t i = 0; i < length; i++) {
                    // TODO: Ensure templated for faster access.
                    not_na[i] = get_code_as_int64(i) != -1;
                }
                break;
            case bodo_array_type::NUMPY:
                if (dtype == Bodo_CTypes::TIMEDELTA) {
                    int64_t* data =
                        (int64_t*)this->data1<bodo_array_type::NUMPY>();
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

    std::unique_ptr<bodo::DataType> data_type() {
        if (arr_type == bodo_array_type::ARRAY_ITEM) {
            auto inner = child_arrays[0]->data_type();
            return std::make_unique<bodo::ArrayType>(std::move(inner));
        } else if (arr_type == bodo_array_type::STRUCT) {
            std::vector<std::unique_ptr<bodo::DataType>> child_types;
            for (auto const& child : child_arrays) {
                child_types.push_back(child->data_type());
            }
            return std::make_unique<bodo::StructType>(std::move(child_types));
        } else if (arr_type == bodo_array_type::MAP) {
            std::shared_ptr<array_info> inner_struct =
                child_arrays[0]->child_arrays[0];
            std::unique_ptr<bodo::DataType> key_type =
                inner_struct->child_arrays[0]->data_type();
            std::unique_ptr<bodo::DataType> value_type =
                inner_struct->child_arrays[1]->data_type();
            return std::make_unique<bodo::MapType>(std::move(key_type),
                                                   std::move(value_type));
        } else {
            return std::make_unique<bodo::DataType>(
                arr_type, dtype, this->precision, this->scale, this->timezone);
        }
    }
};

std::unique_ptr<array_info> alloc_numpy(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

/**
 * @brief Allocate a numpy array with all nulls.
 * If the provided dtype does not have a sentinel representation for nulls,
 * this function will throw an exception.
 *
 * @param length The length of the array
 * @param typ_enum The dtype of the array. Only some dtypes are supported.
 * @param pool The buffer pool to use for allocations.
 * @param mm The memory manager to use for allocations.
 * @return std::unique_ptr<array_info> The allocated array of all nulls.
 */
std::unique_ptr<array_info> alloc_numpy_array_all_nulls(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

//  NOTE: extra_null_bytes is used to account for padding in null buffer
//  for process boundaries in shuffle (bits of two different processes cannot be
//  packed in the same byte).
std::unique_ptr<array_info> alloc_array_item(
    int64_t n_arrays, std::shared_ptr<array_info> inner_arr,
    int64_t extra_null_bytes = 0,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

/**
 * @brief Allocate array_item_array with all nulls. The inner array is required
 * for type stability, but may be of length 0.
 *
 */
std::unique_ptr<array_info> alloc_array_item_all_nulls(
    int64_t n_arrays, std::shared_ptr<array_info> inner_arr,
    int64_t extra_null_bytes = 0,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    const std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

/**
 * @brief Allocate a STRUCT array
 * @param length length of the STRUCT array
 * @param child_arrays child arrays of the STRUCT array
 * @param extra_null_bytes used to account for padding in null buffer
 * for process boundaries in shuffle (bits of two different processes cannot be
 * packed in the same byte).
 *
 * @return pointer to the allocated array_info
 */
std::unique_ptr<array_info> alloc_struct(
    int64_t length, std::vector<std::shared_ptr<array_info>> child_arrays,
    int64_t extra_null_bytes = 0,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

std::unique_ptr<array_info> alloc_map(int64_t n_rows,
                                      std::shared_ptr<array_info> inner_arr);

std::unique_ptr<array_info> alloc_categorical(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum, int64_t num_categories,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

std::unique_ptr<array_info> alloc_categorical_array_all_nulls(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum, int64_t num_categories,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

std::unique_ptr<array_info> alloc_nullable_array(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum,
    int64_t extra_null_bytes = 0,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager(),
    std::string timezone = "");

std::unique_ptr<array_info> alloc_nullable_array_no_nulls(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum,
    int64_t extra_null_bytes = 0,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

std::unique_ptr<array_info> alloc_nullable_array_all_nulls(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum,
    int64_t extra_null_bytes = 0,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

std::unique_ptr<array_info> alloc_string_array(
    Bodo_CTypes::CTypeEnum typ_enum, int64_t length, int64_t n_chars,
    int64_t array_id = -1, int64_t extra_null_bytes = 0,
    bool is_globally_replicated = false, bool is_locally_unique = false,
    bool is_locally_sorted = false,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

std::unique_ptr<array_info> alloc_string_array_all_nulls(
    Bodo_CTypes::CTypeEnum typ_enum, int64_t length,
    int64_t extra_null_bytes = 0,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

std::unique_ptr<array_info> alloc_interval_array(
    int64_t length, Bodo_CTypes::CTypeEnum typ_enum,
    bodo::IBufferPool* const pool,
    const std::shared_ptr<::arrow::MemoryManager> mm);

//  NOTE: extra_null_bytes is used to account for padding in null buffer
//  for process boundaries in shuffle (bits of two different processes cannot be
//  packed in the same byte).
std::unique_ptr<array_info> alloc_dict_string_array(
    int64_t length, int64_t n_keys, int64_t n_chars_keys,
    int64_t extra_null_bytes = 0,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

/**
 * @brief Allocate an empty dictionary string array. By default
 * this generates an empty dictionary, but callers can "swap" the
 * underlying dictionary to unify with another dictionary.
 *
 * @param length The number of null rows.
 * @param extra_null_bytes An extra null bytes to allocate.
 * @param pool The buffer pool to use for allocations.
 * @param mm The memory manager to use for allocations.
 * @return std::unique_ptr<array_info> The allocated dictionary string array.
 */
std::unique_ptr<array_info> alloc_dict_string_array_all_nulls(
    int64_t length, int64_t extra_null_bytes = 0,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

std::unique_ptr<array_info> alloc_timestamptz_array(
    int64_t length, int64_t extra_null_bytes = 0,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

std::unique_ptr<array_info> alloc_timestamptz_array_all_nulls(
    int64_t length, int64_t extra_null_bytes = 0,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

/**
 * @brief Allocate a fully null array for each array type. This throws
 * an error if the array type + dtype combination is not supported.
 *
 * @param length The length of the array to allocate.
 * @param arr_type The array type to allocate.
 * @param dtype The dtype to allocate.
 * @param extra_null_bytes The extra null bytes to allocate.
 * @param num_categories The number of categories for categorical arrays.
 * @param pool The operator pool to use for allocations.
 * @param mm The memory manager to use for allocations.
 * @return std::unique_ptr<array_info> The allocated array filled entirely
 * with null values.
 */
std::unique_ptr<array_info> alloc_all_null_array_top_level(
    int64_t length, bodo_array_type::arr_type_enum arr_type,
    Bodo_CTypes::CTypeEnum dtype, int64_t extra_null_bytes = 0,
    int64_t num_categories = 0,
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
 * @param pool Memory pool to use for allocations during the execution of this
 * function.
 * @param mm Memory manager associated with the pool.
 * @return std::shared_ptr<array_info> A string type array info constructed from
 * the vector of strings.
 */
std::unique_ptr<array_info> create_string_array(
    Bodo_CTypes::CTypeEnum typ_enum, bodo::vector<uint8_t> const& null_bitmap,
    bodo::vector<std::string> const& list_string, int64_t array_id = -1,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

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

/**
 * The allocations array function for the function.
 * NOTE: extra_null_bytes is used to account for padding in null buffer
 * for process boundaries in shuffle (bits of two different processes cannot be
 * packed in the same byte).
 *
 * In the case of NUMPY, CATEGORICAL, NULLABLE_INT_BOOL or STRUCT:
 * -- length is the number of rows, and n_sub_elems, n_sub_sub_elems do not
 * matter.
 * In the case of STRING:
 * -- length is the number of rows (= number of strings)
 * -- n_sub_elems is the total number of characters.
 * In the case of DICT:
 * -- length is the number of rows (same as the number of indices)
 * -- n_sub_elems is the number of keys in the dictionary
 * -- n_sub_sub_elems is the total number of characters for
 *    the keys in the dictionary
 * In the case of ARRAY_ITEM or STRUCT:
 * -- length is the number of rows (same as the number of indices)
 * -- Dummy child arrays are returned. The caller is responsible for
 * initializing the child arrays
 */
template <
    bodo_array_type::arr_type_enum const_arr_type = bodo_array_type::UNKNOWN>
std::unique_ptr<array_info> alloc_array_top_level(
    int64_t length, int64_t n_sub_elems, int64_t n_sub_sub_elems,
    bodo_array_type::arr_type_enum arr_type, Bodo_CTypes::CTypeEnum dtype,
    int64_t array_id = -1, int64_t extra_null_bytes = 0,
    int64_t num_categories = 0, bool is_globally_replicated = false,
    bool is_locally_unique = false, bool is_locally_sorted = false,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager(),
    std::string timezone = "") {
    switch (const_arr_type != bodo_array_type::UNKNOWN ? const_arr_type
                                                       : arr_type) {
        case bodo_array_type::STRING:
            return alloc_string_array(dtype, length, n_sub_elems, array_id,
                                      extra_null_bytes, is_globally_replicated,
                                      is_locally_unique, is_locally_sorted,
                                      pool, std::move(mm));

        case bodo_array_type::NULLABLE_INT_BOOL:
            return alloc_nullable_array(length, dtype, extra_null_bytes, pool,
                                        std::move(mm), timezone);

        case bodo_array_type::INTERVAL:
            return alloc_interval_array(length, dtype, pool, std::move(mm));

        case bodo_array_type::NUMPY:
            return alloc_numpy(length, dtype, pool, std::move(mm));

        case bodo_array_type::CATEGORICAL:
            return alloc_categorical(length, dtype, num_categories, pool,
                                     std::move(mm));

        case bodo_array_type::DICT:
            return alloc_dict_string_array(length, n_sub_elems, n_sub_sub_elems,
                                           extra_null_bytes, pool,
                                           std::move(mm));
        case bodo_array_type::TIMESTAMPTZ:
            return alloc_timestamptz_array(length, extra_null_bytes, pool,
                                           std::move(mm));
        case bodo_array_type::ARRAY_ITEM:
            return alloc_array_item(length, nullptr, extra_null_bytes, pool,
                                    std::move(mm));
        case bodo_array_type::STRUCT:
            return alloc_struct(length, {}, extra_null_bytes, pool,
                                std::move(mm));
        case bodo_array_type::MAP: {
            std::unique_ptr<array_info> inner_array = alloc_array_item(
                length, nullptr, extra_null_bytes, pool, std::move(mm));
            return alloc_map(length, std::move(inner_array));
        }
        default:
            throw std::runtime_error("alloc_array: array type (" +
                                     GetArrType_as_string(arr_type) +
                                     ") not supported");
    }
}

/**
 * @brief Allocate an empty array with the same schema as 'in_arr', similar to
 * alloc_table_like function. Currently only used by alloc_table_like function.
 *
 * @param in_arr Reference array
 * @return std::unique_ptr<array_info> Pointer to the allocated array
 */
std::unique_ptr<array_info> alloc_array_like(
    std::shared_ptr<array_info> in_arr, bool reuse_dictionaries = true,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

/* The "get-value" functionality for array_info.
   This is the equivalent of at functionality.
   We cannot use at(idx) statements.
 */

template <typename T,
          bodo_array_type::arr_type_enum arr_type = bodo_array_type::UNKNOWN>
inline T& getv(const std::shared_ptr<array_info>& arr, size_t idx) {
    return ((T*)arr->data1<arr_type>())[idx];
}

template <typename T,
          bodo_array_type::arr_type_enum arr_type = bodo_array_type::UNKNOWN>
inline T& getv(const std::unique_ptr<array_info>& arr, size_t idx) {
    return ((T*)arr->data1<arr_type>())[idx];
}

struct mpi_comm_info {
    int myrank, n_pes;
    int64_t n_rows_send = 0, n_rows_recv = 0;
    bool has_nulls;
    // generally required MPI counts
    std::vector<int64_t> send_count;
    std::vector<int64_t> recv_count;
    std::vector<int64_t> send_disp;
    std::vector<int64_t> recv_disp;
    // counts for arrays with null bitmask
    std::vector<int64_t> send_count_null;
    std::vector<int64_t> recv_count_null;
    std::vector<int64_t> send_disp_null;
    std::vector<int64_t> recv_disp_null;
    size_t n_null_bytes = 0;
    // Store input row to destination rank. This way we only need to do
    // hash_to_rank() once during shuffle. Also stores the result of filtering
    // rows with bloom filters (removed rows have dest == -1), so that we only
    // query the filter once.
    bodo::vector<int> row_dest;
    // true if using a bloom filter to discard rows before shuffling
    bool filtered = false;

    /**
     * @brief Table level constructor for mpi_comm_info. Initialize
     * mpi_comm_info counts and displacement vectors and stats based on input,
     * and computes all information needed for the shuffling of
     * data1/data2/data3 and their sizes.
     *
     * @param arrays Vector of array from input table
     * @param hashes Hashes of all the rows
     * @param is_parallel True if data is distributed (used to indicate whether
     * tracing should be parallel or not)
     * @param filter Bloom filter. Rows whose hash is not in the filter will be
     * discarded from shuffling. If no filter is provided no filtering will
     * happen.
     * @param keep_row_bitmask bitmask specifying if a row should be kept or
     * not. In those cases, the rows will be handled based on the value of the
     * templated parameter keep_filter_misses
     * @param keep_filter_misses : In case a Bloom filter is provided
     * and a key is not present in the bloom filter, should we keep the value on
     * this rank (i.e. not discard it altogether). Similarly, there is a special
     * case that arises when keep_row_bitmask is provided and a row is set to
     * not true, which can happen for example if a row has a null in a key
     * column (i.e. this row cannot match with any other in case of SQL joins)
     *  In these cases, this flag determines
     * whether to keep the row on this rank (i.e. not discard it altogether).
     * This is useful in the outer join case. In groupby, this is used in the
     * MRNF and nunique cases to drop rows using a local reduction before
     * shuffling them. Defaults to false.
     * @param send_only only initialize send counts and not recv counts. This
     * avoids alltoall collectives which is necessary for async shuffle.
     */
    explicit mpi_comm_info(
        const std::vector<std::shared_ptr<array_info>>& arrays,
        const std::shared_ptr<uint32_t[]>& hashes, bool is_parallel,
        const SimdBlockFilterFixed<::hashing::SimpleMixSplit>* filter = nullptr,
        const uint8_t* keep_row_bitmask = nullptr,
        bool keep_filter_misses = false, bool send_only = false);

    /**
     * @brief Construct mpi_comm_info for inner array of array item array.
     * Initialize mpi_comm_info counts and displacement vectors and stats based
     * on the parent array and row dest information.
     *
     * @param parent_arr The parent ARRAY_ITEM array
     * @param parent_comm_info The mpi_comm_info of parent_arr
     * @param send_only only initialize send counts and not recv counts. This
     avoids alltoall collectives which is necessary for async shuffle.
     */
    explicit mpi_comm_info(const std::shared_ptr<array_info>& parent_arr,
                           const mpi_comm_info& parent_comm_info,
                           bool _has_nulls, bool send_only = false);

   private:
    /**
     * @brief Set row_dest and send count vectors
     */
    template <bool keep_filter_misses = false>
    void set_send_count(
        const std::shared_ptr<uint32_t[]>& hashes,
        const SimdBlockFilterFixed<::hashing::SimpleMixSplit>*& filter,
        const uint8_t* keep_row_bitmask, const uint64_t& n_rows);
};

struct mpi_str_comm_info {
    int64_t n_sub_send = 0, n_sub_recv = 0;
    // counts required for string arrays
    std::vector<int64_t> send_count_sub;
    std::vector<int64_t> recv_count_sub;
    std::vector<int64_t> send_disp_sub;
    std::vector<int64_t> recv_disp_sub;

    /**
     * @brief Initialize mpi_str_comm_info based on the array type
     *
     * @param arr_info The input array. If arr_info is neither string nor list
     * string array, the constructor will be a no-op
     * @param comm_info The mpi_comm_info of arr_info
     * @param send_only only initialize send counts and not recv counts. This
     * avoids alltoall collectives which is necessary for async shuffle.
     */
    mpi_str_comm_info(const std::shared_ptr<array_info>& arr_info,
                      const mpi_comm_info& comm_info, bool send_only = false);
};

struct table_info {
    std::vector<std::shared_ptr<array_info>> columns;
    std::vector<std::string> column_names;
    std::shared_ptr<bodo::TableMetadata> metadata;
    // keep shuffle info to be able to reverse the shuffle if necessary
    // currently used in groupby apply
    // TODO: refactor out?
    std::shared_ptr<mpi_comm_info> comm_info;
    std::shared_ptr<uint32_t[]> hashes;
    int id;
    uint64_t _nrows = 0;
    table_info() {}
    explicit table_info(std::vector<std::shared_ptr<array_info>>&& _columns)
        : columns(std::move(_columns)) {}
    explicit table_info(std::vector<std::shared_ptr<array_info>>& _columns)
        : columns(_columns) {}
    explicit table_info(std::vector<std::shared_ptr<array_info>>& _columns,
                        uint64_t nrows)
        : columns(_columns), _nrows(nrows) {}
    explicit table_info(std::vector<std::shared_ptr<array_info>>& _columns,
                        std::vector<std::string> column_names,
                        std::shared_ptr<bodo::TableMetadata> metadata)
        : columns(_columns), column_names(column_names), metadata(metadata) {}
    explicit table_info(std::vector<std::shared_ptr<array_info>>& _columns,
                        uint64_t nrows, std::vector<std::string> column_names,
                        std::shared_ptr<bodo::TableMetadata> metadata)
        : columns(_columns),
          column_names(column_names),
          metadata(metadata),
          _nrows(nrows) {}
    uint64_t ncols() const { return columns.size(); }
    uint64_t nrows() const {
        // TODO: Replace with _nrows always.
        return this->ncols() == 0 ? _nrows : columns[0]->length;
    }
    std::shared_ptr<array_info> operator[](size_t idx) { return columns[idx]; }
    const std::shared_ptr<array_info> operator[](size_t idx) const {
        return columns[idx];
    }

    /// @brief Extract a bodo::Schema from a table_info
    std::unique_ptr<bodo::Schema> schema() {
        std::vector<std::unique_ptr<bodo::DataType>> column_types;
        for (size_t i = 0; i < columns.size(); i++) {
            auto col = columns[i];
            column_types.push_back(col->data_type());
        }

        return std::make_unique<bodo::Schema>(std::move(column_types),
                                              column_names, metadata);
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

    template <typename X, typename... Args>
    friend class ::bodo::pin_guard;
};

/**
 * @brief Get the dtypes and arr types from an existing table
 *
 * @param table Reference table
 * @return std::tuple<std::vector<int8_t>, std::vector<int8_t>> Vector of
 * C types and vector of array types
 */
std::tuple<std::vector<int8_t>, std::vector<int8_t>>
get_dtypes_arr_types_from_table(const std::shared_ptr<table_info>& table);

/**
 * @brief Get the dtypes and arr types from an existing array
 *
 * @param[in] array Reference array
 * @param[in, out] arr_c_types Reference to a vector to append dtypes into
 * @param[in, out] arr_array_types Reference to a vector to append arr types
 * into
 */
void _get_dtypes_arr_types_from_array(const std::shared_ptr<array_info>& array,
                                      std::vector<int8_t>& arr_c_types,
                                      std::vector<int8_t>& arr_array_types);

/**
 * @brief Get the dtypes and arr types from an existing array
 * @param array Reference array
 * @return std::tuple<std::vector<int8_t>, std::vector<int8_t>> Vector of
 * C types and vector of array types
 */
std::tuple<std::vector<int8_t>, std::vector<int8_t>>
get_dtypes_arr_types_from_array(const std::shared_ptr<array_info>& array);

/**
 * @brief Generate a map from column index to the actual index to the array type
 * and C type arrays
 *
 * @param arr_array_types The span of array types
 * @return A vector that maps from the column index to the start index in the
 * type array
 */
std::vector<size_t> get_col_idx_map(
    const std::span<const int8_t>& arr_array_types);

/**
 * @brief Helper function for early reference (and potentially memory)
 * release of a column in a table (to reduce peak memory usage wherever
 * possible). This is useful in performance and memory critical regions to
 * release memory of columns that are no longer needed. However, calling
 * reset on the column is only safe if this is the last reference to the
 * table. We pass the shared_ptr by reference to not incref the table_info
 * during the function call. NOTE: This function is only useful for
 * performance reasons and should only be used when you know what you are
 * doing. See existing usages of this in our code to understand the use
 * cases.
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

/**
 * @brief Calculate the toal memory of local array. Note that the array must be
 * pinned if approximate_string_size is set to false.
 *
 * @param earr input array
 * @param include_dict_size Should the size of dictionaries be included?
 * @param include_children Should the size of children be included?
 * @param approximate_string_size Should the total number of chars in string
 * arrays be computed or approximated by buffer size?
 * @return int64_t total size of input array in memory
 */
int64_t array_memory_size(std::shared_ptr<array_info> earr,
                          bool include_dict_size, bool include_children = true,
                          bool approximate_string_size = false);

/**
 * @brief Calculate the total memory of the dictionaries of the array
 * @param earr input array
 * @return int64_t total size of the dictionaries of the array
 */
int64_t array_dictionary_memory_size(std::shared_ptr<array_info> earr);
/**
 * Compute the total memory of local chunk of the table on current rank. Note
 * that the table must be pinned if approximate_string_size is set to false.
 *
 * @param table : The input table
 * @param include_dict_size : Should the size of dictionaries be included?
 * @param approximate_string_size Should the total number of chars in string
 * arrays be computed or approximated by buffer size?
 * @return the total size of the local chunk of the table
 */
int64_t table_local_memory_size(const std::shared_ptr<table_info>& table,
                                bool include_dict_size,
                                bool approximate_string_size = false);
/**
 * Compute the total memory of the dictionaries of the table on current rank
 *
 * @param table : The input table
 * @return the total size of the dictionaries of the table
 */
int64_t table_local_dictionary_memory_size(
    const std::shared_ptr<table_info>& table);

/* Compute the total memory of the table across all processors.
 *
 * @param table : The input table
 * @return the total size of the table all over the processors
 */
int64_t table_global_memory_size(const std::shared_ptr<table_info>& table);

/// Initialize numpy_item_size and verify size of dtypes
void bodo_common_init();

/**
 * @brief Make a copy of the given array
 *
 * @param arr the array to be copied
 * @param shallow_copy_child_arrays whether to shallow copy of child arrays for
 * ARRAY_ITEM, STRUCT and MAP array
 * @return The copy
 */
std::shared_ptr<array_info> copy_array(std::shared_ptr<array_info> arr,
                                       bool shallow_copy_child_arrays = false);

/**
 * @brief Calculate the (estimated) size of one row for a table with given
 * the table's schema.
 *
 * @param schema Schema of the table.
 * @return size_t Estimated total size of the row
 */
size_t get_row_bytes(const std::shared_ptr<bodo::Schema>& schema);

/**
 * Free underlying array of array_info pointer and delete the pointer
 */
void delete_info(array_info* arr);

/**
 * delete table pointer and its column array_info pointers (but not the arrays).
 */
void delete_table(table_info* table);

/**
 * @brief Create a new table with table's map columns converted to list(struct)
 *
 * @param table input table
 * @return table_info* output table with map columns converted
 */
table_info* cpp_table_map_to_list(table_info* table);

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

/** Converts the decimal precision to the smallest integer byte
 *  size  it fits in.
 */
int32_t decimal_precision_to_integer_bytes(int32_t precision);

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
    if (!val) {
        memset(ptr, 0, n_bytes);
    } else {
        memset(ptr, 0xff, n_bytes);
    }
}

inline bool is_na(const uint8_t* null_bitmap, int64_t i) {
    return (null_bitmap[i / 8] & kBitmask[i % 8]) == 0;
}
}

/**
 * @brief Convert Arrow DataType to Bodo DataType, including nested types
 *
 * @param arrow_type input Arrow DataType
 * @return std::unique_ptr<bodo::DataType> equivalent Bodo DataType
 */
std::unique_ptr<bodo::DataType> arrow_type_to_bodo_data_type(
    const std::shared_ptr<arrow::DataType> arrow_type);

// Retrieve the bodo version as a string.
std::string get_bodo_version();

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

template <typename T>
struct numba_optional {
    T* value;
    // 0 = false, 1 = true
    // Numba assumes this is 1 byte, the C++ spec doesn't guarantee sizeof(bool)
    // == 1 so we use uint8_t
    uint8_t has_value;
    numba_optional() { numba_optional(nullptr, false); }
    numba_optional(T* _value, uint8_t _has_value)
        : value(_value), has_value(_has_value) {
        // This has to be standard layout since it is intended to be passed from
        // numba generated code
        static_assert(std::is_standard_layout<numba_optional<T>>::value,
                      "numba_optional must be standard layout");
    }
};
