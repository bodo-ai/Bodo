// Copyright (C) 2019 Bodo Inc. All rights reserved.
#pragma once

#include <complex>
#include <concepts>
#include <set>
#include <span>

#include "_bodo_common.h"
#include "_decimal_ext.h"
#include "hyperloglog.hpp"

// convert Array dtype to C type using a trait class
// similar to:
// https://github.com/rapidsai/cudf/blob/c4a1389bca6f2fd521bd5e768eda7407aa3e66b5/cpp/include/cudf/utilities/type_dispatcher.hpp#L141
template <Bodo_CTypes::CTypeEnum T>
struct dtype_to_type {
    using type = void;
};

#ifndef DTYPE_TO_C_TYPE
#define DTYPE_TO_C_TYPE(Type, Id) \
    template <>                   \
    struct dtype_to_type<Id> {    \
        using type = Type;        \
    };
#endif

DTYPE_TO_C_TYPE(int8_t, Bodo_CTypes::INT8)
DTYPE_TO_C_TYPE(int16_t, Bodo_CTypes::INT16)
DTYPE_TO_C_TYPE(int32_t, Bodo_CTypes::INT32)
DTYPE_TO_C_TYPE(int64_t, Bodo_CTypes::INT64)
DTYPE_TO_C_TYPE(uint8_t, Bodo_CTypes::UINT8)
DTYPE_TO_C_TYPE(uint16_t, Bodo_CTypes::UINT16)
DTYPE_TO_C_TYPE(uint32_t, Bodo_CTypes::UINT32)
DTYPE_TO_C_TYPE(uint64_t, Bodo_CTypes::UINT64)
DTYPE_TO_C_TYPE(float, Bodo_CTypes::FLOAT32)
DTYPE_TO_C_TYPE(double, Bodo_CTypes::FLOAT64)
DTYPE_TO_C_TYPE(bool, Bodo_CTypes::_BOOL)

DTYPE_TO_C_TYPE(char*, Bodo_CTypes::STRING)
DTYPE_TO_C_TYPE(char*, Bodo_CTypes::BINARY)
// NOTE: for functions that only need a C type with similar size (for copy,
// equality, ...) but not the actual semantics (e.g. isna)
DTYPE_TO_C_TYPE(int64_t, Bodo_CTypes::DATETIME)
DTYPE_TO_C_TYPE(int64_t, Bodo_CTypes::TIMEDELTA)
DTYPE_TO_C_TYPE(int64_t, Bodo_CTypes::TIME)
DTYPE_TO_C_TYPE(int32_t, Bodo_CTypes::DATE)
DTYPE_TO_C_TYPE(__int128, Bodo_CTypes::DECIMAL)
DTYPE_TO_C_TYPE(__int128, Bodo_CTypes::INT128)
DTYPE_TO_C_TYPE(std::complex<double>, Bodo_CTypes::COMPLEX128)
DTYPE_TO_C_TYPE(std::complex<float>, Bodo_CTypes::COMPLEX64)

#define DICT_INDEX_C_TYPE int32_t

// ------------------------------------------------

template <Bodo_CTypes::CTypeEnum DType>
concept datetime_timedelta =
    (DType == Bodo_CTypes::DATETIME) || (DType == Bodo_CTypes::TIMEDELTA);

template <Bodo_CTypes::CTypeEnum DType>
concept decimal = DType == Bodo_CTypes::DECIMAL;

template <Bodo_CTypes::CTypeEnum DType>
concept float_dtype =
    (DType == Bodo_CTypes::FLOAT32) || (DType == Bodo_CTypes::FLOAT64);

template <Bodo_CTypes::CTypeEnum DType>
concept bool_dtype = DType == Bodo_CTypes::_BOOL;

template <Bodo_CTypes::CTypeEnum DType>
concept integer_dtype =
    (DType == Bodo_CTypes::UINT8) || (DType == Bodo_CTypes::INT8) ||
    (DType == Bodo_CTypes::UINT16) || (DType == Bodo_CTypes::INT16) ||
    (DType == Bodo_CTypes::UINT32) || (DType == Bodo_CTypes::INT32) ||
    (DType == Bodo_CTypes::UINT64) || (DType == Bodo_CTypes::INT64);

template <Bodo_CTypes::CTypeEnum DType>
concept complex_dtype =
    (DType == Bodo_CTypes::COMPLEX64) || (DType == Bodo_CTypes::COMPLEX128);

template <Bodo_CTypes::CTypeEnum DType>
concept complex128_dtype = DType == Bodo_CTypes::COMPLEX128;

template <Bodo_CTypes::CTypeEnum DType>
concept complex64_dtype = DType == Bodo_CTypes::COMPLEX64;

template <Bodo_CTypes::CTypeEnum DType>
concept numeric_dtype = decimal<DType> || float_dtype<DType> ||
                        integer_dtype<DType> || complex_dtype<DType>;

// ------------------------------------------------

// select dtypes that can have sentinel nulls
template <Bodo_CTypes::CTypeEnum DType>
concept NullSentinelDtype = (float_dtype<DType> || datetime_timedelta<DType>);

// select dtypes that can have sentinel NAs in SQL which
// represent NULL.
template <Bodo_CTypes::CTypeEnum DType>
concept SQLNASentinelDtype = datetime_timedelta<DType>;

/**
 * @brief Returns a NA value for the corresponding type. Used for
 * dtypes that do not have a value for NA.
 *
 * For comparisons, use isnan_alltype instead!
 *
 * @return Nothing, throws a runtime error.
 */
template <typename T, Bodo_CTypes::CTypeEnum DType>
constexpr inline T nan_val() {
    throw std::runtime_error(
        "_array_utils.h::nan_val: No NA val exists for this type");
}

/**
 * @brief Returns a NA value for the corresponding type. Used for
 * dtypes that use a special value (e.g. the smallest integer)
 * as a sentinal value for NA.
 *
 * For comparisons, use isnan_alltype instead!
 *
 * @return The sentinel value that stands in for NA (or specifically NaT)
 */
template <typename T, Bodo_CTypes::CTypeEnum DType>
    requires datetime_timedelta<DType>
constexpr inline T nan_val() {
    return std::numeric_limits<T>::min();
}

/**
 * @brief Returns a NA value for the corresponding type. Used for
 * dtypes that have a real NaN value.
 *
 * For comparisons, use isnan_alltype instead!
 *
 * @return The NA value from the corresponding dtype.
 */
template <typename T, Bodo_CTypes::CTypeEnum DType>
    requires(float_dtype<DType> || decimal<DType>)
constexpr inline T nan_val() {
    return std::numeric_limits<T>::quiet_NaN();
}

// ------------------------------------------------

inline void CheckEqualityArrayType(std::shared_ptr<array_info> arr1,
                                   std::shared_ptr<array_info> arr2) {
    if (arr1->arr_type != arr2->arr_type) {
        throw std::runtime_error(
            "array_info passed to Cpp code have different arr_type");
    }
    if (arr1->arr_type != bodo_array_type::STRING) {
        if (arr1->dtype != arr2->dtype) {
            throw std::runtime_error(
                "array_info passed to Cpp code have different dtype");
        }
    }
}

/** Getting the expression of a string of characters as a T value
 *
 * The template parameter is T.
 * @param dtype the bodo data type.
 * @param ptr the value of the pointer passed in argument
 * @return the value as a T value.
 */
template <typename T>
inline T GetTentry(char* ptr) {
    T* ptr_T = (T*)ptr;
    return *ptr_T;
}

/** Getting the expression of a string of characters as a double value
 *
 * @param dtype the bodo data type.
 * @param ptr the value of the pointer passed in argument
 * @return the value as a double.
 */
inline double GetDoubleEntry(Bodo_CTypes::CTypeEnum dtype, char* ptr) {
    if (dtype == Bodo_CTypes::INT8)
        return double(GetTentry<int8_t>(ptr));
    if (dtype == Bodo_CTypes::UINT8)
        return double(GetTentry<uint8_t>(ptr));
    if (dtype == Bodo_CTypes::INT16)
        return double(GetTentry<int16_t>(ptr));
    if (dtype == Bodo_CTypes::UINT16)
        return double(GetTentry<uint16_t>(ptr));
    if (dtype == Bodo_CTypes::INT32)
        return double(GetTentry<int32_t>(ptr));
    if (dtype == Bodo_CTypes::UINT32)
        return double(GetTentry<uint32_t>(ptr));
    if (dtype == Bodo_CTypes::INT64)
        return double(GetTentry<int64_t>(ptr));
    if (dtype == Bodo_CTypes::UINT64)
        return double(GetTentry<uint64_t>(ptr));
    if (dtype == Bodo_CTypes::FLOAT32)
        return double(GetTentry<float>(ptr));
    if (dtype == Bodo_CTypes::FLOAT64)
        return double(GetTentry<double>(ptr));
    if (dtype == Bodo_CTypes::DATE)
        return double(GetTentry<int32_t>(ptr));
    if (dtype == Bodo_CTypes::DATETIME)
        return double(GetTentry<int64_t>(ptr));
    if (dtype == Bodo_CTypes::TIMEDELTA)
        return double(GetTentry<int64_t>(ptr));
    if (dtype == Bodo_CTypes::DECIMAL)
        return decimal_to_double(GetTentry<__int128>(ptr));
    throw std::runtime_error(
        "_array_utils.h::GetDoubleEntry: Unsupported case in GetDoubleEntry");
}

/** This function uses the combinatorial information computed in the
 * "ListPairWrite" array. The other arguments shift1, shift2 and ChoiceColumn
 * are for the choice of column
 * ---For inserting a left data, ChoiceColumn = 0 indicates retrieving column
 * from the left.
 * ---For inserting a right data, ChoiceColumn = 1 indicates retrieving column
 * from the right.
 * ---For inserting a key, we need to access both to left and right columns.
 *    This corresponds to the columns shift1 and shift2.
 *
 * The code considers all the cases in turn and creates the new array from it.
 *
 * The keys in output re used twice: In the left and on the right and so they
 * are outputed twice.
 *
 * No error is thrown but input is assumed to be coherent.
 *
 * @param arr1: the first column
 * @param arr2: the second column
 * @param short_write_idxs is the span of indices in the short table
 * @param long_write_idxs is the span of indices in the long table
 * @return one column of the table output.
 */
std::shared_ptr<array_info> RetrieveArray_TwoColumns(
    std::shared_ptr<array_info> const& arr1,
    std::shared_ptr<array_info> const& arr2,
    const std::span<const int64_t> short_write_idxs,
    const std::span<const int64_t> long_write_idxs);

/** This function returns the column with the rows with the rows given in
 * "ListIdx"
 *
 * @param std::shared_ptr<array_info> : the input pointer
 * @param ListIdx is the vector of list of rows selected
 * @param use_nullable_arr use nullable int/float output for Numpy array input
 * @param pool Memory pool to use for allocations during the execution of this
 * function.
 * @param mm Memory manager associated with the pool.
 * @return one array
 */
std::shared_ptr<array_info> RetrieveArray_SingleColumn(
    std::shared_ptr<array_info> in_arr, const std::span<const int64_t> ListIdx,
    bool use_nullable_arr = false,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

/** This function uses the combinatorial information computed in the
 * "ListIdx" array and return the column with the selected rows.
 *
 * @param in_arr : the input column
 * @param idx_arr : the index column
 * @param use_nullable_arr use nullable int/float output for Numpy array input
 * @param pool Memory pool to use for allocations during the execution of this
 * function.
 * @param mm Memory manager associated with the pool.
 * @return one array
 */
std::shared_ptr<array_info> RetrieveArray_SingleColumn_arr(
    std::shared_ptr<array_info> in_arr, std::shared_ptr<array_info> idx_arr,
    bool use_nullable_arr = false,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

/** This function takes a table, a list of rows and returns the rows obtained
 * by selecting the rows.
 *
 * @param in_table     : the input table.
 * @param ListIdx      : is the vector of indices to be selected.
 * @param n_cols_arg   : The number of columns selected. If -1 then all are
 * selected
 * @param use_nullable_arr use nullable int/float output for Numpy array input
 * @param pool Memory pool to use for allocations during the execution of this
 * function.
 * @param mm Memory manager associated with the pool.
 * @return the table output.
 */
std::shared_ptr<table_info> RetrieveTable(
    std::shared_ptr<table_info> const in_table,
    const std::span<const int64_t> ListIdx, int const& n_cols_arg = -1,
    const bool use_nullable_arr = false,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

/**
 * @brief select rows and columns in input table specified by list of indices
 * and return a new table.
 *
 * @param in_table input table to select from
 * @param rowInds list of row indices to select
 * @param colInds list of column indices to select
 * @param use_nullable_arr use nullable int/float output for Numpy array input
 * @param pool Memory pool to use for allocations during the execution of this
 * function.
 * @param mm Memory manager associated with the pool.
 * @return std::shared_ptr<table_info> output table with selected rows/columns
 */
std::shared_ptr<table_info> RetrieveTable(
    std::shared_ptr<table_info> const in_table,
    const std::span<const int64_t> rowInds,
    std::vector<uint64_t> const& colInds, const bool use_nullable_arr = false,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());

/** This code test if two keys are equal (Before that the hash should have been
 * used) It is used that way because we assume that the left key have the same
 * type as the right keys. The computation is for just one column and it is used
 * keys or the right keys. Equality means that the column/rows are the same.
 *
 * @param arr1 the first column for the comparison
 * @param iRow1 the row of the first key
 * @param arr2 the second column for the comparison
 * @param iRow2 the row of the second key
 * @param is_na_equal should na values be considered equal
 * @return True if they are equal and false otherwise.
 */
bool TestEqualColumn(const std::shared_ptr<array_info>& arr1, int64_t pos1,
                     const std::shared_ptr<array_info>& arr2, int64_t pos2,
                     bool is_na_equal);

/* @brief Get the sort indices of a slice of an array
 * @param arr Array to sort
 * @param pos_s Start position of the slice
 * @param pos_e End position of the slice
 * @return Array of sort indices
 */
std::shared_ptr<arrow::ArrayData> get_sort_indices_of_slice_arrow(
    std::shared_ptr<arrow::Array> const& arr, int64_t pos_s, int64_t pos_e);

/* This function test if two rows of two arrow columns (which may or may not be
 * the same) are equal, greater or lower than the other.
 *
 * @param arr1            : the first arrow array
 * @param pos1_s, pos1_e  : the starting and ending positions
 * @param arr2            : the second arrow array
 * @param pos2_s, pos2_e  : the starting and ending positions
 * @param na_position_bis : Whether the missing data is first or last
 * @param is_na_equal      : should na values be considered equal
 * @return 1 is arr1[pos1_s:pos1_e] < arr2[pos2_s:pos2_e], 0 is equality, -1 if
 * >.
 */
int ComparisonArrowColumn(std::shared_ptr<arrow::Array> const& arr1,
                          int64_t pos1_s, int64_t pos1_e,
                          std::shared_ptr<arrow::Array> const& arr2,
                          int64_t pos2_s, int64_t pos2_e,
                          bool const& na_position_bis, bool const& is_na_equal);

/** This code test if two keys are equal (Before that the hash should have been
 * used) It is used that way because we assume that the left key have the same
 * type as the right keys. The shift is used to precise whether we use the left
 * keys or the right keys. Equality means that all the columns are the same.
 * Thus the test iterates over the columns and if one is different then result
 * is false. We consider all types of bodo_array_type
 *
 * @param columns the vector of columns
 * @param n_key the number of keys considered for the comparison
 * @param shift_key1 the column shift for the first key
 * @param iRow1 the row of the first key
 * @param shift_key2 the column for the second key
 * @param iRow2 the row of the second key
 * @return True if they are equal and false otherwise.
 */
inline bool TestEqual(std::vector<std::shared_ptr<array_info>> const& columns,
                      size_t const& n_key, size_t const& shift_key1,
                      size_t const& iRow1, size_t const& shift_key2,
                      size_t const& iRow2) {
    // iteration over the list of key for the comparison.
    for (size_t iKey = 0; iKey < n_key; iKey++) {
        bool test = TestEqualColumn(columns[shift_key1 + iKey], iRow1,
                                    columns[shift_key2 + iKey], iRow2, true);
        if (!test) {
            return false;
        }
    }
    // If all keys are equal then we are ok and the keys are equals.
    return true;
};

/** This code test if two keys are equal (Before that the hash should have been
 * used) It is used that way because we assume that the left key have the same
 * type as the right keys. Equality means that all the columns are the same.
 * Thus the test iterates over the columns and if one is different then result
 * is false. We consider all types of bodo_array_type
 *
 * @param table1: the first table
 * @param table2: the second table
 * @param iRow1 the row of the first key
 * @param iRow2 the row of the second key
 * @param n_key the number of keys considered for the comparison
 * @param is_na_equal Are NA values considered equal
 * @return True if they are equal and false otherwise.
 */
inline bool TestEqualJoin(const std::shared_ptr<const table_info>& table1,
                          const std::shared_ptr<const table_info>& table2,
                          size_t const& iRow1, size_t const& iRow2,
                          size_t const& n_key, bool is_na_equal) {
    // iteration over the list of key for the comparison.
    for (size_t iKey = 0; iKey < n_key; iKey++) {
        bool test = TestEqualColumn(table1->columns[iKey], iRow1,
                                    table2->columns[iKey], iRow2, is_na_equal);
        if (!test) {
            return false;
        }
    }
    // If all keys are equal then we are ok and the keys are equals.
    return true;
};

template <typename T, int dtype>
    requires std::integral<T>
inline bool isnan_categorical(T const& val) {
    T miss_idx = -1;
    return val == miss_idx;
}

template <typename T, int dtype>
inline bool isnan_categorical(T const& val) {
    return false;
}

template <typename T, int dtype>
    requires std::integral<T>
inline void set_na_if_num_categories(T& val, int64_t num_categories) {
    if (val == T(num_categories)) {
        val = T(-1);
    }
}

template <typename T, int dtype>
inline void set_na_if_num_categories(T& val, int64_t num_categories) {
    return;
}

/** This function is used to determine if the value in a Categorical pointer
 * (pointer to a single value in a CategoricalArrayType) isnan.
 * @param the data type for the codes.
 * @param the Categorical Pointer
 * @returns if the value stored at the ptr is nan
 */
inline bool isnan_categorical_ptr(int dtype, char* ptr) {
    switch (dtype) {
        case Bodo_CTypes::INT8:
            return isnan_categorical<int8_t, Bodo_CTypes::INT8>(
                *((const int8_t*)ptr));
        case Bodo_CTypes::INT16:
            return isnan_categorical<int16_t, Bodo_CTypes::INT16>(
                *((const int16_t*)ptr));
        case Bodo_CTypes::INT32:
            return isnan_categorical<int32_t, Bodo_CTypes::INT32>(
                *((const int32_t*)ptr));
        case Bodo_CTypes::INT64:
            return isnan_categorical<int64_t, Bodo_CTypes::INT64>(
                *((const int64_t*)ptr));

        default:
            throw std::runtime_error(
                "_array_utils.h::NumericComparison: Invalid dtype put on "
                "CategoricalArrayType.");
    }
}

template <typename T, Bodo_CTypes::CTypeEnum DType>
    requires std::floating_point<T>
constexpr inline bool isnan_alltype(T const& val) {
    return isnan(val);
}

template <typename T, Bodo_CTypes::CTypeEnum DType>
    requires datetime_timedelta<DType>
constexpr inline bool isnan_alltype(T const& val) {
    return val == nan_val<T, DType>();
}

template <typename T, Bodo_CTypes::CTypeEnum DType>
constexpr inline bool isnan_alltype(T const& val) {
    return false;
}

/**
 * The comparison function for integer types.
 *
 * @param ptr1: char* pointer to the first value
 * @param ptr2: char* pointer to the second value
 * @param na_position: true for NaN being last, false for NaN being first (not
 * used)
 * @return 1 if *ptr1 < *ptr2
 */
template <typename T>
int NumericComparison_int(char* ptr1, char* ptr2, bool const& na_position) {
    T* ptr1_T = (T*)ptr1;
    T* ptr2_T = (T*)ptr2;
    if (*ptr1_T > *ptr2_T) {
        return -1;
    } else if (*ptr1_T < *ptr2_T) {
        return 1;
    } else {
        return 0;
    }
}

/**
 * The comparison function for decimal types.
 *
 * @param ptr1: char* pointer to the first value
 * @param ptr2: char* pointer to the second value
 * @param na_position: true for NaN being last, false for NaN being first (not
 * used)
 * @return 1 if *ptr1 < *ptr2
 */
inline int NumericComparison_decimal(char* ptr1, char* ptr2,
                                     bool const& na_position) {
    __int128* ptr1_dec = (__int128*)ptr1;
    __int128* ptr2_dec = (__int128*)ptr2;
    double value1 = decimal_to_double(*ptr1_dec);
    double value2 = decimal_to_double(*ptr2_dec);
    if (value1 > value2) {
        return -1;
    } else if (value1 < value2) {
        return 1;
    } else {
        return 0;
    }
}

/**
 * The comparison function for floating points.
 * If na_position = True then the NaN are considered larger than any other.
 *
 * @param ptr1: char* pointer to the first value
 * @param ptr2: char* pointer to the second value
 * @param na_position: true for NaN being larger, false for NaN being smallest
 * @return 1 if *ptr1 < *ptr2
 */
template <typename T>
int NumericComparison_float(char* ptr1, char* ptr2, bool const& na_position) {
    T* ptr1_T = (T*)ptr1;
    T* ptr2_T = (T*)ptr2;
    T val1 = *ptr1_T;
    T val2 = *ptr2_T;
    if (isnan(val1) && isnan(val2)) {
        return 0;
    } else if (isnan(val2)) {
        if (na_position) {
            return 1;
        } else {
            return -1;
        }
    } else if (isnan(val1)) {
        if (na_position) {
            return -1;
        } else {
            return 1;
        }
    } else if (val1 > val2) {
        return -1;
    } else if (val1 < val2) {
        return 1;
    } else {
        return 0;
    }
}

/**
 * The comparison function for dates
 * If na_position = True then the NaN are considered larger than any other.
 * The minimum values are considered to be missing values.
 *
 * @param ptr1: char* pointer to the first value
 * @param ptr2: char* pointer to the second value
 * @param na_position: true for NaN being larger, false for NaN being smallest
 * @return 1 if *ptr1 < *ptr2
 */
template <typename T>
int NumericComparison_date(char* ptr1, char* ptr2, bool const& na_position) {
    T* ptr1_T = (T*)ptr1;
    T* ptr2_T = (T*)ptr2;
    T val1 = *ptr1_T;
    T val2 = *ptr2_T;
    auto isnan_date = [&](T const& val) {
        return val == std::numeric_limits<T>::min();
    };
    if (isnan_date(val1) && isnan_date(val2)) {
        return 0;
    } else if (isnan_date(val2)) {
        if (na_position) {
            return 1;
        } else {
            return -1;
        }
    } else if (isnan_date(val1)) {
        if (na_position) {
            return -1;
        } else {
            return 1;
        }
    } else if (val1 > val2) {
        return -1;
    } else if (val1 < val2) {
        return 1;
    } else {
        return 0;
    }
}

/**
 * The comparison function for integer/floating point
 * If na_position = True then the NaN are considered larger than any other.
 *
 * @param ptr1: char* pointer to the first value
 * @param ptr2: char* pointer to the second value
 * @param na_position: true for NaN being last, false for NaN being first
 * @return 1 if *ptr1 < *ptr2, 0 if equal and -1 if >
 */
inline int NumericComparison(Bodo_CTypes::CTypeEnum const& dtype, char* ptr1,
                             char* ptr2, bool const& na_position) {
    if (dtype == Bodo_CTypes::_BOOL)
        return NumericComparison_int<bool>(ptr1, ptr2, na_position);
    if (dtype == Bodo_CTypes::INT8)
        return NumericComparison_int<int8_t>(ptr1, ptr2, na_position);
    if (dtype == Bodo_CTypes::UINT8)
        return NumericComparison_int<uint8_t>(ptr1, ptr2, na_position);
    if (dtype == Bodo_CTypes::INT16)
        return NumericComparison_int<int16_t>(ptr1, ptr2, na_position);
    if (dtype == Bodo_CTypes::UINT16)
        return NumericComparison_int<uint16_t>(ptr1, ptr2, na_position);
    if (dtype == Bodo_CTypes::INT32 || dtype == Bodo_CTypes::DATE)
        return NumericComparison_int<int32_t>(ptr1, ptr2, na_position);
    if (dtype == Bodo_CTypes::UINT32)
        return NumericComparison_int<uint32_t>(ptr1, ptr2, na_position);
    // for DATE/TIME, the missing value is done via NULLABLE_INT_BOOL
    // TODO: [BE-4106] Split Time into Time32 and Time64
    if (dtype == Bodo_CTypes::INT64 || dtype == Bodo_CTypes::TIME)
        return NumericComparison_int<int64_t>(ptr1, ptr2, na_position);
    if (dtype == Bodo_CTypes::UINT64)
        return NumericComparison_int<uint64_t>(ptr1, ptr2, na_position);
    // For DATETIME/TIMESTAMPTZ/TIMEDELTA the NA is done via the
    // std::numeric_limits<int64_t>::min()
    if (dtype == Bodo_CTypes::DATETIME || dtype == Bodo_CTypes::TIMEDELTA ||
        dtype == Bodo_CTypes::TIMESTAMPTZ)
        return NumericComparison_date<int64_t>(ptr1, ptr2, na_position);
    if (dtype == Bodo_CTypes::FLOAT32)
        return NumericComparison_float<float>(ptr1, ptr2, na_position);
    if (dtype == Bodo_CTypes::FLOAT64)
        return NumericComparison_float<double>(ptr1, ptr2, na_position);
    if (dtype == Bodo_CTypes::DECIMAL)
        return NumericComparison_decimal(ptr1, ptr2, na_position);
    throw std::runtime_error(
        "_array_utils.h::NumericComparison: Invalid dtype put on input to "
        "NumericComparison.");
}

/** This code test keys if two keys are greater or equal
 * The code is done so as to give identical results to the Python comparison.
 * It is used that way because we assume that the left key have the same type as
 * the right keys. The shift is used to precise whether we use the left keys or
 * the right keys. 0 means that the columns are equals and 1,-1 that the keys
 * are different. Thus the test iterates over the columns and if one is
 * different then we can conclude. We consider all types of bodo_array_type.
 *
 * @param n_key the number of keys considered for the comparison
 * @param vect_ascending the vector of ascending values for the comparison
 * @param columns1 the list of columns of the first table
 * @param shift_key1 the column shift for the first key
 * @param iRow1 the row of the first key
 * @param columns2 the list of columns of the second table
 * @param shift_key2 the column for the second key
 * @param iRow2 the row of the second key
 * @param na_position: the vector of null locations. if true NaN values are
 * largest, if false smallest.
 * @return true if (shift_key1,iRow1) < (shift_key2,iRow2) , false otherwise
 */
bool KeyComparisonAsPython(
    size_t const& n_key, int64_t* vect_ascending,
    std::vector<std::shared_ptr<array_info>> const& columns1,
    size_t const& shift_key1, size_t const& iRow1,
    std::vector<std::shared_ptr<array_info>> const& columns2,
    size_t const& shift_key2, size_t const& iRow2, int64_t* na_position);

int KeyComparisonAsPython_Column(bool const& na_position_bis,
                                 const std::shared_ptr<array_info>& arr1,
                                 size_t const& iRow1,
                                 const std::shared_ptr<array_info>& arr2,
                                 size_t const& iRow2);

/**
 * @brief Create a null-bitmask that is a bitwise AND of the bitmasks of the
 * specified arrays. This is equivalent to computing if _any_ column in a row is
 * a null (since it will get set to 0). This is useful in SQL joins where we
 * compute this information for the join keys to filter rows that cannot match
 * with any other rows.
 * For certain array types where a null-bitmask already exists, we will use it
 * directly, and for others like NUMPY, CATEGORICAL and ARROW, we will compute
 * one dynamically.
 *
 * @param arrays Vector of arrays whose bitmasks need to be bitwise AND-ed.
 * @param is_parallel Are the arrays distributed (for tracing purposes).
 * @return uint8_t* Output bitmask.
 */
uint8_t* bitwise_and_null_bitmasks(
    const std::vector<std::shared_ptr<array_info>>& arrays,
    const bool is_parallel);

// ----------------------- Debug functions -----------------------

/* This is a function used by "DEBUG_PrintSetOfColumn"
 * It takes a column and returns a vector of string on output
 *
 * @param os is the output stream
 * @param arr is the pointer.
 */
void DEBUG_PrintColumn(std::ostream& os, const std::shared_ptr<array_info> arr);

/** The DEBUG_PrintSetOfColumn is printing the contents of the table to
 * the output stream.
 * All cases are supported (NUMPY, STRING, NULLABLE_INT_BOOL) as well as
 * all integer and floating types.
 *
 * The number of rows in the columns do not have to be the same.
 *
 * @param the output stream (e.g. std::cerr or std::cout)
 * @param ListArr the list of columns in input
 * @return Nothing. Everything is put in the stream
 */
void DEBUG_PrintSetOfColumn(
    std::ostream& os, std::vector<std::shared_ptr<array_info>> const& ListArr);

/**
 * @brief Print the contents of a table to the output stream.
 * See DEBUG_PrintSetOfColumn for more details
 */
void DEBUG_PrintTable(std::ostream& os, table_info* table);

/**
 * @brief Print the contents of a table to the output stream.
 * See DEBUG_PrintSetOfColumn for more details
 */
void DEBUG_PrintTable(std::ostream& os,
                      const std::shared_ptr<const table_info>& table);

/**
 * @brief Prints contents of a std::unordered_map to the output stream.
 * Only the case where both the key and value type is uint64_t is supported
 * at this point.
 *
 * @param os the output stream (e.g. std::cerr or std::cout)
 * @param map the unordered_map to print
 * @return Nothing. Everything is put in the stream
 */
void DEBUG_PrintUnorderedMap(std::ostream& os,
                             std::unordered_map<uint64_t, uint64_t> map);

/** This is a function used for debugging.
 * It prints the nature of the columns of the tables
 *
 * @param the output stream (for example std::cerr or std::cout)
 * @param The list of columns in output
 * @return nothing. Everything is printed to the stream
 */
void DEBUG_PrintRefct(std::ostream& os,
                      std::vector<std::shared_ptr<array_info>> const& ListArr);

void DEBUG_append_to_out_array(std::shared_ptr<arrow::Array> input_array,
                               int64_t start_offset, int64_t end_offset,
                               std::string& string_builder);

inline bool does_keys_have_nulls(
    std::vector<std::shared_ptr<array_info>> const& key_cols) {
    for (auto key_col : key_cols) {
        if ((key_col->arr_type == bodo_array_type::NUMPY &&
             (key_col->dtype == Bodo_CTypes::FLOAT32 ||
              key_col->dtype == Bodo_CTypes::FLOAT64 ||
              key_col->dtype == Bodo_CTypes::DATETIME ||
              key_col->dtype == Bodo_CTypes::TIMEDELTA)) ||
            key_col->arr_type == bodo_array_type::STRING ||
            key_col->arr_type == bodo_array_type::CATEGORICAL ||
            key_col->arr_type == bodo_array_type::DICT ||
            key_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
            return true;
        }
    }
    return false;
}

inline bool does_row_has_nulls(
    const std::vector<std::shared_ptr<array_info>>& key_cols,
    int64_t const& i) {
    for (const auto& key_col : key_cols) {
        if (key_col->arr_type == bodo_array_type::CATEGORICAL) {
            std::vector<char> vectNaN = RetrieveNaNentry(key_col->dtype);
            size_t siztype = numpy_item_size[key_col->dtype];
            if (memcmp(key_col->data1() + i * siztype, vectNaN.data(),
                       siztype) == 0) {
                return true;
            }
        } else if (key_col->arr_type == bodo_array_type::STRING ||
                   key_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL ||
                   key_col->arr_type == bodo_array_type::DICT) {
            if (!GetBit((uint8_t*)key_col->null_bitmask(), i)) {
                return true;
            }
        } else if (key_col->arr_type == bodo_array_type::NUMPY) {
            if ((key_col->dtype == Bodo_CTypes::FLOAT32 &&
                 isnan(key_col->at<float>(i))) ||
                (key_col->dtype == Bodo_CTypes::FLOAT64 &&
                 isnan(key_col->at<double>(i))) ||
                (key_col->dtype == Bodo_CTypes::DATETIME &&
                 key_col->at<int64_t>(i) ==
                     std::numeric_limits<int64_t>::min()) ||
                (key_col->dtype == Bodo_CTypes::TIMEDELTA &&
                 key_col->at<int64_t>(i) ==
                     std::numeric_limits<int64_t>::min())) {
                return true;
            }
        }
    }
    return false;
}

/**
 * Given an array of hashes, returns estimate of number of unique hashes.
 * @param hashes: pointer to array of hashes
 * @param len: number of hashes
 * @param is_parallel: true if data is distributed (used to indicate whether
 * tracing should be parallel or not)
 */
size_t get_nunique_hashes(const std::shared_ptr</*const*/ uint32_t[]>& hashes,
                          const size_t len, bool is_parallel);

/**
 * Given an array of hashes, returns estimate of number of unique hashes
 * of the local array, and global estimate doing a reduction over all ranks.
 * @param hashes: pointer to array of hashes
 * @param len: number of hashes
 * @param is_parallel: true if data is distributed (used to indicate whether
 * tracing should be parallel or not)
 */
std::pair<size_t, size_t> get_nunique_hashes_global(
    const std::shared_ptr<uint32_t[]> hashes, const size_t len,
    bool is_parallel);

/**
 * @brief concatenate tables vertically into a single table.
 * Input tables are assumed to have the same schema, and will
 * be fully deleted (decref arrays and delete pointers).
 *
 * @param table_chunks input tables which are assumed to have the same schema
 * @return std::shared_ptr<table_info> concatenated table
 */
std::shared_ptr<table_info> concat_tables(
    const std::vector<std::shared_ptr<table_info>>& table_chunks);

/**
 * @brief Concatenate the arrays into a single array.
 *
 * For now, we're just wrapping it around concat_tables.
 * Data types for our use case are integers, floats, decimal, date and
 * datetime. For all of these, excluding Date, we re-use the Bodo buffers
 * when creating the intermediate Arrow array, so it should be fairly
 * efficient.
 *
 * XXX Switch to a different implementation later if needed for performance.
 *
 * Function assumes that all array-infos have the same array type and dtype.
 * All input arrays will be fully decref-ed and deleted (due to behavior of
 * concat_tables)
 *
 * @param arrays Vector of arrays to concatenate together.
 * @return std::shared_ptr<array_info> Concatenated array.
 */
std::shared_ptr<array_info> concat_arrays(
    std::vector<std::shared_ptr<array_info>>& arrays);

/**
 * @brief Check if the interval (arr1[idx], arr2[idx]) is within
 * the bounds for a certain rank.
 * We consider a row within the bounds of a rank if bounds[rank-1] <=
 * arr2[idx] AND arr1[idx] <= bounds[rank]. When rank == 0, we skip the
 * first check (and consider it true), since bounds[-1] is essentially
 * -infinity. When rank == n_pes - 1, we skip the second check (and consider
 * it true), since bounds[n_pes - 1] is essentially +infinity.
 *
 * @param bounds_arr Array of length n_pes - 1 with bounds for the ranks.
 * @param rank Rank to check the bound for
 * @param n_pes Total number of processes
 * @param arr1 Array for the start of the interval
 * @param arr2 Array for the end of the interval
 * @param idx Index to check
 */
inline bool within_bounds_of_rank(const std::shared_ptr<array_info>& bounds_arr,
                                  uint32_t rank, int n_pes,
                                  const std::shared_ptr<array_info>& arr1,
                                  const std::shared_ptr<array_info>& arr2,
                                  uint64_t idx) {
    // na_position_bis is true in our case since asc = true and na_last = true
    // which means that na_bis = (!na_last) ^ asc = true
    return ((rank == 0) || (KeyComparisonAsPython_Column(
                                true, bounds_arr, rank - 1, arr2, idx) >= 0)) &&
           ((rank == uint32_t(n_pes - 1)) ||
            (KeyComparisonAsPython_Column(true, arr1, idx, bounds_arr, rank) >=
             0));
}

/**
 * @brief Determine if an interval pair in a row is a bad interval.
 *
 * A bad interval is an interval [A, B] where A > B or when A == B
 * in the case when strict == true
 *
 * @param arr1 The array containing the start of the interval
 * @param arr2 The array containing the end of the interval
 * @param idx The row index of the interval in both arrays
 * @param strict Whether to consider [A, A] to be a bad interval
 * @return If the interval [arr1[idx], arr2[idx]] is bad
 */
inline bool is_bad_interval(const std::shared_ptr<array_info>& arr1,
                            const std::shared_ptr<array_info>& arr2,
                            uint64_t idx, bool strict = true) {
    auto comp = KeyComparisonAsPython_Column(true, arr1, idx, arr2, idx);
    // strict == true: comp == -1 (A > B) or comp == 0 (A == B)
    // strict == false: comp == -1 (A > B)
    return strict ? comp <= 0 : comp < 0;
}

/**
 * @brief Flatten a vector of vectors into a single vector.
 * The vectors are placed in the order that they occur in vec.
 *
 * @tparam T Data type of the vector.
 * @param vec Vector of vectors to flatten
 * @param total_size Total length of all the vectors.
 * @return std::vector<T> Flattened vector.
 */
template <typename T, typename S, typename V>
inline std::vector<T, S> flatten(std::vector<std::vector<T, S>, V> const& vec,
                                 uint64_t total_size) {
    std::vector<T, S> flattened;
    flattened.reserve(total_size);
    for (auto const& v : vec) {
        flattened.insert(flattened.end(), v.begin(), v.end());
    }
    return flattened;
}

/**
 * @brief Move columns specified in col_ids to the front of the table.
 * This is done by creating a new vector of array_info's where the
 * specified columns are put in the front, and then the rest of the columns
 * are put afterwards.
 *
 * e.g. If there are 5 columns (A, B, C, D, E; in that order)
 * and col_ids = [1,2],
 * we will re-arrange it so that the columns are [B, C, A, D, E].
 *
 * To enable putting the columns back in the correct order using
 * restore_col_order function, we return a map which maps the old index to the
 * new index.
 *
 * @param table Table to modify
 * @param col_ids Column ids to move to the front
 * @return const std::unordered_map<uint64_t, uint64_t> Map of new index to old
 * index that can be used in restore_col_order to restore the original ordering.
 */
inline const std::unordered_map<uint64_t, uint64_t> move_cols_to_front(
    std::shared_ptr<table_info> table, std::vector<uint64_t> col_ids) {
    std::vector<std::shared_ptr<array_info>> new_columns(table->columns.size());
    std::set<uint64_t> col_ids_set(col_ids.begin(), col_ids.end());
    std::unordered_map<uint64_t, uint64_t> col_mapping;
    uint64_t itr = 0;
    // Move specified ones to the front
    for (size_t i = 0; i < col_ids.size(); i++) {
        new_columns[itr] = table->columns[col_ids[i]];
        col_mapping[col_ids[i]] = itr;
        itr++;
    }
    // Add the rest
    for (size_t i = 0; i < table->columns.size(); i++) {
        if (!col_ids_set.contains(i)) {  // If not already added
            new_columns[itr] = table->columns[i];
            col_mapping[i] = itr;
            itr++;
        }
    }
    table->columns.clear();
    table->columns = new_columns;
    // TODO Verify that it's returned without copy
    return col_mapping;
}

/**
 * @brief Restore the original column ordering using the map returned by
 * move_cols_to_front.
 *
 * @param table Table to restore the order of.
 * @param col_mapping Map of old column id to new column id. This is the output
 * of move_cols_to_front.
 */
inline void restore_col_order(
    std::shared_ptr<table_info> table,
    std::unordered_map<uint64_t, uint64_t>& col_mapping) {
    std::vector<std::shared_ptr<array_info>> new_columns(table->columns.size());
    for (auto it = col_mapping.begin(); it != col_mapping.end(); it++) {
        new_columns[it->first] = table->columns[it->second];
    }
    table->columns.clear();
    table->columns = new_columns;
}

/**
 * @brief Convert the given value to a double.
 *
 * @tparam T The input type.
 * @tparam dtype The underlying dtype of the input type.
 * @param val The value to convert
 * @return Val as a double.
 */
template <typename T, Bodo_CTypes::CTypeEnum DType>
    requires decimal<DType>
inline double to_double(T const& val) {
    return decimal_to_double(val);
}

template <typename T, Bodo_CTypes::CTypeEnum DType>
inline double to_double(T const& val) {
    return static_cast<double>(val);
}

// Several concepts and utilities used for arr_type/dtype agnostic array
// interactions.

template <bodo_array_type::arr_type_enum ArrType>
concept numpy_array = ArrType == bodo_array_type::arr_type_enum::NUMPY;

template <bodo_array_type::arr_type_enum ArrType>
concept nullable_array =
    ArrType == bodo_array_type::arr_type_enum::NULLABLE_INT_BOOL;

template <bodo_array_type::arr_type_enum ArrType>
concept string_array = ArrType == bodo_array_type::arr_type_enum::STRING;

template <bodo_array_type::arr_type_enum ArrType>
concept dict_array = ArrType == bodo_array_type::arr_type_enum::DICT;

template <bodo_array_type::arr_type_enum ArrType>
concept string_or_dict = string_array<ArrType> || dict_array<ArrType>;

template <bodo_array_type::arr_type_enum ArrType>
concept array_item_array =
    ArrType == bodo_array_type::arr_type_enum::ARRAY_ITEM;

/**
 * @brief Retrieves an item from an array.
 *
 * @param[in] arr - The array to extract the value from.
 * @param[in] idx - Index of item to extract.
 *
 * @return The item at the given index.
 */
template <bodo_array_type::arr_type_enum ArrType, typename T,
          Bodo_CTypes::CTypeEnum DType>
inline T get_arr_item(array_info& arr, int64_t idx) {
    return ((T*)arr.data1())[idx];
}

/**
 * @brief Retrieves an item from an array.
 *
 * Used for nullable arrays of booleans.
 *
 * @param[in] arr - The array to extract the value from.
 * @param[in] idx - Index of item to extract.
 *
 * @return The item at the given index.
 */
template <bodo_array_type::arr_type_enum ArrType, typename T,
          Bodo_CTypes::CTypeEnum DType>
    requires(nullable_array<ArrType> && bool_dtype<DType>)
inline T get_arr_item(array_info& arr, int64_t idx) {
    return GetBit((uint8_t*)arr.data1(), idx);
}

/**
 * @brief Retrieves an item from an array, as a std::string_view.
 *
 * Used for string arrays.
 *
 * @param[in] arr - The array to extract the value from.
 * @param[in] idx - Index of item to extract.
 *
 * @return The item at the given index, as a string view.
 * Note: this return is essentially a dangling pointer, which is
 * invalid once the original array is de allocated.
 */
template <bodo_array_type::arr_type_enum ArrType, typename T,
          Bodo_CTypes::CTypeEnum DType>
    requires(string_array<ArrType>)
inline std::string_view get_arr_item_str(array_info& arr, int64_t idx) {
    char* data = arr.data1();
    offset_t* offsets = (offset_t*)arr.data2();
    offset_t start_offset = offsets[idx];
    offset_t end_offset = offsets[idx + 1];
    offset_t len = end_offset - start_offset;
    std::string_view substr(&data[start_offset], len);
    return substr;
}

/**
 * @brief Retrieves an item from an array, as a std::string_view.
 *
 * Used for dictionary encoded arrays.
 *
 * @param[in] arr - The array to extract the value from.
 * @param[in] idx - Index of item to extract.
 *
 * @return The item at the given index, as a string view.
 * Note: this return is essentially a dangling pointer, which is
 * invalid once the original array is de allocated.
 */
template <bodo_array_type::arr_type_enum ArrType, typename T,
          Bodo_CTypes::CTypeEnum DType>
    requires(dict_array<ArrType>)
inline std::string_view get_arr_item_str(array_info& arr, int64_t idx) {
    std::shared_ptr<array_info> indices = arr.child_arrays[1];
    int64_t dict_idx = get_arr_item<bodo_array_type::NULLABLE_INT_BOOL, int64_t,
                                    Bodo_CTypes::INT64>(*indices, idx);
    char* data = arr.child_arrays[0]->data1();
    offset_t* offsets = (offset_t*)arr.child_arrays[0]->data2();
    offset_t start_offset = offsets[dict_idx];
    offset_t end_offset = offsets[dict_idx + 1];
    offset_t len = end_offset - start_offset;
    std::string_view substr(&data[start_offset], len);
    return substr;
}

/**
 * Check if an element of an array is non-null.
 *
 * @param[in] arr: Reference to array to check for nulls.
 * @param[in] idx: Index of element to check for nulls.
 *
 * @return True if the array at the requested index is non-null.
 */
template <bodo_array_type::arr_type_enum ArrType, typename T,
          Bodo_CTypes::CTypeEnum DType>
inline bool non_null_at(array_info& arr, size_t idx) {
    return arr.get_null_bit(idx);
}

/**
 * Check if element of an array is non-null
 *
 * Used for general numpy arrays, which never have nulls.
 *
 * @param[in] arr: Reference to array to check for nulls.
 * @param[in] idx: Index of element to check for nulls.
 *
 * @return True always, since numpy arrays without sentinels never have nulls
 */
template <bodo_array_type::arr_type_enum ArrType, typename T,
          Bodo_CTypes::CTypeEnum DType>
    requires numpy_array<ArrType>
inline bool non_null_at(array_info& arr, size_t idx) {
    return true;
}

/**
 * Check if element of an array is non-null
 *
 * Used for numpy arrays of datetime or timedelta, which have sentinels for NaT.
 *
 * This (and other helpers) needs to be changed when nullable datetime/timedelta
 * arrays are implemented.
 *
 * @param[in] arr: Reference to array to check for nulls.
 * @param[in] idx: Index of element to check for nulls.
 *
 * @return True if the array at the requested index contains a sentinel value
 * not corresponding to null
 */
template <bodo_array_type::arr_type_enum ArrType, typename T,
          Bodo_CTypes::CTypeEnum DType>
    requires(numpy_array<ArrType> && NullSentinelDtype<DType>)
inline bool non_null_at(array_info& arr, size_t idx) {
    return !isnan_alltype<T, DType>(get_arr_item<ArrType, T, DType>(arr, idx));
}

/**
 * Check if an element of an array is null.
 *
 * @param[in] arr: Reference to array to check for nulls.
 * @param[in] idx: Index of element to check for nulls.
 *
 * @return True if the array at the requested index is null.
 */
template <bodo_array_type::arr_type_enum ArrType, typename T,
          Bodo_CTypes::CTypeEnum DType>
inline bool is_null_at(array_info& arr, size_t idx) {
    return !arr.get_null_bit(idx);
}

/**
 * Check if element of an array is null
 *
 * Used for general numpy arrays, which never have nulls.
 *
 * @param[in] arr: Reference to array to check for nulls.
 * @param[in] idx: Index of element to check for nulls.
 *
 * @return False always, since numpy arrays without sentinels never have nulls
 */
template <bodo_array_type::arr_type_enum ArrType, typename T,
          Bodo_CTypes::CTypeEnum DType>
    requires numpy_array<ArrType>
inline bool is_null_at(array_info& arr, size_t idx) {
    return false;
}

/**
 * Check if element of an array is null
 *
 * Used for numpy arrays of datetime or timedelta, which have sentinels for NaT.
 *
 * This (and other helpers) needs to be changed when nullable datetime/timedelta
 * arrays are implemented.
 *
 * @param[in] arr: Reference to array to check for nulls.
 * @param[in] idx: Index of element to check for nulls.
 *
 * @return True if the array at the requested index contains a sentinel value
 * corresponding to null
 */
template <bodo_array_type::arr_type_enum ArrType, typename T,
          Bodo_CTypes::CTypeEnum DType>
    requires(numpy_array<ArrType> && NullSentinelDtype<DType>)
inline bool is_null_at(array_info& arr, size_t idx) {
    return isnan_alltype<T, DType>(get_arr_item<ArrType, T, DType>(arr, idx));
}

/**
 * Sets the null bit at the given index to true, indicating that the array
 * at the specified location is non-null. Used for nullable arrays.
 *
 * @param[in,out] arr: The array to operate on.
 * @param[in] idx: Index being set to non-null.
 */
template <bodo_array_type::arr_type_enum ArrType, typename T,
          Bodo_CTypes::CTypeEnum DType>
inline void set_non_null(array_info& arr, size_t idx) {
    arr.set_null_bit(idx, true);
}

/**
 * Sets the null bit at the given index to true, indicating that the array
 * at the specified location is non-null. Used for numpy arrays.
 *
 * @param[in,out] arr: The array to operate on.
 * @param[in] idx: Index being set to non-null.
 *
 * Note: this function does nothing because there is no null bitmask to alter
 * in a numpy array.
 */
template <bodo_array_type::arr_type_enum ArrType, typename T,
          Bodo_CTypes::CTypeEnum DType>
    requires numpy_array<ArrType>
inline void set_non_null(array_info& arr, size_t idx) {}

/**
 * Sets the null bit at the given index to false, indicating that the array
 * at the specified location is null. Used for nullable arrays.
 *
 * @param[in,out] arr: The array to operate on.
 * @param[in] idx: Index being set to non-null.
 */
template <bodo_array_type::arr_type_enum ArrType, typename T,
          Bodo_CTypes::CTypeEnum DType>
inline void set_to_null(array_info& arr, size_t idx) {
    arr.set_null_bit(idx, false);
}

/**
 * Sets the null bit at the given index to false, indicating that the array
 * at the specified location is null. Used for numpy arrays.
 *
 * @param[in,out] arr: The array to operate on.
 * @param[in] idx: Index being set to non-null.
 *
 * Note: this function does nothing because there is no null bitmask to alter
 * in a numpy array.
 */
template <bodo_array_type::arr_type_enum ArrType, typename T,
          Bodo_CTypes::CTypeEnum DType>
    requires numpy_array<ArrType>
inline void set_to_null(array_info& arr, size_t idx) {}

/**
 * Sets an item in an array. Used for nullable arrays (except for booleans),
 * or numpy arrays.
 *
 * @param[in,out] arr - The array to set the item in.
 * @param[in] idx - The index of the item to set.
 * @param[in] val - The value to set the item to.
 */
template <bodo_array_type::arr_type_enum ArrType, typename T,
          Bodo_CTypes::CTypeEnum DType>
inline void set_arr_item(array_info& arr, size_t idx, T val) {
    ((T*)arr.data1())[idx] = val;
}

/**
 * Sets an item in an array. Used for nullable arrays of booleans
 *
 * @param[in,out] arr - The array to set the item in.
 * @param[in] idx - The index of the item to set.
 * @param[in] val - The value to set the item to.
 */
template <bodo_array_type::arr_type_enum ArrType, typename T,
          Bodo_CTypes::CTypeEnum DType>
    requires(nullable_array<ArrType> && bool_dtype<DType>)
inline void set_arr_item(array_info& arr, size_t idx, T val) {
    SetBitTo((uint8_t*)arr.data1(), idx, val);
}
