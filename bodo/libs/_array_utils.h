// Copyright (C) 2019 Bodo Inc. All rights reserved.
#ifndef _ARRAY_UTILS_H_INCLUDED
#define _ARRAY_UTILS_H_INCLUDED

#include "_bodo_common.h"
#include "_decimal_ext.h"
#include "hyperloglog.hpp"

// ------------------------------------------------
// Always include robin and hopscotch maps because they are used regardless of
// default hash map implementation
#include <include/tsl/hopscotch_map.h>
#include <include/tsl/robin_map.h>

// Choose default implementation for unordered map and set
#undef USE_STD
#undef USE_TSL_ROBIN
#undef USE_TSL_SPARSE
#undef USE_TSL_HOPSCOTCH
#define USE_ROBIN_HOOD_FLAT
#undef USE_ROBIN_HOOD_NODE

#ifdef USE_STD
#include <unordered_map>
#include <unordered_set>
#define UNORD_MAP_CONTAINER std::unordered_map
#define UNORD_SET_CONTAINER std::unordered_set
#endif

#ifdef USE_TSL_ROBIN
// The robin_map can store hashes internally, which helps improve performance.
// To enable this, search for UNORD_MAP_CONTAINER and add a `true` template
// parameter and allocation. E.g.
//
// UNORD_MAP_CONTAINER<size_t, size_t, HashHashJoinTable,
//                     KeyEqualHashJoinTable,
//                     std::allocator<std::pair<size_t, size_t>>,
//                     true>  // StoreHash
#define UNORDERED_MAP_MAX_LOAD_FACTOR 0.5
#include <include/tsl/robin_set.h>
#define UNORD_MAP_CONTAINER tsl::robin_map
#define UNORD_SET_CONTAINER tsl::robin_set
#endif

#ifdef USE_TSL_SPARSE
#define UNORDERED_MAP_MAX_LOAD_FACTOR 0.9
#include <include/tsl/sparse_map.h>
#include <include/tsl/sparse_set.h>
#define UNORD_MAP_CONTAINER tsl::sparse_map
#define UNORD_SET_CONTAINER tsl::sparse_set
#endif

#ifdef USE_TSL_HOPSCOTCH
#include <include/tsl/hopscotch_set.h>
#define UNORD_MAP_CONTAINER tsl::hopscotch_map
#define UNORD_SET_CONTAINER tsl::hopscotch_set
#endif

#ifdef USE_ROBIN_HOOD_FLAT
#define UNORDERED_MAP_MAX_LOAD_FACTOR 0.8
#include "robin_hood.h"
#define UNORD_MAP_CONTAINER robin_hood::unordered_flat_map
#define UNORD_SET_CONTAINER robin_hood::unordered_flat_set
#endif

#ifdef USE_ROBIN_HOOD_NODE
#define UNORDERED_MAP_MAX_LOAD_FACTOR 0.8
#include "robin_hood.h"
#define UNORD_MAP_CONTAINER robin_hood::unordered_node_map
#define UNORD_SET_CONTAINER robin_hood::unordered_node_set
#endif
// ------------------------------------------------

inline void CheckEqualityArrayType(array_info* arr1, array_info* arr2) {
    if (arr1->arr_type != arr2->arr_type) {
        Bodo_PyErr_SetString(
            PyExc_RuntimeError,
            "array_info passed to Cpp code have different arr_type");
        return;
    }
    if (arr1->arr_type != bodo_array_type::STRING) {
        if (arr1->dtype != arr2->dtype) {
            Bodo_PyErr_SetString(
                PyExc_RuntimeError,
                "array_info passed to Cpp code have different dtype");
            return;
        }
    }
}

/** Getting the expression of a string of characters as a T value
 *
 * The template paramter is T.
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
    if (dtype == Bodo_CTypes::INT8) return double(GetTentry<int8_t>(ptr));
    if (dtype == Bodo_CTypes::UINT8) return double(GetTentry<uint8_t>(ptr));
    if (dtype == Bodo_CTypes::INT16) return double(GetTentry<int16_t>(ptr));
    if (dtype == Bodo_CTypes::UINT16) return double(GetTentry<uint16_t>(ptr));
    if (dtype == Bodo_CTypes::INT32) return double(GetTentry<int32_t>(ptr));
    if (dtype == Bodo_CTypes::UINT32) return double(GetTentry<uint32_t>(ptr));
    if (dtype == Bodo_CTypes::INT64) return double(GetTentry<int64_t>(ptr));
    if (dtype == Bodo_CTypes::UINT64) return double(GetTentry<uint64_t>(ptr));
    if (dtype == Bodo_CTypes::FLOAT32) return double(GetTentry<float>(ptr));
    if (dtype == Bodo_CTypes::FLOAT64) return double(GetTentry<double>(ptr));
    if (dtype == Bodo_CTypes::DATE) return double(GetTentry<int64_t>(ptr));
    if (dtype == Bodo_CTypes::DATETIME) return double(GetTentry<int64_t>(ptr));
    if (dtype == Bodo_CTypes::TIMEDELTA) return double(GetTentry<int64_t>(ptr));
    if (dtype == Bodo_CTypes::DECIMAL)
        return decimal_to_double(GetTentry<decimal_value_cpp>(ptr));
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
 * @param ListPairWrite is the vector of list of pairs for the writing of the
 * output table
 * @param ChoiceColumn is the chosen option
 * @param map_integer_type is the choice of mapping the integer type from
 *    NUMPY to NULLABLE_INT_ARRAY if not available.
 * @return one column of the table output.
 */
array_info* RetrieveArray_TwoColumns(
    array_info* const& arr1, array_info* const& arr2,
    std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> const& ListPairWrite,
    int const& ChoiceColumn, bool const& map_integer_type);

/** This function returns the column with the rows with the rows given in
 * "ListIdx"
 *
 * @param array_info* : the input pointer
 * @param ListIdx is the vector of list of rows selected
 * @return one array
 */
array_info* RetrieveArray_SingleColumn(array_info* in_arr,
                                       std::vector<int64_t> const& ListIdx);

/** This function uses the combinatorial information computed in the
 * "ListIdx" array and return the coulm with the selecetd rows.
 *
 * @param in_arr : the input column
 * @param idx_arr : the index column
 * @return one array
 */
array_info* RetrieveArray_SingleColumn_arr(array_info* in_arr,
                                           array_info* idx_arr);

/** This function takes a table, a list of rows and returns the rows obtained
 * by selecting the rows.
 *
 * @param in_table     : the input table.
 * @param ListIdx      : is the vector of indices to be selected.
 * @param n_col        : The number of columns selected. If -1 then all are
 * selected
 * @return the table output.
 */
table_info* RetrieveTable(table_info* const& in_table,
                          std::vector<int64_t> const& ListIdx,
                          int const& n_col);

/** This code test if two keys are equal (Before that the hash should have been
 * used) It is used that way because we assume that the left key have the same
 * type as the right keys. The computation is for just one column and it is used
 * keys or the right keys. Equality means that the column/rows are the same.
 *
 * @param arr1 the first column for the comparison
 * @param iRow1 the row of the first key
 * @param arr2 the second columne for the comparison
 * @param iRow2 the row of the second key
 * @param is_na_equal should na values be considered equal
 * @return True if they are equal and false otherwise.
 */
bool TestEqualColumn(array_info* arr1, int64_t pos1, array_info* arr2,
                     int64_t pos2, bool is_na_equal);

/* This function test if two rows of two arrow columns (which may or may not be
 * the same) are equal, greater or lower than the other.
 *
 * @param arr1            : the first arrow array
 * @param pos1_s, pos1_e  : the starting and ending positions
 * @param arr2            : the second arrow array
 * @param pos2_s, pos2_e  : the starting and ending positions
 * @param na_position_bis : Whether the missing data is first or last
 * @return 1 is arr1[pos1_s:pos1_e] < arr2[pos2_s:pos2_e], 0 is equality, -1 if
 * >.
 */
int ComparisonArrowColumn(std::shared_ptr<arrow::Array> const& arr1,
                          int64_t pos1_s, int64_t pos1_e,
                          std::shared_ptr<arrow::Array> const& arr2,
                          int64_t pos2_s, int64_t pos2_e,
                          bool const& na_position_bis);

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
inline bool TestEqual(std::vector<array_info*> const& columns,
                      size_t const& n_key, size_t const& shift_key1,
                      size_t const& iRow1, size_t const& shift_key2,
                      size_t const& iRow2) {
    // iteration over the list of key for the comparison.
    for (size_t iKey = 0; iKey < n_key; iKey++) {
        bool test = TestEqualColumn(columns[shift_key1 + iKey], iRow1,
                                    columns[shift_key2 + iKey], iRow2, true);
        if (!test) return false;
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
inline bool TestEqualJoin(table_info* table1, table_info* table2,
                          size_t const& iRow1, size_t const& iRow2,
                          size_t const& n_key, bool is_na_equal) {
    // iteration over the list of key for the comparison.
    for (size_t iKey = 0; iKey < n_key; iKey++) {
        bool test = TestEqualColumn(table1->columns[iKey], iRow1,
                                    table2->columns[iKey], iRow2, is_na_equal);
        if (!test) return false;
    }
    // If all keys are equal then we are ok and the keys are equals.
    return true;
};

// is_datetime_timedelta

template <int dtype>
struct is_datetime_timedelta {
    static const bool value = false;
};

template <>
struct is_datetime_timedelta<Bodo_CTypes::DATETIME> {
    static const bool value = true;
};

template <>
struct is_datetime_timedelta<Bodo_CTypes::TIMEDELTA> {
    static const bool value = true;
};

// is_decimal

template <int dtype>
struct is_decimal {
    static const bool value = false;
};

template <>
struct is_decimal<Bodo_CTypes::DECIMAL> {
    static const bool value = true;
};

template <typename T, int dtype>
inline typename std::enable_if<std::is_integral<T>::value, bool>::type
isnan_categorical(T const& val) {
    T miss_idx = -1;
    return val == miss_idx;
}

template <typename T, int dtype>
inline typename std::enable_if<!std::is_integral<T>::value, bool>::type
isnan_categorical(T const& val) {
    return false;
}

template<typename T, int dtype>
inline typename std::enable_if<std::is_integral<T>::value, void>::type
set_na_if_num_categories(T& val, int64_t num_categories) {
    if (val == num_categories) {
        val = -1;
    }
}

template<typename T, int dtype>
inline typename std::enable_if<!std::is_integral<T>::value, void>::type
set_na_if_num_categories(T& val, int64_t num_categories) {
    return;
}

/** This function is used to determine if the value in a Categorical pointer
 * (pointer to a single value in a CategoricalArrayType) isnan.
 * @param the data type for the codes.
 * @param the Categorical Pointer
 * @returns if the value stored at the ptr is nan
 */
bool isnan_categorical_ptr(int dtype, char* ptr);

template <typename T, int dtype>
inline typename std::enable_if<std::is_floating_point<T>::value, bool>::type
isnan_alltype(T const& val) {
    return isnan(val);
}

template <typename T, int dtype>
inline typename std::enable_if<!std::is_floating_point<T>::value &&
                                   is_datetime_timedelta<dtype>::value,
                               bool>::type
isnan_alltype(T const& val) {
    return val == std::numeric_limits<T>::min();
}

template <typename T, int dtype>
inline typename std::enable_if<!std::is_floating_point<T>::value &&
                                   !is_datetime_timedelta<dtype>::value,
                               bool>::type
isnan_alltype(T const& val) {
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
    if (*ptr1_T > *ptr2_T) return -1;
    if (*ptr1_T < *ptr2_T) return 1;
    return 0;
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
    decimal_value_cpp* ptr1_dec = (decimal_value_cpp*)ptr1;
    decimal_value_cpp* ptr2_dec = (decimal_value_cpp*)ptr2;
    double value1 = decimal_to_double(*ptr1_dec);
    double value2 = decimal_to_double(*ptr2_dec);
    if (value1 > value2) return -1;
    if (value1 < value2) return 1;
    return 0;
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
    if (isnan(val1) && isnan(val2)) return 0;
    if (isnan(val2)) {
        if (na_position) return 1;
        return -1;
    }
    if (isnan(val1)) {
        if (na_position) return -1;
        return 1;
    }
    if (val1 > val2) return -1;
    if (val1 < val2) return 1;
    return 0;
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
    if (isnan_date(val1) && isnan_date(val2)) return 0;
    if (isnan_date(val2)) {
        if (na_position) return 1;
        return -1;
    }
    if (isnan_date(val1)) {
        if (na_position) return -1;
        return 1;
    }
    if (val1 > val2) return -1;
    if (val1 < val2) return 1;
    return 0;
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
    if (dtype == Bodo_CTypes::INT32)
        return NumericComparison_int<int32_t>(ptr1, ptr2, na_position);
    if (dtype == Bodo_CTypes::UINT32)
        return NumericComparison_int<uint32_t>(ptr1, ptr2, na_position);
    // for DATE, the missing value is done via NULLABLE_INT_BOOL
    if (dtype == Bodo_CTypes::INT64 || dtype == Bodo_CTypes::DATE)
        return NumericComparison_int<int64_t>(ptr1, ptr2, na_position);
    if (dtype == Bodo_CTypes::UINT64)
        return NumericComparison_int<uint64_t>(ptr1, ptr2, na_position);
    // For DATETIME and TIMEDELTA the NA is done via the
    // std::numeric_limits<int64_t>::min()
    if (dtype == Bodo_CTypes::DATETIME || dtype == Bodo_CTypes::TIMEDELTA)
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
bool KeyComparisonAsPython(size_t const& n_key, int64_t* vect_ascending,
                           std::vector<array_info*> const& columns1,
                           size_t const& shift_key1, size_t const& iRow1,
                           std::vector<array_info*> const& columns2,
                           size_t const& shift_key2, size_t const& iRow2,
                           int64_t* na_position);

int KeyComparisonAsPython_Column(bool const& na_position_bis, array_info* arr1,
                                 size_t const& iRow1, array_info* arr2,
                                 size_t const& iRow2);

// ----------------------- Debug functions -----------------------

/* This is a function used by "DEBUG_PrintSetOfColumn"
 * It takes a column and returns a vector of string on output
 *
 * @param os is the output stream
 * @param arr is the pointer.
 */
void DEBUG_PrintColumn(std::ostream& os, array_info* arr);

/* This is a function used by "DEBUG_PrintSetOfColumn"
 * It takes a column and returns a vector of string on output
 *
 * @param os is the output stream
 * @param arr is the pointer.
 */
void DEBUG_PrintColumn(std::ostream& os, multiple_array_info* arr);

/** The DEBUG_PrintSetOfColumn is printing the contents of the table to
 * the output stream.
 * All cases are supported (NUMPY, SRING, NULLABLE_INT_BOOL) as well as
 * all integer and floating types.
 *
 * The number of rows in the columns do not have to be the same.
 *
 * @param the output stream (e.g. std::cerr or std::cout)
 * @param ListArr the list of columns in input
 * @return Nothing. Everything is put in the stream
 */
void DEBUG_PrintSetOfColumn(std::ostream& os,
                            std::vector<array_info*> const& ListArr);

/** This is a function used for debugging.
 * It prints the nature of the columns of the tables
 *
 * @param the output stream (for example std::cerr or std::cout)
 * @param The list of columns in output
 * @return nothing. Everything is printed to the stream
 */
void DEBUG_PrintRefct(std::ostream& os,
                      std::vector<array_info*> const& ListArr);

inline bool does_keys_have_nulls(std::vector<array_info*> const& key_cols) {
    for (auto key_col : key_cols) {
        if ((key_col->arr_type == bodo_array_type::NUMPY &&
             (key_col->dtype == Bodo_CTypes::FLOAT32 ||
              key_col->dtype == Bodo_CTypes::FLOAT64 ||
              key_col->dtype == Bodo_CTypes::DATETIME ||
              key_col->dtype == Bodo_CTypes::TIMEDELTA)) ||
            key_col->arr_type == bodo_array_type::STRING ||
            key_col->arr_type == bodo_array_type::CATEGORICAL ||
            key_col->arr_type == bodo_array_type::DICT ||
            key_col->arr_type == bodo_array_type::LIST_STRING ||
            key_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
            return true;
        }
    }
    return false;
}

inline bool does_row_has_nulls(std::vector<array_info*> const& key_cols,
                               int64_t const& i) {
    for (auto key_col : key_cols) {
        if (key_col->arr_type == bodo_array_type::CATEGORICAL) {
            std::vector<char> vectNaN = RetrieveNaNentry(key_col->dtype);
            size_t siztype = numpy_item_size[key_col->dtype];
            if (memcmp(key_col->data1 + i * siztype, vectNaN.data(), siztype) ==
                0)
                return true;
        } else if (key_col->arr_type == bodo_array_type::STRING ||
                   key_col->arr_type == bodo_array_type::LIST_STRING ||
                   key_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL ||
                   key_col->arr_type == bodo_array_type::DICT) {
            if (!GetBit((uint8_t*)key_col->null_bitmask, i)) return true;
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
                     std::numeric_limits<int64_t>::min()))
                return true;
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
size_t get_nunique_hashes(uint32_t const* const hashes, const size_t len,
                          bool is_parallel);

/**
 * Given an array of hashes, returns estimate of number of unique hashes
 * of the local array, and global estimate doing a reduction over all ranks.
 * @param hashes: pointer to array of hashes
 * @param len: number of hashes
 * @param is_parallel: true if data is distributed (used to indicate whether
 * tracing should be parallel or not)
 */
std::pair<size_t, size_t> get_nunique_hashes_global(
    uint32_t const* const hashes, const size_t len, bool is_parallel);

#endif  // _ARRAY_UTILS_H_INCLUDED
