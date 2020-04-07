// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include "_bodo_common.h"

// ------------------------------------------------
// Choose implementation for unordered map and set
#undef USE_STD
#define USE_TSL_ROBIN
#undef USE_TSL_SPARSE
#undef USE_TSL_HOPSCOTCH

#ifdef USE_STD
#include <unordered_map>
#include <unordered_set>
#define MAP_CONTAINER std::unordered_map
#define SET_CONTAINER std::unordered_set
#endif
#ifdef USE_TSL_ROBIN
#include <include/tsl/robin_map.h>
#include <include/tsl/robin_set.h>
#define MAP_CONTAINER tsl::robin_map
#define SET_CONTAINER tsl::robin_set
#endif
#ifdef USE_TSL_SPARSE
#include <include/tsl/sparse_map.h>
#include <include/tsl/sparse_set.h>
#define MAP_CONTAINER tsl::sparse_map
#define SET_CONTAINER tsl::sparse_set
#endif
#ifdef USE_TSL_HOPSCOTCH
#include <include/tsl/hopscotch_map.h>
#include <include/tsl/hopscotch_set.h>
#define MAP_CONTAINER tsl::hopscotch_map
#define SET_CONTAINER tsl::hopscotch_set
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
    if (dtype == Bodo_CTypes::INT8) return GetTentry<int8_t>(ptr);
    if (dtype == Bodo_CTypes::UINT8) return GetTentry<uint8_t>(ptr);
    if (dtype == Bodo_CTypes::INT16) return GetTentry<int16_t>(ptr);
    if (dtype == Bodo_CTypes::UINT16) return GetTentry<uint16_t>(ptr);
    if (dtype == Bodo_CTypes::INT32) return GetTentry<int32_t>(ptr);
    if (dtype == Bodo_CTypes::UINT32) return GetTentry<uint32_t>(ptr);
    if (dtype == Bodo_CTypes::INT64) return GetTentry<int64_t>(ptr);
    if (dtype == Bodo_CTypes::UINT64) return GetTentry<uint64_t>(ptr);
    if (dtype == Bodo_CTypes::FLOAT32) return GetTentry<float>(ptr);
    if (dtype == Bodo_CTypes::FLOAT64) return GetTentry<double>(ptr);
    if (dtype == Bodo_CTypes::DATE) return GetTentry<int64_t>(ptr);
    if (dtype == Bodo_CTypes::DATETIME) return GetTentry<int64_t>(ptr);
    if (dtype == Bodo_CTypes::TIMEDELTA) return GetTentry<int64_t>(ptr);
    Bodo_PyErr_SetString(PyExc_RuntimeError, "Unsupported case in GetDoubleEntry");
    return 0;
}

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
                             "In DATE case missing values are handled by NULLABLE_INT_BOOL so this case is impossible");
    }
    if (dtype == Bodo_CTypes::DATETIME || dtype == Bodo_CTypes::TIMEDELTA)
        return GetCharVector<int64_t>(std::numeric_limits<int64_t>::min());
    if (dtype == Bodo_CTypes::FLOAT32)
        return GetCharVector<float>(std::nanf("1"));
    if (dtype == Bodo_CTypes::FLOAT64)
        return GetCharVector<double>(std::nan("1"));
    return {};
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
 * @param in_table is the input table.
 * @param ListPairWrite is the vector of list of pairs for the writing of the
 * output table
 * @param shift1 is the first shift (of the left array)
 * @param shift2 is the second shift (of the left array)
 * @param ChoiceColumn is the chosen option
 * @param map_integer_type is the choice of mapping the integer type from
 *    NUMPY to NULLABLE_INT_ARRAY if not available.
 * @return one column of the table output.
 */
array_info* RetrieveArray(
    table_info* const& in_table,
    std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> const& ListPairWrite,
    size_t const& shift1, size_t const& shift2, int const& ChoiceColumn,
    bool const& map_integer_type);

/** This function takes a table, a list of rows and returns the rows obtained
 * by selecting the rows.
 *
 * @param in_table is the input table.
 * @param ListPairWrite is the vector of list of pairs for the writing of the
 * output table
 * @param the number of columns to be selected. if equal to -1 then all columns are selected.
 * @return the table output.
 */
table_info* RetrieveTable(
    table_info* const& in_table,
    std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> const&
    ListPairWrite, int const& n_col);

/** This code test if two keys are equal (Before that the hash should have been
 * used) It is used that way because we assume that the left key have the same
 * type as the right keys. The computation is for just one column and it is used
 * keys or the right keys. Equality means that the column/rows are the same.
 *
 * @param arr1 the first column for the comparison
 * @param iRow1 the row of the first key
 * @param arr2 the second columne for the comparison
 * @param iRow2 the row of the second key
 * @return True if they are equal and false otherwise.
 */
bool TestEqualColumn(array_info* arr1, int64_t pos1, array_info* arr2,
                     int64_t pos2);

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
                                    columns[shift_key2 + iKey], iRow2);
        if (!test) return false;
    }
    // If all keys are equal then we are ok and the keys are equals.
    return true;
};

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
inline typename std::enable_if<!std::is_floating_point<T>::value, int>::type
NumericComparison_int(char* ptr1, char* ptr2, bool const& na_position) {
    T* ptr1_T = (T*)ptr1;
    T* ptr2_T = (T*)ptr2;
    if (*ptr1_T > *ptr2_T) return -1;
    if (*ptr1_T < *ptr2_T) return 1;
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
inline typename std::enable_if<std::is_floating_point<T>::value, int>::type
NumericComparison_float(char* ptr1, char* ptr2, bool const& na_position) {
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
inline typename std::enable_if<!std::is_floating_point<T>::value, int>::type
NumericComparison_date(char* ptr1, char* ptr2, bool const& na_position) {
    T* ptr1_T = (T*)ptr1;
    T* ptr2_T = (T*)ptr2;
    T val1 = *ptr1_T;
    T val2 = *ptr2_T;
    auto isnan_date=[&](T const& val) {
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
    // For DATETIME and TIMEDELTA the NA is done via the std::numeric_limits<int64_t>::min()
    if (dtype == Bodo_CTypes::DATETIME || dtype == Bodo_CTypes::TIMEDELTA)
        return NumericComparison_date<int64_t>(ptr1, ptr2, na_position);
    if (dtype == Bodo_CTypes::FLOAT32)
        return NumericComparison_float<float>(ptr1, ptr2, na_position);
    if (dtype == Bodo_CTypes::FLOAT64)
        return NumericComparison_float<double>(ptr1, ptr2, na_position);
    Bodo_PyErr_SetString(PyExc_RuntimeError,
                         "Invalid dtype put on input to NumericComparison");
    return 0;
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
 * @param na_position: if true NaN values are largest, if false smallest.
 * @return true if (shift_key1,iRow1) < (shift_key2,iRow2) , false otherwise
 */
bool KeyComparisonAsPython(size_t const& n_key, int64_t* vect_ascending,
                           std::vector<array_info*> const& columns1,
                           size_t const& shift_key1, size_t const& iRow1,
                           std::vector<array_info*> const& columns2,
                           size_t const& shift_key2, size_t const& iRow2,
                           bool const& na_position);

// ----------------------- Debug functions -----------------------

/* This is a function used by "DEBUG_PrintSetOfColumn"
 * It takes a column and returns a vector of string on output
 *
 * @param arr is the pointer.
 * @return The vector of strings to be used later.
 */
std::vector<std::string> DEBUG_PrintColumn(array_info* arr);

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
