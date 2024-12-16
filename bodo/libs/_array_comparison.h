#pragma once
#include <vector>

#include "_bodo_common.h"

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
    size_t const& n_key, const int64_t* vect_ascending,
    std::vector<std::shared_ptr<array_info>> const& columns1,
    size_t const& shift_key1, size_t const& iRow1,
    std::vector<std::shared_ptr<array_info>> const& columns2,
    size_t const& shift_key2, size_t const& iRow2, const int64_t* na_position);

/// Convenience wrapper that calls KeyComparisonAsPython with the shift_key
/// set to 0
bool KeyComparisonAsPython(
    size_t const& n_key, const int64_t* vect_ascending,
    std::vector<std::shared_ptr<array_info>> const& columns1,
    size_t const& iRow1,
    std::vector<std::shared_ptr<array_info>> const& columns2,
    size_t const& iRow2, const int64_t* na_position);

int KeyComparisonAsPython_Column(bool const& na_position_bis,
                                 const std::shared_ptr<array_info>& arr1,
                                 size_t const& iRow1,
                                 const std::shared_ptr<array_info>& arr2,
                                 size_t const& iRow2);
