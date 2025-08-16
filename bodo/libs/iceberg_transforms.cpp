
#include "iceberg_transforms.h"

#include <boost/xpressive/xpressive.hpp>
#include "_bodo_common.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "_array_hash.h"
#include "_datetime_ext.h"
#include "_datetime_utils.h"

//
// Utilities for Iceberg Transforms
//

/**
 * @brief Helper function to convert int32 date arrays to int64 date arrays,
 * which are used for hashing dates.
 *
 * @param in_arr Input DATE array (NULLABLE)
 * @return array_info* Nullable INT64 array.
 */
std::shared_ptr<array_info> convert_date_to_int64(
    std::shared_ptr<array_info> in_arr) {
    tracing::Event ev("convert_date_to_int64");
    const uint64_t nRow = in_arr->length;
    ev.add_attribute("nRows", nRow);

    if (in_arr->dtype != Bodo_CTypes::DATE) {
        throw std::runtime_error("Unsupported dtype type '" +
                                 GetDtype_as_string(in_arr->dtype) +
                                 "' provided to convert_date_to_int64. "
                                 "Expected Bodo_CTypes::DATE array");
    }

    if (in_arr->arr_type != bodo_array_type::NULLABLE_INT_BOOL) {
        throw std::runtime_error(
            "Unsupported arr_type '" + GetArrType_as_string(in_arr->arr_type) +
            "' provided to convert_date_to_int64. Expected "
            "bodo_array_type::NULLABLE_INT_BOOL.");
    }

    std::shared_ptr<array_info> out_arr =
        alloc_nullable_array(nRow, Bodo_CTypes::INT64, 0);
    for (uint64_t i = 0; i < nRow; i++) {
        out_arr->at<int64_t, bodo_array_type::NULLABLE_INT_BOOL>(i) =
            in_arr->at<int32_t, bodo_array_type::NULLABLE_INT_BOOL>(i);
    }

    int64_t n_bytes = ((nRow + 7) >> 3);
    std::copy(
        in_arr->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>(),
        in_arr->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>() + n_bytes,
        out_arr->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>());

    return out_arr;
}

/**
 * @brief Helper function to convert nanoseconds to microseconds.
 * Essentially creates a new nullable int64 array where the values are divided
 * by 1000. NaTs in original array are converted to nulls.
 *
 * @param in_arr Input DATETIME array (NUMPY)
 * @return std::shared_ptr<array_info> Nullable INT64 array.
 */
std::shared_ptr<array_info> convert_datetime_ns_to_us(
    std::shared_ptr<array_info> in_arr) {
    tracing::Event ev("convert_datetime_ns_to_us");
    const uint64_t nRow = in_arr->length;
    ev.add_attribute("nRows", nRow);

    if (in_arr->dtype != Bodo_CTypes::DATETIME) {
        throw std::runtime_error(
            "Unsupported dtype type '" + GetDtype_as_string(in_arr->dtype) +
            "' provided to convert_datetime_ns_to_us. Expected "
            "Bodo_CTypes::DATETIME array.");
    }

    if (in_arr->arr_type != bodo_array_type::NUMPY &&
        in_arr->arr_type != bodo_array_type::NULLABLE_INT_BOOL) {
        throw std::runtime_error(
            "Unsupported arr_type '" + GetArrType_as_string(in_arr->arr_type) +
            "' provided to convert_datetime_ns_to_us. Expected "
            "bodo_array_type::NUMPY or bodo_array_type::NULLABLE_INT_BOOL.");
    }

    std::shared_ptr<array_info> out_arr =
        alloc_nullable_array(nRow, Bodo_CTypes::INT64, 0);
    for (uint64_t i = 0; i < nRow; i++) {
        if (in_arr->at<int64_t>(i) == std::numeric_limits<int64_t>::min()) {
            // if value is NaT (equals
            // std::numeric_limits<int64_t>::min()) we set it as
            // null in the transformed array.
            out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i, false);
            // Set it to -1 to avoid non-deterministic behavior (e.g. during
            // hashing)
            out_arr->at<int64_t, bodo_array_type::NULLABLE_INT_BOOL>(i) = -1;
        } else {
            // convert from nanoseconds to microseconds
            out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i, true);
            out_arr->at<int64_t, bodo_array_type::NULLABLE_INT_BOOL>(i) =
                in_arr->at<int64_t>(i) / 1000;
        }
    }
    return out_arr;
}

// Copied from _datetime_ext.cpp
int64_t extract_unit(int64_t* d, int64_t unit) {
    assert(unit > 0);
    npy_int64 div = *d / unit;
    npy_int64 mod = *d % unit;
    if (mod < 0) {
        mod += unit;
        div -= 1;
    }
    assert(mod >= 0);
    return div;
}

/**
 * @brief Get the years since epoch from datetime object
 *
 * @param dt DATETIME (as int64)
 * @return int32_t Number of years since epoch
 */
static inline int32_t get_years_since_epoch_from_datetime(int64_t* dt) {
    constexpr int64_t perday = 24LL * 60LL * 60LL * 1000LL * 1000LL * 1000LL;
    int64_t days = extract_unit(dt, perday);
    // Get years from 1970
    return days_to_yearsdays(&days) - 1970;
}

/**
 * @brief Helper function used in get_months_since_epoch_from_datetime which
 * calculates the month if we were to go days many days into the given year.
 *
 * @param year The year to calculate months value for (required for leapyear
 * calculation)
 * @param days Number of days to go into the year. It is assumed that it is
 * <=365 (for non-leap year) and <=366 (for leap years)
 * @param[out] month Month of year (from 0 to 11)
 */
static void get_remaining_months(int64_t year, int64_t* days, int64_t* month) {
    const int* month_lengths = days_per_month_table[is_leapyear(year)];
    for (int i = 0; i < 12; ++i) {
        if (*days < month_lengths[i]) {
            *month = i;
            return;
        } else {
            *days -= month_lengths[i];
        }
    }
}

/**
 * @brief Get the months since epoch from DATETIME object (represented as int64)
 *
 * @param dt DATETIME (as int64)
 * @return int32_t Number of months since epoch for the provided DATETIME
 */
static inline int32_t get_months_since_epoch_from_datetime(int64_t* dt) {
    constexpr int64_t perday = 24LL * 60LL * 60LL * 1000LL * 1000LL * 1000LL;
    int64_t days = extract_unit(dt, perday);
    int64_t year = days_to_yearsdays(&days);
    int64_t rem_months = -1;
    get_remaining_months(year, &days, &rem_months);
    // Compute months from 1970-01-01.
    return ((year - 1970) * 12) + rem_months;
}

/**
 * @brief Get the days since epoch from DATETIME object (represented as int64)
 *
 * @param dt DATETIME object (as int64)
 * @return int64_t Number of days since epoch for the provided DATETIME object
 */
static inline int64_t get_days_since_epoch_from_datetime(int64_t* dt) {
    constexpr int64_t perday = 24LL * 60LL * 60LL * 1000LL * 1000LL * 1000LL;
    int64_t days = extract_unit(dt, perday);
    return days;
}

/**
 * @brief Get the hours since epoch from DATETIME object (represented as int64)
 *
 * @param dt DATETIME object (as int64)
 * @return int32_t Number of hours since epoch for the provided DATETIME object
 */
static inline int32_t get_hours_since_epoch_from_datetime(int64_t* dt) {
    constexpr int64_t perhour = 60LL * 60LL * 1000LL * 1000LL * 1000LL;
    // hours from 1970-01-01 00:00:00 UTC
    return extract_unit(dt, perhour);
}

//
// Iceberg Partition Transforms
//

std::shared_ptr<array_info> array_transform_bucket_N(
    std::shared_ptr<array_info> in_arr, int64_t N, bool is_parallel) {
    assert(N > 0);
    tracing::Event ev("array_transform_bucket_N", is_parallel);
    const uint64_t nRow = in_arr->length;
    ev.add_attribute("nRows", nRow);

    std::shared_ptr<array_info> out_arr =
        alloc_nullable_array(nRow, Bodo_CTypes::UINT32, 0);
    std::unique_ptr<uint32_t[]> hashes = std::unique_ptr<uint32_t[]>(
        (uint32_t*)out_arr->data1<bodo_array_type::NULLABLE_INT_BOOL>());

    int64_t n_bytes = ((nRow + 7) >> 3);

    // We calculate DICT case separately since `hash_array`
    // hashes the indices instead of the actual
    // strings, which doesn't work for Iceberg.
    if (in_arr->arr_type == bodo_array_type::DICT) {
        // Calculate hashes on the dict
        std::unique_ptr<uint32_t[]> dict_hashes =
            std::make_unique<uint32_t[]>(in_arr->child_arrays[0]->length);
        hash_array</*use_murmurhash*/ true>(
            dict_hashes.get(), in_arr->child_arrays[0],
            in_arr->child_arrays[0]->length, 0, is_parallel, false);
        // Iterate over the elements and assign hash from the dict
        for (uint64_t i = 0; i < nRow; i++) {
            if (!in_arr->child_arrays[1]
                     ->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i)) {
                // in case index i is null,
                // in_arr->child_arrays[1]->at<int32_t>(i) is a garbage value
                // and might be out of bounds for dict_hashes, so we set the
                // hash to -1 to avoid access errors.
                hashes[i] = -1;
            } else {
                hashes[i] =
                    dict_hashes[in_arr->child_arrays[1]
                                    ->at<dict_indices_t,
                                         bodo_array_type::NULLABLE_INT_BOOL>(
                                        i)];
            }
        }
        // Copy the null bitmask from the indices array
        memcpy(out_arr->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>(),
               in_arr->child_arrays[1]
                   ->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>(),
               n_bytes);
    } else {
        // hash_array doesn't hash nulls
        // as we need. In particular, we need null
        // to hash to null. Hence, we copied over
        // the null bitmask.
        if (in_arr->dtype == Bodo_CTypes::DATETIME) {
            // In case of datetime, first convert to microsecond (as int64 which
            // is important for correctness) and then hash
            std::shared_ptr<array_info> us_array =
                convert_datetime_ns_to_us(in_arr);
            assert(us_array->arr_type == bodo_array_type::NULLABLE_INT_BOOL);
            // DATETIME arrays are always NUMPY so there's no null_bitmask
            // in in_arr. The array could have NaTs though, which
            // convert_datetime_ns_to_us will convert to nulls, so we can
            // use the bitmask from us_array as is for the out_arr.
            memcpy(out_arr->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>(),
                   us_array->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>(),
                   n_bytes);
            hash_array</*use_murmurhash*/ true>(hashes.get(), us_array, nRow, 0,
                                                is_parallel, false);
        } else if (in_arr->dtype == Bodo_CTypes::DATE) {
            // Based on
            // https://iceberg.apache.org/spec/#appendix-b-32-bit-hash-requirements,
            // dates should be hashed as int32 (after computing number of days
            // since epoch), however Spark, Iceberg-Python and example in
            // Iceberg Spec seems to use int64.
            std::shared_ptr<array_info> i64_array(
                convert_date_to_int64(in_arr));
            assert(i64_array->arr_type == bodo_array_type::NULLABLE_INT_BOOL);
            hash_array</*use_murmurhash*/ true>(hashes.get(), i64_array, nRow,
                                                0, is_parallel, false);
            // DATE arrays are always NULLABLE_INT_BOOL arrays, so we can copy
            // over the null_bitmask as is.
            memcpy(out_arr->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>(),
                   in_arr->null_bitmask(), n_bytes);
        } else if (in_arr->dtype == Bodo_CTypes::INT32) {
            // in case of int32, we need to cast to int64 before hashing
            // (https://iceberg.apache.org/spec/#appendix-b-32-bit-hash-requirements)
            std::shared_ptr<array_info> int64_arr =
                alloc_array_top_level(in_arr->length, in_arr->n_sub_elems(), 0,
                                      in_arr->arr_type, Bodo_CTypes::INT64);
            if (in_arr->null_bitmask()) {
                // Copy the null bitmask if it exists for the arr type
                memcpy(int64_arr->null_bitmask(), in_arr->null_bitmask(),
                       n_bytes);
                // Copy it over to the out_arr null_bitmask as well
                memcpy(
                    out_arr->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>(),
                    in_arr->null_bitmask(), n_bytes);
            } else {
                // In case this array type (NUMPY) doesn't have a null_bitmask
                // there should be no nulls in the array and we can set
                // null_bitmask to all 1s.
                memset(
                    out_arr->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>(),
                    0xFF, n_bytes);
            }
            for (uint64_t i = 0; i < nRow; i++) {
                int64_arr->at<int64_t>(i) = (int64_t)in_arr->at<int32_t>(i);
            }
            hash_array</*use_murmurhash*/ true>(hashes.get(), int64_arr, nRow,
                                                0, is_parallel, false);
        } else {
            hash_array</*use_murmurhash*/ true>(hashes.get(), in_arr, nRow, 0,
                                                is_parallel, false);
            if (in_arr->null_bitmask()) {
                // Copy the null bitmask if it exists for the arr type
                memcpy(
                    out_arr->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>(),
                    in_arr->null_bitmask(), n_bytes);
            } else {
                // Otherwise set it to all 1s.
                // DATETIME which uses a NUMPY array (but can have NaTs)
                // is handled separately above, and floats (which can have NaNs)
                // are not supported for the bucket transform at all, so this
                // should be safe.
                memset(
                    out_arr->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>(),
                    0xFF, n_bytes);
            }
        }
    }

    for (uint64_t i = 0; i < nRow; i++) {
        hashes[i] = (hashes[i] & INT_MAX) % N;
    }
    // Release the ptr to hashes because we don't have ownership
    hashes.release();

    return out_arr;
}

template <typename T>
static inline T int_truncate_helper(T& v, int64_t W) {
    return v - (((v % W) + W) % W);
}

std::shared_ptr<array_info> array_transform_truncate_W(
    std::shared_ptr<array_info> in_arr, int64_t width, bool is_parallel) {
    assert(width > 0);
    tracing::Event ev("array_transform_truncate_W", is_parallel);
    ev.add_attribute("W", width);
    const uint64_t nRow = in_arr->length;
    ev.add_attribute("nRows", nRow);

    std::shared_ptr<array_info> out_arr;
    if (in_arr->arr_type == bodo_array_type::STRING) {
        offset_t* in_offsets =
            (offset_t*)in_arr->data2<bodo_array_type::STRING>();
        offset_t str_len;
        uint64_t n_chars = 0;
        // Calculate the total chars first
        for (uint64_t i = 0; i < nRow; i++) {
            if (!in_arr->get_null_bit<bodo_array_type::STRING>(i)) {
                continue;
            }
            str_len = in_offsets[i + 1] - in_offsets[i];
            n_chars += std::min(str_len, (offset_t)width);
        }
        // Allocate output array. If this string array is a dictionary then
        // it is still globally replicated and sorted, but the truncation
        // may break uniqueness.
        out_arr = alloc_string_array(in_arr->dtype, nRow, n_chars, -1, 0,
                                     in_arr->is_globally_replicated, false,
                                     in_arr->is_locally_sorted);
        // Copy over truncated strings to the new array
        offset_t* out_offsets =
            (offset_t*)out_arr->data2<bodo_array_type::STRING>();
        out_offsets[0] = 0;
        for (uint64_t i = 0; i < nRow; i++) {
            if (!in_arr->get_null_bit<bodo_array_type::STRING>(i)) {
                out_arr->set_null_bit<bodo_array_type::STRING>(i, false);
                out_offsets[i + 1] = out_offsets[i];
                continue;
            }
            str_len = in_offsets[i + 1] - in_offsets[i];
            str_len = std::min(str_len, (offset_t)width);
            memcpy(out_arr->data1<bodo_array_type::STRING>() + out_offsets[i],
                   in_arr->data1<bodo_array_type::STRING>() + in_offsets[i],
                   sizeof(char) * str_len);
            out_offsets[i + 1] = out_offsets[i] + str_len;
            out_arr->set_null_bit<bodo_array_type::STRING>(i, true);
        }
        return out_arr;
    }
    if (in_arr->arr_type == bodo_array_type::NUMPY) {
        if (in_arr->dtype == Bodo_CTypes::INT32) {
            out_arr = alloc_numpy(nRow, in_arr->dtype);
            for (uint64_t i = 0; i < nRow; i++) {
                out_arr->at<int32_t, bodo_array_type::NUMPY>(i) =
                    int_truncate_helper<int32_t>(
                        in_arr->at<int32_t, bodo_array_type::NUMPY>(i), width);
            }
            return out_arr;
        }
        if (in_arr->dtype == Bodo_CTypes::INT64) {
            out_arr = alloc_numpy(nRow, in_arr->dtype);
            for (uint64_t i = 0; i < nRow; i++) {
                out_arr->at<int64_t, bodo_array_type::NUMPY>(i) =
                    int_truncate_helper<int64_t>(
                        in_arr->at<int64_t, bodo_array_type::NUMPY>(i), width);
            }
            return out_arr;
        }
        if (in_arr->dtype == Bodo_CTypes::DECIMAL) {
            throw std::runtime_error(
                "array_transform_truncate_W not yet implemented for decimals");
        }
    }
    if (in_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        if (in_arr->dtype == Bodo_CTypes::INT32) {
            out_arr = alloc_nullable_array(nRow, in_arr->dtype, 0);
            for (uint64_t i = 0; i < nRow; i++) {
                if (!in_arr->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                        i)) {
                    out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                        i, false);
                    continue;
                }
                out_arr->at<int32_t, bodo_array_type::NULLABLE_INT_BOOL>(i) =
                    int_truncate_helper<int32_t>(
                        in_arr->at<int32_t, bodo_array_type::NULLABLE_INT_BOOL>(
                            i),
                        width);
                out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i,
                                                                          true);
            }
            return out_arr;
        }
        if (in_arr->dtype == Bodo_CTypes::INT64) {
            out_arr = alloc_nullable_array(nRow, in_arr->dtype, 0);
            for (uint64_t i = 0; i < nRow; i++) {
                if (!in_arr->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                        i)) {
                    out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                        i, false);
                    continue;
                }
                out_arr->at<int64_t>(i) = int_truncate_helper<int64_t>(
                    in_arr->at<int64_t, bodo_array_type::NULLABLE_INT_BOOL>(i),
                    width);
                out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i,
                                                                          true);
            }
            return out_arr;
        }
    }
    if (in_arr->arr_type == bodo_array_type::DICT) {
        std::shared_ptr<array_info> dict_data_arr = in_arr->child_arrays[0];
        // Call recursively on the dictionary
        std::shared_ptr<array_info> trunc_dict_data_arr =
            array_transform_truncate_W(dict_data_arr, width, is_parallel);
        // Make a copy of the indices
        std::shared_ptr<array_info> indices_copy =
            copy_array(in_arr->child_arrays[1]);
        // Note that if the input dictionary was sorted, the
        // transformed dictionary would be sorted too since
        // it's just a truncation. It might not have all
        // unique elements though.
        out_arr = create_dict_string_array(trunc_dict_data_arr, indices_copy);
        return out_arr;
    }
    // Throw an error if type is not supported (e.g. CATEGORICAL)
    std::string err = "array_transform_truncate_W not supported.";
    throw std::runtime_error(err);
}

std::shared_ptr<array_info> array_transform_void(
    std::shared_ptr<array_info> in_arr, bool is_parallel) {
    tracing::Event ev("array_transform_void", is_parallel);
    // We simply return a nullable int32 array
    // with all nulls (as per the Iceberg spec)
    return alloc_nullable_array_all_nulls(in_arr->length, Bodo_CTypes::INT32);
}

std::shared_ptr<array_info> array_transform_year(
    std::shared_ptr<array_info> in_arr, bool is_parallel) {
    tracing::Event ev("array_transform_year", is_parallel);
    const uint64_t nRow = in_arr->length;
    ev.add_attribute("nRows", nRow);
    std::shared_ptr<array_info> out_arr =
        alloc_nullable_array(nRow, Bodo_CTypes::INT32, 0);

    int64_t n_bytes = ((nRow + 7) >> 3);

    if ((in_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) &&
        (in_arr->dtype == Bodo_CTypes::DATE)) {
        for (uint64_t i = 0; i < nRow; i++) {
            int64_t days =
                in_arr->at<int32_t, bodo_array_type::NULLABLE_INT_BOOL>(i);
            int64_t years = days_to_yearsdays(&days);
            out_arr->at<int32_t, bodo_array_type::NULLABLE_INT_BOOL>(i) =
                years - 1970;
        }
        // Copy the null bitmask as is
        memcpy(out_arr->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>(),
               in_arr->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>(),
               n_bytes);
        return out_arr;
    }
    if ((in_arr->arr_type == bodo_array_type::NUMPY ||
         in_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) &&
        (in_arr->dtype == Bodo_CTypes::DATETIME)) {
        for (uint64_t i = 0; i < nRow; i++) {
            if (in_arr->at<int64_t>(i) == std::numeric_limits<int64_t>::min()) {
                // if value is NaT (equals
                // std::numeric_limits<int64_t>::min()) we set it as
                // null in the transformed array.
                out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                    i, false);
                // Set it to -1 to avoid non-deterministic behavior (e.g. during
                // hashing)
                out_arr->at<int32_t, bodo_array_type::NULLABLE_INT_BOOL>(i) =
                    -1;
            } else {
                out_arr->at<int32_t, bodo_array_type::NULLABLE_INT_BOOL>(i) =
                    get_years_since_epoch_from_datetime(
                        &in_arr->at<int64_t>(i));
                out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i,
                                                                          true);
            }
        }
        return out_arr;
    }
    // Throw an error if type is not supported.
    std::string err = "array_transform_year not supported.";
    throw std::runtime_error(err);
}

std::shared_ptr<array_info> array_transform_month(
    std::shared_ptr<array_info> in_arr, bool is_parallel) {
    tracing::Event ev("array_transform_month", is_parallel);
    const uint64_t nRow = in_arr->length;
    ev.add_attribute("nRows", nRow);
    std::shared_ptr<array_info> out_arr =
        alloc_nullable_array(nRow, Bodo_CTypes::INT32, 0);
    int64_t n_bytes = ((nRow + 7) >> 3);

    if ((in_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) &&
        (in_arr->dtype == Bodo_CTypes::DATE)) {
        for (uint64_t i = 0; i < nRow; i++) {
            int64_t days =
                        in_arr->at<int32_t, bodo_array_type::NULLABLE_INT_BOOL>(
                            i),
                    month = 0;
            int64_t years = days_to_yearsdays(&days);
            get_month_day(years, days, &month, &days);
            out_arr->at<int32_t, bodo_array_type::NULLABLE_INT_BOOL>(i) =
                (years - 1970) * 12 + month;
        }
        // Copy the null bitmask as is
        memcpy(out_arr->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>(),
               in_arr->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>(),
               n_bytes);
        return out_arr;
    }

    if ((in_arr->arr_type == bodo_array_type::NUMPY ||
         in_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) &&
        (in_arr->dtype == Bodo_CTypes::DATETIME)) {
        for (uint64_t i = 0; i < nRow; i++) {
            if (in_arr->at<int64_t>(i) == std::numeric_limits<int64_t>::min()) {
                // if value is NaT (equals
                // std::numeric_limits<int64_t>::min()) we set it as
                // null in the transformed array.
                out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                    i, false);
                // Set it to -1 to avoid non-deterministic behavior (e.g. during
                // hashing)
                out_arr->at<int32_t, bodo_array_type::NULLABLE_INT_BOOL>(i) =
                    -1;
            } else {
                out_arr->at<int32_t, bodo_array_type::NULLABLE_INT_BOOL>(i) =
                    (int32_t)get_months_since_epoch_from_datetime(
                        &in_arr->at<int64_t>(i));
                out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i,
                                                                          true);
            }
        }
        return out_arr;
    }

    // Throw an error if type is not supported.
    std::string err = "array_transform_month not supported.";
    throw std::runtime_error(err);
}

std::shared_ptr<array_info> array_transform_day(
    std::shared_ptr<array_info> in_arr, bool is_parallel) {
    tracing::Event ev("array_transform_day", is_parallel);
    const uint64_t nRow = in_arr->length;
    ev.add_attribute("nRows", nRow);
    std::shared_ptr<array_info> out_arr =
        alloc_nullable_array(nRow, Bodo_CTypes::INT64, 0);

    int64_t n_bytes = ((nRow + 7) >> 3);

    if ((in_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) &&
        (in_arr->dtype == Bodo_CTypes::DATE)) {
        for (uint64_t i = 0; i < nRow; i++) {
            out_arr->at<int64_t, bodo_array_type::NULLABLE_INT_BOOL>(i) =
                in_arr->at<int32_t, bodo_array_type::NULLABLE_INT_BOOL>(i);
        }
        // Copy the null bitmask as is
        memcpy(out_arr->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>(),
               in_arr->null_bitmask<bodo_array_type::NULLABLE_INT_BOOL>(),
               n_bytes);
        return out_arr;
    }

    if ((in_arr->arr_type == bodo_array_type::NUMPY ||
         in_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) &&
        (in_arr->dtype == Bodo_CTypes::DATETIME)) {
        for (uint64_t i = 0; i < nRow; i++) {
            if (in_arr->at<int64_t>(i) == std::numeric_limits<int64_t>::min()) {
                // if value is NaT (equals
                // std::numeric_limits<int64_t>::min()) we set it as
                // null in the transformed array.
                out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                    i, false);
                // Set it to -1 to avoid non-deterministic behavior (e.g. during
                // hashing)
                out_arr->at<int64_t, bodo_array_type::NULLABLE_INT_BOOL>(i) =
                    -1;
            } else {
                out_arr->at<int64_t, bodo_array_type::NULLABLE_INT_BOOL>(i) =
                    get_days_since_epoch_from_datetime(&in_arr->at<int64_t>(i));
                out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i,
                                                                          true);
            }
        }
        return out_arr;
    }

    // Throw an error if type is not supported.
    std::string err = "array_transform_day not supported.";
    throw std::runtime_error(err);
}

std::shared_ptr<array_info> array_transform_hour(
    std::shared_ptr<array_info> in_arr, bool is_parallel) {
    tracing::Event ev("array_transform_hour", is_parallel);
    const uint64_t nRow = in_arr->length;
    ev.add_attribute("nRows", nRow);

    std::shared_ptr<array_info> out_arr =
        alloc_nullable_array(nRow, Bodo_CTypes::INT32, 0);

    if ((in_arr->arr_type == bodo_array_type::NUMPY ||
         in_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) &&
        (in_arr->dtype == Bodo_CTypes::DATETIME)) {
        for (uint64_t i = 0; i < nRow; i++) {
            if (in_arr->at<int64_t>(i) == std::numeric_limits<int64_t>::min()) {
                // if value is NaT (equals
                // std::numeric_limits<int64_t>::min()) we set it as
                // null in the transformed array.
                out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                    i, false);
                // Set it to -1 to avoid non-deterministic behavior (e.g. during
                // hashing)
                out_arr->at<int32_t, bodo_array_type::NULLABLE_INT_BOOL>(i) =
                    -1;
            } else {
                out_arr->at<int32_t, bodo_array_type::NULLABLE_INT_BOOL>(i) =
                    get_hours_since_epoch_from_datetime(
                        &in_arr->at<int64_t>(i));
                out_arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i,
                                                                          true);
            }
        }
        return out_arr;
    }

    // Throw an error if type is not supported.
    std::string err = "array_transform_hour not supported.";
    throw std::runtime_error(err);
}

std::shared_ptr<array_info> iceberg_identity_transform(
    std::shared_ptr<array_info> in_arr, bool is_parallel) {
    tracing::Event ev("iceberg_identity_transform", is_parallel);
    const uint64_t nRow = in_arr->length;
    ev.add_attribute("nRows", nRow);

    if (in_arr->dtype == Bodo_CTypes::DATETIME) {
        // For partitioning, we need the partition value to be in microseconds
        // instead of the default which is nanoseconds. Note that it is
        // important due to precision issues during partitioning. e.g. two
        // different nanosecond values could correspond to the same microsecond
        // value, so during partitioning it's important for them to be in the
        // same partition.
        // For sorting it would be fine to keep it in
        // nanoseconds, but we are not optimizing that at this time.
        return convert_datetime_ns_to_us(in_arr);
    } else {
        return in_arr;  // return as is
    }
}

std::shared_ptr<array_info> iceberg_transform(
    std::shared_ptr<array_info> in_arr, std::string transform_name, int64_t arg,
    bool is_parallel) {
    if (transform_name == "bucket") {
        return array_transform_bucket_N(in_arr, arg, is_parallel);
    }
    if (transform_name == "truncate") {
        return array_transform_truncate_W(in_arr, arg, is_parallel);
    }
    if (transform_name == "void") {
        return array_transform_void(in_arr, is_parallel);
    }
    if (transform_name == "year") {
        return array_transform_year(in_arr, is_parallel);
    }
    if (transform_name == "month") {
        return array_transform_month(in_arr, is_parallel);
    }
    if (transform_name == "day") {
        return array_transform_day(in_arr, is_parallel);
    }
    if (transform_name == "hour") {
        return array_transform_hour(in_arr, is_parallel);
    }

    throw std::runtime_error("Unrecognized transform '" + transform_name +
                             "' provided.");
}

std::string transform_val_to_str(std::string transform_name,
                                 std::shared_ptr<array_info> in_arr,
                                 std::shared_ptr<array_info> transformed_arr,
                                 size_t idx) {
    if (transformed_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL ||
        transformed_arr->arr_type == bodo_array_type::STRING ||
        transformed_arr->arr_type == bodo_array_type::DICT) {
        if (!transformed_arr->get_null_bit(idx)) {
            return std::string("null");
        }
    }

    if (transform_name == "year") {
        return std::to_string(transformed_arr->at<int32_t>(idx) + 1970);
    }
    if ((transform_name == "identity" || transform_name == "day") &&
        in_arr->dtype == Bodo_CTypes::DATE) {
        return array_val_to_str(in_arr, idx);
    }
    if (transform_name == "day") {
        // day transform is only supported for DATETIME and DATE types.
        // DATE is handled right above, so this must be DATETIME.
        assert(in_arr->dtype == Bodo_CTypes::DATETIME);

        // If the value in in_arr was NaT, the day transform
        // would've set the null bit to false in the output
        // nullable int64 array, and that case would've been caught
        // at the top of the function. If it reaches here, we can
        // safely assume that it's a valid value and not NaT.

        int64_t day = transformed_arr->at<int64_t>(idx);
        int64_t year = days_to_yearsdays(&day);

        int64_t rem_months = -1;
        get_remaining_months(year, &day, &rem_months);
        int64_t month = rem_months + 1;
        day += 1;

        std::string date_str;
        date_str.reserve(10);
        date_str += std::to_string(year) + "-";
        if (month < 10) {
            date_str += "0";
        }
        date_str += std::to_string(month) + "-";
        if (day < 10) {
            date_str += "0";
        }
        date_str += std::to_string(day);
        return date_str;
    }
    return array_val_to_str(transformed_arr, idx);
}

PyObject* iceberg_transformed_val_to_py(std::shared_ptr<array_info> arr,
                                        size_t idx) {
    // Return Python None in case the null bit is set
    if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL ||
        arr->arr_type == bodo_array_type::STRING ||
        arr->arr_type == bodo_array_type::DICT) {
        if (!arr->get_null_bit(idx)) {
            return Py_None;
        }
    }
    switch (arr->dtype) {
        case Bodo_CTypes::INT32:
        case Bodo_CTypes::DATE:
            return PyLong_FromLong(arr->at<int32_t>(idx));
        case Bodo_CTypes::UINT32:
            return PyLong_FromUnsignedLong(arr->at<uint32_t>(idx));
        case Bodo_CTypes::INT64:
            return PyLong_FromLongLong(arr->at<int64_t>(idx));
        case Bodo_CTypes::UINT64:
            return PyLong_FromUnsignedLongLong(arr->at<uint64_t>(idx));

        case Bodo_CTypes::FLOAT32:
            return PyFloat_FromDouble((double)arr->at<float>(idx));
        case Bodo_CTypes::FLOAT64:
            return PyFloat_FromDouble(arr->at<double>(idx));

        case Bodo_CTypes::_BOOL:
            bool is_true;
            if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
                // Nullable booleans have 1 bit per boolean
                is_true = GetBit(
                    (uint8_t*)arr->data1<bodo_array_type::NULLABLE_INT_BOOL>(),
                    idx);
            } else {
                is_true = arr->at<bool>(idx);
            }
            if (is_true) {
                return Py_True;
            } else {
                return Py_False;
            }

        case Bodo_CTypes::STRING: {
            switch (arr->arr_type) {
                case bodo_array_type::DICT: {
                    int32_t dict_idx =
                        arr->child_arrays[1]
                            ->at<dict_indices_t,
                                 bodo_array_type::NULLABLE_INT_BOOL>(idx);
                    offset_t* offsets = (offset_t*)arr->child_arrays[0]
                                            ->data2<bodo_array_type::STRING>();
                    return PyUnicode_FromString(
                        std::string(arr->child_arrays[0]
                                            ->data1<bodo_array_type::STRING>() +
                                        offsets[dict_idx],
                                    offsets[dict_idx + 1] - offsets[dict_idx])
                            .c_str());
                }
                default: {
                    offset_t* offsets =
                        (offset_t*)arr->data2<bodo_array_type::STRING>();
                    return PyUnicode_FromString(
                        std::string(arr->data1<bodo_array_type::STRING>() +
                                        offsets[idx],
                                    offsets[idx + 1] - offsets[idx])
                            .c_str());
                }
            }
        }

        default: {
            std::vector<char> error_msg(100);
            snprintf(
                error_msg.data(), error_msg.size(),
                "iceberg_transformed_val_to_py not implemented for dtype %d",
                arr->dtype);
            throw std::runtime_error(error_msg.data());
        }
    }
    // We explicitly don't support DATE and DATETIME since the output of
    // the transforms should never be these dtypes. In case of identity or day
    // transform, these are both transformed into int64 arrays. In case of void,
    // we return a int32 nullable array with all nulls.
}
