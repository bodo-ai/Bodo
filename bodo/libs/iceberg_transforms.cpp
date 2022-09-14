// Copyright (C) 2022 Bodo Inc. All rights reserved.

#include "iceberg_transforms.h"

//
// Utilities for Iceberg Transforms
//

/**
 * @brief Helper function to convert nanoseconds to microseconds.
 * Essentially creates a new nullable int64 array where the values are divided
 * by 1000. NaTs in original array are converted to nulls.
 *
 * @param in_arr Input DATETIME array (NUMPY)
 * @return array_info* Nullable INT64 array.
 */
array_info* convert_datetime_ns_to_us(array_info* in_arr) {
    tracing::Event ev("convert_datetime_ns_to_us");
    const uint64_t nRow = in_arr->length;
    ev.add_attribute("nRows", nRow);

    if (in_arr->dtype != Bodo_CTypes::DATETIME) {
        throw std::runtime_error(
            "Unsupported dtype type '" + GetDtype_as_string(in_arr->dtype) +
            "' provided to convert_datetime_ns_to_us. Expected "
            "Bodo_CTypes::DATETIME array.");
    }

    if (in_arr->arr_type != bodo_array_type::NUMPY) {
        throw std::runtime_error(
            "Unsupported arr_type '" + GetArrType_as_string(in_arr->arr_type) +
            "' provided to convert_datetime_ns_to_us. Expected "
            "bodo_array_type::NUMPY.");
    }

    array_info* out_arr = alloc_nullable_array(nRow, Bodo_CTypes::INT64, 0);
    for (uint64_t i = 0; i < nRow; i++) {
        if (in_arr->at<int64_t>(i) == std::numeric_limits<int64_t>::min()) {
            // if value is NaT (equals
            // std::numeric_limits<int64_t>::min()) we set it as
            // null in the transformed array.
            out_arr->set_null_bit(i, false);
            // Set it to -1 to avoid non-deterministic behavior (e.g. during
            // hashing)
            out_arr->at<int64_t>(i) = -1;
        } else {
            // convert from nanoseconds to microseconds
            out_arr->set_null_bit(i, true);
            out_arr->at<int64_t>(i) = in_arr->at<int64_t>(i) / 1000;
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
 * @brief Get the years since epoch from date object
 *
 * @param dt DATE (as int64)
 * @return int32_t Number of years since epoch
 */
static inline int32_t get_years_since_epoch_from_date(int64_t* dt) {
    return (*dt >> 32) - 1970;
}

/**
 * @brief Get the days since epoch from date object
 * Same as bodo_date64_to_arrow_date32 in parque_write.cpp
 *
 * @param dt DATE (as int64)
 * @return int64_t Number of days since epoch
 */
static inline int64_t get_days_since_epoch_from_date(int64_t* dt) {
    int64_t year = *dt >> 32;
    int64_t month = (*dt >> 16) & 0xFFFF;
    int64_t day = *dt & 0xFFFF;
    return get_days_from_date(year, month, day);
}

/**
 * @brief Convert a DATE array (NULLABLE_INT) into a nullable int64 array with
 * days since epoch.
 *
 * @param in_arr DATE array to transform
 * @return array_info* int64 array containing days since epoch for each DATE
 * element.
 */
array_info* convert_date_to_days_since_epoch(array_info* in_arr) {
    tracing::Event ev("convert_date_to_days_since_epoch");
    const uint64_t nRow = in_arr->length;
    ev.add_attribute("nRows", nRow);

    if (in_arr->dtype != Bodo_CTypes::DATE) {
        throw std::runtime_error(
            "Unsupported dtype type '" + GetDtype_as_string(in_arr->dtype) +
            "' provided to convert_date_to_days_since_epoch. Expected "
            "Bodo_CTypes::DATE array.");
    }
    if (in_arr->arr_type != bodo_array_type::NULLABLE_INT_BOOL) {
        throw std::runtime_error(
            "Unsupported arr_type '" + GetArrType_as_string(in_arr->arr_type) +
            "' provided to convert_date_to_days_since_epoch. Expected "
            "bodo_array_type::NULLABLE_INT_BOOL.");
    }

    // Iceberg Spec is unclear on whether "date" which is number
    // of days since epoch, is int32 or int64. We choose int64
    // because it works better for the bucket transform
    // (see note in array_transform_bucket_N).
    array_info* out_arr = alloc_nullable_array(nRow, Bodo_CTypes::INT64, 0);
    for (uint64_t i = 0; i < nRow; i++) {
        out_arr->at<int64_t>(i) =
            get_days_since_epoch_from_date(&in_arr->at<int64_t>(i));
    }
    int64_t n_bytes = ((nRow + 7) >> 3);
    // Copy the null bitmask as is
    memcpy(out_arr->null_bitmask, in_arr->null_bitmask, n_bytes);
    return out_arr;
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
    int64_t rem_months;
    get_remaining_months(year, &days, &rem_months);
    // Compute months from 1970-01-01.
    return ((year - 1970) * 12) + rem_months;
}

/**
 * @brief Get the months since epoch from DATE object (represented as int64)
 *
 * @param dt DATE object (int64)
 * @return int32_t Number of months since epoch for the provided DATE object
 */
static inline int32_t get_months_since_epoch_from_date(int64_t* dt) {
    int32_t year = (*dt >> 32);
    int32_t rem_months = ((*dt >> 16) & 0xFFFF) - 1;
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

array_info* array_transform_bucket_N(array_info* in_arr, int64_t N,
                                     bool is_parallel) {
    assert(N > 0);
    tracing::Event ev("array_transform_bucket_N", is_parallel);
    const uint64_t nRow = in_arr->length;
    ev.add_attribute("nRows", nRow);

    array_info* out_arr = alloc_nullable_array(nRow, Bodo_CTypes::UINT32, 0);
    uint32_t* hashes = (uint32_t*)out_arr->data1;

    int64_t n_bytes = ((nRow + 7) >> 3);

    // We calculate DICT case separately since `hash_array`
    // hashes the indices instead of the actual
    // strings, which doesn't work for Iceberg.
    if (in_arr->arr_type == bodo_array_type::DICT) {
        // Calculate hashes on the dict
        uint32_t* dict_hashes = new uint32_t[in_arr->info1->length];
        hash_array(dict_hashes, in_arr->info1, in_arr->info1->length, 0,
                   is_parallel, false, /*use_murmurhash=*/true);
        // Iterate over the elements and assign hash from the dict
        for (uint64_t i = 0; i < nRow; i++) {
            if (!in_arr->info2->get_null_bit(i)) {
                // in case index i is null, in_arr->info2->at<uint32_t>(i)
                // is a garbage value and might be out of bounds for
                // dict_hashes, so we set the hash to -1 to avoid access errors.
                hashes[i] = -1;
            } else {
                hashes[i] = dict_hashes[in_arr->info2->at<uint32_t>(i)];
            }
        }
        // Copy the null bitmask from the indices array
        memcpy(out_arr->null_bitmask, in_arr->info2->null_bitmask, n_bytes);
    } else {
        // hash_array doesn't hash nulls
        // as we need. In particular, we need null
        // to hash to null. Hence, we copied over
        // the null bitmask.
        if (in_arr->dtype == Bodo_CTypes::DATETIME) {
            // In case of datetime, first convert to microsecond (as int64 which
            // is important for correctness) and then hash
            array_info* us_array = convert_datetime_ns_to_us(in_arr);
            // DATETIME arrays are always NUMPY so there's no null_bitmask
            // in in_arr. The array could have NaTs though, which
            // convert_datetime_ns_to_us will convert to nulls, so we can
            // use the bitmask from us_array as is for the out_arr.
            memcpy(out_arr->null_bitmask, us_array->null_bitmask, n_bytes);
            hash_array(hashes, us_array, nRow, 0, is_parallel, false,
                       /*use_murmurhash=*/true);
            decref_array(us_array);
            delete us_array;
        } else if (in_arr->dtype == Bodo_CTypes::DATE) {
            // For date, we need to convert to days from epoch before hashing.
            // Based on
            // https://iceberg.apache.org/spec/#appendix-b-32-bit-hash-requirements,
            // dates should be hashed as int32 (after computing number of days
            // since epoch), however Spark, Iceberg-Python and example in
            // Iceberg Spec seems to use int64.
            array_info* days_since_epoch_array =
                convert_date_to_days_since_epoch(in_arr);
            hash_array(hashes, days_since_epoch_array, nRow, 0, is_parallel,
                       false,
                       /*use_murmurhash=*/true);
            // DATE arrays are always NULLABLE_INT_BOOL arrays, so we can copy
            // over the null_bitmask as is. convert_date_to_days_since_epoch
            // also copies over the null_bitmask, so it's the same.
            memcpy(out_arr->null_bitmask, in_arr->null_bitmask, n_bytes);
            decref_array(days_since_epoch_array);
            delete days_since_epoch_array;
        } else if (in_arr->dtype == Bodo_CTypes::INT32) {
            // in case of int32, we need to cast to int64 before hashing
            // (https://iceberg.apache.org/spec/#appendix-b-32-bit-hash-requirements)
            array_info* int64_arr = alloc_array(
                in_arr->length, in_arr->n_sub_elems, in_arr->n_sub_sub_elems,
                in_arr->arr_type, Bodo_CTypes::INT64, 0, 0);
            if (in_arr->null_bitmask) {
                // Copy the null bitmask if it exists for the arr type
                memcpy(int64_arr->null_bitmask, in_arr->null_bitmask, n_bytes);
                // Copy it over to the out_arr null_bitmask as well
                memcpy(out_arr->null_bitmask, in_arr->null_bitmask, n_bytes);
            } else {
                // In case this array type (NUMPY) doesn't have a null_bitmask
                // there should be no nulls in the array and we can set
                // null_bitmask to all 1s.
                memset(out_arr->null_bitmask, 0xFF, n_bytes);
            }
            for (uint64_t i = 0; i < nRow; i++) {
                int64_arr->at<int64_t>(i) = (int64_t)in_arr->at<int32_t>(i);
            }
            hash_array(hashes, int64_arr, nRow, 0, is_parallel, false,
                       /*use_murmurhash=*/true);
            decref_array(int64_arr);
            delete int64_arr;
        } else {
            hash_array(hashes, in_arr, nRow, 0, is_parallel, false,
                       /*use_murmurhash=*/true);
            if (in_arr->null_bitmask) {
                // Copy the null bitmask if it exists for the arr type
                memcpy(out_arr->null_bitmask, in_arr->null_bitmask, n_bytes);
            } else {
                // Otherwise set it to all 1s.
                // DATETIME which uses a NUMPY array (but can have NaTs)
                // is handled separately above, and floats (which can have NaNs)
                // are not supported for the bucket transform at all, so this
                // should be safe.
                memset(out_arr->null_bitmask, 0xFF, n_bytes);
            }
        }
    }

    for (uint64_t i = 0; i < nRow; i++) {
        hashes[i] = (hashes[i] & INT_MAX) % N;
    }

    return out_arr;
}

template <typename T>
static inline T int_truncate_helper(T& v, int64_t W) {
    return v - (((v % W) + W) % W);
}

array_info* array_transform_truncate_W(array_info* in_arr, int64_t width,
                                       bool is_parallel) {
    assert(width > 0);
    tracing::Event ev("array_transform_truncate_W", is_parallel);
    ev.add_attribute("W", width);
    const uint64_t nRow = in_arr->length;
    ev.add_attribute("nRows", nRow);

    array_info* out_arr;
    if (in_arr->arr_type == bodo_array_type::STRING) {
        offset_t* in_offsets = (offset_t*)in_arr->data2;
        offset_t str_len;
        uint64_t n_chars = 0;
        // Calculate the total chars first
        for (uint64_t i = 0; i < nRow; i++) {
            if (!in_arr->get_null_bit(i)) {
                continue;
            }
            str_len = in_offsets[i + 1] - in_offsets[i];
            n_chars += std::min(str_len, (offset_t)width);
        }
        // Allocate output array
        out_arr = alloc_string_array(nRow, n_chars, 0);
        // Copy over truncated strings to the new array
        offset_t* out_offsets = (offset_t*)out_arr->data2;
        out_offsets[0] = 0;
        for (uint64_t i = 0; i < nRow; i++) {
            if (!in_arr->get_null_bit(i)) {
                out_arr->set_null_bit(i, false);
                out_offsets[i + 1] = out_offsets[i];
                continue;
            }
            str_len = in_offsets[i + 1] - in_offsets[i];
            str_len = std::min(str_len, (offset_t)width);
            memcpy(out_arr->data1 + out_offsets[i],
                   in_arr->data1 + in_offsets[i], sizeof(char) * str_len);
            out_offsets[i + 1] = out_offsets[i] + str_len;
            out_arr->set_null_bit(i, true);
        }
        return out_arr;
    }
    if (in_arr->arr_type == bodo_array_type::NUMPY) {
        if (in_arr->dtype == Bodo_CTypes::INT32) {
            out_arr = alloc_numpy(nRow, in_arr->dtype);
            for (uint64_t i = 0; i < nRow; i++) {
                out_arr->at<int32_t>(i) =
                    int_truncate_helper<int32_t>(in_arr->at<int32_t>(i), width);
            }
            return out_arr;
        }
        if (in_arr->dtype == Bodo_CTypes::INT64) {
            out_arr = alloc_numpy(nRow, in_arr->dtype);
            for (uint64_t i = 0; i < nRow; i++) {
                out_arr->at<int64_t>(i) =
                    int_truncate_helper<int64_t>(in_arr->at<int64_t>(i), width);
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
                if (!in_arr->get_null_bit(i)) {
                    out_arr->set_null_bit(i, false);
                    continue;
                }
                out_arr->at<int32_t>(i) =
                    int_truncate_helper<int32_t>(in_arr->at<int32_t>(i), width);
                out_arr->set_null_bit(i, true);
            }
            return out_arr;
        }
        if (in_arr->dtype == Bodo_CTypes::INT64) {
            out_arr = alloc_nullable_array(nRow, in_arr->dtype, 0);
            for (uint64_t i = 0; i < nRow; i++) {
                if (!in_arr->get_null_bit(i)) {
                    out_arr->set_null_bit(i, false);
                    continue;
                }
                out_arr->at<int64_t>(i) =
                    int_truncate_helper<int64_t>(in_arr->at<int64_t>(i), width);
                out_arr->set_null_bit(i, true);
            }
            return out_arr;
        }
    }
    if (in_arr->arr_type == bodo_array_type::DICT) {
        array_info* dict_data_arr = in_arr->info1;
        // Call recursively on the dictionary
        array_info* trunc_dict_data_arr =
            array_transform_truncate_W(dict_data_arr, width, is_parallel);
        // Make a copy of the indices
        array_info* indices_copy = copy_array(in_arr->info2);
        // Note that if the input dictionary was sorted, the
        // transformed dictionary would be sorted too since
        // it's just a truncation. It might not have all
        // unique elements though.
        out_arr = new array_info(
            bodo_array_type::DICT, in_arr->dtype, in_arr->length, -1, -1, NULL,
            NULL, NULL, indices_copy->null_bitmask, NULL, NULL, NULL, NULL, 0,
            0, 0, in_arr->has_global_dictionary, in_arr->has_sorted_dictionary,
            trunc_dict_data_arr, indices_copy);
        return out_arr;
    }
    // Throw an error if type is not supported (e.g. CATEGORICAL)
    std::string err = "array_transform_truncate_W not supported.";
    throw std::runtime_error(err);
}

array_info* array_transform_void(array_info* in_arr, bool is_parallel) {
    tracing::Event ev("array_transform_void", is_parallel);
    // We simply return a nullable int32 array
    // with all nulls (as per the Iceberg spec)
    return alloc_nullable_array_all_nulls(in_arr->length, Bodo_CTypes::INT32,
                                          0);
}

array_info* array_transform_year(array_info* in_arr, bool is_parallel) {
    tracing::Event ev("array_transform_year", is_parallel);
    const uint64_t nRow = in_arr->length;
    ev.add_attribute("nRows", nRow);
    array_info* out_arr = alloc_nullable_array(nRow, Bodo_CTypes::INT32, 0);

    int64_t n_bytes = ((nRow + 7) >> 3);

    if ((in_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) &&
        (in_arr->dtype == Bodo_CTypes::DATE)) {
        for (uint64_t i = 0; i < nRow; i++) {
            out_arr->at<int32_t>(i) =
                get_years_since_epoch_from_date(&in_arr->at<int64_t>(i));
        }
        // Copy the null bitmask as is
        memcpy(out_arr->null_bitmask, in_arr->null_bitmask, n_bytes);
        return out_arr;
    }
    if ((in_arr->arr_type == bodo_array_type::NUMPY) &&
        (in_arr->dtype == Bodo_CTypes::DATETIME)) {
        for (uint64_t i = 0; i < nRow; i++) {
            if (in_arr->at<int64_t>(i) == std::numeric_limits<int64_t>::min()) {
                // if value is NaT (equals
                // std::numeric_limits<int64_t>::min()) we set it as
                // null in the transformed array.
                out_arr->set_null_bit(i, false);
                // Set it to -1 to avoid non-deterministic behavior (e.g. during
                // hashing)
                out_arr->at<int32_t>(i) = -1;
            } else {
                out_arr->at<int32_t>(i) = get_years_since_epoch_from_datetime(
                    &in_arr->at<int64_t>(i));
                out_arr->set_null_bit(i, true);
            }
        }
        return out_arr;
    }
    // Throw an error if type is not supported.
    std::string err = "array_transform_year not supported.";
    throw std::runtime_error(err);
}

array_info* array_transform_month(array_info* in_arr, bool is_parallel) {
    tracing::Event ev("array_transform_month", is_parallel);
    const uint64_t nRow = in_arr->length;
    ev.add_attribute("nRows", nRow);
    array_info* out_arr = alloc_nullable_array(nRow, Bodo_CTypes::INT32, 0);
    int64_t n_bytes = ((nRow + 7) >> 3);

    if ((in_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) &&
        (in_arr->dtype == Bodo_CTypes::DATE)) {
        for (uint64_t i = 0; i < nRow; i++) {
            out_arr->at<int32_t>(i) =
                get_months_since_epoch_from_date(&in_arr->at<int64_t>(i));
        }
        // Copy the null bitmask as is
        memcpy(out_arr->null_bitmask, in_arr->null_bitmask, n_bytes);
        return out_arr;
    }

    if ((in_arr->arr_type == bodo_array_type::NUMPY) &&
        (in_arr->dtype == Bodo_CTypes::DATETIME)) {
        for (uint64_t i = 0; i < nRow; i++) {
            if (in_arr->at<int64_t>(i) == std::numeric_limits<int64_t>::min()) {
                // if value is NaT (equals
                // std::numeric_limits<int64_t>::min()) we set it as
                // null in the transformed array.
                out_arr->set_null_bit(i, false);
                // Set it to -1 to avoid non-deterministic behavior (e.g. during
                // hashing)
                out_arr->at<int32_t>(i) = -1;
            } else {
                out_arr->at<int32_t>(i) =
                    (int32_t)get_months_since_epoch_from_datetime(
                        &in_arr->at<int64_t>(i));
                out_arr->set_null_bit(i, true);
            }
        }
        return out_arr;
    }

    // Throw an error if type is not supported.
    std::string err = "array_transform_month not supported.";
    throw std::runtime_error(err);
}

array_info* array_transform_day(array_info* in_arr, bool is_parallel) {
    tracing::Event ev("array_transform_day", is_parallel);
    const uint64_t nRow = in_arr->length;
    ev.add_attribute("nRows", nRow);
    array_info* out_arr = alloc_nullable_array(nRow, Bodo_CTypes::INT64, 0);

    int64_t n_bytes = ((nRow + 7) >> 3);

    if ((in_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) &&
        (in_arr->dtype == Bodo_CTypes::DATE)) {
        for (uint64_t i = 0; i < nRow; i++) {
            out_arr->at<int64_t>(i) =
                get_days_since_epoch_from_date(&in_arr->at<int64_t>(i));
        }
        // Copy the null bitmask as is
        memcpy(out_arr->null_bitmask, in_arr->null_bitmask, n_bytes);
        return out_arr;
    }

    if ((in_arr->arr_type == bodo_array_type::NUMPY) &&
        (in_arr->dtype == Bodo_CTypes::DATETIME)) {
        for (uint64_t i = 0; i < nRow; i++) {
            if (in_arr->at<int64_t>(i) == std::numeric_limits<int64_t>::min()) {
                // if value is NaT (equals
                // std::numeric_limits<int64_t>::min()) we set it as
                // null in the transformed array.
                out_arr->set_null_bit(i, false);
                // Set it to -1 to avoid non-deterministic behavior (e.g. during
                // hashing)
                out_arr->at<int64_t>(i) = -1;
            } else {
                out_arr->at<int64_t>(i) =
                    get_days_since_epoch_from_datetime(&in_arr->at<int64_t>(i));
                out_arr->set_null_bit(i, true);
            }
        }
        return out_arr;
    }

    // Throw an error if type is not supported.
    std::string err = "array_transform_day not supported.";
    throw std::runtime_error(err);
}

array_info* array_transform_hour(array_info* in_arr, bool is_parallel) {
    tracing::Event ev("array_transform_hour", is_parallel);
    const uint64_t nRow = in_arr->length;
    ev.add_attribute("nRows", nRow);

    array_info* out_arr = alloc_nullable_array(nRow, Bodo_CTypes::INT32, 0);

    if ((in_arr->arr_type == bodo_array_type::NUMPY) &&
        (in_arr->dtype == Bodo_CTypes::DATETIME)) {
        for (uint64_t i = 0; i < nRow; i++) {
            if (in_arr->at<int64_t>(i) == std::numeric_limits<int64_t>::min()) {
                // if value is NaT (equals
                // std::numeric_limits<int64_t>::min()) we set it as
                // null in the transformed array.
                out_arr->set_null_bit(i, false);
                // Set it to -1 to avoid non-deterministic behavior (e.g. during
                // hashing)
                out_arr->at<int32_t>(i) = -1;
            } else {
                out_arr->at<int32_t>(i) = get_hours_since_epoch_from_datetime(
                    &in_arr->at<int64_t>(i));
                out_arr->set_null_bit(i, true);
            }
        }
        return out_arr;
    }

    // Throw an error if type is not supported.
    std::string err = "array_transform_hour not supported.";
    throw std::runtime_error(err);
}

array_info* iceberg_identity_transform(array_info* in_arr, bool* new_alloc,
                                       bool is_parallel) {
    tracing::Event ev("iceberg_identity_transform", is_parallel);
    const uint64_t nRow = in_arr->length;
    ev.add_attribute("nRows", nRow);

    if (in_arr->dtype == Bodo_CTypes::DATETIME) {
        // For paritioning, we need the partition value to be in microseconds
        // instead of the default which is nanoseconds. Note that it is
        // important due to precision issues during partitioning. e.g. two
        // different nanosecond values could correspond to the same microsecond
        // value, so during partitioning it's important for them to be in the
        // same partition.
        // For sorting it would be fine to keep it in
        // nanoseconds, but we are not optimizing that at this time.
        *new_alloc = true;
        return convert_datetime_ns_to_us(in_arr);
    } else if (in_arr->dtype == Bodo_CTypes::DATE) {
        // For partitioning, we need the partition value to be the number
        // of days since epoch, instead of the bit-style DATE representation.
        // XXX TODO Since we seem to need to return a string anyway, maybe we
        // don't need this anymore?
        *new_alloc = true;
        return convert_date_to_days_since_epoch(in_arr);
    } else {
        *new_alloc = false;
        return in_arr;  // return as is
    }
}

array_info* iceberg_transform(array_info* in_arr, std::string transform_name,
                              int64_t arg, bool is_parallel) {
    if (transform_name == "bucket")
        return array_transform_bucket_N(in_arr, arg, is_parallel);
    if (transform_name == "truncate")
        return array_transform_truncate_W(in_arr, arg, is_parallel);
    if (transform_name == "void")
        return array_transform_void(in_arr, is_parallel);
    if (transform_name == "year")
        return array_transform_year(in_arr, is_parallel);
    if (transform_name == "month")
        return array_transform_month(in_arr, is_parallel);
    if (transform_name == "day")
        return array_transform_day(in_arr, is_parallel);
    if (transform_name == "hour")
        return array_transform_hour(in_arr, is_parallel);

    throw std::runtime_error("Unrecognized transform '" + transform_name +
                             "' provided.");
}

std::string transform_val_to_str(std::string transform_name, array_info* in_arr,
                                 array_info* transformed_arr, size_t idx) {
    if (transformed_arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL ||
        transformed_arr->arr_type == bodo_array_type::STRING ||
        transformed_arr->arr_type == bodo_array_type::DICT) {
        if (!transformed_arr->get_null_bit(idx)) {
            return std::string("null");
        }
    }

    if (transform_name == "year")
        return std::to_string(transformed_arr->at<int32_t>(idx) + 1970);
    if ((transform_name == "identity" || transform_name == "day") &&
        in_arr->dtype == Bodo_CTypes::DATE)
        return in_arr->val_to_str(idx);
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

        int64_t rem_months;
        get_remaining_months(year, &day, &rem_months);
        int64_t month = rem_months + 1;
        day += 1;

        std::string date_str;
        date_str.reserve(10);
        date_str += std::to_string(year) + "-";
        if (month < 10) date_str += "0";
        date_str += std::to_string(month) + "-";
        if (day < 10) date_str += "0";
        date_str += std::to_string(day);
        return date_str;
    }
    return transformed_arr->val_to_str(idx);
}

PyObject* iceberg_transformed_val_to_py(array_info* arr, size_t idx) {
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
            if (arr->at<bool>(idx)) return Py_True;
            return Py_False;

        case Bodo_CTypes::STRING: {
            switch (arr->arr_type) {
                case bodo_array_type::DICT: {
                    int32_t dict_idx = arr->info2->at<int32_t>(idx);
                    offset_t* offsets = (offset_t*)arr->info1->data2;
                    return PyUnicode_FromString(
                        std::string(arr->info1->data1 + offsets[dict_idx],
                                    offsets[dict_idx + 1] - offsets[dict_idx])
                            .c_str());
                }
                default: {
                    offset_t* offsets = (offset_t*)arr->data2;
                    return PyUnicode_FromString(
                        std::string(arr->data1 + offsets[idx],
                                    offsets[idx + 1] - offsets[idx])
                            .c_str());
                }
            }
        }

        default: {
            std::vector<char> error_msg(100);
            sprintf(
                error_msg.data(),
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
