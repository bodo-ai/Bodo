/**
 * @file _lead_lag.cpp
 * @author Benjamin Owad (benjamin@bodo.ai)
 * @brief Implementation for lead/lag SQL functions.
 *
 * @copyright Copyright (C) 2023 Bodo Inc. All rights reserved.
 *
 */
#include "_lead_lag.h"

#include <algorithm>
#include <concepts>
#include "_bodo_common.h"

/*
 * Search for next non-null value shift_amt away from write_i,
 * updating values_between in the process.
 *
 * Returns new values for read_i and values_between in a tuple.
 */
template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType,
          typename T>
inline std::tuple<int64_t, int64_t> search_for_next_ign_nulls_val(
    const std::shared_ptr<array_info> &in_col, int64_t read_i,
    int64_t values_between, int64_t shift_amt_abs, int64_t step, int64_t n) {
    // If write_i isn't shift_amt non-null values away from read_i,
    // then we need to search until we are shift_amt non-null values away.
    while (values_between < shift_amt_abs) {
        read_i += step;
        if (read_i >= 0 && read_i < n) {
            values_between += non_null_at<ArrType, T, DType>(*in_col, read_i);
        } else {
            break;
        }
    }
    return std::make_tuple(std::move(read_i), std::move(values_between));
}

/*
 * Copies value or null from in_arr at read_i to out_arr at write_i.
 */
template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType,
          bool ignore_nulls, typename T>
inline void copy_value(const std::shared_ptr<array_info> &in_col,
                       const std::unique_ptr<array_info> &out_col,
                       int64_t read_i, int64_t write_i) {
    // If we are in a nullable array and respecting nulls (i.e. is this case
    // possible)
    if constexpr (ArrType == bodo_array_type::NULLABLE_INT_BOOL &&
                  !ignore_nulls) {
        // If we have a null at read_i
        if (is_null_at<ArrType, T, DType>(*in_col, read_i)) {
            // Copy it to the position at write_i
            out_col->set_null_bit<ArrType>(write_i, false);
            return;
        }
    }
    // Otherwise, copy the value itself.
    T new_value = get_arr_item<ArrType, T, DType>(*in_col, read_i);
    set_arr_item<bodo_array_type::NULLABLE_INT_BOOL, T, DType>(
        *out_col, write_i, new_value);
}

/*
 * If we have a default value, copy it into the specified position in the
 * out_col. No default value? No problem! Copy a null instead.
 */
template <Bodo_CTypes::CTypeEnum DType, bool has_default, typename T>
inline void handle_out_of_bounds(const std::unique_ptr<array_info> &out_col,
                                 int64_t write_i,
                                 std::optional<T> default_fill_val) {
    if constexpr (has_default) {
        set_arr_item<bodo_array_type::NULLABLE_INT_BOOL, T, DType>(
            *out_col, write_i, *default_fill_val);
    } else {
        out_col->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(write_i,
                                                                  false);
    }
}

template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType,
          bool ignore_nulls, bool has_default, typename T>
inline void handle_shift(const std::shared_ptr<array_info> &in_col,
                         const std::unique_ptr<array_info> &out_col,
                         int64_t &write_i, const int64_t n, const int step,
                         const int64_t shift_amt, const int64_t shift_amt_abs,
                         const bool loop_backwards,
                         const std::optional<T> &default_fill_val) {
    // Cursor referring to position in the input that we want to copy *from*.
    // In the respect nulls case, start in exactly the right
    // place, where it will remain for the duration of the function.
    int64_t read_i = write_i - shift_amt;

    // Handle entries in bounds, copying values to their new positions in
    // out_col
    while (read_i >= 0 && read_i < n) {
        // Copy the values
        copy_value<ArrType, DType, ignore_nulls, T>(in_col, out_col, read_i,
                                                    write_i);
        // Increment write_i and read_i
        write_i += step;
        read_i += step;
    }
    // Handle out of bounds entries after our copy range.
    for (int i = 0; i < shift_amt_abs; i++) {
        handle_out_of_bounds<DType, has_default, T>(out_col, write_i,
                                                    default_fill_val);
        write_i += step;
    }
}

template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType,
          bool ignore_nulls, bool has_default, typename T>
    requires(ignore_nulls)
inline void handle_shift(const std::shared_ptr<array_info> &in_col,
                         const std::unique_ptr<array_info> &out_col,
                         int64_t &write_i, const int64_t n, const int step,
                         const int64_t shift_amt, const int64_t shift_amt_abs,
                         const bool loop_backwards,
                         const std::optional<T> &default_fill_val) {
    // Count of how many non-null values exist between our two cursors (read_i
    // and write_i). Excludes write_i, includes read_i
    int64_t values_between = 0;

    // Cursor referring to position in the input that we want to copy *from*.
    // In the ignore nulls case, start out at write_i.
    // Then, search until we are shift_amt non-null values away.
    int64_t read_i = write_i;
    std::tie(read_i, values_between) =
        search_for_next_ign_nulls_val<ArrType, DType, T>(
            in_col, read_i, values_between, shift_amt_abs, step, n);

    // Handle entries in bounds, copying values to their new positions in
    // out_col
    while (read_i >= 0 && read_i < n) {
        // Copy the values
        copy_value<ArrType, DType, ignore_nulls, T>(in_col, out_col, read_i,
                                                    write_i);
        // Increment write_i
        write_i += step;

        // Subtract non-null values exiting the range (write_i exclusive, so
        // subtract if *new* write_i non-null)
        values_between -= non_null_at<ArrType, T, DType>(*in_col, write_i);

        // Search for next read_i position, ignoring nulls
        std::tie(read_i, values_between) =
            search_for_next_ign_nulls_val<ArrType, DType, T>(
                in_col, read_i, values_between, shift_amt_abs, step, n);
    }
    // Handle any remaining entries after our copy range.
    while (write_i < n && write_i >= 0) {
        handle_out_of_bounds<DType, has_default, T>(out_col, write_i,
                                                    default_fill_val);
        write_i += step;
    }
}

/*
 * Standalone sequential implementation of lead/lag, for the general case
 * (should work with in_col of nullable or numpy arr_types)
 */
template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType,
          bool ignore_nulls, bool has_default, typename T>
    requires std::same_as<T, typename dtype_to_type<DType>::type>
std::unique_ptr<array_info> lead_lag_seq(
    const std::shared_ptr<array_info> &in_col, int64_t shift_amt,
    const std::optional<T> &default_fill_val,
    int64_t default_fill_val_len = 0) {
    // Note slight precision loss here from uint64_t
    const int64_t n = static_cast<int64_t>(in_col->length);

    // Maximum shift_amt is n
    shift_amt = std::clamp(shift_amt, -n, n);
    const int64_t shift_amt_abs = std::abs(shift_amt);
    // We want to loop "backwards," right to left if our shift value is positive
    const bool loop_backwards = shift_amt_abs == shift_amt;
    int step = loop_backwards ? -1 : 1;

    // Allocate output array as nullable array with every element non-null
    std::unique_ptr<array_info> out_col =
        alloc_nullable_array_no_nulls(n, in_col->dtype, 0);

    // Cursor referring to position in the output that we want to copy *to*.
    int64_t write_i = loop_backwards ? n - 1 : 0;

    handle_shift<ArrType, DType, ignore_nulls, has_default, T>(
        in_col, out_col, write_i, n, step, shift_amt, shift_amt_abs,
        loop_backwards, default_fill_val);

    return out_col;
}

/*
 * Copies offsets from read_i to write_i, incrementing allocation_size in the
 * process.
 */
template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType,
          bool ignore_nulls, typename T>
inline void copy_offsets(
    const std::shared_ptr<array_info> &in_col,
    std::vector<std::optional<std::pair<offset_t, offset_t>>> &out_offsets,
    int64_t read_i, int64_t write_i, offset_t &allocation_size) {
    assert(in_col->arr_type == ArrType);
    // If we have a null, mark a null to copy in this location.
    if (is_null_at<ArrType, T, DType>(*in_col, read_i)) {
        return;
    }

    const auto in_offsets = (offset_t *)in_col->data2<ArrType>();

    // Find the beginning and end of the string to copy
    const offset_t start_offset = in_offsets[read_i];
    const offset_t end_offset = in_offsets[read_i + 1];
    const offset_t len = end_offset - start_offset;

    // Mark its offsets for the copy later
    out_offsets[write_i] = {start_offset, end_offset};
    allocation_size += len;
}

/*
 * If we have a default value, add a pair of negative ones to our offsets here
 * (which indicates a default value by my self-defined convention). Otherwise,
 * do nothing since std::nullopt is the default.
 */
template <bool has_default>
inline void handle_out_of_bounds_str(
    std::vector<std::optional<std::pair<offset_t, offset_t>>> &out_offsets,
    int64_t write_i, offset_t default_fill_val_len, offset_t &allocation_size) {
    if constexpr (has_default) {
        out_offsets[write_i] = {-1, -1};
        allocation_size += default_fill_val_len;
    }
}

template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType,
          bool ignore_nulls, bool has_default, typename T>
inline void handle_shift_offsets(
    const std::shared_ptr<array_info> &in_col,
    std::vector<std::optional<std::pair<offset_t, offset_t>>>
        &offsets_to_copy_from,
    int64_t &write_i, offset_t &allocation_size, const int64_t n,
    const int step, const int64_t shift_amt, const int64_t shift_amt_abs,
    const bool loop_backwards, const offset_t default_fill_val_len) {
    // Cursor referring to position in the input that we want to copy *from*.
    // In the respect nulls case, start in exactly the right
    // place, where it will remain for the duration of the function.
    int64_t read_i = write_i - shift_amt;

    // Handle entries in bounds, copying offsets to their new positions
    while (read_i >= 0 && read_i < n) {
        // Copy the values
        copy_offsets<ArrType, DType, ignore_nulls, T>(
            in_col, offsets_to_copy_from, read_i, write_i, allocation_size);
        // Increment write_i and read_i
        write_i += step;
        read_i += step;
    }
    // Handle out of bounds entries after our copy range.
    for (int i = 0; i < shift_amt_abs; i++) {
        handle_out_of_bounds_str<has_default>(offsets_to_copy_from, write_i,
                                              default_fill_val_len,
                                              allocation_size);
        write_i += step;
    }
}

template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType,
          bool ignore_nulls, bool has_default, typename T>
    requires(ignore_nulls)
inline void handle_shift_offsets(
    const std::shared_ptr<array_info> &in_col,
    std::vector<std::optional<std::pair<offset_t, offset_t>>>
        &offsets_to_copy_from,
    int64_t &write_i, offset_t &allocation_size, const int64_t n,
    const int step, const int64_t shift_amt, const int64_t shift_amt_abs,
    const bool loop_backwards, const offset_t default_fill_val_len) {
    // Count of how many non-null values exist between our two cursors (read_i
    // and write_i). Excludes write_i, includes read_i
    int64_t values_between = 0;

    // Cursor referring to position in the input that we want to copy *from*.
    // In the ignore nulls case, start out at write_i.
    // Then, search until we are shift_amt non-null values away.
    int64_t read_i = write_i;
    std::tie(read_i, values_between) =
        search_for_next_ign_nulls_val<ArrType, DType, T>(
            in_col, read_i, values_between, shift_amt_abs, step, n);

    // Handle entries in bounds, copying offsets to their new positions
    while (read_i >= 0 && read_i < n) {
        // Copy the values
        copy_offsets<ArrType, DType, ignore_nulls, T>(
            in_col, offsets_to_copy_from, read_i, write_i, allocation_size);
        // Increment write_i
        write_i += step;

        // Subtract non-null values exiting the range (write_i exclusive, so
        // subtract if *new* write_i non-null)
        values_between -= non_null_at<ArrType, T, DType>(*in_col, write_i);

        // Search for next read_i position, ignoring nulls
        std::tie(read_i, values_between) =
            search_for_next_ign_nulls_val<ArrType, DType, T>(
                in_col, read_i, values_between, shift_amt_abs, step, n);
    }
    // Handle any remaining entries after our copy range.
    while (write_i < n && write_i >= 0) {
        handle_out_of_bounds_str<has_default>(offsets_to_copy_from, write_i,
                                              default_fill_val_len,
                                              allocation_size);
        write_i += step;
    }
}

/*
 * Standalone sequential implementation of lead/lag, for the string arr_type.
 */
template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType,
          bool ignore_nulls, bool has_default, typename T>
    requires((DType == Bodo_CTypes::BINARY || DType == Bodo_CTypes::STRING) &&
             string_array<ArrType> &&
             std::same_as<T, typename dtype_to_type<DType>::type>)
std::unique_ptr<array_info> lead_lag_seq(
    const std::shared_ptr<array_info> &in_col, int64_t shift_amt,
    const std::optional<T> &default_fill_val,
    int64_t default_fill_val_len = 0) {
    assert(in_col->arr_type == ArrType);
    // Note slight precision loss here from uint64_t
    const int64_t n = static_cast<int64_t>(in_col->length);

    // Maximum shift_amt is n
    shift_amt = std::clamp(shift_amt, -n, n);
    const int64_t shift_amt_abs = std::abs(shift_amt);
    // We want to loop "backwards," right to left if our shift value is positive
    const bool loop_backwards = shift_amt_abs == shift_amt;
    int step = loop_backwards ? -1 : 1;

    // Cursor referring to position in the output that we want to copy *to*.
    int64_t write_i = loop_backwards ? n - 1 : 0;

    // First, we build our offsets_to_copy_from, which will allow us to build
    // our new string array.

    // Array of offset pairs, representing offsets in the input to be
    // copied at this position in the array. If the value is std::nullopt,
    // then the output value should be null. If the value is {-1, -1}
    // (which underflows), then the output value should be the default.
    auto offsets_to_copy_from =
        std::vector<std::optional<std::pair<offset_t, offset_t>>>(n,
                                                                  std::nullopt);
    offset_t allocation_size = 0;

    handle_shift_offsets<ArrType, DType, ignore_nulls, has_default, T>(
        in_col, offsets_to_copy_from, write_i, allocation_size, n, step,
        shift_amt, shift_amt_abs, loop_backwards, default_fill_val_len);

    // Since string arrays can only be allocated in one direction, we use our
    // separately calculated offsets_to_copy_from here to build up our actual
    // string array.

    // Allocate output array
    std::unique_ptr<array_info> out_col =
        alloc_string_array(in_col->dtype, n, allocation_size);

    // Offsets from output
    const auto out_offsets =
        (offset_t *)out_col->data2<bodo_array_type::STRING>();

    // Copy strings from input to output appropriatel
    // The cursor keeps track of what offsets we are copying *to* in the output
    offset_t cursor = 0;
    for (int i = 0; i < n; i++) {
        // Subtle but integral piece here--sets the first offset to the current
        // cursor position always. Remember, cursor starts as 0. Next iteration
        // it'll be the length of the first string minus one, which corresponds
        // to the end of the first string and the start of the second string.
        out_offsets[i] = cursor;
        if (offsets_to_copy_from[i].has_value()) {
            auto &[start, end] = offsets_to_copy_from[i].value();
            // {-1, -1} case
            if (start == std::numeric_limits<offset_t>::max() &&
                end == std::numeric_limits<offset_t>::max()) {
                // We copy the default value in!
                memcpy(out_col->data1<bodo_array_type::STRING>() + cursor,
                       *default_fill_val, default_fill_val_len);

                // ...and increment our cursor
                cursor += default_fill_val_len;

                continue;
            }

            const offset_t len = end - start;

            // Copy the actual value from the appropriate offests
            memcpy(out_col->data1<bodo_array_type::STRING>() + cursor,
                   in_col->data1<ArrType>() + start, len);

            // Increment our copy cursor by the amount of data we have added
            // to keep track of the offsets.
            cursor += len;
        } else {
            // Nullopt case--corresponds to a null value
            out_col->set_null_bit<bodo_array_type::STRING>(i, false);
        }
    }

    out_offsets[n] = cursor;

    return out_col;
}

/*
 * Sequential implementation of lead/lag specifically for dictionary-encoded
 * strings, making use of the general implementation as a helper.
 *
 * In the case where we have no default value,
 * we simply shift the offsets using the general implementation and construct
 * our new dictionary encoded array.
 *
 * In the case where we do have a default value, we have two cases--if it is
 * already in our data array, then we can proceed as usual and shift the offsets
 * with the same data. If the default string is not present in our existing data
 * array, then we allocate a new data array with the default string appended to
 * the old data.
 */
template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType,
          bool ignore_nulls, bool has_default, typename T>
    requires((DType == Bodo_CTypes::BINARY || DType == Bodo_CTypes::STRING) &&
             dict_array<ArrType> &&
             std::same_as<T, typename dtype_to_type<DType>::type>)
std::unique_ptr<array_info> lead_lag_seq(
    const std::shared_ptr<array_info> &in_col, int64_t shift_amt,
    const std::optional<T> &default_fill_val,
    int64_t default_fill_val_len = 0) {
    assert(in_col->arr_type == bodo_array_type::DICT);
    const std::shared_ptr<array_info> in_data = in_col->child_arrays[0];
    const std::shared_ptr<array_info> in_indices = in_col->child_arrays[1];

    // The index of our default value (if any) in our input data array (if
    // present)
    std::optional<int64_t> default_index = std::nullopt;

    std::shared_ptr<array_info> out_data = in_data;

    if constexpr (has_default) {
        const auto default_str = std::basic_string_view(*default_fill_val);

        const char *in_data_chars = in_data->data1<bodo_array_type::STRING>();
        const offset_t *in_data_offsets =
            (offset_t *)in_data->data2<bodo_array_type::STRING>();
        const uint64_t in_data_len = in_data->length;
        const uint64_t in_data_n_chars = in_data_offsets[in_data_len];

        // Allocate a new data array with room for one more string of the
        // default_str's length. The default is present in all paths, so we
        // maintain any global replication. However, we lose uniqueness and
        // sort ordering.
        out_data = alloc_string_array(DType, in_data_len + 1,
                                      in_data_n_chars + default_str.length(), 0,
                                      0, in_data->is_globally_replicated);
        char *out_data_chars = out_data->data1<bodo_array_type::STRING>();
        offset_t *out_data_offsets =
            (offset_t *)out_data->data2<bodo_array_type::STRING>();

        // Copy our original string data to the new array
        std::copy(in_data_chars, in_data_chars + in_data_n_chars,
                  out_data_chars);

        // copy original offsets
        std::copy(in_data_offsets, in_data_offsets + in_data_len + 1,
                  out_data_offsets);

        // copy new default to the end of the chars array
        std::copy(default_str.begin(), default_str.end(),
                  out_data_chars + in_data_n_chars);

        // Add end offset for new default
        out_data_offsets[in_data_len + 1] =
            in_data_n_chars + default_str.length();

        // Make this our default index for shifting indices
        default_index = in_data_len;
    }

    // Shift our indices with possible default value
    std::shared_ptr<array_info> out_indices =
        lead_lag_seq<bodo_array_type::arr_type_enum::NULLABLE_INT_BOOL,
                     Bodo_CTypes::CTypeEnum::INT32, ignore_nulls, has_default,
                     int32_t>(in_indices, shift_amt, default_index, 0);

    // Allocate our new dictionary array with new indices and possibly new data
    std::unique_ptr<array_info> output_dict_array =
        create_dict_string_array(out_data, out_indices);

    return output_dict_array;
}

/*
 * Simple templating wrapper to do runtime checks so that one doesn't have to
 * specify ignore_nulls and has_default template parameters.
 */
template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType,
          typename T>
inline std::unique_ptr<array_info> lead_lag_seq_wrapper(
    const std::shared_ptr<array_info> &in_col, int64_t shift_amt,
    const std::optional<T> &default_fill_val, const bool ignore_nulls,
    int64_t default_fill_val_len) {
    if (ignore_nulls) {
        if (default_fill_val.has_value()) {
            return lead_lag_seq<ArrType, DType, true, true>(
                in_col, shift_amt, default_fill_val, default_fill_val_len);
        } else {
            return lead_lag_seq<ArrType, DType, true, false>(
                in_col, shift_amt, default_fill_val, default_fill_val_len);
        }
    } else {
        if (default_fill_val.has_value()) {
            return lead_lag_seq<ArrType, DType, false, true>(
                in_col, shift_amt, default_fill_val, default_fill_val_len);
        } else {
            return lead_lag_seq<ArrType, DType, false, false>(
                in_col, shift_amt, default_fill_val, default_fill_val_len);
        }
    }
}

// For binary and string values, value of T (which is char*) is directly passed
// from Python instead of T*
template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType>
    requires(DType == Bodo_CTypes::BINARY || DType == Bodo_CTypes::STRING)
inline array_info *lead_lag_seq_py_helper(
    const std::shared_ptr<array_info> &in_col, int64_t shift_amt,
    void *default_fill_raw, int64_t default_fill_val_len, bool ignore_nulls) {
    using T = typename dtype_to_type<DType>::type;
    std::optional<T> default_fill_val = std::nullopt;

    if (default_fill_raw != nullptr) {
        const auto fill_val_casted = static_cast<T>(default_fill_raw);
        default_fill_val = std::make_optional(fill_val_casted);
    }

    return lead_lag_seq_wrapper<ArrType, DType, T>(
               in_col, shift_amt, default_fill_val, ignore_nulls,
               default_fill_val_len)
        .release();
}

/*
 * Small helper to reduce sheer quantity of duplicated code in lead_lag_seq_py.
 * Effectively converts values from C types to C++ types, namely creates
 * std::optional.
 */
template <bodo_array_type::arr_type_enum ArrType, Bodo_CTypes::CTypeEnum DType>
inline array_info *lead_lag_seq_py_helper(
    const std::shared_ptr<array_info> &in_col, int64_t shift_amt,
    void *default_fill_raw, int64_t default_fill_val_len, bool ignore_nulls) {
    using T = typename dtype_to_type<DType>::type;
    std::optional<T> default_fill_val = std::nullopt;

    if constexpr (numpy_array<ArrType> && datetime_timedelta<DType>) {
        default_fill_val = nan_val<T, DType>();
    }

    if (default_fill_raw != nullptr) {
        const auto fill_val_casted = *(static_cast<T *>(default_fill_raw));
        default_fill_val = std::make_optional(fill_val_casted);
    }

    return lead_lag_seq_wrapper<ArrType, DType, T>(
               in_col, shift_amt, default_fill_val, ignore_nulls,
               default_fill_val_len)
        .release();
}

/*
 * Helper that templates on ArrType as well as DType.
 */
template <Bodo_CTypes::CTypeEnum DType>
inline array_info *lead_lag_seq_py_array_impl(
    const std::shared_ptr<array_info> &in_col, int64_t shift_amt,
    void *default_fill_raw, int64_t default_fill_val_len, bool ignore_nulls) {
    switch (in_col->arr_type) {
        case bodo_array_type::arr_type_enum::NULLABLE_INT_BOOL: {
            return lead_lag_seq_py_helper<
                bodo_array_type::arr_type_enum::NULLABLE_INT_BOOL, DType>(
                in_col, shift_amt, default_fill_raw, default_fill_val_len,
                ignore_nulls);
        }
        case bodo_array_type::arr_type_enum::NUMPY: {
            return lead_lag_seq_py_helper<bodo_array_type::arr_type_enum::NUMPY,
                                          DType>(
                in_col, shift_amt, default_fill_raw, default_fill_val_len,
                ignore_nulls);
        }
        case bodo_array_type::arr_type_enum::STRING: {
            return lead_lag_seq_py_helper<
                bodo_array_type::arr_type_enum::STRING, DType>(
                in_col, shift_amt, default_fill_raw, default_fill_val_len,
                ignore_nulls);
        }
        case bodo_array_type::arr_type_enum::DICT: {
            return lead_lag_seq_py_helper<bodo_array_type::arr_type_enum::DICT,
                                          DType>(
                in_col, shift_amt, default_fill_raw, default_fill_val_len,
                ignore_nulls);
        }
        default: {
            throw std::runtime_error(
                "_lead_lag.cpp::lead_lag_seq_py_helper:"
                " unsupported array arr_type");
            return nullptr;
        }
    }
}

/*
 * Python entrypoint for lead_lag_seq. Obligatory dtype to templating
 * conversion.
 */
array_info *lead_lag_seq_py_entry(array_info *in_raw, int64_t shift_amt,
                                  void *default_fill_raw,
                                  int64_t default_fill_val_len,
                                  bool ignore_nulls) {
    std::shared_ptr<array_info> in_col(in_raw);
    try {
        switch (in_col->dtype) {
            case Bodo_CTypes::INT8: {
                return lead_lag_seq_py_array_impl<Bodo_CTypes::INT8>(
                    in_col, shift_amt, default_fill_raw, default_fill_val_len,
                    ignore_nulls);
            }
            case Bodo_CTypes::UINT8: {
                return lead_lag_seq_py_array_impl<Bodo_CTypes::UINT8>(
                    in_col, shift_amt, default_fill_raw, default_fill_val_len,
                    ignore_nulls);
            }
            case Bodo_CTypes::INT16: {
                return lead_lag_seq_py_array_impl<Bodo_CTypes::INT16>(
                    in_col, shift_amt, default_fill_raw, default_fill_val_len,
                    ignore_nulls);
            }
            case Bodo_CTypes::UINT16: {
                return lead_lag_seq_py_array_impl<Bodo_CTypes::UINT16>(
                    in_col, shift_amt, default_fill_raw, default_fill_val_len,
                    ignore_nulls);
            }
            case Bodo_CTypes::INT32: {
                return lead_lag_seq_py_array_impl<Bodo_CTypes::INT32>(
                    in_col, shift_amt, default_fill_raw, default_fill_val_len,
                    ignore_nulls);
            }
            case Bodo_CTypes::UINT32: {
                return lead_lag_seq_py_array_impl<Bodo_CTypes::UINT32>(
                    in_col, shift_amt, default_fill_raw, default_fill_val_len,
                    ignore_nulls);
            }
            case Bodo_CTypes::INT64: {
                return lead_lag_seq_py_array_impl<Bodo_CTypes::INT64>(
                    in_col, shift_amt, default_fill_raw, default_fill_val_len,
                    ignore_nulls);
            }
            case Bodo_CTypes::UINT64: {
                return lead_lag_seq_py_array_impl<Bodo_CTypes::UINT64>(
                    in_col, shift_amt, default_fill_raw, default_fill_val_len,
                    ignore_nulls);
            }
            case Bodo_CTypes::FLOAT32: {
                return lead_lag_seq_py_array_impl<Bodo_CTypes::FLOAT32>(
                    in_col, shift_amt, default_fill_raw, default_fill_val_len,
                    ignore_nulls);
            }
            case Bodo_CTypes::FLOAT64: {
                return lead_lag_seq_py_array_impl<Bodo_CTypes::FLOAT64>(
                    in_col, shift_amt, default_fill_raw, default_fill_val_len,
                    ignore_nulls);
            }
            case Bodo_CTypes::STRING: {
                return lead_lag_seq_py_array_impl<Bodo_CTypes::STRING>(
                    in_col, shift_amt, default_fill_raw, default_fill_val_len,
                    ignore_nulls);
            }
            case Bodo_CTypes::_BOOL: {
                return lead_lag_seq_py_array_impl<Bodo_CTypes::_BOOL>(
                    in_col, shift_amt, default_fill_raw, default_fill_val_len,
                    ignore_nulls);
            }
            case Bodo_CTypes::DECIMAL: {
                return lead_lag_seq_py_array_impl<Bodo_CTypes::DECIMAL>(
                    in_col, shift_amt, default_fill_raw, default_fill_val_len,
                    ignore_nulls);
            }
            case Bodo_CTypes::DATE: {
                return lead_lag_seq_py_array_impl<Bodo_CTypes::DATE>(
                    in_col, shift_amt, default_fill_raw, default_fill_val_len,
                    ignore_nulls);
            }
            case Bodo_CTypes::TIME: {
                return lead_lag_seq_py_array_impl<Bodo_CTypes::TIME>(
                    in_col, shift_amt, default_fill_raw, default_fill_val_len,
                    ignore_nulls);
            }
            case Bodo_CTypes::DATETIME: {
                return lead_lag_seq_py_array_impl<Bodo_CTypes::DATETIME>(
                    in_col, shift_amt, default_fill_raw, default_fill_val_len,
                    ignore_nulls);
            }
            case Bodo_CTypes::TIMEDELTA: {
                return lead_lag_seq_py_array_impl<Bodo_CTypes::TIMEDELTA>(
                    in_col, shift_amt, default_fill_raw, default_fill_val_len,
                    ignore_nulls);
            }
            case Bodo_CTypes::INT128: {
                return lead_lag_seq_py_array_impl<Bodo_CTypes::INT128>(
                    in_col, shift_amt, default_fill_raw, default_fill_val_len,
                    ignore_nulls);
            }
            case Bodo_CTypes::BINARY: {
                return lead_lag_seq_py_array_impl<Bodo_CTypes::BINARY>(
                    in_col, shift_amt, default_fill_raw, default_fill_val_len,
                    ignore_nulls);
            }
            default:
                throw std::runtime_error(
                    "_lead_lag.cpp::lead_lag_seq_py:"
                    " unsupported array dtype");
                return nullptr;
        }
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

// Initialize lead_lag_seq_py function for usage with python
PyMODINIT_FUNC PyInit_lead_lag(void) {
    PyObject *m;
    MOD_DEF(m, "lead_lag", "No docs", NULL);
    if (m == NULL) {
        return NULL;
    }

    bodo_common_init();

    SetAttrStringFromVoidPtr(m, lead_lag_seq_py_entry);

    return m;
}
