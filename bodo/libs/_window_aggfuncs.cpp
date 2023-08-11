// Copyright (C) 2023 Bodo Inc. All rights reserved.
#include "_window_aggfuncs.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_groupby_common.h"
#include "_groupby_ftypes.h"

/**
 * @brief Computes a window aggregation using the window frame
 * ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING for
 * each partition. Scans through all of the rows in the partition
 * to populate an accumulator, then performs the calculation
 * and uses the results to fill all the values in the partition.
 *
 * @param[in] in_arr: The input array whose data is being aggregated.
 * @param[in,out] out_arr: The array where the results are written to.
 * @param[in] sorted_groups: the sorted array of group numbers indicating
 * which partition the current row belongs to.
 * @param[in] sorted_idx: The mapping of sorted indices back to their
 * locations in the original table.
 */
template <Bodo_FTypes::FTypeEnum window_func,
          bodo_array_type::arr_type_enum ArrayType, typename T,
          Bodo_CTypes::CTypeEnum DType>
void window_frame_computation_no_frame(
    std::shared_ptr<array_info> in_arr, std::shared_ptr<array_info> out_arr,
    std::shared_ptr<array_info> sorted_groups,
    std::shared_ptr<array_info> sorted_idx) {
    int64_t n = in_arr->length;
    // Keep track of which group number came before so we can tell when
    // we have entered a new partition, and also keep track of the index
    // where the current group started.
    int64_t prev_group = -1;
    int64_t group_start_idx = 0;
    // Create and initialize the accumulator.
    WindowAggfunc aggfunc(in_arr, out_arr, sorted_idx);
    aggfunc
        .init<window_func, ArrayType, T, DType, window_frame_enum::NO_FRAME>();
    for (int64_t i = 0; i < n; i++) {
        // Check we have crossed into a new group (and we are not in row zero,
        // since that will always have (curr_group != prev_group) as true)
        int64_t curr_group = getv<int64_t>(sorted_groups, i);
        if (curr_group != prev_group && i > 0) {
            // Compute the value for the partition and set all of the
            // values in the range to the accumulated result, then reset
            // the accumulator.
            aggfunc.compute_partition<window_func, ArrayType, T, DType>(
                group_start_idx, i);
            aggfunc.init<window_func, ArrayType, T, DType,
                         window_frame_enum::NO_FRAME>();
            group_start_idx = i;
        }
        // Insert the current index into the accumulator.
        aggfunc.enter<window_func, ArrayType, T, DType,
                      window_frame_enum::NO_FRAME>(i);
        prev_group = curr_group;
    }
    // Repeat the aggregation/population step one more time for the final group
    aggfunc.compute_partition<window_func, ArrayType, T, DType>(group_start_idx,
                                                                n);
    // After the computation is complete for each partition, do any necessary
    // final operations (e.g. converting string vectors to proper arrays)
    aggfunc.cleanup<window_func, ArrayType, T, DType,
                    window_frame_enum::NO_FRAME>();
}

/**
 * @brief Computes a window aggregation with a sliding window frame
 * on a partition of the input array. E.g.:
 *
 * ROWS BETWEEN 10 PRECEDING AND 5 FOLLOWING
 *
 * Starts by adding the elements to the accumulator corresponding to
 * the window frame positioned about the first index of the partition.
 * Slides the frame forward with each index, inserting/removing elements
 * as they enter/exit the window frame.
 *
 * @param[in] in_arr: The input array whose data is being aggregated.
 * @param[in,out] out_arr: The array where the results are written to.
 * @param[in] sorted_idx: The mapping of sorted indices back to their
 * locations in the original table.
 * @param[in] lo: The number of indices after the current row where the
 * window frame begins (inclusive).
 * @param[in] hi: The number of indices after the current row where the
 * window frame ends (inclusive).
 * @param[in] group_start_idx: the row where the current partition begins
 * (inclusive).
 * @param[in] group_end_idx: the row where the current partition ends
 * (exclusive).
 */
template <Bodo_FTypes::FTypeEnum window_func,
          bodo_array_type::arr_type_enum ArrayType, typename T,
          Bodo_CTypes::CTypeEnum DType>
void window_frame_sliding_computation(std::shared_ptr<array_info> in_arr,
                                      std::shared_ptr<array_info> out_arr,
                                      std::shared_ptr<array_info> sorted_idx,
                                      int64_t lo, int64_t hi,
                                      const int64_t group_start_idx,
                                      const int64_t group_end_idx) {
    // Create and initialize the accumulator.
    WindowAggfunc aggfunc(in_arr, out_arr, sorted_idx);
    aggfunc
        .init<window_func, ArrayType, T, DType, window_frame_enum::SLIDING>();
    // Entering = the index where the next value to enter the window frame
    // will come from. Exiting = the index where the next value to leave
    // the window frame will come from.
    int64_t entering = group_start_idx + hi + 1;
    int64_t exiting = group_start_idx + lo;
    // Add all of the values into the accumulator that are part of the
    // sliding frame for the first row.
    for (int64_t i = std::max(group_start_idx, exiting);
         i < std::min(std::max(group_start_idx, entering), group_end_idx);
         i++) {
        aggfunc.enter<window_func, ArrayType, T, DType,
                      window_frame_enum::SLIDING>(i);
    }
    // Loop through all of the rows in the partition, each time calculating
    // the current aggregation value and storing it in the array. Add and
    // remove values from the accumulator as they come into range.
    for (int64_t i = group_start_idx; i < group_end_idx; i++) {
        // Compute the indices lower_bound and upper_bound such that
        // the value at index i corresponds to the aggregate value
        // of the input array in the range of indices [lower_bound,
        // upper_bound).
        int64_t lower_bound =
            std::min(std::max(group_start_idx, exiting), group_end_idx);
        int64_t upper_bound =
            std::min(std::max(group_start_idx, entering), group_end_idx);
        aggfunc.compute_frame<window_func, ArrayType, T, DType,
                              window_frame_enum::SLIDING>(i, lower_bound,
                                                          upper_bound);
        // If exiting is in bounds, remove that index from the accumulator
        if (group_start_idx <= exiting && exiting < group_end_idx) {
            aggfunc.exit<window_func, ArrayType, T, DType,
                         window_frame_enum::SLIDING>(exiting);
        }
        // If entering is in bounds, add that index to the accumulator
        if (group_start_idx <= entering && entering < group_end_idx) {
            aggfunc.enter<window_func, ArrayType, T, DType,
                          window_frame_enum::SLIDING>(entering);
        }
        exiting++;
        entering++;
    }
    if (static_cast<uint64_t>(group_end_idx) == in_arr->length) {
        // After the computation is complete for each partition, do any
        // necessary final operations (e.g. converting string vectors to proper
        // arrays)
        aggfunc.cleanup<window_func, ArrayType, T, DType,
                        window_frame_enum::SLIDING>();
    }
}

/**
 * @brief Computes a window aggregation with a prefix window frame
 * on a partition of the input array. E.g.:
 *
 * ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
 *
 * Starts at the start of the partition and adds elements to the accumulator
 * in ascending order without ever removing any elements.
 *
 * @param[in] in_arr: The input array whose data is being aggregated.
 * @param[in,out] out_arr: The array where the results are written to.
 * @param[in] sorted_idx: The mapping of sorted indices back to their
 * locations in the original table.
 * @param[in] hi: The number of indices after the current row that the
 * prefix frame includes (inclusive).
 * @param[in] group_start_idx: the row where the current partition begins
 * (inclusive).
 * @param[in] group_end_idx: the row where the current partition ends
 * (exclusive).
 */
template <Bodo_FTypes::FTypeEnum window_func,
          bodo_array_type::arr_type_enum ArrayType, typename T,
          Bodo_CTypes::CTypeEnum DType>
void window_frame_prefix_computation(std::shared_ptr<array_info> in_arr,
                                     std::shared_ptr<array_info> out_arr,
                                     std::shared_ptr<array_info> sorted_idx,
                                     const int64_t hi,
                                     const int64_t group_start_idx,
                                     const int64_t group_end_idx) {
    // Create and initialize the accumulator.
    WindowAggfunc aggfunc(in_arr, out_arr, sorted_idx);
    aggfunc.init<window_func, ArrayType, T, DType,
                 window_frame_enum::CUMULATIVE>();
    // Entering = the index where the next value to enter the window frame
    // will come from.
    int64_t entering = group_start_idx + hi + 1;
    // Add all of the values into the accumulator that are part of the
    // prefix frame for the first row.
    for (int64_t i = group_start_idx;
         i < std::min(std::max(group_start_idx, entering), group_end_idx);
         i++) {
        aggfunc.enter<window_func, ArrayType, T, DType,
                      window_frame_enum::CUMULATIVE>(i);
    }
    // Loop through each row of the partition in ascending order, inserting
    // values into the prefix and calculating the current value of the
    // aggregation to populate the current row of the output array.
    for (int64_t i = group_start_idx; i < group_end_idx; i++) {
        // Compute the index upper_bound such that the current value
        // the value at index i corresponds to the aggregate value
        // of the input array in the range of indices [0, upper_bound).
        int64_t upper_bound =
            std::min(std::max(group_start_idx, entering), group_end_idx);
        aggfunc.compute_frame<window_func, ArrayType, T, DType,
                              window_frame_enum::CUMULATIVE>(i, group_start_idx,
                                                             upper_bound);
        // If entering is in bounds, add that index to the accumulator
        if (group_start_idx <= entering && entering < group_end_idx) {
            aggfunc.enter<window_func, ArrayType, T, DType,
                          window_frame_enum::CUMULATIVE>(entering);
        }
        entering++;
    }
    if (static_cast<uint64_t>(group_end_idx) == in_arr->length) {
        // After the computation is complete for each partition, do any
        // necessary final operations (e.g. converting string vectors to proper
        // arrays)
        aggfunc.cleanup<window_func, ArrayType, T, DType,
                        window_frame_enum::CUMULATIVE>();
    }
}

/**
 * @brief Computes a window aggregation with a suffix window frame
 * on a partition of the input array. E.g.:
 *
 * ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
 *
 * Starts at the end of the partition and adds elements to the accumulator
 * in reverse order without ever removing any elements.
 *
 * @param[in] in_arr: The input array whose data is being aggregated.
 * @param[in,out] out_arr: The array where the results are written to.
 * @param[in] sorted_idx: The mapping of sorted indices back to their
 * locations in the original table.
 * @param[in] hi: The number of indices after the current row that the
 * prefix frame includes (inclusive).
 * @param[in] group_start_idx: the row where the current partition begins
 * (inclusive).
 * @param[in] group_end_idx: the row where the current partition ends
 * (exclusive).
 */
template <Bodo_FTypes::FTypeEnum window_func,
          bodo_array_type::arr_type_enum ArrayType, typename T,
          Bodo_CTypes::CTypeEnum DType>
void window_frame_suffix_computation(std::shared_ptr<array_info> in_arr,
                                     std::shared_ptr<array_info> out_arr,
                                     std::shared_ptr<array_info> sorted_idx,
                                     const int64_t lo,
                                     const int64_t group_start_idx,
                                     const int64_t group_end_idx) {
    // Create and initialize the accumulator.
    WindowAggfunc aggfunc(in_arr, out_arr, sorted_idx);
    aggfunc.init<window_func, ArrayType, T, DType,
                 window_frame_enum::CUMULATIVE>();
    // Entering = the index where the next value to enter the window frame
    // will come from.
    int64_t entering = group_end_idx + lo - 1;
    // Add all of the values into the accumulator that are part of the
    // suffix frame for the last row.
    for (int64_t i = group_end_idx - 1;
         i >= std::max(group_start_idx, group_end_idx + lo); i--) {
        aggfunc.enter<window_func, ArrayType, T, DType,
                      window_frame_enum::CUMULATIVE>(i);
    }
    // Loop through each row of the partition in descending order, inserting
    // values into the prefix and calculating the current value of the
    // aggregation to populate the current row of the output array.
    for (int64_t i = group_end_idx - 1; i >= group_start_idx; i--) {
        // Compute the index lower_bound such that the current value
        // the value at index i corresponds to the aggregate value
        // of the input array in the range of indices [lower_bound, n).
        int64_t lower_bound =
            std::min(std::max(group_start_idx, entering), group_end_idx);
        aggfunc.compute_frame<window_func, ArrayType, T, DType,
                              window_frame_enum::CUMULATIVE>(i, lower_bound,
                                                             group_end_idx);
        // If entering is in bounds, add that index to the accumulator
        entering--;
        if (group_start_idx <= entering && entering < group_end_idx) {
            aggfunc.enter<window_func, ArrayType, T, DType,
                          window_frame_enum::CUMULATIVE>(entering);
        }
    }
    if (static_cast<uint64_t>(group_end_idx) == in_arr->length) {
        // After the computation is complete for each partition, do any
        // necessary final operations (e.g. converting string vectors to proper
        // arrays)
        aggfunc.cleanup<window_func, ArrayType, T, DType,
                        window_frame_enum::CUMULATIVE>();
    }
}

/**
 * @brief Computes a window aggregation with a window frame by
 * identifying which of the 3 cases (prefix, suffix or sliding)
 * the bounds match with and using the corresponding helper function.
 * The function iterates across the rows and each time it detects the
 * end of a partition it calls the helper function on the subset of
 * rows corresponding to a complete partition.
 *
 * @param[in] in_arr: The input array whose data is being aggregated.
 * @param[in,out] out_arr: The array where the results are written to.
 * @param[in] sorted_groups: the sorted array of group numbers indicating
 * which partition the current row belongs to.
 * @param[in] sorted_idx: The mapping of sorted indices back to their
 * locations in the original table.
 * @param[in] frame_lo: Pointer to the lower bound value (nullptr if
 * the lower bound is UNBOUNDED PRECEDING)
 * @param[in] frame_hi: Pointer to the upper bound value (nullptr if
 * the upper bound is UNBOUNDED FOLLOWING)
 *
 * See window_frame_computation_bounds_handler for more details on the
 * interpretation of frame_lo / frame_hi
 */
template <Bodo_FTypes::FTypeEnum window_func,
          bodo_array_type::arr_type_enum ArrayType, typename T,
          Bodo_CTypes::CTypeEnum DType>
void window_frame_computation_with_frame(
    std::shared_ptr<array_info> in_arr, std::shared_ptr<array_info> out_arr,
    std::shared_ptr<array_info> sorted_groups,
    std::shared_ptr<array_info> sorted_idx, const int64_t* frame_lo,
    const int64_t* frame_hi) {
    int64_t n = in_arr->length;
    int64_t prev_group = -1;
    int64_t group_start_idx = 0;
    for (int64_t i = 0; i < n; i++) {
        // If i = 0 (curr_group != prev_group) will always be true since
        // prev_group is a dummy value. We are only interested in rows
        // that have a different group from the previous row
        int64_t curr_group = getv<int64_t>(sorted_groups, i);
        if ((i > 0) && (curr_group != prev_group)) {
            if (frame_lo == nullptr) {
                // Case 1: cumulative window frame from the beginning of
                // the partition to some row relative to the current row
                window_frame_prefix_computation<window_func, ArrayType, T,
                                                DType>(
                    in_arr, out_arr, sorted_idx, *frame_hi, group_start_idx, i);
            } else if (frame_hi == nullptr) {
                // Case 2: cumulative window frame from the end of
                // the partition to some row relative to the current row
                window_frame_suffix_computation<window_func, ArrayType, T,
                                                DType>(
                    in_arr, out_arr, sorted_idx, *frame_lo, group_start_idx, i);
            } else {
                // Case 3: sliding window frame relative to the current row
                window_frame_sliding_computation<window_func, ArrayType, T,
                                                 DType>(
                    in_arr, out_arr, sorted_idx, *frame_lo, *frame_hi,
                    group_start_idx, i);
            }
            group_start_idx = i;
        }
        prev_group = curr_group;
    }
    // Repeat the process one more time on the final group
    if (frame_lo == nullptr) {
        window_frame_prefix_computation<window_func, ArrayType, T, DType>(
            in_arr, out_arr, sorted_idx, *frame_hi, group_start_idx, n);
    } else if (frame_hi == nullptr) {
        window_frame_suffix_computation<window_func, ArrayType, T, DType>(
            in_arr, out_arr, sorted_idx, *frame_lo, group_start_idx, n);
    } else {
        window_frame_sliding_computation<window_func, ArrayType, T, DType>(
            in_arr, out_arr, sorted_idx, *frame_lo, *frame_hi, group_start_idx,
            n);
    }
}

/**
 * @brief Computes a window aggregation by identifying whether the
 * aggregation is being done with or without a window frame and forwarding
 * to the corresponding helper function.
 *
 * @param[in] in_arr: The input array whose data is being aggregated.
 * @param[in,out] out_arr: The array where the results are written to.
 * @param[in] sorted_groups: the sorted array of group numbers indicating
 * which partition the current row belongs to.
 * @param[in] sorted_idx: The mapping of sorted indices back to their
 * locations in the original table.
 * @param[in] frame_lo: Pointer to the lower bound value (nullptr if
 * the lower bound is UNBOUNDED PRECEDING)
 * @param[in] frame_hi: Pointer to the upper bound value (nullptr if
 * the upper bound is UNBOUNDED FOLLOWING)
 *
 * Note: if frame_lo/frame_hi are non-null then the values stored within
 * them correspond to inclusive bounds of a window frame relative to
 * the current index. For example:
 *
 * UNBOUNDED PRECEDING to CURRENT ROW -> null & 0
 *         3 PRECEDING to 4 FOLLOWING ->   -3 & +4
 *        10 PRECEDING to 1 PRECEDING ->  -10 & -1
 * 1 FOLLOWING to UNBOUNDED FOLLOWING -  >  1 & null
 */
template <Bodo_FTypes::FTypeEnum window_func,
          bodo_array_type::arr_type_enum ArrayType, typename T,
          Bodo_CTypes::CTypeEnum DType>
void window_frame_computation_bounds_handler(
    std::shared_ptr<array_info> in_arr, std::shared_ptr<array_info> out_arr,
    std::shared_ptr<array_info> sorted_groups,
    std::shared_ptr<array_info> sorted_idx, int64_t* frame_lo,
    int64_t* frame_hi) {
    if (frame_lo == nullptr && frame_hi == nullptr) {
        window_frame_computation_no_frame<window_func, ArrayType, T, DType>(
            in_arr, out_arr, sorted_groups, sorted_idx);
    } else {
        window_frame_computation_with_frame<window_func, ArrayType, T, DType>(
            in_arr, out_arr, sorted_groups, sorted_idx, frame_lo, frame_hi);
    }
}

/**
 * @brief Computes a window aggregation by switching on the function
 * type being calculated.
 *
 * @param[in] in_arr: The input array whose data is being aggregated.
 * @param[in,out] out_arr: The array where the results are written to.
 * @param[in] sorted_groups: the sorted array of group numbers indicating
 * which partition the current row belongs to.
 * @param[in] sorted_idx: The mapping of sorted indices back to their
 * locations in the original table.
 * @param[in] frame_lo: Pointer to the lower bound value (nullptr if
 * the lower bound is UNBOUNDED PRECEDING)
 * @param[in] frame_hi: Pointer to the upper bound value (nullptr if
 * the upper bound is UNBOUNDED FOLLOWING)
 *
 * See window_frame_computation_bounds_handler for more details on the
 * interpretation of frame_lo / frame_hi
 */
template <bodo_array_type::arr_type_enum ArrayType, typename T,
          Bodo_CTypes::CTypeEnum DType>
void window_frame_computation_ftype_handler(
    std::shared_ptr<array_info> in_arr, std::shared_ptr<array_info> out_arr,
    std::shared_ptr<array_info> sorted_groups,
    std::shared_ptr<array_info> sorted_idx, int64_t* frame_lo,
    int64_t* frame_hi, int ftype) {
    switch (ftype) {
        case Bodo_FTypes::size:
            window_frame_computation_bounds_handler<Bodo_FTypes::size,
                                                    ArrayType, T, DType>(
                in_arr, out_arr, sorted_groups, sorted_idx, frame_lo, frame_hi);
            break;
        case Bodo_FTypes::ratio_to_report:
            window_frame_computation_bounds_handler<
                Bodo_FTypes::ratio_to_report, ArrayType, T, DType>(
                in_arr, out_arr, sorted_groups, sorted_idx, frame_lo, frame_hi);
            break;
        case Bodo_FTypes::count:
            window_frame_computation_bounds_handler<Bodo_FTypes::count,
                                                    ArrayType, T, DType>(
                in_arr, out_arr, sorted_groups, sorted_idx, frame_lo, frame_hi);
            break;
        case Bodo_FTypes::count_if:
            window_frame_computation_bounds_handler<Bodo_FTypes::count_if,
                                                    ArrayType, T, DType>(
                in_arr, out_arr, sorted_groups, sorted_idx, frame_lo, frame_hi);
            break;
        case Bodo_FTypes::any_value:
            window_frame_computation_bounds_handler<Bodo_FTypes::any_value,
                                                    ArrayType, T, DType>(
                in_arr, out_arr, sorted_groups, sorted_idx, frame_lo, frame_hi);
            break;
        default:
            throw std::runtime_error(
                "Invalid window function for frame based computation: " +
                std::string(get_name_for_Bodo_FTypes(ftype)));
    }
}

/**
 * @brief Computes a window aggregation by casing on the array type
 * and forwarding  to the helper function with the array type templated.
 *
 * @param[in] in_arr: The input array whose data is being aggregated.
 * @param[in,out] out_arr: The array where the results are written to.
 * @param[in] sorted_groups: the sorted array of group numbers indicating
 * which partition the current row belongs to.
 * @param[in] sorted_idx: The mapping of sorted indices back to their
 * locations in the original table.
 * @param[in] frame_lo: Pointer to the lower bound value (nullptr if
 * the lower bound is UNBOUNDED PRECEDING)
 * @param[in] frame_hi: Pointer to the upper bound value (nullptr if
 * the upper bound is UNBOUNDED FOLLOWING)
 *
 * See window_frame_computation_bounds_handler for more details on the
 * interpretation of frame_lo / frame_hi
 */
template <bodo_array_type::arr_type_enum ArrayType>
void window_frame_computation_dtype_handler(
    std::shared_ptr<array_info> in_arr, std::shared_ptr<array_info> out_arr,
    std::shared_ptr<array_info> sorted_groups,
    std::shared_ptr<array_info> sorted_idx, int64_t* frame_lo,
    int64_t* frame_hi, int ftype) {
    switch (in_arr->dtype) {
        case Bodo_CTypes::_BOOL:
            window_frame_computation_ftype_handler<
                ArrayType, dtype_to_type<Bodo_CTypes::_BOOL>::type,
                Bodo_CTypes::_BOOL>(in_arr, out_arr, sorted_groups, sorted_idx,
                                    frame_lo, frame_hi, ftype);
            break;
        case Bodo_CTypes::INT8:
            window_frame_computation_ftype_handler<
                ArrayType, dtype_to_type<Bodo_CTypes::INT8>::type,
                Bodo_CTypes::INT8>(in_arr, out_arr, sorted_groups, sorted_idx,
                                   frame_lo, frame_hi, ftype);
            break;
        case Bodo_CTypes::UINT8:
            window_frame_computation_ftype_handler<
                ArrayType, dtype_to_type<Bodo_CTypes::UINT8>::type,
                Bodo_CTypes::UINT8>(in_arr, out_arr, sorted_groups, sorted_idx,
                                    frame_lo, frame_hi, ftype);
            break;
        case Bodo_CTypes::INT16:
            window_frame_computation_ftype_handler<
                ArrayType, dtype_to_type<Bodo_CTypes::INT16>::type,
                Bodo_CTypes::INT16>(in_arr, out_arr, sorted_groups, sorted_idx,
                                    frame_lo, frame_hi, ftype);
            break;
        case Bodo_CTypes::UINT16:
            window_frame_computation_ftype_handler<
                ArrayType, dtype_to_type<Bodo_CTypes::UINT16>::type,
                Bodo_CTypes::UINT16>(in_arr, out_arr, sorted_groups, sorted_idx,
                                     frame_lo, frame_hi, ftype);
            break;
        case Bodo_CTypes::INT32:
            window_frame_computation_ftype_handler<
                ArrayType, dtype_to_type<Bodo_CTypes::INT32>::type,
                Bodo_CTypes::INT32>(in_arr, out_arr, sorted_groups, sorted_idx,
                                    frame_lo, frame_hi, ftype);
            break;
        case Bodo_CTypes::UINT32:
            window_frame_computation_ftype_handler<
                ArrayType, dtype_to_type<Bodo_CTypes::UINT32>::type,
                Bodo_CTypes::UINT32>(in_arr, out_arr, sorted_groups, sorted_idx,
                                     frame_lo, frame_hi, ftype);
            break;
        case Bodo_CTypes::INT64:
            window_frame_computation_ftype_handler<
                ArrayType, dtype_to_type<Bodo_CTypes::INT64>::type,
                Bodo_CTypes::INT64>(in_arr, out_arr, sorted_groups, sorted_idx,
                                    frame_lo, frame_hi, ftype);
            break;
        case Bodo_CTypes::UINT64:
            window_frame_computation_ftype_handler<
                ArrayType, dtype_to_type<Bodo_CTypes::UINT64>::type,
                Bodo_CTypes::UINT64>(in_arr, out_arr, sorted_groups, sorted_idx,
                                     frame_lo, frame_hi, ftype);
            break;
        case Bodo_CTypes::FLOAT32:
            window_frame_computation_ftype_handler<
                ArrayType, dtype_to_type<Bodo_CTypes::FLOAT32>::type,
                Bodo_CTypes::FLOAT32>(in_arr, out_arr, sorted_groups,
                                      sorted_idx, frame_lo, frame_hi, ftype);
            break;
        case Bodo_CTypes::FLOAT64:
            window_frame_computation_ftype_handler<
                ArrayType, dtype_to_type<Bodo_CTypes::FLOAT64>::type,
                Bodo_CTypes::FLOAT64>(in_arr, out_arr, sorted_groups,
                                      sorted_idx, frame_lo, frame_hi, ftype);
            break;
        case Bodo_CTypes::DECIMAL:
            window_frame_computation_ftype_handler<
                ArrayType, dtype_to_type<Bodo_CTypes::DECIMAL>::type,
                Bodo_CTypes::DECIMAL>(in_arr, out_arr, sorted_groups,
                                      sorted_idx, frame_lo, frame_hi, ftype);
            break;
        case Bodo_CTypes::DATE:
            window_frame_computation_ftype_handler<
                ArrayType, dtype_to_type<Bodo_CTypes::DATE>::type,
                Bodo_CTypes::DATE>(in_arr, out_arr, sorted_groups, sorted_idx,
                                   frame_lo, frame_hi, ftype);
            break;
        case Bodo_CTypes::DATETIME:
            window_frame_computation_ftype_handler<
                ArrayType, dtype_to_type<Bodo_CTypes::DATETIME>::type,
                Bodo_CTypes::DATETIME>(in_arr, out_arr, sorted_groups,
                                       sorted_idx, frame_lo, frame_hi, ftype);
            break;
        case Bodo_CTypes::TIMEDELTA:
            window_frame_computation_ftype_handler<
                ArrayType, dtype_to_type<Bodo_CTypes::TIMEDELTA>::type,
                Bodo_CTypes::TIMEDELTA>(in_arr, out_arr, sorted_groups,
                                        sorted_idx, frame_lo, frame_hi, ftype);
            break;
        case Bodo_CTypes::TIME:
            window_frame_computation_ftype_handler<
                ArrayType, dtype_to_type<Bodo_CTypes::TIME>::type,
                Bodo_CTypes::TIME>(in_arr, out_arr, sorted_groups, sorted_idx,
                                   frame_lo, frame_hi, ftype);
            break;
        default:
            throw std::runtime_error(
                "Unsupported dtype for window functions: " +
                GetDtype_as_string(in_arr->dtype));
    }
}

/**
 * @brief Computes a window aggregation by casing on the array type
 * and forwarding  to the helper function with the array type templated.
 *
 * @param[in] in_arr: The input array whose data is being aggregated.
 * @param[in,out] out_arr: The array where the results are written to.
 * @param[in] sorted_groups: the sorted array of group numbers indicating
 * which partition the current row belongs to.
 * @param[in] sorted_idx: The mapping of sorted indices back to their
 * locations in the original table.
 * @param[in] frame_lo: Pointer to the lower bound value (nullptr if
 * the lower bound is UNBOUNDED PRECEDING)
 * @param[in] frame_hi: Pointer to the upper bound value (nullptr if
 * the upper bound is UNBOUNDED FOLLOWING)
 *
 * See window_frame_computation_bounds_handler for more details on the
 * interpretation of frame_lo / frame_hi
 */
void window_frame_computation(std::shared_ptr<array_info> in_arr,
                              std::shared_ptr<array_info> out_arr,
                              std::shared_ptr<array_info> sorted_groups,
                              std::shared_ptr<array_info> sorted_idx,
                              int64_t* frame_lo, int64_t* frame_hi, int ftype) {
    switch (in_arr->arr_type) {
        case bodo_array_type::NUMPY:
            window_frame_computation_dtype_handler<bodo_array_type::NUMPY>(
                in_arr, out_arr, sorted_groups, sorted_idx, frame_lo, frame_hi,
                ftype);
            break;
        case bodo_array_type::NULLABLE_INT_BOOL:
            window_frame_computation_dtype_handler<
                bodo_array_type::NULLABLE_INT_BOOL>(in_arr, out_arr,
                                                    sorted_groups, sorted_idx,
                                                    frame_lo, frame_hi, ftype);
            break;
        case bodo_array_type::STRING:
            window_frame_computation_ftype_handler<
                bodo_array_type::STRING, std::string_view, Bodo_CTypes::STRING>(
                in_arr, out_arr, sorted_groups, sorted_idx, frame_lo, frame_hi,
                ftype);
            break;
        case bodo_array_type::DICT:
            window_frame_computation_ftype_handler<
                bodo_array_type::DICT, std::string_view, Bodo_CTypes::STRING>(
                in_arr, out_arr, sorted_groups, sorted_idx, frame_lo, frame_hi,
                ftype);
            break;
        default:
            throw std::runtime_error(
                "Unsupported array for window functions: " +
                GetArrType_as_string(in_arr->arr_type));
    }
}
