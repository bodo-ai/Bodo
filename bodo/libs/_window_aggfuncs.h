// Copyright (C) 2023 Bodo Inc. All rights reserved.
#ifndef _WINDOW_AGGFUNCS_H_INCLUDED
#define _WINDOW_AGGFUNCS_H_INCLUDED

#include "_array_utils.h"
#include "_bodo_common.h"
#include "_groupby_common.h"
#include "_groupby_ftypes.h"

// Enums to specify which type of window frame is being computer, in case
// a window function should have different implementaitons for some of
// the formats (e.g. min/max can accumulate a running min/max for
// cumulative/no_frame, but needs to do a full slice aggregation
// on sliding frames)
enum window_frame_enum { CUMULATIVE, SLIDING, NO_FRAME };

// Class to store any accumulator information for a window frame computation.
// Each new ftype that is supported using this infrastructure will have its
// own implementation of the methods templated on that ftype. However, one
// ftype can have multiple such implementations templated on the window
// frame type (cumulative vs sliding vs no frame) in case there are big
// differences in how the data should be accumulated. The class is templated
// on the array type, typename, and dtype so that type-agnostic getters/setters
// can be used for the various array types.
class WindowAggfunc {
   public:
    /**
     * @brief Constructs a new accumulator and calls the reset method.
     *
     * @param[in,out] w_info: The accumulator struct that is being inseretd
     * into.
     * @param[in] in_arr: The input array whose data is being inserted into the
     * accumulator.
     * @param[in] sorted_idx: The mapping of sorted indices back to their
     * locations in the original table.
     */
    WindowAggfunc(std::shared_ptr<array_info> _in_arr,
                  std::shared_ptr<array_info> _out_arr,
                  std::shared_ptr<array_info> _sorted_idx)
        : in_arr(_in_arr), out_arr(_out_arr), sorted_idx(_sorted_idx) {}

    // Fallback implementations.

    /**
     * @brief Initializes/resets the class with any default values
     * that the accumulator should hold when the window is starting
     * from a clean slate (or in a new partition).
     */
    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
    inline void init() {
        throw std::runtime_error(
            "WindowAggfunc::init: Unimplemented for window function " +
            std::string(get_name_for_Bodo_FTypes(ftype)));
    }
    /**
     * @brief Updates the acummulator by inserting the value of the input array
     * at index i (using sorted_idx as a layer of indirection).
     *
     * @param[in] i: The index being inserted into the accumulator.
     */
    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
    inline void enter(int64_t i) {
        throw std::runtime_error(
            "WindowAggfunc::enter: Unimplemented for window function " +
            std::string(get_name_for_Bodo_FTypes(ftype)));
    }

    /**
     * @brief Updates the acummulator by removing the value of the input array
     * at index i (using sorted_idx as a layer of indirection).
     *
     * @param[in] i: The index being inserted into the accumulator.
     *
     * Note: any ftype that does not support window frames should throw an error
     * instead of providing an actual implementaiton for this funciton.
     */
    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
    inline void exit(int64_t i) {
        throw std::runtime_error(
            "WindowAggfunc::exit: Unimplemented for window function " +
            std::string(get_name_for_Bodo_FTypes(ftype)));
    }

    /**
     * @brief Using the values stored in the accumulator, computes the window
     * aggregate value for the entire partition and writes it to every row.
     *
     * @param[in] start_idx: The index where the partition begins (inclusive).
     * @param[in] start_idx: The index where the partition begins (exclusive).
     */
    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType>
    inline void compute_partition(int64_t start_idx, int64_t end_idx) {
        throw std::runtime_error(
            "WindowAggfunc::compute_partition: Unimplemented for window "
            "function " +
            std::string(get_name_for_Bodo_FTypes(ftype)));
    }

    /**
     * @brief Using the values stored in the accumulator, computes the window
     * aggregate value for a specific row and its corresponding window frame,
     * and writes it to that row of the output array.
     *
     * @param[in] i: The index that the frame is relative to (and where the
     * output is written).
     * @param[in] lower_bound: The index where the frame begins (inclusive).
     * @param[in] upper_bound: The index where the frame begins (exclusive).
     *
     * Note: any ftype that does not support window frames should throw an error
     * instead of providing an actual implementaiton for this funciton.
     */
    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
    inline void compute_frame(int64_t i, int64_t lower_bound,
                              int64_t upper_bound) {
        throw std::runtime_error(
            "WindowAggfunc::compute_frame: Unimplemented for window function " +
            std::string(get_name_for_Bodo_FTypes(ftype)));
    }

    // COUNT(*) implementations.

    // COUNT(*) doesn't do anything for init, enter or exit because the entire
    // computation can be done in terms of the partition/frame start/stop
    // indices.

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires(ftype == Bodo_FTypes::size)
    inline void init() {}

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires(ftype == Bodo_FTypes::size)
    inline void enter(int64_t i) {}

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires(ftype == Bodo_FTypes::size)
    inline void exit(int64_t i) {}

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType>
        requires(ftype == Bodo_FTypes::size)
    inline void compute_partition(int64_t start_idx, int64_t end_idx) {
        // Fill the entire partition with the length of the partition
        int64_t count = end_idx - start_idx;
        for (int64_t i = start_idx; i < end_idx; i++) {
            int64_t idx = getv<int64_t>(sorted_idx, i);
            getv<int64_t>(out_arr, idx) = count;
        }
    }

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires(ftype == Bodo_FTypes::size)
    inline void compute_frame(int64_t i, int64_t lower_bound,
                              int64_t upper_bound) {
        // The aggrevate value is the number of indices in the window frame
        int64_t idx = getv<int64_t>(sorted_idx, i);
        getv<int64_t>(out_arr, idx) = upper_bound - lower_bound;
    }

    // COUNT implementations.

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires(ftype == Bodo_FTypes::count)
    inline void init() {
        in_window = 0;
    }

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires(ftype == Bodo_FTypes::count)
    inline void enter(int64_t i) {
        int64_t idx = getv<int64_t>(sorted_idx, i);
        if (non_null_at<ArrayType, T, DType>(*in_arr, idx)) {
            in_window++;
        }
    }

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires(ftype == Bodo_FTypes::count)
    inline void exit(int64_t i) {
        int64_t idx = getv<int64_t>(sorted_idx, i);
        if (non_null_at<ArrayType, T, DType>(*in_arr, idx)) {
            in_window--;
        }
    }

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType>
        requires(ftype == Bodo_FTypes::count)
    inline void compute_partition(int64_t start_idx, int64_t end_idx) {
        // Fills the entire partition with the value of in_window after scanning
        // through the entire partition
        for (int64_t i = start_idx; i < end_idx; i++) {
            int64_t idx = getv<int64_t>(sorted_idx, i);
            getv<int64_t>(out_arr, idx) = in_window;
        }
    }

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires(ftype == Bodo_FTypes::count)
    inline void compute_frame(int64_t i, int64_t lower_bound,
                              int64_t upper_bound) {
        // The aggregate value is the current value of in_window
        int64_t idx = getv<int64_t>(sorted_idx, i);
        getv<int64_t>(out_arr, idx) = in_window;
    }

    // COUNT_IF implementations.

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires(ftype == Bodo_FTypes::count_if)
    inline void init() {
        in_window = 0;
    }

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires(ftype == Bodo_FTypes::count_if)
    inline void enter(int64_t i) {
        int64_t idx = getv<int64_t>(sorted_idx, i);
        if (non_null_at<ArrayType, T, DType>(*in_arr, idx) &&
            get_arr_item<ArrayType, bool, Bodo_CTypes::_BOOL>(*in_arr, idx)) {
            in_window++;
        }
    }

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires(ftype == Bodo_FTypes::count_if)
    inline void exit(int64_t i) {
        int64_t idx = getv<int64_t>(sorted_idx, i);
        if (non_null_at<ArrayType, T, DType>(*in_arr, idx) &&
            get_arr_item<ArrayType, bool, Bodo_CTypes::_BOOL>(*in_arr, idx)) {
            in_window--;
        }
    }

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType>
        requires(ftype == Bodo_FTypes::count_if)
    inline void compute_partition(int64_t start_idx, int64_t end_idx) {
        // Fills the entire partition with the value of in_window after scanning
        // through the entire partition
        for (int64_t i = start_idx; i < end_idx; i++) {
            int64_t idx = getv<int64_t>(sorted_idx, i);
            getv<int64_t>(out_arr, idx) = in_window;
        }
    }

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires(ftype == Bodo_FTypes::count_if)
    inline void compute_frame(int64_t i, int64_t lower_bound,
                              int64_t upper_bound) {
        // The aggregate value is the current value of in_window
        int64_t idx = getv<int64_t>(sorted_idx, i);
        getv<int64_t>(out_arr, idx) = in_window;
    }

   private:
    // Columns storing the input data, output data, and lookup from sorted to
    // unsorted indices
    std::shared_ptr<array_info> in_arr;
    std::shared_ptr<array_info> out_arr;
    std::shared_ptr<array_info> sorted_idx;
    // Accumulator values
    int64_t in_window;
};

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
                              int64_t* frame_lo, int64_t* frame_hi, int ftype);

#endif  // _WINDOW_AGGFUNCS_H_INCLUDED
