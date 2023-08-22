// Copyright (C) 2023 Bodo Inc. All rights reserved.
#ifndef _WINDOW_AGGFUNCS_H_INCLUDED
#define _WINDOW_AGGFUNCS_H_INCLUDED

#include "_array_utils.h"
#include "_bodo_common.h"
#include "_groupby_common.h"
#include "_groupby_ftypes.h"

template <int ftype>
concept first = ftype == Bodo_FTypes::first;

template <int ftype>
concept last = ftype == Bodo_FTypes::last;

template <int ftype>
concept var = ftype == Bodo_FTypes::var;

template <int ftype>
concept var_pop = ftype == Bodo_FTypes::var_pop;

template <int ftype>
concept stddev = ftype == Bodo_FTypes::std;

template <int ftype>
concept stddev_pop = ftype == Bodo_FTypes::std_pop;

template <int ftype>
concept mean = ftype == Bodo_FTypes::mean;

template <int ftype>
concept ratio_to_report = ftype == Bodo_FTypes::ratio_to_report;

template <int ftype>
concept any_value = ftype == Bodo_FTypes::any_value;

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
            std::string(get_name_for_Bodo_FTypes(ftype)) + " with array type " +
            std::string(GetArrType_as_string(ArrayType)) + " and dtype " +
            std::string(GetDtype_as_string(DType)));
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
            std::string(get_name_for_Bodo_FTypes(ftype)) + " with array type " +
            std::string(GetArrType_as_string(ArrayType)) + " and dtype " +
            std::string(GetDtype_as_string(DType)));
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
            std::string(get_name_for_Bodo_FTypes(ftype)) + " with array type " +
            std::string(GetArrType_as_string(ArrayType)) + " and dtype " +
            std::string(GetDtype_as_string(DType)));
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
            std::string(get_name_for_Bodo_FTypes(ftype)) + " with array type " +
            std::string(GetArrType_as_string(ArrayType)) + " and dtype " +
            std::string(GetDtype_as_string(DType)));
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
     * instead of providing an actual implementaiton for this function.
     */
    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
    inline void compute_frame(int64_t i, int64_t lower_bound,
                              int64_t upper_bound) {
        throw std::runtime_error(
            "WindowAggfunc::compute_frame: Unimplemented for window function " +
            std::string(get_name_for_Bodo_FTypes(ftype)) + " with array type " +
            std::string(GetArrType_as_string(ArrayType)) + " and dtype " +
            std::string(GetDtype_as_string(DType)));
    }

    /**
     * @brief Performs any final operations that need to be done after all
     * the values in each partition have been calculated. Default implementation
     * is to do nothing.
     */
    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
    inline void cleanup() {}

    /**
     * @brief Performs any final operations that need to be done after all
     * the values in each partition have been calculated. The implementation
     * for string input arrays is to convert the string / null
     * vectors into the final array.
     *
     * The implementation is only used for ftypes that will have a string
     * output.
     */
    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires((first<ftype> || last<ftype> || any_value<ftype>) &&
                 string_array<ArrayType>)
    inline void cleanup() {
        std::shared_ptr<array_info> new_out_arr = create_string_array(
            Bodo_CTypes::STRING, *null_vector, *string_vector);
        *out_arr = std::move(*new_out_arr);
    }

    /**
     * @brief Performs any final operations that need to be done after all
     * the values in each partition have been calculated. The implementation
     * for dictionary encoded input arrays is to create a new dictionary
     * encoded array using the old dictionary and the new indices.
     *
     * The implementation is only used for ftypes that will have a string output
     * where all strings come from the input dictionary (e.g. min, first_value).
     */
    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires((first<ftype> || last<ftype> || any_value<ftype>) &&
                 dict_array<ArrayType>)
    inline void cleanup() {
        std::shared_ptr<array_info> new_out_arr =
            create_dict_string_array(in_arr->child_arrays[0], out_indices);
        *out_arr = std::move(*new_out_arr);
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

    // RATIO_TO_REPORT implementations.

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires(ratio_to_report<ftype> && !string_or_dict<ArrayType>)
    inline void init() {
        m1 = 0.0;
    }

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires(ratio_to_report<ftype> && !string_or_dict<ArrayType>)
    inline void enter(int64_t i) {
        int64_t idx = getv<int64_t>(sorted_idx, i);
        if (non_null_at<ArrayType, T, DType>(*in_arr, idx)) {
            m1 += to_double<T, DType>(
                get_arr_item<ArrayType, T, DType>(*in_arr, idx));
        }
    }

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType>
        requires(ratio_to_report<ftype> && !string_or_dict<ArrayType>)
    inline void compute_partition(int64_t start_idx, int64_t end_idx) {
        // For each row in the partition, divides the row by the sum
        // of values in the partition.
        const bool all_null = m1 == 0.0;
        for (int64_t i = start_idx; i < end_idx; i++) {
            int64_t idx = getv<int64_t>(sorted_idx, i);
            if (all_null || is_null_at<ArrayType, T, DType>(*in_arr, idx)) {
                set_to_null<bodo_array_type::NULLABLE_INT_BOOL, double,
                            Bodo_CTypes::FLOAT64>(*out_arr, idx);
            } else {
                double val = to_double<T, DType>(
                    get_arr_item<ArrayType, T, DType>(*in_arr, idx));
                set_arr_item<bodo_array_type::NULLABLE_INT_BOOL, double,
                             Bodo_CTypes::FLOAT64>(*out_arr, idx, val / m1);
                set_non_null<bodo_array_type::NULLABLE_INT_BOOL, double,
                             Bodo_CTypes::FLOAT64>(*out_arr, idx);
            }
        }
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
        requires(ftype == Bodo_FTypes::count_if && !string_array<ArrayType> &&
                 !dict_array<ArrayType>)
    inline void enter(int64_t i) {
        int64_t idx = getv<int64_t>(sorted_idx, i);
        if (non_null_at<ArrayType, T, DType>(*in_arr, idx) &&
            get_arr_item<ArrayType, bool, Bodo_CTypes::_BOOL>(*in_arr, idx)) {
            in_window++;
        }
    }

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires(ftype == Bodo_FTypes::count_if && !string_array<ArrayType> &&
                 !dict_array<ArrayType>)
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

    // AVG / VAR / STD implementations.

    // The minimum number of elememnts required in the window for the
    // function to be non-null.
    template <int ftype>
    static constexpr int min_elements() {
        return 1;
    }

    template <int ftype>
        requires(var<ftype> || stddev<ftype>)
    static constexpr int min_elements() {
        return 2;
    }

    // Computes the mean / variance / standard deviation in terms of
    // in_window, m1, m2, etc.

    template <int ftype>
        requires(mean<ftype>)
    inline double compute_moment_function() {
        return m1 / in_window;
    }

    template <int ftype>
        requires(var<ftype>)
    inline double compute_moment_function() {
        return (m2 - (m1 * m1) / in_window) / (in_window - 1);
    }

    template <int ftype>
        requires(var_pop<ftype>)
    inline double compute_moment_function() {
        return (m2 - (m1 * m1) / in_window) / in_window;
    }

    template <int ftype>
        requires(stddev<ftype>)
    inline double compute_moment_function() {
        return std::sqrt((m2 - (m1 * m1) / in_window) / (in_window - 1));
    }

    template <int ftype>
        requires(stddev_pop<ftype>)
    inline double compute_moment_function() {
        return std::sqrt((m2 - (m1 * m1) / in_window) / in_window);
    }

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires(mean<ftype> || var<ftype> || var_pop<ftype> || stddev<ftype> ||
                 stddev_pop<ftype>)
    inline void init() {
        in_window = 0;
        nan_counter = 0;
        m1 = 0.0;
        m2 = 0.0;
    }

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires((mean<ftype> || var<ftype> || var_pop<ftype> ||
                  stddev<ftype> || stddev_pop<ftype>) &&
                 numeric_dtype<DType>)
    inline void enter(int64_t i) {
        int64_t idx = getv<int64_t>(sorted_idx, i);
        if (non_null_at<ArrayType, T, DType>(*in_arr, idx)) {
            T val = get_arr_item<ArrayType, T, DType>(*in_arr, idx);
            if (isnan_alltype<T, DType>(val)) {
                nan_counter++;
            } else {
                double val = to_double<T, DType>(
                    get_arr_item<ArrayType, T, DType>(*in_arr, idx));
                m1 += val;
                m2 += val * val;
            }
            in_window++;
        }
    }

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires((mean<ftype> || var<ftype> || var_pop<ftype> ||
                  stddev<ftype> || stddev_pop<ftype>) &&
                 numeric_dtype<DType>)
    inline void exit(int64_t i) {
        int64_t idx = getv<int64_t>(sorted_idx, i);
        if (non_null_at<ArrayType, T, DType>(*in_arr, idx)) {
            T val = get_arr_item<ArrayType, T, DType>(*in_arr, idx);
            if (isnan_alltype<T, DType>(val)) {
                nan_counter--;
            } else {
                double val = to_double<T, DType>(
                    get_arr_item<ArrayType, T, DType>(*in_arr, idx));
                m1 -= val;
                m2 -= val * val;
            }
            in_window--;
        }
    }

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType>
        requires(mean<ftype> || var<ftype> || var_pop<ftype> || stddev<ftype> ||
                 stddev_pop<ftype>)
    inline void compute_partition(int64_t start_idx, int64_t end_idx) {
        // If there are no non-null entries, set the entire partition to null.
        if (in_window < min_elements<ftype>()) {
            for (int64_t i = start_idx; i < end_idx; i++) {
                int64_t idx = getv<int64_t>(sorted_idx, i);
                set_to_null<bodo_array_type::NULLABLE_INT_BOOL, double,
                            Bodo_CTypes::FLOAT64>(*out_arr, idx);
            }
            return;
        }
        double result;
        if (nan_counter > 0) {
            // If there is at least 1 NaN entry, set the entire partition to
            // NaN.
            result = std::numeric_limits<double>::quiet_NaN();
        } else {
            // Otherwise, fill the entire partition to the result
            // of the corresponding comptation in terms of in_window,
            // m1, m2, etc.
            result = compute_moment_function<ftype>();
            // result = m1 / in_window;
        }
        for (int64_t i = start_idx; i < end_idx; i++) {
            int64_t idx = getv<int64_t>(sorted_idx, i);
            set_non_null<bodo_array_type::NULLABLE_INT_BOOL, double,
                         Bodo_CTypes::FLOAT64>(*out_arr, idx);
            set_arr_item<bodo_array_type::NULLABLE_INT_BOOL, double,
                         Bodo_CTypes::FLOAT64>(*out_arr, idx, result);
        }
    }

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires(mean<ftype> || var<ftype> || var_pop<ftype> || stddev<ftype> ||
                 stddev_pop<ftype>)
    inline void compute_frame(int64_t i, int64_t lower_bound,
                              int64_t upper_bound) {
        // If there are no non-null entries, set to null.
        if (in_window < min_elements<ftype>()) {
            int64_t idx = getv<int64_t>(sorted_idx, i);
            set_to_null<bodo_array_type::NULLABLE_INT_BOOL, double,
                        Bodo_CTypes::FLOAT64>(*out_arr, idx);
            return;
        }
        double result;
        if (nan_counter > 0) {
            // If there is at least 1 NaN entry, set to NaN.
            result = std::numeric_limits<double>::quiet_NaN();
        } else {
            // Otherwise, set to the result of the corresponding
            // comptation in terms of in_window, m1, m2, etc.
            result = compute_moment_function<ftype>();
            // result = m1 / in_window;
        }
        int64_t idx = getv<int64_t>(sorted_idx, i);
        set_non_null<bodo_array_type::NULLABLE_INT_BOOL, double,
                     Bodo_CTypes::FLOAT64>(*out_arr, idx);
        set_arr_item<bodo_array_type::NULLABLE_INT_BOOL, double,
                     Bodo_CTypes::FLOAT64>(*out_arr, idx, result);
    }

    // FIRST_VALUE / LAST_VALUE / ANY_VALUE implementations (nullable/numpy
    // arrays).

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires((first<ftype> || last<ftype> || any_value<ftype>) &&
                 !string_or_dict<ArrayType>)
    inline void init() {}

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires(first<ftype> || last<ftype> || any_value<ftype>)
    inline void enter(int64_t i) {}

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires(first<ftype> || last<ftype>)
    inline void exit(int64_t i) {}

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType>
        requires(any_value<ftype> && !string_or_dict<ArrayType>)
    inline void compute_partition(int64_t start_idx, int64_t end_idx) {
        int64_t target_idx = get_value_fn_target_idx<ftype>(start_idx, end_idx);
        if (target_idx == -1 ||
            is_null_at<ArrayType, T, DType>(*in_arr, target_idx)) {
            // If the first element is null, set the entire partition
            // to null (for simplicity).
            for (int64_t i = start_idx; i < end_idx; i++) {
                int64_t idx = getv<int64_t>(sorted_idx, i);
                set_to_null<ArrayType, T, DType>(*out_arr, idx);
            }
        } else {
            // Otherwise, set the entire partition equal to the first
            // value (for simplicity).
            T val = get_arr_item<ArrayType, T, DType>(*in_arr, target_idx);
            for (int64_t i = start_idx; i < end_idx; i++) {
                int64_t idx = getv<int64_t>(sorted_idx, i);
                set_arr_item<ArrayType, T, DType>(*out_arr, idx, val);
                set_non_null<ArrayType, T, DType>(*out_arr, idx);
            }
        }
    }

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType>
        requires((first<ftype> || last<ftype>) && !string_or_dict<ArrayType>)
    inline void compute_partition(int64_t start_idx, int64_t end_idx) {
        int64_t target_idx = get_value_fn_target_idx<ftype>(start_idx, end_idx);
        if (target_idx == -1 ||
            is_null_at<ArrayType, T, DType>(*in_arr, target_idx)) {
            // If the first element is null, set the entire partition
            // to null (for simplicity).
            for (int64_t i = start_idx; i < end_idx; i++) {
                int64_t idx = getv<int64_t>(sorted_idx, i);
                set_to_null<bodo_array_type::NULLABLE_INT_BOOL, T, DType>(
                    *out_arr, idx);
            }
        } else {
            // Otherwise, set the entire partition equal to the first
            // value (for simplicity).
            T val = get_arr_item<ArrayType, T, DType>(*in_arr, target_idx);
            for (int64_t i = start_idx; i < end_idx; i++) {
                int64_t idx = getv<int64_t>(sorted_idx, i);
                set_arr_item<bodo_array_type::NULLABLE_INT_BOOL, T, DType>(
                    *out_arr, idx, val);
                set_non_null<bodo_array_type::NULLABLE_INT_BOOL, T, DType>(
                    *out_arr, idx);
            }
        }
    }

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires((first<ftype> || last<ftype>) && !string_or_dict<ArrayType>)
    inline void compute_frame(int64_t i, int64_t lower_bound,
                              int64_t upper_bound) {
        int64_t target_idx =
            get_value_fn_target_idx<ftype>(lower_bound, upper_bound);
        if (target_idx == -1 ||
            is_null_at<ArrayType, T, DType>(*in_arr, target_idx)) {
            // If the first element is null or the frame is empty, the answer is
            // null
            int64_t idx = getv<int64_t>(sorted_idx, i);
            // Note: the output array is nullable even if the input is
            // a numpy array
            set_to_null<bodo_array_type::NULLABLE_INT_BOOL, T, DType>(*out_arr,
                                                                      idx);
        } else {
            // Otherwise, the answer is the value located at lower_bound
            T val = get_arr_item<ArrayType, T, DType>(*in_arr, target_idx);
            int64_t idx = getv<int64_t>(sorted_idx, i);
            set_arr_item<bodo_array_type::NULLABLE_INT_BOOL, T, DType>(
                *out_arr, idx, val);
            set_non_null<bodo_array_type::NULLABLE_INT_BOOL, T, DType>(*out_arr,
                                                                       idx);
        }
    }

    // FIRST/LAST/ANY_VALUE implementations (string arrays).

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires((first<ftype> || last<ftype> || any_value<ftype>) &&
                 string_array<ArrayType>)
    inline void init() {
        // If the vectors have not been allocated yet, do so.
        if (null_vector == nullptr) {
            int64_t length = in_arr->length;
            string_vector = std::make_shared<bodo::vector<std::string>>(length);
            null_vector = std::make_shared<bodo::vector<uint8_t>>(length);
        }
    }

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType>
        requires((first<ftype> || last<ftype> || any_value<ftype>) &&
                 string_array<ArrayType>)
    inline void compute_partition(int64_t start_idx, int64_t end_idx) {
        int64_t target_idx = get_value_fn_target_idx<ftype>(start_idx, end_idx);
        if (is_null_at<ArrayType, T, DType>(*in_arr, target_idx)) {
            // If the first element is null, set the entire partition
            // to null (for simplicity).
            for (int64_t i = start_idx; i < end_idx; i++) {
                int64_t idx = getv<int64_t>(sorted_idx, i);
                SetBitTo(null_vector->data(), idx, false);
            }
        } else {
            // Otherwise, set the entire partition equal to the first
            // value (for simplicity).
            std::string_view val =
                get_arr_item_str<ArrayType, T, DType>(*in_arr, target_idx);
            for (int64_t i = start_idx; i < end_idx; i++) {
                int64_t idx = getv<int64_t>(sorted_idx, i);
                (*string_vector)[idx] = val;
                SetBitTo(null_vector->data(), idx, true);
            }
        }
    }

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires((first<ftype> || last<ftype>) && string_array<ArrayType>)
    inline void compute_frame(int64_t i, int64_t lower_bound,
                              int64_t upper_bound) {
        int64_t target_idx =
            get_value_fn_target_idx<ftype>(lower_bound, upper_bound);
        if (target_idx == -1 ||
            is_null_at<ArrayType, T, DType>(*in_arr, target_idx)) {
            // If the first element is null or the frame is empty, the answer is
            // null
            int64_t idx = getv<int64_t>(sorted_idx, i);
            SetBitTo(null_vector->data(), idx, false);
        } else {
            // Otherwise, the answer is the value located at lower_bound
            std::string_view val =
                get_arr_item_str<ArrayType, T, DType>(*in_arr, target_idx);
            int64_t idx = getv<int64_t>(sorted_idx, i);
            (*string_vector)[idx] = val;
            SetBitTo(null_vector->data(), idx, true);
        }
    }

    // Helper utility for FIRST/LAST/ANY/NTH_VALUE functions. Takes in
    // the lower/upper bounds of the current partition/frame and returns
    // the index where the value should be sought from

    template <int ftype>
        requires(any_value<ftype> || first<ftype>)
    inline int64_t get_value_fn_target_idx(int64_t lower_bound,
                                           int64_t upper_bound) {
        if (lower_bound < upper_bound) {
            return getv<int64_t>(sorted_idx, lower_bound);
        }
        return -1;
    }

    template <int ftype>
        requires(last<ftype>)
    inline int64_t get_value_fn_target_idx(int64_t lower_bound,
                                           int64_t upper_bound) {
        if (lower_bound < upper_bound) {
            return getv<int64_t>(sorted_idx, upper_bound - 1);
        }
        return -1;
    }

    // FIRST/LAST/ANY_VALUE implementations (dictionary encoded arrays).

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires((first<ftype> || last<ftype> || any_value<ftype>) &&
                 dict_array<ArrayType>)
    inline void init() {
        // If the indices array has not been allocated yet, do so.
        if (out_indices == nullptr) {
            int64_t length = in_arr->length;
            out_indices =
                alloc_nullable_array_all_nulls(length, Bodo_CTypes::INT32, 0);
        }
    }

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType>
        requires((first<ftype> || last<ftype> || any_value<ftype>) &&
                 dict_array<ArrayType>)
    inline void compute_partition(int64_t start_idx, int64_t end_idx) {
        int64_t target_idx = get_value_fn_target_idx<ftype>(start_idx, end_idx);
        if (is_null_at<ArrayType, T, DType>(*in_arr, target_idx)) {
            // If the first element is null, set the entire partition
            // to null (for simplicity).
            for (int64_t i = start_idx; i < end_idx; i++) {
                int64_t idx = getv<int64_t>(sorted_idx, i);
                set_to_null<bodo_array_type::NULLABLE_INT_BOOL, int32_t,
                            Bodo_CTypes::INT32>(*out_indices, idx);
            }
        } else {
            // Otherwise, set the entire partition equal to the first
            // value (for simplicity).
            int32_t val = get_arr_item<bodo_array_type::NULLABLE_INT_BOOL,
                                       int32_t, Bodo_CTypes::INT32>(
                *(in_arr->child_arrays[1]), target_idx);
            for (int64_t i = start_idx; i < end_idx; i++) {
                int64_t idx = getv<int64_t>(sorted_idx, i);
                set_arr_item<bodo_array_type::NULLABLE_INT_BOOL, int32_t,
                             Bodo_CTypes::INT32>(*out_indices, idx, val);
                set_non_null<bodo_array_type::NULLABLE_INT_BOOL, int32_t,
                             Bodo_CTypes::INT32>(*out_indices, idx);
            }
        }
    }

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires((first<ftype> || last<ftype>) && dict_array<ArrayType>)
    inline void compute_frame(int64_t i, int64_t lower_bound,
                              int64_t upper_bound) {
        int64_t target_idx =
            get_value_fn_target_idx<ftype>(lower_bound, upper_bound);
        if (target_idx == -1 ||
            is_null_at<ArrayType, T, DType>(*in_arr, target_idx)) {
            // If the first element is null or the frame is empty, the answer is
            // null
            int64_t idx = getv<int64_t>(sorted_idx, i);
            set_to_null<bodo_array_type::NULLABLE_INT_BOOL, int32_t,
                        Bodo_CTypes::INT32>(*out_indices, idx);
        } else {
            // Otherwise, the answer is the value located at lower_bound
            int32_t val = get_arr_item<bodo_array_type::NULLABLE_INT_BOOL,
                                       int32_t, Bodo_CTypes::INT32>(
                *(in_arr->child_arrays[1]), target_idx);
            int64_t idx = getv<int64_t>(sorted_idx, i);
            set_arr_item<bodo_array_type::NULLABLE_INT_BOOL, int32_t,
                         Bodo_CTypes::INT32>(*out_indices, idx, val);
            set_non_null<bodo_array_type::NULLABLE_INT_BOOL, int32_t,
                         Bodo_CTypes::INT32>(*out_indices, idx);
        }
    }

   private:
    // Columns storing the input data, output data, lookup from sorted to
    // unsorted indices, plus the argument vector.
    std::shared_ptr<array_info> in_arr;
    std::shared_ptr<array_info> out_arr;
    std::shared_ptr<array_info> sorted_idx;
    // Accumulator values.
    int64_t in_window;
    int64_t nan_counter;
    double m1;
    double m2;
    // Vectors to store the intermediary string values before they are placed
    // into a final array.
    std::shared_ptr<bodo::vector<std::string>> string_vector = nullptr;
    std::shared_ptr<bodo::vector<uint8_t>> null_vector = nullptr;
    // Array info to store the new indices when creating a dictionary encoded
    // array.
    std::shared_ptr<array_info> out_indices = nullptr;
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
