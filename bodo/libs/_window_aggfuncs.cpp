// Copyright (C) 2023 Bodo Inc. All rights reserved.
// See this link for an explanation of how these aggfuncs work:
// https://bodo.atlassian.net/wiki/spaces/B/pages/1384546305/Using+and+Expanding+the+Groupby.Window+Infrastructure
#include "_window_aggfuncs.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_groupby_common.h"
#include "_groupby_ftypes.h"

// Class to store any accumulator information for a window frame computation.
// Each new ftype that is supported using this infrastructure will have its
// own implementation of the methods templated on that ftype. However, one
// ftype can have multiple such implementations templated on the window
// frame type (cumulative vs sliding vs no frame) in case there are big
// differences in how the data should be accumulated. Besides the ftype and
// frame type, the methods of the class are templated on the array type,
// typename, and dtype so that type-agnostic getters/setters can be used for the
// various array types.
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

    // Implementations for the counting functions

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires(count_fns<ftype>)
    inline void init() {
        // At the start of each partition, reset the count to zero.
        in_window = 0;
    }

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires(size<ftype>)
    inline void enter(int64_t i) {
        // For COUNT(*), whenever anything enters the window increment the
        // count.
        in_window++;
    }

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires(size<ftype>)
    inline void exit(int64_t i) {
        // For COUNT(*), whenever anything leaves the window decrement the
        // count.
        in_window--;
    }

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires(count<ftype>)
    inline void enter(int64_t i) {
        int64_t idx = getv<int64_t>(sorted_idx, i);
        // For COUNT, whenever any non-null value enters the window increment
        // the count.
        if (non_null_at<ArrayType, T, DType>(*in_arr, idx)) {
            in_window++;
        }
    }

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires(count<ftype>)
    inline void exit(int64_t i) {
        // For COUNT, whenever any non-null value leaves the window decrement
        // the count.
        int64_t idx = getv<int64_t>(sorted_idx, i);
        if (non_null_at<ArrayType, T, DType>(*in_arr, idx)) {
            in_window--;
        }
    }
    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires(count_if<ftype> && !string_or_dict<ArrayType>)
    inline void enter(int64_t i) {
        // For COUNT, whenever any TRUE value enters the window increment the
        // count.
        int64_t idx = getv<int64_t>(sorted_idx, i);
        if (non_null_at<ArrayType, T, DType>(*in_arr, idx) &&
            get_arr_item<ArrayType, bool, Bodo_CTypes::_BOOL>(*in_arr, idx)) {
            in_window++;
        }
    }

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires(count_if<ftype> && !string_or_dict<ArrayType>)
    inline void exit(int64_t i) {
        // For COUNT, whenever any TRUE value leaves the window decrement the
        // count.
        int64_t idx = getv<int64_t>(sorted_idx, i);
        if (non_null_at<ArrayType, T, DType>(*in_arr, idx) &&
            get_arr_item<ArrayType, bool, Bodo_CTypes::_BOOL>(*in_arr, idx)) {
            in_window--;
        }
    }

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType>
        requires(count_fns<ftype>)
    inline void compute_partition(int64_t start_idx, int64_t end_idx) {
        // Fill the entire partition with the current value of in_window.
        for (int64_t i = start_idx; i < end_idx; i++) {
            int64_t idx = getv<int64_t>(sorted_idx, i);
            getv<int64_t>(out_arr, idx) = in_window;
        }
    }

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires(count_fns<ftype>)
    inline void compute_frame(int64_t i, int64_t lower_bound,
                              int64_t upper_bound) {
        // The aggrevate value is  the current value of in_window.
        int64_t idx = getv<int64_t>(sorted_idx, i);
        getv<int64_t>(out_arr, idx) = in_window;
    }

    // RATIO_TO_REPORT implementations.

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires(ratio_to_report<ftype> && !string_or_dict<ArrayType>)
    inline void init() {
        // At the start of each partition, reset the sum to zero.
        m1 = 0.0;
    }

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires(ratio_to_report<ftype> && !string_or_dict<ArrayType>)
    inline void enter(int64_t i) {
        int64_t idx = getv<int64_t>(sorted_idx, i);
        // When a non-null value enters the window, add it to the current sum.
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
        // Compute the mean of the values currently in the window.
        return m1 / in_window;
    }

    template <int ftype>
        requires(var<ftype>)
    inline double compute_moment_function() {
        // Compute the sample variance of the values currently in the window.
        return (m2 - (m1 * m1) / in_window) / (in_window - 1);
    }

    template <int ftype>
        requires(var_pop<ftype>)
    inline double compute_moment_function() {
        // Compute the population variance of the values currently in the
        // window.
        return (m2 - (m1 * m1) / in_window) / in_window;
    }

    template <int ftype>
        requires(stddev<ftype>)
    inline double compute_moment_function() {
        // Compute the sample stddev of the values currently in the window.
        return std::sqrt((m2 - (m1 * m1) / in_window) / (in_window - 1));
    }

    template <int ftype>
        requires(stddev_pop<ftype>)
    inline double compute_moment_function() {
        // Compute the population stddev of the values currently in the window.
        return std::sqrt((m2 - (m1 * m1) / in_window) / in_window);
    }

    template <int ftype, bodo_array_type::arr_type_enum ArrayType, typename T,
              Bodo_CTypes::CTypeEnum DType, window_frame_enum frame_type>
        requires(mean<ftype> || var<ftype> || var_pop<ftype> || stddev<ftype> ||
                 stddev_pop<ftype>)
    inline void init() {
        // At the start of each partition, reset the counters and sums to zero.
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
        // When a non-null entry enters the window, if it is a NaN then
        // increment the nan counter, otherwise increment the moment sums.
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
        // When a non-null entry leaves the window, if it is a NaN then
        // decrement the nan counter, otherwise decrement the moment sums.
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

    // Helper utility for FIRST/LAST/ANY/NTH_VALUE functions. Takes in
    // the lower/upper bounds of the current partition/frame and returns
    // the index where the value should be sought from

    template <int ftype>
        requires(any_value<ftype> || first<ftype>)
    inline int64_t get_value_fn_target_idx(int64_t lower_bound,
                                           int64_t upper_bound) {
        // For first/any value, return the value at the start of the partition.
        if (lower_bound < upper_bound) {
            return getv<int64_t>(sorted_idx, lower_bound);
        }
        return -1;
    }

    template <int ftype>
        requires(last<ftype>)
    inline int64_t get_value_fn_target_idx(int64_t lower_bound,
                                           int64_t upper_bound) {
        // For last value, return the value at the end of the partition.
        if (lower_bound < upper_bound) {
            return getv<int64_t>(sorted_idx, upper_bound - 1);
        }
        return -1;
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
    int64_t in_window;    // How many entries are currently in the window (usage
                          // depends on the ftype).
    int64_t nan_counter;  // How many NaN entries are currently in the window.
    double m1;            // The sum of elements currently in the window.
    double m2;  // The sum of squares of elements currently in the window.
    // Vectors to store the intermediary string values before they are placed
    // into a final array.
    std::shared_ptr<bodo::vector<std::string>> string_vector = nullptr;
    std::shared_ptr<bodo::vector<uint8_t>> null_vector = nullptr;
    // Array info to store the new indices when creating a dictionary encoded
    // array.
    std::shared_ptr<array_info> out_indices = nullptr;
};

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
                                      const int64_t group_end_idx,
                                      WindowAggfunc& aggfunc) {
    // Initialize the accumulator.
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
                                     const int64_t group_end_idx,
                                     WindowAggfunc& aggfunc) {
    // Initialize the accumulator.
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
                                     const int64_t group_end_idx,
                                     WindowAggfunc& aggfunc) {
    // Initialize the accumulator.
    aggfunc.init<window_func, ArrayType, T, DType,
                 window_frame_enum::CUMULATIVE>();
    // Entering = the index where the next value to enter the window frame
    // will come from.
    int64_t entering = group_end_idx + lo - 1;
    // Add all of the values into the accumulator that are part of the
    // suffix frame for the last row.
    for (int64_t i = group_end_idx - 1;
         i >= std::max(group_start_idx, group_end_idx + lo - 1); i--) {
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
    WindowAggfunc aggfunc(in_arr, out_arr, sorted_idx);
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
                    in_arr, out_arr, sorted_idx, *frame_hi, group_start_idx, i,
                    aggfunc);
            } else if (frame_hi == nullptr) {
                // Case 2: cumulative window frame from the end of
                // the partition to some row relative to the current row
                window_frame_suffix_computation<window_func, ArrayType, T,
                                                DType>(
                    in_arr, out_arr, sorted_idx, *frame_lo, group_start_idx, i,
                    aggfunc);
            } else {
                // Case 3: sliding window frame relative to the current row
                window_frame_sliding_computation<window_func, ArrayType, T,
                                                 DType>(
                    in_arr, out_arr, sorted_idx, *frame_lo, *frame_hi,
                    group_start_idx, i, aggfunc);
            }
            group_start_idx = i;
        }
        prev_group = curr_group;
    }
    // Repeat the process one more time on the final group
    if (frame_lo == nullptr) {
        window_frame_prefix_computation<window_func, ArrayType, T, DType>(
            in_arr, out_arr, sorted_idx, *frame_hi, group_start_idx, n,
            aggfunc);
    } else if (frame_hi == nullptr) {
        window_frame_suffix_computation<window_func, ArrayType, T, DType>(
            in_arr, out_arr, sorted_idx, *frame_lo, group_start_idx, n,
            aggfunc);
    } else {
        window_frame_sliding_computation<window_func, ArrayType, T, DType>(
            in_arr, out_arr, sorted_idx, *frame_lo, *frame_hi, group_start_idx,
            n, aggfunc);
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
        case Bodo_FTypes::first:
            window_frame_computation_bounds_handler<Bodo_FTypes::first,
                                                    ArrayType, T, DType>(
                in_arr, out_arr, sorted_groups, sorted_idx, frame_lo, frame_hi);
            break;
        case Bodo_FTypes::last:
            window_frame_computation_bounds_handler<Bodo_FTypes::last,
                                                    ArrayType, T, DType>(
                in_arr, out_arr, sorted_groups, sorted_idx, frame_lo, frame_hi);
            break;
        case Bodo_FTypes::var:
            window_frame_computation_bounds_handler<Bodo_FTypes::var, ArrayType,
                                                    T, DType>(
                in_arr, out_arr, sorted_groups, sorted_idx, frame_lo, frame_hi);
            break;
        case Bodo_FTypes::var_pop:
            window_frame_computation_bounds_handler<Bodo_FTypes::var_pop,
                                                    ArrayType, T, DType>(
                in_arr, out_arr, sorted_groups, sorted_idx, frame_lo, frame_hi);
            break;
        case Bodo_FTypes::std:
            window_frame_computation_bounds_handler<Bodo_FTypes::std, ArrayType,
                                                    T, DType>(
                in_arr, out_arr, sorted_groups, sorted_idx, frame_lo, frame_hi);
            break;
        case Bodo_FTypes::std_pop:
            window_frame_computation_bounds_handler<Bodo_FTypes::std_pop,
                                                    ArrayType, T, DType>(
                in_arr, out_arr, sorted_groups, sorted_idx, frame_lo, frame_hi);
            break;
        case Bodo_FTypes::mean:
            window_frame_computation_bounds_handler<Bodo_FTypes::mean,
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
        case Bodo_CTypes::INT128:
            window_frame_computation_ftype_handler<
                ArrayType, dtype_to_type<Bodo_CTypes::INT128>::type,
                Bodo_CTypes::INT128>(in_arr, out_arr, sorted_groups, sorted_idx,
                                     frame_lo, frame_hi, ftype);
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
