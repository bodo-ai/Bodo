// Copyright (C) 2023 Bodo Inc. All rights reserved.
#pragma once

#include <concepts>
#include <span>
#include "../_array_utils.h"
#include "../_bodo_common.h"
#include "_groupby_ftypes.h"

/**
 * The file contains the templated aggregate functions that are used
 * for the update step of groupby and will be inlined.
 *
 */

// Helper Functions used to implement the aggregate functions that
// are not shared with any other code.

/**
 * @brief Implements the comparison operator for lists of strings
 * equivalent to the comparison operator for strings.
 *
 * @param v1 The first list of strings.
 * @param v2 The second list of strings.
 * @return -1 if v1 < v2, 0 if v1=v2 and 1 if v1 > v2
 */
int compare_list_string(
    const std::span<const std::pair<std::string, bool>> v1,
    const std::span<const std::pair<std::string, bool>> v2) {
    size_t len1 = v1.size();
    size_t len2 = v2.size();
    size_t minlen = len1;
    if (len2 < len1)
        minlen = len2;
    for (size_t i = 0; i < minlen; i++) {
        bool bit1 = v1[i].second;
        bool bit2 = v2[i].second;
        if (bit1 && !bit2) {
            return 1;
        } else if (!bit1 && bit2) {
            return -1;
        } else if (v1[i] < v2[i]) {
            return -1;
        } else if (v1[i] > v2[i]) {
            return 1;
        }
    }
    if (len1 < len2) {
        return -1;
    } else if (len1 > len2) {
        return 1;
    } else {
        return 0;
    }
}

// Template definitions used by multiple aggregate functions

/**
 * This template is used for functions that take two values of the same dtype.
 * This is defined for the generic aggfunc that doesn't require any additional
 * arguments.
 */
template <typename T, Bodo_CTypes::CTypeEnum DType, int ftype>
struct aggfunc {
    /**
     * Apply the function.
     * @param[in,out] first input value, and holds the result
     * @param[in] second input value.
     */
    static void apply(T& v1, T& v2) {}
};

/**
 * This template is used for functions that have (possibly) different
 * input and output types, T_in and T_out, and don't require additional
 * arguments.
 */
template <typename T_out, typename T_in, Bodo_CTypes::CTypeEnum In_DType,
          int ftype>
struct casted_aggfunc {
    /**
     * Apply the function.
     * @param[in,out] v1: current aggregate value, holds the result
     * @param[in] v2: other input value.
     */
    static void apply(T_out& v1, T_in& v2) {}
};

/**
 * This template is used for functions that take an input value and
 * reduce its result to a boolean output.
 */
template <typename T, Bodo_CTypes::CTypeEnum DType, int ftype>
struct bool_aggfunc {
    /**
     * Apply the function.
     * @param[in,out] current aggregate value, holds the result
     * @param[in] other input value.
     */
    static void apply(const std::shared_ptr<array_info>& arr, int64_t idx,
                      T& v2) {}
};

/**
 * This template is used for common string functions that both take
 * in strings and output a string (e.g. sum, min, max)
 */

template <int ftype>
struct aggstring {
    /**
     * Apply the function.
     * @param[in,out] first input value, and holds the result
     * @param[in] second input value.
     */
    static void apply(std::string& v1, std::string& v2) {}
};

/**
 * This template is used for common operations that are performed
 * on dictionary encoded string arrays (e.g. min, max, last)
 */
template <int ftype>
struct aggdict {
    /**
     * Apply the function.
     * @param[in,out] first input index value to be updated.
     * @param[in] second input index value.
     * @param[in] first_string input string value.
     * @param[in] second_string input string value.
     */
    static void apply(int32_t& v1, int32_t& v2, std::string& s1,
                      std::string& s2) {}
};

/**
 * This template defines the common operations that are performed
 * on lists of strings.
 */
template <int ftype>
struct aggliststring {
    /**
     * Apply the function.
     * @param[in,out] first input value, and holds the result
     * @param[in] second input value.
     */
    template <typename Alloc1, typename Alloc2>
    static void apply(std::vector<std::pair<std::string, bool>, Alloc1>& v1,
                      std::vector<std::pair<std::string, bool>, Alloc2>& v2) {}
};

// sum

template <typename T, Bodo_CTypes::CTypeEnum DType>
struct aggfunc<T, DType, Bodo_FTypes::sum> {
    /**
     * Aggregation function for sum. Modifies current sum if value is not a nan
     *
     * @param[in,out] v1 sum value, and holds the result
     * @param v2 input value.
     */
    inline static void apply(T& v1, T& v2) {
        if (!isnan_alltype<T, DType>(v2)) {
            v1 += v2;
        }
    }
};

template <typename T_out, typename T_in, Bodo_CTypes::CTypeEnum In_DType>
struct casted_aggfunc<T_out, T_in, In_DType, Bodo_FTypes::sum> {
    /**
     * @brief Used for the case where we upcast integers to their 64-bit
     * variants to prevent overflow when summing up the values.
     *
     * @param[in, out] v1 Current sum value, holds the result.
     * @param v2 Input value
     */
    inline static void apply(T_out& v1, T_in& v2) {
        if (!isnan_alltype<T_in, In_DType>(v2)) {
            v1 += v2;
        }
    }
};

/**
 * Aggregation function for sum of booleans. Increases total by 1 if the
 * value is not null and truthy. This is used by count_if and sum.
 *
 * @param[in,out] current sum
 * @param second input value.
 */
inline static void bool_sum(int64_t& v1, bool& v2) {
    if (v2) {
        v1 += 1;
    }
}

template <>
struct aggliststring<Bodo_FTypes::sum> {
    template <typename Alloc1, typename Alloc2>
    inline static void apply(
        std::vector<std::pair<std::string, bool>, Alloc1>& v1,
        std::vector<std::pair<std::string, bool>, Alloc2>& v2) {
        v1.insert(v1.end(), v2.begin(), v2.end());
    }
};

// min

template <typename T, Bodo_CTypes::CTypeEnum DType>
struct aggfunc<T, DType, Bodo_FTypes::min> {
    /**
     * Aggregation function for min. Modifies current min if value is not a nan
     *
     * @param[in,out] current min value (or nan for floats if no min value found
     * yet)
     * @param second input value.
     */
    inline static void apply(T& v1, T& v2) {
        if (!isnan_alltype<T, DType>(v2)) {
            v1 = std::min(v2, v1);  // Max(x, Nan) = x
        }
    }
};

// TODO: Can this be merged with the integer implementation?
// Right now this requires too many modifications to the existing
// code.
template <>
struct aggstring<Bodo_FTypes::min> {
    inline static void apply(std::string& v1, std::string& v2) {
        v1 = std::min(v1, v2);
    }
};

template <>
struct aggdict<Bodo_FTypes::min> {
    inline static void apply(int32_t& v1, int32_t& v2, std::string& s1,
                             std::string& s2) {
        // TODO: Optimize on sorted dictionary to only compare integers
        if (s1.compare(s2) > 0) {
            v1 = v2;
        }
    }
};

template <>
struct aggliststring<Bodo_FTypes::min> {
    template <typename Alloc1, typename Alloc2>
    inline static void apply(
        std::vector<std::pair<std::string, bool>, Alloc1>& v1,
        std::vector<std::pair<std::string, bool>, Alloc2>& v2) {
        if (compare_list_string(v1, v2) == 1) {
            v1 = v2;
        }
    }
};

// nullable boolean implementation
template <>
struct bool_aggfunc<bool, Bodo_CTypes::_BOOL, Bodo_FTypes::min> {
    inline static void apply(const std::shared_ptr<array_info>& arr,
                             int64_t idx, bool& v2) {
        assert(arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL);
        bool v1 = GetBit(
            (uint8_t*)arr->data1<bodo_array_type::NULLABLE_INT_BOOL>(), idx);
        // And to get the output.
        v1 = v1 && v2;
        SetBitTo((uint8_t*)arr->data1<bodo_array_type::NULLABLE_INT_BOOL>(),
                 idx, v1);
    }
};

// max

template <typename T, Bodo_CTypes::CTypeEnum DType>
struct aggfunc<T, DType, Bodo_FTypes::max> {
    /**
     * Aggregation function for max. Modifies current max if value is not a nan
     *
     * @param[in,out] current max value (or nan for floats if no max value found
     * yet)
     * @param second input value.
     */
    inline static void apply(T& v1, T& v2) {
        if (!isnan_alltype<T, DType>(v2)) {
            v1 = std::max(v2, v1);  // Max(x, Nan) = x
        }
    }
};

// TODO: Can this be merged with the integer implementation?
// Right now this requires too many modifications to the existing
// code.
template <>
struct aggstring<Bodo_FTypes::max> {
    inline static void apply(std::string& v1, std::string& v2) {
        v1 = std::max(v1, v2);
    }
};

template <>
struct aggdict<Bodo_FTypes::max> {
    inline static void apply(int32_t& v1, int32_t& v2, std::string& s1,
                             std::string& s2) {
        // TODO: Optimize on sorted dictionary to only compare integers
        if (s1.compare(s2) < 0) {
            v1 = v2;
        }
    }
};

template <>
struct aggliststring<Bodo_FTypes::max> {
    template <typename Alloc1, typename Alloc2>
    inline static void apply(
        std::vector<std::pair<std::string, bool>, Alloc1>& v1,
        std::vector<std::pair<std::string, bool>, Alloc2>& v2) {
        if (compare_list_string(v1, v2) == -1) {
            v1 = v2;
        }
    }
};

// nullable boolean implementation
template <>
struct bool_aggfunc<bool, Bodo_CTypes::_BOOL, Bodo_FTypes::max> {
    inline static void apply(const std::shared_ptr<array_info>& arr,
                             int64_t idx, bool& v2) {
        assert(arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL);
        bool v1 = GetBit(
            (uint8_t*)arr->data1<bodo_array_type::NULLABLE_INT_BOOL>(), idx);
        // OR to get the output.
        v1 = v1 || v2;
        SetBitTo((uint8_t*)arr->data1<bodo_array_type::NULLABLE_INT_BOOL>(),
                 idx, v1);
    }
};

// prod
// Note: product of date and timedelta is not possible

template <typename T, Bodo_CTypes::CTypeEnum DType>
struct aggfunc<T, DType, Bodo_FTypes::prod> {
    /**
     * Aggregation function for product. Modifies current product if value is
     * not a nan
     *
     * @param[in,out] current product
     * @param second input value.
     */
    inline static void apply(T& v1, T& v2) {
        if (!isnan_alltype<T, DType>(v2)) {
            v1 *= v2;
        }
    }
};

template <>
struct aggfunc<bool, Bodo_CTypes::_BOOL, Bodo_FTypes::prod> {
    inline static void apply(bool& v1, bool& v2) { v1 = v1 && v2; }
};

// nullable boolean implementation
template <>
struct bool_aggfunc<bool, Bodo_CTypes::_BOOL, Bodo_FTypes::prod> {
    inline static void apply(const std::shared_ptr<array_info>& arr,
                             int64_t idx, bool& v2) {
        assert(arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL);
        bool v1 = GetBit(
            (uint8_t*)arr->data1<bodo_array_type::NULLABLE_INT_BOOL>(), idx);
        v1 = v1 && v2;
        SetBitTo((uint8_t*)arr->data1<bodo_array_type::NULLABLE_INT_BOOL>(),
                 idx, v1);
    }
};

// idxmin

template <typename T, Bodo_CTypes::CTypeEnum DType>
struct idxmin_agg {
    static void apply(T& v1, T& v2, uint64_t& index_pos, int64_t i);
};

template <typename T, Bodo_CTypes::CTypeEnum DType>
    requires(!std::floating_point<T>)
struct idxmin_agg<T, DType> {
    inline static void apply(T& v1, T& v2, uint64_t& index_pos, int64_t i) {
        // TODO should it be >=?
        if (!isnan_alltype<T, DType>(v2) && (v1 > v2)) {
            v1 = v2;
            index_pos = i;
        }
    }
};

// TODO: Should we get rid of the float specialization? This extra isna
// check is not needed for other types but may not be possible to optimize
// away.
template <typename T, Bodo_CTypes::CTypeEnum DType>
    requires std::floating_point<T>
struct idxmin_agg<T, DType> {
    inline static void apply(T& v1, T& v2, uint64_t& index_pos, int64_t i) {
        if (!isnan(v2)) {
            // v1 is initialized as NaN
            // TODO should it be >=?
            if (isnan(v1) || (v1 > v2)) {
                v1 = v2;
                index_pos = i;
            }
        }
    }
};

inline static void idxmin_string(std::string& v1, std::string& v2,
                                 uint64_t& index_pos, int64_t i) {
    if (v1.compare(v2) > 0) {
        v1 = v2;
        index_pos = i;
    }
}

inline static void idxmin_dict(int32_t& v1, int32_t& v2, std::string& s1,
                               std::string& s2, uint64_t& index_pos,
                               int64_t i) {
    // TODO: Optimize on sorted dictionary to only compare integers
    if (s1.compare(s2) > 0) {
        v1 = v2;
        index_pos = i;
    }
}

inline static void idxmin_bool(const std::shared_ptr<array_info>& arr,
                               int64_t grp_num, bool& v2, uint64_t& index_pos,
                               int64_t i) {
    bool v1 = GetBit((uint8_t*)arr->data1(), grp_num);
    if (v2 < v1) {
        SetBitTo((uint8_t*)arr->data1(), grp_num, v2);
        index_pos = i;
    }
}

// idxmax

template <typename T, Bodo_CTypes::CTypeEnum DType>
struct idxmax_agg {
    static void apply(T& v1, T& v2, uint64_t& index_pos, int64_t i);
};

template <typename T, Bodo_CTypes::CTypeEnum DType>
    requires(!std::floating_point<T>)
struct idxmax_agg<T, DType> {
    inline static void apply(T& v1, T& v2, uint64_t& index_pos, int64_t i) {
        // TODO should it be <=?
        if (!isnan_alltype<T, DType>(v2) && (v1 < v2)) {
            v1 = v2;
            index_pos = i;
        }
    }
};

// TODO: Should we get rid of the float specialization? This extra isna
// check is not needed for other types but may not be possible to optimize
// away.
template <typename T, Bodo_CTypes::CTypeEnum DType>
    requires std::floating_point<T>
struct idxmax_agg<T, DType> {
    inline static void apply(T& v1, T& v2, uint64_t& index_pos, int64_t i) {
        if (!isnan(v2)) {
            // v1 is initialized as NaN
            // TODO should it be <=?
            if (isnan(v1) || (v1 < v2)) {
                v1 = v2;
                index_pos = i;
            }
        }
    }
};

inline static void idxmax_string(std::string& v1, std::string& v2,
                                 uint64_t& index_pos, int64_t i) {
    if (v1.compare(v2) < 0) {
        v1 = v2;
        index_pos = i;
    }
}

inline static void idxmax_dict(int32_t& v1, int32_t& v2, std::string& s1,
                               std::string& s2, uint64_t& index_pos,
                               int64_t i) {
    // TODO: Optimize on sorted dictionary to only compare integers
    if (s1.compare(s2) < 0) {
        v1 = v2;
        index_pos = i;
    }
}

inline static void idxmax_bool(const std::shared_ptr<array_info>& arr,
                               int64_t grp_num, bool& v2, uint64_t& index_pos,
                               int64_t i) {
    bool v1 = GetBit((uint8_t*)arr->data1(), grp_num);
    if (v2 > v1) {
        SetBitTo((uint8_t*)arr->data1(), grp_num, v2);
        index_pos = i;
    }
}

// boolor_agg

template <typename T, Bodo_CTypes::CTypeEnum DType>
    requires(!decimal<DType>)
struct bool_aggfunc<T, DType, Bodo_FTypes::boolor_agg> {
    /**
     * Aggregation function for boolor_agg. Note this implementation
     * handles both integer and floating point data.
     *
     * @param[in,out] arr The array holding the current aggregation info. This
     * is necessary because nullable booleans have 1 bit per boolean
     * @param idx The index to load/store for array.
     * @param v2 other input value.
     */
    inline static void apply(const std::shared_ptr<array_info>& arr,
                             int64_t idx, T& v2) {
        bool v1 = GetBit((uint8_t*)arr->data1(), idx);
        v1 = (v1 || (v2 != 0));
        SetBitTo((uint8_t*)arr->data1(), idx, v1);
    }
};

template <typename T, Bodo_CTypes::CTypeEnum DType>
    requires decimal<DType>
struct bool_aggfunc<T, DType, Bodo_FTypes::boolor_agg> {
    /**
     * Aggregation function for boolor_agg. Note this implementation
     * handles decimal data.
     *
     * @param[in,out] arr The array holding the current aggregation info. This
     * is necessary because nullable booleans have 1 bit per boolean
     * @param idx The index to load/store for array.
     * @param v2 other input value.
     */
    // TODO: Compare decimal directly?
    inline static void apply(const std::shared_ptr<array_info>& arr,
                             int64_t idx, T& v2) {
        bool v1 = GetBit((uint8_t*)arr->data1(), idx);
        v1 = v1 || ((decimal_to_double(v2)) != 0.0);
        SetBitTo((uint8_t*)arr->data1(), idx, v1);
    }
};

// booland_agg

template <typename T, Bodo_CTypes::CTypeEnum DType>
    requires(!decimal<DType>)
struct bool_aggfunc<T, DType, Bodo_FTypes::booland_agg> {
    /**
     * Aggregation function for booland_agg. Note this implementation
     * handles both integer and floating point data.
     *
     * @param[in,out] arr The array holding the current aggregation info. This
     * is necessary because nullable booleans have 1 bit per boolean
     * @param idx The index to load/store for array.
     * @param v2 other input value.
     */
    inline static void apply(const std::shared_ptr<array_info>& arr,
                             int64_t idx, T& v2) {
        bool v1 = GetBit((uint8_t*)arr->data1(), idx);
        v1 = (v1 && (v2 != 0));
        SetBitTo((uint8_t*)arr->data1(), idx, v1);
    }
};

template <typename T, Bodo_CTypes::CTypeEnum DType>
    requires decimal<DType>
struct bool_aggfunc<T, DType, Bodo_FTypes::booland_agg> {
    /**
     * Aggregation function for booland_agg. Note this implementation
     * handles decimal data data.
     *
     * @param[in,out] arr The array holding the current aggregation info. This
     * is necessary because nullable booleans have 1 bit per boolean
     * @param idx The index to load/store for array.
     * @param v2 other input value.
     */
    // TODO: Compare decimal directly?
    inline static void apply(const std::shared_ptr<array_info>& arr,
                             int64_t idx, T& v2) {
        bool v1 = GetBit((uint8_t*)arr->data1(), idx);
        v1 = v1 && ((decimal_to_double(v2)) != 0.0);
        SetBitTo((uint8_t*)arr->data1(), idx, v1);
    }
};

// last
template <typename T, Bodo_CTypes::CTypeEnum DType>
struct aggfunc<T, DType, Bodo_FTypes::last> {
    /**
     * Aggregation function for last. Always selects v2
     * if its not NaN.
     *
     * @param[in,out] current count
     * @param second input value.
     */
    inline static void apply(T& v1, T& v2) {
        if (!isnan_alltype<T, DType>(v2)) {
            v1 = v2;
        }
    }
};

// TODO: Fuse all implementations into a generic
// aggfunc since we don't need any type specific behavior.
// Right now this requires too much code rewrite to only
// benefit aggdict (since we can avoid string allocations).

template <>
struct aggstring<Bodo_FTypes::last> {
    inline static void apply(std::string& v1, std::string& v2) { v1 = v2; }
};

template <>
struct aggliststring<Bodo_FTypes::last> {
    template <typename Alloc1, typename Alloc2>
    inline static void apply(
        std::vector<std::pair<std::string, bool>, Alloc1>& v1,
        std::vector<std::pair<std::string, bool>, Alloc2>& v2) {
        v1 = v2;
    }
};

// nullable boolean implementation
template <>
struct bool_aggfunc<bool, Bodo_CTypes::_BOOL, Bodo_FTypes::last> {
    inline static void apply(const std::shared_ptr<array_info>& arr,
                             int64_t idx, bool& v2) {
        assert(arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL);
        SetBitTo((uint8_t*)arr->data1<bodo_array_type::NULLABLE_INT_BOOL>(),
                 idx, v2);
    }
};

// count

template <typename T, Bodo_CTypes::CTypeEnum DType>
struct count_agg {
    /**
     * Aggregation function for count. Increases count if value is not a nan
     *
     * @param[in,out] current count
     * @param second input value.
     */
    inline static void apply(int64_t& v1, T& v2) {
        if (!isnan_alltype<T, DType>(v2)) {
            v1 += 1;
        }
    }
};

// size

template <typename T, Bodo_CTypes::CTypeEnum DType>
struct size_agg {
    /**
     * Aggregation function for size. Increases size
     *
     * @param[in,out] current count
     */
    inline static void apply(int64_t& v1) { v1 += 1; }
};

// mean

template <typename T, Bodo_CTypes::CTypeEnum DType>
struct mean_agg {
    /**
     * Aggregation function for mean. Modifies count and sum of observed input
     * values
     *
     * @param[in,out] contains the current sum of observed values
     * @param an observed input value
     * @param[in,out] count: current number of observations
     */
    inline static void apply(double& v1, T& v2, uint64_t& count) {
        if (!isnan_alltype<T, DType>(v2)) {
            v1 += to_double<T, DType>(v2);
            count += 1;
        }
    }
};

// variance

template <typename T, Bodo_CTypes::CTypeEnum DType>
struct var_agg {
    /**
     * Aggregation function for variance. Modifies count, mean and m2 (sum of
     * squares of differences from the current mean) based on the observed input
     * values. See
     * https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
     * for more information.
     *
     * @param[in] v: observed value
     * @param[in,out] count: current number of observations
     * @param[in,out] mean_x: current mean
     * @param[in,out] m2: sum of squares of differences from the current mean
     */
    inline static void apply(T val, uint64_t& count, double& mean_x,
                             double& m2) {
        if (!isnan_alltype<T, DType>(val)) {
            double val_double = to_double<T, DType>(val);
            count += 1;
            double delta = val_double - mean_x;
            mean_x += delta / count;
            double delta2 = val_double - mean_x;
            m2 += delta * delta2;
        }
    }
};

// Skew

template <typename T, Bodo_CTypes::CTypeEnum DType>
struct skew_agg {
    /**
     * Aggregation function for skew. The same principle as variance, but
     * used to calculate the third moment.
     *
     * @param[in] val: observed value
     * @param[in,out] count: current number of observations
     * @param[in,out] m1: current sum of elements
     * @param[in,out] m2: current sum of squares of elements
     * @param[in,out] m3: current sum of cubes of elements
     */
    inline static void apply(T val, uint64_t& count, double& m1, double& m2,
                             double& m3) {
        if (!isnan_alltype<T, DType>(val)) {
            double val_double = to_double<T, DType>(val);
            count += 1;
            m1 += val_double;
            m2 += val_double * val_double;
            m3 += val_double * val_double * val_double;
        }
    }
};

// kurtosis

template <typename T, Bodo_CTypes::CTypeEnum DType>
struct kurt_agg {
    /**
     * Aggregation function for kurtosis. The same principle as variance, but
     * used to calculate the fourth moment
     *
     * @param[in] val: observed value
     * @param[in,out] count: current number of observations
     * @param[in,out] m1: current sum of elements
     * @param[in,out] m2: current sum of squares of elements
     * @param[in,out] m3: current sum of cubes of elements
     * @param[in,out] m4: current sum of fourths of elements
     */
    inline static void apply(T val, uint64_t& count, double& m1, double& m2,
                             double& m3, double& m4) {
        if (!isnan_alltype<T, DType>(val)) {
            double val_double = to_double<T, DType>(val);
            count += 1;
            m1 += val_double;
            m2 += val_double * val_double;
            m3 += val_double * val_double * val_double;
            m4 += val_double * val_double * val_double * val_double;
        }
    }
};

// boolxor_agg

template <typename T, Bodo_CTypes::CTypeEnum DType>
struct boolxor_agg {
    /**
     * Aggregation function for boolxor_agg. The goal is to
     * count the number of non-null occurrences and the number of true
     * occurrences.
     *
     * @param T: input value
     * @param[in,out] one_arr: boolean array indicating which groups have 1+
     * nonzero observations
     * @param[in,out] two_arr: boolean array indicating which groups have 2+
     * nonzero observations
     * @param[in] i_grp: index of the corresponding group
     */
    inline static void apply(const T val,
                             const std::shared_ptr<array_info>& one_arr,
                             const std::shared_ptr<array_info>& two_arr,
                             int64_t i_grp) {
        if (val != 0) {
            bool old_one = GetBit((uint8_t*)one_arr->data1(), i_grp);
            SetBitTo((uint8_t*)one_arr->data1(), i_grp, true);
            SetBitTo((uint8_t*)two_arr->data1(), i_grp, old_one);
        }
    }
};

// bitor_agg

template <typename T_out, typename T_in, Bodo_CTypes::CTypeEnum DType>
    requires std::integral<T_in> && std::same_as<T_in, T_out>
struct casted_aggfunc<T_out, T_in, DType, Bodo_FTypes::bitor_agg> {
    /**
     * Applies BITOR_AGG for *integer* inputs.
     * For integers, T_out and T_in should be the same dtype.
     *
     * @param[in,out] v1: current aggregate value, holds the result
     * @param[in] v2: other input value.
     */
    static void apply(T_out& v1, T_in& v2) { v1 |= v2; }
};

template <typename T_in, Bodo_CTypes::CTypeEnum DType>
    requires std::floating_point<T_in>
struct casted_aggfunc<int64_t, T_in, DType, Bodo_FTypes::bitor_agg> {
    /**
     * Applies BITOR_AGG for *floating point* inputs.
     *
     * @param[in,out] v1: current aggregate value, holds the result
     * @param[in] v2: other input value.
     */
    static void apply(int64_t& v1, T_in& v2) { v1 |= std::lround(v2); }
};

// bitand_agg

template <typename T_out, typename T_in, Bodo_CTypes::CTypeEnum DType>
    requires std::integral<T_in> && std::same_as<T_in, T_out>
struct casted_aggfunc<T_out, T_in, DType, Bodo_FTypes::bitand_agg> {
    /**
     * Applies BITAND_AGG for *integer* inputs.
     * For integers, T_out and T_in should be the same dtype.
     *
     * @param[in,out] v1: current aggregate value, holds the result
     * @param[in] v2: other input value.
     */
    static void apply(T_out& v1, T_in& v2) { v1 &= v2; }
};

template <typename T_in, Bodo_CTypes::CTypeEnum DType>
    requires std::floating_point<T_in>
struct casted_aggfunc<int64_t, T_in, DType, Bodo_FTypes::bitand_agg> {
    /**
     * Applies BITAND_AGG for *floating point* inputs.
     *
     * @param[in,out] v1: current aggregate value, holds the result
     * @param[in] v2: other input value.
     */
    static void apply(int64_t& v1, T_in& v2) { v1 &= std::lround(v2); }
};

// bitand_agg

template <typename T_out, typename T_in, Bodo_CTypes::CTypeEnum DType>
    requires std::integral<T_in> && std::same_as<T_in, T_out>
struct casted_aggfunc<T_out, T_in, DType, Bodo_FTypes::bitxor_agg> {
    /**
     * Applies BITXOR_AGG for *integer* inputs.
     * For integers, T_out and T_in should be the same dtype.
     *
     * @param[in,out] v1: current aggregate value, holds the result
     * @param[in] v2: other input value.
     */
    static void apply(T_out& v1, T_in& v2) { v1 ^= v2; }
};

template <typename T_in, Bodo_CTypes::CTypeEnum DType>
    requires std::floating_point<T_in>
struct casted_aggfunc<int64_t, T_in, DType, Bodo_FTypes::bitxor_agg> {
    /**
     * Applies BITXOR_AGG for *floating point* inputs.
     *
     * @param[in,out] v1: current aggregate value, holds the result
     * @param[in] v2: other input value.
     */
    static void apply(int64_t& v1, T_in& v2) { v1 ^= std::lround(v2); }
};

// Non inlined operations over multiple columns

/**
 * @brief Perform an idx*** column comparison for a column. This is used when we
 * need to find the "first" value produced by a min_row_number_filter() call
 * that contains several orderby columns, each of which may have separate types
 * and different NA first/last or ascending/descending order. This returns true
 * if the comparison is not a tie and false if the comparison is a tie. This is
 * used to signal if we should continue searching that later columns.
 *
 * @param[in, out] out_arr The output index array. This is used to load the
 * current member index of the "leading" value and store the new index if the
 * new value is a valid replacement.
 * @param grp_num The current group number used to load from out_arr.
 * @param[in] in_arr The input array used to compare value.
 * @param in_idx The index of the new row to compare.
 * @param asc Is this in ascending order? Here asc=True means we want to do a
 * min and asc=False means we want to do a max.
 * @param na_pos Are NAs placed last. If true this means NA values will be
 * replaced with any non-NA value. If false this means non-NA values will be
 * replaced with any NA value. Not NaN is the largest float and not NA.
 * @return true The comparison is not a tie.
 * @return false The comparison is a tie. If there are more columns we should
 * continue iterating.
 */
bool idx_compare_column(const std::shared_ptr<array_info>& out_arr,
                        int64_t grp_num,
                        const std::shared_ptr<array_info>& in_arr,
                        int64_t in_idx, bool asc, bool na_pos);
