// Copyright (C) 2023 Bodo Inc. All rights reserved.

#ifndef _GROUPBY_AGG_FUNCS_H_INCLUDED
#define _GROUPBY_AGG_FUNCS_H_INCLUDED

#include "_array_utils.h"
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
int compare_list_string(std::vector<std::pair<std::string, bool>> const& v1,
                        std::vector<std::pair<std::string, bool>> const& v2) {
    size_t len1 = v1.size();
    size_t len2 = v2.size();
    size_t minlen = len1;
    if (len2 < len1) minlen = len2;
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

/**
 * @brief Convert the given value to a double.
 *
 * @tparam T The input type.
 * @tparam dtype The underlying dtype of the input type.
 * @param val The value to convert
 * @return Val as a double.
 */
template <typename T, int dtype>
inline typename std::enable_if<!is_decimal<dtype>::value, double>::type
to_double(T const& val) {
    return static_cast<double>(val);
}

template <typename T, int dtype>
inline typename std::enable_if<is_decimal<dtype>::value, double>::type
to_double(T const& val) {
    return decimal_to_double(val);
}

// Template definitions used by multiple aggregate functions

/**
 * This template is used for functions that take two values of the same dtype.
 * This is defined for the generic aggfunc that doesn't require any additional
 * arguments.
 */
template <typename T, int dtype, int ftype>
struct aggfunc {
    /**
     * Apply the function.
     * @param[in,out] first input value, and holds the result
     * @param[in] second input value.
     */
    static void apply(T& v1, T& v2) {}
};

/**
 * This template is used for functions that take an input value and
 * reduce its result to a boolean output.
 */
template <typename T, int dtype, int ftype, typename Enable = void>
struct bool_aggfunc {
    /**
     * Apply the function.
     * @param[in,out] current aggregate value, holds the result
     * @param[in] other input value.
     */
    static void apply(bool& v1, T& v2);
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
    static void apply(std::vector<std::pair<std::string, bool>>& v1,
                      std::vector<std::pair<std::string, bool>>& v2) {}
};

// sum

template <typename T, int dtype>
struct aggfunc<T, dtype, Bodo_FTypes::sum> {
    /**
     * Aggregation function for sum. Modifies current sum if value is not a nan
     *
     * @param[in,out] current sum value, and holds the result
     * @param second input value.
     */
    inline static void apply(T& v1, T& v2) {
        if (!isnan_alltype<T, dtype>(v2)) {
            v1 += v2;
        }
    }
};

template <typename T, int dtype, typename Enable = void>
struct bool_sum {
    /**
     * Aggregation function for sum of booleans. Increases total by 1 if the
     * value is not null and truthy.
     *
     * @param[in,out] current sum
     * @param second input value.
     */
    static void apply(int64_t& v1, T& v2);
};

template <typename T, int dtype>
struct bool_sum<T, dtype> {
    inline static void apply(int64_t& v1, T& v2) {
        if (v2) {
            v1 += 1;
        }
    }
};

template <>
struct aggliststring<Bodo_FTypes::sum> {
    inline static void apply(std::vector<std::pair<std::string, bool>>& v1,
                             std::vector<std::pair<std::string, bool>>& v2) {
        v1.insert(v1.end(), v2.begin(), v2.end());
    }
};

// min

template <typename T, int dtype>
struct aggfunc<T, dtype, Bodo_FTypes::min> {
    /**
     * Aggregation function for min. Modifies current min if value is not a nan
     *
     * @param[in,out] current min value (or nan for floats if no min value found
     * yet)
     * @param second input value.
     */
    inline static void apply(T& v1, T& v2) {
        if (!isnan_alltype<T, dtype>(v2)) {
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
    inline static void apply(std::vector<std::pair<std::string, bool>>& v1,
                             std::vector<std::pair<std::string, bool>>& v2) {
        if (compare_list_string(v1, v2) == 1) {
            v1 = v2;
        }
    }
};

// max

template <typename T, int dtype>
struct aggfunc<T, dtype, Bodo_FTypes::max> {
    /**
     * Aggregation function for max. Modifies current max if value is not a nan
     *
     * @param[in,out] current max value (or nan for floats if no max value found
     * yet)
     * @param second input value.
     */
    inline static void apply(T& v1, T& v2) {
        if (!isnan_alltype<T, dtype>(v2)) {
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
    inline static void apply(std::vector<std::pair<std::string, bool>>& v1,
                             std::vector<std::pair<std::string, bool>>& v2) {
        if (compare_list_string(v1, v2) == -1) {
            v1 = v2;
        }
    }
};

// prod
// Note: product of date and timedelta is not possible

template <typename T, int dtype>
struct aggfunc<T, dtype, Bodo_FTypes::prod> {
    /**
     * Aggregation function for product. Modifies current product if value is
     * not a nan
     *
     * @param[in,out] current product
     * @param second input value.
     */
    inline static void apply(T& v1, T& v2) {
        if (!isnan_alltype<T, dtype>(v2)) {
            v1 *= v2;
        }
    }
};

template <>
struct aggfunc<bool, Bodo_CTypes::_BOOL, Bodo_FTypes::prod> {
    inline static void apply(bool& v1, bool& v2) { v1 = v1 && v2; }
};

// idxmin

template <typename T, int dtype, typename Enable = void>
struct idxmin_agg {
    static void apply(T& v1, T& v2, uint64_t& index_pos, int64_t i);
};

template <typename T, int dtype>
struct idxmin_agg<
    T, dtype,
    typename std::enable_if<!std::is_floating_point<T>::value>::type> {
    inline static void apply(T& v1, T& v2, uint64_t& index_pos, int64_t i) {
        // TODO should it be >=?
        if (!isnan_alltype<T, dtype>(v2) && (v1 > v2)) {
            v1 = v2;
            index_pos = i;
        }
    }
};

// TODO: Should we get rid of the float specialization? This extra isna
// check is not needed for other types but may not be possible to optimize
// away.
template <typename T, int dtype>
struct idxmin_agg<
    T, dtype, typename std::enable_if<std::is_floating_point<T>::value>::type> {
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

// idxmax

template <typename T, int dtype, typename Enable = void>
struct idxmax_agg {
    static void apply(T& v1, T& v2, uint64_t& index_pos, int64_t i);
};

template <typename T, int dtype>
struct idxmax_agg<
    T, dtype,
    typename std::enable_if<!std::is_floating_point<T>::value>::type> {
    inline static void apply(T& v1, T& v2, uint64_t& index_pos, int64_t i) {
        // TODO should it be <=?
        if (!isnan_alltype<T, dtype>(v2) && (v1 < v2)) {
            v1 = v2;
            index_pos = i;
        }
    }
};

// TODO: Should we get rid of the float specialization? This extra isna
// check is not needed for other types but may not be possible to optimize
// away.
template <typename T, int dtype>
struct idxmax_agg<
    T, dtype, typename std::enable_if<std::is_floating_point<T>::value>::type> {
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

// boolor_agg

template <typename T, int dtype>
struct bool_aggfunc<T, dtype, Bodo_FTypes::boolor_agg,
                    typename std::enable_if<!is_decimal<dtype>::value>::type> {
    /**
     * Aggregation function for boolor_agg. Note this implementation
     * handles both integer and floating point data.
     *
     * @param[in,out] current aggregate value, holds the result
     * @param other input value.
     */
    inline static void apply(bool& v1, T& v2) { v1 = (v1 || (v2 != 0)); }
};

template <typename T, int dtype>
struct bool_aggfunc<T, dtype, Bodo_FTypes::boolor_agg,
                    typename std::enable_if<is_decimal<dtype>::value>::type> {
    /**
     * Aggregation function for boolor_agg. Note this implementation
     * handles only decimal data.
     *
     * @param[in,out] current aggregate value, holds the result
     * @param other input value.
     */
    // TODO: Compare decimal directly?
    inline static void apply(bool& v1, T& v2) {
        v1 = v1 || ((decimal_to_double(v2)) != 0.0);
    }
};

// last
template <typename T, int dtype>
struct aggfunc<T, dtype, Bodo_FTypes::last> {
    /**
     * Aggregation function for last. Always selects v2
     * if its not NaN.
     *
     * @param[in,out] current count
     * @param second input value.
     */
    inline static void apply(T& v1, T& v2) {
        if (!isnan_alltype<T, dtype>(v2)) {
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
    inline static void apply(std::vector<std::pair<std::string, bool>>& v1,
                             std::vector<std::pair<std::string, bool>>& v2) {
        v1 = v2;
    }
};

// count

template <typename T, int dtype>
struct count_agg {
    /**
     * Aggregation function for count. Increases count if value is not a nan
     *
     * @param[in,out] current count
     * @param second input value.
     */
    inline static void apply(int64_t& v1, T& v2) {
        if (!isnan_alltype<T, dtype>(v2)) {
            v1 += 1;
        }
    }
};

// size

template <typename T, int dtype>
struct size_agg {
    /**
     * Aggregation function for size. Increases size
     *
     * @param[in,out] current count
     * @param second input value.
     */
    inline static void apply(int64_t& v1, T& v2) { v1 += 1; }
};

// mean

template <typename T, int dtype>
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
        if (!isnan_alltype<T, dtype>(v2)) {
            v1 += to_double<T, dtype>(v2);
            count += 1;
        }
    }
};

// variance

template <typename T, int dtype>
struct var_agg {
    /**
     * Aggregation function for variance. Modifies count, mean and m2 (sum of
     * squares of differences from the current mean) based on the observed input
     * values. See
     * https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
     * for more information.
     *
     * @param[in] v2: observed value
     * @param[in,out] count: current number of observations
     * @param[in,out] mean_x: current mean
     * @param[in,out] m2: sum of squares of differences from the current mean
     */
    inline static void apply(T& v2, uint64_t& count, double& mean_x,
                             double& m2) {
        if (!isnan_alltype<T, dtype>(v2)) {
            double v2_double = to_double<T, dtype>(v2);
            count += 1;
            double delta = v2_double - mean_x;
            mean_x += delta / count;
            double delta2 = v2_double - mean_x;
            m2 += delta * delta2;
        }
    }
};

// ngroup

template <typename T, int dtype>
struct ngroup_agg {
    /**
     * Aggregation function for ngroup.
     * Assign v2 (group number) to v1 (output_column[i])
     *
     * @param v1 [out] output value
     * @param v2 input value.
     */
    static void apply(int64_t& v1, T& v2) { v1 = v2; }
};

#endif  // _GROUPBY_AGG_FUNCS_H_INCLUDED
