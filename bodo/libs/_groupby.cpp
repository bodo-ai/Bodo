// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include "_groupby.h"
#include <functional>
#include <limits>
#include "_array_hash.h"
#include "_array_operations.h"
#include "_array_utils.h"
#include "_decimal_ext.h"
#include "_distributed.h"
#include "_murmurhash3.h"
#include "_shuffle.h"

#undef DEBUG_GROUPBY
#undef DEBUG_GROUPBY_SYMBOL
#undef DEBUG_GROUPBY_FULL

/**
 * Enum of aggregation, combine and eval functions used by groubpy.
 * Some functions like sum can be used for multiple purposes, like aggregation
 * and combine. Some operations like sum don't need eval.
 */
struct Bodo_FTypes {
    // !!! IMPORTANT: this is supposed to match the positions in
    // supported_agg_funcs in aggregate.py
    enum FTypeEnum {
        sum,
        count,
        nunique,
        median,
        cumsum,
        cumprod,
        cummin,
        cummax,
        mean,
        min,
        max,
        prod,
        first,
        last,
        idxmin,
        idxmax,
        var,
        std,
        udf,
        num_funcs,  // num_funcs is used to know how many functions up to this
                    // point
        mean_eval,
        var_eval,
        std_eval
    };
};

static std::vector<Bodo_FTypes::FTypeEnum> combine_funcs(
    Bodo_FTypes::num_funcs);

void groupby_init() {
    static bool initialized = false;
    if (initialized) {
        Bodo_PyErr_SetString(PyExc_RuntimeError, "groupby already initialized");
        return;
    }
    initialized = true;

    // this mapping is used by BasicColSet operations to know what combine
    // function to use for a given aggregation function
    combine_funcs[Bodo_FTypes::sum] = Bodo_FTypes::sum;
    combine_funcs[Bodo_FTypes::count] = Bodo_FTypes::sum;
    combine_funcs[Bodo_FTypes::mean] =
        Bodo_FTypes::sum;  // sum totals and counts
    combine_funcs[Bodo_FTypes::min] = Bodo_FTypes::min;
    combine_funcs[Bodo_FTypes::max] = Bodo_FTypes::max;
    combine_funcs[Bodo_FTypes::prod] = Bodo_FTypes::prod;
    combine_funcs[Bodo_FTypes::first] = Bodo_FTypes::first;
    combine_funcs[Bodo_FTypes::last] = Bodo_FTypes::last;
}

/**
 * Function pointer for groupby update and combine operations that are
 * executed in JIT-compiled code (also see udfinfo_t).
 *
 * @param input table
 * @param output table
 * @param row to group mapping (tells to which group -row in output table-
          the row i of input table goes to)
 */
typedef void (*udf_table_op_fn)(table_info* in_table, table_info* out_table,
                                int64_t* row_to_group);
/**
 * Function pointer for groupby eval operation that is executed in JIT-compiled
 * code (also see udfinfo_t).
 *
 * @param table containing the output columns and reduction variables columns
 */
typedef void (*udf_eval_fn)(table_info*);

/*
 * This struct stores info that is used when groupby.agg() has JIT-compiled
 * user-defined functions. Such JIT-compiled code will be invoked by the C++
 * library via function pointers.
 */
struct udfinfo_t {
    /*
     * This empty table is used to tell the C++ library the types to use
     * to allocate the columns (output and redvar) for udfs
     */
    table_info* udf_table_dummy;
    /*
     * Function pointer to "update" code which performs the initial
     * local groupby and aggregation.
     */
    udf_table_op_fn update;
    /*
     * Function pointer to "combine" code which combines the results
     * after shuffle.
     */
    udf_table_op_fn combine;
    /*
     * Function pointer to "eval" code which performs post-processing and
     * sets the final output value for each group.
     */
    udf_eval_fn eval;
};

/**
 * This template is used for functions that take two values of the same dtype.
 */
template <typename T, int dtype, int ftype, typename Enable = void>
struct aggfunc {
    /**
     * Apply the function.
     * @param[in,out] first input value, and holds the result
     * @param[in] second input value.
     */
    static void apply(T& v1, T& v2) {}
};

template <typename T, int dtype>
struct aggfunc<
    T, dtype, Bodo_FTypes::sum,
    typename std::enable_if<!std::is_floating_point<T>::value &&
                            is_datetime_timedelta<dtype>::value>::type> {
    /**
     * Aggregation function for sum. Modifies current sum if value is not a nan
     *
     * @param[in,out] current sum value, and holds the result
     * @param second input value.
     */
    static void apply(T& v1, T& v2) {
        if (v2 != std::numeric_limits<T>::min()) v1 += v2;
    }
};

template <typename T, int dtype>
struct aggfunc<
    T, dtype, Bodo_FTypes::sum,
    typename std::enable_if<!std::is_floating_point<T>::value &&
                            !is_datetime_timedelta<dtype>::value>::type> {
    /**
     * Aggregation function for sum. Modifies current sum if value is not a nan
     *
     * @param[in,out] current sum value, and holds the result
     * @param second input value.
     */
    static void apply(T& v1, T& v2) { v1 += v2; }
};

template <typename T, int dtype>
struct aggfunc<
    T, dtype, Bodo_FTypes::sum,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
    static void apply(T& v1, T& v2) {
        if (!isnan(v2)) v1 += v2;
    }
};

// min

template <typename T, int dtype>
struct aggfunc<
    T, dtype, Bodo_FTypes::min,
    typename std::enable_if<!std::is_floating_point<T>::value &&
                            is_datetime_timedelta<dtype>::value>::type> {
    /**
     * Aggregation function for min. Modifies current min if value is not a nan
     *
     * @param[in,out] current min value (or nan for floats if no min value found
     * yet)
     * @param second input value.
     */
    static void apply(T& v1, T& v2) {
        if (v2 != std::numeric_limits<T>::min()) v1 = std::min(v1, v2);
    }
};

template <typename T, int dtype>
struct aggfunc<
    T, dtype, Bodo_FTypes::min,
    typename std::enable_if<!std::is_floating_point<T>::value &&
                            !is_datetime_timedelta<dtype>::value>::type> {
    /**
     * Aggregation function for min. Modifies current min if value is not a nan
     *
     * @param[in,out] current min value (or nan for floats if no min value found
     * yet)
     * @param second input value.
     */
    static void apply(T& v1, T& v2) { v1 = std::min(v1, v2); }
};

template <typename T, int dtype>
struct aggfunc<
    T, dtype, Bodo_FTypes::min,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
    static void apply(T& v1, T& v2) {
        if (!isnan(v2))
            v1 = std::min(v2, v1);  // std::min(x,NaN) = x
                                    // (v1 is initialized as NaN)
    }
};

// max

template <typename T, int dtype>
struct aggfunc<
    T, dtype, Bodo_FTypes::max,
    typename std::enable_if<!std::is_floating_point<T>::value &&
                            is_datetime_timedelta<dtype>::value>::type> {
    /**
     * Aggregation function for max. Modifies current max if value is not a nan
     *
     * @param[in,out] current max value (or nan for floats if no max value found
     * yet)
     * @param second input value.
     */
    static void apply(T& v1, T& v2) {
        if (v2 != std::numeric_limits<T>::min()) v1 = std::max(v1, v2);
    }
};

template <typename T, int dtype>
struct aggfunc<
    T, dtype, Bodo_FTypes::max,
    typename std::enable_if<!std::is_floating_point<T>::value &&
                            !is_datetime_timedelta<dtype>::value>::type> {
    /**
     * Aggregation function for max. Modifies current max if value is not a nan
     *
     * @param[in,out] current max value (or nan for floats if no max value found
     * yet)
     * @param second input value.
     */
    static void apply(T& v1, T& v2) { v1 = std::max(v1, v2); }
};

template <typename T, int dtype>
struct aggfunc<
    T, dtype, Bodo_FTypes::max,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
    static void apply(T& v1, T& v2) {
        if (!isnan(v2)) {
            v1 = std::max(v2, v1);  // std::max(x,NaN) = x
                                    // (v1 is initialized as NaN)
        }
    }
};

// prod
// product of date and timedelta is not possible so no need to support it

template <typename T, int dtype>
struct aggfunc<
    T, dtype, Bodo_FTypes::prod,
    typename std::enable_if<!std::is_floating_point<T>::value>::type> {
    /**
     * Aggregation function for product. Modifies current product if value is
     * not a nan
     *
     * @param[in,out] current product
     * @param second input value.
     */
    static void apply(T& v1, T& v2) { v1 *= v2; }
};

template <>
struct aggfunc<bool, Bodo_CTypes::_BOOL, Bodo_FTypes::prod> {
    static void apply(bool& v1, bool& v2) { v1 = v1 && v2; }
};

template <typename T, int dtype>
struct aggfunc<
    T, dtype, Bodo_FTypes::prod,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
    static void apply(T& v1, T& v2) {
        if (!isnan(v2)) v1 *= v2;
    }
};

// idxmin

template <typename T, int dtype, typename Enable = void>
struct idxmin_agg {
    static void apply(T& v1, T& v2, uint64_t& index_pos, int64_t i);
};

template <typename T, int dtype>
struct idxmin_agg<
    T, dtype,
    typename std::enable_if<!std::is_floating_point<T>::value &&
                            is_datetime_timedelta<dtype>::value>::type> {
    static void apply(T& v1, T& v2, uint64_t& index_pos, int64_t i) {
        // TODO should it be >=?
        if ((v2 != std::numeric_limits<T>::max()) && (v1 > v2)) {
            v1 = v2;
            index_pos = i;
        }
    }
};

template <typename T, int dtype>
struct idxmin_agg<
    T, dtype,
    typename std::enable_if<!std::is_floating_point<T>::value &&
                            !is_datetime_timedelta<dtype>::value>::type> {
    static void apply(T& v1, T& v2, uint64_t& index_pos, int64_t i) {
        // TODO should it be >=?
        if (v1 > v2) {
            v1 = v2;
            index_pos = i;
        }
    }
};

template <typename T, int dtype>
struct idxmin_agg<
    T, dtype, typename std::enable_if<std::is_floating_point<T>::value>::type> {
    static void apply(T& v1, T& v2, uint64_t& index_pos, int64_t i) {
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

// idxmax

template <typename T, int dtype, typename Enable = void>
struct idxmax_agg {
    static void apply(T& v1, T& v2, uint64_t& index_pos, int64_t i);
};

template <typename T, int dtype>
struct idxmax_agg<
    T, dtype,
    typename std::enable_if<!std::is_floating_point<T>::value &&
                            is_datetime_timedelta<dtype>::value>::type> {
    static void apply(T& v1, T& v2, uint64_t& index_pos, int64_t i) {
        // TODO should it be <=?
        if ((v2 != std::numeric_limits<T>::min()) && (v1 < v2)) {
            v1 = v2;
            index_pos = i;
        }
    }
};

template <typename T, int dtype>
struct idxmax_agg<
    T, dtype,
    typename std::enable_if<!std::is_floating_point<T>::value &&
                            !is_datetime_timedelta<dtype>::value>::type> {
    static void apply(T& v1, T& v2, uint64_t& index_pos, int64_t i) {
        // TODO should it be <=?
        if (v1 < v2) {
            v1 = v2;
            index_pos = i;
        }
    }
};

template <typename T, int dtype>
struct idxmax_agg<
    T, dtype, typename std::enable_if<std::is_floating_point<T>::value>::type> {
    static void apply(T& v1, T& v2, uint64_t& index_pos, int64_t i) {
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

// aggstring: support for the string operations (sum, min, max)

template <int ftype, typename Enable = void>
struct aggstring {
    /**
     * Apply the function.
     * @param[in,out] first input value, and holds the result
     * @param[in] second input value.
     */
    static void apply(std::string& v1, std::string& v2) {}
};

template <>
struct aggstring<Bodo_FTypes::sum> {
    static void apply(std::string& v1, std::string& v2) { v1 += v2; }
};

template <>
struct aggstring<Bodo_FTypes::min> {
    static void apply(std::string& v1, std::string& v2) {
        v1 = std::min(v1, v2);
    }
};

template <>
struct aggstring<Bodo_FTypes::max> {
    static void apply(std::string& v1, std::string& v2) {
        v1 = std::max(v1, v2);
    }
};

template <>
struct aggstring<Bodo_FTypes::last> {
    static void apply(std::string& v1, std::string& v2) { v1 = v2; }
};

using pair_str_bool = std::pair<std::string, bool>;

template <int ftype, typename Enable = void>
struct aggliststring {
    /**
     * Apply the function.
     * @param[in,out] first input value, and holds the result
     * @param[in] second input value.
     */
    static void apply(std::vector<pair_str_bool>& v1,
                      std::vector<pair_str_bool>& v2) {}
};

template <>
struct aggliststring<Bodo_FTypes::sum> {
    static void apply(std::vector<pair_str_bool>& v1,
                      std::vector<pair_str_bool>& v2) {
        v1.insert(v1.end(), v2.begin(), v2.end());
    }
};

// returns -1 if v1 < v2, 0 if v1=v2 and 1 if v1 > v2
int compare_list_string(std::vector<pair_str_bool> const& v1,
                        std::vector<pair_str_bool> const& v2) {
    size_t len1 = v1.size();
    size_t len2 = v2.size();
    size_t minlen = len1;
    if (len2 < len1) minlen = len2;
    for (size_t i = 0; i < minlen; i++) {
        bool bit1 = v1[i].second;
        bool bit2 = v2[i].second;
        if (bit1 && !bit2) return 1;
        if (!bit1 && bit2) return -1;
        if (v1[i] < v2[i]) return -1;
        if (v1[i] > v2[i]) return 1;
    }
    if (len1 < len2) return -1;
    if (len1 > len2) return 1;
    return 0;
}

template <>
struct aggliststring<Bodo_FTypes::min> {
    static void apply(std::vector<pair_str_bool>& v1,
                      std::vector<pair_str_bool>& v2) {
        if (compare_list_string(v1, v2) == 1) v1 = v2;
    }
};

template <>
struct aggliststring<Bodo_FTypes::max> {
    static void apply(std::vector<pair_str_bool>& v1,
                      std::vector<pair_str_bool>& v2) {
        if (compare_list_string(v1, v2) == -1) v1 = v2;
    }
};

template <>
struct aggliststring<Bodo_FTypes::last> {
    static void apply(std::vector<pair_str_bool>& v1,
                      std::vector<pair_str_bool>& v2) {
        v1 = v2;
    }
};

// common template function

template <typename T, int dtype>
inline typename std::enable_if<!is_decimal<dtype>::value, double>::type
to_double(T const& val) {
    return (double)val;
}

template <typename T, int dtype>
inline typename std::enable_if<is_decimal<dtype>::value, double>::type
to_double(T const& val) {
    return decimal_to_double(val);
}

// count

template <typename T, int dtype, typename Enable = void>
struct count_agg {
    /**
     * Aggregation function for count. Increases count if value is not a nan
     *
     * @param[in,out] current count
     * @param second input value.
     */
    static void apply(int64_t& v1, T& v2);
};

template <typename T, int dtype>
struct count_agg<
    T, dtype,
    typename std::enable_if<!std::is_floating_point<T>::value &&
                            is_datetime_timedelta<dtype>::value>::type> {
    static void apply(int64_t& v1, T& v2) {
        if (v2 != std::numeric_limits<T>::min()) v1 += 1;
    }
};

template <typename T, int dtype>
struct count_agg<
    T, dtype,
    typename std::enable_if<!std::is_floating_point<T>::value &&
                            !is_datetime_timedelta<dtype>::value>::type> {
    static void apply(int64_t& v1, T& v2) { v1 += 1; }
};

template <typename T, int dtype>
struct count_agg<
    T, dtype, typename std::enable_if<std::is_floating_point<T>::value>::type> {
    static void apply(int64_t& v1, T& v2) {
        if (!isnan(v2)) v1 += 1;
    }
};

// mean

template <typename T, int dtype, typename Enable = void>
struct mean_agg {
    /**
     * Aggregation function for mean. Modifies count and sum of observed input
     * values
     *
     * @param[in,out] contains the current sum of observed values
     * @param an observed input value
     * @param[in,out] count: current number of observations
     */
    static void apply(double& v1, T& v2, uint64_t& count);
};

template <typename T, int dtype>
struct mean_agg<
    T, dtype,
    typename std::enable_if<!std::is_floating_point<T>::value &&
                            is_datetime_timedelta<dtype>::value>::type> {
    static void apply(double& v1, T& v2, uint64_t& count) {
        if (v2 != std::numeric_limits<T>::min()) {
            v1 += (double)v2;
            count += 1;
        }
    }
};

template <typename T, int dtype>
struct mean_agg<
    T, dtype,
    typename std::enable_if<!std::is_floating_point<T>::value &&
                            !is_datetime_timedelta<dtype>::value>::type> {
    static void apply(double& v1, T& v2, uint64_t& count) {
        v1 += to_double<T, dtype>(v2);
        count += 1;
    }
};

template <typename T, int dtype>
struct mean_agg<T, dtype,
                typename std::enable_if<std::is_floating_point<T>::value &&
                                        !is_decimal<dtype>::value>::type> {
    static void apply(double& v1, T& v2, uint64_t& count) {
        if (!isnan(v2)) {
            v1 += (double)v2;
            count += 1;
        }
    }
};

/**
 * Final evaluation step for mean, which calculates the mean based on the
 * sum of observed values and the number of values.
 *
 * @param[in,out] sum of observed values, will be modified to contain the mean
 * @param count: number of observations
 */
static void mean_eval(double& result, uint64_t& count) { result /= count; }

// variance

template <typename T, int dtype, typename Enable = void>
struct var_agg {
    /**
     * Aggregation function for variance. Modifies count, mean and m2 (sum of
     * squares of differences from the current mean) based on the observed input
     * values. See
     * https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
     * for more information.
     *
     * @param[in] observed value
     * @param[in,out] count: current number of observations
     * @param[in,out] mean_x: current mean
     * @param[in,out] m2: sum of squares of differences from the current mean
     */
    static void apply(T& v2, uint64_t& count, double& mean_x, double& m2);
};

template <typename T, int dtype>
struct var_agg<
    T, dtype,
    typename std::enable_if<!std::is_floating_point<T>::value &&
                            is_datetime_timedelta<dtype>::value>::type> {
    inline static void apply(T& v2, uint64_t& count, double& mean_x,
                             double& m2) {
        if (v2 != std::numeric_limits<T>::min()) {
            count += 1;
            double delta = (double)v2 - mean_x;
            mean_x += delta / count;
            double delta2 = (double)v2 - mean_x;
            m2 += delta * delta2;
        }
    }
};

template <typename T, int dtype>
struct var_agg<
    T, dtype,
    typename std::enable_if<!std::is_floating_point<T>::value &&
                            !is_datetime_timedelta<dtype>::value>::type> {
    inline static void apply(T& v2, uint64_t& count, double& mean_x,
                             double& m2) {
        double v2_double = to_double<T, dtype>(v2);
        count += 1;
        double delta = v2_double - mean_x;
        mean_x += delta / count;
        double delta2 = v2_double - mean_x;
        m2 += delta * delta2;
    }
};

template <typename T, int dtype>
struct var_agg<T, dtype,
               typename std::enable_if<std::is_floating_point<T>::value &&
                                       !is_decimal<dtype>::value>::type> {
    inline static void apply(T& v2, uint64_t& count, double& mean_x,
                             double& m2) {
        if (!isnan(v2)) {
            count += 1;
            double delta = (double)v2 - mean_x;
            mean_x += delta / count;
            double delta2 = (double)v2 - mean_x;
            m2 += delta * delta2;
        }
    }
};

/**
 * Perform combine operation for variance, which for a set of rows belonging
 * to the same group with count (# observations), mean and m2, reduces
 * the values to one row. See
 * https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
 * for more information.
 *
 * @param input array of counts (can include multiple values per group and
 * multiple groups)
 * @param input array of means (can include multiple values per group and
 * multiple groups)
 * @param input array of m2 (can include multiple values per group and multiple
 * groups)
 * @param output array of counts (one row per group)
 * @param output array of means (one row per group)
 * @param output array of m2 (one row per group)
 * @param maps row numbers in input arrays to row number in output array
 */
static void var_combine(array_info* count_col_in, array_info* mean_col_in,
                        array_info* m2_col_in, array_info* count_col_out,
                        array_info* mean_col_out, array_info* m2_col_out,
                        const std::vector<int64_t>& row_to_group) {
    for (int64_t i = 0; i < count_col_in->length; i++) {
        uint64_t& count_a = count_col_out->at<uint64_t>(row_to_group[i]);
        uint64_t& count_b = count_col_in->at<uint64_t>(i);
        double& mean_a = mean_col_out->at<double>(row_to_group[i]);
        double& mean_b = mean_col_in->at<double>(i);
        double& m2_a = m2_col_out->at<double>(row_to_group[i]);
        double& m2_b = m2_col_in->at<double>(i);

        uint64_t count = count_a + count_b;
        double delta = mean_b - mean_a;
        mean_a = (count_a * mean_a + count_b * mean_b) / count;
        m2_a = m2_a + m2_b + delta * delta * count_a * count_b / count;
        count_a = count;
    }
}

/**
 * Perform final evaluation step for variance, which calculates the variance
 * based on the count and m2 values. See
 * https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
 * for more information.
 *
 * @param[in,out] stores the calculated variance
 * @param count: number of observations
 * @param m2: sum of squares of differences from the current mean
 */
static void var_eval(double& result, uint64_t& count, double& m2) {
    if (count == 0)
        result = std::numeric_limits<double>::quiet_NaN();
    else
        result = m2 / (count - 1);
}

/**
 * Perform final evaluation step for std, which calculates the standard
 * deviation based on the count and m2 values. See
 * https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
 * for more information.
 *
 * @param[in,out] stores the calculated std
 * @param count: number of observations
 * @param m2: sum of squares of differences from the current mean
 */
static void std_eval(double& result, uint64_t& count, double& m2) {
    if (count == 0)
        result = std::numeric_limits<double>::quiet_NaN();
    else
        result = sqrt(m2 / (count - 1));
}

// last

template <typename T, int dtype>
struct aggfunc<
    T, dtype, Bodo_FTypes::last,
    typename std::enable_if<!std::is_floating_point<T>::value &&
                            is_datetime_timedelta<dtype>::value>::type> {
    /**
     * Aggregation function for last. Assigns value if not a nat
     *
     * @param[in,out] last value, and holds the result
     * @param second input value.
     */
    static void apply(T& v1, T& v2) {
        if (v2 != std::numeric_limits<T>::min()) v1 = v2;
    }
};

template <typename T, int dtype>
struct aggfunc<
    T, dtype, Bodo_FTypes::last,
    typename std::enable_if<!std::is_floating_point<T>::value &&
                            !is_datetime_timedelta<dtype>::value>::type> {
    /**
     * Aggregation function for last. Just assigns value
     *
     * @param[in,out] last value, and holds the result
     * @param second input value.
     */
    static void apply(T& v1, T& v2) { v1 = v2; }
};

template <typename T, int dtype>
struct aggfunc<
    T, dtype, Bodo_FTypes::last,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
    static void apply(T& v1, T& v2) {
        if (!isnan(v2)) v1 = v2;
    }
};

/** Data structure used for the computation of groups.

    @data row_to_group       : This takes the index and returns the group
    @data group_to_first_row : This takes the group index and return the first
   row index.
    @data next_row_in_group  : for a row in the list returns the next row in the
   list if existent. if non-existent value is -1.
    @data list_missing       : list of rows which are missing and NaNs.

    This is only one data structure but it has two use cases.
    -- get_group_info computes only the entries row_to_group and
   group_to_first_row. This is the data structure used for groupby operations
   such as sum, mean, etc. for which the full group structure does not need to
   be known.
    -- get_group_info_iterate computes all the entries. This is needed for some
   operations such as nunique, median, and cumulative operations. The entry
   list_missing is computed only for cumulative operations and computed only if
   needed.
 */
struct grouping_info {
    std::vector<int64_t> row_to_group;
    std::vector<int64_t> group_to_first_row;
    std::vector<int64_t> next_row_in_group;
    std::vector<int64_t> list_missing;
};

/**
 * Multi column key used for hashing keys to determine group membership in
 * groupby
 */
struct multi_col_key {
    uint32_t hash;
    table_info* table;
    int64_t row;

    multi_col_key(uint32_t _hash, table_info* _table, int64_t _row)
        : hash(_hash), table(_table), row(_row) {}

    bool operator==(const multi_col_key& other) const {
        for (int64_t i = 0; i < table->num_keys; i++) {
            array_info* c1 = table->columns[i];
            array_info* c2 = other.table->columns[i];
            size_t siztype;
            switch (c1->arr_type) {
                case bodo_array_type::ARROW: {
                    int64_t pos1_s = row;
                    int64_t pos1_e = row + 1;
                    int64_t pos2_s = other.row;
                    int64_t pos2_e = other.row + 1;
                    bool na_position_bis = true;
                    int test = ComparisonArrowColumn(c1->array, pos1_s, pos1_e,
                                                     c2->array, pos2_s, pos2_e,
                                                     na_position_bis);
                    if (test != 0) return false;
                }
                    continue;
                case bodo_array_type::NULLABLE_INT_BOOL:
                    if (GetBit((uint8_t*)c1->null_bitmask, row) !=
                        GetBit((uint8_t*)c2->null_bitmask, other.row))
                        return false;
                    if (!GetBit((uint8_t*)c1->null_bitmask, row)) continue;
                case bodo_array_type::CATEGORICAL:  // Even in missing case
                                                    // (value -1) this works
                case bodo_array_type::NUMPY:
                    siztype = numpy_item_size[c1->dtype];
                    if (memcmp(c1->data1 + siztype * row,
                               c2->data1 + siztype * other.row, siztype) != 0) {
                        return false;
                    }
                    continue;
                case bodo_array_type::STRING: {
                    uint8_t* c1_null_bitmask = (uint8_t*)c1->null_bitmask;
                    uint8_t* c2_null_bitmask = (uint8_t*)c2->null_bitmask;
                    if (GetBit(c1_null_bitmask, row) !=
                        GetBit(c2_null_bitmask, other.row))
                        return false;
                    uint32_t* c1_offsets = (uint32_t*)c1->data2;
                    uint32_t* c2_offsets = (uint32_t*)c2->data2;
                    uint32_t c1_str_len = c1_offsets[row + 1] - c1_offsets[row];
                    uint32_t c2_str_len =
                        c2_offsets[other.row + 1] - c2_offsets[other.row];
                    if (c1_str_len != c2_str_len) return false;
                    char* c1_str = c1->data1 + c1_offsets[row];
                    char* c2_str = c2->data1 + c2_offsets[other.row];
                    if (strncmp(c1_str, c2_str, c1_str_len) != 0) return false;
                }
                    continue;
                case bodo_array_type::LIST_STRING:
                    uint8_t* c1_null_bitmask = (uint8_t*)c1->null_bitmask;
                    uint8_t* c2_null_bitmask = (uint8_t*)c2->null_bitmask;
                    if (GetBit(c1_null_bitmask, row) !=
                        GetBit(c2_null_bitmask, other.row))
                        return false;
                    uint8_t* c1_sub_null_bitmask =
                        (uint8_t*)c1->sub_null_bitmask;
                    uint8_t* c2_sub_null_bitmask =
                        (uint8_t*)c2->sub_null_bitmask;
                    uint32_t* c1_index_offsets = (uint32_t*)c1->data3;
                    uint32_t* c2_index_offsets = (uint32_t*)c2->data3;
                    uint32_t* c1_data_offsets = (uint32_t*)c1->data2;
                    uint32_t* c2_data_offsets = (uint32_t*)c2->data2;
                    // Comparing the number of strings.
                    uint32_t c1_index_len =
                        c1_index_offsets[row + 1] - c1_index_offsets[row];
                    uint32_t c2_index_len = c2_index_offsets[other.row + 1] -
                                            c2_index_offsets[other.row];
                    if (c1_index_len != c2_index_len) return false;
                    // comparing the length of the strings.
                    for (uint32_t u = 0; u < c1_index_len; u++) {
                        uint32_t size_data1 =
                            c1_data_offsets[c1_index_offsets[row] + u + 1] -
                            c1_data_offsets[c1_index_offsets[row] + u];
                        uint32_t size_data2 =
                            c2_data_offsets[c2_index_offsets[other.row] + u +
                                            1] -
                            c2_data_offsets[c2_index_offsets[other.row] + u];
                        if (size_data1 != size_data2) return false;
                        bool str_bit1 = GetBit(c1_sub_null_bitmask,
                                               c1_index_offsets[row] + u);
                        bool str_bit2 = GetBit(c2_sub_null_bitmask,
                                               c2_index_offsets[other.row] + u);
                        if (str_bit1 != str_bit2) return false;
                    }
                    // Now comparing the strings. Their length is the same since
                    // we pass above check
                    uint32_t common_len =
                        c1_data_offsets[c1_index_offsets[row + 1]] -
                        c1_data_offsets[c1_index_offsets[row]];
                    char* c1_strB =
                        c1->data1 + c1_data_offsets[c1_index_offsets[row]];
                    char* c2_strB =
                        c2->data1 +
                        c2_data_offsets[c2_index_offsets[other.row]];
                    if (strncmp(c1_strB, c2_strB, common_len) != 0)
                        return false;
            }
        }
        return true;
    }
};

struct key_hash {
    std::size_t operator()(const multi_col_key& k) const { return k.hash; }
};

/**
 * Given a table with n key columns, this function calculates the row to group
 * mapping for every row based on its key.
 * For every row in the table, this only does *one* lookup in the hash map.
 *
 * @param the table
 * @param[out] vector that maps row number in the table to a group number
 * @param[out] vector that maps group number to the first row in the table
 *                that belongs to that group
 */
void get_group_info(table_info& table, std::vector<int64_t>& row_to_group,
                    std::vector<int64_t>& group_to_first_row,
                    bool check_for_null_keys) {
#ifdef DEBUG_GROUPBY
    std::cout << "Beginning of get_group_info\n";
#endif
    std::vector<array_info*> key_cols = std::vector<array_info*>(
        table.columns.begin(), table.columns.begin() + table.num_keys);
    uint32_t seed = SEED_HASH_GROUPBY_SHUFFLE;
    uint32_t* hashes = hash_keys(key_cols, seed);

    row_to_group.reserve(table.nrows());
    // start at 1 because I'm going to use 0 to mean nothing was inserted yet
    // in the map (but note that the group values I record in the output go from
    // 0 to num_groups - 1)
    int next_group = 1;
    UNORD_MAP_CONTAINER<multi_col_key, int64_t, key_hash> key_to_group;
    bool key_is_nullable = false;
    if (check_for_null_keys) {
        key_is_nullable = does_keys_have_nulls(key_cols);
    }
#ifdef DEBUG_GROUPBY
    std::cout << "check_for_null_keys=" << check_for_null_keys
              << " key_is_nullable=" << key_is_nullable << "\n";
#endif
    for (int64_t i = 0; i < table.nrows(); i++) {
        if (key_is_nullable) {
            if (does_row_has_nulls(key_cols, i)) {
                row_to_group.emplace_back(-1);
                continue;
            }
        }
        multi_col_key key(hashes[i], &table, i);
        int64_t& group = key_to_group[key];  // this inserts 0 into the map if
                                             // key doesn't exist
        if (group == 0) {
            group = next_group++;  // this updates the value in the map without
                                   // another lookup
            group_to_first_row.emplace_back(i);
        }
        row_to_group.emplace_back(group - 1);
    }
    delete[] hashes;
}

/**
 * Given a table with n key columns, this function calculates the row to group
 * mapping for every row based on its key.
 * For every row in the table, this only does *one* lookup in the hash map.
 *
 * @param            table: the table
 * @param consider_missing: whether to return the list of missing rows or not
 * @return vector that maps group number to the first row in the table
 *                that belongs to that group
 */
grouping_info get_group_info_iterate(table_info* table, bool consider_missing) {
#ifdef DEBUG_GROUPBY
    std::cout << "Beginning of get_group_info_iterate\n";
#endif
    std::vector<int64_t> row_to_group(table->nrows());
    std::vector<int64_t> group_to_first_row;
    std::vector<int64_t> next_row_in_group(table->nrows(), -1);
    std::vector<int64_t> active_group_repr;
    std::vector<int64_t> list_missing;

    std::vector<array_info*> key_cols = std::vector<array_info*>(
        table->columns.begin(), table->columns.begin() + table->num_keys);
    uint32_t seed = SEED_HASH_GROUPBY_SHUFFLE;
    uint32_t* hashes = hash_keys(key_cols, seed);

    bool key_is_nullable = does_keys_have_nulls(key_cols);
    // start at 1 because I'm going to use 0 to mean nothing was inserted yet
    // in the map (but note that the group values I record in the output go from
    // 0 to num_groups - 1)
    int next_group = 1;
    UNORD_MAP_CONTAINER<multi_col_key, int64_t, key_hash> key_to_group;
    for (int64_t i = 0; i < table->nrows(); i++) {
        if (key_is_nullable) {
            if (does_row_has_nulls(key_cols, i)) {
                row_to_group[i] = -1;
                if (consider_missing) list_missing.push_back(i);
                continue;
            }
        }
        multi_col_key key(hashes[i], table, i);
        int64_t& group = key_to_group[key];  // this inserts 0 into the map if
                                             // key doesn't exist
        if (group == 0) {
            group = next_group++;  // this updates the value in the map without
                                   // another lookup
            group_to_first_row.emplace_back(i);
            active_group_repr.emplace_back(i);
        } else {
            int64_t prev_elt = active_group_repr[group - 1];
            next_row_in_group[prev_elt] = i;
            active_group_repr[group - 1] = i;
        }
        row_to_group[i] = group - 1;
    }
    delete[] hashes;
    return {std::move(row_to_group), std::move(group_to_first_row),
            std::move(next_row_in_group), std::move(list_missing)};
}

/**
 * The cumulative_computation function. It uses the symbolic information
 * to compute the cumsum/cumprod/cummin/cummax
 *
 * @param The column on which we do the computation
 * @param The array containing information on how the rows are organized
 * @param skipna: Whether to skip NaN values or not.
 * @return the returning array.
 */
template <typename T, int dtype>
void cumulative_computation_T(array_info* arr, array_info* out_arr,
                              grouping_info const& grp_inf,
                              int32_t const& ftype, bool const& skipna) {
    size_t num_group = grp_inf.group_to_first_row.size();
    if (arr->arr_type == bodo_array_type::STRING ||
        arr->arr_type == bodo_array_type::LIST_STRING) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "There is no cumulative operation for the string "
                             "or list string case");
        return;
    }
    auto cum_computation =
        [&](std::function<std::pair<bool, T>(int64_t)> const& get_entry,
            std::function<void(int64_t, std::pair<bool, T> const&)> const&
                set_entry) -> void {
        for (size_t igrp = 0; igrp < num_group; igrp++) {
            int64_t i = grp_inf.group_to_first_row[igrp];
            T initVal;
            if (ftype == Bodo_FTypes::cumsum) initVal = 0;
            if (ftype == Bodo_FTypes::cummin)
                initVal = std::numeric_limits<T>::max();
            if (ftype == Bodo_FTypes::cummax)
                initVal = std::numeric_limits<T>::min();
            if (ftype == Bodo_FTypes::cumprod) initVal = 1;
            std::pair<bool, T> ePair{false, initVal};
            while (true) {
                std::pair<bool, T> fPair = get_entry(i);
                if (fPair.first) {  // the value is a NaN.
                    if (skipna) {
                        set_entry(i, fPair);
                    } else {
                        ePair = fPair;
                        set_entry(i, ePair);
                    }
                } else {  // The value is a normal one.
                    if (ftype == Bodo_FTypes::cumsum)
                        ePair.second += fPair.second;
                    if (ftype == Bodo_FTypes::cumprod)
                        ePair.second *= fPair.second;
                    if (ftype == Bodo_FTypes::cummin)
                        ePair.second = std::min(ePair.second, fPair.second);
                    if (ftype == Bodo_FTypes::cummax)
                        ePair.second = std::max(ePair.second, fPair.second);
                    set_entry(i, ePair);
                }
                i = grp_inf.next_row_in_group[i];
                if (i == -1) break;
            }
        }
        T eVal_nan = GetTentry<T>(
            RetrieveNaNentry((Bodo_CTypes::CTypeEnum)dtype).data());
        std::pair<bool, T> pairNaN{true, eVal_nan};
        for (auto& idx_miss : grp_inf.list_missing)
            set_entry(idx_miss, pairNaN);
    };

    if (arr->arr_type == bodo_array_type::NUMPY) {
        if (dtype == Bodo_CTypes::DATETIME || dtype == Bodo_CTypes::TIMEDELTA) {
            cum_computation(
                [=](int64_t pos) -> std::pair<bool, T> {
                    // in DATETIME/TIMEDELTA case, the types is necessarily
                    // int64_t
                    T eVal = arr->at<T>(pos);
                    bool isna = (eVal == std::numeric_limits<T>::min());
                    return {isna, eVal};
                },
                [=](int64_t pos, std::pair<bool, T> const& ePair) -> void {
                    if (ePair.first)
                        out_arr->at<T>(pos) = std::numeric_limits<T>::min();
                    else
                        out_arr->at<T>(pos) = ePair.second;
                });
        } else {
            cum_computation(
                [=](int64_t pos) -> std::pair<bool, T> {
                    T eVal = arr->at<T>(pos);
                    bool isna = isnan_alltype<T, dtype>(eVal);
                    return {isna, eVal};
                },
                [=](int64_t pos, std::pair<bool, T> const& ePair) -> void {
                    out_arr->at<T>(pos) = ePair.second;
                });
        }
    }
    if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        uint8_t* null_bitmask_i = (uint8_t*)arr->null_bitmask;
        uint8_t* null_bitmask_o = (uint8_t*)out_arr->null_bitmask;
        cum_computation(
            [=](int64_t pos) -> std::pair<bool, T> {
                return {!GetBit(null_bitmask_i, pos), arr->at<T>(pos)};
            },
            [=](int64_t pos, std::pair<bool, T> const& ePair) -> void {
                SetBitTo(null_bitmask_o, pos, !ePair.first);
                out_arr->at<T>(pos) = ePair.second;
            });
    }
}

void cumulative_computation_list_string(array_info* arr, array_info* out_arr,
                                        grouping_info const& grp_inf,
                                        int32_t const& ftype,
                                        bool const& skipna) {
#ifdef DEBUG_GROUPBY
    std::cout << "Beginning of cumulative_computation_list_string\n";
#endif
    if (ftype != Bodo_FTypes::cumsum) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "So far only cumulative sums for list-strings");
    }
    int64_t n = arr->length;
    using Tbas = std::pair<bool, std::string>;
    using T = std::pair<bool, std::vector<Tbas>>;
    std::vector<T> V(n);
    uint8_t* null_bitmask = (uint8_t*)arr->null_bitmask;
    uint8_t* sub_null_bitmask = (uint8_t*)arr->sub_null_bitmask;
    char* data = arr->data1;
    uint32_t* data_offsets = (uint32_t*)arr->data2;
    uint32_t* index_offsets = (uint32_t*)arr->data3;
    auto get_entry = [&](int64_t i) -> T {
        bool isna = !GetBit(null_bitmask, i);
        uint32_t start_idx_offset = index_offsets[i];
        uint32_t end_idx_offset = index_offsets[i + 1];
        std::vector<Tbas> LEnt;
        for (uint32_t idx = start_idx_offset; idx < end_idx_offset; idx++) {
            uint32_t str_len = data_offsets[idx + 1] - data_offsets[idx];
            uint32_t start_data_offset = data_offsets[idx];
            bool bit = GetBit(sub_null_bitmask, idx);
#ifdef DEBUG_GROUPBY
            std::cout << "get_entry i=" << i << " idx=" << idx << " bit=" << bit
                      << "\n";
#endif
            std::string val(&data[start_data_offset], str_len);
            Tbas eEnt = {bit, val};
            LEnt.push_back(eEnt);
        }
        return {isna, LEnt};
    };
    size_t num_group = grp_inf.group_to_first_row.size();
    for (size_t igrp = 0; igrp < num_group; igrp++) {
        int64_t i = grp_inf.group_to_first_row[igrp];
        T ePair{false, {}};
        while (true) {
            T fPair = get_entry(i);
            if (fPair.first) {  // the value is a NaN.
                if (skipna) {
                    V[i] = fPair;
                } else {
                    ePair = fPair;
                    V[i] = ePair;
                }
            } else {  // The value is a normal one.
                for (auto& eStr : fPair.second) ePair.second.push_back(eStr);
                V[i] = ePair;
            }
            i = grp_inf.next_row_in_group[i];
            if (i == -1) break;
        }
    }
    T pairNaN{true, {}};
    for (auto& idx_miss : grp_inf.list_missing) V[idx_miss] = pairNaN;
    // Now writing down in the array.
    int64_t nb_char = 0, nb_str = 0;
    for (auto& kPair : V) {
        nb_str += kPair.second.size();
        for (auto& estr : kPair.second) nb_char += estr.second.size();
    }
    // Doing the alocation trickery.
    array_info* new_out_col = alloc_list_string_array(n, nb_str, nb_char, 0);
    uint32_t* index_offsets_o = (uint32_t*)new_out_col->data3;
    uint32_t* data_offsets_o = (uint32_t*)new_out_col->data2;
    char* data_o = (char*)new_out_col->data1;
    uint8_t* null_bitmask_o = (uint8_t*)new_out_col->null_bitmask;
    uint8_t* sub_null_bitmask_o = (uint8_t*)new_out_col->sub_null_bitmask;
    // Writing the strings in output
    uint32_t pos_idx = 0;
    uint32_t pos_char = 0;
    for (int64_t i = 0; i < n; i++) {
        index_offsets_o[i] = pos_idx;
        T uPair = V[i];
        SetBitTo(null_bitmask_o, i, !uPair.first);
        size_t n_str = uPair.second.size();
        for (size_t i_str = 0; i_str < n_str; i_str++) {
            Tbas eBas = uPair.second[i_str];
            bool bit = eBas.first;
            std::string estr = eBas.second;
            size_t len = estr.size();
            memcpy(data_o, estr.data(), len);
            data_offsets_o[pos_idx + i_str] = pos_char;
            SetBitTo(sub_null_bitmask_o, pos_idx + i_str, bit);
            pos_char += len;
            data_o += len;
        }
        pos_idx += n_str;
    }
    index_offsets_o[n] = pos_idx;
    data_offsets_o[nb_str] = pos_char;
#ifdef DEBUG_GROUPBY
    std::cout << "End of cumulative_computation_list_string\n";
#endif
    *out_arr = std::move(*new_out_col);
#ifdef DEBUG_GROUPBY
    std::cout << "out_arr : ";
    DEBUG_PrintColumn(std::cout, out_arr);
#endif
    delete new_out_col;
}

void cumulative_computation_string(array_info* arr, array_info* out_arr,
                                   grouping_info const& grp_inf,
                                   int32_t const& ftype, bool const& skipna) {
#ifdef DEBUG_GROUPBY
    std::cout << "Beginning of cumulative_computation_string\n";
#endif
    if (ftype != Bodo_FTypes::cumsum) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "So far only cumulative sums for strings");
    }
    int64_t n = arr->length;
    using T = std::pair<bool, std::string>;
    std::vector<T> V(n);
    uint8_t* null_bitmask = (uint8_t*)arr->null_bitmask;
    char* data = arr->data1;
    uint32_t* offsets = (uint32_t*)arr->data2;
    auto get_entry = [&](int64_t i) -> T {
        bool isna = !GetBit(null_bitmask, i);
        uint32_t start_offset = offsets[i];
        uint32_t end_offset = offsets[i + 1];
        uint32_t len = end_offset - start_offset;
        std::string val(&data[start_offset], len);
        return {isna, val};
    };
    size_t num_group = grp_inf.group_to_first_row.size();
    for (size_t igrp = 0; igrp < num_group; igrp++) {
        int64_t i = grp_inf.group_to_first_row[igrp];
        T ePair{false, ""};
        while (true) {
            T fPair = get_entry(i);
            if (fPair.first) {  // the value is a NaN.
                if (skipna) {
                    V[i] = fPair;
                } else {
                    ePair = fPair;
                    V[i] = ePair;
                }
            } else {  // The value is a normal one.
                ePair.second += fPair.second;
                V[i] = ePair;
            }
            i = grp_inf.next_row_in_group[i];
            if (i == -1) break;
        }
    }
    T pairNaN{true, ""};
    for (auto& idx_miss : grp_inf.list_missing) V[idx_miss] = pairNaN;
    // Now writing down in the array.
    int64_t nb_char = 0;
    for (auto& kPair : V) nb_char += kPair.second.size();
    array_info* new_out_col = alloc_string_array(n, nb_char, 0);
    char* data_o = new_out_col->data1;
    uint32_t* offsets_o = (uint32_t*)new_out_col->data2;
    uint8_t* null_bitmask_o = (uint8_t*)new_out_col->null_bitmask;
    // Writing the strings in output
    uint32_t pos = 0;
    for (int64_t i = 0; i < n; i++) {
        offsets_o[i] = pos;
        T uPair = V[i];
        SetBitTo(null_bitmask_o, i, !uPair.first);
        size_t len = uPair.second.size();
        memcpy(data_o, uPair.second.data(), len);
        data_o += len;
        pos += len;
    }
    offsets_o[n] = pos;
#ifdef DEBUG_GROUPBY
    std::cout << "End of cumulative_computation_string\n";
#endif
    *out_arr = std::move(*new_out_col);
#ifdef DEBUG_GROUPBY
    std::cout << "out_arr : ";
    DEBUG_PrintColumn(std::cout, out_arr);
#endif
    delete new_out_col;
}

void cumulative_computation(array_info* arr, array_info* out_arr,
                            grouping_info const& grp_inf, int32_t const& ftype,
                            bool const& skipna) {
    Bodo_CTypes::CTypeEnum dtype = arr->dtype;
    if (arr->arr_type == bodo_array_type::STRING)
        return cumulative_computation_string(arr, out_arr, grp_inf, ftype,
                                             skipna);
    if (arr->arr_type == bodo_array_type::LIST_STRING)
        return cumulative_computation_list_string(arr, out_arr, grp_inf, ftype,
                                                  skipna);
    if (dtype == Bodo_CTypes::INT8)
        return cumulative_computation_T<int8_t, Bodo_CTypes::INT8>(
            arr, out_arr, grp_inf, ftype, skipna);
    if (dtype == Bodo_CTypes::UINT8)
        return cumulative_computation_T<uint8_t, Bodo_CTypes::UINT8>(
            arr, out_arr, grp_inf, ftype, skipna);

    if (dtype == Bodo_CTypes::INT16)
        return cumulative_computation_T<int16_t, Bodo_CTypes::INT16>(
            arr, out_arr, grp_inf, ftype, skipna);
    if (dtype == Bodo_CTypes::UINT16)
        return cumulative_computation_T<uint16_t, Bodo_CTypes::UINT16>(
            arr, out_arr, grp_inf, ftype, skipna);

    if (dtype == Bodo_CTypes::INT32)
        return cumulative_computation_T<int32_t, Bodo_CTypes::INT32>(
            arr, out_arr, grp_inf, ftype, skipna);
    if (dtype == Bodo_CTypes::UINT32)
        return cumulative_computation_T<uint32_t, Bodo_CTypes::UINT32>(
            arr, out_arr, grp_inf, ftype, skipna);

    if (dtype == Bodo_CTypes::INT64)
        return cumulative_computation_T<int64_t, Bodo_CTypes::INT64>(
            arr, out_arr, grp_inf, ftype, skipna);
    if (dtype == Bodo_CTypes::UINT64)
        return cumulative_computation_T<uint64_t, Bodo_CTypes::UINT64>(
            arr, out_arr, grp_inf, ftype, skipna);

    if (dtype == Bodo_CTypes::FLOAT32)
        return cumulative_computation_T<float, Bodo_CTypes::FLOAT32>(
            arr, out_arr, grp_inf, ftype, skipna);
    if (dtype == Bodo_CTypes::FLOAT64)
        return cumulative_computation_T<double, Bodo_CTypes::FLOAT64>(
            arr, out_arr, grp_inf, ftype, skipna);

    if (dtype == Bodo_CTypes::DATE)
        return cumulative_computation_T<int64_t, Bodo_CTypes::DATE>(
            arr, out_arr, grp_inf, ftype, skipna);
    if (dtype == Bodo_CTypes::DATETIME)
        return cumulative_computation_T<int64_t, Bodo_CTypes::DATETIME>(
            arr, out_arr, grp_inf, ftype, skipna);
    if (dtype == Bodo_CTypes::TIMEDELTA)
        return cumulative_computation_T<int64_t, Bodo_CTypes::TIMEDELTA>(
            arr, out_arr, grp_inf, ftype, skipna);
}

/**
 * The median_computation function. It uses the symbolic information to compute
 * the median results.
 *
 * @param The column on which we do the computation
 * @param The array containing information on how the rows are organized
 * @param skipna: Whether to skip NaN values or not.
 */
void median_computation(array_info* arr, array_info* out_arr,
                        grouping_info const& grp_inf, bool const& skipna) {
    size_t num_group = grp_inf.group_to_first_row.size();
    size_t siztype = numpy_item_size[arr->dtype];
    if (arr->arr_type == bodo_array_type::STRING ||
        arr->arr_type == bodo_array_type::LIST_STRING) {
        Bodo_PyErr_SetString(
            PyExc_RuntimeError,
            "There is no median for the string or list string case");
        return;
    }
    if (arr->dtype == Bodo_CTypes::DATE ||
        arr->dtype == Bodo_CTypes::DATETIME ||
        arr->dtype == Bodo_CTypes::TIMEDELTA) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "There is no median for the datetime case");
        return;
    }
    auto median_operation =
        [&](std::function<bool(size_t)> const& isnan_entry) -> void {
        for (size_t igrp = 0; igrp < num_group; igrp++) {
            int64_t i = grp_inf.group_to_first_row[igrp];
            std::vector<double> ListValue;
            bool HasNaN = false;
            while (true) {
                if (!isnan_entry(i)) {
                    char* ptr = arr->data1 + i * siztype;
                    double eVal = GetDoubleEntry(arr->dtype, ptr);
                    ListValue.emplace_back(eVal);
                } else {
                    if (!skipna) {
                        HasNaN = true;
                        break;
                    }
                }
                i = grp_inf.next_row_in_group[i];
                if (i == -1) break;
            }
            auto GetKthValue = [&](size_t const& pos) -> double {
                std::nth_element(ListValue.begin(), ListValue.begin() + pos,
                                 ListValue.end());
                return ListValue[pos];
            };
            double valReturn;
            if (HasNaN) {
                valReturn = std::nan("1");
            } else {
                size_t len = ListValue.size();
                int res = len % 2;
                if (res == 0) {
                    size_t kMid1 = len / 2;
                    size_t kMid2 = kMid1 - 1;
                    valReturn = (GetKthValue(kMid1) + GetKthValue(kMid2)) / 2;
                } else {
                    size_t kMid = len / 2;
                    valReturn = GetKthValue(kMid);
                }
            }
            out_arr->at<double>(igrp) = valReturn;
        }
    };
    if (arr->arr_type == bodo_array_type::NUMPY) {
        median_operation([=](size_t pos) -> bool {
            if (arr->dtype == Bodo_CTypes::FLOAT32) {
                return isnan(arr->at<float>(pos));
            }
            if (arr->dtype == Bodo_CTypes::FLOAT64) {
                return isnan(arr->at<double>(pos));
            }
            return false;
        });
    }
    if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        uint8_t* null_bitmask = (uint8_t*)arr->null_bitmask;
        median_operation(
            [=](size_t pos) -> bool { return !GetBit(null_bitmask, pos); });
    }
}

/**
 * The nunique_computation function. It uses the symbolic information to compute
 * the nunique results.
 *
 * @param The column on which we do the computation
 * @param The array containing information on how the rows are organized
 * @param The boolean dropna indicating whether we drop or not the NaN values
 * from the nunique computation.
 */
void nunique_computation(array_info* arr, array_info* out_arr,
                         grouping_info const& grp_inf, bool const& dropna) {
    size_t num_group = grp_inf.group_to_first_row.size();
    if (arr->arr_type == bodo_array_type::NUMPY) {
        /**
         * Check if a pointer points to a NaN or not
         *
         * @param the char* pointer
         * @param the type of the data in input
         */
        auto isnan_entry = [&](char* ptr) -> bool {
            if (arr->dtype == Bodo_CTypes::FLOAT32) {
                float* ptr_f = (float*)ptr;
                return isnan(*ptr_f);
            }
            if (arr->dtype == Bodo_CTypes::FLOAT64) {
                double* ptr_d = (double*)ptr;
                return isnan(*ptr_d);
            }
            if (arr->dtype == Bodo_CTypes::DATETIME ||
                arr->dtype == Bodo_CTypes::TIMEDELTA) {
                int64_t* ptr_i = (int64_t*)ptr;
                return *ptr_i == std::numeric_limits<int64_t>::min();
            }
            return false;
        };
        size_t siztype = numpy_item_size[arr->dtype];
        for (size_t igrp = 0; igrp < num_group; igrp++) {
            std::function<size_t(int64_t)> hash_fct = [&](int64_t i) -> size_t {
                char* ptr = arr->data1 + i * siztype;
                size_t retval = 0;
                memcpy(&retval, ptr, std::min(siztype, sizeof(size_t)));
                return retval;
            };
            std::function<bool(int64_t, int64_t)> equal_fct =
                [&](int64_t i1, int64_t i2) -> bool {
                char* ptr1 = arr->data1 + i1 * siztype;
                char* ptr2 = arr->data1 + i2 * siztype;
                return memcmp(ptr1, ptr2, siztype) == 0;
            };
            UNORD_SET_CONTAINER<int64_t, std::function<size_t(int64_t)>,
                                std::function<bool(int64_t, int64_t)>>
                eset({}, hash_fct, equal_fct);
            int64_t i = grp_inf.group_to_first_row[igrp];
            bool HasNullRow = false;
            while (true) {
                char* ptr = arr->data1 + (i * siztype);
                if (!isnan_entry(ptr)) {
                    eset.insert(i);
                } else {
                    HasNullRow = true;
                }
                i = grp_inf.next_row_in_group[i];
                if (i == -1) break;
            }
            int64_t size = eset.size();
            if (HasNullRow && !dropna) size++;
            out_arr->at<int64_t>(igrp) = size;
        }
    }
    if (arr->arr_type == bodo_array_type::LIST_STRING) {
        uint32_t* in_index_offsets = (uint32_t*)arr->data3;
        uint32_t* in_data_offsets = (uint32_t*)arr->data2;
        uint8_t* null_bitmask = (uint8_t*)arr->null_bitmask;
        uint8_t* sub_null_bitmask = (uint8_t*)arr->sub_null_bitmask;
        uint32_t seed = SEED_HASH_CONTAINER;
        for (size_t igrp = 0; igrp < num_group; igrp++) {
            std::function<size_t(int64_t)> hash_fct = [&](int64_t i) -> size_t {
                // We do not put the lengths and bitmask in the hash
                // computation. after all, it is just a hash
                char* val_chars =
                    arr->data1 + in_data_offsets[in_index_offsets[i]];
                int len = in_data_offsets[in_index_offsets[i + 1]] -
                          in_data_offsets[in_index_offsets[i]];
                uint32_t val;
                hash_string_32(val_chars, len, seed, &val);
                return size_t(val);
            };
            std::function<bool(int64_t, int64_t)> equal_fct =
                [&](int64_t i1, int64_t i2) -> bool {
                bool bit1 = GetBit(null_bitmask, i1);
                bool bit2 = GetBit(null_bitmask, i2);
                if (bit1 != bit2)
                    return false;  // That first case, might not be necessary.
                size_t len1 = in_index_offsets[i1 + 1] - in_index_offsets[i1];
                size_t len2 = in_index_offsets[i2 + 1] - in_index_offsets[i2];
                if (len1 != len2) return false;
                for (size_t u = 0; u < len1; u++) {
                    uint32_t len_str1 =
                        in_data_offsets[in_index_offsets[i1] + 1] -
                        in_data_offsets[in_index_offsets[i1]];
                    uint32_t len_str2 =
                        in_data_offsets[in_index_offsets[i2] + 1] -
                        in_data_offsets[in_index_offsets[i2]];
                    if (len_str1 != len_str2) return false;
                    bool bit1 = GetBit(sub_null_bitmask, in_index_offsets[i1]);
                    bool bit2 = GetBit(sub_null_bitmask, in_index_offsets[i2]);
                    if (bit1 != bit2) return false;
                }
                uint32_t nb_char1 = in_data_offsets[in_index_offsets[i1 + 1]] -
                                    in_data_offsets[in_index_offsets[i1]];
                uint32_t nb_char2 = in_data_offsets[in_index_offsets[i2 + 1]] -
                                    in_data_offsets[in_index_offsets[i2]];
                if (nb_char1 != nb_char2) return false;
                char* ptr1 =
                    arr->data1 +
                    sizeof(uint32_t) * in_data_offsets[in_index_offsets[i1]];
                char* ptr2 =
                    arr->data1 +
                    sizeof(uint32_t) * in_data_offsets[in_index_offsets[i2]];
                return strncmp(ptr1, ptr2, len1) == 0;
            };
            UNORD_SET_CONTAINER<int64_t, std::function<size_t(int64_t)>,
                                std::function<bool(int64_t, int64_t)>>
                eset({}, hash_fct, equal_fct);
            int64_t i = grp_inf.group_to_first_row[igrp];
            bool HasNullRow = false;
            while (true) {
                if (GetBit(null_bitmask, i)) {
                    eset.insert(i);
                } else {
                    HasNullRow = true;
                }
                i = grp_inf.next_row_in_group[i];
                if (i == -1) break;
            }
            int64_t size = eset.size();
            if (HasNullRow && !dropna) size++;
            out_arr->at<int64_t>(igrp) = size;
        }
    }
    if (arr->arr_type == bodo_array_type::STRING) {
        uint32_t* in_offsets = (uint32_t*)arr->data2;
        uint8_t* null_bitmask = (uint8_t*)arr->null_bitmask;
        uint32_t seed = SEED_HASH_CONTAINER;

        for (size_t igrp = 0; igrp < num_group; igrp++) {
            std::function<size_t(int64_t)> hash_fct = [&](int64_t i) -> size_t {
                char* val_chars = arr->data1 + in_offsets[i];
                int len = in_offsets[i + 1] - in_offsets[i];
                uint32_t val;
                hash_string_32(val_chars, len, seed, &val);
                return size_t(val);
            };
            std::function<bool(int64_t, int64_t)> equal_fct =
                [&](int64_t i1, int64_t i2) -> bool {
                size_t len1 = in_offsets[i1 + 1] - in_offsets[i1];
                size_t len2 = in_offsets[i2 + 1] - in_offsets[i2];
                if (len1 != len2) return false;
                char* ptr1 = arr->data1 + in_offsets[i1];
                char* ptr2 = arr->data1 + in_offsets[i2];
                return strncmp(ptr1, ptr2, len1) == 0;
            };
            UNORD_SET_CONTAINER<int64_t, std::function<size_t(int64_t)>,
                                std::function<bool(int64_t, int64_t)>>
                eset({}, hash_fct, equal_fct);
            int64_t i = grp_inf.group_to_first_row[igrp];
            bool HasNullRow = false;
            while (true) {
                if (GetBit(null_bitmask, i)) {
                    eset.insert(i);
                } else {
                    HasNullRow = true;
                }
                i = grp_inf.next_row_in_group[i];
                if (i == -1) break;
            }
            int64_t size = eset.size();
            if (HasNullRow && !dropna) size++;
            out_arr->at<int64_t>(igrp) = size;
        }
    }
    if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        uint8_t* null_bitmask = (uint8_t*)arr->null_bitmask;
        size_t siztype = numpy_item_size[arr->dtype];
        for (size_t igrp = 0; igrp < num_group; igrp++) {
            std::function<size_t(int64_t)> hash_fct = [&](int64_t i) -> size_t {
                char* ptr = arr->data1 + i * siztype;
                size_t retval = 0;
                size_t* size_t_ptrA = &retval;
                char* size_t_ptrB = (char*)size_t_ptrA;
                for (size_t i = 0; i < std::min(siztype, sizeof(size_t)); i++)
                    size_t_ptrB[i] = ptr[i];
                return retval;
            };
            std::function<bool(int64_t, int64_t)> equal_fct =
                [&](int64_t i1, int64_t i2) -> bool {
                char* ptr1 = arr->data1 + i1 * siztype;
                char* ptr2 = arr->data1 + i2 * siztype;
                return memcmp(ptr1, ptr2, siztype) == 0;
            };
            UNORD_SET_CONTAINER<int64_t, std::function<size_t(int64_t)>,
                                std::function<bool(int64_t, int64_t)>>
                eset({}, hash_fct, equal_fct);
            int64_t i = grp_inf.group_to_first_row[igrp];
            bool HasNullRow = false;
            while (true) {
                if (GetBit(null_bitmask, i)) {
                    eset.insert(i);
                } else {
                    HasNullRow = true;
                }
                i = grp_inf.next_row_in_group[i];
                if (i == -1) break;
            }
            int64_t size = eset.size();
            if (HasNullRow && !dropna) size++;
            out_arr->at<int64_t>(igrp) = size;
        }
    }
}

/**
 * Apply a function to a column(s), save result to (possibly reduced) output
 * column(s) Semantics of this function right now vary depending on function
 * type (ftype).
 *
 * @param column containing input values
 * @param output column
 * @param auxiliary input/output columns used for mean, var, std
 * @param maps row numbers in input columns to group numbers (for reduction
 * operations)
 */
template <typename T, int ftype, int dtype>
void apply_to_column(array_info* in_col, array_info* out_col,
                     std::vector<array_info*>& aux_cols,
                     const grouping_info& grp_info) {
    switch (in_col->arr_type) {
        case bodo_array_type::CATEGORICAL:
            if (ftype == Bodo_FTypes::count) {
                for (int64_t i = 0; i < in_col->length; i++) {
                    if (grp_info.row_to_group[i] != -1) {
                        T& val = in_col->at<T>(i);
                        if (!isnan_categorical<T, dtype>(val)) {
                            count_agg<T, dtype>::apply(
                                out_col->at<int64_t>(grp_info.row_to_group[i]),
                                in_col->at<T>(i));
                        }
                    }
                }
                return;
            }
        case bodo_array_type::NUMPY:
            if (ftype == Bodo_FTypes::mean) {
                array_info* count_col = aux_cols[0];
                for (int64_t i = 0; i < in_col->length; i++)
                    if (grp_info.row_to_group[i] != -1)
                        mean_agg<T, dtype>::apply(
                            out_col->at<double>(grp_info.row_to_group[i]),
                            in_col->at<T>(i),
                            count_col->at<uint64_t>(grp_info.row_to_group[i]));
            } else if (ftype == Bodo_FTypes::mean_eval) {
                for (int64_t i = 0; i < in_col->length; i++)
                    mean_eval(out_col->at<double>(i), in_col->at<uint64_t>(i));
            } else if (ftype == Bodo_FTypes::var) {
                array_info* count_col = aux_cols[0];
                array_info* mean_col = aux_cols[1];
                array_info* m2_col = aux_cols[2];
                for (int64_t i = 0; i < in_col->length; i++)
                    if (grp_info.row_to_group[i] != -1)
                        var_agg<T, dtype>::apply(
                            in_col->at<T>(i),
                            count_col->at<uint64_t>(grp_info.row_to_group[i]),
                            mean_col->at<double>(grp_info.row_to_group[i]),
                            m2_col->at<double>(grp_info.row_to_group[i]));
            } else if (ftype == Bodo_FTypes::var_eval) {
                array_info* count_col = aux_cols[0];
                array_info* m2_col = aux_cols[2];
                for (int64_t i = 0; i < in_col->length; i++)
                    var_eval(out_col->at<double>(i), count_col->at<uint64_t>(i),
                             m2_col->at<double>(i));
            } else if (ftype == Bodo_FTypes::std_eval) {
                array_info* count_col = aux_cols[0];
                array_info* m2_col = aux_cols[2];
                for (int64_t i = 0; i < in_col->length; i++)
                    std_eval(out_col->at<double>(i), count_col->at<uint64_t>(i),
                             m2_col->at<double>(i));
            } else if (ftype == Bodo_FTypes::count) {
                for (int64_t i = 0; i < in_col->length; i++)
                    if (grp_info.row_to_group[i] != -1)
                        count_agg<T, dtype>::apply(
                            out_col->at<int64_t>(grp_info.row_to_group[i]),
                            in_col->at<T>(i));
            } else if (ftype == Bodo_FTypes::first) {
                // create a temporary bitmask to know if we have set a
                // value for each row/group
                int64_t n_bytes = ((out_col->length + 7) >> 3);
                std::vector<char> out_col_bitmask_vec(n_bytes, 0);
                uint8_t* out_col_bitmask = (uint8_t*)out_col_bitmask_vec.data();
                for (int64_t i = 0; i < in_col->length; i++) {
                    const int64_t& group = grp_info.row_to_group[i];
                    T& val = in_col->at<T>(i);
                    if ((group != -1) && !GetBit(out_col_bitmask, group) &&
                        !isnan_alltype<T, dtype>(val)) {
                        out_col->at<T>(group) = val;
                        SetBitTo(out_col_bitmask, group, true);
                    }
                }
            } else if (ftype == Bodo_FTypes::idxmax) {
                array_info* index_pos = aux_cols[0];
                for (int64_t i = 0; i < in_col->length; i++) {
                    int64_t grp = grp_info.row_to_group[i];
                    if (grp != -1) {
                        idxmax_agg<T, dtype>::apply(
                            out_col->at<T>(grp), in_col->at<T>(i),
                            index_pos->at<uint64_t>(grp), i);
                    }
                }
            } else if (ftype == Bodo_FTypes::idxmin) {
                array_info* index_pos = aux_cols[0];
                for (int64_t i = 0; i < in_col->length; i++) {
                    int64_t grp = grp_info.row_to_group[i];
                    if (grp != -1) {
                        idxmin_agg<T, dtype>::apply(
                            out_col->at<T>(grp), in_col->at<T>(i),
                            index_pos->at<uint64_t>(grp), i);
                    }
                }
            } else {
                for (int64_t i = 0; i < in_col->length; i++)
                    if (grp_info.row_to_group[i] != -1)
                        aggfunc<T, dtype, ftype>::apply(
                            out_col->at<T>(grp_info.row_to_group[i]),
                            in_col->at<T>(i));
            }
            return;
        // for list strings, we are supporting count, sum, max, min, first, last
        case bodo_array_type::LIST_STRING:
            switch (ftype) {
                case Bodo_FTypes::count:
                    for (int64_t i = 0; i < in_col->length; i++) {
                        if ((grp_info.row_to_group[i] != -1) &&
                            GetBit((uint8_t*)in_col->null_bitmask, i))
                            count_agg<T, dtype>::apply(
                                out_col->at<int64_t>(grp_info.row_to_group[i]),
                                in_col->at<T>(i));
                    }
                    return;
                default:
                    size_t num_groups = grp_info.group_to_first_row.size();
                    std::vector<std::vector<pair_str_bool>> ListListPair(
                        num_groups);
                    char* data_i = in_col->data1;
                    uint32_t* index_offsets_i = (uint32_t*)in_col->data3;
                    uint32_t* data_offsets_i = (uint32_t*)in_col->data2;
                    uint8_t* null_bitmask_i = (uint8_t*)in_col->null_bitmask;
                    uint8_t* sub_null_bitmask_i =
                        (uint8_t*)in_col->sub_null_bitmask;
                    // Computing the strings used in output.
                    uint64_t n_bytes = (num_groups + 7) >> 3;
                    std::vector<uint8_t> Vmask(n_bytes, 0);
                    for (int64_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = grp_info.row_to_group[i];
                        if ((i_grp != -1) && GetBit(null_bitmask_i, i)) {
                            bool out_bit_set = GetBit(Vmask.data(), i_grp);
                            if (ftype == Bodo_FTypes::first && out_bit_set)
                                continue;
                            uint32_t start_offset = index_offsets_i[i];
                            uint32_t end_offset = index_offsets_i[i + 1];
                            uint32_t len = end_offset - start_offset;
                            std::vector<pair_str_bool> LStrB(len);
                            for (uint32_t i = 0; i < len; i++) {
                                uint32_t len_str =
                                    data_offsets_i[start_offset + i + 1] -
                                    data_offsets_i[start_offset + i];
                                uint32_t pos_start =
                                    data_offsets_i[start_offset + i];
                                std::string val(&data_i[pos_start], len_str);
                                bool str_bit = GetBit(sub_null_bitmask_i,
                                                      start_offset + i);
                                LStrB[i] = {val, str_bit};
                            }
                            if (out_bit_set) {
                                aggliststring<ftype>::apply(ListListPair[i_grp],
                                                            LStrB);
                            } else {
                                ListListPair[i_grp] = LStrB;
                                SetBitTo(Vmask.data(), i_grp, true);
                            }
                        }
                    }
                    // Determining the number of characters in output.
                    size_t nb_string = 0;
                    size_t nb_char = 0;
                    for (int64_t i_grp = 0; i_grp < int64_t(num_groups);
                         i_grp++) {
                        if (GetBit(Vmask.data(), i_grp)) {
                            nb_string += ListListPair[i_grp].size();
                            for (auto& e_str : ListListPair[i_grp])
                                nb_char += e_str.first.size();
                        }
                    }
                    // Allocation needs to be done through
                    // alloc_list_string_array, which allocates with meminfos
                    // and same data structs that Python uses. We need to
                    // re-allocate here because number of strings and chars has
                    // been determined here (previous out_col was just an empty
                    // dummy allocation).

                    array_info* new_out_col = alloc_list_string_array(
                        out_col->length, nb_string, nb_char, 0);
                    uint32_t* index_offsets_o = (uint32_t*)new_out_col->data3;
                    uint32_t* data_offsets_o = (uint32_t*)new_out_col->data2;
                    uint8_t* null_bitmask_o =
                        (uint8_t*)new_out_col->null_bitmask;
                    uint8_t* sub_null_bitmask_o =
                        (uint8_t*)new_out_col->sub_null_bitmask;
                    // Writing the list_strings in output
                    char* data_o = new_out_col->data1;
                    data_offsets_o[0] = 0;
                    uint32_t pos_index = 0;
                    uint32_t pos_data = 0;
                    for (int64_t i_grp = 0; i_grp < int64_t(num_groups);
                         i_grp++) {
                        bool bit = GetBit(Vmask.data(), i_grp);
                        SetBitTo(null_bitmask_o, i_grp, bit);
                        index_offsets_o[i_grp] = pos_index;
                        if (bit) {
                            uint32_t n_string = ListListPair[i_grp].size();
                            for (uint32_t i_str = 0; i_str < n_string;
                                 i_str++) {
                                std::string& estr =
                                    ListListPair[i_grp][i_str].first;
                                uint32_t n_char = estr.size();
                                memcpy(data_o, estr.data(), n_char);
                                data_o += n_char;
                                pos_data++;
                                data_offsets_o[pos_data] =
                                    data_offsets_o[pos_data - 1] + n_char;
                                bool bit = ListListPair[i_grp][i_str].second;
                                SetBitTo(sub_null_bitmask_o, pos_index + i_str,
                                         bit);
                            }
                            pos_index += n_string;
                            SetBitTo((uint8_t*)new_out_col->null_bitmask, i_grp,
                                     true);
                        } else {
                            SetBitTo((uint8_t*)new_out_col->null_bitmask, i_grp,
                                     false);
                        }
                    }
                    index_offsets_o[num_groups] = pos_index;
                    *out_col = std::move(*new_out_col);
                    delete new_out_col;
                    return;
            }

        // For the STRING we compute the count, sum, max, min, first, last
        case bodo_array_type::STRING:
            switch (ftype) {
                case Bodo_FTypes::count:
                    for (int64_t i = 0; i < in_col->length; i++) {
                        if ((grp_info.row_to_group[i] != -1) &&
                            GetBit((uint8_t*)in_col->null_bitmask, i))
                            count_agg<T, dtype>::apply(
                                out_col->at<int64_t>(grp_info.row_to_group[i]),
                                in_col->at<T>(i));
                    }
                    return;
                default:
                    size_t num_groups = grp_info.group_to_first_row.size();
                    std::vector<std::string> ListString(num_groups);
                    char* data = in_col->data1;
                    uint32_t* offsets = (uint32_t*)in_col->data2;
                    uint8_t* null_bitmask_i = (uint8_t*)in_col->null_bitmask;
                    uint8_t* null_bitmask_o = (uint8_t*)out_col->null_bitmask;
                    // Computing the strings used in output.
                    for (int64_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = grp_info.row_to_group[i];
                        if ((i_grp != -1) && GetBit(null_bitmask_i, i)) {
                            bool out_bit_set = GetBit(null_bitmask_o, i_grp);
                            if (ftype == Bodo_FTypes::first && out_bit_set)
                                continue;
                            uint32_t start_offset = offsets[i];
                            uint32_t end_offset = offsets[i + 1];
                            uint32_t len = end_offset - start_offset;
                            std::string val(&data[start_offset], len);
                            if (out_bit_set) {
                                aggstring<ftype>::apply(ListString[i_grp], val);
                            } else {
                                ListString[i_grp] = val;
                                SetBitTo(null_bitmask_o, i_grp, true);
                            }
                        }
                    }
                    // Determining the number of characters in output.
                    size_t nb_char = 0;
                    for (int64_t i_grp = 0; i_grp < int64_t(num_groups);
                         i_grp++) {
                        if (GetBit(null_bitmask_o, i_grp))
                            nb_char += ListString[i_grp].size();
                    }
                    // resize output string array to fit result size
                    array_item_arr_numpy_payload* payload =
                        (array_item_arr_numpy_payload*)out_col->meminfo->data;

                    // decref existing data array
                    decref_numpy_payload(payload->data);

                    // update string array payload to reflect change
                    payload->data = allocate_numpy_payload(
                        nb_char, Bodo_CTypes::CTypeEnum::UINT8);
                    out_col->data1 = payload->data.data;
                    out_col->n_sub_elems = nb_char;

                    // Writing the strings in output
                    char* data_o = out_col->data1;
                    uint32_t* offsets_o = (uint32_t*)out_col->data2;
                    uint32_t pos = 0;
                    for (int64_t i_grp = 0; i_grp < int64_t(num_groups);
                         i_grp++) {
                        offsets_o[i_grp] = pos;
                        if (GetBit(null_bitmask_o, i_grp)) {
                            int len = int(ListString[i_grp].size());
                            memcpy(data_o, ListString[i_grp].data(), len);
                            data_o += len;
                            pos += len;
                        }
                    }
                    offsets_o[num_groups] = pos;
                    return;
            }
        case bodo_array_type::NULLABLE_INT_BOOL:
            switch (ftype) {
                case Bodo_FTypes::count:
                    for (int64_t i = 0; i < in_col->length; i++) {
                        if ((grp_info.row_to_group[i] != -1) &&
                            GetBit((uint8_t*)in_col->null_bitmask, i))
                            count_agg<T, dtype>::apply(
                                out_col->at<int64_t>(grp_info.row_to_group[i]),
                                in_col->at<T>(i));
                    }
                    return;
                case Bodo_FTypes::mean:
                    for (int64_t i = 0; i < in_col->length; i++) {
                        if ((grp_info.row_to_group[i] != -1) &&
                            GetBit((uint8_t*)in_col->null_bitmask, i)) {
                            mean_agg<T, dtype>::apply(
                                out_col->at<double>(grp_info.row_to_group[i]),
                                in_col->at<T>(i),
                                aux_cols[0]->at<uint64_t>(
                                    grp_info.row_to_group[i]));
                        }
                    }
                    return;
                case Bodo_FTypes::var:
                    for (int64_t i = 0; i < in_col->length; i++) {
                        if ((grp_info.row_to_group[i] != -1) &&
                            GetBit((uint8_t*)in_col->null_bitmask, i))
                            var_agg<T, dtype>::apply(
                                in_col->at<T>(i),
                                aux_cols[0]->at<uint64_t>(
                                    grp_info.row_to_group[i]),
                                aux_cols[1]->at<double>(
                                    grp_info.row_to_group[i]),
                                aux_cols[2]->at<double>(
                                    grp_info.row_to_group[i]));
                    }
                    return;
                case Bodo_FTypes::first:
                    for (int64_t i = 0; i < in_col->length; i++) {
                        uint8_t* out_col_null_bitmask =
                            (uint8_t*)out_col->null_bitmask;
                        const int64_t& group = grp_info.row_to_group[i];
                        if ((group != -1) &&
                            !GetBit(out_col_null_bitmask, group) &&
                            GetBit((uint8_t*)in_col->null_bitmask, i)) {
                            out_col->at<T>(group) = in_col->at<T>(i);
                            SetBitTo(out_col_null_bitmask, group, true);
                        }
                    }
                    return;
                default:
                    for (int64_t i = 0; i < in_col->length; i++) {
                        if ((grp_info.row_to_group[i] != -1) &&
                            GetBit((uint8_t*)in_col->null_bitmask, i)) {
                            aggfunc<T, dtype, ftype>::apply(
                                out_col->at<T>(grp_info.row_to_group[i]),
                                in_col->at<T>(i));
                            SetBitTo((uint8_t*)out_col->null_bitmask,
                                     grp_info.row_to_group[i], true);
                        }
                    }
                    return;
            }
        default:
            Bodo_PyErr_SetString(PyExc_RuntimeError,
                                 "apply_to_column: incorrect array type");
            return;
    }
}

/**
 * Invokes the correct template instance of apply_to_column depending on
 * function (ftype) and dtype. See 'apply_to_column'
 *
 * @param column containing input values
 * @param output column
 * @param auxiliary input/output columns used for mean, var, std
 * @param maps row numbers in input columns to group numbers (for reduction
 * operations)
 * @param function to apply
 */
void do_apply_to_column(array_info* in_col, array_info* out_col,
                        std::vector<array_info*>& aux_cols,
                        const grouping_info& grp_info, int ftype) {
#ifdef DEBUG_GROUPBY
    std::cout << "Beginning of do_apply_to_coulm\n";
#endif
    if (in_col->arr_type == bodo_array_type::STRING ||
        in_col->arr_type == bodo_array_type::LIST_STRING) {
        switch (ftype) {
            // NOTE: The int template argument is not used in this call to
            // apply_to_column
            case Bodo_FTypes::sum:
                return apply_to_column<int, Bodo_FTypes::sum,
                                       Bodo_CTypes::STRING>(in_col, out_col,
                                                            aux_cols, grp_info);
            case Bodo_FTypes::min:
                return apply_to_column<int, Bodo_FTypes::min,
                                       Bodo_CTypes::STRING>(in_col, out_col,
                                                            aux_cols, grp_info);
            case Bodo_FTypes::max:
                return apply_to_column<int, Bodo_FTypes::max,
                                       Bodo_CTypes::STRING>(in_col, out_col,
                                                            aux_cols, grp_info);
            case Bodo_FTypes::first:
                return apply_to_column<int, Bodo_FTypes::first,
                                       Bodo_CTypes::STRING>(in_col, out_col,
                                                            aux_cols, grp_info);
            case Bodo_FTypes::last:
                return apply_to_column<int, Bodo_FTypes::last,
                                       Bodo_CTypes::STRING>(in_col, out_col,
                                                            aux_cols, grp_info);
        }
    }
    if (ftype == Bodo_FTypes::count) {
        switch (in_col->dtype) {
            case Bodo_CTypes::FLOAT32:
                // data will only be used to check for nans
                return apply_to_column<float, Bodo_FTypes::count,
                                       Bodo_CTypes::FLOAT32>(
                    in_col, out_col, aux_cols, grp_info);
            case Bodo_CTypes::FLOAT64:
                // data will only be used to check for nans
                return apply_to_column<double, Bodo_FTypes::count,
                                       Bodo_CTypes::FLOAT64>(
                    in_col, out_col, aux_cols, grp_info);
            default:
                // data will be ignored in this case, so type doesn't matter
                return apply_to_column<int8_t, Bodo_FTypes::count,
                                       Bodo_CTypes::INT8>(in_col, out_col,
                                                          aux_cols, grp_info);
        }
    }

    switch (in_col->dtype) {
        case Bodo_CTypes::_BOOL:
            switch (ftype) {
                case Bodo_FTypes::min:
                    return apply_to_column<bool, Bodo_FTypes::min,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<bool, Bodo_FTypes::max,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<bool, Bodo_FTypes::prod,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::first:
                    return apply_to_column<bool, Bodo_FTypes::first,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::last:
                    return apply_to_column<bool, Bodo_FTypes::last,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<bool, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<bool, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info);
                default:
                    Bodo_PyErr_SetString(
                        PyExc_RuntimeError,
                        "unsuported aggregation for boolean type column");
                    return;
            }
        case Bodo_CTypes::INT8:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<int8_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<int8_t, Bodo_FTypes::min,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<int8_t, Bodo_FTypes::max,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<int8_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::first:
                    return apply_to_column<int8_t, Bodo_FTypes::first,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::last:
                    return apply_to_column<int8_t, Bodo_FTypes::last,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<int8_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int8_t, Bodo_FTypes::var,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<int8_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<int8_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::UINT8:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<uint8_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<uint8_t, Bodo_FTypes::min,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<uint8_t, Bodo_FTypes::max,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<uint8_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::first:
                    return apply_to_column<uint8_t, Bodo_FTypes::first,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::last:
                    return apply_to_column<uint8_t, Bodo_FTypes::last,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<uint8_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<uint8_t, Bodo_FTypes::var,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<uint8_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<uint8_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::INT16:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<int16_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<int16_t, Bodo_FTypes::min,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<int16_t, Bodo_FTypes::max,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<int16_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::first:
                    return apply_to_column<int16_t, Bodo_FTypes::first,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::last:
                    return apply_to_column<int16_t, Bodo_FTypes::last,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<int16_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int16_t, Bodo_FTypes::var,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<int16_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<int16_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::UINT16:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<uint16_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<uint16_t, Bodo_FTypes::min,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<uint16_t, Bodo_FTypes::max,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<uint16_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::first:
                    return apply_to_column<uint16_t, Bodo_FTypes::first,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::last:
                    return apply_to_column<uint16_t, Bodo_FTypes::last,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<uint16_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<uint16_t, Bodo_FTypes::var,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<uint16_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<uint16_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::INT32:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<int32_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<int32_t, Bodo_FTypes::min,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<int32_t, Bodo_FTypes::max,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<int32_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::first:
                    return apply_to_column<int32_t, Bodo_FTypes::first,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::last:
                    return apply_to_column<int32_t, Bodo_FTypes::last,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<int32_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int32_t, Bodo_FTypes::var,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<int32_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<int32_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::UINT32:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<uint32_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<uint32_t, Bodo_FTypes::min,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<uint32_t, Bodo_FTypes::max,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<uint32_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::first:
                    return apply_to_column<uint32_t, Bodo_FTypes::first,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::last:
                    return apply_to_column<uint32_t, Bodo_FTypes::last,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<uint32_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<uint32_t, Bodo_FTypes::var,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<uint32_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<uint32_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::INT64:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<int64_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<int64_t, Bodo_FTypes::min,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<int64_t, Bodo_FTypes::max,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<int64_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::first:
                    return apply_to_column<int64_t, Bodo_FTypes::first,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::last:
                    return apply_to_column<int64_t, Bodo_FTypes::last,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<int64_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int64_t, Bodo_FTypes::var,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<int64_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<int64_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::UINT64:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<uint64_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<uint64_t, Bodo_FTypes::min,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<uint64_t, Bodo_FTypes::max,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<uint64_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::first:
                    return apply_to_column<uint64_t, Bodo_FTypes::first,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::last:
                    return apply_to_column<uint64_t, Bodo_FTypes::last,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<uint64_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<uint64_t, Bodo_FTypes::var,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<uint64_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<uint64_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::DATE:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<int64_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<int64_t, Bodo_FTypes::min,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<int64_t, Bodo_FTypes::max,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<int64_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::first:
                    return apply_to_column<int64_t, Bodo_FTypes::first,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::last:
                    return apply_to_column<int64_t, Bodo_FTypes::last,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<int64_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int64_t, Bodo_FTypes::var,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<int64_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<int64_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::DATETIME:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<int64_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<int64_t, Bodo_FTypes::min,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<int64_t, Bodo_FTypes::max,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<int64_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::first:
                    return apply_to_column<int64_t, Bodo_FTypes::first,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::last:
                    return apply_to_column<int64_t, Bodo_FTypes::last,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<int64_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int64_t, Bodo_FTypes::var,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<int64_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<int64_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::TIMEDELTA:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<int64_t, Bodo_FTypes::sum,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<int64_t, Bodo_FTypes::min,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<int64_t, Bodo_FTypes::max,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<int64_t, Bodo_FTypes::prod,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::first:
                    return apply_to_column<int64_t, Bodo_FTypes::first,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::last:
                    return apply_to_column<int64_t, Bodo_FTypes::last,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<int64_t, Bodo_FTypes::mean,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int64_t, Bodo_FTypes::var,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<int64_t, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<int64_t, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::FLOAT32:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<float, Bodo_FTypes::sum,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<float, Bodo_FTypes::min,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<float, Bodo_FTypes::max,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<float, Bodo_FTypes::prod,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::first:
                    return apply_to_column<float, Bodo_FTypes::first,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::last:
                    return apply_to_column<float, Bodo_FTypes::last,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<float, Bodo_FTypes::mean,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean_eval:
                    return apply_to_column<float, Bodo_FTypes::mean_eval,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<float, Bodo_FTypes::var,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var_eval:
                    return apply_to_column<float, Bodo_FTypes::var_eval,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::std_eval:
                    return apply_to_column<float, Bodo_FTypes::std_eval,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<float, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<float, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::FLOAT64:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<double, Bodo_FTypes::sum,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<double, Bodo_FTypes::min,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<double, Bodo_FTypes::max,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<double, Bodo_FTypes::prod,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::first:
                    return apply_to_column<double, Bodo_FTypes::first,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::last:
                    return apply_to_column<double, Bodo_FTypes::last,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<double, Bodo_FTypes::mean,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean_eval:
                    return apply_to_column<double, Bodo_FTypes::mean_eval,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<double, Bodo_FTypes::var,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var_eval:
                    return apply_to_column<double, Bodo_FTypes::var_eval,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::std_eval:
                    return apply_to_column<double, Bodo_FTypes::std_eval,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<double, Bodo_FTypes::idxmin,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<double, Bodo_FTypes::idxmax,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::DECIMAL:
            switch (ftype) {
                case Bodo_FTypes::first:
                    return apply_to_column<decimal_value_cpp,
                                           Bodo_FTypes::first,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::last:
                    return apply_to_column<decimal_value_cpp, Bodo_FTypes::last,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<decimal_value_cpp, Bodo_FTypes::min,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<decimal_value_cpp, Bodo_FTypes::max,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<decimal_value_cpp, Bodo_FTypes::mean,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean_eval:
                    return apply_to_column<decimal_value_cpp,
                                           Bodo_FTypes::mean_eval,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<decimal_value_cpp, Bodo_FTypes::var,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var_eval:
                    return apply_to_column<decimal_value_cpp,
                                           Bodo_FTypes::var_eval,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::std_eval:
                    return apply_to_column<decimal_value_cpp,
                                           Bodo_FTypes::std_eval,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<decimal_value_cpp,
                                           Bodo_FTypes::idxmin,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<decimal_value_cpp,
                                           Bodo_FTypes::idxmax,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info);
            }
        default:
            std::cerr << "do_apply_to_column: invalid array dtype" << std::endl;
            return;
    }
}

/**
 * Initialize an output column that will be used to store the result of an
 * aggregation function. Initialization depends on the function:
 * default: zero initialization
 * prod: 1
 * min: max dtype value, or quiet_NaN if float (so that result is nan if all
 * input values are nan) max: min dtype value, or quiet_NaN if float (so that
 * result is nan if all input values are nan)
 *
 * @param output column
 * @param function identifier
 */
void aggfunc_output_initialize(array_info* out_col, int ftype) {
    if (out_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        if (ftype == Bodo_FTypes::min || ftype == Bodo_FTypes::max ||
            ftype == Bodo_FTypes::first || ftype == Bodo_FTypes::last)
            // if input is all nulls, max, min, first and last output will be
            // null
            InitializeBitMask((uint8_t*)out_col->null_bitmask, out_col->length,
                              false);
        else
            // for other functions (count, sum, etc.) output will never be null
            InitializeBitMask((uint8_t*)out_col->null_bitmask, out_col->length,
                              true);
    }
    if (out_col->arr_type == bodo_array_type::STRING ||
        out_col->arr_type == bodo_array_type::LIST_STRING) {
        InitializeBitMask((uint8_t*)out_col->null_bitmask, out_col->length,
                          false);
    }
    switch (ftype) {
        case Bodo_FTypes::prod:
            switch (out_col->dtype) {
                case Bodo_CTypes::_BOOL:
                    std::fill((bool*)out_col->data1,
                              (bool*)out_col->data1 + out_col->length, true);
                    return;
                case Bodo_CTypes::INT8:
                    std::fill((int8_t*)out_col->data1,
                              (int8_t*)out_col->data1 + out_col->length, 1);
                    return;
                case Bodo_CTypes::UINT8:
                    std::fill((uint8_t*)out_col->data1,
                              (uint8_t*)out_col->data1 + out_col->length, 1);
                    return;
                case Bodo_CTypes::INT16:
                    std::fill((int16_t*)out_col->data1,
                              (int16_t*)out_col->data1 + out_col->length, 1);
                    return;
                case Bodo_CTypes::UINT16:
                    std::fill((uint16_t*)out_col->data1,
                              (uint16_t*)out_col->data1 + out_col->length, 1);
                    return;
                case Bodo_CTypes::INT32:
                    std::fill((int32_t*)out_col->data1,
                              (int32_t*)out_col->data1 + out_col->length, 1);
                    return;
                case Bodo_CTypes::UINT32:
                    std::fill((uint32_t*)out_col->data1,
                              (uint32_t*)out_col->data1 + out_col->length, 1);
                    return;
                case Bodo_CTypes::INT64:
                    std::fill((int64_t*)out_col->data1,
                              (int64_t*)out_col->data1 + out_col->length, 1);
                    return;
                case Bodo_CTypes::UINT64:
                    std::fill((uint64_t*)out_col->data1,
                              (uint64_t*)out_col->data1 + out_col->length, 1);
                    return;
                case Bodo_CTypes::FLOAT32:
                    std::fill((float*)out_col->data1,
                              (float*)out_col->data1 + out_col->length, 1);
                    return;
                case Bodo_CTypes::FLOAT64:
                    std::fill((double*)out_col->data1,
                              (double*)out_col->data1 + out_col->length, 1);
                    return;
                case Bodo_CTypes::STRING:
                default:
                    Bodo_PyErr_SetString(PyExc_RuntimeError,
                                         "unsupported/not implemented");
                    return;
            }
        case Bodo_FTypes::min:
            switch (out_col->dtype) {
                case Bodo_CTypes::_BOOL:
                    std::fill((bool*)out_col->data1,
                              (bool*)out_col->data1 + out_col->length, true);
                    return;
                case Bodo_CTypes::INT8:
                    std::fill((int8_t*)out_col->data1,
                              (int8_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<int8_t>::max());
                    return;
                case Bodo_CTypes::UINT8:
                    std::fill((uint8_t*)out_col->data1,
                              (uint8_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<uint8_t>::max());
                    return;
                case Bodo_CTypes::INT16:
                    std::fill((int16_t*)out_col->data1,
                              (int16_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<int16_t>::max());
                    return;
                case Bodo_CTypes::UINT16:
                    std::fill((uint16_t*)out_col->data1,
                              (uint16_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<uint16_t>::max());
                    return;
                case Bodo_CTypes::INT32:
                    std::fill((int32_t*)out_col->data1,
                              (int32_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<int32_t>::max());
                    return;
                case Bodo_CTypes::UINT32:
                    std::fill((uint32_t*)out_col->data1,
                              (uint32_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<uint32_t>::max());
                    return;
                case Bodo_CTypes::INT64:
                    std::fill((int64_t*)out_col->data1,
                              (int64_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<int64_t>::max());
                    return;
                case Bodo_CTypes::UINT64:
                    std::fill((uint64_t*)out_col->data1,
                              (uint64_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<uint64_t>::max());
                    return;
                case Bodo_CTypes::DATE:
                case Bodo_CTypes::DATETIME:
                case Bodo_CTypes::TIMEDELTA:
                    std::fill((int64_t*)out_col->data1,
                              (int64_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<int64_t>::max());
                    return;
                case Bodo_CTypes::FLOAT32:
                    // initialize to quiet_NaN so that result is nan if all
                    // input values are nan
                    std::fill((float*)out_col->data1,
                              (float*)out_col->data1 + out_col->length,
                              std::numeric_limits<float>::quiet_NaN());
                    return;
                case Bodo_CTypes::FLOAT64:
                    // initialize to quiet_NaN so that result is nan if all
                    // input values are nan
                    std::fill((double*)out_col->data1,
                              (double*)out_col->data1 + out_col->length,
                              std::numeric_limits<double>::quiet_NaN());
                    return;
                case Bodo_CTypes::DECIMAL:
                    std::fill((int64_t*)out_col->data1,
                              (int64_t*)out_col->data1 + 2 * out_col->length,
                              std::numeric_limits<int64_t>::max());
                    return;
                case Bodo_CTypes::STRING:
                    // Nothing to initilize with in the case of strings.
                    return;
                case Bodo_CTypes::LIST_STRING:
                    // Nothing to initilize with in the case of list strings.
                    return;
                default:
                    Bodo_PyErr_SetString(PyExc_RuntimeError,
                                         "unsupported/not implemented");
                    return;
            }
        case Bodo_FTypes::max:
            switch (out_col->dtype) {
                case Bodo_CTypes::_BOOL:
                    std::fill((bool*)out_col->data1,
                              (bool*)out_col->data1 + out_col->length, false);
                    return;
                case Bodo_CTypes::INT8:
                    std::fill((int8_t*)out_col->data1,
                              (int8_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<int8_t>::min());
                    return;
                case Bodo_CTypes::UINT8:
                    std::fill((uint8_t*)out_col->data1,
                              (uint8_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<uint8_t>::min());
                    return;
                case Bodo_CTypes::INT16:
                    std::fill((int16_t*)out_col->data1,
                              (int16_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<int16_t>::min());
                    return;
                case Bodo_CTypes::UINT16:
                    std::fill((uint16_t*)out_col->data1,
                              (uint16_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<uint16_t>::min());
                    return;
                case Bodo_CTypes::INT32:
                    std::fill((int32_t*)out_col->data1,
                              (int32_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<int32_t>::min());
                    return;
                case Bodo_CTypes::UINT32:
                    std::fill((uint32_t*)out_col->data1,
                              (uint32_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<uint32_t>::min());
                    return;
                case Bodo_CTypes::INT64:
                    std::fill((int64_t*)out_col->data1,
                              (int64_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<int64_t>::min());
                    return;
                case Bodo_CTypes::UINT64:
                    std::fill((uint64_t*)out_col->data1,
                              (uint64_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<uint64_t>::min());
                    return;
                case Bodo_CTypes::DATE:
                case Bodo_CTypes::DATETIME:
                case Bodo_CTypes::TIMEDELTA:
                    std::fill((int64_t*)out_col->data1,
                              (int64_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<int64_t>::min());
                    return;
                case Bodo_CTypes::FLOAT32:
                    // initialize to quiet_NaN so that result is nan if all
                    // input values are nan
                    std::fill((float*)out_col->data1,
                              (float*)out_col->data1 + out_col->length,
                              std::numeric_limits<float>::quiet_NaN());
                    return;
                case Bodo_CTypes::FLOAT64:
                    // initialize to quiet_NaN so that result is nan if all
                    // input values are nan
                    std::fill((double*)out_col->data1,
                              (double*)out_col->data1 + out_col->length,
                              std::numeric_limits<double>::quiet_NaN());
                    return;
                case Bodo_CTypes::DECIMAL:
                    std::fill((int64_t*)out_col->data1,
                              (int64_t*)out_col->data1 + 2 * out_col->length,
                              std::numeric_limits<int64_t>::min());
                    return;
                case Bodo_CTypes::STRING:
                    // nothing to initialize in the case of strings
                    return;
                case Bodo_CTypes::LIST_STRING:
                    // nothing to initialize in the case of list strings
                    return;
                default:
                    Bodo_PyErr_SetString(PyExc_RuntimeError,
                                         "unsupported/not implemented");
                    return;
            }
        case Bodo_FTypes::first:
        case Bodo_FTypes::last:
            switch (out_col->dtype) {
                // for first & last, we only need an initial value for the
                // non-null bitmask cases where the datatype has a nan
                // representation
                case Bodo_CTypes::DATE:
                case Bodo_CTypes::DATETIME:
                case Bodo_CTypes::TIMEDELTA:
                    // nat representation for date values is int64_t min value
                    std::fill((int64_t*)out_col->data1,
                              (int64_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<int64_t>::min());
                    return;
                case Bodo_CTypes::FLOAT32:
                    // initialize to quiet_NaN so that result is nan if all
                    // input values are nan
                    std::fill((float*)out_col->data1,
                              (float*)out_col->data1 + out_col->length,
                              std::numeric_limits<float>::quiet_NaN());
                    return;
                case Bodo_CTypes::FLOAT64:
                    // initialize to quiet_NaN so that result is nan if all
                    // input values are nan
                    std::fill((double*)out_col->data1,
                              (double*)out_col->data1 + out_col->length,
                              std::numeric_limits<double>::quiet_NaN());
                    return;
                default:
                    // for most cases we don't need an initial value, first/last
                    // will just replace that with the first/last value
                    return;
            }
        default:
            // zero initialize
            memset(out_col->data1, 0,
                   numpy_item_size[out_col->dtype] * out_col->length);
    }
}

/**
 * Returns the array type and dtype required for output columns based on the
 * aggregation function and input dtype.
 *
 * @param function identifier
 * @param[in,out] array type (caller sets a default, this function only changes
 * in certain cases)
 * @param[in,out] output dtype (caller sets a default, this function only
 * changes in certain cases)
 * @param true if column is key column (in this case ignore because output type
 * will be the same)
 */
static void get_groupby_output_dtype(int ftype,
                                     bodo_array_type::arr_type_enum& array_type,
                                     Bodo_CTypes::CTypeEnum& dtype,
                                     bool is_key) {
    if (is_key) return;
    switch (ftype) {
        case Bodo_FTypes::nunique:
        case Bodo_FTypes::count:
            array_type = bodo_array_type::NUMPY;
            dtype = Bodo_CTypes::INT64;
            return;
        case Bodo_FTypes::median:
        case Bodo_FTypes::mean:
        case Bodo_FTypes::var:
        case Bodo_FTypes::std:
            array_type = bodo_array_type::NUMPY;
            dtype = Bodo_CTypes::FLOAT64;
            return;
        default:
            return;
    }
}

/*
 An instance of GroupbyPipeline class manages a groupby operation. In a
 groupby operation, an arbitrary number of functions can be applied to each
 input column. The functions can vary between input columns. Each combination
 of (input column, function) is an operation that produces a column in the
 output table. The computation of each (input column, function) pair is
 encapsulated in what is called a "column set" (for lack of a better name).
 There are different column sets for different types of operations (e.g. var,
 mean, median, udfs, basic operations...). Each column set creates,
 initializes, operates on and manages the arrays needed to perform its
 computation. Different column set types may require different number of
 columns and dtypes. The main control flow of groupby is in
 GroupbyPipeline::run(). It invokes update, shuffle, combine and eval steps
 (as needed), and these steps iterate through the column sets and invoke
 their operations.
*/

/*
 * This is the base column set class which is used by most operations (like
 * sum, prod, count, etc.). Several subclasses also rely on some of the methods
 * of this base class.
 */
class BasicColSet {
   public:
    /**
     * Construct column set corresponding to function of type ftype applied to
     * the input column in_col
     * @param input column of groupby associated with this column set
     * @param ftype associated with this column set
     * @param tells the column set whether GroupbyPipeline is going to perform
     *        a combine operation or not. If false, this means that either
     *        shuffling is not necessary or that it will be done at the
     *        beginning of the pipeline.
     */
    BasicColSet(array_info* in_col, int ftype, bool combine_step)
        : in_col(in_col), ftype(ftype), combine_step(combine_step) {}
    virtual ~BasicColSet() {}

    /**
     * Allocate my columns for update step.
     * @param number of groups found in the input table
     * @param[in,out] vector of columns of update table. This method adds
     *                columns to this vector.
     */
    virtual void alloc_update_columns(size_t num_groups,
                                      std::vector<array_info*>& out_cols) {
        bodo_array_type::arr_type_enum arr_type = in_col->arr_type;
        Bodo_CTypes::CTypeEnum dtype = in_col->dtype;
        int64_t num_categories = in_col->num_categories;
        // calling this modifies arr_type and dtype
        get_groupby_output_dtype(ftype, arr_type, dtype, false);
#ifdef DEBUG_GROUPBY
        std::cout << "num_groups=" << num_groups
                  << " num_categories=" << num_categories << "\n";
        std::cout << "arr_type=" << int(arr_type) << " dtype=" << int(dtype)
                  << "\n";
#endif
        out_cols.push_back(
            alloc_array(num_groups, 1, 1, arr_type, dtype, 0, num_categories));
        update_cols.push_back(out_cols.back());
    }

    /**
     * Perform update step for this column set. This will fill my columns with
     * the result of the aggregation operation corresponding to this column set
     * @param grouping info calculated by GroupbyPipeline
     */
    virtual void update(const grouping_info& grp_info) {
        std::vector<array_info*> aux_cols;
        aggfunc_output_initialize(update_cols[0], ftype);
        do_apply_to_column(in_col, update_cols[0], aux_cols, grp_info, ftype);
    }

    /**
     * When GroupbyPipeline shuffles the table after update, the column set
     * needs to be updated with the columns from the new shuffled table. This
     * method is called by GroupbyPipeline with an iterator pointing to my
     * first column. The column set will update its columns and return an
     * iterator pointing to the next set of columns.
     * @param iterator pointing to the first column in this column set
     */
    virtual std::vector<array_info*>::iterator update_after_shuffle(
        std::vector<array_info*>::iterator& it) {
        update_cols.assign(it, it + update_cols.size());
        return it + update_cols.size();
    }

    /**
     * Allocate my columns for combine step.
     * @param number of groups found in the input table (which is the update
     * table)
     * @param[in,out] vector of columns of combine table. This method adds
     *                columns to this vector.
     */
    virtual void alloc_combine_columns(size_t num_groups,
                                       std::vector<array_info*>& out_cols) {
        Bodo_FTypes::FTypeEnum combine_ftype = combine_funcs[ftype];
        for (auto col : update_cols) {
            bodo_array_type::arr_type_enum arr_type = col->arr_type;
            Bodo_CTypes::CTypeEnum dtype = col->dtype;
            int64_t num_categories = col->num_categories;
            // calling this modifies arr_type and dtype
            get_groupby_output_dtype(combine_ftype, arr_type, dtype, false);
            out_cols.push_back(alloc_array(num_groups, 1, 1, arr_type, dtype, 0,
                                           num_categories));
            combine_cols.push_back(out_cols.back());
        }
    }

    /**
     * Perform combine step for this column set. This will fill my columns with
     * the result of the aggregation operation corresponding to this column set
     * @param grouping info calculated by GroupbyPipeline
     */
    virtual void combine(const grouping_info& grp_info) {
        Bodo_FTypes::FTypeEnum combine_ftype = combine_funcs[ftype];
        std::vector<array_info*> aux_cols(combine_cols.begin() + 1,
                                          combine_cols.end());
        for (auto col : combine_cols)
            aggfunc_output_initialize(col, combine_ftype);
        do_apply_to_column(update_cols[0], combine_cols[0], aux_cols, grp_info,
                           combine_ftype);
    }

    /**
     * Perform eval step for this column set. This will fill the output column
     * with the final result of the aggregation operation corresponding to this
     * column set
     * @param grouping info calculated by GroupbyPipeline
     */
    virtual void eval(const grouping_info& grp_info) {}

    /**
     * Obtain the final output column resulting from the groupby operation on
     * this column set. This will free all other intermediate or auxiliary
     * columns (if any) used by the column set (like reduction variables).
     */
    virtual array_info* getOutputColumn() {
        std::vector<array_info*>* mycols;
        if (combine_step)
            mycols = &combine_cols;
        else
            mycols = &update_cols;
        array_info* out_col = mycols->at(0);
        for (auto it = mycols->begin() + 1; it != mycols->end(); it++) {
            array_info* a = *it;
            decref_array(a);
            delete a;
        }
        return out_col;
    }

   protected:
    friend class GroupbyPipeline;
    array_info* in_col;  // the input column (from groupby input table) to which
                         // this column set corresponds to
    int ftype;
    bool combine_step;  // GroupbyPipeline is going to perform a combine
                        // operation or not
    std::vector<array_info*> update_cols;   // columns for update step
    std::vector<array_info*> combine_cols;  // columns for combine step
};

class MeanColSet : public BasicColSet {
   public:
    MeanColSet(array_info* in_col, bool combine_step)
        : BasicColSet(in_col, Bodo_FTypes::mean, combine_step) {}
    virtual ~MeanColSet() {}

    virtual void alloc_update_columns(size_t num_groups,
                                      std::vector<array_info*>& out_cols) {
        array_info* c1 =
            alloc_array(num_groups, 1, 1, bodo_array_type::NUMPY,
                        Bodo_CTypes::FLOAT64, 0, 0);  // for sum and result
        array_info* c2 = alloc_array(num_groups, 1, 1, bodo_array_type::NUMPY,
                                     Bodo_CTypes::UINT64, 0, 0);  // for counts
        out_cols.push_back(c1);
        out_cols.push_back(c2);
        update_cols.push_back(c1);
        update_cols.push_back(c2);
    }

    virtual void update(const grouping_info& grp_info) {
        std::vector<array_info*> aux_cols = {update_cols[1]};
        aggfunc_output_initialize(update_cols[0], ftype);
        aggfunc_output_initialize(update_cols[1], ftype);
        do_apply_to_column(in_col, update_cols[0], aux_cols, grp_info, ftype);
    }

    virtual void combine(const grouping_info& grp_info) {
        std::vector<array_info*> aux_cols;
        aggfunc_output_initialize(combine_cols[0], Bodo_FTypes::sum);
        aggfunc_output_initialize(combine_cols[1], Bodo_FTypes::sum);
        do_apply_to_column(update_cols[0], combine_cols[0], aux_cols, grp_info,
                           Bodo_FTypes::sum);
        do_apply_to_column(update_cols[1], combine_cols[1], aux_cols, grp_info,
                           Bodo_FTypes::sum);
    }

    virtual void eval(const grouping_info& grp_info) {
        std::vector<array_info*> aux_cols;
        if (combine_step)
            do_apply_to_column(combine_cols[1], combine_cols[0], aux_cols,
                               grp_info, Bodo_FTypes::mean_eval);
        else
            do_apply_to_column(update_cols[1], update_cols[0], aux_cols,
                               grp_info, Bodo_FTypes::mean_eval);
    }
};

class IdxMinMaxColSet : public BasicColSet {
   public:
    IdxMinMaxColSet(array_info* in_col, array_info* _index_col, int ftype,
                    bool combine_step)
        : BasicColSet(in_col, ftype, combine_step), index_col(_index_col) {}
    virtual ~IdxMinMaxColSet() {}

    virtual void alloc_update_columns(size_t num_groups,
                                      std::vector<array_info*>& out_cols) {
        // output column containing index values. dummy for now. will be
        // assigned the real data at the end of update()
        array_info* out_col = alloc_array(num_groups, 1, 1, index_col->arr_type,
                                          index_col->dtype, 0, 0);
        // create array to store min/max value
        array_info* max_col = alloc_array(num_groups, 1, 1, in_col->arr_type,
                                          in_col->dtype, 0, 0);  // for min/max
        // create array to store index position of min/max value
        array_info* index_pos_col =
            alloc_array(num_groups, 1, 1, bodo_array_type::NUMPY,
                        Bodo_CTypes::UINT64, 0, 0);
        out_cols.push_back(out_col);
        out_cols.push_back(max_col);
        update_cols.push_back(out_col);
        update_cols.push_back(max_col);
        update_cols.push_back(index_pos_col);
    }

    virtual void update(const grouping_info& grp_info) {
        array_info* index_pos_col = update_cols[2];
        std::vector<array_info*> aux_cols = {index_pos_col};
        if (ftype == Bodo_FTypes::idxmax)
            aggfunc_output_initialize(update_cols[1], Bodo_FTypes::max);
        if (ftype == Bodo_FTypes::idxmin)
            aggfunc_output_initialize(update_cols[1], Bodo_FTypes::min);
        aggfunc_output_initialize(index_pos_col,
                                  Bodo_FTypes::count);  // zero init
        do_apply_to_column(in_col, update_cols[1], aux_cols, grp_info, ftype);

        array_info* real_out_col =
            RetrieveArray_SingleColumn_arr(index_col, index_pos_col);
        array_info* out_col = update_cols[0];
        *out_col = std::move(*real_out_col);
        delete real_out_col;
        decref_array(index_pos_col);
        delete index_pos_col;
        update_cols.pop_back();
    }

    virtual void alloc_combine_columns(size_t num_groups,
                                       std::vector<array_info*>& out_cols) {
        // output column containing index values. dummy for now. will be
        // assigned the real data at the end of combine()
        array_info* out_col = alloc_array(num_groups, 1, 1, index_col->arr_type,
                                          index_col->dtype, 0, 0);
        // create array to store min/max value
        array_info* max_col = alloc_array(num_groups, 1, 1, in_col->arr_type,
                                          in_col->dtype, 0, 0);  // for min/max
        // create array to store index position of min/max value
        array_info* index_pos_col =
            alloc_array(num_groups, 1, 1, bodo_array_type::NUMPY,
                        Bodo_CTypes::UINT64, 0, 0);
        out_cols.push_back(out_col);
        out_cols.push_back(max_col);
        combine_cols.push_back(out_col);
        combine_cols.push_back(max_col);
        combine_cols.push_back(index_pos_col);
    }

    virtual void combine(const grouping_info& grp_info) {
        array_info* index_pos_col = combine_cols[2];
        std::vector<array_info*> aux_cols = {index_pos_col};
        if (ftype == Bodo_FTypes::idxmax)
            aggfunc_output_initialize(combine_cols[1], Bodo_FTypes::max);
        if (ftype == Bodo_FTypes::idxmin)
            aggfunc_output_initialize(combine_cols[1], Bodo_FTypes::min);
        aggfunc_output_initialize(index_pos_col,
                                  Bodo_FTypes::count);  // zero init
        do_apply_to_column(update_cols[1], combine_cols[1], aux_cols, grp_info,
                           ftype);

        array_info* real_out_col =
            RetrieveArray_SingleColumn_arr(update_cols[0], index_pos_col);
        array_info* out_col = combine_cols[0];
        *out_col = std::move(*real_out_col);
        delete real_out_col;
        decref_array(index_pos_col);
        delete index_pos_col;
        combine_cols.pop_back();
    }

   private:
    array_info* index_col;
};

class VarStdColSet : public BasicColSet {
   public:
    VarStdColSet(array_info* in_col, int ftype, bool combine_step)
        : BasicColSet(in_col, ftype, combine_step) {}
    virtual ~VarStdColSet() {}

    virtual void alloc_update_columns(size_t num_groups,
                                      std::vector<array_info*>& out_cols) {
        if (!combine_step) {
            // need to create output column now
            array_info* col =
                alloc_array(num_groups, 1, 1, bodo_array_type::NUMPY,
                            Bodo_CTypes::FLOAT64, 0, 0);  // for result
            out_cols.push_back(col);
            update_cols.push_back(col);
        }
        array_info* count_col =
            alloc_array(num_groups, 1, 1, bodo_array_type::NUMPY,
                        Bodo_CTypes::UINT64, 0, 0);
        array_info* mean_col =
            alloc_array(num_groups, 1, 1, bodo_array_type::NUMPY,
                        Bodo_CTypes::FLOAT64, 0, 0);
        array_info* m2_col =
            alloc_array(num_groups, 1, 1, bodo_array_type::NUMPY,
                        Bodo_CTypes::FLOAT64, 0, 0);
        aggfunc_output_initialize(count_col,
                                  Bodo_FTypes::count);  // zero initialize
        aggfunc_output_initialize(mean_col,
                                  Bodo_FTypes::count);  // zero initialize
        aggfunc_output_initialize(m2_col,
                                  Bodo_FTypes::count);  // zero initialize
        out_cols.push_back(count_col);
        out_cols.push_back(mean_col);
        out_cols.push_back(m2_col);
        update_cols.push_back(count_col);
        update_cols.push_back(mean_col);
        update_cols.push_back(m2_col);
    }

    virtual void update(const grouping_info& grp_info) {
        if (!combine_step) {
            std::vector<array_info*> aux_cols = {update_cols[1], update_cols[2],
                                                 update_cols[3]};
            do_apply_to_column(in_col, update_cols[1], aux_cols, grp_info,
                               ftype);
        } else {
            std::vector<array_info*> aux_cols = {update_cols[0], update_cols[1],
                                                 update_cols[2]};
            do_apply_to_column(in_col, update_cols[0], aux_cols, grp_info,
                               ftype);
        }
    }

    virtual void alloc_combine_columns(size_t num_groups,
                                       std::vector<array_info*>& out_cols) {
        array_info* col =
            alloc_array(num_groups, 1, 1, bodo_array_type::NUMPY,
                        Bodo_CTypes::FLOAT64, 0, 0);  // for result
        out_cols.push_back(col);
        combine_cols.push_back(col);
        BasicColSet::alloc_combine_columns(num_groups, out_cols);
    }

    virtual void combine(const grouping_info& grp_info) {
        array_info* count_col_in = update_cols[0];
        array_info* mean_col_in = update_cols[1];
        array_info* m2_col_in = update_cols[2];
        array_info* count_col_out = combine_cols[1];
        array_info* mean_col_out = combine_cols[2];
        array_info* m2_col_out = combine_cols[3];
        aggfunc_output_initialize(count_col_out, Bodo_FTypes::count);
        aggfunc_output_initialize(mean_col_out, Bodo_FTypes::count);
        aggfunc_output_initialize(m2_col_out, Bodo_FTypes::count);
        var_combine(count_col_in, mean_col_in, m2_col_in, count_col_out,
                    mean_col_out, m2_col_out, grp_info.row_to_group);
    }

    virtual void eval(const grouping_info& grp_info) {
        std::vector<array_info*>* mycols;
        if (combine_step)
            mycols = &combine_cols;
        else
            mycols = &update_cols;

        std::vector<array_info*> aux_cols = {mycols->at(1), mycols->at(2),
                                             mycols->at(3)};
        if (ftype == Bodo_FTypes::var)
            do_apply_to_column(mycols->at(0), mycols->at(0), aux_cols, grp_info,
                               Bodo_FTypes::var_eval);
        else
            do_apply_to_column(mycols->at(0), mycols->at(0), aux_cols, grp_info,
                               Bodo_FTypes::std_eval);
    }
};

class UdfColSet : public BasicColSet {
   public:
    UdfColSet(array_info* in_col, bool combine_step, table_info* udf_table,
              int udf_table_idx, int n_redvars)
        : BasicColSet(in_col, Bodo_FTypes::udf, combine_step),
          udf_table(udf_table),
          udf_table_idx(udf_table_idx),
          n_redvars(n_redvars) {}
    virtual ~UdfColSet() {}

    virtual void alloc_update_columns(size_t num_groups,
                                      std::vector<array_info*>& out_cols) {
        int offset = 0;
        if (combine_step) offset = 1;
        // for update table we only need redvars (skip first column which is
        // output column)
        for (int i = udf_table_idx + offset; i < udf_table_idx + 1 + n_redvars;
             i++) {
            // we get the type from the udf dummy table that was passed to C++
            // library
            bodo_array_type::arr_type_enum arr_type =
                udf_table->columns[i]->arr_type;
            Bodo_CTypes::CTypeEnum dtype = udf_table->columns[i]->dtype;
            int64_t num_categories = udf_table->columns[i]->num_categories;
            out_cols.push_back(alloc_array(num_groups, 1, 1, arr_type, dtype, 0,
                                           num_categories));
            if (!combine_step) update_cols.push_back(out_cols.back());
        }
    }

    virtual void update(const grouping_info& grp_info) {
        // do nothing because this is done in JIT-compiled code (invoked from
        // GroupbyPipeline once for all udf columns sets)
    }

    virtual std::vector<array_info*>::iterator update_after_shuffle(
        std::vector<array_info*>::iterator& it) {
        // UdfColSet doesn't keep the update cols, return the updated iterator
        return it + n_redvars;
    }

    virtual void alloc_combine_columns(size_t num_groups,
                                       std::vector<array_info*>& out_cols) {
        for (int i = udf_table_idx; i < udf_table_idx + 1 + n_redvars; i++) {
            // we get the type from the udf dummy table that was passed to C++
            // library
            bodo_array_type::arr_type_enum arr_type =
                udf_table->columns[i]->arr_type;
            Bodo_CTypes::CTypeEnum dtype = udf_table->columns[i]->dtype;
            int64_t num_categories = udf_table->columns[i]->num_categories;
            out_cols.push_back(alloc_array(num_groups, 1, 1, arr_type, dtype, 0,
                                           num_categories));
            combine_cols.push_back(out_cols.back());
        }
    }

    virtual void combine(const grouping_info& grp_info) {
        // do nothing because this is done in JIT-compiled code (invoked from
        // GroupbyPipeline once for all udf columns sets)
    }

    virtual void eval(const grouping_info& grp_info) {
        // do nothing because this is done in JIT-compiled code (invoked from
        // GroupbyPipeline once for all udf columns sets)
    }

   private:
    table_info* udf_table;  // the table containing type info for UDF columns
    int udf_table_idx;      // index to my information in the udf table
    int n_redvars;          // number of redvar columns this UDF uses
};

class MedianColSet : public BasicColSet {
   public:
    MedianColSet(array_info* in_col, bool _skipna)
        : BasicColSet(in_col, Bodo_FTypes::median, false), skipna(_skipna) {}
    virtual ~MedianColSet() {}

    virtual void update(const grouping_info& grp_info) {
        median_computation(in_col, update_cols[0], grp_info, skipna);
    }

   private:
    bool skipna;
};

class NUniqueColSet : public BasicColSet {
   public:
    NUniqueColSet(array_info* in_col, bool _dropna)
        : BasicColSet(in_col, Bodo_FTypes::nunique, false), dropna(_dropna) {}
    virtual ~NUniqueColSet() {}

    virtual void update(const grouping_info& grp_info) {
        nunique_computation(in_col, update_cols[0], grp_info, dropna);
    }

   private:
    bool dropna;
};

class CumOpColSet : public BasicColSet {
   public:
    CumOpColSet(array_info* in_col, int ftype, bool _skipna)
        : BasicColSet(in_col, ftype, false), skipna(_skipna) {}
    virtual ~CumOpColSet() {}

    virtual void alloc_update_columns(size_t num_groups,
                                      std::vector<array_info*>& out_cols) {
        // NOTE: output size of cum ops is the same as input size
        //       (NOT the number of groups)
        out_cols.push_back(alloc_array(in_col->length, 1, 1, in_col->arr_type,
                                       in_col->dtype, 0,
                                       in_col->num_categories));
        update_cols.push_back(out_cols.back());
    }

    virtual void update(const grouping_info& grp_info) {
        cumulative_computation(in_col, update_cols[0], grp_info, ftype, skipna);
    }

   private:
    bool skipna;
};

class GroupbyPipeline {
   public:
    GroupbyPipeline(table_info* _in_table, int64_t _num_keys,
                    bool input_has_index, bool _is_parallel, int* ftypes,
                    int* func_offsets, int* _udf_nredvars,
                    table_info* _udf_table, udf_table_op_fn update_cb,
                    udf_table_op_fn combine_cb, udf_eval_fn eval_cb,
                    bool skipna, bool _return_key, bool _return_index)
        : in_table(_in_table),
          num_keys(_num_keys),
          is_parallel(_is_parallel),
          return_key(_return_key),
          return_index(_return_index),
          udf_table(_udf_table),
          udf_n_redvars(_udf_nredvars) {
        udf_info = {udf_table, update_cb, combine_cb, eval_cb};
        // if true, the last column is the index on input and output.
        // this is relevant only to cumulative operation like cumsum.
        int index_i = int(input_has_index);
#ifdef DEBUG_GROUPBY
        std::cout << " index_i=" << index_i << "\n";
#endif
        // NOTE cumulative operations (cumsum, cumprod, etc.) cannot be mixed
        // with non cumulative ops. This is checked at compile time in
        // aggregate.py

        for (int i = 0;
             i < func_offsets[in_table->ncols() - num_keys - index_i]; i++) {
            int ftype = ftypes[i];
            if (ftype == Bodo_FTypes::nunique || ftype == Bodo_FTypes::median ||
                ftype == Bodo_FTypes::cumsum || ftype == Bodo_FTypes::cumprod ||
                ftype == Bodo_FTypes::cummin || ftype == Bodo_FTypes::cummax) {
                // these operations first require shuffling the data to
                // gather all rows with the same key in the same process
                if (is_parallel) shuffle_before_update = true;
                // these operations require extended group info
                req_extended_group_info = true;
                if (ftype == Bodo_FTypes::cumsum ||
                    ftype == Bodo_FTypes::cummin ||
                    ftype == Bodo_FTypes::cumprod ||
                    ftype == Bodo_FTypes::cummax)
                    cumulative_op = true;
                break;
            }
        }
#ifdef DEBUG_GROUPBY
        std::cout << "cumulative_op=" << cumulative_op << "\n";
        std::cout << "req_extended_group_info=" << req_extended_group_info
                  << "\n";
        std::cout << "shuffle_before_update=" << shuffle_before_update << "\n";
#endif
        if (shuffle_before_update) {
            // Code below is equivalent to
            // table_info* shuf_table = shuffle_table(update_table, num_keys);
            // We do this more complicated construction because we may need
            // later the hashes and comm_info.
            comm_info_ptr = new mpi_comm_info(in_table->columns);
            hashes = hash_keys_table(in_table, num_keys, SEED_HASH_PARTITION);
            comm_info_ptr->set_counts(hashes);
            // shuffle_table_kernel steals the reference but we still need it
            // for the code after C++ groupby
            for (auto a : in_table->columns) incref_array(a);
            in_table = shuffle_table_kernel(in_table, hashes, *comm_info_ptr);
            if (!cumulative_op) {
                delete hashes;
                delete comm_info_ptr;
            }
        }
#ifdef DEBUG_GROUPBY
        std::cout << "After shuffle\n";
#endif

        // a combine operation is only necessary when data is distributed and
        // a shuffle has not been done at the start of the groupby pipeline
        do_combine = is_parallel && !shuffle_before_update;
#ifdef DEBUG_GROUPBY
        std::cout << "do_combine=" << do_combine << "\n";
#endif

        array_info* index_col = nullptr;
        if (input_has_index)
            index_col = in_table->columns[in_table->columns.size() - 1];

        // construct the column sets, one for each (input_column, func) pair
        // ftypes is an array of function types received from generated code,
        // and has one ftype for each (input_column, func) pair
        int k = 0;
        for (int64_t i = num_keys; i < in_table->ncols() - index_i; i++, k++) {
            array_info* col = in_table->columns[i];
            int start = func_offsets[k];
            int end = func_offsets[k + 1];
            for (int j = start; j != end; j++) {
                col_sets.push_back(
                    makeColSet(col, index_col, ftypes[j], do_combine, skipna));
            }
        }
#ifdef DEBUG_GROUPBY
        std::cout << "End of constructor\n";
#endif
    }

    ~GroupbyPipeline() {
        for (auto col_set : col_sets) delete col_set;
    }

    /**
     * This is the main control flow of the Groupby pipeline.
     */
    table_info* run() {
#ifdef DEBUG_GROUPBY
        std::cout << "Before update()\n";
#endif
        update();
#ifdef DEBUG_GROUPBY
        std::cout << "After update()\n";
#endif
        if (shuffle_before_update)
            // in_table was created in C++ during shuffling and not needed
            // anymore
            delete_table_decref_arrays(in_table);
        if (is_parallel && !shuffle_before_update) {
            shuffle();
#ifdef DEBUG_GROUPBY
            std::cout << "After shuffle()\n";
#endif
            combine();
#ifdef DEBUG_GROUPBY
            std::cout << "After combine()\n";
#endif
        }
        eval();
#ifdef DEBUG_GROUPBY
        std::cout << "After eval()\n";
#endif
        return getOutputTable();
    }

   private:
    /**
     * Construct and return a column set based on the ftype.
     * @param groupby input column associated with this column set.
     * @param ftype function type associated with this column set.
     * @param do_combine whether GroupbyPipeline will perform combine operation
     *        or not.
     * @param skipna option used for nunique, cumsum, cumprod, cummin, cummax
     */
    BasicColSet* makeColSet(array_info* in_col, array_info* index_col,
                            int ftype, bool do_combine, bool skipna) {
        BasicColSet* col_set;
        switch (ftype) {
            case Bodo_FTypes::udf:
                col_set = new UdfColSet(in_col, do_combine, udf_table,
                                        udf_table_idx, udf_n_redvars[n_udf]);
                udf_table_idx += (1 + udf_n_redvars[n_udf]);
                n_udf++;
                return col_set;
            case Bodo_FTypes::median:
                return new MedianColSet(in_col, skipna);
            case Bodo_FTypes::nunique:
                return new NUniqueColSet(in_col, skipna);
            case Bodo_FTypes::cumsum:
            case Bodo_FTypes::cummin:
            case Bodo_FTypes::cummax:
            case Bodo_FTypes::cumprod:
                return new CumOpColSet(in_col, ftype, skipna);
            case Bodo_FTypes::mean:
                return new MeanColSet(in_col, do_combine);
            case Bodo_FTypes::var:
            case Bodo_FTypes::std:
                return new VarStdColSet(in_col, ftype, do_combine);
            case Bodo_FTypes::idxmin:
            case Bodo_FTypes::idxmax:
                return new IdxMinMaxColSet(in_col, index_col, ftype,
                                           do_combine);
            default:
                return new BasicColSet(in_col, ftype, do_combine);
        }
    }

    /**
     * The update step groups rows in the input table based on keys, and
     * aggregates them based on the function to be applied to the columns.
     * More specifically, it will invoke the update method of each column set.
     */
    void update() {
#ifdef DEBUG_GROUPBY
        std::cout << "Beginning of update()\n";
#endif
        in_table->num_keys = num_keys;
#ifdef DEBUG_GROUPBY
        std::cout << "req_extended_group_info=" << req_extended_group_info
                  << "\n";
#endif
        if (req_extended_group_info) {
            bool consider_missing = cumulative_op;
            grp_info = get_group_info_iterate(in_table, consider_missing);
        } else
            get_group_info(*in_table, grp_info.row_to_group,
                           grp_info.group_to_first_row, true);
        num_groups = grp_info.group_to_first_row.size();
#ifdef DEBUG_GROUPBY
        std::cout << "num_groups=" << num_groups << "\n";
#endif

        update_table = cur_table = new table_info();
        if (cumulative_op)
            num_keys = 0;  // there are no key columns in output of cumulative
                           // operations
        else
            alloc_init_keys(in_table, update_table);

        for (auto col_set : col_sets) {
            col_set->alloc_update_columns(num_groups, update_table->columns);
            col_set->update(grp_info);
        }
#ifdef DEBUG_GROUPBY
        std::cout << "After alloc_update_columns\n";
#endif
        if (return_index)
            update_table->columns.push_back(
                copy_array(in_table->columns.back()));
#ifdef DEBUG_GROUPBY
        std::cout << "After return_index\n";
#endif
        if (n_udf > 0)
            udf_info.update(in_table, update_table,
                            grp_info.row_to_group.data());
#ifdef DEBUG_GROUPBY
        std::cout << "After n_udf\n";
#endif
    }

    /**
     * Shuffles the update table and updates the column sets with the newly
     * shuffled table.
     */
    void shuffle() {
        comm_info_ptr = new mpi_comm_info(update_table->columns);
        hashes = hash_keys_table(update_table, num_keys, SEED_HASH_PARTITION);
        comm_info_ptr->set_counts(hashes);
        table_info* shuf_table =
            shuffle_table_kernel(update_table, hashes, *comm_info_ptr);
        if (!cumulative_op) {
            delete hashes;
            delete comm_info_ptr;
        }
        // NOTE: shuffle_table_kernel decrefs input arrays
        delete_table(update_table);
        update_table = cur_table = shuf_table;

        // update column sets with columns from shuffled table
        auto it = update_table->columns.begin() + num_keys;
        for (auto col_set : col_sets) it = col_set->update_after_shuffle(it);
    }

    /**
     * The combine step is performed after update and shuffle. It groups rows
     * in shuffled table based on keys, and aggregates them based on the
     * function to be applied to the columns. More specifically, it will invoke
     * the combine method of each column set.
     */
    void combine() {
        grp_info.row_to_group.clear();
        grp_info.group_to_first_row.clear();
        update_table->num_keys = num_keys;
        get_group_info(*update_table, grp_info.row_to_group,
                       grp_info.group_to_first_row, false);
        num_groups = grp_info.group_to_first_row.size();

        combine_table = cur_table = new table_info();
        alloc_init_keys(update_table, combine_table);
        for (auto col_set : col_sets) {
            col_set->alloc_combine_columns(num_groups, combine_table->columns);
            col_set->combine(grp_info);
        }
        if (n_udf > 0)
            udf_info.combine(update_table, combine_table,
                             grp_info.row_to_group.data());
        delete_table_decref_arrays(update_table);
    }

    /**
     * The eval step generates the final result (output column) for each column
     * set. It call the eval method of each column set.
     */
    void eval() {
        for (auto col_set : col_sets) col_set->eval(grp_info);
        if (n_udf > 0) udf_info.eval(cur_table);
    }

    /**
     * Returns the final output table which is the result of the groupby.
     */
    table_info* getOutputTable() {
        table_info* out_table = new table_info();
        if (return_key)
            out_table->columns.assign(cur_table->columns.begin(),
                                      cur_table->columns.begin() + num_keys);
        for (BasicColSet* col_set : col_sets)
            out_table->columns.push_back(col_set->getOutputColumn());
        if (return_index)
            out_table->columns.push_back(cur_table->columns.back());
        if (cumulative_op && is_parallel) {
#ifdef DEBUG_GROUPBY
            std::cout << "Before reverse_shuffle_table_kernel\n";
#endif
            table_info* revshuf_table =
                reverse_shuffle_table_kernel(out_table, hashes, *comm_info_ptr);
            delete hashes;
            delete comm_info_ptr;
#ifdef DEBUG_GROUPBY
            std::cout << "After reverse_shuffle_table_kernel\n";
#endif
            delete_table(out_table);
#ifdef DEBUG_GROUPBY
            std::cout << "After delete_table_decref_arrays\n";
#endif
            out_table = revshuf_table;
#ifdef DEBUG_GROUPBY
            std::cout << "After out_table assignation\n";
#endif
        }
        delete cur_table;
        return out_table;
    }

    /**
     * Allocate and fill key columns, based on grouping info. It uses the
     * values of key columns from from_table to populate out_table.
     */
    void alloc_init_keys(table_info* from_table, table_info* out_table) {
#ifdef DEBUG_GROUPBY
        std::cout << "Beginning of alloc_init_keys\n";
#endif
        for (int64_t i = 0; i < num_keys; i++) {
            const array_info* key_col = (*from_table)[i];
            array_info* new_key_col;
            if (key_col->arr_type == bodo_array_type::NUMPY ||
                key_col->arr_type == bodo_array_type::CATEGORICAL ||
                key_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
                new_key_col =
                    alloc_array(num_groups, 1, 1, key_col->arr_type,
                                key_col->dtype, 0, key_col->num_categories);
                int64_t dtype_size = numpy_item_size[key_col->dtype];
                for (size_t j = 0; j < num_groups; j++)
                    memcpy(new_key_col->data1 + j * dtype_size,
                           key_col->data1 +
                               grp_info.group_to_first_row[j] * dtype_size,
                           dtype_size);
                if (key_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
                    uint8_t* in_null_bitmask = (uint8_t*)key_col->null_bitmask;
                    uint8_t* out_null_bitmask =
                        (uint8_t*)new_key_col->null_bitmask;
                    for (size_t j = 0; j < num_groups; j++) {
                        size_t in_row = grp_info.group_to_first_row[j];
                        SetBitTo(out_null_bitmask, j,
                                 GetBit(in_null_bitmask, in_row));
                    }
                }
            }
            if (key_col->arr_type == bodo_array_type::STRING) {
                // new key col will have num_groups rows containing the
                // string for each group
                int64_t n_chars = 0;  // total number of chars of all keys for
                                      // this column
                uint32_t* in_offsets = (uint32_t*)key_col->data2;
                for (size_t j = 0; j < num_groups; j++) {
                    int64_t row = grp_info.group_to_first_row[j];
                    n_chars += in_offsets[row + 1] - in_offsets[row];
                }
                new_key_col =
                    alloc_array(num_groups, n_chars, 1, key_col->arr_type,
                                key_col->dtype, 0, key_col->num_categories);

                uint8_t* in_null_bitmask = (uint8_t*)key_col->null_bitmask;
                uint8_t* out_null_bitmask = (uint8_t*)new_key_col->null_bitmask;
                uint32_t* out_offsets = (uint32_t*)new_key_col->data2;
                uint32_t pos = 0;
                for (size_t j = 0; j < num_groups; j++) {
                    size_t in_row = grp_info.group_to_first_row[j];
                    uint32_t start_offset = in_offsets[in_row];
                    uint32_t str_len = in_offsets[in_row + 1] - start_offset;
                    out_offsets[j] = pos;
                    memcpy(&new_key_col->data1[pos],
                           &key_col->data1[start_offset], str_len);
                    pos += str_len;
                    SetBitTo(out_null_bitmask, j,
                             GetBit(in_null_bitmask, in_row));
                }
                out_offsets[num_groups] = pos;
            }
            if (key_col->arr_type == bodo_array_type::LIST_STRING) {
                // new key col will have num_groups rows containing the
                // list string for each group
                int64_t n_strings = 0;  // total number of strings of all keys
                                        // for this column
                int64_t n_chars = 0;  // total number of chars of all keys for
                                      // this column
                uint32_t* in_index_offsets = (uint32_t*)key_col->data3;
                uint32_t* in_data_offsets = (uint32_t*)key_col->data2;
                for (size_t j = 0; j < num_groups; j++) {
                    int64_t row = grp_info.group_to_first_row[j];
                    n_strings +=
                        in_index_offsets[row + 1] - in_index_offsets[row];
                    n_chars += in_data_offsets[in_index_offsets[row + 1]] -
                               in_data_offsets[in_index_offsets[row]];
                }
                new_key_col = alloc_array(num_groups, n_strings, n_chars,
                                          key_col->arr_type, key_col->dtype, 0,
                                          key_col->num_categories);
                uint8_t* in_null_bitmask = (uint8_t*)key_col->null_bitmask;
                uint8_t* out_null_bitmask = (uint8_t*)new_key_col->null_bitmask;
                uint8_t* in_sub_null_bitmask =
                    (uint8_t*)key_col->sub_null_bitmask;
                uint8_t* out_sub_null_bitmask =
                    (uint8_t*)new_key_col->sub_null_bitmask;
                uint32_t* out_index_offsets = (uint32_t*)new_key_col->data3;
                uint32_t* out_data_offsets = (uint32_t*)new_key_col->data2;
                uint32_t pos_data = 0;
                uint32_t pos_index = 0;
                out_data_offsets[0] = 0;
                out_index_offsets[0] = 0;
                for (size_t j = 0; j < num_groups; j++) {
                    size_t in_row = grp_info.group_to_first_row[j];
                    uint32_t size_index =
                        in_index_offsets[in_row + 1] - in_index_offsets[in_row];
                    uint32_t pos_start = in_index_offsets[in_row];
                    for (uint32_t i_str = 0; i_str < size_index; i_str++) {
                        uint32_t len_str =
                            in_data_offsets[pos_start + i_str + 1] -
                            in_data_offsets[pos_start + i_str];
                        pos_index++;
                        out_data_offsets[pos_index] =
                            out_data_offsets[pos_index - 1] + len_str;
                        bool bit =
                            GetBit(in_sub_null_bitmask, pos_start + i_str);
                        SetBitTo(out_sub_null_bitmask, pos_index, bit);
                    }
                    out_index_offsets[j + 1] = pos_index;
                    // Now the strings themselves
                    uint32_t in_start_offset =
                        in_data_offsets[in_index_offsets[in_row]];
                    uint32_t n_chars_o =
                        in_data_offsets[in_index_offsets[in_row + 1]] -
                        in_data_offsets[in_index_offsets[in_row]];
                    memcpy(&new_key_col->data1[pos_data],
                           &key_col->data1[in_start_offset], n_chars_o);
                    pos_data += n_chars_o;
                    SetBitTo(out_null_bitmask, j,
                             GetBit(in_null_bitmask, in_row));
                }
            }
            out_table->columns.push_back(new_key_col);
        }
#ifdef DEBUG_GROUPBY
        std::cout << "End of alloc_init_keys\n";
#endif
    }

    table_info* in_table;  // input table of groupby
    int64_t num_keys;
    bool is_parallel;
    bool return_key;
    bool return_index;
    std::vector<BasicColSet*> col_sets;
    table_info* udf_table;
    int* udf_n_redvars;
    int n_udf = 0;
    int udf_table_idx = 0;
    // shuffling before update requires more communication and is needed
    // when one of the groupby functions is
    // median/nunique/cumsum/cumprod/cummin/cummax
    bool shuffle_before_update = false;
    bool cumulative_op = false;
    bool req_extended_group_info = false;
    bool do_combine;

    udfinfo_t udf_info;

    table_info* update_table = nullptr;
    table_info* combine_table = nullptr;
    table_info* cur_table = nullptr;

    grouping_info grp_info;
    size_t num_groups;
    // shuffling stuff
    uint32_t* hashes;
    mpi_comm_info* comm_info_ptr = nullptr;
};

template <typename Tkey, typename T, int dtype>
void mpi_exscan_computation_numpy_T(std::vector<array_info*>& out_arrs,
                                    array_info* cat_column,
                                    table_info* in_table, int64_t num_keys,
                                    int64_t k, int* ftypes, int* func_offsets,
                                    bool is_parallel, bool skipdropna) {
    int64_t n_rows = in_table->nrows();
    int start = func_offsets[k];
    int end = func_offsets[k + 1];
    int n_oper = end - start;
    int64_t max_row_idx = cat_column->num_categories;
#ifdef DEBUG_GROUPBY
    std::cout << "Beginning of mpi_exscan_computation_numpy_T\n";
    std::cout << "k=" << k << " max_row_idx=" << max_row_idx
              << " n_oper=" << n_oper << " is_parallel=" << is_parallel << "\n";
#endif
    std::vector<T> cumulative(max_row_idx * n_oper);
    for (int j = start; j != end; j++) {
        int ftype = ftypes[j];
        T value_init = -1;  // Not correct value
        if (ftype == Bodo_FTypes::cumsum) value_init = 0;
        if (ftype == Bodo_FTypes::cumprod) value_init = 1;
        if (ftype == Bodo_FTypes::cummax)
            value_init = std::numeric_limits<T>::min();
        if (ftype == Bodo_FTypes::cummin)
            value_init = std::numeric_limits<T>::max();
#ifdef DEBUG_GROUPBY
        std::cout << "j=" << j << " value_init=" << value_init << "\n";
#endif
        for (int i_row = 0; i_row < max_row_idx; i_row++)
            cumulative[i_row + max_row_idx * (j - start)] = value_init;
    }
#ifdef DEBUG_GROUPBY
    std::cout << "mpi_exscan_computation_numpy_T, step 2\n";
#endif
    std::vector<T> cumulative_recv = cumulative;
    array_info* in_col = in_table->columns[k + num_keys];
    T nan_value =
        GetTentry<T>(RetrieveNaNentry((Bodo_CTypes::CTypeEnum)dtype).data());
    Tkey miss_idx = -1;
    for (int j = start; j != end; j++) {
        array_info* work_col = out_arrs[j];
        int ftype = ftypes[j];
#ifdef DEBUG_GROUPBY
        std::cout << "j=" << j << " ftype=" << ftype << "\n";
#endif
        auto apply_oper = [&](std::function<T(T, T)> oper) -> void {
#ifdef DEBUG_GROUPBY_SYMBOL
            std::cout << "Beginning of apply_oper\n";
#endif
            for (int64_t i_row = 0; i_row < n_rows; i_row++) {
                Tkey idx = cat_column->at<Tkey>(i_row);
                if (idx == miss_idx) {
                    work_col->at<T>(i_row) = nan_value;
                } else {
                    size_t pos = idx + max_row_idx * (j - start);
                    T val = in_col->at<T>(i_row);
                    if (skipdropna && isnan_alltype<T, dtype>(val)) {
                        work_col->at<T>(i_row) = val;
                    } else {
                        T new_val = oper(val, cumulative[pos]);
                        work_col->at<T>(i_row) = new_val;
                        cumulative[pos] = new_val;
                    }
                }
#ifdef DEBUG_GROUPBY_FULL
                T out_val = work_col->at<T>(i_row);
                std::cout << "i_row=" << i_row << " idx=" << idx
                          << " out_val=" << out_val << "\n";
#endif
            }
#ifdef DEBUG_GROUPBY_SYMBOL
            std::cout << "Ending of apply_oper\n";
#endif
        };
        if (ftype == Bodo_FTypes::cumsum)
            apply_oper([](T val1, T val2) -> T { return val1 + val2; });
        if (ftype == Bodo_FTypes::cumprod)
            apply_oper([](T val1, T val2) -> T { return val1 * val2; });
        if (ftype == Bodo_FTypes::cummax)
            apply_oper(
                [](T val1, T val2) -> T { return std::max(val1, val2); });
        if (ftype == Bodo_FTypes::cummin)
            apply_oper(
                [](T val1, T val2) -> T { return std::min(val1, val2); });
    }
#ifdef DEBUG_GROUPBY
    std::cout << "mpi_exscan_computation_numpy_T, step 3 is_parallel="
              << is_parallel << "\n";
#endif
    if (!is_parallel) return;
    MPI_Datatype mpi_typ = get_MPI_typ(dtype);
    for (int j = start; j != end; j++) {
        T* data_s = cumulative.data() + max_row_idx * (j - start);
        T* data_r = cumulative_recv.data() + max_row_idx * (j - start);
        int ftype = ftypes[j];
        if (ftype == Bodo_FTypes::cumsum)
            MPI_Exscan(data_s, data_r, max_row_idx, mpi_typ, MPI_SUM,
                       MPI_COMM_WORLD);
        if (ftype == Bodo_FTypes::cumprod)
            MPI_Exscan(data_s, data_r, max_row_idx, mpi_typ, MPI_PROD,
                       MPI_COMM_WORLD);
        if (ftype == Bodo_FTypes::cummax)
            MPI_Exscan(data_s, data_r, max_row_idx, mpi_typ, MPI_MAX,
                       MPI_COMM_WORLD);
        if (ftype == Bodo_FTypes::cummin)
            MPI_Exscan(data_s, data_r, max_row_idx, mpi_typ, MPI_MIN,
                       MPI_COMM_WORLD);
    }
#ifdef DEBUG_GROUPBY
    std::cout << "mpi_exscan_computation_numpy_T, step 4\n";
#endif
    for (int j = start; j != end; j++) {
        array_info* work_col = out_arrs[j];
        int ftype = ftypes[j];
#ifdef DEBUG_GROUPBY
        std::cout << "j=" << j << "\n";
#endif
        // For skipdropna:
        //   The cumulative is never a NaN. The sum therefore works
        //   correctly whether val is a NaN or not.
        // For !skipdropna:
        //   the cumulative can be a NaN. The sum also works correctly.
        auto apply_oper = [&](std::function<T(T, T)> oper) -> void {
            for (int64_t i_row = 0; i_row < n_rows; i_row++) {
                Tkey idx = cat_column->at<Tkey>(i_row);
                if (idx != miss_idx) {
                    size_t pos = idx + max_row_idx * (j - start);
                    T val = work_col->at<T>(i_row);
                    T new_val = oper(val, cumulative_recv[pos]);
                    work_col->at<T>(i_row) = new_val;
                }
            }
        };
        if (ftype == Bodo_FTypes::cumsum)
            apply_oper([](T val1, T val2) -> T { return val1 + val2; });
        if (ftype == Bodo_FTypes::cumprod)
            apply_oper([](T val1, T val2) -> T { return val1 * val2; });
        if (ftype == Bodo_FTypes::cummax)
            apply_oper(
                [](T val1, T val2) -> T { return std::max(val1, val2); });
        if (ftype == Bodo_FTypes::cummin)
            apply_oper(
                [](T val1, T val2) -> T { return std::min(val1, val2); });
    }
#ifdef DEBUG_GROUPBY
    std::cout << "Leaving of mpi_exscan_computation_numpy_T\n";
#endif
}

template <typename Tkey, typename T, int dtype>
void mpi_exscan_computation_nullable_T(std::vector<array_info*>& out_arrs,
                                       array_info* cat_column,
                                       table_info* in_table, int64_t num_keys,
                                       int64_t k, int* ftypes,
                                       int* func_offsets, bool is_parallel,
                                       bool skipdropna) {
    int64_t n_rows = in_table->nrows();
    int start = func_offsets[k];
    int end = func_offsets[k + 1];
    int n_oper = end - start;
    int64_t max_row_idx = cat_column->num_categories;
#ifdef DEBUG_GROUPBY
    std::cout << "Beginning of mpi_exscan_computation_nullable_T\n";
    std::cout << "k=" << k << " max_row_idx=" << max_row_idx
              << " n_oper=" << n_oper << "\n";
#endif
    std::vector<T> cumulative(max_row_idx * n_oper);
    for (int j = start; j != end; j++) {
        int ftype = ftypes[j];
        T value_init = -1;  // Not correct value
        if (ftype == Bodo_FTypes::cumsum) value_init = 0;
        if (ftype == Bodo_FTypes::cumprod) value_init = 1;
        if (ftype == Bodo_FTypes::cummax)
            value_init = std::numeric_limits<T>::min();
        if (ftype == Bodo_FTypes::cummin)
            value_init = std::numeric_limits<T>::max();
#ifdef DEBUG_GROUPBY
        std::cout << "j=" << j << " value_init=" << value_init << "\n";
#endif
        for (int i_row = 0; i_row < max_row_idx; i_row++)
            cumulative[i_row + max_row_idx * (j - start)] = value_init;
    }
    std::vector<T> cumulative_recv = cumulative;
    std::vector<uint8_t> cumulative_mask, cumulative_mask_recv;
    // If we use skipdropna then we do not need to keep track of
    // the previous values
    if (!skipdropna) {
        cumulative_mask = std::vector<uint8_t>(max_row_idx * n_oper, 0);
        cumulative_mask_recv = std::vector<uint8_t>(max_row_idx * n_oper, 0);
    }
    array_info* in_col = in_table->columns[k + num_keys];
    uint8_t* null_bitmask_i = (uint8_t*)in_col->null_bitmask;
    Tkey miss_idx = -1;
    for (int j = start; j != end; j++) {
        array_info* work_col = out_arrs[j];
        uint8_t* null_bitmask_o = (uint8_t*)work_col->null_bitmask;
        int ftype = ftypes[j];
        auto apply_oper = [&](std::function<T(T, T)> oper) -> void {
            for (int64_t i_row = 0; i_row < n_rows; i_row++) {
                Tkey idx = cat_column->at<Tkey>(i_row);
                if (idx == miss_idx) {
                    SetBitTo(null_bitmask_o, i_row, false);
                } else {
                    size_t pos = idx + max_row_idx * (j - start);
                    T val = in_col->at<T>(i_row);
                    bool bit_i = GetBit(null_bitmask_i, i_row);
                    T new_val = oper(val, cumulative[pos]);
                    bool bit_o = bit_i;
                    work_col->at<T>(i_row) = new_val;
                    if (skipdropna) {
                        if (bit_i) cumulative[pos] = new_val;
                    } else {
                        if (bit_i) {
                            if (cumulative_mask[pos] == 1)
                                bit_o = false;
                            else
                                cumulative[pos] = new_val;
                        } else
                            cumulative_mask[pos] = 1;
                    }
                    SetBitTo(null_bitmask_o, i_row, bit_o);
                }
            }
        };
        if (ftype == Bodo_FTypes::cumsum)
            apply_oper([](T val1, T val2) -> T { return val1 + val2; });
        if (ftype == Bodo_FTypes::cumprod)
            apply_oper([](T val1, T val2) -> T { return val1 * val2; });
        if (ftype == Bodo_FTypes::cummax)
            apply_oper(
                [](T val1, T val2) -> T { return std::max(val1, val2); });
        if (ftype == Bodo_FTypes::cummin)
            apply_oper(
                [](T val1, T val2) -> T { return std::min(val1, val2); });
    }
    if (!is_parallel) return;
    MPI_Datatype mpi_typ = get_MPI_typ(dtype);
    for (int j = start; j != end; j++) {
        T* data_s = cumulative.data() + max_row_idx * (j - start);
        T* data_r = cumulative_recv.data() + max_row_idx * (j - start);
        int ftype = ftypes[j];
        if (ftype == Bodo_FTypes::cumsum)
            MPI_Exscan(data_s, data_r, max_row_idx, mpi_typ, MPI_SUM,
                       MPI_COMM_WORLD);
        if (ftype == Bodo_FTypes::cumprod)
            MPI_Exscan(data_s, data_r, max_row_idx, mpi_typ, MPI_PROD,
                       MPI_COMM_WORLD);
        if (ftype == Bodo_FTypes::cummax)
            MPI_Exscan(data_s, data_r, max_row_idx, mpi_typ, MPI_MAX,
                       MPI_COMM_WORLD);
        if (ftype == Bodo_FTypes::cummin)
            MPI_Exscan(data_s, data_r, max_row_idx, mpi_typ, MPI_MIN,
                       MPI_COMM_WORLD);
    }
    if (!skipdropna) {
        mpi_typ = get_MPI_typ(Bodo_CTypes::UINT8);
        MPI_Exscan(cumulative_mask.data(), cumulative_mask_recv.data(),
                   max_row_idx * n_oper, mpi_typ, MPI_MAX, MPI_COMM_WORLD);
    }
    for (int j = start; j != end; j++) {
        array_info* work_col = out_arrs[j];
        uint8_t* null_bitmask_o = (uint8_t*)work_col->null_bitmask;
        int ftype = ftypes[j];
        auto apply_oper = [&](std::function<T(T, T)> oper) -> void {
            for (int64_t i_row = 0; i_row < n_rows; i_row++) {
                Tkey idx = cat_column->at<Tkey>(i_row);
                if (idx != miss_idx) {
                    size_t pos = idx + max_row_idx * (j - start);
                    T val = work_col->at<T>(i_row);
                    T new_val = oper(val, cumulative_recv[pos]);
                    work_col->at<T>(i_row) = new_val;
                    if (!skipdropna && cumulative_mask_recv[pos] == 1) {
                        SetBitTo(null_bitmask_o, i_row, false);
                    }
                }
            }
        };
        if (ftype == Bodo_FTypes::cumsum)
            apply_oper([](T val1, T val2) -> T { return val1 + val2; });
        if (ftype == Bodo_FTypes::cumprod)
            apply_oper([](T val1, T val2) -> T { return val1 * val2; });
        if (ftype == Bodo_FTypes::cummax)
            apply_oper(
                [](T val1, T val2) -> T { return std::max(val1, val2); });
        if (ftype == Bodo_FTypes::cummin)
            apply_oper(
                [](T val1, T val2) -> T { return std::min(val1, val2); });
    }
#ifdef DEBUG_GROUPBY
    std::cout << "Leaving of mpi_exscan_computation_nullable_T\n";
#endif
}

template <typename Tkey, typename T, int dtype>
void mpi_exscan_computation_T(std::vector<array_info*>& out_arrs,
                              array_info* cat_column, table_info* in_table,
                              int64_t num_keys, int64_t k, int* ftypes,
                              int* func_offsets, bool is_parallel,
                              bool skipdropna) {
#ifdef DEBUG_GROUPBY
    std::cout << "Beginning of mpi_exscan_computation_T\n";
#endif
    array_info* in_col = in_table->columns[k + num_keys];
    if (in_col->arr_type == bodo_array_type::NUMPY)
        return mpi_exscan_computation_numpy_T<Tkey, T, dtype>(
            out_arrs, cat_column, in_table, num_keys, k, ftypes, func_offsets,
            is_parallel, skipdropna);
    else
        return mpi_exscan_computation_nullable_T<Tkey, T, dtype>(
            out_arrs, cat_column, in_table, num_keys, k, ftypes, func_offsets,
            is_parallel, skipdropna);
}

template <typename Tkey>
table_info* mpi_exscan_computation_Tkey(array_info* cat_column,
                                        table_info* in_table, int64_t num_keys,
                                        int* ftypes, int* func_offsets,
                                        bool is_parallel, bool skipdropna,
                                        bool return_key, bool return_index) {
#ifdef DEBUG_GROUPBY_SYMBOL
    std::cout << "mpi_exscan_computation_Tkey (in_table)\n";
#ifdef DEBUG_GROUPBY_FULL
    DEBUG_PrintSetOfColumn(std::cout, in_table->columns);
#endif
    DEBUG_PrintRefct(std::cout, in_table->columns);
#endif
    std::vector<array_info*> out_arrs;
    // We do not return the keys in output in the case of cumulative operations.
    int64_t n_rows = in_table->nrows();
    int return_index_i = return_index;
    int k = 0;
    for (int64_t i = num_keys; i < in_table->ncols() - return_index_i;
         i++, k++) {
        array_info* col = in_table->columns[i];
        int start = func_offsets[k];
        int end = func_offsets[k + 1];
        for (int j = start; j != end; j++) {
            array_info* out_col =
                alloc_array(n_rows, 1, 1, col->arr_type, col->dtype, 0,
                            col->num_categories);
            int ftype = ftypes[j];
            aggfunc_output_initialize(out_col, ftype);
            out_arrs.push_back(out_col);
        }
    }
    // Since each column can have different data type and MPI_Exscan can only do
    // one type at a time. thus we have an iteration over the columns of the
    // input table. But we can consider the various cumsum / cumprod / cummax /
    // cummin in turn.
    k = 0;
    for (int64_t i = num_keys; i < in_table->ncols() - return_index_i;
         i++, k++) {
        array_info* col = in_table->columns[i];
        const Bodo_CTypes::CTypeEnum dtype = col->dtype;
#ifdef DEBUG_GROUPBY
        std::cout << "MPI_EXSCAN_COMPUTATION_TKEY i=" << i
                  << " dtype=" << int(dtype) << "\n";
#endif
        if (dtype == Bodo_CTypes::INT8)
            mpi_exscan_computation_T<Tkey, int8_t, Bodo_CTypes::INT8>(
                out_arrs, cat_column, in_table, num_keys, k, ftypes,
                func_offsets, is_parallel, skipdropna);
        if (dtype == Bodo_CTypes::UINT8)
            mpi_exscan_computation_T<Tkey, uint8_t, Bodo_CTypes::UINT8>(
                out_arrs, cat_column, in_table, num_keys, k, ftypes,
                func_offsets, is_parallel, skipdropna);
        if (dtype == Bodo_CTypes::INT16)
            mpi_exscan_computation_T<Tkey, int16_t, Bodo_CTypes::INT16>(
                out_arrs, cat_column, in_table, num_keys, k, ftypes,
                func_offsets, is_parallel, skipdropna);
        if (dtype == Bodo_CTypes::UINT16)
            mpi_exscan_computation_T<Tkey, uint16_t, Bodo_CTypes::UINT16>(
                out_arrs, cat_column, in_table, num_keys, k, ftypes,
                func_offsets, is_parallel, skipdropna);
        if (dtype == Bodo_CTypes::INT32)
            mpi_exscan_computation_T<Tkey, int32_t, Bodo_CTypes::INT32>(
                out_arrs, cat_column, in_table, num_keys, k, ftypes,
                func_offsets, is_parallel, skipdropna);
        if (dtype == Bodo_CTypes::UINT32)
            mpi_exscan_computation_T<Tkey, uint32_t, Bodo_CTypes::UINT32>(
                out_arrs, cat_column, in_table, num_keys, k, ftypes,
                func_offsets, is_parallel, skipdropna);
        if (dtype == Bodo_CTypes::INT64)
            mpi_exscan_computation_T<Tkey, int64_t, Bodo_CTypes::INT64>(
                out_arrs, cat_column, in_table, num_keys, k, ftypes,
                func_offsets, is_parallel, skipdropna);
        if (dtype == Bodo_CTypes::UINT64)
            mpi_exscan_computation_T<Tkey, uint64_t, Bodo_CTypes::UINT64>(
                out_arrs, cat_column, in_table, num_keys, k, ftypes,
                func_offsets, is_parallel, skipdropna);
        if (dtype == Bodo_CTypes::FLOAT32)
            mpi_exscan_computation_T<Tkey, float, Bodo_CTypes::FLOAT32>(
                out_arrs, cat_column, in_table, num_keys, k, ftypes,
                func_offsets, is_parallel, skipdropna);
        if (dtype == Bodo_CTypes::FLOAT64)
            mpi_exscan_computation_T<Tkey, double, Bodo_CTypes::FLOAT64>(
                out_arrs, cat_column, in_table, num_keys, k, ftypes,
                func_offsets, is_parallel, skipdropna);
    }
    if (return_index) out_arrs.push_back(copy_array(in_table->columns.back()));
#ifdef DEBUG_GROUPBY
    std::cout << "mpi_exscan_computation(out_arrs)\n";
    DEBUG_PrintSetOfColumn(std::cout, out_arrs);
    DEBUG_PrintRefct(std::cout, out_arrs);
#endif
    return new table_info(out_arrs);
}

table_info* mpi_exscan_computation(array_info* cat_column, table_info* in_table,
                                   int64_t num_keys, int* ftypes,
                                   int* func_offsets, bool is_parallel,
                                   bool skipdropna, bool return_key,
                                   bool return_index) {
    const Bodo_CTypes::CTypeEnum dtype = cat_column->dtype;
#ifdef DEBUG_GROUPBY
    std::cout << "mpi_exscan_computation calling mpi_exscan_computation_Tkey "
                 "with dtype="
              << dtype << "\n";
#endif
    if (dtype == Bodo_CTypes::INT8)
        return mpi_exscan_computation_Tkey<int8_t>(
            cat_column, in_table, num_keys, ftypes, func_offsets, is_parallel,
            skipdropna, return_key, return_index);
    if (dtype == Bodo_CTypes::UINT8)
        return mpi_exscan_computation_Tkey<uint8_t>(
            cat_column, in_table, num_keys, ftypes, func_offsets, is_parallel,
            skipdropna, return_key, return_index);
    if (dtype == Bodo_CTypes::INT16)
        return mpi_exscan_computation_Tkey<int16_t>(
            cat_column, in_table, num_keys, ftypes, func_offsets, is_parallel,
            skipdropna, return_key, return_index);
    if (dtype == Bodo_CTypes::UINT16)
        return mpi_exscan_computation_Tkey<uint16_t>(
            cat_column, in_table, num_keys, ftypes, func_offsets, is_parallel,
            skipdropna, return_key, return_index);
    if (dtype == Bodo_CTypes::INT32)
        return mpi_exscan_computation_Tkey<int32_t>(
            cat_column, in_table, num_keys, ftypes, func_offsets, is_parallel,
            skipdropna, return_key, return_index);
    if (dtype == Bodo_CTypes::UINT32)
        return mpi_exscan_computation_Tkey<uint32_t>(
            cat_column, in_table, num_keys, ftypes, func_offsets, is_parallel,
            skipdropna, return_key, return_index);
    if (dtype == Bodo_CTypes::INT64)
        return mpi_exscan_computation_Tkey<int64_t>(
            cat_column, in_table, num_keys, ftypes, func_offsets, is_parallel,
            skipdropna, return_key, return_index);
    if (dtype == Bodo_CTypes::UINT64)
        return mpi_exscan_computation_Tkey<uint64_t>(
            cat_column, in_table, num_keys, ftypes, func_offsets, is_parallel,
            skipdropna, return_key, return_index);
    //
    Bodo_PyErr_SetString(PyExc_RuntimeError, "failed to find matching dtype");
    return nullptr;
}

array_info* compute_categorical_index(table_info* in_table, int64_t num_keys,
                                      bool is_parallel) {
#ifdef DEBUG_GROUPBY
    std::cout << "compute_categorical_index num_keys=" << num_keys
              << " is_parallel=" << is_parallel << "\n";
#endif
    // A rare case of incref since we are going to need the in_table after the
    // computation of red_table.
    for (int64_t i_key = 0; i_key < num_keys; i_key++)
        incref_array(in_table->columns[i_key]);
    table_info* red_table =
        drop_duplicates_nonnull_keys(in_table, num_keys, is_parallel);
#ifdef DEBUG_GROUPBY
    std::cout << "We have red_table\n";
#endif
    size_t n_rows_full, n_rows = red_table->nrows();
    if (is_parallel)
        MPI_Allreduce(&n_rows, &n_rows_full, 1, MPI_LONG_LONG_INT, MPI_SUM,
                      MPI_COMM_WORLD);
    else
        n_rows_full = n_rows;
#ifdef DEBUG_GROUPBY
    std::cout << "compute_categorical_index n_rows=" << n_rows
              << " n_rows_full=" << n_rows_full << "\n";
#endif
    if (n_rows_full > max_global_number_groups_exscan) {
      delete_table_decref_arrays(red_table);
      return nullptr;
    }

    // We are below threshold. Now doing an allgather for determining the keys.
    bool all_gather = true;
#ifdef DEBUG_GROUPBY
    std::cout << "Before gather_table\n";
#endif
    table_info* full_table = gather_table(red_table, num_keys, all_gather);
    delete_table(red_table);
#ifdef DEBUG_GROUPBY
    std::cout << "After gather_table\n";
#endif
    // Now building the map_container.
    uint32_t* hashes_full =
        hash_keys_table(full_table, num_keys, SEED_HASH_MULTIKEY);
    uint32_t* hashes_in_table =
        hash_keys_table(in_table, num_keys, SEED_HASH_MULTIKEY);
    std::vector<array_info*> concat_column(
        full_table->columns.begin(), full_table->columns.begin() + num_keys);
    concat_column.insert(concat_column.end(), in_table->columns.begin(),
                         in_table->columns.begin() + num_keys);
#ifdef DEBUG_GROUPBY
    std::cout << "concat_column has been built\n";
#endif
    std::function<size_t(size_t)> hash_fct = [&](size_t const& iRow) -> size_t {
        if (iRow < n_rows_full)
            return size_t(hashes_full[iRow]);
        else
            return size_t(hashes_in_table[iRow - n_rows_full]);
    };
    std::function<bool(size_t, size_t)> equal_fct =
        [&](size_t const& iRowA, size_t const& iRowB) -> bool {
        size_t jRowA, jRowB, shift_A, shift_B;
        if (iRowA < n_rows_full) {
            shift_A = 0;
            jRowA = iRowA;
        } else {
            shift_A = num_keys;
            jRowA = iRowA - n_rows_full;
        }
        if (iRowB < n_rows_full) {
            shift_B = 0;
            jRowB = iRowB;
        } else {
            shift_B = num_keys;
            jRowB = iRowB - n_rows_full;
        }
        bool test =
            TestEqual(concat_column, num_keys, shift_A, jRowA, shift_B, jRowB);
        return test;
    };
    UNORD_MAP_CONTAINER<size_t, size_t, std::function<size_t(size_t)>,
                        std::function<bool(size_t, size_t)>>
        entSet({}, hash_fct, equal_fct);
    for (size_t iRow = 0; iRow < size_t(n_rows_full); iRow++)
        entSet[iRow] = iRow;
    size_t n_rows_in = in_table->nrows();
#ifdef DEBUG_GROUPBY
    std::cout << "compute_categorical_index n_rows_full=" << n_rows_full
              << "\n";
#endif
    array_info* out_arr =
        alloc_categorical(n_rows_in, Bodo_CTypes::INT32, n_rows_full);
#ifdef DEBUG_GROUPBY
    std::cout << "scale=" << out_arr->scale
              << " precision=" << out_arr->precision
              << " num_categories=" << out_arr->num_categories << "\n";
#endif
    std::vector<array_info*> key_cols(in_table->columns.begin(),
                                      in_table->columns.begin() + num_keys);
    bool has_nulls = does_keys_have_nulls(key_cols);
    for (size_t iRow = 0; iRow < n_rows_in; iRow++) {
        int32_t pos;
        if (has_nulls) {
            if (does_row_has_nulls(key_cols, iRow))
                pos = -1;
            else
                pos = entSet[iRow + n_rows_full];
        } else {
            pos = entSet[iRow + n_rows_full];
        }
#ifdef DEBUG_GROUPBY_FULL
        std::cout << "compute_categorical_index iRow=" << iRow << " pos=" << pos
                  << "\n";
#endif
        out_arr->at<int32_t>(iRow) = pos;
    }
    delete_table_decref_arrays(full_table);
    return out_arr;
}

/* Determine the strategy to be used for the computation of the groupby.
   This computation is done whether a strategy is possible. But it also
   makes heuristic computation in order to reach a decision.
   ---
   @param in_table : input table
   @param num_keys : number of keys
   @param ftypes : the type of operations.
   @param func_offsets : the function offsets
   @param input_has_index : whether input table contains index col in last
   position
   @return strategy to use :
   ---0 will use the GroupbyPipeline based on hash partitioning
   ---1 will use the MPI_Exscan strategy with CATEGORICAL column
   ---2 will use the MPI_Exscan strategy with determination of the columns
 */
int determine_groupby_strategy(table_info* in_table, int64_t num_keys,
                               int* ftypes, int* func_offsets, bool is_parallel,
                               bool input_has_index) {
    // First decision: If it is cumulative, then we can use the MPI_Exscan.
    // Otherwise no
    bool has_non_cumulative_op = false;
    bool has_cumulative_op = false;
    int index_i = int(input_has_index);
    for (int i = 0; i < func_offsets[in_table->ncols() - num_keys - index_i];
         i++) {
        int ftype = ftypes[i];
        if (ftype == Bodo_FTypes::cumsum || ftype == Bodo_FTypes::cummin ||
            ftype == Bodo_FTypes::cumprod || ftype == Bodo_FTypes::cummax) {
            has_cumulative_op = true;
        } else {
            has_non_cumulative_op = true;
        }
    }
    if (has_non_cumulative_op)
        return 0;  // No choice, we have to use the classic hash scheme
    if (!has_cumulative_op)
        return 0;  // It does not make sense to use MPI_exscan here.
    // Second decision: Whether it is arithmetic or not. If arithmetic, we can
    // use MPI_Exscan. If not, we may make it work for cumsum of strings or list
    // of strings but that would be definitely quite complicated and use more
    // than just MPI_Exscan.
    bool has_non_arithmetic_type = false;
    for (int64_t i = num_keys; i < in_table->ncols() - index_i; i++) {
        array_info* oper_col = in_table->columns[i];
        if (oper_col->arr_type != bodo_array_type::NUMPY &&
            oper_col->arr_type != bodo_array_type::NULLABLE_INT_BOOL)
            has_non_arithmetic_type = true;
    }
    if (has_non_arithmetic_type)
        return 0;  // No choice, we have to use the classic hash scheme
    // Third decision: Whether we use categorical with just one key. Working
    // with other keys would require some preprocessing.
    if (num_keys > 1)
        return 2;  // For more than 1 key column, use multikey mpi_exscan
    bodo_array_type::arr_type_enum key_arr_type =
        in_table->columns[0]->arr_type;
    if (key_arr_type != bodo_array_type::CATEGORICAL)
        return 2;  // For key column that are not categorical, use multikey
                   // mpi_exscan
    if (in_table->columns[0]->num_categories > max_global_number_groups_exscan)
        return 0;  // For too many categories the hash partition will be better
    return 1;      // all conditions satisfied. Let's go for EXSCAN code
}

table_info* groupby_and_aggregate(table_info* in_table, int64_t num_keys,
                                  bool input_has_index, int* ftypes,
                                  int* func_offsets, int* udf_nredvars,
                                  bool is_parallel, bool skipdropna,
                                  bool return_key, bool return_index,
                                  void* update_cb, void* combine_cb,
                                  void* eval_cb, table_info* udf_dummy_table) {
#ifdef DEBUG_GROUPBY_SYMBOL
    std::cout << "IN_TABLE (groupby):\n";
#ifdef DEBUG_GROUPBY_FULL
    DEBUG_PrintSetOfColumn(std::cout, in_table->columns);
#endif
    DEBUG_PrintRefct(std::cout, in_table->columns);
    std::cout << "num_keys=" << num_keys << " is_parallel=" << is_parallel
              << " input_has_index=" << input_has_index << "\n";
#endif

    int strategy = determine_groupby_strategy(
        in_table, num_keys, ftypes, func_offsets, is_parallel, input_has_index);
#ifdef DEBUG_GROUPBY
    std::cout << "groupby : strategy = " << strategy << "\n";
#endif

    auto implement_strategy0 = [&]() -> table_info* {
        GroupbyPipeline groupby(
            in_table, num_keys, input_has_index, is_parallel, ftypes,
            func_offsets, udf_nredvars, udf_dummy_table,
            (udf_table_op_fn)update_cb, (udf_table_op_fn)combine_cb,
            (udf_eval_fn)eval_cb, skipdropna, return_key, return_index);

        table_info* ret_table = groupby.run();
#ifdef DEBUG_GROUPBY_SYMBOL
        std::cout << "RET_TABLE (groupby, classic code path):\n";
#ifdef DEBUG_GROUPBY_FULL
        DEBUG_PrintSetOfColumn(std::cout, ret_table->columns);
#endif
        DEBUG_PrintRefct(std::cout, ret_table->columns);
#endif
        return ret_table;
    };
    auto implement_categorical_exscan =
        [&](array_info* cat_column) -> table_info* {
        table_info* ret_table = mpi_exscan_computation(
            cat_column, in_table, num_keys, ftypes, func_offsets, is_parallel,
            skipdropna, return_key, return_index);
#ifdef DEBUG_GROUPBY_SYMBOL
        std::cout << "RET_TABLE (groupby, categorical_exscan code path):\n";
#ifdef DEBUG_GROUPBY_FULL
        DEBUG_PrintSetOfColumn(std::cout, ret_table->columns);
#endif
        DEBUG_PrintRefct(std::cout, ret_table->columns);
#endif
#ifdef DEBUG_GROUPBY_SYMBOL
        std::cout
            << "IN_TABLE (groupby implement_categorical_exscan on exit):\n";
        DEBUG_PrintRefct(std::cout, in_table->columns);
#endif
        return ret_table;
    };
    if (strategy == 0) return implement_strategy0();
    if (strategy == 1) {
        array_info* cat_column = in_table->columns[0];
        return implement_categorical_exscan(cat_column);
    }
    if (strategy == 2) {
        array_info* cat_column =
            compute_categorical_index(in_table, num_keys, is_parallel);
        if (cat_column == nullptr) {  // It turns out that there are too many
                                      // different keys for exscan to be ok.
            return implement_strategy0();
        } else {
#ifdef DEBUG_GROUPBY
            std::cout << "Before the implement_categorical_exscan\n";
            std::cout << "num_categories=" << cat_column->num_categories
                      << "\n";
#endif
            table_info* ret_table = implement_categorical_exscan(cat_column);
            delete_info_decref_array(cat_column);
            return ret_table;
        }
    }
    return nullptr;
}
