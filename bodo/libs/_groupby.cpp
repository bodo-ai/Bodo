// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include "_groupby.h"
#include <functional>
#include <limits>
#include <map>
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
        no_op = 0,  // To make sure ftypes[0] isn't accidently matched with any
                    // of the supported functions.
        head,
        transform,
        size,
        shift,
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
        gen_udf,
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

    // this mapping is used by BasicColSet operations to know what combine (i.e.
    // step (c)) function to use for a given aggregation function
    combine_funcs[Bodo_FTypes::size] = Bodo_FTypes::sum;
    combine_funcs[Bodo_FTypes::sum] = Bodo_FTypes::sum;
    combine_funcs[Bodo_FTypes::count] = Bodo_FTypes::sum;
    combine_funcs[Bodo_FTypes::mean] =
        Bodo_FTypes::sum;  // sum totals and counts
    combine_funcs[Bodo_FTypes::min] = Bodo_FTypes::min;
    combine_funcs[Bodo_FTypes::max] = Bodo_FTypes::max;
    combine_funcs[Bodo_FTypes::prod] = Bodo_FTypes::prod;
    combine_funcs[Bodo_FTypes::first] = Bodo_FTypes::first;
    combine_funcs[Bodo_FTypes::last] = Bodo_FTypes::last;
    combine_funcs[Bodo_FTypes::nunique] =
        Bodo_FTypes::sum;  // used in nunique_mode = 2
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

/**
 * Function pointer for general UDFs executed in JIT-compiled code (see
 * also udfinfo_t).
 *
 * @param num_groups Number of groups in input data
 * @param in_table Input table only for columns with general UDFs. This is in
 *        *non-conventional* format. Given n groups, for each input column
 *        of groupby, this table contains n columns (containing the input
 *        data for group 0,1,...,n-1).
 * @param out_table Groupby output table. Has columns for *all* output,
 *        including for columns with no general UDFs.
 */
typedef void (*udf_general_fn)(int64_t num_groups, table_info* in_table,
                               table_info* out_table);
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

    /*
     * Function pointer to general UDF code (takes input data -by groups-
     * for all input columns with general UDF and fills in the corresponding
     * columns in the output table).
     */
    udf_general_fn general_udf;
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

// isnan makes sense only for floating point.
template <typename T, int dtype>
struct count_agg<
    T, dtype, typename std::enable_if<std::is_floating_point<T>::value>::type> {
    static void apply(int64_t& v1, T& v2) {
        if (!isnan(v2)) v1 += 1;
    }
};

template <typename T, int dtype, typename Enable = void>
struct size_agg {
    /**
     * Aggregation function for size. Increases size
     *
     * @param[in,out] current count
     * @param second input value.
     */
    static void apply(int64_t& v1, T& v2) { v1 += 1; }
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
     * @param[in] v2: observed value
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
    table_info* dispatch_table = nullptr;
    table_info* dispatch_info = nullptr;
    size_t num_groups;
    size_t n_pivot;
    int mode;  // 1: for the update, 2: for the combine
};

/*
  The construction of the array_info from standard vectors.
  It covers array_info and multiple_array_info.
 */
array_info* create_string_array_iter(
    std::vector<uint8_t> const& V,
    std::vector<std::string>::const_iterator& iter, size_t const& len,
    size_t start_idx) {
    size_t nb_char = 0;
    std::vector<std::string>::const_iterator iter_b = iter;
    for (size_t i_grp = 0; i_grp < len; i_grp++) {
        if (GetBit(V.data(), i_grp)) {
            nb_char += iter_b->size();
        }
        iter_b++;
    }
    size_t extra_bytes = 0;
    array_info* out_col = alloc_string_array(len, nb_char, extra_bytes);
    // update string array payload to reflect change
    char* data_o = out_col->data1;
    offset_t* offsets_o = (offset_t*)out_col->data2;
    offset_t pos = 0;
    iter_b = iter;
    for (size_t i_grp = 0; i_grp < len; i_grp++) {
        offsets_o[i_grp] = pos;
        bool bit = GetBit(V.data(), start_idx + i_grp);
        if (bit) {
            size_t len_str = size_t(iter_b->size());
            memcpy(data_o, iter_b->data(), len_str);
            data_o += len_str;
            pos += len_str;
        }
        out_col->set_null_bit(i_grp, bit);
        iter_b++;
    }
    offsets_o[len] = pos;
    return out_col;
}

template <typename ARRAY>
typename std::enable_if<!is_multiple_array<ARRAY>::value, ARRAY*>::type
create_string_array(grouping_info const& grp_info,
                    std::vector<uint8_t> const& V,
                    std::vector<std::string> const& ListString) {
    std::vector<std::string>::const_iterator iter = ListString.begin();
    size_t start_idx = 0;
    return create_string_array_iter(V, iter, ListString.size(), start_idx);
}

template <typename ARRAY>
typename std::enable_if<is_multiple_array<ARRAY>::value, ARRAY*>::type
create_string_array(grouping_info const& grp_info,
                    std::vector<uint8_t> const& V,
                    std::vector<std::string> const& ListString) {
    size_t num_groups = grp_info.num_groups;
    size_t len_loc = grp_info.group_to_first_row.size();
    size_t n_block = num_groups / len_loc;
    std::vector<array_info*> vect_arr(n_block);
    std::vector<std::string>::const_iterator iter = ListString.begin();
    for (size_t i_block = 0; i_block < n_block; i_block++) {
        size_t start_idx = i_block * len_loc;
        vect_arr[i_block] =
            create_string_array_iter(V, iter, len_loc, start_idx);
        iter += len_loc;
    }
    return new multiple_array_info(vect_arr);
}

array_info* create_list_string_array_iter(
    std::vector<uint8_t> const& V,
    std::vector<std::vector<pair_str_bool>>::const_iterator const& iter,
    size_t len, size_t start_idx) {
    // Determining the number of characters in output.
    size_t nb_string = 0;
    size_t nb_char = 0;
    std::vector<std::vector<pair_str_bool>>::const_iterator iter_b = iter;
    for (size_t i_grp = 0; i_grp < len; i_grp++) {
        if (GetBit(V.data(), i_grp)) {
            std::vector<pair_str_bool> e_list = *iter_b;
            nb_string += e_list.size();
            for (auto& e_str : e_list) nb_char += e_str.first.size();
        }
        iter_b++;
    }
    // Allocation needs to be done through
    // alloc_list_string_array, which allocates with meminfos
    // and same data structs that Python uses. We need to
    // re-allocate here because number of strings and chars has
    // been determined here (previous out_col was just an empty
    // dummy allocation).

    array_info* new_out_col =
        alloc_list_string_array(len, nb_string, nb_char, 0);
    offset_t* index_offsets_o = (offset_t*)new_out_col->data3;
    offset_t* data_offsets_o = (offset_t*)new_out_col->data2;
    uint8_t* sub_null_bitmask_o = (uint8_t*)new_out_col->sub_null_bitmask;
    // Writing the list_strings in output
    char* data_o = new_out_col->data1;
    data_offsets_o[0] = 0;
    offset_t pos_index = 0;
    offset_t pos_data = 0;
    iter_b = iter;
    for (size_t i_grp = 0; i_grp < len; i_grp++) {
        bool bit = GetBit(V.data(), i_grp);
        new_out_col->set_null_bit(i_grp, bit);
        index_offsets_o[i_grp] = pos_index;
        if (bit) {
            std::vector<pair_str_bool> e_list = *iter_b;
            offset_t n_string = e_list.size();
            for (offset_t i_str = 0; i_str < n_string; i_str++) {
                std::string& estr = e_list[i_str].first;
                offset_t n_char = estr.size();
                memcpy(data_o, estr.data(), n_char);
                data_o += n_char;
                pos_data++;
                data_offsets_o[pos_data] =
                    data_offsets_o[pos_data - 1] + n_char;
                bool bit = e_list[i_str].second;
                SetBitTo(sub_null_bitmask_o, pos_index + i_str, bit);
            }
            pos_index += n_string;
        }
        iter_b++;
    }
    index_offsets_o[len] = pos_index;
    return new_out_col;
}

template <typename ARRAY>
typename std::enable_if<!is_multiple_array<ARRAY>::value, ARRAY*>::type
create_list_string_array(
    grouping_info const& grp_info, std::vector<uint8_t> const& V,
    std::vector<std::vector<pair_str_bool>> const& ListListPair) {
    std::vector<std::vector<pair_str_bool>>::const_iterator iter =
        ListListPair.begin();
    size_t start_idx = 0;
    return create_list_string_array_iter(V, iter, ListListPair.size(),
                                         start_idx);
}

template <typename ARRAY>
typename std::enable_if<is_multiple_array<ARRAY>::value, ARRAY*>::type
create_list_string_array(
    grouping_info const& grp_info, std::vector<uint8_t> const& V,
    std::vector<std::vector<pair_str_bool>> const& ListListPair) {
    size_t num_groups = grp_info.num_groups;
    size_t len_loc = grp_info.group_to_first_row.size();
    size_t n_block = num_groups / len_loc;
    std::vector<array_info*> vect_arr(n_block);
    std::vector<std::vector<pair_str_bool>>::const_iterator iter =
        ListListPair.begin();
    for (size_t i_block = 0; i_block < n_block; i_block++) {
        size_t start_idx = i_block * len_loc;
        vect_arr[i_block] =
            create_list_string_array_iter(V, iter, len_loc, start_idx);
        iter += len_loc;
    }
    return new multiple_array_info(vect_arr);
}

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
template <typename ARRAY, typename F>
void var_combine_F(ARRAY* count_col_in, ARRAY* mean_col_in, ARRAY* m2_col_in,
                   ARRAY* count_col_out, ARRAY* mean_col_out, ARRAY* m2_col_out,
                   F f) {
    for (int64_t i = 0; i < count_col_in->length; i++) {
        int64_t j = f(i);
        uint64_t& count_a = getv<ARRAY, uint64_t>(count_col_out, j);
        uint64_t& count_b = getv<ARRAY, uint64_t>(count_col_in, i);
        // in the pivot case we can receive dummy values from other ranks
        // when combining the results (in this case with count == 0). This is
        // because the pivot case groups on index and creates n_pivot columns,
        // and so for each of its index values a rank will have n_pivot fields,
        // even if a rank does not have a particular (index, pivot_value) pair
        if (count_b == 0) continue;
        double& mean_a = getv<ARRAY, double>(mean_col_out, j);
        double& mean_b = getv<ARRAY, double>(mean_col_in, i);
        double& m2_a = getv<ARRAY, double>(m2_col_out, j);
        double& m2_b = getv<ARRAY, double>(m2_col_in, i);
        uint64_t count = count_a + count_b;
        double delta = mean_b - mean_a;
        mean_a = (count_a * mean_a + count_b * mean_b) / count;
        m2_a = m2_a + m2_b + delta * delta * count_a * count_b / count;
        count_a = count;
    }
}

template <typename ARRAY>
typename std::enable_if<!is_multiple_array<ARRAY>::value, void>::type
var_combine(ARRAY* count_col_in, ARRAY* mean_col_in, ARRAY* m2_col_in,
            ARRAY* count_col_out, ARRAY* mean_col_out, ARRAY* m2_col_out,
            grouping_info const& grp_info) {
    auto f = [&](int64_t const& i_row) -> int64_t {
        return grp_info.row_to_group[i_row];
    };
    return var_combine_F<ARRAY, decltype(f)>(count_col_in, mean_col_in,
                                             m2_col_in, count_col_out,
                                             mean_col_out, m2_col_out, f);
}

template <typename ARRAY>
typename std::enable_if<is_multiple_array<ARRAY>::value, void>::type
var_combine(ARRAY* count_col_in, ARRAY* mean_col_in, ARRAY* m2_col_in,
            ARRAY* count_col_out, ARRAY* mean_col_out, ARRAY* m2_col_out,
            grouping_info const& grp_info) {
    table_info* dispatch_table = grp_info.dispatch_table;
    table_info* dispatch_info = grp_info.dispatch_info;
    int n_cols = dispatch_table->ncols();
    int n_dis = dispatch_table->nrows();
    size_t n_pivot = grp_info.n_pivot;
    bool na_position_bis = true;
    auto KeyComparisonAsPython_Table = [&](int i_info, int i_dis) -> bool {
        for (int64_t i_col = 0; i_col < n_cols; i_col++) {
            int test = KeyComparisonAsPython_Column(
                na_position_bis, dispatch_table->columns[i_col], i_dis,
                dispatch_info->columns[i_col], i_info);
            if (test != 0) return false;
        }
        return true;
    };
    auto get_i_dis = [&](int64_t const& i_info) -> int64_t {
        for (int64_t i_dis = 0; i_dis < n_dis; i_dis++) {
            bool test = KeyComparisonAsPython_Table(i_info, i_dis);
            if (test) return i_dis;
        }
        return -1;
    };
#ifdef DEBUG_GROUPBY
    std::cout << "grp_info.mode=" << grp_info.mode << "\n";
#endif
    auto f = [&](int64_t const& i_row) -> int64_t {
        if (grp_info.mode == 1) {
            int64_t i_grp = grp_info.row_to_group[i_row];
            int i_dis = get_i_dis(i_row);
            if (i_dis == -1) return -1;
            return i_dis + n_pivot * i_grp;
        } else {
            int i_dis = i_row % n_pivot;
            int i_row_red = i_row / n_pivot;
            int64_t i_grp_red = grp_info.row_to_group[i_row_red];
            if (i_grp_red == -1) return -1;
            int64_t i_grp = i_dis + n_pivot * i_grp_red;
            return i_grp;
        }
    };
    return var_combine_F<ARRAY, decltype(f)>(count_col_in, mean_col_in,
                                             m2_col_in, count_col_out,
                                             mean_col_out, m2_col_out, f);
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

// TODO which load factor to use?
#define ROBIN_MAP_MAX_LOAD_FACTOR 0.5
#define HOPSCOTCH_MAP_MAX_LOAD_FACTOR 0.9

bool calc_use_robin_map(size_t nunique_hashes, tracing::Event* ev = nullptr) {
    // If the number of groups is very large, we should avoid very small load
    // factors because the map will use too much memory, and also allocation
    // and deallocation will become very expensive
    bool use_robin_map = true;
    if (nunique_hashes > 1000000) {  // TODO which threshold is best?
        double robin_map_load_factor = -1;
        double hopscotch_map_load_factor = -1;
        int64_t bucket_size = 1;
        while (true) {
            if (bucket_size > nunique_hashes) {
                // robin_map works best with load_factor <= 0.5, otherwise
                // hopscotch is better:
                // https://tessil.github.io/2016/08/29/benchmark-hopscotch-map.html
                double load_factor = nunique_hashes / double(bucket_size);
                if (load_factor <= ROBIN_MAP_MAX_LOAD_FACTOR &&
                    robin_map_load_factor < 0)
                    robin_map_load_factor = load_factor;
                if (load_factor < HOPSCOTCH_MAP_MAX_LOAD_FACTOR &&
                    hopscotch_map_load_factor < 0)
                    hopscotch_map_load_factor = load_factor;
                if (robin_map_load_factor > 0 &&
                    hopscotch_map_load_factor > 0) {
                    if (robin_map_load_factor >
                        0.35)  // TODO which threshold is best?
                        use_robin_map = true;
                    else if (hopscotch_map_load_factor >
                             robin_map_load_factor + 0.1)
                        use_robin_map = false;
                    break;
                }
            }
            // tsl::robin_map and tsl::hopscotch_map uses a power of 2 growth
            // policy by default (for faster hash to bucket calculation)
            bucket_size = bucket_size << 1;
        }
        if (ev) {
            ev->add_attribute("robin_map_load_factor", robin_map_load_factor);
            ev->add_attribute("hopscotch_map_load_factor",
                              hopscotch_map_load_factor);
        }
    }
    return use_robin_map;
}

/**
 * The main get_group_info loop which populates a grouping_info structure
 * (map rows from input to their group number, and store the first input row
 * for each group).
 *
 * @param[in,out] key_to_group: The hash map used to populate the grouping_info
 * structure, maps row index from input data to group numbers (the group numbers
 * in the map are the real group number + 1)
 * @param[in] key_cols: key columns
 * @param[in,out] grp_info: The grouping_info structure that we are populating
 * @param key_drop_nulls : whether to drop null keys
 * @param nrows : number of input rows
 * @param is_parallel: true if data is distributed
 */
template <typename T>
static void get_group_info_loop(T& key_to_group,
                                std::vector<array_info*>& key_cols,
                                grouping_info& grp_info,
                                const bool key_drop_nulls, const int64_t nrows,
                                bool is_parallel) {
    tracing::Event ev("get_group_info_loop", is_parallel);
    // Start at 1 because I'm going to use 0 to mean nothing was inserted yet
    // in the map (but note that the group values recorded in grp_info go from
    // 0 to num_groups - 1)
    int64_t next_group = 1;
    // There are two versions of the loop because a `if (key_drop_nulls)`
    // branch inside the loop has a bit of a performance hit and
    // `get_group_info_loop` is one of the most expensive computations.
    // To not duplicate code, we put the common portion of the loop in
    // MAIN_LOOP_BODY macro

#define MAIN_LOOP_BODY                                                      \
    int64_t& group = key_to_group[i]; /* this inserts 0 into the map if key \
                                         doesn't exist */                   \
    if (group == 0) {                                                       \
        group = next_group++; /* this updates the value in the map without  \
                                 another lookup */                          \
        grp_info.group_to_first_row.emplace_back(i);                        \
    }                                                                       \
    grp_info.row_to_group[i] = group - 1

    if (!key_drop_nulls) {
        for (int64_t i = 0; i < nrows; i++) {
            MAIN_LOOP_BODY;
        }
    } else {
        for (int64_t i = 0; i < nrows; i++) {
            if (does_row_has_nulls(key_cols, i)) {
                grp_info.row_to_group[i] = -1;
                continue;
            }
            MAIN_LOOP_BODY;
        }
    }
}

namespace {
/**
 * Look up a hash in a table.
 *
 * Don't use std::function to reduce call overhead.
 */
struct HashLookupIn32bitTable {
    uint32_t operator()(const int64_t iRow) const { return hashes[iRow]; }
    uint32_t* hashes;
};

/**
 * Check if keys are equal by lookup in a table.
 *
 * Don't use std::function to reduce call overhead.
 */
struct KeyEqualLookupIn32bitTable {
    bool operator()(const int64_t iRowA, const int64_t iRowB) const {
        return TestEqualJoin(table, table, iRowA, iRowB, n_keys, true);
    }
    int64_t n_keys;
    table_info* table;
};

// both get_group_info and get_groupby_labes use the dealloc measurement, so
// share that code.
template <bool CalledFromGroupInfo, typename Map>
void do_map_dealloc(uint32_t*& hashes, Map& key_to_group, bool is_parallel) {
    tracing::Event ev_dealloc(CalledFromGroupInfo
                                  ? "get_group_info_dealloc"
                                  : "get_groupby_labels_dealloc",
                              is_parallel);
    delete[] hashes;
    hashes = nullptr;  // updates hashes ptr at caller

    if (ev_dealloc.is_tracing()) {
        ev_dealloc.add_attribute("map size", key_to_group.size());
        ev_dealloc.add_attribute("map bucket_count",
                                 key_to_group.bucket_count());
        ev_dealloc.add_attribute("map load_factor", key_to_group.load_factor());
        ev_dealloc.add_attribute("map max_load_factor",
                                 key_to_group.max_load_factor());
    }
    key_to_group.clear();
    key_to_group.reserve(0);  // try to force dealloc of hash map
    ev_dealloc.finalize();
}

template <typename Map>
void get_group_info_impl(Map& key_to_group, tracing::Event& ev,
                         grouping_info& grp_info, table_info* const table,
                         std::vector<array_info*>& key_cols, uint32_t*& hashes,
                         const size_t nunique_hashes,
                         const bool check_for_null_keys, const bool key_dropna,
                         const double load_factor, bool is_parallel) {
    tracing::Event ev_alloc("get_group_info_alloc", is_parallel);
    key_to_group.max_load_factor(load_factor);
    key_to_group.reserve(nunique_hashes);
    ev_alloc.add_attribute("map bucket_count", key_to_group.bucket_count());
    ev_alloc.add_attribute("map max_load_factor",
                           key_to_group.max_load_factor());

    // XXX Don't want to initialize data (just reserve and set size) but
    // std::vector doesn't support this. Since we know the size of the
    // vector, doing this is better than reserving and then doing
    // emplace_backs in get_group_info_loop, which are slower than []
    // operator
    grp_info.row_to_group.resize(table->nrows());
    grp_info.group_to_first_row.reserve(nunique_hashes * 1.1);
    ev_alloc.finalize();

    const bool key_is_nullable =
        check_for_null_keys ? does_keys_have_nulls(key_cols) : false;
    const bool key_drop_nulls = key_is_nullable && key_dropna;
    ev.add_attribute("g_key_is_nullable", key_is_nullable);
    ev.add_attribute("g_key_dropna", key_dropna);
    ev.add_attribute("g_key_drop_nulls", key_drop_nulls);

    get_group_info_loop<Map>(key_to_group, key_cols, grp_info, key_drop_nulls,
                             table->nrows(), is_parallel);
    grp_info.num_groups = grp_info.group_to_first_row.size();
    ev.add_attribute("num_groups", size_t(grp_info.num_groups));
    do_map_dealloc<true>(hashes, key_to_group, is_parallel);
}
}  // namespace

/**
 * Given a set of tables with n key columns, this function calculates the row to
 * group mapping for every row based on its key. For every row in the tables,
 * this only does *one* lookup in the hash map.
 *
 * @param[in] tables the tables
 * @param[in] hashes hashes if they have already been calculated. nullptr
 * otherwise
 * @param[in] nunique_hashes estimated number of unique hashes if hashes are
 * provided
 * @param[out] grp_infos is grouping_info structures that map row numbers to
 * group numbers
 * @param[in] check_for_null_keys whether to check for null keys. If a key is
 * null and key_dropna=True that row will not be mapped to any group
 * @param[in] key_dropna whether to allow NA values in group keys or not.
 * @param[in] is_parallel: true if data is distributed
 */
void get_group_info(std::vector<table_info*>& tables, uint32_t*& hashes,
                    size_t nunique_hashes,
                    std::vector<grouping_info>& grp_infos,
                    bool check_for_null_keys, bool key_dropna,
                    bool is_parallel) {
    tracing::Event ev("get_group_info", is_parallel);
    if (tables.size() == 0) {
        throw std::runtime_error("get_group_info: tables is empty");
    }
    table_info* table = tables[0];
    ev.add_attribute("input_table_nrows", table->nrows());
    std::vector<array_info*> key_cols = std::vector<array_info*>(
        table->columns.begin(), table->columns.begin() + table->num_keys);
    if (hashes == nullptr) {
        hashes = hash_keys(key_cols, SEED_HASH_GROUPBY_SHUFFLE, is_parallel);
        nunique_hashes =
            get_nunique_hashes(hashes, table->nrows(), is_parallel);
    }
    ev.add_attribute("nunique_hashes_est", nunique_hashes);
    grp_infos.emplace_back();
    grouping_info& grp_info = grp_infos.back();
    const int64_t n_keys = table->num_keys;

    HashLookupIn32bitTable hash_fct{hashes};
    KeyEqualLookupIn32bitTable equal_fct{n_keys, table};

    using rh_flat_t =
        robin_hood::unordered_flat_map<int64_t, int64_t, HashLookupIn32bitTable,
                                       KeyEqualLookupIn32bitTable>;
    rh_flat_t key_to_group_rh_flat({}, hash_fct, equal_fct);
    get_group_info_impl(key_to_group_rh_flat, ev, grp_info, table, key_cols,
                        hashes, nunique_hashes, check_for_null_keys, key_dropna,
                        UNORDERED_MAP_MAX_LOAD_FACTOR, is_parallel);

    if (tables.size() > 1) {
        // This case is not currently used
        throw std::runtime_error(
            "get_group_info not implemented for multiple tables");
    }
}

template <typename T>
static int64_t get_groupby_labels_loop(T& key_to_group,
                                       std::vector<array_info*>& key_cols,
                                       int64_t* row_to_group, int64_t* sort_idx,
                                       const bool key_drop_nulls,
                                       const int64_t nrows, bool is_parallel) {
    tracing::Event ev("get_groupby_labels_loop", is_parallel);
    // Start at 1 because I'm going to use 0 to mean nothing was inserted yet
    // in the map (but note that the group values recorded in grp_info go from
    // 0 to num_groups - 1)
    int64_t next_group = 1;
    // There are two versions of the loop because a `if (key_drop_nulls)`
    // branch inside the loop has a bit of a performance hit and
    // this loop is one of the most expensive computations.
    // To not duplicate code, we put the common portion of the loop in
    // MAIN_LOOP_BODY macro

    std::vector<std::vector<int64_t>> group_rows;

#define MAIN_LOOP_BODY                                                      \
    int64_t& group = key_to_group[i]; /* this inserts 0 into the map if key \
                                         doesn't exist */                   \
    if (group == 0) {                                                       \
        group = next_group++; /* this updates the value in the map without  \
                                 another lookup */                          \
        group_rows.emplace_back();                                          \
    }                                                                       \
    group_rows[group - 1].push_back(i);                                     \
    row_to_group[i] = group - 1

    // keep track of how many NA values in the column
    int64_t na_pos = 0;
    if (!key_drop_nulls) {
        for (int64_t i = 0; i < nrows; i++) {
            MAIN_LOOP_BODY;
        }
    } else {
        for (int64_t i = 0; i < nrows; i++) {
            if (does_row_has_nulls(key_cols, i)) {
                row_to_group[i] = -1;
                // We need to keep position of all values in the sort_idx
                // regardless of being dropped or not Indexes of all NA values
                // are first in sort_idx since their group number is -1
                sort_idx[na_pos] = i;
                na_pos++;
                continue;
            }
            MAIN_LOOP_BODY;
        }
    }
    int64_t pos = 0 + na_pos;
    for (size_t i = 0; i < group_rows.size(); i++) {
        memcpy(sort_idx + pos, group_rows[i].data(),
               group_rows[i].size() * sizeof(int64_t));
        pos += group_rows[i].size();
    }
    return next_group - 1;
}

namespace {
template <typename Map>
int64_t get_groupby_labels_impl(Map& key_to_group, tracing::Event& ev,
                                int64_t* out_labels, int64_t* sort_idx,
                                table_info* const table,
                                std::vector<array_info*>& key_cols,
                                uint32_t*& hashes, const size_t nunique_hashes,
                                const bool check_for_null_keys,
                                const bool key_dropna, const double load_factor,
                                bool is_parallel) {
    tracing::Event ev_alloc("get_groupby_labels_alloc", is_parallel);
    key_to_group.max_load_factor(load_factor);
    key_to_group.reserve(nunique_hashes);
    ev_alloc.add_attribute("map bucket_count", key_to_group.bucket_count());
    ev_alloc.add_attribute("map max_load_factor",
                           key_to_group.max_load_factor());
    ev_alloc.finalize();

    const bool key_is_nullable =
        check_for_null_keys ? does_keys_have_nulls(key_cols) : false;
    const bool key_drop_nulls = key_is_nullable && key_dropna;
    ev.add_attribute("g_key_is_nullable", key_is_nullable);
    ev.add_attribute("g_key_dropna", key_dropna);
    ev.add_attribute("g_key_drop_nulls", key_drop_nulls);

    const int64_t num_groups = get_groupby_labels_loop<Map>(
        key_to_group, key_cols, out_labels, sort_idx, key_drop_nulls,
        table->nrows(), is_parallel);
    ev.add_attribute("num_groups", num_groups);
    do_map_dealloc<false>(hashes, key_to_group, is_parallel);
    return num_groups;
}
}  // namespace

/**
 * @brief Get groupby labels for input key arrays
 *
 * @param table a table of all key arrays
 * @param[out] out_labels output array to fill
 * @param[out] sort_idx sorted group indices
 * @param[in] key_dropna whether to allow NA values in group keys or not.
 * @param is_parallel: true if data is distributed
 * @return int64_t total number of groups
 */
int64_t get_groupby_labels(table_info* table, int64_t* out_labels,
                           int64_t* sort_idx, bool key_dropna,
                           bool is_parallel) {
    tracing::Event ev("get_groupby_labels", is_parallel);
    ev.add_attribute("input_table_nrows", table->nrows());
    // TODO(ehsan): refactor to avoid code duplication with get_group_info
    // This function is similar to get_group_info. See that function for
    // more comments
    table->num_keys = table->columns.size();
    std::vector<array_info*> key_cols = table->columns;
    uint32_t seed = SEED_HASH_GROUPBY_SHUFFLE;
    uint32_t* hashes = hash_keys(key_cols, seed, is_parallel);

    size_t nunique_hashes =
        get_nunique_hashes(hashes, table->nrows(), is_parallel);
    ev.add_attribute("nunique_hashes_est", nunique_hashes);
    const int64_t n_keys = table->num_keys;

    HashLookupIn32bitTable hash_fct{hashes};
    KeyEqualLookupIn32bitTable equal_fct{n_keys, table};

    const bool check_for_null_keys = true;
    using rh_flat_t =
        robin_hood::unordered_flat_map<int64_t, int64_t, HashLookupIn32bitTable,
                                       KeyEqualLookupIn32bitTable>;
    rh_flat_t key_to_group_rh_flat({}, hash_fct, equal_fct);
    return get_groupby_labels_impl(
        key_to_group_rh_flat, ev, out_labels, sort_idx, table, key_cols, hashes,
        nunique_hashes, check_for_null_keys, key_dropna,
        UNORDERED_MAP_MAX_LOAD_FACTOR, is_parallel);
}

/**
 * Given a set of tables with n key columns, this function calculates the row to
 * group mapping for every row based on its key. For every row in the tables,
 * this only does *one* lookup in the hash map.
 *
 * @param           tables: the tables
 * @param[in] hashes hashes for first table in tables, if they have already
 * been calculated. nullptr otherwise
 * @param[in] nunique_hashes estimated number of unique hashes if hashes are
 * provided (for first table)
 * @param[out]      grouping_info structures that map row numbers to group
 * numbers
 * @param[in] consider_missing: whether to return the list of missing rows or
 * not
 * @param[in] key_dropna whether to allow NA values in group keys or not.
 * @param[in] is_parallel: true if data is distributed
 */
void get_group_info_iterate(std::vector<table_info*>& tables, uint32_t*& hashes,
                            size_t nunique_hashes,
                            std::vector<grouping_info>& grp_infos,
                            const bool consider_missing, bool key_dropna,
                            bool is_parallel) {
    tracing::Event ev("get_group_info_iterate", is_parallel);
    if (tables.size() == 0) {
        throw std::runtime_error("get_group_info: tables is empty");
    }
    table_info* table = tables[0];
    std::vector<array_info*> key_cols = std::vector<array_info*>(
        table->columns.begin(), table->columns.begin() + table->num_keys);
    // TODO: if |tables| > 1 then we probably need to use hashes from all the
    // tables to get an accurate nunique_hashes estimate. We can do it, but
    // it would mean calculating all hashes in advance
    if (hashes == nullptr) {
        hashes = hash_keys(key_cols, SEED_HASH_GROUPBY_SHUFFLE, is_parallel);
        nunique_hashes =
            get_nunique_hashes(hashes, table->nrows(), is_parallel);
    }
    grp_infos.emplace_back();
    grouping_info& grp_info = grp_infos.back();

    int64_t max_rows = 0;
    for (table_info* table : tables)
        max_rows = std::max(max_rows, table->nrows());
    grp_info.row_to_group.reserve(max_rows);
    grp_info.row_to_group.resize(table->nrows());
    grp_info.next_row_in_group.reserve(max_rows);
    grp_info.next_row_in_group.resize(table->nrows(), -1);
    grp_info.group_to_first_row.reserve(nunique_hashes * 1.1);
    std::vector<int64_t> active_group_repr;
    active_group_repr.reserve(nunique_hashes * 1.1);

    // TODO Incorporate or adapt other optimizations from `get_group_info`

    bool key_is_nullable = does_keys_have_nulls(key_cols);
    bool key_drop_nulls = key_is_nullable && key_dropna;

    // Start at 1 because I'm going to use 0 to mean nothing was inserted yet
    // in the map (but note that the group values recorded in grp_info go from
    // 0 to num_groups - 1)
    int64_t next_group = 1;
    UNORD_MAP_CONTAINER<multi_col_key, int64_t, multi_col_key_hash>
        key_to_group;
    key_to_group.reserve(nunique_hashes);
    for (int64_t i = 0; i < table->nrows(); i++) {
        if (key_drop_nulls) {
            if (does_row_has_nulls(key_cols, i)) {
                grp_info.row_to_group[i] = -1;
                if (consider_missing) grp_info.list_missing.push_back(i);
                continue;
            }
        }
        multi_col_key key(hashes[i], table, i);
        int64_t& group = key_to_group[key];  // this inserts 0 into the map if
                                             // key doesn't exist
        if (group == 0) {
            group = next_group++;  // this updates the value in the map without
                                   // another lookup
            grp_info.group_to_first_row.emplace_back(i);
            active_group_repr.emplace_back(i);
        } else {
            int64_t prev_elt = active_group_repr[group - 1];
            grp_info.next_row_in_group[prev_elt] = i;
            active_group_repr[group - 1] = i;
        }
        grp_info.row_to_group[i] = group - 1;
    }
    delete[] hashes;
    hashes = nullptr;
    grp_info.num_groups = grp_info.group_to_first_row.size();

    for (size_t j = 1; j < tables.size(); j++) {
        int64_t num_groups = next_group - 1;
        // IMPORTANT: Assuming all the tables have the same number and type of
        // key columns (but not the same values in key columns)
        table = tables[j];
        key_cols = std::vector<array_info*>(
            table->columns.begin(), table->columns.begin() + table->num_keys);
        hashes = hash_keys(key_cols, SEED_HASH_GROUPBY_SHUFFLE, is_parallel);
        grp_infos.emplace_back();
        grouping_info& grp_info = grp_infos.back();
        grp_info.row_to_group.resize(table->nrows());
        grp_info.next_row_in_group.resize(table->nrows(), -1);
        grp_info.group_to_first_row.resize(num_groups, -1);
        active_group_repr.resize(num_groups);

        for (int64_t i = 0; i < table->nrows(); i++) {
            if (key_drop_nulls) {
                if (does_row_has_nulls(key_cols, i)) {
                    grp_info.row_to_group[i] = -1;
                    if (consider_missing) grp_info.list_missing.push_back(i);
                    continue;
                }
            }
            multi_col_key key(hashes[i], table, i);
            int64_t& group = key_to_group[key];  // this inserts 0 into the map
                                                 // if key doesn't exist
            if ((group == 0) ||
                (grp_info.group_to_first_row[group - 1] == -1)) {
                if (group == 0) {
                    group = next_group++;  // this updates the value in the map
                                           // without another lookup
                    grp_info.group_to_first_row.emplace_back(i);
                    active_group_repr.emplace_back(i);
                } else {
                    grp_info.group_to_first_row[group - 1] = i;
                    active_group_repr[group - 1] = i;
                }
            } else {
                int64_t prev_elt = active_group_repr[group - 1];
                grp_info.next_row_in_group[prev_elt] = i;
                active_group_repr[group - 1] = i;
            }
            grp_info.row_to_group[i] = group - 1;
        }
        delete[] hashes;
        hashes = nullptr;
        grp_info.num_groups = grp_info.group_to_first_row.size();
    }

    // set same num_groups in every group_info
    int64_t num_groups = next_group - 1;
    for (auto& grp_info : grp_infos) {
        grp_info.group_to_first_row.resize(num_groups, -1);
        grp_info.num_groups = num_groups;
    }
    ev.add_attribute("num_groups", size_t(num_groups));
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
                              grouping_info const& grp_info,
                              int32_t const& ftype, bool const& skipna) {
    size_t num_group = grp_info.group_to_first_row.size();
    if (arr->arr_type == bodo_array_type::STRING ||
        arr->arr_type == bodo_array_type::LIST_STRING) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "There is no cumulative operation for the string "
                             "or list string case");
        return;
    }
    auto cum_computation = [&](auto const& get_entry,
                               auto const& set_entry) -> void {
        for (size_t igrp = 0; igrp < num_group; igrp++) {
            int64_t i = grp_info.group_to_first_row[igrp];
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
                i = grp_info.next_row_in_group[i];
                if (i == -1) break;
            }
        }
        T eVal_nan = GetTentry<T>(
            RetrieveNaNentry((Bodo_CTypes::CTypeEnum)dtype).data());
        std::pair<bool, T> pairNaN{true, eVal_nan};
        for (auto& idx_miss : grp_info.list_missing)
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
        cum_computation(
            [=](int64_t pos) -> std::pair<bool, T> {
                return {!arr->get_null_bit(pos), arr->at<T>(pos)};
            },
            [=](int64_t pos, std::pair<bool, T> const& ePair) -> void {
                out_arr->set_null_bit(pos, !ePair.first);
                out_arr->at<T>(pos) = ePair.second;
            });
    }
}

void cumulative_computation_list_string(array_info* arr, array_info* out_arr,
                                        grouping_info const& grp_info,
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
    using T = std::pair<bool, std::vector<pair_str_bool>>;
    std::vector<T> V(n);
    uint8_t* null_bitmask = (uint8_t*)arr->null_bitmask;
    uint8_t* sub_null_bitmask = (uint8_t*)arr->sub_null_bitmask;
    char* data = arr->data1;
    offset_t* data_offsets = (offset_t*)arr->data2;
    offset_t* index_offsets = (offset_t*)arr->data3;
    auto get_entry = [&](int64_t i) -> T {
        bool isna = !GetBit(null_bitmask, i);
        offset_t start_idx_offset = index_offsets[i];
        offset_t end_idx_offset = index_offsets[i + 1];
        std::vector<pair_str_bool> LEnt;
        for (offset_t idx = start_idx_offset; idx < end_idx_offset; idx++) {
            offset_t str_len = data_offsets[idx + 1] - data_offsets[idx];
            offset_t start_data_offset = data_offsets[idx];
            bool bit = GetBit(sub_null_bitmask, idx);
            std::string val(&data[start_data_offset], str_len);
            pair_str_bool eEnt = {val, bit};
            LEnt.push_back(eEnt);
        }
        return {isna, LEnt};
    };
    size_t num_group = grp_info.group_to_first_row.size();
    for (size_t igrp = 0; igrp < num_group; igrp++) {
        int64_t i = grp_info.group_to_first_row[igrp];
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
            i = grp_info.next_row_in_group[i];
            if (i == -1) break;
        }
    }
    T pairNaN{true, {}};
    for (auto& idx_miss : grp_info.list_missing) V[idx_miss] = pairNaN;
    //
    size_t n_bytes = (n + 7) >> 3;
    std::vector<uint8_t> Vmask(n_bytes, 0);
    std::vector<std::vector<pair_str_bool>> ListListPair(n);
    for (int i = 0; i < n; i++) {
        SetBitTo(Vmask.data(), i, !V[i].first);
        ListListPair[i] = V[i].second;
    }
    array_info* new_out_col =
        create_list_string_array<array_info>(grp_info, Vmask, ListListPair);
#ifdef DEBUG_GROUPBY
    std::cout << "new_out_col : ";
    DEBUG_PrintColumn(std::cout, new_out_col);
#endif
    *out_arr = std::move(*new_out_col);
    delete new_out_col;
}

void cumulative_computation_string(array_info* arr, array_info* out_arr,
                                   grouping_info const& grp_info,
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
    offset_t* offsets = (offset_t*)arr->data2;
    auto get_entry = [&](int64_t i) -> T {
        bool isna = !GetBit(null_bitmask, i);
        offset_t start_offset = offsets[i];
        offset_t end_offset = offsets[i + 1];
        offset_t len = end_offset - start_offset;
        std::string val(&data[start_offset], len);
        return {isna, val};
    };
    size_t num_group = grp_info.group_to_first_row.size();
    for (size_t igrp = 0; igrp < num_group; igrp++) {
        int64_t i = grp_info.group_to_first_row[igrp];
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
            i = grp_info.next_row_in_group[i];
            if (i == -1) break;
        }
    }
    T pairNaN{true, ""};
    for (auto& idx_miss : grp_info.list_missing) V[idx_miss] = pairNaN;
    // Now writing down in the array.
    size_t n_bytes = (n + 7) >> 3;
    std::vector<uint8_t> Vmask(n_bytes, 0);
    std::vector<std::string> ListString(n);
    for (int64_t i = 0; i < n; i++) {
        SetBitTo(Vmask.data(), i, !V[i].first);
        ListString[i] = V[i].second;
    }
    array_info* new_out_col =
        create_string_array<array_info>(grp_info, Vmask, ListString);
    *out_arr = std::move(*new_out_col);
#ifdef DEBUG_GROUPBY
    std::cout << "out_arr : ";
    DEBUG_PrintColumn(std::cout, out_arr);
    std::cout << "End of cumulative_computation_string\n";
#endif
    delete new_out_col;
}

template <typename ARRAY>
typename std::enable_if<!is_multiple_array<ARRAY>::value, void>::type
cumulative_computation(array_info* arr, array_info* out_arr,
                       grouping_info const& grp_info, int32_t const& ftype,
                       bool const& skipna) {
    Bodo_CTypes::CTypeEnum dtype = arr->dtype;
    if (arr->arr_type == bodo_array_type::STRING)
        return cumulative_computation_string(arr, out_arr, grp_info, ftype,
                                             skipna);
    if (arr->arr_type == bodo_array_type::LIST_STRING)
        return cumulative_computation_list_string(arr, out_arr, grp_info, ftype,
                                                  skipna);
    if (dtype == Bodo_CTypes::INT8)
        return cumulative_computation_T<int8_t, Bodo_CTypes::INT8>(
            arr, out_arr, grp_info, ftype, skipna);
    if (dtype == Bodo_CTypes::UINT8)
        return cumulative_computation_T<uint8_t, Bodo_CTypes::UINT8>(
            arr, out_arr, grp_info, ftype, skipna);
    if (dtype == Bodo_CTypes::INT16)
        return cumulative_computation_T<int16_t, Bodo_CTypes::INT16>(
            arr, out_arr, grp_info, ftype, skipna);
    if (dtype == Bodo_CTypes::UINT16)
        return cumulative_computation_T<uint16_t, Bodo_CTypes::UINT16>(
            arr, out_arr, grp_info, ftype, skipna);
    if (dtype == Bodo_CTypes::INT32)
        return cumulative_computation_T<int32_t, Bodo_CTypes::INT32>(
            arr, out_arr, grp_info, ftype, skipna);
    if (dtype == Bodo_CTypes::UINT32)
        return cumulative_computation_T<uint32_t, Bodo_CTypes::UINT32>(
            arr, out_arr, grp_info, ftype, skipna);
    if (dtype == Bodo_CTypes::INT64)
        return cumulative_computation_T<int64_t, Bodo_CTypes::INT64>(
            arr, out_arr, grp_info, ftype, skipna);
    if (dtype == Bodo_CTypes::UINT64)
        return cumulative_computation_T<uint64_t, Bodo_CTypes::UINT64>(
            arr, out_arr, grp_info, ftype, skipna);
    if (dtype == Bodo_CTypes::FLOAT32)
        return cumulative_computation_T<float, Bodo_CTypes::FLOAT32>(
            arr, out_arr, grp_info, ftype, skipna);
    if (dtype == Bodo_CTypes::FLOAT64)
        return cumulative_computation_T<double, Bodo_CTypes::FLOAT64>(
            arr, out_arr, grp_info, ftype, skipna);
    if (dtype == Bodo_CTypes::DATE)
        return cumulative_computation_T<int64_t, Bodo_CTypes::DATE>(
            arr, out_arr, grp_info, ftype, skipna);
    if (dtype == Bodo_CTypes::DATETIME)
        return cumulative_computation_T<int64_t, Bodo_CTypes::DATETIME>(
            arr, out_arr, grp_info, ftype, skipna);
    if (dtype == Bodo_CTypes::TIMEDELTA)
        return cumulative_computation_T<int64_t, Bodo_CTypes::TIMEDELTA>(
            arr, out_arr, grp_info, ftype, skipna);
}

template <typename ARRAY>
typename std::enable_if<is_multiple_array<ARRAY>::value, void>::type
cumulative_computation(array_info* arr, ARRAY* out_arr,
                       grouping_info const& grp_info, int32_t const& ftype,
                       bool const& skipna) {
    Bodo_PyErr_SetString(PyExc_RuntimeError,
                         "while cumulative operation makes sense for "
                         "pivot_table/crosstab the functionality is missing "
                         "right now");
}

/**
 * The shift_computation function.
 * Shift rows per group N times (up or down).
 * @param arr column on which we do the computation
 * @param out_arr column data after being shifted
 * @param grp_info: grouping_info about groups and rows organization
 * @param periods: Number of periods to shift
 */
template <typename ARRAY>
typename std::enable_if<!is_multiple_array<ARRAY>::value, void>::type
shift_computation(array_info* arr, array_info* out_arr,
                  grouping_info const& grp_info, int64_t const& periods) {
    size_t num_rows = grp_info.row_to_group.size();
    size_t num_groups = grp_info.num_groups;
    int64_t tmp_periods = periods;
    // 1. Shift operation taken from pandas
    // https://github.com/pandas-dev/pandas/blob/master/pandas/_libs/groupby.pyx#L293

    size_t ii, offset;
    int sign;
    // If periods<0, shift up (i.e. iterate backwards)
    if (periods < 0) {
        tmp_periods = -periods;
        offset = num_rows - 1;
        sign = -1;
    } else {
        offset = 0;
        sign = 1;
    }

    std::vector<int64_t> row_list(num_rows);
    if (tmp_periods == 0) {
        for (size_t i = 0; i < num_rows; i++) {
            row_list[i] = i;
        }
    } else {
        int64_t gid;       // group number
        int64_t cur_pos;   // new value for current row
        int64_t prev_val;  // previous value
        std::vector<int64_t> nrows_per_group(
            num_groups);  // array holding number of rows per group
        std::vector<std::vector<int64_t>> p_values(
            num_groups,
            std::vector<int64_t>(tmp_periods));  // 2d array holding most recent
                                                 // N=periods elements per group
        // For each row value, find if it should be NaN or it will get a value
        // that is N=periods away from it. It's a NaN if it's row_number <
        // periods, otherwise get it's new value from (row_number -/+ periods)
        for (size_t i = 0; i < num_rows; i++) {
            ii = offset + sign * i;
            gid = grp_info.row_to_group[ii];
            if (gid == -1) {
                row_list[ii] = -1;
                continue;
            }

            nrows_per_group[gid]++;
            cur_pos = nrows_per_group[gid] % tmp_periods;
            prev_val = p_values[gid][cur_pos];
            if (nrows_per_group[gid] > tmp_periods) {
                row_list[ii] = prev_val;
            } else {
                row_list[ii] = -1;
            }
            p_values[gid][cur_pos] = ii;
        }  // end-row_loop
    }
    // 2. Retrieve column and put it in update_cols
    array_info* updated_col = RetrieveArray_SingleColumn(arr, row_list);
    *out_arr = std::move(*updated_col);
    delete updated_col;
}
template <typename ARRAY>
typename std::enable_if<is_multiple_array<ARRAY>::value, void>::type
shift_computation(array_info* arr, ARRAY* out_arr,
                  grouping_info const& grp_info, int64_t const& periods) {
    Bodo_PyErr_SetString(PyExc_RuntimeError,
                         "multiarray shift functionality is missing right now");
}
/**
 * The head_computation function.
 * Copy rows identified by row_list from input to output column
 * @param arr column on which we do the computation
 * @param out_arr output column data
 * @param grp_info: grouping_info about groups and rows organization
 * @param row_list: row indices to copy
 */
template <typename ARRAY>
typename std::enable_if<!is_multiple_array<ARRAY>::value, void>::type
head_computation(array_info* arr, array_info* out_arr,
                 const std::vector<int64_t>& row_list) {
    array_info* updated_col = RetrieveArray_SingleColumn(arr, row_list);
    *out_arr = std::move(*updated_col);
    delete updated_col;
}
template <typename ARRAY>
typename std::enable_if<is_multiple_array<ARRAY>::value, void>::type
head_computation(array_info* arr, ARRAY* out_arr,
                 const std::vector<int64_t>& row_list) {
    throw std::runtime_error(
        "gb.head() is not implemented for multiple_array.");
}

/**
 * The median_computation function. It uses the symbolic information to compute
 * the median results.
 *
 * @param The column on which we do the computation
 * @param The array containing information on how the rows are organized
 * @param skipna: Whether to skip NaN values or not.
 */
template <typename ARRAY>
typename std::enable_if<!is_multiple_array<ARRAY>::value, void>::type
median_computation(array_info* arr, array_info* out_arr,
                   grouping_info const& grp_info, bool const& skipna) {
    size_t num_group = grp_info.group_to_first_row.size();
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
    auto median_operation = [&](auto const& isnan_entry) -> void {
        for (size_t igrp = 0; igrp < num_group; igrp++) {
            int64_t i = grp_info.group_to_first_row[igrp];
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
                i = grp_info.next_row_in_group[i];
                if (i == -1) break;
            }
            auto GetKthValue = [&](size_t const& pos) -> double {
                std::nth_element(ListValue.begin(), ListValue.begin() + pos,
                                 ListValue.end());
                return ListValue[pos];
            };
            double valReturn;
            // a group can be empty if it has all NaNs so output will be NaN
            // even if skipna=True
            if (HasNaN || ListValue.size() == 0) {
                valReturn = std::nan("1");
            } else {
                size_t len = ListValue.size();
                size_t res = len % 2;
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
        median_operation(
            [=](size_t pos) -> bool { return !arr->get_null_bit(pos); });
    }
}

template <typename ARRAY>
typename std::enable_if<is_multiple_array<ARRAY>::value, void>::type
median_computation(array_info* arr, ARRAY* out_arr,
                   grouping_info const& grp_info, bool const& skipna) {
    Bodo_PyErr_SetString(PyExc_RuntimeError,
                         "while median makes sense for pivot_table/crosstab "
                         "the functionality is missing right now");
}

namespace {
/**
 * Compute hash for numpy or nullable_int_bool bodo types
 *
 * Don't use std::function to reduce call overhead.
 */
struct HashNuniqueComputationNumpyOrNullableIntBool {
    uint32_t operator()(const int64_t i) const {
        char* ptr = arr->data1 + i * siztype;
        uint32_t retval = 0;
        hash_string_32(ptr, siztype, seed, &retval);
        return retval;
    }
    array_info* arr;
    size_t siztype;
    uint32_t seed;
};

/**
 * Key comparison for numpy or nullable_int_bool bodo types
 *
 * Don't use std::function to reduce call overhead.
 */
struct KeyEqualNuniqueComputationNumpyOrNullableIntBool {
    bool operator()(const int64_t i1, const int64_t i2) const {
        char* ptr1 = arr->data1 + i1 * siztype;
        char* ptr2 = arr->data1 + i2 * siztype;
        return memcmp(ptr1, ptr2, siztype) == 0;
    }

    array_info* arr;
    size_t siztype;
};

/**
 * Compute hash for list string bodo types.
 *
 * Don't use std::function to reduce call overhead.
 */
struct HashNuniqueComputationListString {
    size_t operator()(const int64_t i) const {
        // We do not put the lengths and bitmask in the hash
        // computation. after all, it is just a hash
        char* val_chars = arr->data1 + in_data_offsets[in_index_offsets[i]];
        int len = in_data_offsets[in_index_offsets[i + 1]] -
                  in_data_offsets[in_index_offsets[i]];
        uint32_t val;
        hash_string_32(val_chars, len, seed, &val);
        return static_cast<size_t>(val);
    }
    array_info* arr;
    offset_t* in_index_offsets;
    offset_t* in_data_offsets;
    uint32_t seed;
};

/**
 * Key comparison for list string bodo types
 *
 * Don't use std::function to reduce call overhead.
 */
struct KeyEqualNuniqueComputationListString {
    bool operator()(const int64_t i1, const int64_t i2) const {
        bool bit1 = arr->get_null_bit(i1);
        bool bit2 = arr->get_null_bit(i2);
        if (bit1 != bit2)
            return false;  // That first case, might not be necessary.
        size_t len1 = in_index_offsets[i1 + 1] - in_index_offsets[i1];
        size_t len2 = in_index_offsets[i2 + 1] - in_index_offsets[i2];
        if (len1 != len2) return false;
        for (size_t u = 0; u < len1; u++) {
            offset_t len_str1 = in_data_offsets[in_index_offsets[i1] + 1] -
                                in_data_offsets[in_index_offsets[i1]];
            offset_t len_str2 = in_data_offsets[in_index_offsets[i2] + 1] -
                                in_data_offsets[in_index_offsets[i2]];
            if (len_str1 != len_str2) return false;
            bool bit1 = GetBit(sub_null_bitmask, in_index_offsets[i1]);
            bool bit2 = GetBit(sub_null_bitmask, in_index_offsets[i2]);
            if (bit1 != bit2) return false;
        }
        offset_t nb_char1 = in_data_offsets[in_index_offsets[i1 + 1]] -
                            in_data_offsets[in_index_offsets[i1]];
        offset_t nb_char2 = in_data_offsets[in_index_offsets[i2 + 1]] -
                            in_data_offsets[in_index_offsets[i2]];
        if (nb_char1 != nb_char2) return false;
        char* ptr1 = arr->data1 +
                     sizeof(offset_t) * in_data_offsets[in_index_offsets[i1]];
        char* ptr2 = arr->data1 +
                     sizeof(offset_t) * in_data_offsets[in_index_offsets[i2]];
        return memcmp(ptr1, ptr2, len1) == 0;
    }

    array_info* arr;
    offset_t* in_index_offsets;
    offset_t* in_data_offsets;
    uint8_t* sub_null_bitmask;
    uint32_t seed;
};

/**
 * Compute hash for string bodo types.
 *
 * Don't use std::function to reduce call overhead.
 */
struct HashNuniqueComputationString {
    size_t operator()(const int64_t i) const {
        char* val_chars = arr->data1 + in_offsets[i];
        size_t len = in_offsets[i + 1] - in_offsets[i];
        uint32_t val;
        hash_string_32(val_chars, len, seed, &val);
        return size_t(val);
    }
    array_info* arr;
    offset_t* in_offsets;
    uint32_t seed;
};

/**
 * Key comparison for string bodo types
 *
 * Don't use std::function to reduce call overhead.
 */
struct KeyEqualNuniqueComputationString {
    bool operator()(const int64_t i1, const int64_t i2) const {
        size_t len1 = in_offsets[i1 + 1] - in_offsets[i1];
        size_t len2 = in_offsets[i2 + 1] - in_offsets[i2];
        if (len1 != len2) {
            return false;
        }
        char* ptr1 = arr->data1 + in_offsets[i1];
        char* ptr2 = arr->data1 + in_offsets[i2];
        return memcmp(ptr1, ptr2, len1) == 0;
    }

    array_info* arr;
    offset_t* in_offsets;
};
}  // namespace

/**
 * The nunique_computation function. It uses the symbolic information to compute
 * the nunique results.
 *
 * @param arr The column on which we do the computation
 * @param out_arr[out] The column which contains nunique results
 * @param grp_info The array containing information on how the rows are
 * organized
 * @param dropna The boolean dropna indicating whether we drop or not the NaN
 * values from the nunique computation.
 * @param is_parallel: true if data is distributed (used to indicate whether
 * tracing should be parallel or not)
 */
template <typename ARRAY>
typename std::enable_if<!is_multiple_array<ARRAY>::value, void>::type
nunique_computation(array_info* arr, array_info* out_arr,
                    grouping_info const& grp_info, bool const& dropna,
                    bool const& is_parallel) {
    tracing::Event ev("nunique_computation", is_parallel);
    size_t num_group = grp_info.group_to_first_row.size();
    if (num_group == 0) return;
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
        const size_t siztype = numpy_item_size[arr->dtype];
        const uint32_t seed = SEED_HASH_CONTAINER;

        HashNuniqueComputationNumpyOrNullableIntBool hash_fct{arr, siztype,
                                                              seed};
        KeyEqualNuniqueComputationNumpyOrNullableIntBool equal_fct{arr,
                                                                   siztype};
        UNORD_SET_CONTAINER<int64_t,
                            HashNuniqueComputationNumpyOrNullableIntBool,
                            KeyEqualNuniqueComputationNumpyOrNullableIntBool>
            eset({}, hash_fct, equal_fct);
        eset.reserve(double(arr->length) / num_group);  // NOTE: num_group > 0
        eset.max_load_factor(UNORDERED_MAP_MAX_LOAD_FACTOR);

        for (size_t igrp = 0; igrp < num_group; igrp++) {
            int64_t i = grp_info.group_to_first_row[igrp];
            // with nunique mode=2 some groups might not be present in the
            // nunique table
            if (i < 0) continue;
            eset.clear();
            bool HasNullRow = false;
            while (true) {
                char* ptr = arr->data1 + (i * siztype);
                if (!isnan_entry(ptr)) {
                    eset.insert(i);
                } else {
                    HasNullRow = true;
                }
                i = grp_info.next_row_in_group[i];
                if (i == -1) break;
            }
            int64_t size = eset.size();
            if (HasNullRow && !dropna) size++;
            out_arr->at<int64_t>(igrp) = size;
        }
    }

    if (arr->arr_type == bodo_array_type::LIST_STRING) {
        offset_t* in_index_offsets = (offset_t*)arr->data3;
        offset_t* in_data_offsets = (offset_t*)arr->data2;
        uint8_t* sub_null_bitmask = (uint8_t*)arr->sub_null_bitmask;
        const uint32_t seed = SEED_HASH_CONTAINER;

        HashNuniqueComputationListString hash_fct{arr, in_index_offsets,
                                                  in_data_offsets, seed};
        KeyEqualNuniqueComputationListString equal_fct{
            arr, in_index_offsets, in_data_offsets, sub_null_bitmask, seed};
        UNORD_SET_CONTAINER<int64_t, HashNuniqueComputationListString,
                            KeyEqualNuniqueComputationListString>
            eset({}, hash_fct, equal_fct);
        eset.reserve(double(arr->length) / num_group);  // NOTE: num_group > 0
        eset.max_load_factor(UNORDERED_MAP_MAX_LOAD_FACTOR);

        for (size_t igrp = 0; igrp < num_group; igrp++) {
            int64_t i = grp_info.group_to_first_row[igrp];
            // with nunique mode=2 some groups might not be present in the
            // nunique table
            if (i < 0) continue;
            eset.clear();
            bool HasNullRow = false;
            while (true) {
                if (arr->get_null_bit(i)) {
                    eset.insert(i);
                } else {
                    HasNullRow = true;
                }
                i = grp_info.next_row_in_group[i];
                if (i == -1) break;
            }
            int64_t size = eset.size();
            if (HasNullRow && !dropna) size++;
            out_arr->at<int64_t>(igrp) = size;
        }
    }

    if (arr->arr_type == bodo_array_type::STRING) {
        offset_t* in_offsets = (offset_t*)arr->data2;
        const uint32_t seed = SEED_HASH_CONTAINER;

        HashNuniqueComputationString hash_fct{arr, in_offsets, seed};
        KeyEqualNuniqueComputationString equal_fct{arr, in_offsets};
        UNORD_SET_CONTAINER<int64_t, HashNuniqueComputationString,
                            KeyEqualNuniqueComputationString>
            eset({}, hash_fct, equal_fct);
        eset.reserve(double(arr->length) / num_group);  // NOTE: num_group > 0
        eset.max_load_factor(UNORDERED_MAP_MAX_LOAD_FACTOR);

        for (size_t igrp = 0; igrp < num_group; igrp++) {
            int64_t i = grp_info.group_to_first_row[igrp];
            // with nunique mode=2 some groups might not be present in the
            // nunique table
            if (i < 0) continue;
            eset.clear();
            bool HasNullRow = false;
            while (true) {
                if (arr->get_null_bit(i)) {
                    eset.insert(i);
                } else {
                    HasNullRow = true;
                }
                i = grp_info.next_row_in_group[i];
                if (i == -1) break;
            }
            int64_t size = eset.size();
            if (HasNullRow && !dropna) size++;
            out_arr->at<int64_t>(igrp) = size;
        }
    }

    if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        const size_t siztype = numpy_item_size[arr->dtype];
        HashNuniqueComputationNumpyOrNullableIntBool hash_fct{arr, siztype};
        KeyEqualNuniqueComputationNumpyOrNullableIntBool equal_fct{arr,
                                                                   siztype};
        UNORD_SET_CONTAINER<int64_t,
                            HashNuniqueComputationNumpyOrNullableIntBool,
                            KeyEqualNuniqueComputationNumpyOrNullableIntBool>
            eset({}, hash_fct, equal_fct);
        eset.reserve(double(arr->length) / num_group);  // NOTE: num_group > 0
        eset.max_load_factor(UNORDERED_MAP_MAX_LOAD_FACTOR);

        for (size_t igrp = 0; igrp < num_group; igrp++) {
            int64_t i = grp_info.group_to_first_row[igrp];
            // with nunique mode=2 some groups might not be present in the
            // nunique table
            if (i < 0) continue;
            eset.clear();
            bool HasNullRow = false;
            while (true) {
                if (arr->get_null_bit(i)) {
                    eset.insert(i);
                } else {
                    HasNullRow = true;
                }
                i = grp_info.next_row_in_group[i];
                if (i == -1) break;
            }
            int64_t size = eset.size();
            if (HasNullRow && !dropna) size++;
            out_arr->at<int64_t>(igrp) = size;
        }
    }
}

template <typename ARRAY>
typename std::enable_if<is_multiple_array<ARRAY>::value, void>::type
nunique_computation(array_info* arr, ARRAY* out_arr,
                    grouping_info const& grp_info, bool const& skipna,
                    bool const& is_parallel) {
    Bodo_PyErr_SetString(PyExc_RuntimeError,
                         "while nunique makes sense for pivot_table/crosstab "
                         "the functionality is missing right now");
}

template <typename ARR_I, typename ARR_O, typename F, int ftype>
typename std::enable_if<!is_multiple_array<ARR_I>::value, ARR_O*>::type
apply_to_column_list_string(ARR_I* in_col, ARR_O* out_col,
                            const grouping_info& grp_info, F f) {
    size_t num_groups = grp_info.num_groups;
    std::vector<std::vector<pair_str_bool>> ListListPair(num_groups);
    char* data_i = in_col->data1;
    offset_t* index_offsets_i = (offset_t*)in_col->data3;
    offset_t* data_offsets_i = (offset_t*)in_col->data2;
    uint8_t* sub_null_bitmask_i = (uint8_t*)in_col->sub_null_bitmask;
    // Computing the strings used in output.
    uint64_t n_bytes = (num_groups + 7) >> 3;
    std::vector<uint8_t> Vmask(n_bytes, 0);
    for (int64_t i = 0; i < in_col->length; i++) {
        int64_t i_grp = f(i);
        if ((i_grp != -1) && in_col->get_null_bit(i)) {
            bool out_bit_set = out_col->get_null_bit(i_grp);
            if (ftype == Bodo_FTypes::first && out_bit_set) continue;
            offset_t start_offset = index_offsets_i[i];
            offset_t end_offset = index_offsets_i[i + 1];
            offset_t len = end_offset - start_offset;
            std::vector<pair_str_bool> LStrB(len);
            for (offset_t i = 0; i < len; i++) {
                offset_t len_str = data_offsets_i[start_offset + i + 1] -
                                   data_offsets_i[start_offset + i];
                offset_t pos_start = data_offsets_i[start_offset + i];
                std::string val(&data_i[pos_start], len_str);
                bool str_bit = GetBit(sub_null_bitmask_i, start_offset + i);
                LStrB[i] = {val, str_bit};
            }
            if (out_bit_set) {
                aggliststring<ftype>::apply(ListListPair[i_grp], LStrB);
            } else {
                ListListPair[i_grp] = LStrB;
                out_col->set_null_bit(i_grp, true);
            }
        }
    }
    return create_list_string_array<ARR_O>(grp_info, Vmask, ListListPair);
}

template <typename ARR_I, typename ARR_O, typename F, int ftype>
typename std::enable_if<is_multiple_array<ARR_I>::value, ARR_O*>::type
apply_to_column_list_string(ARR_I* in_col, ARR_O* out_col,
                            const grouping_info& grp_info, F f) {
    Bodo_PyErr_SetString(PyExc_RuntimeError,
                         "The code is missing for this possibility");
    return nullptr;
}

template <typename ARR_I, typename ARR_O, typename F, int ftype>
typename std::enable_if<!is_multiple_array<ARR_I>::value, ARR_O*>::type
apply_to_column_string(ARR_I* in_col, ARR_O* out_col,
                       const grouping_info& grp_info, F f) {
#ifdef DEBUG_GROUPBY
    std::cout << "Beginning of apply_to_column_string\n";
#endif
    size_t num_groups = grp_info.num_groups;
    size_t n_bytes = (num_groups + 7) >> 3;
    std::vector<uint8_t> V(n_bytes, 0);
    std::vector<std::string> ListString(num_groups);
    char* data_i = in_col->data1;
    offset_t* offsets_i = (offset_t*)in_col->data2;
    // Computing the strings used in output.
    for (int64_t i = 0; i < in_col->length; i++) {
        int64_t i_grp = f(i);
        if ((i_grp != -1) && in_col->get_null_bit(i)) {
            bool out_bit_set = GetBit(V.data(), i_grp);
            if (ftype == Bodo_FTypes::first && out_bit_set) continue;
            offset_t start_offset = offsets_i[i];
            offset_t end_offset = offsets_i[i + 1];
            offset_t len = end_offset - start_offset;
            std::string val(&data_i[start_offset], len);
            if (out_bit_set) {
                aggstring<ftype>::apply(ListString[i_grp], val);
            } else {
                ListString[i_grp] = val;
                SetBitTo(V.data(), i_grp, true);
            }
        }
    }
    // Determining the number of characters in output.
    return create_string_array<ARR_O>(grp_info, V, ListString);
}

template <typename ARR_I, typename ARR_O, typename F, int ftype>
typename std::enable_if<is_multiple_array<ARR_I>::value, ARR_O*>::type
apply_to_column_string(ARR_I* in_col, ARR_O* out_col,
                       const grouping_info& grp_info, F f) {
    Bodo_PyErr_SetString(PyExc_RuntimeError,
                         "The code is missing for this possibility");
    return nullptr;
}

template <typename ARR_I, typename ARR_O, typename F, int ftype>
typename std::enable_if<!(is_multiple_array<ARR_I>::value ||
                          is_multiple_array<ARR_O>::value),
                        ARR_O*>::type
apply_sum_to_column_string(ARR_I* in_col, ARR_O* out_col,
                           const grouping_info& grp_info, F f) {
#ifdef DEBUG_GROUPBY
    std::cout << "Beginning of apply_to_column_string\n";
#endif

    // allocate output array (length is number of groups, number of chars same
    // as input)
    size_t num_groups = grp_info.num_groups;
    int64_t n_chars = in_col->n_sub_elems;
    array_info* out_arr = alloc_string_array(num_groups, n_chars, 0);
    size_t n_bytes = (num_groups + 7) >> 3;
    memset(out_arr->null_bitmask, 0xff, n_bytes);  // null not possible

    // find offsets for each output string
    std::vector<offset_t> str_offsets(num_groups + 1, 0);
    char* data_i = in_col->data1;
    offset_t* offsets_i = (offset_t*)in_col->data2;
    char* data_o = out_arr->data1;
    offset_t* offsets_o = (offset_t*)out_arr->data2;

    for (int64_t i = 0; i < in_col->length; i++) {
        int64_t i_grp = f(i);
        if ((i_grp != -1) && in_col->get_null_bit(i)) {
            offset_t len = offsets_i[i + 1] - offsets_i[i];
            str_offsets[i_grp + 1] += len;
        }
    }
    std::partial_sum(str_offsets.begin(), str_offsets.end(),
                     str_offsets.begin());
    memcpy(offsets_o, str_offsets.data(), (num_groups + 1) * sizeof(offset_t));

    // copy characters to output
    for (int64_t i = 0; i < in_col->length; i++) {
        int64_t i_grp = f(i);
        if ((i_grp != -1) && in_col->get_null_bit(i)) {
            offset_t len = offsets_i[i + 1] - offsets_i[i];
            memcpy(&data_o[str_offsets[i_grp]], data_i + offsets_i[i], len);
            str_offsets[i_grp] += len;
        }
    }
    return out_arr;
}

template <typename ARR_I, typename ARR_O, typename F, int ftype>
typename std::enable_if<(is_multiple_array<ARR_I>::value ||
                         is_multiple_array<ARR_O>::value),
                        ARR_O*>::type
apply_sum_to_column_string(ARR_I* in_col, ARR_O* out_col,
                           const grouping_info& grp_info, F f) {
    Bodo_PyErr_SetString(PyExc_RuntimeError,
                         "Invalid/unsupported groupyby string sum case");
    return nullptr;
}

template <typename ARR_I>
typename std::enable_if<!is_multiple_array<ARR_I>::value, bool>::type
do_computation(ARR_I* in_col, int64_t i) {
    return true;
}

template <typename ARR_I>
typename std::enable_if<is_multiple_array<ARR_I>::value, bool>::type
do_computation(ARR_I* in_col, int64_t i) {
    return in_col->get_access_bit(i);
}

/**
 * Apply a function to a column(s), save result to (possibly reduced) output
 * column(s) Semantics of this function right now vary depending on function
 * type (ftype).
 *
 * template parameters are:
 * For groupby operation: ARR_I = ARR_O = array_info during the update and
 *                        combine steps
 * For pivot_table / crosstab:
 * --- during update:
 *    ARR_I = array_info   ARR_O = multiple_array_info
 * --- during combine:
 *    ARR_I = ARR_OI = multiple_array_info
 * During pivot_table / crosstab the "do_computation" is used for the missing
 * data. That is whether it was accessed or not. For groupby, this collapses to
 * true by the template evaluation of the AST.
 *
 * @param column containing input values
 * @param output column
 * @param auxiliary input/output columns used for mean, var, std
 * @param maps row numbers in input columns to group numbers (for reduction
 * operations)
 */
template <typename ARR_I, typename ARR_O, typename F, typename T, int ftype,
          int dtype>
void apply_to_column_F(ARR_I* in_col, ARR_O* out_col,
                       std::vector<ARR_O*>& aux_cols,
                       const grouping_info& grp_info, F f) {
    switch (in_col->arr_type) {
        case bodo_array_type::CATEGORICAL:
            if (ftype == Bodo_FTypes::count) {
                for (int64_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = f(i);
                    if (i_grp != -1) {
                        T& val = getv<ARR_I, T>(in_col, i);
                        if (!isnan_categorical<T, dtype>(val)) {
                            count_agg<T, dtype>::apply(
                                getv<ARR_O, int64_t>(out_col, i_grp), val);
                        }
                    }
                }
                return;
            } else if (ftype == Bodo_FTypes::min ||
                       ftype == Bodo_FTypes::last) {
                // NOTE: Bodo_FTypes::max is handled for categorical type since
                // NA is -1.
                for (int64_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = f(i);
                    if (i_grp != -1) {
                        T& val = getv<ARR_I, T>(in_col, i);
                        if (!isnan_categorical<T, dtype>(val)) {
                            aggfunc<T, dtype, ftype>::apply(
                                getv<ARR_O, T>(out_col, i_grp), val);
                        }
                    }
                }
                // aggfunc_output_initialize_kernel, min defaults
                // to num_categories if all entries are NA
                // this needs to be replaced with -1
                if (ftype == Bodo_FTypes::min) {
                    for (int64_t i = 0; i < out_col->length; i++) {
                        T& val = getv<ARR_O, T>(out_col, i);
                        set_na_if_num_categories<T, dtype>(
                            val, out_col->num_categories);
                    }
                }
                return;
            } else if (ftype == Bodo_FTypes::first) {
                int64_t n_bytes = ((out_col->length + 7) >> 3);
                std::vector<uint8_t> bitmask_vec(n_bytes, 0);
                for (int64_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = f(i);
                    T val = getv<ARR_I, T>(in_col, i);
                    if ((i_grp != -1) && !GetBit(bitmask_vec.data(), i_grp) &&
                        !isnan_categorical<T, dtype>(val)) {
                        getv<ARR_O, T>(out_col, i_grp) = val;
                        SetBitTo(bitmask_vec.data(), i_grp, true);
                    }
                }
            }
        case bodo_array_type::NUMPY:
            if (ftype == Bodo_FTypes::mean) {
                ARR_O* count_col = aux_cols[0];
                for (int64_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = f(i);
                    if (i_grp != -1)
                        mean_agg<T, dtype>::apply(
                            getv<ARR_O, double>(out_col, i_grp),
                            getv<ARR_I, T>(in_col, i),
                            getv<ARR_O, uint64_t>(count_col, i_grp));
                }
            } else if (ftype == Bodo_FTypes::mean_eval) {
                for (int64_t i = 0; i < in_col->length; i++)
                    mean_eval(getv<ARR_O, double>(out_col, i),
                              getv<ARR_I, uint64_t>(in_col, i));
            } else if (ftype == Bodo_FTypes::var) {
                ARR_O* count_col = aux_cols[0];
                ARR_O* mean_col = aux_cols[1];
                ARR_O* m2_col = aux_cols[2];
                for (int64_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = f(i);
                    if (i_grp != -1)
                        var_agg<T, dtype>::apply(
                            getv<ARR_I, T>(in_col, i),
                            getv<ARR_O, uint64_t>(count_col, i_grp),
                            getv<ARR_O, double>(mean_col, i_grp),
                            getv<ARR_O, double>(m2_col, i_grp));
                }
            } else if (ftype == Bodo_FTypes::var_eval) {
                ARR_O* count_col = aux_cols[0];
                ARR_O* m2_col = aux_cols[2];
                for (int64_t i = 0; i < in_col->length; i++)
                    var_eval(getv<ARR_O, double>(out_col, i),
                             getv<ARR_O, uint64_t>(count_col, i),
                             getv<ARR_O, double>(m2_col, i));
            } else if (ftype == Bodo_FTypes::std_eval) {
                ARR_O* count_col = aux_cols[0];
                ARR_O* m2_col = aux_cols[2];
                for (int64_t i = 0; i < in_col->length; i++)
                    std_eval(getv<ARR_O, double>(out_col, i),
                             getv<ARR_O, uint64_t>(count_col, i),
                             getv<ARR_O, double>(m2_col, i));
            } else if (ftype == Bodo_FTypes::count) {
                for (int64_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = f(i);
                    if (i_grp != -1)
                        count_agg<T, dtype>::apply(
                            getv<ARR_O, int64_t>(out_col, i_grp),
                            getv<ARR_I, T>(in_col, i));
                }
            } else if (ftype == Bodo_FTypes::first) {
                // create a temporary bitmask to know if we have set a
                // value for each row/group
                int64_t n_bytes = ((out_col->length + 7) >> 3);
                std::vector<uint8_t> bitmask_vec(n_bytes, 0);
                for (int64_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = f(i);
                    T val = getv<ARR_I, T>(in_col, i);
                    if ((i_grp != -1) && !GetBit(bitmask_vec.data(), i_grp) &&
                        !isnan_alltype<T, dtype>(val)) {
                        getv<ARR_O, T>(out_col, i_grp) = val;
                        SetBitTo(bitmask_vec.data(), i_grp, true);
                    }
                }
            } else if (ftype == Bodo_FTypes::idxmax) {
                ARR_O* index_pos = aux_cols[0];
                for (int64_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = f(i);
                    if (i_grp != -1) {
                        idxmax_agg<T, dtype>::apply(
                            getv<ARR_O, T>(out_col, i_grp),
                            getv<ARR_I, T>(in_col, i),
                            getv<ARR_O, uint64_t>(index_pos, i_grp), i);
                    }
                }
            } else if (ftype == Bodo_FTypes::idxmin) {
                ARR_O* index_pos = aux_cols[0];
                for (int64_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = f(i);
                    if (i_grp != -1) {
                        idxmin_agg<T, dtype>::apply(
                            getv<ARR_O, T>(out_col, i_grp),
                            getv<ARR_I, T>(in_col, i),
                            getv<ARR_O, uint64_t>(index_pos, i_grp), i);
                    }
                }
            } else {
                for (int64_t i = 0; i < in_col->length; i++) {
                    int64_t i_grp = f(i);
                    if (i_grp != -1 && do_computation<ARR_I>(in_col, i))
                        aggfunc<T, dtype, ftype>::apply(
                            getv<ARR_O, T>(out_col, i_grp),
                            getv<ARR_I, T>(in_col, i));
                }
            }
            return;
        // for list strings, we are supporting count, sum, max, min, first, last
        case bodo_array_type::LIST_STRING:
            switch (ftype) {
                case Bodo_FTypes::count: {
                    for (int64_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        if ((i_grp != -1) && in_col->get_null_bit(i))
                            count_agg<T, dtype>::apply(
                                getv<ARR_O, int64_t>(out_col, i_grp),
                                getv<ARR_I, T>(in_col, i));
                    }
                    return;
                }
                default:
                    ARR_O* new_out_col =
                        apply_to_column_list_string<ARR_I, ARR_O, F, ftype>(
                            in_col, out_col, grp_info, f);
                    *out_col = std::move(*new_out_col);
                    delete new_out_col;
                    return;
            }

        // For the STRING we compute the count, sum, max, min, first, last
        case bodo_array_type::STRING:
            switch (ftype) {
                case Bodo_FTypes::count: {
                    for (int64_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        if ((i_grp != -1) && in_col->get_null_bit(i))
                            count_agg<T, dtype>::apply(
                                getv<ARR_O, int64_t>(out_col, i_grp),
                                getv<ARR_I, T>(in_col, i));
                    }
                    return;
                }
                // optimized groupby sum for strings (concatenation)
                case Bodo_FTypes::sum: {
                    ARR_O* new_out_col =
                        apply_sum_to_column_string<ARR_I, ARR_O, F, ftype>(
                            in_col, out_col, grp_info, f);
                    *out_col = std::move(*new_out_col);
                    delete new_out_col;
                    return;
                }
                default:
                    ARR_O* new_out_col =
                        apply_to_column_string<ARR_I, ARR_O, F, ftype>(
                            in_col, out_col, grp_info, f);
                    *out_col = std::move(*new_out_col);
                    delete new_out_col;
                    return;
            }
        case bodo_array_type::NULLABLE_INT_BOOL:
            switch (ftype) {
                case Bodo_FTypes::count: {
                    for (int64_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        if ((i_grp != -1) && in_col->get_null_bit(i))
                            count_agg<T, dtype>::apply(
                                getv<ARR_O, int64_t>(out_col, i_grp),
                                getv<ARR_I, T>(in_col, i));
                    }
                    return;
                }
                case Bodo_FTypes::mean:
                    for (int64_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        if ((i_grp != -1) && in_col->get_null_bit(i))
                            mean_agg<T, dtype>::apply(
                                getv<ARR_O, double>(out_col, i_grp),
                                getv<ARR_I, T>(in_col, i),
                                getv<ARR_O, uint64_t>(aux_cols[0], i_grp));
                    }
                    return;
                case Bodo_FTypes::var:
                    for (int64_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        if ((i_grp != -1) && in_col->get_null_bit(i) &&
                            do_computation<ARR_I>(in_col, i))
                            var_agg<T, dtype>::apply(
                                getv<ARR_I, T>(in_col, i),
                                getv<ARR_O, uint64_t>(aux_cols[0], i_grp),
                                getv<ARR_O, double>(aux_cols[1], i_grp),
                                getv<ARR_O, double>(aux_cols[2], i_grp));
                    }
                    return;
                case Bodo_FTypes::first:
                    for (int64_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        if ((i_grp != -1) && !out_col->get_null_bit(i_grp) &&
                            in_col->get_null_bit(i) &&
                            do_computation<ARR_I>(in_col, i)) {
                            getv<ARR_O, T>(out_col, i_grp) =
                                getv<ARR_I, T>(in_col, i);
                            out_col->set_null_bit(i_grp, true);
                        }
                    }
                    return;
                case Bodo_FTypes::idxmax:
                    for (int64_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        if (i_grp != -1) {
                            idxmax_agg<T, dtype>::apply(
                                getv<ARR_O, T>(out_col, i_grp),
                                getv<ARR_I, T>(in_col, i),
                                getv<ARR_O, uint64_t>(aux_cols[0], i_grp), i);
                        }
                    }
                    return;
                case Bodo_FTypes::idxmin:
                    for (int64_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        if (i_grp != -1) {
                            idxmin_agg<T, dtype>::apply(
                                getv<ARR_O, T>(out_col, i_grp),
                                getv<ARR_I, T>(in_col, i),
                                getv<ARR_O, uint64_t>(aux_cols[0], i_grp), i);
                        }
                    }
                    return;
                default: {
                    for (int64_t i = 0; i < in_col->length; i++) {
                        int64_t i_grp = f(i);
                        if ((i_grp != -1) && in_col->get_null_bit(i) &&
                            do_computation<ARR_I>(in_col, i)) {
                            aggfunc<T, dtype, ftype>::apply(
                                getv<ARR_O, T>(out_col, i_grp),
                                getv<ARR_I, T>(in_col, i));
                            out_col->set_null_bit(i_grp, true);
                        }
                    }
                    return;
                }
            }
        default:
            Bodo_PyErr_SetString(PyExc_RuntimeError,
                                 "apply_to_column: incorrect array type");
            return;
    }
}

template <typename ARR_I, typename ARR_O, typename T, int ftype, int dtype>
inline typename std::enable_if<!is_multiple_array<ARR_O>::value, void>::type
apply_to_column(ARR_I* in_col, ARR_O* out_col, std::vector<ARR_O*>& aux_cols,
                const grouping_info& grp_info) {
#ifdef DEBUG_GROUPBY
    std::cout << "apply_to_column single case\n";
#endif
    auto f = [&](int64_t const& i_row) -> int64_t {
        return grp_info.row_to_group[i_row];
    };
    return apply_to_column_F<ARR_I, ARR_O, decltype(f), T, ftype, dtype>(
        in_col, out_col, aux_cols, grp_info, f);
}

template <typename ARR_I, typename ARR_O, typename T, int ftype, int dtype>
inline typename std::enable_if<is_multiple_array<ARR_O>::value, void>::type
apply_to_column(ARR_I* in_col, ARR_O* out_col, std::vector<ARR_O*>& aux_cols,
                const grouping_info& grp_info) {
#ifdef DEBUG_GROUPBY
    std::cout << "apply_to_column multiple case mode=" << grp_info.mode << "\n";
#endif
    table_info* dispatch_table = grp_info.dispatch_table;
    table_info* dispatch_info = grp_info.dispatch_info;
    int n_cols = dispatch_table->ncols();
    int n_dis = dispatch_table->nrows();
    size_t n_pivot = grp_info.n_pivot;
    bool na_position_bis = true;
    auto KeyComparisonAsPython_Table = [&](int i_info, int i_dis) -> bool {
        for (int64_t i_col = 0; i_col < n_cols; i_col++) {
            int test = KeyComparisonAsPython_Column(
                na_position_bis, dispatch_table->columns[i_col], i_dis,
                dispatch_info->columns[i_col], i_info);
            if (test != 0) return false;
        }
        return true;
    };
    auto get_i_dis = [&](int64_t const& i_info) -> int64_t {
        for (int64_t i_dis = 0; i_dis < n_dis; i_dis++) {
            bool test = KeyComparisonAsPython_Table(i_info, i_dis);
            if (test) return i_dis;
        }
        return -1;
    };
    auto f = [&](int64_t const& i_row) -> int64_t {
        if (grp_info.mode == 1) {
            int64_t i_grp = grp_info.row_to_group[i_row];
            int i_dis = get_i_dis(i_row);
            // If i_dis = -1 it can be because the input was wrong.
            // But it can be because a column has been dropped out by the
            // compiler passes.
            if (i_dis == -1) return -1;
            return i_dis + n_pivot * i_grp;
        } else {
            int i_dis = i_row % n_pivot;
            int i_row_red = i_row / n_pivot;
            int64_t i_grp_red = grp_info.row_to_group[i_row_red];
            if (i_grp_red == -1) {
                return -1;
            }
            int64_t i_grp = i_dis + n_pivot * i_grp_red;
            return i_grp;
        }
    };
    return apply_to_column_F<ARR_I, ARR_O, decltype(f), T, ftype, dtype>(
        in_col, out_col, aux_cols, grp_info, f);
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
template <typename ARR_I, typename ARR_O>
void do_apply_to_column(ARR_I* in_col, ARR_O* out_col,
                        std::vector<ARR_O*>& aux_cols,
                        const grouping_info& grp_info, int ftype) {
    // size operation is the same regardless of type of data.
    // Hence, just compute number of rows per group here.
    if (ftype == Bodo_FTypes::size) {
        for (int64_t i = 0; i < in_col->length; i++) {
            int64_t i_grp = grp_info.row_to_group[i];
            if (i_grp != -1)
                size_agg<int64_t, Bodo_CTypes::INT64>::apply(
                    getv<ARR_O, int64_t>(out_col, i_grp),
                    getv<ARR_I, int64_t>(in_col, i));
        }
        return;
    }
    if (in_col->arr_type == bodo_array_type::STRING ||
        in_col->arr_type == bodo_array_type::LIST_STRING) {
        switch (ftype) {
            // NOTE: The int template argument is not used in this call to
            // apply_to_column
            case Bodo_FTypes::sum:
                return apply_to_column<ARR_I, ARR_O, int, Bodo_FTypes::sum,
                                       Bodo_CTypes::STRING>(in_col, out_col,
                                                            aux_cols, grp_info);
            case Bodo_FTypes::min:
                return apply_to_column<ARR_I, ARR_O, int, Bodo_FTypes::min,
                                       Bodo_CTypes::STRING>(in_col, out_col,
                                                            aux_cols, grp_info);
            case Bodo_FTypes::max:
                return apply_to_column<ARR_I, ARR_O, int, Bodo_FTypes::max,
                                       Bodo_CTypes::STRING>(in_col, out_col,
                                                            aux_cols, grp_info);
            case Bodo_FTypes::first:
                return apply_to_column<ARR_I, ARR_O, int, Bodo_FTypes::first,
                                       Bodo_CTypes::STRING>(in_col, out_col,
                                                            aux_cols, grp_info);
            case Bodo_FTypes::last:
                return apply_to_column<ARR_I, ARR_O, int, Bodo_FTypes::last,
                                       Bodo_CTypes::STRING>(in_col, out_col,
                                                            aux_cols, grp_info);
        }
    }
    if (ftype == Bodo_FTypes::count) {
        switch (in_col->dtype) {
            case Bodo_CTypes::FLOAT32:
                // data will only be used to check for nans
                return apply_to_column<ARR_I, ARR_O, float, Bodo_FTypes::count,
                                       Bodo_CTypes::FLOAT32>(
                    in_col, out_col, aux_cols, grp_info);
            case Bodo_CTypes::FLOAT64:
                // data will only be used to check for nans
                return apply_to_column<ARR_I, ARR_O, double, Bodo_FTypes::count,
                                       Bodo_CTypes::FLOAT64>(
                    in_col, out_col, aux_cols, grp_info);
            case Bodo_CTypes::DATETIME:
            case Bodo_CTypes::TIMEDELTA:
                // data will only be used to check for NATs
                return apply_to_column<ARR_I, ARR_O, int64_t,
                                       Bodo_FTypes::count,
                                       Bodo_CTypes::DATETIME>(
                    in_col, out_col, aux_cols, grp_info);
            default:
                // data will be ignored in this case, so type doesn't matter
                return apply_to_column<ARR_I, ARR_O, int8_t, Bodo_FTypes::count,
                                       Bodo_CTypes::INT8>(in_col, out_col,
                                                          aux_cols, grp_info);
        }
    }

    switch (in_col->dtype) {
        case Bodo_CTypes::_BOOL:
            switch (ftype) {
                case Bodo_FTypes::min:
                    return apply_to_column<ARR_I, ARR_O, bool, Bodo_FTypes::min,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<ARR_I, ARR_O, bool, Bodo_FTypes::max,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<ARR_I, ARR_O, bool,
                                           Bodo_FTypes::prod,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::first:
                    return apply_to_column<ARR_I, ARR_O, bool,
                                           Bodo_FTypes::first,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::last:
                    return apply_to_column<ARR_I, ARR_O, bool,
                                           Bodo_FTypes::last,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<ARR_I, ARR_O, bool,
                                           Bodo_FTypes::idxmin,
                                           Bodo_CTypes::_BOOL>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<ARR_I, ARR_O, bool,
                                           Bodo_FTypes::idxmax,
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
                    return apply_to_column<ARR_I, ARR_O, int8_t,
                                           Bodo_FTypes::sum, Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<ARR_I, ARR_O, int8_t,
                                           Bodo_FTypes::min, Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<ARR_I, ARR_O, int8_t,
                                           Bodo_FTypes::max, Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<ARR_I, ARR_O, int8_t,
                                           Bodo_FTypes::prod,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::first:
                    return apply_to_column<ARR_I, ARR_O, int8_t,
                                           Bodo_FTypes::first,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::last:
                    return apply_to_column<ARR_I, ARR_O, int8_t,
                                           Bodo_FTypes::last,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<ARR_I, ARR_O, int8_t,
                                           Bodo_FTypes::mean,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<ARR_I, ARR_O, int8_t,
                                           Bodo_FTypes::var, Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<ARR_I, ARR_O, int8_t,
                                           Bodo_FTypes::idxmin,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<ARR_I, ARR_O, int8_t,
                                           Bodo_FTypes::idxmax,
                                           Bodo_CTypes::INT8>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::UINT8:
            switch (ftype) {
                case Bodo_FTypes::sum: {
                    return apply_to_column<ARR_I, ARR_O, uint8_t,
                                           Bodo_FTypes::sum,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info);
                }
                case Bodo_FTypes::min:
                    return apply_to_column<ARR_I, ARR_O, uint8_t,
                                           Bodo_FTypes::min,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<ARR_I, ARR_O, uint8_t,
                                           Bodo_FTypes::max,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<ARR_I, ARR_O, uint8_t,
                                           Bodo_FTypes::prod,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::first:
                    return apply_to_column<ARR_I, ARR_O, uint8_t,
                                           Bodo_FTypes::first,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::last:
                    return apply_to_column<ARR_I, ARR_O, uint8_t,
                                           Bodo_FTypes::last,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<ARR_I, ARR_O, uint8_t,
                                           Bodo_FTypes::mean,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<ARR_I, ARR_O, uint8_t,
                                           Bodo_FTypes::var,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<ARR_I, ARR_O, uint8_t,
                                           Bodo_FTypes::idxmin,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<ARR_I, ARR_O, uint8_t,
                                           Bodo_FTypes::idxmax,
                                           Bodo_CTypes::UINT8>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::INT16:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<ARR_I, ARR_O, int16_t,
                                           Bodo_FTypes::sum,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<ARR_I, ARR_O, int16_t,
                                           Bodo_FTypes::min,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<ARR_I, ARR_O, int16_t,
                                           Bodo_FTypes::max,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<ARR_I, ARR_O, int16_t,
                                           Bodo_FTypes::prod,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::first:
                    return apply_to_column<ARR_I, ARR_O, int16_t,
                                           Bodo_FTypes::first,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::last:
                    return apply_to_column<ARR_I, ARR_O, int16_t,
                                           Bodo_FTypes::last,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<ARR_I, ARR_O, int16_t,
                                           Bodo_FTypes::mean,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<ARR_I, ARR_O, int16_t,
                                           Bodo_FTypes::var,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<ARR_I, ARR_O, int16_t,
                                           Bodo_FTypes::idxmin,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<ARR_I, ARR_O, int16_t,
                                           Bodo_FTypes::idxmax,
                                           Bodo_CTypes::INT16>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::UINT16:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<ARR_I, ARR_O, uint16_t,
                                           Bodo_FTypes::sum,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<ARR_I, ARR_O, uint16_t,
                                           Bodo_FTypes::min,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<ARR_I, ARR_O, uint16_t,
                                           Bodo_FTypes::max,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<ARR_I, ARR_O, uint16_t,
                                           Bodo_FTypes::prod,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::first:
                    return apply_to_column<ARR_I, ARR_O, uint16_t,
                                           Bodo_FTypes::first,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::last:
                    return apply_to_column<ARR_I, ARR_O, uint16_t,
                                           Bodo_FTypes::last,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<ARR_I, ARR_O, uint16_t,
                                           Bodo_FTypes::mean,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<ARR_I, ARR_O, uint16_t,
                                           Bodo_FTypes::var,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<ARR_I, ARR_O, uint16_t,
                                           Bodo_FTypes::idxmin,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<ARR_I, ARR_O, uint16_t,
                                           Bodo_FTypes::idxmax,
                                           Bodo_CTypes::UINT16>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::INT32:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<ARR_I, ARR_O, int32_t,
                                           Bodo_FTypes::sum,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<ARR_I, ARR_O, int32_t,
                                           Bodo_FTypes::min,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<ARR_I, ARR_O, int32_t,
                                           Bodo_FTypes::max,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<ARR_I, ARR_O, int32_t,
                                           Bodo_FTypes::prod,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::first:
                    return apply_to_column<ARR_I, ARR_O, int32_t,
                                           Bodo_FTypes::first,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::last:
                    return apply_to_column<ARR_I, ARR_O, int32_t,
                                           Bodo_FTypes::last,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<ARR_I, ARR_O, int32_t,
                                           Bodo_FTypes::mean,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<ARR_I, ARR_O, int32_t,
                                           Bodo_FTypes::var,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<ARR_I, ARR_O, int32_t,
                                           Bodo_FTypes::idxmin,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<ARR_I, ARR_O, int32_t,
                                           Bodo_FTypes::idxmax,
                                           Bodo_CTypes::INT32>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::UINT32:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<ARR_I, ARR_O, uint32_t,
                                           Bodo_FTypes::sum,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<ARR_I, ARR_O, uint32_t,
                                           Bodo_FTypes::min,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<ARR_I, ARR_O, uint32_t,
                                           Bodo_FTypes::max,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<ARR_I, ARR_O, uint32_t,
                                           Bodo_FTypes::prod,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::first:
                    return apply_to_column<ARR_I, ARR_O, uint32_t,
                                           Bodo_FTypes::first,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::last:
                    return apply_to_column<ARR_I, ARR_O, uint32_t,
                                           Bodo_FTypes::last,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<ARR_I, ARR_O, uint32_t,
                                           Bodo_FTypes::mean,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<ARR_I, ARR_O, uint32_t,
                                           Bodo_FTypes::var,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<ARR_I, ARR_O, uint32_t,
                                           Bodo_FTypes::idxmin,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<ARR_I, ARR_O, uint32_t,
                                           Bodo_FTypes::idxmax,
                                           Bodo_CTypes::UINT32>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::INT64:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::sum,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::min,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::max,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::prod,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::first:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::first,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::last:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::last,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::mean,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::var,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::idxmin,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::idxmax,
                                           Bodo_CTypes::INT64>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::UINT64:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<ARR_I, ARR_O, uint64_t,
                                           Bodo_FTypes::sum,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<ARR_I, ARR_O, uint64_t,
                                           Bodo_FTypes::min,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<ARR_I, ARR_O, uint64_t,
                                           Bodo_FTypes::max,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<ARR_I, ARR_O, uint64_t,
                                           Bodo_FTypes::prod,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::first:
                    return apply_to_column<ARR_I, ARR_O, uint64_t,
                                           Bodo_FTypes::first,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::last:
                    return apply_to_column<ARR_I, ARR_O, uint64_t,
                                           Bodo_FTypes::last,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<ARR_I, ARR_O, uint64_t,
                                           Bodo_FTypes::mean,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<ARR_I, ARR_O, uint64_t,
                                           Bodo_FTypes::var,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<ARR_I, ARR_O, uint64_t,
                                           Bodo_FTypes::idxmin,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<ARR_I, ARR_O, uint64_t,
                                           Bodo_FTypes::idxmax,
                                           Bodo_CTypes::UINT64>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::DATE:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::sum, Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::min, Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::max, Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::prod,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::first:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::first,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::last:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::last,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::mean,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::var, Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::idxmin,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::idxmax,
                                           Bodo_CTypes::DATE>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::DATETIME:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::sum,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::min,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::max,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::prod,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::first:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::first,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::last:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::last,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::mean,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::var,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::idxmin,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::idxmax,
                                           Bodo_CTypes::DATETIME>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::TIMEDELTA:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::sum,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::min,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::max,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::prod,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::first:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::first,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::last:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::last,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::mean,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::var,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::idxmin,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<ARR_I, ARR_O, int64_t,
                                           Bodo_FTypes::idxmax,
                                           Bodo_CTypes::TIMEDELTA>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::FLOAT32:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<ARR_I, ARR_O, float,
                                           Bodo_FTypes::sum,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<ARR_I, ARR_O, float,
                                           Bodo_FTypes::min,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<ARR_I, ARR_O, float,
                                           Bodo_FTypes::max,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<ARR_I, ARR_O, float,
                                           Bodo_FTypes::prod,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::first:
                    return apply_to_column<ARR_I, ARR_O, float,
                                           Bodo_FTypes::first,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::last:
                    return apply_to_column<ARR_I, ARR_O, float,
                                           Bodo_FTypes::last,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<ARR_I, ARR_O, float,
                                           Bodo_FTypes::mean,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean_eval:
                    return apply_to_column<ARR_I, ARR_O, float,
                                           Bodo_FTypes::mean_eval,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<ARR_I, ARR_O, float,
                                           Bodo_FTypes::var,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var_eval:
                    return apply_to_column<ARR_I, ARR_O, float,
                                           Bodo_FTypes::var_eval,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::std_eval:
                    return apply_to_column<ARR_I, ARR_O, float,
                                           Bodo_FTypes::std_eval,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<ARR_I, ARR_O, float,
                                           Bodo_FTypes::idxmin,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<ARR_I, ARR_O, float,
                                           Bodo_FTypes::idxmax,
                                           Bodo_CTypes::FLOAT32>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::FLOAT64:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<ARR_I, ARR_O, double,
                                           Bodo_FTypes::sum,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<ARR_I, ARR_O, double,
                                           Bodo_FTypes::min,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<ARR_I, ARR_O, double,
                                           Bodo_FTypes::max,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<ARR_I, ARR_O, double,
                                           Bodo_FTypes::prod,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::first:
                    return apply_to_column<ARR_I, ARR_O, double,
                                           Bodo_FTypes::first,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::last:
                    return apply_to_column<ARR_I, ARR_O, double,
                                           Bodo_FTypes::last,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<ARR_I, ARR_O, double,
                                           Bodo_FTypes::mean,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean_eval:
                    return apply_to_column<ARR_I, ARR_O, double,
                                           Bodo_FTypes::mean_eval,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<ARR_I, ARR_O, double,
                                           Bodo_FTypes::var,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var_eval:
                    return apply_to_column<ARR_I, ARR_O, double,
                                           Bodo_FTypes::var_eval,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::std_eval:
                    return apply_to_column<ARR_I, ARR_O, double,
                                           Bodo_FTypes::std_eval,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<ARR_I, ARR_O, double,
                                           Bodo_FTypes::idxmin,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<ARR_I, ARR_O, double,
                                           Bodo_FTypes::idxmax,
                                           Bodo_CTypes::FLOAT64>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::DECIMAL:
            switch (ftype) {
                case Bodo_FTypes::first:
                    return apply_to_column<ARR_I, ARR_O, decimal_value_cpp,
                                           Bodo_FTypes::first,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::last:
                    return apply_to_column<ARR_I, ARR_O, decimal_value_cpp,
                                           Bodo_FTypes::last,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<ARR_I, ARR_O, decimal_value_cpp,
                                           Bodo_FTypes::min,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<ARR_I, ARR_O, decimal_value_cpp,
                                           Bodo_FTypes::max,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<ARR_I, ARR_O, decimal_value_cpp,
                                           Bodo_FTypes::mean,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean_eval:
                    return apply_to_column<ARR_I, ARR_O, decimal_value_cpp,
                                           Bodo_FTypes::mean_eval,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<ARR_I, ARR_O, decimal_value_cpp,
                                           Bodo_FTypes::var,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var_eval:
                    return apply_to_column<ARR_I, ARR_O, decimal_value_cpp,
                                           Bodo_FTypes::var_eval,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::std_eval:
                    return apply_to_column<ARR_I, ARR_O, decimal_value_cpp,
                                           Bodo_FTypes::std_eval,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmin:
                    return apply_to_column<ARR_I, ARR_O, decimal_value_cpp,
                                           Bodo_FTypes::idxmin,
                                           Bodo_CTypes::DECIMAL>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::idxmax:
                    return apply_to_column<ARR_I, ARR_O, decimal_value_cpp,
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
void aggfunc_output_initialize_kernel(array_info* out_col, int ftype,
                                      bool is_groupby) {
    if (out_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        bool init_val;
        if (ftype == Bodo_FTypes::min || ftype == Bodo_FTypes::max ||
            ftype == Bodo_FTypes::first || ftype == Bodo_FTypes::last) {
            // if input is all nulls, max, min, first and last output will be
            // null
            if (is_groupby) {
                init_val = false;
            } else {
                init_val = true;
            }
        } else {
            init_val = true;
        }
        InitializeBitMask((uint8_t*)out_col->null_bitmask, out_col->length,
                          init_val);
    }
    if (out_col->arr_type == bodo_array_type::STRING ||
        out_col->arr_type == bodo_array_type::LIST_STRING) {
        InitializeBitMask((uint8_t*)out_col->null_bitmask, out_col->length,
                          false);
    }
    if (out_col->arr_type == bodo_array_type::CATEGORICAL) {
        if (ftype == Bodo_FTypes::min || ftype == Bodo_FTypes::max ||
            ftype == Bodo_FTypes::first || ftype == Bodo_FTypes::last) {
            int init_val = -1;
            // if input is all nulls, max, first and last output will be -1
            // min will be num of categories
            if (ftype == Bodo_FTypes::min) {
                init_val = out_col->num_categories;
            }
            switch (out_col->dtype) {
                case Bodo_CTypes::INT8:
                    std::fill((int8_t*)out_col->data1,
                              (int8_t*)out_col->data1 + out_col->length,
                              init_val);
                    return;
                case Bodo_CTypes::INT16:
                    std::fill((int16_t*)out_col->data1,
                              (int16_t*)out_col->data1 + out_col->length,
                              init_val);
                    return;
                case Bodo_CTypes::INT32:
                    std::fill((int32_t*)out_col->data1,
                              (int32_t*)out_col->data1 + out_col->length,
                              init_val);
                    return;
                case Bodo_CTypes::INT64:
                    std::fill((int64_t*)out_col->data1,
                              (int64_t*)out_col->data1 + out_col->length,
                              init_val);
                    return;
            }
        }
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
                case Bodo_CTypes::BINARY:
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
                case Bodo_CTypes::BINARY:
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
                case Bodo_CTypes::BINARY:
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

void aggfunc_output_initialize(array_info* out_col, int ftype) {
    bool is_groupby = true;
    aggfunc_output_initialize_kernel(out_col, ftype, is_groupby);
}

void aggfunc_output_initialize(multiple_array_info* out_col, int ftype) {
    bool is_groupby = false;
    for (array_info* eout : out_col->vect_arr)
        aggfunc_output_initialize_kernel(eout, ftype, is_groupby);
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
template <typename ARRAY>
typename std::enable_if<!is_multiple_array<ARRAY>::value, void>::type
get_groupby_output_dtype(int ftype, bodo_array_type::arr_type_enum& array_type,
                         Bodo_CTypes::CTypeEnum& dtype, bool is_key,
                         bool is_combine, bool is_crosstab) {
    if (is_combine) ftype = combine_funcs[ftype];
    if (is_key) return;
    switch (ftype) {
        case Bodo_FTypes::nunique:
        case Bodo_FTypes::count:
        case Bodo_FTypes::size:
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
template <typename ARRAY>
typename std::enable_if<is_multiple_array<ARRAY>::value, void>::type
get_groupby_output_dtype(int ftype, bodo_array_type::arr_type_enum& array_type,
                         Bodo_CTypes::CTypeEnum& dtype, bool is_key,
                         bool is_combine, bool is_crosstab) {
    int input_ftype = ftype;
    if (is_combine) ftype = combine_funcs[ftype];
    if (is_key) return;
    switch (ftype) {
        case Bodo_FTypes::nunique:
        case Bodo_FTypes::size:
        case Bodo_FTypes::count: {
            if (is_crosstab) {
                array_type = bodo_array_type::NUMPY;
            } else {
                array_type = bodo_array_type::NULLABLE_INT_BOOL;
            }
            dtype = Bodo_CTypes::INT64;
            return;
        }
        case Bodo_FTypes::max:
        case Bodo_FTypes::min:
        case Bodo_FTypes::sum:
        case Bodo_FTypes::prod: {
            bool is_float =
                dtype == Bodo_CTypes::FLOAT64 || dtype == Bodo_CTypes::FLOAT32;
            if (ftype == input_ftype && !is_float) {
                array_type = bodo_array_type::NULLABLE_INT_BOOL;
            }
            return;
        }
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

template <typename ARRAY>
inline typename std::enable_if<!is_multiple_array<ARRAY>::value, ARRAY*>::type
alloc_array_groupby(int64_t length, int64_t n_sub_elems,
                    int64_t n_sub_sub_elems,
                    bodo_array_type::arr_type_enum arr_type,
                    Bodo_CTypes::CTypeEnum dtype, int64_t extra_null_bytes,
                    int64_t num_categories, int64_t n_pivot) {
    return alloc_array(length, n_sub_elems, n_sub_sub_elems, arr_type, dtype,
                       extra_null_bytes, num_categories);
}

template <typename ARRAY>
inline typename std::enable_if<is_multiple_array<ARRAY>::value, ARRAY*>::type
alloc_array_groupby(int64_t length, int64_t n_sub_elems,
                    int64_t n_sub_sub_elems,
                    bodo_array_type::arr_type_enum arr_type,
                    Bodo_CTypes::CTypeEnum dtype, int64_t extra_null_bytes,
                    int64_t num_categories, int64_t n_pivot) {
    std::vector<array_info*> vect_arr;
    for (int i_pivot = 0; i_pivot < n_pivot; i_pivot++)
        vect_arr.push_back(alloc_array(length, n_sub_elems, n_sub_sub_elems,
                                       arr_type, dtype, extra_null_bytes,
                                       num_categories));
    return new multiple_array_info(vect_arr);
}

template <typename ARRAY>
inline typename std::enable_if<is_multiple_array<ARRAY>::value, void>::type
free_array_groupby(ARRAY* arr) {
    for (auto& e_arr : arr->vect_arr) delete_info_decref_array(e_arr);
    for (auto& e_arr : arr->vect_access) delete_info_decref_array(e_arr);
    delete arr;
}

template <typename ARRAY>
inline typename std::enable_if<!is_multiple_array<ARRAY>::value, void>::type
free_array_groupby(ARRAY* arr) {
    delete_info_decref_array(arr);
}

template <typename ARRAY>
inline typename std::enable_if<is_multiple_array<ARRAY>::value, ARRAY*>::type
get_array_from_iterator(typename std::vector<array_info*>::iterator& it,
                        int n_pivot) {
    std::vector<array_info*> vect_arr;
    for (int i_pivot = 0; i_pivot < n_pivot; i_pivot++) {
        vect_arr.push_back(*it);
        it++;
    }
    int n_access = (n_pivot + 7) >> 3;
    std::vector<array_info*> vect_access;
    for (int i_access = 0; i_access < n_access; i_access++) {
        vect_access.push_back(*it);
        it++;
    }
    return new multiple_array_info(vect_arr, vect_access);
}

template <typename ARRAY>
inline typename std::enable_if<!is_multiple_array<ARRAY>::value, ARRAY*>::type
get_array_from_iterator(typename std::vector<array_info*>::iterator& it,
                        int n_pivot) {
    array_info* arr = *it;
    it++;
    return arr;
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
template <typename ARRAY>
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
    virtual void alloc_update_columns(size_t num_groups, size_t n_pivot,
                                      bool is_crosstab,
                                      std::vector<ARRAY*>& out_cols) {
        bodo_array_type::arr_type_enum arr_type = in_col->arr_type;
        Bodo_CTypes::CTypeEnum dtype = in_col->dtype;
        int64_t num_categories = in_col->num_categories;
        // calling this modifies arr_type and dtype
        bool is_combine = false;
        get_groupby_output_dtype<ARRAY>(ftype, arr_type, dtype, false,
                                        is_combine, is_crosstab);
        out_cols.push_back(alloc_array_groupby<ARRAY>(
            num_groups, 1, 1, arr_type, dtype, 0, num_categories, n_pivot));
        update_cols.push_back(out_cols.back());
    }

    /**
     * Perform update step for this column set. This will fill my columns with
     * the result of the aggregation operation corresponding to this column set
     * @param grouping info calculated by GroupbyPipeline
     */
    virtual void update(const std::vector<grouping_info>& grp_infos) {
        std::vector<ARRAY*> aux_cols;
        aggfunc_output_initialize(update_cols[0], ftype);
        do_apply_to_column(in_col, update_cols[0], aux_cols, grp_infos[0],
                           ftype);
    }

    /**
     * When GroupbyPipeline shuffles the table after update, the column set
     * needs to be updated with the columns from the new shuffled table. This
     * method is called by GroupbyPipeline with an iterator pointing to my
     * first column. The column set will update its columns and return an
     * iterator pointing to the next set of columns.
     * @param iterator pointing to the first column in this column set
     */
    virtual typename std::vector<array_info*>::iterator update_after_shuffle(
        typename std::vector<array_info*>::iterator& it, int n_pivot) {
        for (size_t i_col = 0; i_col < update_cols.size(); i_col++)
            update_cols[i_col] = get_array_from_iterator<ARRAY>(it, n_pivot);
        return it;
    }

    /**
     * Allocate my columns for combine step.
     * @param number of groups found in the input table (which is the update
     * table)
     * @param[in,out] vector of columns of combine table. This method adds
     *                columns to this vector.
     */
    virtual void alloc_combine_columns(size_t num_groups, size_t n_pivot,
                                       bool is_crosstab,
                                       std::vector<ARRAY*>& out_cols) {
        for (auto col : update_cols) {
            bodo_array_type::arr_type_enum arr_type = col->arr_type;
            Bodo_CTypes::CTypeEnum dtype = col->dtype;
            int64_t num_categories = col->num_categories;
            // calling this modifies arr_type and dtype
            bool is_combine = true;
            get_groupby_output_dtype<ARRAY>(ftype, arr_type, dtype, false,
                                            is_combine, is_crosstab);
            out_cols.push_back(alloc_array_groupby<ARRAY>(
                num_groups, 1, 1, arr_type, dtype, 0, num_categories, n_pivot));
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
        std::vector<ARRAY*> aux_cols(combine_cols.begin() + 1,
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
    virtual ARRAY* getOutputColumn() {
        std::vector<ARRAY*>* mycols;
        if (combine_step)
            mycols = &combine_cols;
        else
            mycols = &update_cols;
        ARRAY* out_col = mycols->at(0);
        for (auto it = mycols->begin() + 1; it != mycols->end(); it++) {
            ARRAY* a = *it;
            free_array_groupby(a);
        }
        return out_col;
    }

   protected:
    array_info* in_col;  // the input column (from groupby input table) to which
                         // this column set corresponds to
    int ftype;
    bool combine_step;  // GroupbyPipeline is going to perform a combine
                        // operation or not
    std::vector<ARRAY*> update_cols;   // columns for update step
    std::vector<ARRAY*> combine_cols;  // columns for combine step
};

template <typename ARRAY>
class MeanColSet : public BasicColSet<ARRAY> {
   public:
    MeanColSet(array_info* in_col, bool combine_step)
        : BasicColSet<ARRAY>(in_col, Bodo_FTypes::mean, combine_step) {}
    virtual ~MeanColSet() {}

    virtual void alloc_update_columns(size_t num_groups, size_t n_pivot,
                                      bool is_crosstab,
                                      std::vector<ARRAY*>& out_cols) {
        ARRAY* c1 = alloc_array_groupby<ARRAY>(
            num_groups, 1, 1, bodo_array_type::NUMPY, Bodo_CTypes::FLOAT64, 0,
            0, n_pivot);  // for sum and result
        ARRAY* c2 = alloc_array_groupby<ARRAY>(
            num_groups, 1, 1, bodo_array_type::NUMPY, Bodo_CTypes::UINT64, 0, 0,
            n_pivot);  // for counts
        out_cols.push_back(c1);
        out_cols.push_back(c2);
        this->update_cols.push_back(c1);
        this->update_cols.push_back(c2);
    }

    virtual void update(const std::vector<grouping_info>& grp_infos) {
        std::vector<ARRAY*> aux_cols = {this->update_cols[1]};
        aggfunc_output_initialize(this->update_cols[0], this->ftype);
        aggfunc_output_initialize(this->update_cols[1], this->ftype);
        do_apply_to_column(this->in_col, this->update_cols[0], aux_cols,
                           grp_infos[0], this->ftype);
    }

    virtual void combine(const grouping_info& grp_info) {
        std::vector<ARRAY*> aux_cols;
        aggfunc_output_initialize(this->combine_cols[0], Bodo_FTypes::sum);
        aggfunc_output_initialize(this->combine_cols[1], Bodo_FTypes::sum);
        do_apply_to_column(this->update_cols[0], this->combine_cols[0],
                           aux_cols, grp_info, Bodo_FTypes::sum);
        do_apply_to_column(this->update_cols[1], this->combine_cols[1],
                           aux_cols, grp_info, Bodo_FTypes::sum);
    }

    virtual void eval(const grouping_info& grp_info) {
        std::vector<ARRAY*> aux_cols;
        if (this->combine_step)
            do_apply_to_column(this->combine_cols[1], this->combine_cols[0],
                               aux_cols, grp_info, Bodo_FTypes::mean_eval);
        else
            do_apply_to_column(this->update_cols[1], this->update_cols[0],
                               aux_cols, grp_info, Bodo_FTypes::mean_eval);
    }
};

template <typename ARRAY>
typename std::enable_if<!is_multiple_array<ARRAY>::value, ARRAY*>::type
RetrieveArray_SingleColumn_ARRAY(array_info* index_col, ARRAY* index_pos) {
    return RetrieveArray_SingleColumn_arr(index_col, index_pos);
}

template <typename ARRAY>
typename std::enable_if<is_multiple_array<ARRAY>::value, ARRAY*>::type
RetrieveArray_SingleColumn_ARRAY(array_info* index_col, ARRAY* index_pos) {
    std::vector<array_info*> vect_arr;
    for (auto& e_arr : index_pos->vect_arr)
        vect_arr.push_back(RetrieveArray_SingleColumn_arr(index_col, e_arr));
    return new multiple_array_info(vect_arr);
}

template <typename ARRAY>
typename std::enable_if<!is_multiple_array<ARRAY>::value, ARRAY*>::type
RetrieveArray_SingleColumn_Multiple_ARRAY(ARRAY* index_col, ARRAY* index_pos) {
    return RetrieveArray_SingleColumn_arr(index_col, index_pos);
}

template <typename ARRAY>
typename std::enable_if<is_multiple_array<ARRAY>::value, ARRAY*>::type
RetrieveArray_SingleColumn_Multiple_ARRAY(ARRAY* index_col, ARRAY* index_pos) {
    Bodo_PyErr_SetString(
        PyExc_RuntimeError,
        "The code is missing in RetrieveArray_SingleColumn_Multiple_ARRAY");
    return nullptr;
}

template <typename ARRAY>
class IdxMinMaxColSet : public BasicColSet<ARRAY> {
   public:
    IdxMinMaxColSet(array_info* in_col, array_info* _index_col, int ftype,
                    bool combine_step)
        : BasicColSet<ARRAY>(in_col, ftype, combine_step),
          index_col(_index_col) {}
    virtual ~IdxMinMaxColSet() {}

    virtual void alloc_update_columns(size_t num_groups, size_t n_pivot,
                                      bool is_crosstab,
                                      std::vector<ARRAY*>& out_cols) {
        // output column containing index values. dummy for now. will be
        // assigned the real data at the end of update()
        ARRAY* out_col =
            alloc_array_groupby<ARRAY>(num_groups, 1, 1, index_col->arr_type,
                                       index_col->dtype, 0, 0, n_pivot);
        // create array to store min/max value
        ARRAY* max_col = alloc_array_groupby<ARRAY>(
            num_groups, 1, 1, this->in_col->arr_type, this->in_col->dtype, 0, 0,
            n_pivot);  // for min/max
        // create array to store index position of min/max value
        ARRAY* index_pos_col =
            alloc_array_groupby<ARRAY>(num_groups, 1, 1, bodo_array_type::NUMPY,
                                       Bodo_CTypes::UINT64, 0, 0, n_pivot);
        out_cols.push_back(out_col);
        out_cols.push_back(max_col);
        this->update_cols.push_back(out_col);
        this->update_cols.push_back(max_col);
        this->update_cols.push_back(index_pos_col);
    }

    virtual void update(const std::vector<grouping_info>& grp_infos) {
        ARRAY* index_pos_col = this->update_cols[2];
        std::vector<ARRAY*> aux_cols = {index_pos_col};
        if (this->ftype == Bodo_FTypes::idxmax)
            aggfunc_output_initialize(this->update_cols[1], Bodo_FTypes::max);
        if (this->ftype == Bodo_FTypes::idxmin)
            aggfunc_output_initialize(this->update_cols[1], Bodo_FTypes::min);
        aggfunc_output_initialize(index_pos_col,
                                  Bodo_FTypes::count);  // zero init
        do_apply_to_column(this->in_col, this->update_cols[1], aux_cols,
                           grp_infos[0], this->ftype);

        ARRAY* real_out_col =
            RetrieveArray_SingleColumn_ARRAY(index_col, index_pos_col);
        ARRAY* out_col = this->update_cols[0];
        *out_col = std::move(*real_out_col);
        delete real_out_col;
        free_array_groupby(index_pos_col);
        this->update_cols.pop_back();
    }

    virtual void alloc_combine_columns(size_t num_groups, size_t n_pivot,
                                       bool is_crosstab,
                                       std::vector<ARRAY*>& out_cols) {
        // output column containing index values. dummy for now. will be
        // assigned the real data at the end of combine()
        ARRAY* out_col =
            alloc_array_groupby<ARRAY>(num_groups, 1, 1, index_col->arr_type,
                                       index_col->dtype, 0, 0, n_pivot);
        // create array to store min/max value
        ARRAY* max_col = alloc_array_groupby<ARRAY>(
            num_groups, 1, 1, this->in_col->arr_type, this->in_col->dtype, 0, 0,
            n_pivot);  // for min/max
        // create array to store index position of min/max value
        ARRAY* index_pos_col =
            alloc_array_groupby<ARRAY>(num_groups, 1, 1, bodo_array_type::NUMPY,
                                       Bodo_CTypes::UINT64, 0, 0, n_pivot);
        out_cols.push_back(out_col);
        out_cols.push_back(max_col);
        this->combine_cols.push_back(out_col);
        this->combine_cols.push_back(max_col);
        this->combine_cols.push_back(index_pos_col);
    }

    virtual void combine(const grouping_info& grp_info) {
        ARRAY* index_pos_col = this->combine_cols[2];
        std::vector<ARRAY*> aux_cols = {index_pos_col};
        if (this->ftype == Bodo_FTypes::idxmax)
            aggfunc_output_initialize(this->combine_cols[1], Bodo_FTypes::max);
        if (this->ftype == Bodo_FTypes::idxmin)
            aggfunc_output_initialize(this->combine_cols[1], Bodo_FTypes::min);
        aggfunc_output_initialize(index_pos_col,
                                  Bodo_FTypes::count);  // zero init
        do_apply_to_column(this->update_cols[1], this->combine_cols[1],
                           aux_cols, grp_info, this->ftype);

        ARRAY* real_out_col = RetrieveArray_SingleColumn_Multiple_ARRAY(
            this->update_cols[0], index_pos_col);
        ARRAY* out_col = this->combine_cols[0];
        *out_col = std::move(*real_out_col);
        delete real_out_col;
        free_array_groupby(index_pos_col);
        this->combine_cols.pop_back();
    }

   private:
    array_info* index_col;
};

template <typename ARRAY>
class VarStdColSet : public BasicColSet<ARRAY> {
   public:
    VarStdColSet(array_info* in_col, int ftype, bool combine_step)
        : BasicColSet<ARRAY>(in_col, ftype, combine_step) {}
    virtual ~VarStdColSet() {}

    virtual void alloc_update_columns(size_t num_groups, size_t n_pivot,
                                      bool is_crosstab,
                                      std::vector<ARRAY*>& out_cols) {
        if (!this->combine_step) {
            // need to create output column now
            ARRAY* col = alloc_array_groupby<ARRAY>(
                num_groups, 1, 1, bodo_array_type::NUMPY, Bodo_CTypes::FLOAT64,
                0, 0, n_pivot);  // for result
            out_cols.push_back(col);
            this->update_cols.push_back(col);
        }
        ARRAY* count_col =
            alloc_array_groupby<ARRAY>(num_groups, 1, 1, bodo_array_type::NUMPY,
                                       Bodo_CTypes::UINT64, 0, 0, n_pivot);
        ARRAY* mean_col =
            alloc_array_groupby<ARRAY>(num_groups, 1, 1, bodo_array_type::NUMPY,
                                       Bodo_CTypes::FLOAT64, 0, 0, n_pivot);
        ARRAY* m2_col =
            alloc_array_groupby<ARRAY>(num_groups, 1, 1, bodo_array_type::NUMPY,
                                       Bodo_CTypes::FLOAT64, 0, 0, n_pivot);
        aggfunc_output_initialize(count_col,
                                  Bodo_FTypes::count);  // zero initialize
        aggfunc_output_initialize(mean_col,
                                  Bodo_FTypes::count);  // zero initialize
        aggfunc_output_initialize(m2_col,
                                  Bodo_FTypes::count);  // zero initialize
        out_cols.push_back(count_col);
        out_cols.push_back(mean_col);
        out_cols.push_back(m2_col);
        this->update_cols.push_back(count_col);
        this->update_cols.push_back(mean_col);
        this->update_cols.push_back(m2_col);
    }

    virtual void update(const std::vector<grouping_info>& grp_infos) {
        if (!this->combine_step) {
            std::vector<ARRAY*> aux_cols = {this->update_cols[1],
                                            this->update_cols[2],
                                            this->update_cols[3]};
            do_apply_to_column(this->in_col, this->update_cols[1], aux_cols,
                               grp_infos[0], this->ftype);
        } else {
            std::vector<ARRAY*> aux_cols = {this->update_cols[0],
                                            this->update_cols[1],
                                            this->update_cols[2]};
            do_apply_to_column(this->in_col, this->update_cols[0], aux_cols,
                               grp_infos[0], this->ftype);
        }
    }

    virtual void alloc_combine_columns(size_t num_groups, size_t n_pivot,
                                       bool is_crosstab,
                                       std::vector<ARRAY*>& out_cols) {
        ARRAY* col = alloc_array_groupby<ARRAY>(
            num_groups, 1, 1, bodo_array_type::NUMPY, Bodo_CTypes::FLOAT64, 0,
            0, n_pivot);  // for result
        out_cols.push_back(col);
        this->combine_cols.push_back(col);
        BasicColSet<ARRAY>::alloc_combine_columns(num_groups, n_pivot,
                                                  is_crosstab, out_cols);
    }

    virtual void combine(const grouping_info& grp_info) {
        ARRAY* count_col_in = this->update_cols[0];
        ARRAY* mean_col_in = this->update_cols[1];
        ARRAY* m2_col_in = this->update_cols[2];
        ARRAY* count_col_out = this->combine_cols[1];
        ARRAY* mean_col_out = this->combine_cols[2];
        ARRAY* m2_col_out = this->combine_cols[3];
        aggfunc_output_initialize(count_col_out, Bodo_FTypes::count);
        aggfunc_output_initialize(mean_col_out, Bodo_FTypes::count);
        aggfunc_output_initialize(m2_col_out, Bodo_FTypes::count);
        var_combine(count_col_in, mean_col_in, m2_col_in, count_col_out,
                    mean_col_out, m2_col_out, grp_info);
    }

    virtual void eval(const grouping_info& grp_info) {
        std::vector<ARRAY*>* mycols;
        if (this->combine_step)
            mycols = &this->combine_cols;
        else
            mycols = &this->update_cols;

        std::vector<ARRAY*> aux_cols = {mycols->at(1), mycols->at(2),
                                        mycols->at(3)};
        if (this->ftype == Bodo_FTypes::var)
            do_apply_to_column(mycols->at(0), mycols->at(0), aux_cols, grp_info,
                               Bodo_FTypes::var_eval);
        else
            do_apply_to_column(mycols->at(0), mycols->at(0), aux_cols, grp_info,
                               Bodo_FTypes::std_eval);
    }
};

template <typename ARRAY>
class UdfColSet : public BasicColSet<ARRAY> {
   public:
    UdfColSet(array_info* in_col, bool combine_step, table_info* udf_table,
              int udf_table_idx, int n_redvars)
        : BasicColSet<ARRAY>(in_col, Bodo_FTypes::udf, combine_step),
          udf_table(udf_table),
          udf_table_idx(udf_table_idx),
          n_redvars(n_redvars) {}
    virtual ~UdfColSet() {}

    virtual void alloc_update_columns(size_t num_groups, size_t n_pivot,
                                      bool is_crosstab,
                                      std::vector<ARRAY*>& out_cols) {
        int offset = 0;
        if (this->combine_step) offset = 1;
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
            out_cols.push_back(alloc_array_groupby<ARRAY>(
                num_groups, 1, 1, arr_type, dtype, 0, num_categories, n_pivot));
            if (!this->combine_step)
                this->update_cols.push_back(out_cols.back());
        }
    }

    virtual void update(const std::vector<grouping_info>& grp_infos) {
        // do nothing because this is done in JIT-compiled code (invoked from
        // GroupbyPipeline once for all udf columns sets)
    }

    virtual typename std::vector<array_info*>::iterator update_after_shuffle(
        typename std::vector<array_info*>::iterator& it, int n_pivot) {
        // UdfColSet doesn't keep the update cols, return the updated iterator
        return it + n_pivot * n_redvars;
    }

    virtual void alloc_combine_columns(size_t num_groups, size_t n_pivot,
                                       bool is_crosstab,
                                       std::vector<ARRAY*>& out_cols) {
        for (int i = udf_table_idx; i < udf_table_idx + 1 + n_redvars; i++) {
            // we get the type from the udf dummy table that was passed to C++
            // library
            bodo_array_type::arr_type_enum arr_type =
                udf_table->columns[i]->arr_type;
            Bodo_CTypes::CTypeEnum dtype = udf_table->columns[i]->dtype;
            int64_t num_categories = udf_table->columns[i]->num_categories;
            out_cols.push_back(alloc_array_groupby<ARRAY>(
                num_groups, 1, 1, arr_type, dtype, 0, num_categories, n_pivot));
            this->combine_cols.push_back(out_cols.back());
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

template <typename ARRAY>
class GeneralUdfColSet : public UdfColSet<ARRAY> {
   public:
    GeneralUdfColSet(array_info* in_col, table_info* udf_table,
                     int udf_table_idx)
        : UdfColSet<ARRAY>(in_col, false, udf_table, udf_table_idx, 0) {}
    virtual ~GeneralUdfColSet() {}

    /**
     * Fill in the input table for general UDF cfunc. See udf_general_fn
     * and aggregate.py::gen_general_udf_cb for more information.
     */
    void fill_in_columns(table_info* general_in_table,
                         const grouping_info& grp_info) const {
        array_info* in_col = this->in_col;
        std::vector<std::vector<int64_t>> group_rows(grp_info.num_groups);
        // get the rows in each group
        for (size_t i = 0; i < in_col->length; i++) {
            int64_t i_grp = grp_info.row_to_group[i];
            group_rows[i_grp].push_back(i);
        }
        // retrieve one column per group from the input column, add it to the
        // general UDF input table
        for (int64_t i = 0; i < grp_info.num_groups; i++) {
            array_info* col = RetrieveArray_SingleColumn(in_col, group_rows[i]);
            general_in_table->columns.push_back(col);
        }
    }
};

template <typename ARRAY>
class MedianColSet : public BasicColSet<ARRAY> {
   public:
    MedianColSet(array_info* in_col, bool _skipna)
        : BasicColSet<ARRAY>(in_col, Bodo_FTypes::median, false),
          skipna(_skipna) {}
    virtual ~MedianColSet() {}

    virtual void update(const std::vector<grouping_info>& grp_infos) {
        median_computation<ARRAY>(this->in_col, this->update_cols[0],
                                  grp_infos[0], this->skipna);
    }

   private:
    bool skipna;
};

template <typename ARRAY>
class NUniqueColSet : public BasicColSet<ARRAY> {
   public:
    NUniqueColSet(array_info* in_col, bool _dropna, table_info* nunique_table,
                  bool do_combine, bool _is_parallel)
        : BasicColSet<ARRAY>(in_col, Bodo_FTypes::nunique, do_combine),
          dropna(_dropna),
          is_parallel(_is_parallel),
          my_nunique_table(nunique_table) {}

    virtual ~NUniqueColSet() {
        if (my_nunique_table != nullptr)
            delete_table_decref_arrays(my_nunique_table);
    }

    virtual void update(const std::vector<grouping_info>& grp_infos) {
        // TODO: check nunique with pivot_table operation
        if (my_nunique_table != nullptr) {
            // use the grouping_info that corresponds to my nunique table
            aggfunc_output_initialize(this->update_cols[0],
                                      Bodo_FTypes::sum);  // zero initialize
            nunique_computation<ARRAY>(this->in_col, this->update_cols[0],
                                       grp_infos[my_nunique_table->id], dropna,
                                       is_parallel);
        } else {
            // use default grouping_info
            nunique_computation<ARRAY>(this->in_col, this->update_cols[0],
                                       grp_infos[0], dropna, is_parallel);
        }
    }

   private:
    bool dropna;
    table_info* my_nunique_table = nullptr;
    bool is_parallel;
};

template <typename ARRAY>
class CumOpColSet : public BasicColSet<ARRAY> {
   public:
    CumOpColSet(array_info* in_col, int ftype, bool _skipna)
        : BasicColSet<ARRAY>(in_col, ftype, false), skipna(_skipna) {}
    virtual ~CumOpColSet() {}

    virtual void alloc_update_columns(size_t num_groups, size_t n_pivot,
                                      bool is_crosstab,
                                      std::vector<ARRAY*>& out_cols) {
        // NOTE: output size of cum ops is the same as input size
        //       (NOT the number of groups)
        out_cols.push_back(alloc_array_groupby<ARRAY>(
            this->in_col->length, 1, 1, this->in_col->arr_type,
            this->in_col->dtype, 0, this->in_col->num_categories, n_pivot));
        this->update_cols.push_back(out_cols.back());
    }

    virtual void update(const std::vector<grouping_info>& grp_infos) {
        cumulative_computation<ARRAY>(this->in_col, this->update_cols[0],
                                      grp_infos[0], this->ftype, this->skipna);
    }

   private:
    bool skipna;
};

template <typename ARRAY>
class ShiftColSet : public BasicColSet<ARRAY> {
   public:
    ShiftColSet(array_info* in_col, int ftype, int64_t _periods)
        : BasicColSet<ARRAY>(in_col, ftype, false), periods(_periods) {}
    virtual ~ShiftColSet() {}

    virtual void alloc_update_columns(size_t num_groups, size_t n_pivot,
                                      bool is_crosstab,
                                      std::vector<ARRAY*>& out_cols) {
        // NOTE: output size of shift is the same as input size
        //       (NOT the number of groups)
        out_cols.push_back(alloc_array_groupby<ARRAY>(
            this->in_col->length, 1, 1, this->in_col->arr_type,
            this->in_col->dtype, 0, this->in_col->num_categories, n_pivot));
        this->update_cols.push_back(out_cols.back());
    }

    virtual void update(const std::vector<grouping_info>& grp_infos) {
        shift_computation<ARRAY>(this->in_col, this->update_cols[0],
                                 grp_infos[0], this->periods);
    }

   private:
    int64_t periods;
};

/**
 * Copy nullable value from tmp_col to all the rows in the
 * corresponding group update_col.
 * @param update_col[out] output column
 * @param tmp_col[in] input column (one value per group)
 * @param grouping_info[in] structures used to get rows for each group
 *
 */
template <typename ARRAY, typename T>
typename std::enable_if<!(is_multiple_array<ARRAY>::value), void>::type
copy_nullable_values_transform(ARRAY* update_col, ARRAY* tmp_col,
                               const grouping_info& grp_info) {
    int64_t nrows = update_col->length;
    bool bit = false;
    for (int64_t iRow = 0; iRow < nrows; iRow++) {
        int64_t igrp = grp_info.row_to_group[iRow];
        bit = tmp_col->get_null_bit(igrp);
        T val = tmp_col->template at<T>(igrp);
        update_col->set_null_bit(iRow, bit);
        update_col->template at<T>(iRow) = val;
    }
}
template <typename ARRAY, typename T>
typename std::enable_if<(is_multiple_array<ARRAY>::value), void>::type
copy_nullable_values_transform(ARRAY* update_col, ARRAY* tmp_col,
                               const grouping_info& grp_info) {
    Bodo_PyErr_SetString(PyExc_RuntimeError,
                         "copy_nullable_values_transform multiple_array_info "
                         "NULLABLE not supported yet.");
}
/**
 * Propagate value from the row in the tmp_col to all the rows in the
 * group update_col.
 * @param update_col[out]: column that has the final result for all rows
 * @param tmp_col[in]: column that has the result per group
 * @param grouping_info[in]: structures used to get rows for each group
 *
 * */
template <typename ARRAY>
typename std::enable_if<!(is_multiple_array<ARRAY>::value), void>::type
copy_string_values_transform(ARRAY* update_col, ARRAY* tmp_col,
                             const grouping_info& grp_info) {
    int64_t num_groups = grp_info.num_groups;
    array_info* out_arr = NULL;
    // first we have to deal with offsets first so we
    // need one first loop to determine the needed length. In the second
    // loop, the assignation is made. If the entries are missing then the
    // bitmask is set to false.
    bodo_array_type::arr_type_enum arr_type = tmp_col->arr_type;
    Bodo_CTypes::CTypeEnum dtype = tmp_col->dtype;
    int64_t n_chars = 0;
    int64_t nRowOut = update_col->length;
    // Store size of data per row
    std::vector<offset_t> ListSizes(nRowOut);
    offset_t* in_offsets = (offset_t*)tmp_col->data2;
    char* in_data1 = tmp_col->data1;
    // 1. Determine needed length (total number of characters)
    // and number of characters per element/row
    // All rows in same group gets same data
    for (int64_t igrp = 0; igrp < num_groups; igrp++) {
        offset_t size = 0;
        offset_t start_offset = in_offsets[igrp];
        offset_t end_offset = in_offsets[igrp + 1];
        size = end_offset - start_offset;
        int64_t idx = grp_info.group_to_first_row[igrp];
        while (true) {
            if (idx == -1) break;
            ListSizes[idx] = size;
            n_chars += size;
            idx = grp_info.next_row_in_group[idx];
        }
    }
    out_arr = alloc_array(nRowOut, n_chars, -1, arr_type, dtype, 0, 0);
    offset_t* out_offsets = (offset_t*)out_arr->data2;
    char* out_data1 = out_arr->data1;
    // keep track of output array position
    offset_t pos = 0;
    // 2. Copy data from tmp_col to corresponding rows in out_arr
    bool bit = false;
    for (int64_t iRow = 0; iRow < nRowOut; iRow++) {
        offset_t size = ListSizes[iRow];
        int64_t igrp = grp_info.row_to_group[iRow];
        offset_t start_offset = in_offsets[igrp];
        char* in_ptr = in_data1 + start_offset;
        char* out_ptr = out_data1 + pos;
        out_offsets[iRow] = pos;
        memcpy(out_ptr, in_ptr, size);
        pos += size;
        bit = tmp_col->get_null_bit(igrp);
        out_arr->set_null_bit(iRow, bit);
    }
    out_offsets[nRowOut] = pos;
    *update_col = std::move(*out_arr);
    delete out_arr;
}
template <typename ARRAY>
typename std::enable_if<(is_multiple_array<ARRAY>::value), void>::type
copy_string_values_transform(ARRAY* update_col, ARRAY* tmp_col,
                             const grouping_info& grp_info) {
    Bodo_PyErr_SetString(PyExc_RuntimeError,
                         "copy_string_values_transform multiple_array_info "
                         "string not supported yet.");
}
/**
 * @param update_col[out]: column that has the final result for all rows
 * @param tmp_col[in]: column that has the result per group
 * @param grouping_info[in]: structures used to get rows for each group
 * Propagate value from the row in the tmp_col to all the rows in the
 * group update_col.
 *
 * */
template <typename ARRAY, typename T>
void copy_values(ARRAY* update_col, ARRAY* tmp_col,
                 const grouping_info& grp_info) {
    if (tmp_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        copy_nullable_values_transform<ARRAY, T>(update_col, tmp_col, grp_info);
        return;
    }
    // Copy result from tmp_col to corressponding group rows in
    // update_col.
    int64_t nrows = update_col->length;
    for (int64_t iRow = 0; iRow < nrows; iRow++) {
        int64_t igrp = grp_info.row_to_group[iRow];
        T& val = getv<ARRAY, T>(tmp_col, igrp);
        T& val2 = getv<ARRAY, T>(update_col, iRow);
        val2 = val;
    }
}

// Add function declaration before usage in TransformColSet
/**
 * Construct and return a column set based on the ftype.
 * @param groupby input column associated with this column set.
 * @param ftype function type associated with this column set.
 * @param do_combine whether GroupbyPipeline will perform combine operation
 *        or not.
 * @param skipna option used for nunique, cumsum, cumprod, cummin, cummax
 * @param periods option used for shift
 * @param transform_func option used for identifying transform function
 *        (currently groupby operation that are already supported)
 * @param head_n: option used for head operation
 */
template <typename ARRAY>
BasicColSet<ARRAY>* makeColSet(array_info* in_col, array_info* index_col,
                               int ftype, bool do_combine, bool skipna,
                               int64_t periods, int64_t transform_func,
                               int64_t head_n, int n_udf, bool is_parallel,
                               int* udf_n_redvars = nullptr,
                               table_info* udf_table = nullptr,
                               int udf_table_idx = 0,
                               table_info* nunique_table = nullptr);

template <typename ARRAY>
class TransformColSet : public BasicColSet<ARRAY> {
   public:
    TransformColSet(array_info* in_col, int ftype, int _func_num,
                    bool do_combine)
        : BasicColSet<ARRAY>(in_col, ftype, false), transform_func(_func_num) {
        transform_op_col = makeColSet<ARRAY>(
            in_col, nullptr, transform_func, do_combine, false, 0,
            transform_func, -1, 0, true);  // is_parallel = true, head_n=-1
    }
    virtual ~TransformColSet() {
        if (transform_op_col != nullptr) delete transform_op_col;
    }

    virtual void alloc_update_columns(size_t num_groups, size_t n_pivot,
                                      bool is_crosstab,
                                      std::vector<ARRAY*>& out_cols) {
        // Allocate child column that does the actual computation
        std::vector<ARRAY*> list_arr;
        transform_op_col->alloc_update_columns(num_groups, n_pivot, is_crosstab,
                                               list_arr);

        // Get output column type based on transform_func and its in_col
        // datatype
        auto arr_type = this->in_col->arr_type;
        auto dtype = this->in_col->dtype;
        int64_t num_categories = this->in_col->num_categories;
        bool is_combine = false;
        get_groupby_output_dtype<ARRAY>(transform_func, arr_type, dtype, false,
                                        is_combine, is_crosstab);
        // NOTE: output size of transform is the same as input size
        //       (NOT the number of groups)
        out_cols.push_back(alloc_array_groupby<ARRAY>(this->in_col->length, 1,
                                                      1, arr_type, dtype, 0,
                                                      num_categories, n_pivot));
        this->update_cols.push_back(out_cols.back());
    }

    // Call corresponding groupby function operation to compute
    // transform_op_col column.
    virtual void update(const std::vector<grouping_info>& grp_infos) {
        transform_op_col->update(grp_infos);
        aggfunc_output_initialize(this->update_cols[0], transform_func);
    }
    // Fill the output column by copying values from the transform_op_col column
    virtual void eval(const grouping_info& grp_info) {
        // Needed to get final result for transform operation on
        // transform_op_col
        transform_op_col->eval(grp_info);
        // copy_values need to know type of the data it'll copy.
        // Hence we use switch case on the column dtype
        ARRAY* child_out_col = this->transform_op_col->getOutputColumn();
        switch (child_out_col->dtype) {
            case Bodo_CTypes::_BOOL:
                copy_values<ARRAY, bool>(this->update_cols[0], child_out_col,
                                         grp_info);
                break;
            case Bodo_CTypes::INT8:
                copy_values<ARRAY, int8_t>(this->update_cols[0], child_out_col,
                                           grp_info);
                break;
            case Bodo_CTypes::UINT8:
                copy_values<ARRAY, uint8_t>(this->update_cols[0], child_out_col,
                                            grp_info);
                break;
            case Bodo_CTypes::INT16:
                copy_values<ARRAY, int16_t>(this->update_cols[0], child_out_col,
                                            grp_info);
                break;
            case Bodo_CTypes::UINT16:
                copy_values<ARRAY, uint16_t>(this->update_cols[0],
                                             child_out_col, grp_info);
                break;
            case Bodo_CTypes::INT32:
                copy_values<ARRAY, int32_t>(this->update_cols[0], child_out_col,
                                            grp_info);
                break;
            case Bodo_CTypes::UINT32:
                copy_values<ARRAY, uint32_t>(this->update_cols[0],
                                             child_out_col, grp_info);
                break;
            case Bodo_CTypes::DATE:
            case Bodo_CTypes::DATETIME:
            case Bodo_CTypes::TIMEDELTA:
            case Bodo_CTypes::INT64:
                copy_values<ARRAY, int64_t>(this->update_cols[0], child_out_col,
                                            grp_info);
                break;
            case Bodo_CTypes::UINT64:
                copy_values<ARRAY, uint64_t>(this->update_cols[0],
                                             child_out_col, grp_info);
                break;
            case Bodo_CTypes::FLOAT32:
                copy_values<ARRAY, float>(this->update_cols[0], child_out_col,
                                          grp_info);
                break;
            case Bodo_CTypes::FLOAT64:
                copy_values<ARRAY, double>(this->update_cols[0], child_out_col,
                                           grp_info);
                break;
            case Bodo_CTypes::STRING:
                copy_string_values_transform<ARRAY>(this->update_cols[0],
                                                    child_out_col, grp_info);
                break;
        }
        free_array_groupby(child_out_col);
    }

   private:
    int64_t transform_func;
    BasicColSet<ARRAY>* transform_op_col = nullptr;
};
template <typename ARRAY>
class HeadColSet : public BasicColSet<ARRAY> {
   public:
    HeadColSet(array_info* in_col, int ftype, int64_t _n)
        : BasicColSet<ARRAY>(in_col, ftype, false), head_n(_n) {}
    virtual ~HeadColSet() {}

    virtual void alloc_update_columns(size_t update_col_len, size_t n_pivot,
                                      bool is_crosstab,
                                      std::vector<ARRAY*>& out_cols) {
        // NOTE: output size of head is dependent on number of rows to
        // get from each group. This is computed in GroupbyPipeline::update().
        out_cols.push_back(alloc_array_groupby<ARRAY>(
            update_col_len, 1, 1, this->in_col->arr_type, this->in_col->dtype,
            0, this->in_col->num_categories, n_pivot));
        this->update_cols.push_back(out_cols.back());
    }

    virtual void update(const std::vector<grouping_info>& grp_infos) {
        head_computation<ARRAY>(this->in_col, this->update_cols[0],
                                head_row_list);
    }
    void set_head_row_list(std::vector<int64_t> row_list) {
        head_row_list = row_list;
    }

   private:
    std::vector<int64_t> head_row_list;
    int64_t head_n;  // number of rows per group to return
};
/* When transmitting data with shuffle, we have only functionality for
   array_info.
   For array_info, nothing needs to be done.
   For multiple_array_info, we need to add the vect_access entries.
*/
template <typename ARRAY>
typename std::enable_if<!is_multiple_array<ARRAY>::value, void>::type
push_back_arrays(std::vector<array_info*>& ListArr, ARRAY* arr) {
    ListArr.push_back(arr);
}

template <typename ARRAY>
typename std::enable_if<is_multiple_array<ARRAY>::value, void>::type
push_back_arrays(std::vector<array_info*>& ListArr, ARRAY* arr) {
    for (auto& earr : arr->vect_arr) ListArr.push_back(earr);
    for (auto& earr : arr->vect_access) ListArr.push_back(earr);
}

/*
  The output_list_array takes the array (whether aray_info or
  multiple_array_info) and write them in their final form
  for output.
  For array_info, nothing needs to be done.
  For multiple_array_info, the nan bits or nan values have
  to be set if there is actually no values to be put.
 */
template <typename ARRAY>
typename std::enable_if<!is_multiple_array<ARRAY>::value, void>::type
output_list_arrays(std::vector<array_info*>& ListArr, ARRAY* arr) {
    ListArr.push_back(arr);
}

template <typename ARRAY>
typename std::enable_if<is_multiple_array<ARRAY>::value, void>::type
output_list_arrays(std::vector<array_info*>& ListArr, ARRAY* arr) {
    int64_t length_loc = arr->length_loc;
    for (size_t i_arr = 0; i_arr < arr->vect_arr.size(); i_arr++) {
        array_info* earr = arr->vect_arr[i_arr];
        int64_t i_arr_access = i_arr / 8;
        int64_t pos_arr_access = i_arr % 8;
        array_info* earr_bitmap = arr->vect_access[i_arr_access];
        if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL ||
            arr->arr_type == bodo_array_type::STRING ||
            arr->arr_type == bodo_array_type::LIST_STRING) {
            uint8_t* ptr = (uint8_t*)earr_bitmap->data1;
            for (int i_row = 0; i_row < length_loc; i_row++) {
                uint8_t* ptr_b = ptr + i_row;
                bool bit = GetBit(ptr_b, pos_arr_access);
                earr->set_null_bit(i_row, bit);
            }
        }
        if (arr->arr_type == bodo_array_type::NUMPY &&
            (arr->dtype == Bodo_CTypes::FLOAT32 ||
             arr->dtype == Bodo_CTypes::FLOAT64)) {
            std::vector<char> vectNaN = RetrieveNaNentry(arr->dtype);
            uint64_t siztype = numpy_item_size[arr->dtype];
            uint8_t* ptr = (uint8_t*)earr_bitmap->data1;
            for (int i_row = 0; i_row < length_loc; i_row++) {
                uint8_t* ptr_b = ptr + i_row;
                bool bit = GetBit(ptr_b, pos_arr_access);
                if (!bit) {
                    memcpy(earr->data1 + siztype * i_row, vectNaN.data(),
                           siztype);
                }
            }
        }
        ListArr.push_back(earr);
    }
    // This is the final stage. The multiple_array_info is not going to be used
    // in the future, therefore we can delete the vect_access data
    for (auto& e_arr : arr->vect_access) delete_info_decref_array(e_arr);
}

template <typename ARRAY>
BasicColSet<ARRAY>* makeColSet(array_info* in_col, array_info* index_col,
                               int ftype, bool do_combine, bool skipna,
                               int64_t periods, int64_t transform_func,
                               int64_t head_n, int n_udf, bool is_parallel,
                               int* udf_n_redvars, table_info* udf_table,
                               int udf_table_idx, table_info* nunique_table) {
    switch (ftype) {
        case Bodo_FTypes::udf:
            return new UdfColSet<ARRAY>(in_col, do_combine, udf_table,
                                        udf_table_idx, udf_n_redvars[n_udf]);
        case Bodo_FTypes::gen_udf:
            return new GeneralUdfColSet<ARRAY>(in_col, udf_table,
                                               udf_table_idx);
        case Bodo_FTypes::median:
            return new MedianColSet<ARRAY>(in_col, skipna);
        case Bodo_FTypes::nunique:
            return new NUniqueColSet<ARRAY>(in_col, skipna, nunique_table,
                                            do_combine, is_parallel);
        case Bodo_FTypes::cumsum:
        case Bodo_FTypes::cummin:
        case Bodo_FTypes::cummax:
        case Bodo_FTypes::cumprod:
            return new CumOpColSet<ARRAY>(in_col, ftype, skipna);
        case Bodo_FTypes::mean:
            return new MeanColSet<ARRAY>(in_col, do_combine);
        case Bodo_FTypes::var:
        case Bodo_FTypes::std:
            return new VarStdColSet<ARRAY>(in_col, ftype, do_combine);
        case Bodo_FTypes::idxmin:
        case Bodo_FTypes::idxmax:
            return new IdxMinMaxColSet<ARRAY>(in_col, index_col, ftype,
                                              do_combine);
        case Bodo_FTypes::shift:
            return new ShiftColSet<ARRAY>(in_col, ftype, periods);
        case Bodo_FTypes::transform:
            return new TransformColSet<ARRAY>(in_col, ftype, transform_func,
                                              do_combine);
        case Bodo_FTypes::head:
            return new HeadColSet<ARRAY>(in_col, ftype, head_n);
        default:
            return new BasicColSet<ARRAY>(in_col, ftype, do_combine);
    }
}

template <typename ARRAY>
class GroupbyPipeline {
   public:
    GroupbyPipeline(table_info* _in_table, int64_t _num_keys,
                    table_info* _dispatch_table, table_info* _dispatch_info,
                    bool input_has_index, bool _is_parallel, bool _is_crosstab,
                    int* ftypes, int* func_offsets, int* _udf_nredvars,
                    table_info* _udf_table, udf_table_op_fn update_cb,
                    udf_table_op_fn combine_cb, udf_eval_fn eval_cb,
                    udf_general_fn general_udfs_cb, bool skipna,
                    int64_t periods, int64_t transform_func, int64_t _head_n,
                    bool _return_key, bool _return_index, bool _key_dropna)
        : in_table(_in_table),
          orig_in_table(_in_table),
          num_keys(_num_keys),
          dispatch_table(_dispatch_table),
          dispatch_info(_dispatch_info),
          is_parallel(_is_parallel),
          is_crosstab(_is_crosstab),
          return_key(_return_key),
          return_index(_return_index),
          key_dropna(_key_dropna),
          udf_table(_udf_table),
          udf_n_redvars(_udf_nredvars),
          head_n(_head_n) {
        tracing::Event ev("GroupbyPipeline()", is_parallel);
        if (dispatch_table == nullptr)
            n_pivot = 1;
        else
            n_pivot = dispatch_table->nrows();
        udf_info = {udf_table, update_cb, combine_cb, eval_cb, general_udfs_cb};
        // if true, the last column is the index on input and output.
        // this is relevant only to cumulative operations like cumsum
        // and transform.
        int index_i = int(input_has_index);
        // NOTE cumulative operations (cumsum, cumprod, etc.) cannot be mixed
        // with non cumulative ops. This is checked at compile time in
        // aggregate.py

        bool has_udf = false;
        bool nunique_op = false;
        int nunique_count = 0;
        const int num_funcs =
            func_offsets[in_table->ncols() - num_keys - index_i];
        for (int i = 0; i < num_funcs; i++) {
            int ftype = ftypes[i];
            if (ftype == Bodo_FTypes::gen_udf && is_parallel)
                shuffle_before_update = true;
            if (ftype == Bodo_FTypes::udf) has_udf = true;
            if (ftype == Bodo_FTypes::head) {
                head_op = true;
                if (is_parallel) shuffle_before_update = true;
                break;
            }
            if (ftype == Bodo_FTypes::nunique) {
                nunique_op = true;
                req_extended_group_info = true;
                nunique_count++;
            } else if (ftype == Bodo_FTypes::median ||
                       ftype == Bodo_FTypes::cumsum ||
                       ftype == Bodo_FTypes::cumprod ||
                       ftype == Bodo_FTypes::cummin ||
                       ftype == Bodo_FTypes::cummax ||
                       ftype == Bodo_FTypes::shift ||
                       ftype == Bodo_FTypes::transform) {
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
                if (ftype == Bodo_FTypes::shift) shift_op = true;
                if (ftype == Bodo_FTypes::transform) transform_op = true;
                break;
            }
        }
        if (nunique_op) {
            if (nunique_count == num_funcs) nunique_only = true;
            ev.add_attribute("nunique_only", nunique_only);
        }

        // if gb.head and data is distribute, last column is key-sort column.
        int head_i = int(head_op && is_parallel);
        // Add key-sorting-column for gb.head() to sort output at the end
        // this is relevant only if data is distributed.
        if (head_i) add_head_key_sort_column();

        // get hashes of keys
        hashes = hash_keys_table(in_table, num_keys, SEED_HASH_PARTITION,
                                 is_parallel);
        size_t nunique_hashes_global;
        // get estimate of number of unique hashes to guide optimization.
        // if shuffle_before_update=true we are going to shuffle everything
        // first so we don't need statistics of current hashes
        if (is_parallel && !shuffle_before_update) {
            if (nunique_op)
                // nunique_hashes_global is currently only used for gb.nunique
                // heuristic
                std::tie(nunique_hashes, nunique_hashes_global) =
                    get_nunique_hashes_global(hashes, in_table->nrows(),
                                              is_parallel);
            else
                nunique_hashes =
                    get_nunique_hashes(hashes, in_table->nrows(), is_parallel);
        } else if (!is_parallel) {
            nunique_hashes =
                get_nunique_hashes(hashes, in_table->nrows(), is_parallel);
        }

        if (is_parallel && (dispatch_table == nullptr) && !has_udf &&
            !shuffle_before_update) {
            // If the estimated number of groups (given by nunique_hashes)
            // is similar to the number of input rows, then it's better to
            // shuffle first instead of doing a local reduction

            // TODO To do this with UDF functions we need to generate
            // two versions of UDFs at compile time (one for
            // shuffle_before_update=true and one for
            // shuffle_before_update=false)

            int shuffle_before_update_local = 0;
            float groups_in_nrows_ratio;
            if (in_table->nrows() == 0)
                groups_in_nrows_ratio = 1.0;
            else
                groups_in_nrows_ratio =
                    float(nunique_hashes) / in_table->nrows();
            if (groups_in_nrows_ratio >= 0.9)  // XXX what threshold is best?
                shuffle_before_update_local = 1;
            ev.add_attribute("groups_in_nrows_ratio", groups_in_nrows_ratio);
            ev.add_attribute("shuffle_before_update_local",
                             shuffle_before_update_local);
            // global count of ranks that decide to shuffle before update
            int shuffle_before_update_count;
            MPI_Allreduce(&shuffle_before_update_local,
                          &shuffle_before_update_count, 1, MPI_INT, MPI_SUM,
                          MPI_COMM_WORLD);
            // TODO Need a better threshold or cost model to decide when
            // to shuffle: https://bodo.atlassian.net/browse/BE-1140
            int num_ranks;
            MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
            if (shuffle_before_update_count >= num_ranks * 0.5)
                shuffle_before_update = true;
        }

        if (shuffle_before_update) {
            // Code below is equivalent to:
            // table_info* in_table = shuffle_table(in_table, num_keys)
            // We do this more complicated construction because we may
            // need the hashes and comm_info later.
            comm_info_ptr = new mpi_comm_info(in_table->columns);
            comm_info_ptr->set_counts(hashes, is_parallel);
            // shuffle_table_kernel steals the reference but we still
            // need it for the code after C++ groupby
            for (auto a : in_table->columns) incref_array(a);
            in_table = shuffle_table_kernel(in_table, hashes, *comm_info_ptr,
                                            is_parallel);
            if (!(cumulative_op || shift_op || transform_op)) {
                delete[] hashes;
                delete comm_info_ptr;
            } else {
                // preserve input table hashes for reverse shuffle at the end
                in_hashes = hashes;
            }
            hashes = nullptr;
        } else if (nunique_op && is_parallel) {
            // **NOTE**: gb_nunique_preprocess can set
            // shuffle_before_update=true in some cases
            gb_nunique_preprocess(ftypes, num_funcs, nunique_hashes_global);
        }

        // a combine operation is only necessary when data is distributed and
        // a shuffle has not been done at the start of the groupby pipeline
        do_combine = is_parallel && !shuffle_before_update;

        array_info* index_col = nullptr;
        if (input_has_index)
            // if gb.head() exclude head_op column as well (if data is
            // distributed).
            index_col =
                in_table->columns[in_table->columns.size() - 1 - head_i];

        // construct the column sets, one for each (input_column, func) pair.
        // ftypes is an array of function types received from generated code,
        // and has one ftype for each (input_column, func) pair
        int k = 0;
        n_udf = 0;
        for (int64_t i = num_keys; i < in_table->ncols() - index_i - head_i;
             i++, k++) {  // for each data column
            array_info* col = in_table->columns[i];
            int start = func_offsets[k];
            int end = func_offsets[k + 1];
            for (int j = start; j != end;
                 j++) {  // for each function applied to this column
                if (ftypes[j] == Bodo_FTypes::nunique &&
                    (nunique_tables.size() > 0)) {
                    array_info* nunique_col =
                        nunique_tables[i]->columns[num_keys];
                    col_sets.push_back(makeColSet<ARRAY>(
                        nunique_col, index_col, ftypes[j], do_combine, skipna,
                        periods, transform_func, head_n, n_udf, is_parallel,
                        udf_n_redvars, udf_table, udf_table_idx,
                        nunique_tables[i]));
                } else {
                    col_sets.push_back(makeColSet<ARRAY>(
                        col, index_col, ftypes[j], do_combine, skipna, periods,
                        transform_func, head_n, n_udf, is_parallel,
                        udf_n_redvars, udf_table, udf_table_idx));
                }
                if (ftypes[j] == Bodo_FTypes::udf ||
                    ftypes[j] == Bodo_FTypes::gen_udf) {
                    udf_table_idx += (1 + udf_n_redvars[n_udf]);
                    n_udf++;
                    if (ftypes[j] == Bodo_FTypes::gen_udf) {
                        gen_udf_col_sets.push_back(
                            dynamic_cast<GeneralUdfColSet<ARRAY>*>(
                                col_sets.back()));
                    }
                }
                ev.add_attribute("g_column_ftype_" + std::to_string(j),
                                 ftypes[j]);
            }
        }
        // This is needed if aggregation was just size operation, it will skip
        // loop (ncols = num_keys + index_i)
        if (col_sets.size() == 0 && ftypes[0] == Bodo_FTypes::size) {
            col_sets.push_back(makeColSet<ARRAY>(
                in_table->columns[0], index_col, ftypes[0], do_combine, skipna,
                periods, transform_func, head_n, n_udf, is_parallel,
                udf_n_redvars, udf_table, udf_table_idx));
        }
        // Add key-sort column and index to col_sets
        // to apply head_computation on them as well.
        if (head_op && return_index) {
            // index-column
            col_sets.push_back(makeColSet<ARRAY>(
                index_col, index_col, Bodo_FTypes::head, do_combine, skipna,
                periods, transform_func, head_n, n_udf, is_parallel,
                udf_n_redvars, udf_table, udf_table_idx));
            if (head_i) {
                array_info* col =
                    in_table->columns[in_table->columns.size() - 1];
                col_sets.push_back(makeColSet<ARRAY>(
                    col, index_col, Bodo_FTypes::head, do_combine, skipna,
                    periods, transform_func, head_n, n_udf, is_parallel,
                    udf_n_redvars, udf_table, udf_table_idx));
            }
        }

        in_table->id = 0;
        ev.add_attribute("g_shuffle_before_update",
                         size_t(shuffle_before_update));
        ev.add_attribute("g_do_combine", size_t(do_combine));
    }

    ~GroupbyPipeline() {
        for (auto col_set : col_sets) delete col_set;
        if (hashes != nullptr) delete[] hashes;
    }
    /**
     * @brief
     * Create key-sort column used to sort table at the end.
     * Set its values and add as the last column in in_table.
     * Column values is in range(start, start+nrows).
     * Each rank will compute its range by identifying
     * start/end index of its set of rows.
     * @return ** void
     */
    void add_head_key_sort_column() {
        array_info* head_sort_col = alloc_array_groupby<array_info>(
            in_table->nrows(), 1, 1, bodo_array_type::NUMPY,
            Bodo_CTypes::UINT64, 0, 0, n_pivot);
        int64_t num_ranks = dist_get_size();
        int64_t my_rank = dist_get_rank();
        // Gather the number of rows on every rank
        int64_t num_rows = in_table->nrows();
        std::vector<int64_t> num_rows_ranks(num_ranks);
        MPI_Allgather(&num_rows, 1, MPI_INT64_T, num_rows_ranks.data(), 1,
                      MPI_INT64_T, MPI_COMM_WORLD);

        // Determine the start/end row number of each rank
        int64_t rank_start_row, rank_end_row;
        rank_end_row = std::accumulate(num_rows_ranks.begin(),
                                       num_rows_ranks.begin() + my_rank + 1, 0);
        rank_start_row = rank_end_row - num_rows;
        // generate start/end range
        for (int64_t i = 0; i < num_rows; i++) {
            uint64_t& val = getv<array_info, uint64_t>(head_sort_col, i);
            val = rank_start_row + i;
        }
        in_table->columns.push_back(head_sort_col);
    }

    /**
     * This is the main control flow of the Groupby pipeline.
     */
    table_info* run() {
        update();
        if (shuffle_before_update) {
            if (in_table != orig_in_table)
                // in_table is temporary table created in C++
                delete_table_decref_arrays(in_table);
        }
        if (is_parallel && !shuffle_before_update) {
            shuffle();
            combine();
        }
        eval();
        // For gb.head() operation, if data is distributed,
        // sort table based on head_sort_col column.
        if (head_op && is_parallel) sort_gb_head_output();
        return getOutputTable();
    }
    /**
     * @brief
     * 1. Put head_sort_col at the beginning of the table.
     * 2. Sort table based on this column.
     * 3. Remove head_sort_col.
     * @return ** void
     */
    void sort_gb_head_output() {
        // Move sort column to the front.
        std::vector<array_info*>::iterator pos = cur_table->columns.end() - 1;
        std::rotate(cur_table->columns.begin(), pos, pos + 1);
        // whether to put NaN first or last.
        // Does not matter in this case (no NaN, values are range(nrows))
        int64_t asc_pos = 1;
        int64_t zero = 0;
        cur_table = sort_values_table(cur_table, 1, &asc_pos, &asc_pos, &zero,
                                      nullptr, is_parallel);
        // Remove key-sort column
        free_array_groupby(cur_table->columns[0]);
        cur_table->columns.erase(cur_table->columns.begin());
    }

   private:
    int64_t compute_head_row_list(grouping_info const& grp_info,
                                  std::vector<int64_t>& head_row_list) {
        // keep track of how many rows found per group so far.
        std::vector<int64_t> nrows_per_grp(grp_info.num_groups);
        int64_t count = 0;  // how many rows found so far
        int64_t iRow = 0;   // index looping over all rows
        for (size_t iRow = 0; iRow < in_table->nrows(); iRow++) {
            int64_t igrp = grp_info.row_to_group[iRow];
            if (igrp != -1 && nrows_per_grp[igrp] < head_n) {
                nrows_per_grp[igrp]++;
                head_row_list.push_back(iRow);
                count++;
            }
        }
        return count;
    }
    /**
     * The update step groups rows in the input table based on keys, and
     * aggregates them based on the function to be applied to the columns.
     * More specifically, it will invoke the update method of each column set.
     */
    void update() {
        tracing::Event ev("update", is_parallel);
        in_table->num_keys = num_keys;
        std::vector<table_info*> tables;
        // If nunique_only and nunique_tables.size() > 0 then all of the input
        // data is in nunique_tables
        if (!(nunique_only && nunique_tables.size() > 0))
            tables.push_back(in_table);
        for (auto it = nunique_tables.begin(); it != nunique_tables.end(); it++)
            tables.push_back(it->second);

        if (req_extended_group_info) {
            const bool consider_missing =
                cumulative_op || shift_op || transform_op;
            get_group_info_iterate(tables, hashes, nunique_hashes, grp_infos,
                                   consider_missing, key_dropna, is_parallel);
        } else
            get_group_info(tables, hashes, nunique_hashes, grp_infos, true,
                           key_dropna, is_parallel);
        grouping_info& grp_info = grp_infos[0];
        grp_info.dispatch_table = dispatch_table;
        grp_info.dispatch_info = dispatch_info;
        grp_info.mode = 1;
        num_groups = grp_info.num_groups;
        grp_info.n_pivot = n_pivot;
        int64_t update_col_len = num_groups;
        std::vector<int64_t> head_row_list;
        if (head_op) {
            update_col_len = compute_head_row_list(grp_infos[0], head_row_list);
        }

        update_table = cur_table = new table_info();
        if (cumulative_op || shift_op || transform_op || head_op)
            num_keys = 0;  // there are no key columns in output of cumulative
                           // operations
        else
            alloc_init_keys(tables, update_table);

        for (auto col_set : col_sets) {
            std::vector<ARRAY*> list_arr;
            col_set->alloc_update_columns(update_col_len, n_pivot, is_crosstab,
                                          list_arr);
            for (auto& e_arr : list_arr)
                push_back_arrays(update_table->columns, e_arr);
            auto head_col = dynamic_cast<HeadColSet<ARRAY>*>(col_set);
            if (head_col) head_col->set_head_row_list(head_row_list);
            col_set->update(grp_infos);
        }
        // gb.head() already added the index to the tables columns.
        // This is need to do head_computation on it as well.
        // since it will not be the same length as the in_table.
        if (!head_op && return_index)
            update_table->columns.push_back(
                copy_array(in_table->columns.back()));
        if (n_udf > 0) {
            int n_gen_udf = gen_udf_col_sets.size();
            if (n_udf > n_gen_udf)
                // regular UDFs
                udf_info.update(in_table, update_table,
                                grp_info.row_to_group.data());
            if (n_gen_udf > 0) {
                table_info* general_in_table = new table_info();
                for (auto udf_col_set : gen_udf_col_sets)
                    udf_col_set->fill_in_columns(general_in_table, grp_info);
                udf_info.general_udf(grp_info.num_groups, general_in_table,
                                     update_table);
                delete_table_decref_arrays(general_in_table);
            }
        }
    }

    /**
     * Shuffles the update table and updates the column sets with the newly
     * shuffled table.
     */
    void shuffle() {
        tracing::Event ev("shuffle", is_parallel);
        table_info* shuf_table =
            shuffle_table(update_table, num_keys, is_parallel);
#ifdef DEBUG_GROUPBY
        std::cout << "After shuffle_table_kernel. shuf_table=\n";
        DEBUG_PrintSetOfColumn(std::cout, shuf_table->columns);
        DEBUG_PrintRefct(std::cout, shuf_table->columns);
#endif
        // NOTE: shuffle_table_kernel decrefs input arrays
        delete_table(update_table);
        update_table = cur_table = shuf_table;

        // update column sets with columns from shuffled table
        auto it = update_table->columns.begin() + num_keys;
        for (auto col_set : col_sets)
            it = col_set->update_after_shuffle(it, n_pivot);
    }

    /**
     * The combine step is performed after update and shuffle. It groups rows
     * in shuffled table based on keys, and aggregates them based on the
     * function to be applied to the columns. More specifically, it will invoke
     * the combine method of each column set.
     */
    void combine() {
        tracing::Event ev("combine", is_parallel);
        update_table->num_keys = num_keys;
        grp_infos.clear();
        std::vector<table_info*> tables = {update_table};
        get_group_info(tables, hashes, nunique_hashes, grp_infos, false,
                       key_dropna, is_parallel);
        grouping_info& grp_info = grp_infos[0];
        num_groups = grp_info.num_groups;
        grp_info.dispatch_table = dispatch_table;
        grp_info.dispatch_info = dispatch_info;
        grp_info.n_pivot = n_pivot;
        grp_info.mode = 2;

        combine_table = cur_table = new table_info();
        alloc_init_keys({update_table}, combine_table);
        std::vector<array_info*> list_arr;
        for (auto col_set : col_sets) {
            std::vector<ARRAY*> list_arr;
            col_set->alloc_combine_columns(num_groups, n_pivot, is_crosstab,
                                           list_arr);
            for (auto& e_arr : list_arr) {
                push_back_arrays(combine_table->columns, e_arr);
            }
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
        tracing::Event ev("eval", is_parallel);
        for (auto col_set : col_sets) col_set->eval(grp_infos[0]);
        // only regular UDFs need eval step
        if (n_udf - gen_udf_col_sets.size() > 0) udf_info.eval(cur_table);
    }

    /**
     * Returns the final output table which is the result of the groupby.
     */
    table_info* getOutputTable() {
        table_info* out_table = new table_info();
        if (return_key)
            out_table->columns.assign(cur_table->columns.begin(),
                                      cur_table->columns.begin() + num_keys);
        // gb.head() with distirbuted data sorted the table so col_sets no
        // longer reflects final
        // output columns.
        if (head_op && is_parallel) {
            for (int i = 0; i < cur_table->ncols(); i++)
                output_list_arrays(out_table->columns, cur_table->columns[i]);
        } else {
            for (BasicColSet<ARRAY>* col_set : col_sets)
                output_list_arrays(out_table->columns,
                                   col_set->getOutputColumn());
            // gb.head() already added index to out_table.
            if (!head_op && return_index)
                out_table->columns.push_back(cur_table->columns.back());
        }
        if ((cumulative_op || shift_op || transform_op) && is_parallel) {
            table_info* revshuf_table = reverse_shuffle_table_kernel(
                out_table, in_hashes, *comm_info_ptr);
            delete[] in_hashes;
            delete comm_info_ptr;
            delete_table(out_table);
            out_table = revshuf_table;
        }
        delete cur_table;
        return out_table;
    }

    /**
     * We enter this algorithm at the beginning of the groupby pipeline, if
     * there are gb.nunique operations and there is no other operation that
     * requires shuffling before update. This algorithm decides, for each
     * nunique column, whether all ranks drop duplicates locally for that column
     * based on average local cardinality estimates across all ranks, and will
     * also decide how to shuffle all the nunique columns (it will use the same
     * scheme to shuffle all the nunique columns since the decision is not
     * based on the characteristics on any particular column). There are two
     * strategies for shuffling:
     * a) Shuffle based on groupby keys. Shuffles nunique data to its final
     *    destination. If there are no other groupby operations other than
     *    nunique then this equals shuffle_before_update=true and we just
     *    need an update and eval step (no second shuffle and combine). But
     *    if there are other operations mixed in, for simplicity we will do
     *    update, shuffle and combine step for nunique columns even though
     *    the nunique data is already in the final destination.
     * b) Shuffle based on keys+value. This is done if the number of *global*
     *    groups is small compared to the number of ranks, since shuffling
     *    based on keys in this case can generate significant load imbalance.
     *    In this case the update step calculates number of unique values
     *    for (key, value) tuples, the second shuffle (after update) collects
     *    the nuniques for a given group on the same rank, and the combine sums
     *    them.
     * @param ftypes: list of groupby function types passed directly from
     * GroupbyPipeline constructor.
     * @param num_funcs: number of functions in ftypes
     * @param nunique_hashes_global: estimated number of global unique hashes
     * of groupby keys (gives an estimate of global number of unique groups)
     */
    void gb_nunique_preprocess(int* ftypes, int num_funcs,
                               size_t nunique_hashes_global) {
        tracing::Event ev("gb_nunique_preprocess", is_parallel);
        if (!is_parallel)
            throw std::runtime_error(
                "gb_nunique_preprocess called for non-distributed data");
        if (shuffle_before_update)
            throw std::runtime_error(
                "gb_nunique_preprocess called with shuffle_before_update=true");

        // If it's just nunique we set table_id_counter to 0 because we won't
        // add in_table to our list of tables. Otherwise, set to 1 as 0 is
        // reserved for in_table
        int table_id_counter;
        if (nunique_only)
            table_id_counter = 0;
        else
            table_id_counter = 1;

        static constexpr float threshold_of_fraction_of_unique_hash = 0.5;
        ev.add_attribute("g_threshold_of_fraction_of_unique_hash",
                         threshold_of_fraction_of_unique_hash);

        // If the number of global groups is small we need to shuffle
        // based on keys *and* values to maximize data distribution
        // and improve scaling. If we only spread based on keys, scaling
        // will be limited by the number of groups.
        int num_ranks;
        MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
        // When number of groups starts to approximate the number of ranks
        // there will be a high chance that a single rank ends up with 2-3
        // times the load (number of groups) than others after shuffling
        // TODO investigate what is the best threshold:
        // https://bodo.atlassian.net/browse/BE-1308
        const bool shuffle_by_keys_and_value =
            (nunique_hashes_global <= num_ranks * 3);
        ev.add_attribute("g_nunique_shuffle_by_keys_and_values",
                         shuffle_by_keys_and_value);

        for (int i = 0, col_idx = num_keys; i < num_funcs; i++, col_idx++) {
            if (ftypes[i] != Bodo_FTypes::nunique) continue;

            table_info* tmp = new table_info();
            tmp->columns.assign(in_table->columns.begin(),
                                in_table->columns.begin() + num_keys);
            tmp->num_keys = num_keys;
            push_back_arrays(tmp->columns, in_table->columns[col_idx]);

            // --------- drop local duplicates ---------
            // If we know that the |set(values)| / len(values)
            // is low on all ranks then it should be beneficial to
            // drop local duplicates before the shuffle.

            const size_t n_rows = static_cast<size_t>(in_table->nrows());
            // get hashes of keys+value
            uint32_t* key_value_hashes = new uint32_t[n_rows];
            memcpy(key_value_hashes, hashes, sizeof(uint32_t) * n_rows);
            // TODO: do a hash combine that writes to an empty hash
            // array to avoid memcpy?
            hash_array_combine(key_value_hashes, tmp->columns[num_keys], n_rows,
                               SEED_HASH_PARTITION,
                               /*global_dict_needed=*/true);

            // Compute the local fraction of unique hashes
            size_t nunique_keyval_hashes =
                get_nunique_hashes(key_value_hashes, n_rows, is_parallel);
            float local_fraction_unique_hashes =
                static_cast<float>(nunique_keyval_hashes) /
                static_cast<float>(n_rows);
            float global_fraction_unique_hashes;
            if (ev.is_tracing())
                ev.add_attribute("nunique_" + std::to_string(i) +
                                     "_local_fraction_unique_hashes",
                                 local_fraction_unique_hashes);
            MPI_Allreduce(&local_fraction_unique_hashes,
                          &global_fraction_unique_hashes, 1, MPI_FLOAT, MPI_SUM,
                          MPI_COMM_WORLD);
            global_fraction_unique_hashes /= static_cast<float>(num_ranks);
            ev.add_attribute("g_nunique_" + std::to_string(i) +
                                 "_global_fraction_unique_hashes",
                             global_fraction_unique_hashes);
            const bool drop_duplicates = global_fraction_unique_hashes <
                                         threshold_of_fraction_of_unique_hash;
            ev.add_attribute(
                "g_nunique_" + std::to_string(i) + "_drop_duplicates",
                drop_duplicates);

            // Regardless of whether we drop duplicates or not, the references
            // to the original input arrays are going to be decremented (by
            // either drop_duplicates_table_inner or shuffle_table), but we
            // still need the references for the code after C++ groupby
            for (auto a : tmp->columns) incref_array(a);
            table_info* tmp2 = nullptr;
            if (drop_duplicates) {
                // Set dropna to false because skipna is handled at
                // a later step. Setting dropna=True here removes NA
                // from the keys, which we do not want
                tmp2 = drop_duplicates_table_inner(tmp, tmp->ncols(), 0, 1,
                                                   is_parallel, false,
                                                   key_value_hashes);
                delete tmp;
                tmp = tmp2;
            }

            // --------- shuffle column ---------
            if (shuffle_by_keys_and_value) {
                if (drop_duplicates)
                    // Note that tmp here no longer contains the
                    // original input arrays
                    tmp2 = shuffle_table(tmp, tmp->ncols(), is_parallel);
                else
                    // Since the arrays are unmodified we can reuse the hashes
                    tmp2 = shuffle_table(tmp, tmp->ncols(), is_parallel, false,
                                         key_value_hashes);
            } else {
                if (drop_duplicates)
                    tmp2 = shuffle_table(tmp, num_keys, is_parallel);
                else
                    tmp2 = shuffle_table(tmp, num_keys, is_parallel, false,
                                         hashes);
            }
            delete[] key_value_hashes;
            delete tmp;
            tmp2->num_keys = num_keys;
            tmp2->id = table_id_counter++;
            nunique_tables[col_idx] = tmp2;
        }

        if (!shuffle_by_keys_and_value && nunique_only)
            // We have shuffled the data to its final destination so this is
            // equivalent to shuffle_before_update=true and we don't need to
            // do a combine step
            shuffle_before_update = true;

        if (nunique_only) {
            // in the case of nunique_only the hashes that we calculated in
            // GroupbyPipeline() are not valid, since we have shuffled all of
            // the input columns
            delete[] hashes;
            hashes = nullptr;
        }
    }

    void find_key_for_group(int64_t group,
                            const std::vector<table_info*>& from_tables,
                            int64_t key_col_idx, array_info*& key_col,
                            int64_t& key_row) {
        for (size_t k = 0; k < grp_infos.size(); k++) {
            key_row = grp_infos[k].group_to_first_row[group];
            if (key_row >= 0) {
                key_col = (*from_tables[k])[key_col_idx];
                return;
            }
        }
        // this is error
    }

    /**
     * Allocate and fill key columns, based on grouping info. It uses the
     * values of key columns from from_table to populate out_table.
     */
    void alloc_init_keys(std::vector<table_info*> from_tables,
                         table_info* out_table) {
        int64_t key_row;
        for (int64_t i = 0; i < num_keys; i++) {
            array_info* key_col = (*from_tables[0])[i];
            array_info* new_key_col;
            if (key_col->arr_type == bodo_array_type::NUMPY ||
                key_col->arr_type == bodo_array_type::CATEGORICAL ||
                key_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
                new_key_col =
                    alloc_array(num_groups, 1, 1, key_col->arr_type,
                                key_col->dtype, 0, key_col->num_categories);
                int64_t dtype_size = numpy_item_size[key_col->dtype];
                for (size_t j = 0; j < num_groups; j++) {
                    find_key_for_group(j, from_tables, i, key_col, key_row);
                    memcpy(new_key_col->data1 + j * dtype_size,
                           key_col->data1 + key_row * dtype_size, dtype_size);
                }
                if (key_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
                    for (size_t j = 0; j < num_groups; j++) {
                        find_key_for_group(j, from_tables, i, key_col, key_row);
                        bool bit = key_col->get_null_bit(key_row);
                        new_key_col->set_null_bit(j, bit);
                    }
                }
            }
            if (key_col->arr_type == bodo_array_type::STRING) {
                // new key col will have num_groups rows containing the
                // string for each group
                int64_t n_chars = 0;  // total number of chars of all keys for
                                      // this column
                offset_t* in_offsets;
                for (size_t j = 0; j < num_groups; j++) {
                    find_key_for_group(j, from_tables, i, key_col, key_row);
                    in_offsets = (offset_t*)key_col->data2;
                    n_chars += in_offsets[key_row + 1] - in_offsets[key_row];
                }
                new_key_col =
                    alloc_array(num_groups, n_chars, 1, key_col->arr_type,
                                key_col->dtype, 0, key_col->num_categories);

                offset_t* out_offsets = (offset_t*)new_key_col->data2;
                offset_t pos = 0;
                for (size_t j = 0; j < num_groups; j++) {
                    find_key_for_group(j, from_tables, i, key_col, key_row);
                    in_offsets = (offset_t*)key_col->data2;
                    offset_t start_offset = in_offsets[key_row];
                    offset_t str_len = in_offsets[key_row + 1] - start_offset;
                    out_offsets[j] = pos;
                    memcpy(&new_key_col->data1[pos],
                           &key_col->data1[start_offset], str_len);
                    pos += str_len;
                    bool bit = key_col->get_null_bit(key_row);
                    new_key_col->set_null_bit(j, bit);
                }
                out_offsets[num_groups] = pos;
            }
            if (key_col->arr_type == bodo_array_type::LIST_STRING) {
                // new key col will have num_groups rows containing the
                // list string for each group
                int64_t n_strings = 0;  // total number of strings of all keys
                                        // for this column
                int64_t n_chars = 0;    // total number of chars of all keys for
                                        // this column
                offset_t* in_index_offsets;
                offset_t* in_data_offsets;
                for (size_t j = 0; j < num_groups; j++) {
                    find_key_for_group(j, from_tables, i, key_col, key_row);
                    in_index_offsets = (offset_t*)key_col->data3;
                    in_data_offsets = (offset_t*)key_col->data2;
                    n_strings += in_index_offsets[key_row + 1] -
                                 in_index_offsets[key_row];
                    n_chars += in_data_offsets[in_index_offsets[key_row + 1]] -
                               in_data_offsets[in_index_offsets[key_row]];
                }
                new_key_col = alloc_array(num_groups, n_strings, n_chars,
                                          key_col->arr_type, key_col->dtype, 0,
                                          key_col->num_categories);
                uint8_t* in_sub_null_bitmask =
                    (uint8_t*)key_col->sub_null_bitmask;
                uint8_t* out_sub_null_bitmask =
                    (uint8_t*)new_key_col->sub_null_bitmask;
                offset_t* out_index_offsets = (offset_t*)new_key_col->data3;
                offset_t* out_data_offsets = (offset_t*)new_key_col->data2;
                offset_t pos_data = 0;
                offset_t pos_index = 0;
                out_data_offsets[0] = 0;
                out_index_offsets[0] = 0;
                for (size_t j = 0; j < num_groups; j++) {
                    find_key_for_group(j, from_tables, i, key_col, key_row);
                    in_index_offsets = (offset_t*)key_col->data3;
                    in_data_offsets = (offset_t*)key_col->data2;
                    offset_t size_index = in_index_offsets[key_row + 1] -
                                          in_index_offsets[key_row];
                    offset_t pos_start = in_index_offsets[key_row];
                    for (offset_t i_str = 0; i_str < size_index; i_str++) {
                        offset_t len_str =
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
                    offset_t in_start_offset =
                        in_data_offsets[in_index_offsets[key_row]];
                    offset_t n_chars_o =
                        in_data_offsets[in_index_offsets[key_row + 1]] -
                        in_data_offsets[in_index_offsets[key_row]];
                    memcpy(&new_key_col->data1[pos_data],
                           &key_col->data1[in_start_offset], n_chars_o);
                    pos_data += n_chars_o;
                    bool bit = key_col->get_null_bit(key_row);
                    new_key_col->set_null_bit(j, bit);
                }
            }
            out_table->columns.push_back(new_key_col);
        }
    }

    table_info*
        orig_in_table;  // original input table of groupby received from Python
    table_info* in_table;  // input table of groupby
    int64_t num_keys;
    table_info* dispatch_table;  // input dispatching table of pivot_table
    table_info* dispatch_info;   // input dispatching info of pivot_table
    bool is_parallel;
    bool is_crosstab;
    bool return_key;
    bool return_index;
    bool key_dropna;
    std::vector<BasicColSet<ARRAY>*> col_sets;
    std::vector<GeneralUdfColSet<ARRAY>*> gen_udf_col_sets;
    table_info* udf_table;
    int* udf_n_redvars;
    // total number of UDFs applied to input columns (includes regular and
    // general UDFs)
    int n_udf = 0;
    int udf_table_idx = 0;
    int n_pivot;
    // shuffling before update requires more communication and is needed
    // when one of the groupby functions is
    // median/nunique/cumsum/cumprod/cummin/cummax/shift/transform
    bool shuffle_before_update = false;
    bool cumulative_op = false;
    bool shift_op = false;
    bool transform_op = false;
    bool head_op = false;
    int64_t head_n;
    bool req_extended_group_info = false;
    bool do_combine;

    // column position in in_table -> table that contains key columns + one
    // nunique column after [dropping local duplicates] + shuffling
    std::map<int, table_info*> nunique_tables;
    bool nunique_only = false;  // there are only groupby nunique operations

    udfinfo_t udf_info;

    table_info* update_table = nullptr;
    table_info* combine_table = nullptr;
    table_info* cur_table = nullptr;

    std::vector<grouping_info> grp_infos;
    size_t num_groups;
    // shuffling stuff
    uint32_t* in_hashes = nullptr;
    mpi_comm_info* comm_info_ptr = nullptr;
    uint32_t* hashes = nullptr;
    size_t nunique_hashes = 0;
};

// MPI_Exscan: https://www.mpich.org/static/docs/v3.1.x/www3/MPI_Exscan.html
// Useful for cumulative functions. Instead of doing shuffling, we compute the
// groups in advance without doing shuffling using MPI_Exscan. We do the
// cumulative operation first locally on each processor, and we use step
// functions on each processor (sum, min, etc.)

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
    std::vector<T> cumulative(max_row_idx * n_oper);
    for (int j = start; j != end; j++) {
        int ftype = ftypes[j];
        T value_init = -1;  // Dummy value set to avoid a compiler warning
        if (ftype == Bodo_FTypes::cumsum) value_init = 0;
        if (ftype == Bodo_FTypes::cumprod) value_init = 1;
        if (ftype == Bodo_FTypes::cummax)
            value_init = std::numeric_limits<T>::min();
        if (ftype == Bodo_FTypes::cummin)
            value_init = std::numeric_limits<T>::max();
        for (int i_row = 0; i_row < max_row_idx; i_row++)
            cumulative[i_row + max_row_idx * (j - start)] = value_init;
    }
    std::vector<T> cumulative_recv = cumulative;
    array_info* in_col = in_table->columns[k + num_keys];
    T nan_value =
        GetTentry<T>(RetrieveNaNentry((Bodo_CTypes::CTypeEnum)dtype).data());
    Tkey miss_idx = -1;
    for (int j = start; j != end; j++) {
        array_info* work_col = out_arrs[j];
        int ftype = ftypes[j];
        auto apply_oper = [&](auto const& oper) -> void {
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
    for (int j = start; j != end; j++) {
        array_info* work_col = out_arrs[j];
        int ftype = ftypes[j];
        // For skipdropna:
        //   The cumulative is never a NaN. The sum therefore works
        //   correctly whether val is a NaN or not.
        // For !skipdropna:
        //   the cumulative can be a NaN. The sum also works correctly.
        auto apply_oper = [&](auto const& oper) -> void {
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
    Tkey miss_idx = -1;
    for (int j = start; j != end; j++) {
        array_info* work_col = out_arrs[j];
        int ftype = ftypes[j];
        auto apply_oper = [&](auto const& oper) -> void {
            for (int64_t i_row = 0; i_row < n_rows; i_row++) {
                Tkey idx = cat_column->at<Tkey>(i_row);
                if (idx == miss_idx) {
                    work_col->set_null_bit(i_row, false);
                } else {
                    size_t pos = idx + max_row_idx * (j - start);
                    T val = in_col->at<T>(i_row);
                    bool bit_i = in_col->get_null_bit(i_row);
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
                    work_col->set_null_bit(i_row, bit_o);
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
        int ftype = ftypes[j];
        auto apply_oper = [&](auto const& oper) -> void {
            for (int64_t i_row = 0; i_row < n_rows; i_row++) {
                Tkey idx = cat_column->at<Tkey>(i_row);
                if (idx != miss_idx) {
                    size_t pos = idx + max_row_idx * (j - start);
                    T val = work_col->at<T>(i_row);
                    T new_val = oper(val, cumulative_recv[pos]);
                    work_col->at<T>(i_row) = new_val;
                    if (!skipdropna && cumulative_mask_recv[pos] == 1)
                        work_col->set_null_bit(i_row, false);
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
}

template <typename Tkey, typename T, int dtype>
void mpi_exscan_computation_T(std::vector<array_info*>& out_arrs,
                              array_info* cat_column, table_info* in_table,
                              int64_t num_keys, int64_t k, int* ftypes,
                              int* func_offsets, bool is_parallel,
                              bool skipdropna) {
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
    DEBUG_PrintRefct(std::cout, in_table->columns);
#ifdef DEBUG_GROUPBY_FULL
    DEBUG_PrintSetOfColumn(std::cout, in_table->columns);
#endif
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
    tracing::Event ev("mpi_exscan_computation", is_parallel);
    const Bodo_CTypes::CTypeEnum dtype = cat_column->dtype;
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

namespace {
/**
 * Compute hash for `compute_categorical_index`
 *
 * Don't use std::function to reduce call overhead.
 */
struct HashComputeCategoricalIndex {
    size_t operator()(size_t const iRow) const {
        if (iRow < n_rows_full) {
            return static_cast<size_t>(hashes_full[iRow]);
        } else {
            return static_cast<size_t>(hashes_in_table[iRow - n_rows_full]);
        }
    }
    uint32_t* hashes_full;
    uint32_t* hashes_in_table;
    size_t n_rows_full;
};

/**
 * Key comparison for `compute_categorical_index`
 *
 * Don't use std::function to reduce call overhead.
 */
struct HashEqualComputeCategoricalIndex {
    bool operator()(size_t const iRowA, size_t const iRowB) const {
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
            TestEqual(*concat_column, num_keys, shift_A, jRowA, shift_B, jRowB);
        return test;
    }
    int64_t num_keys;
    size_t n_rows_full;
    std::vector<array_info*>* concat_column;
};
}  // namespace

/**
 * Basically assign index to each unique category
 * @param in_table : input table
 * @param num_keys : number of keys
 * @param is_parallel: whether we run in parallel or not.
 * @param key_dropna: whether we drop null keys or not.
 * @return key categorical array_info
 */
array_info* compute_categorical_index(table_info* in_table, int64_t num_keys,
                                      bool is_parallel,
                                      bool key_dropna = true) {
    tracing::Event ev("compute_categorical_index", is_parallel);
#ifdef DEBUG_GROUPBY
    std::cout << "compute_categorical_index num_keys=" << num_keys
              << " is_parallel=" << is_parallel << "\n";
#endif
    // A rare case of incref since we are going to need the in_table after the
    // computation of red_table.
    for (int64_t i_key = 0; i_key < num_keys; i_key++)
        incref_array(in_table->columns[i_key]);
    table_info* red_table =
        drop_duplicates_keys(in_table, num_keys, is_parallel, key_dropna);
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
    // Two approaches for cumulative operations : shuffle (then reshuffle) or
    // use exscan. Preferable to do shuffle when we have too many unique values.
    // This is a heuristic to decide approach.
    if (n_rows_full > max_global_number_groups_exscan) {
        delete_table_decref_arrays(red_table);
        return nullptr;
    }
    // We are below threshold. Now doing an allgather for determining the keys.
    bool all_gather = true;
    table_info* full_table =
        gather_table(red_table, num_keys, all_gather, is_parallel);
    delete_table(red_table);
    // Now building the map_container.
    uint32_t* hashes_full =
        hash_keys_table(full_table, num_keys, SEED_HASH_MULTIKEY, is_parallel);
    uint32_t* hashes_in_table =
        hash_keys_table(in_table, num_keys, SEED_HASH_MULTIKEY, is_parallel);
    std::vector<array_info*> concat_column(
        full_table->columns.begin(), full_table->columns.begin() + num_keys);
    concat_column.insert(concat_column.end(), in_table->columns.begin(),
                         in_table->columns.begin() + num_keys);

    HashComputeCategoricalIndex hash_fct{hashes_full, hashes_in_table,
                                         n_rows_full};
    HashEqualComputeCategoricalIndex equal_fct{num_keys, n_rows_full,
                                               &concat_column};
    UNORD_MAP_CONTAINER<size_t, size_t, HashComputeCategoricalIndex,
                        HashEqualComputeCategoricalIndex>
        entSet({}, hash_fct, equal_fct);
    for (size_t iRow = 0; iRow < size_t(n_rows_full); iRow++)
        entSet[iRow] = iRow;
    size_t n_rows_in = in_table->nrows();
    array_info* out_arr =
        alloc_categorical(n_rows_in, Bodo_CTypes::INT32, n_rows_full);
    std::vector<array_info*> key_cols(in_table->columns.begin(),
                                      in_table->columns.begin() + num_keys);
    bool has_nulls = does_keys_have_nulls(key_cols);
    for (size_t iRow = 0; iRow < n_rows_in; iRow++) {
        int32_t pos;
        if (has_nulls) {
            if (key_dropna && does_row_has_nulls(key_cols, iRow))
                pos = -1;
            else
                pos = entSet[iRow + n_rows_full];
        } else {
            pos = entSet[iRow + n_rows_full];
        }
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

/*
  The pivot_table and crosstab functionality
  --
  The dispatch_table contains the columns with the information.
  The dispatch_info contains the dispatching information.
  The rest works as for groupby.
  We use the multiple_array_info template type to make it work
  for pivot_table and groupby.
 */
table_info* pivot_groupby_and_aggregate(
    table_info* in_table, int64_t num_keys, table_info* dispatch_table,
    table_info* dispatch_info, bool input_has_index, int* ftypes,
    int* func_offsets, int* udf_nredvars, bool is_parallel, bool is_crosstab,
    bool skipdropna, bool return_key, bool return_index, void* update_cb,
    void* combine_cb, void* eval_cb, table_info* udf_dummy_table) {
    try {
        GroupbyPipeline<multiple_array_info> groupby(
            in_table, num_keys, dispatch_table, dispatch_info, input_has_index,
            is_parallel, is_crosstab, ftypes, func_offsets, udf_nredvars,
            udf_dummy_table, (udf_table_op_fn)update_cb,
            // TODO: general UDFs
            (udf_table_op_fn)combine_cb, (udf_eval_fn)eval_cb, 0, skipdropna, 0,
            0, -1, return_key,
            // dropna = True for pivot operation
            return_index, true);  // transform_func = periods = 0
                                  // head_n = -1. Not used in
                                  // pivot operation.

        table_info* ret_table = groupby.run();
        return ret_table;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

table_info* groupby_and_aggregate(
    table_info* in_table, int64_t num_keys, bool input_has_index, int* ftypes,
    int* func_offsets, int* udf_nredvars, bool is_parallel, bool skipdropna,
    int64_t periods, int64_t transform_func, int64_t head_n, bool return_key,
    bool return_index, bool key_dropna, void* update_cb, void* combine_cb,
    void* eval_cb, void* general_udfs_cb, table_info* udf_dummy_table) {
    try {
        tracing::Event ev("groupby_and_aggregate", is_parallel);
        int strategy =
            determine_groupby_strategy(in_table, num_keys, ftypes, func_offsets,
                                       is_parallel, input_has_index);
        ev.add_attribute("g_strategy", size_t(strategy));

        auto implement_strategy0 = [&]() -> table_info* {
            table_info* dispatch_info = nullptr;
            table_info* dispatch_table = nullptr;
            GroupbyPipeline<array_info> groupby(
                in_table, num_keys, dispatch_table, dispatch_info,
                input_has_index, is_parallel, true, ftypes, func_offsets,
                udf_nredvars, udf_dummy_table, (udf_table_op_fn)update_cb,
                (udf_table_op_fn)combine_cb, (udf_eval_fn)eval_cb,
                (udf_general_fn)general_udfs_cb, skipdropna, periods,
                transform_func, head_n, return_key, return_index, key_dropna);

            table_info* ret_table = groupby.run();

            return ret_table;
        };
        auto implement_categorical_exscan =
            [&](array_info* cat_column) -> table_info* {
            table_info* ret_table = mpi_exscan_computation(
                cat_column, in_table, num_keys, ftypes, func_offsets,
                is_parallel, skipdropna, return_key, return_index);
            return ret_table;
        };
        if (strategy == 0) return implement_strategy0();
        if (strategy == 1) {
            array_info* cat_column = in_table->columns[0];
            return implement_categorical_exscan(cat_column);
        }
        if (strategy == 2) {
            array_info* cat_column = compute_categorical_index(
                in_table, num_keys, is_parallel, key_dropna);
            if (cat_column ==
                nullptr) {  // It turns out that there are too many
                            // different keys for exscan to be ok.
                return implement_strategy0();
            } else {
                table_info* ret_table =
                    implement_categorical_exscan(cat_column);
                delete_info_decref_array(cat_column);
                return ret_table;
            }
        }
        return nullptr;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}
