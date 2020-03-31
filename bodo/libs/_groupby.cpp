// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include "_groupby.h"
#include <functional>
#include <limits>
#include "_array_hash.h"
#include "_array_utils.h"
#include "_murmurhash3.h"
#include "_shuffle.h"

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
template <typename T, int ftype, typename Enable = void>
struct aggfunc {
    /**
     * Apply the function.
     * @param[in,out] first input value, and holds the result
     * @param[in] second input value.
     */
    static void apply(T& v1, T& v2) {}
};

template <typename T>
struct aggfunc<
    T, Bodo_FTypes::sum,
    typename std::enable_if<!std::is_floating_point<T>::value>::type> {
    /**
     * Aggregation function for sum. Modifies current sum if value is not a nan
     *
     * @param[in,out] current sum value, and holds the result
     * @param second input value.
     */
    static void apply(T& v1, T& v2) { v1 += v2; }
};

template <typename T>
struct aggfunc<
    T, Bodo_FTypes::sum,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
    static void apply(T& v1, T& v2) {
        if (!isnan(v2)) v1 += v2;
    }
};

template <typename T>
struct aggfunc<
    T, Bodo_FTypes::min,
    typename std::enable_if<!std::is_floating_point<T>::value>::type> {
    /**
     * Aggregation function for min. Modifies current min if value is not a nan
     *
     * @param[in,out] current min value (or nan for floats if no min value found
     * yet)
     * @param second input value.
     */
    static void apply(T& v1, T& v2) { v1 = std::min(v1, v2); }
};

template <typename T>
struct aggfunc<
    T, Bodo_FTypes::min,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
    static void apply(T& v1, T& v2) {
        if (!isnan(v2))
            v1 = std::min(v2, v1);  // std::min(x,NaN) = x
                                    // (v1 is initialized as NaN)
    }
};

template <typename T>
struct aggfunc<
    T, Bodo_FTypes::max,
    typename std::enable_if<!std::is_floating_point<T>::value>::type> {
    /**
     * Aggregation function for max. Modifies current max if value is not a nan
     *
     * @param[in,out] current max value (or nan for floats if no max value found
     * yet)
     * @param second input value.
     */
    static void apply(T& v1, T& v2) { v1 = std::max(v1, v2); }
};

template <typename T>
struct aggfunc<
    T, Bodo_FTypes::max,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
    static void apply(T& v1, T& v2) {
        if (!isnan(v2)) {
            v1 = std::max(v2, v1);  // std::max(x,NaN) = x
                                    // (v1 is initialized as NaN)
        }
    }
};

template <typename T>
struct aggfunc<
    T, Bodo_FTypes::prod,
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
struct aggfunc<bool, Bodo_FTypes::prod> {
    static void apply(bool& v1, bool& v2) { v1 = v1 && v2; }
};

template <typename T>
struct aggfunc<
    T, Bodo_FTypes::prod,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
    static void apply(T& v1, T& v2) {
        if (!isnan(v2)) v1 *= v2;
    }
};

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

template <typename T, typename Enable = void>
struct count_agg {
    /**
     * Aggregation function for count. Increases count if value is not a nan
     *
     * @param[in,out] current count
     * @param second input value.
     */
    static void apply(int64_t& v1, T& v2);
};

template <typename T>
struct count_agg<
    T, typename std::enable_if<!std::is_floating_point<T>::value>::type> {
    static void apply(int64_t& v1, T& v2) { v1 += 1; }
};

template <typename T>
struct count_agg<
    T, typename std::enable_if<std::is_floating_point<T>::value>::type> {
    static void apply(int64_t& v1, T& v2) {
        if (!isnan(v2)) v1 += 1;
    }
};

template <typename T, typename Enable = void>
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

template <typename T>
struct mean_agg<
    T, typename std::enable_if<!std::is_floating_point<T>::value>::type> {
    static void apply(double& v1, T& v2, uint64_t& count) {
        v1 += (double)v2;
        count += 1;
    }
};

template <typename T>
struct mean_agg<
    T, typename std::enable_if<std::is_floating_point<T>::value>::type> {
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

template <typename T, typename Enable = void>
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

template <typename T>
struct var_agg<
    T, typename std::enable_if<!std::is_floating_point<T>::value>::type> {
    inline static void apply(T& v2, uint64_t& count, double& mean_x,
                             double& m2) {
        count += 1;
        double delta = (double)v2 - mean_x;
        mean_x += delta / count;
        double delta2 = (double)v2 - mean_x;
        m2 += delta * delta2;
    }
};

template <typename T>
struct var_agg<
    T, typename std::enable_if<std::is_floating_point<T>::value>::type> {
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
   operations such as nunique, median, and cumulative operations. The entry list_missing
   is computed only for cumulative operations and computed only if needed.
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
                case bodo_array_type::NULLABLE_INT_BOOL:
                    if (GetBit((uint8_t*)c1->null_bitmask, row) !=
                        GetBit((uint8_t*)c2->null_bitmask, other.row))
                        return false;
                    if (!GetBit((uint8_t*)c1->null_bitmask, row)) continue;
                case bodo_array_type::NUMPY:
                    siztype = numpy_item_size[c1->dtype];
                    if (memcmp(c1->data1 + siztype * row,
                               c2->data1 + siztype * other.row, siztype) != 0) {
                        return false;
                    }
                    continue;
                case bodo_array_type::STRING:
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
        }
        return true;
    }
};

struct key_hash {
    std::size_t operator()(const multi_col_key& k) const { return k.hash; }
};

static bool does_keys_have_nulls(std::vector<array_info*> const& key_cols) {
    for (auto key_col : key_cols) {
        if ((key_col->arr_type == bodo_array_type::NUMPY &&
             (key_col->dtype == Bodo_CTypes::FLOAT32 ||
              key_col->dtype == Bodo_CTypes::FLOAT64)) ||
            key_col->arr_type == bodo_array_type::STRING ||
            key_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
            return true;
        }
    }
    return false;
}

static bool does_row_has_nulls(std::vector<array_info*> const& key_cols,
                               int64_t const& i) {
    for (auto key_col : key_cols) {
        if (key_col->arr_type == bodo_array_type::STRING ||
            key_col->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
            if (!GetBit((uint8_t*)key_col->null_bitmask, i)) return true;
        } else if (key_col->arr_type == bodo_array_type::NUMPY) {
            if ((key_col->dtype == Bodo_CTypes::FLOAT32 &&
                 isnan(key_col->at<float>(i))) ||
                (key_col->dtype == Bodo_CTypes::FLOAT64 &&
                 isnan(key_col->at<double>(i))))
                return true;
        }
    }
    return false;
}

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
    std::vector<array_info*> key_cols = std::vector<array_info*>(
        table.columns.begin(), table.columns.begin() + table.num_keys);
    uint32_t seed = 0xb0d01288;
    uint32_t* hashes = hash_keys(key_cols, seed);

    row_to_group.reserve(table.nrows());
    // start at 1 because I'm going to use 0 to mean nothing was inserted yet
    // in the map (but note that the group values I record in the output go from
    // 0 to num_groups - 1)
    int next_group = 1;
    MAP_CONTAINER<multi_col_key, int64_t, key_hash> key_to_group;
    bool key_is_nullable = false;
    if (check_for_null_keys) {
        key_is_nullable = does_keys_have_nulls(key_cols);
    }
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
    std::vector<int64_t> row_to_group(table->nrows());
    std::vector<int64_t> group_to_first_row;
    std::vector<int64_t> next_row_in_group(table->nrows(), -1);
    std::vector<int64_t> active_group_repr;
    std::vector<int64_t> list_missing;

    std::vector<array_info*> key_cols = std::vector<array_info*>(
        table->columns.begin(), table->columns.begin() + table->num_keys);
    uint32_t seed = 0xb0d01288;
    uint32_t* hashes = hash_keys(key_cols, seed);

    bool key_is_nullable = does_keys_have_nulls(key_cols);
    // start at 1 because I'm going to use 0 to mean nothing was inserted yet
    // in the map (but note that the group values I record in the output go from
    // 0 to num_groups - 1)
    int next_group = 1;
    MAP_CONTAINER<multi_col_key, int64_t, key_hash> key_to_group;
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
 * The isnan operation done for all types.
 *
 * @param eVal: the T value
 * @return true if for integer and true/false for floating point
 */
template <typename T>
inline typename std::enable_if<!std::is_floating_point<T>::value, bool>::type
isnan_T(char* ptr) {
    return false;
}
template <typename T>
inline typename std::enable_if<std::is_floating_point<T>::value, bool>::type
isnan_T(char* ptr) {
    T* ptr_d = (T*)ptr;
    return isnan(*ptr_d);
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
template <typename T>
void cumulative_computation_T(array_info* arr, array_info* out_arr,
                                  grouping_info const& grp_inf,
                                  int32_t const& ftype, bool const& skipna) {
    size_t num_group = grp_inf.group_to_first_row.size();
    if (arr->arr_type == bodo_array_type::STRING) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "There is no median for the string case");
        return;
    }
    size_t siztype = numpy_item_size[arr->dtype];
    auto cum_computation =
        [&](std::function<std::pair<bool, T>(int64_t)> const& get_entry,
            std::function<void(int64_t, std::pair<bool, T> const&)> const&
                set_entry) -> void {
        for (size_t igrp = 0; igrp < num_group; igrp++) {
            int64_t i = grp_inf.group_to_first_row[igrp];
            T initVal;
            if (ftype == Bodo_FTypes::cumsum) initVal = 0;
            if (ftype == Bodo_FTypes::cummin) initVal = std::numeric_limits<T>::max();
            if (ftype == Bodo_FTypes::cummax) initVal = std::numeric_limits<T>::min();
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
        T eVal_nan = GetTentry<T>(RetrieveNaNentry(arr->dtype).data());
        std::pair<bool, T> pairNaN{true, eVal_nan};
        for (auto& idx_miss : grp_inf.list_missing)
            set_entry(idx_miss, pairNaN);
    };

    if (arr->arr_type == bodo_array_type::NUMPY) {
        cum_computation(
            [=](int64_t pos) -> std::pair<bool, T> {
                char* ptr = arr->data1 + pos * siztype;
                bool isna = isnan_T<T>(ptr);
                T eVal = GetTentry<T>(ptr);
                return {isna, eVal};
            },
            [=](int64_t pos, std::pair<bool, T> const& ePair) -> void {
                out_arr->at<T>(pos) = ePair.second;
            });
    }
    if (arr->arr_type == bodo_array_type::NULLABLE_INT_BOOL) {
        uint8_t* null_bitmask_i = (uint8_t*)arr->null_bitmask;
        uint8_t* null_bitmask_o = (uint8_t*)out_arr->null_bitmask;
        cum_computation(
            [=](int64_t pos) -> std::pair<bool, T> {
                char* ptr = arr->data1 + pos * siztype;
                return {!GetBit(null_bitmask_i, pos), GetTentry<T>(ptr)};
            },
            [=](int64_t pos, std::pair<bool, T> const& ePair) -> void {
                SetBitTo(null_bitmask_o, pos, !ePair.first);
                out_arr->at<T>(pos) = ePair.second;
            });
    }
}

void cumulative_computation(array_info* arr, array_info* out_arr,
                                grouping_info const& grp_inf,
                                int32_t const& ftype, bool const& skipna) {
    if (arr->dtype == Bodo_CTypes::INT8)
        return cumulative_computation_T<int8_t>(arr, out_arr, grp_inf,
                                                ftype, skipna);
    if (arr->dtype == Bodo_CTypes::UINT8)
        return cumulative_computation_T<uint8_t>(arr, out_arr, grp_inf,
                                                 ftype, skipna);

    if (arr->dtype == Bodo_CTypes::INT16)
        return cumulative_computation_T<int16_t>(arr, out_arr, grp_inf,
                                                 ftype, skipna);
    if (arr->dtype == Bodo_CTypes::UINT16)
        return cumulative_computation_T<uint16_t>(arr, out_arr, grp_inf,
                                                  ftype, skipna);

    if (arr->dtype == Bodo_CTypes::INT32)
        return cumulative_computation_T<int32_t>(arr, out_arr, grp_inf,
                                                 ftype, skipna);
    if (arr->dtype == Bodo_CTypes::UINT32)
        return cumulative_computation_T<uint32_t>(arr, out_arr, grp_inf,
                                                  ftype, skipna);

    if (arr->dtype == Bodo_CTypes::INT64)
        return cumulative_computation_T<int64_t>(arr, out_arr, grp_inf,
                                                 ftype, skipna);
    if (arr->dtype == Bodo_CTypes::UINT64)
        return cumulative_computation_T<uint64_t>(arr, out_arr, grp_inf,
                                                  ftype, skipna);

    if (arr->dtype == Bodo_CTypes::FLOAT32)
        return cumulative_computation_T<float>(arr, out_arr, grp_inf, ftype,
                                               skipna);
    if (arr->dtype == Bodo_CTypes::FLOAT64)
        return cumulative_computation_T<double>(arr, out_arr, grp_inf,
                                                ftype, skipna);
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
    std::function<bool(size_t)> isnan_entry;
    size_t siztype = numpy_item_size[arr->dtype];
    if (arr->arr_type == bodo_array_type::STRING) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "There is no median for the string case");
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
            SET_CONTAINER<int64_t, std::function<size_t(int64_t)>,
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
    if (arr->arr_type == bodo_array_type::STRING) {
        uint32_t* in_offsets = (uint32_t*)arr->data2;
        uint8_t* null_bitmask = (uint8_t*)arr->null_bitmask;
        uint32_t seed = 0xb0d01280;

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
            SET_CONTAINER<int64_t, std::function<size_t(int64_t)>,
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
            SET_CONTAINER<int64_t, std::function<size_t(int64_t)>,
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
template <typename T, int ftype>
void apply_to_column(array_info* in_col, array_info* out_col,
                     std::vector<array_info*>& aux_cols,
                     const grouping_info& grp_info) {
    switch (in_col->arr_type) {
        case bodo_array_type::NUMPY:
            if (ftype == Bodo_FTypes::mean) {
                array_info* count_col = aux_cols[0];
                for (int64_t i = 0; i < in_col->length; i++)
                    if (grp_info.row_to_group[i] != -1)
                        mean_agg<T>::apply(
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
                        var_agg<T>::apply(
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
                        count_agg<T>::apply(
                            out_col->at<int64_t>(grp_info.row_to_group[i]),
                            in_col->at<T>(i));
            } else {
                for (int64_t i = 0; i < in_col->length; i++)
                    if (grp_info.row_to_group[i] != -1)
                        aggfunc<T, ftype>::apply(
                            out_col->at<T>(grp_info.row_to_group[i]),
                            in_col->at<T>(i));
            }
            return;
        // for strings, we are only supporting count for now, and count function
        // works for strings because the input value doesn't matter
        case bodo_array_type::STRING:
            switch (ftype) {
                case Bodo_FTypes::count:
                    for (int64_t i = 0; i < in_col->length; i++) {
                        if ((grp_info.row_to_group[i] != -1) &&
                            GetBit((uint8_t*)in_col->null_bitmask, i))
                            count_agg<T>::apply(
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
                        if ((grp_info.row_to_group[i] != -1) &&
                            GetBit(null_bitmask_i, i)) {
                            uint32_t start_offset = offsets[i];
                            uint32_t end_offset = offsets[i + 1];
                            uint32_t len = end_offset - start_offset;
                            int64_t i_grp = grp_info.row_to_group[i];
                            std::string val(&data[start_offset], len);
                            if (GetBit(null_bitmask_o, i_grp)) {
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
                    // Doing the additional needed allocations
                    // FIXME: this code leaks memory
                    // delete[] out_col->data1;
                    out_col->data1 = new char[nb_char];
                    out_col->n_sub_elems = nb_char;
                    // Writing the strings in output
                    char* data_o = out_col->data1;
                    uint32_t* offsets_o = (uint32_t*)out_col->data2;
                    uint32_t pos = 0;
                    for (int64_t i_grp = 0; i_grp < int64_t(num_groups);
                         i_grp++) {
                        offsets_o[i_grp] = pos;
                        if (GetBit(null_bitmask_o, i_grp)) {
                            int len = ListString[i_grp].size();
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
                            count_agg<T>::apply(
                                out_col->at<int64_t>(grp_info.row_to_group[i]),
                                in_col->at<T>(i));
                    }
                    return;
                case Bodo_FTypes::mean:
                    for (int64_t i = 0; i < in_col->length; i++) {
                        if ((grp_info.row_to_group[i] != -1) &&
                            GetBit((uint8_t*)in_col->null_bitmask, i)) {
                            mean_agg<T>::apply(
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
                            var_agg<T>::apply(in_col->at<T>(i),
                                              aux_cols[0]->at<uint64_t>(
                                                  grp_info.row_to_group[i]),
                                              aux_cols[1]->at<double>(
                                                  grp_info.row_to_group[i]),
                                              aux_cols[2]->at<double>(
                                                  grp_info.row_to_group[i]));
                    }
                    return;
                default:
                    for (int64_t i = 0; i < in_col->length; i++) {
                        if ((grp_info.row_to_group[i] != -1) &&
                            GetBit((uint8_t*)in_col->null_bitmask, i)) {
                            aggfunc<T, ftype>::apply(
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
    if (in_col->arr_type == bodo_array_type::STRING) {
        switch (ftype) {
            // NOTE: The int template argument is not used in this call to
            // apply_to_column
            case Bodo_FTypes::sum:
                return apply_to_column<int, Bodo_FTypes::sum>(
                    in_col, out_col, aux_cols, grp_info);
            case Bodo_FTypes::min:
                return apply_to_column<int, Bodo_FTypes::min>(
                    in_col, out_col, aux_cols, grp_info);
            case Bodo_FTypes::max:
                return apply_to_column<int, Bodo_FTypes::max>(
                    in_col, out_col, aux_cols, grp_info);
        }
    }
    if (ftype == Bodo_FTypes::count) {
        switch (in_col->dtype) {
            case Bodo_CTypes::FLOAT32:
                // data will only be used to check for nans
                return apply_to_column<float, Bodo_FTypes::count>(
                    in_col, out_col, aux_cols, grp_info);
            case Bodo_CTypes::FLOAT64:
                // data will only be used to check for nans
                return apply_to_column<double, Bodo_FTypes::count>(
                    in_col, out_col, aux_cols, grp_info);
            default:
                // data will be ignored in this case, so type doesn't matter
                return apply_to_column<int8_t, Bodo_FTypes::count>(
                    in_col, out_col, aux_cols, grp_info);
        }
    }

    switch (in_col->dtype) {
        case Bodo_CTypes::_BOOL:
            switch (ftype) {
                case Bodo_FTypes::min:
                    return apply_to_column<bool, Bodo_FTypes::min>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<bool, Bodo_FTypes::max>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<bool, Bodo_FTypes::prod>(
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
                    return apply_to_column<int8_t, Bodo_FTypes::sum>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<int8_t, Bodo_FTypes::min>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<int8_t, Bodo_FTypes::max>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<int8_t, Bodo_FTypes::prod>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<int8_t, Bodo_FTypes::mean>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int8_t, Bodo_FTypes::var>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::UINT8:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<uint8_t, Bodo_FTypes::sum>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<uint8_t, Bodo_FTypes::min>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<uint8_t, Bodo_FTypes::max>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<uint8_t, Bodo_FTypes::prod>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<uint8_t, Bodo_FTypes::mean>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<uint8_t, Bodo_FTypes::var>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::INT16:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<int16_t, Bodo_FTypes::sum>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<int16_t, Bodo_FTypes::min>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<int16_t, Bodo_FTypes::max>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<int16_t, Bodo_FTypes::prod>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<int16_t, Bodo_FTypes::mean>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int16_t, Bodo_FTypes::var>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::UINT16:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<uint16_t, Bodo_FTypes::sum>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<uint16_t, Bodo_FTypes::min>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<uint16_t, Bodo_FTypes::max>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<uint16_t, Bodo_FTypes::prod>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<uint16_t, Bodo_FTypes::mean>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<uint16_t, Bodo_FTypes::var>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::INT32:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<int32_t, Bodo_FTypes::sum>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<int32_t, Bodo_FTypes::min>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<int32_t, Bodo_FTypes::max>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<int32_t, Bodo_FTypes::prod>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<int32_t, Bodo_FTypes::mean>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int32_t, Bodo_FTypes::var>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::UINT32:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<uint32_t, Bodo_FTypes::sum>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<uint32_t, Bodo_FTypes::min>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<uint32_t, Bodo_FTypes::max>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<uint32_t, Bodo_FTypes::prod>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<uint32_t, Bodo_FTypes::mean>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<uint32_t, Bodo_FTypes::var>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::INT64:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<int64_t, Bodo_FTypes::sum>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<int64_t, Bodo_FTypes::min>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<int64_t, Bodo_FTypes::max>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<int64_t, Bodo_FTypes::prod>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<int64_t, Bodo_FTypes::mean>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<int64_t, Bodo_FTypes::var>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::UINT64:
        case Bodo_CTypes::DATE:
        case Bodo_CTypes::DATETIME:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<uint64_t, Bodo_FTypes::sum>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<uint64_t, Bodo_FTypes::min>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<uint64_t, Bodo_FTypes::max>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<uint64_t, Bodo_FTypes::prod>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<uint64_t, Bodo_FTypes::mean>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<uint64_t, Bodo_FTypes::var>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::FLOAT32:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<float, Bodo_FTypes::sum>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<float, Bodo_FTypes::min>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<float, Bodo_FTypes::max>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<float, Bodo_FTypes::prod>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<float, Bodo_FTypes::mean>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean_eval:
                    return apply_to_column<float, Bodo_FTypes::mean_eval>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<float, Bodo_FTypes::var>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var_eval:
                    return apply_to_column<float, Bodo_FTypes::var_eval>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::std_eval:
                    return apply_to_column<float, Bodo_FTypes::std_eval>(
                        in_col, out_col, aux_cols, grp_info);
            }
        case Bodo_CTypes::FLOAT64:
            switch (ftype) {
                case Bodo_FTypes::sum:
                    return apply_to_column<double, Bodo_FTypes::sum>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::min:
                    return apply_to_column<double, Bodo_FTypes::min>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::max:
                    return apply_to_column<double, Bodo_FTypes::max>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::prod:
                    return apply_to_column<double, Bodo_FTypes::prod>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean:
                    return apply_to_column<double, Bodo_FTypes::mean>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::mean_eval:
                    return apply_to_column<double, Bodo_FTypes::mean_eval>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var:
                case Bodo_FTypes::std:
                    return apply_to_column<double, Bodo_FTypes::var>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::var_eval:
                    return apply_to_column<double, Bodo_FTypes::var_eval>(
                        in_col, out_col, aux_cols, grp_info);
                case Bodo_FTypes::std_eval:
                    return apply_to_column<double, Bodo_FTypes::std_eval>(
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
        if (ftype == Bodo_FTypes::min || ftype == Bodo_FTypes::max)
            // if input is all nulls, max and min output will be null
            InitializeBitMask((uint8_t*)out_col->null_bitmask, out_col->length,
                              false);
        else
            // for other functions (count, sum, etc.) output will never be null
            InitializeBitMask((uint8_t*)out_col->null_bitmask, out_col->length,
                              true);
    }
    if (out_col->arr_type == bodo_array_type::STRING) {
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
                case Bodo_CTypes::DATE:
                case Bodo_CTypes::DATETIME:
                    std::fill((uint64_t*)out_col->data1,
                              (uint64_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<uint64_t>::max());
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
                case Bodo_CTypes::STRING:
                    // Nothing to initilize with in the case of strings.
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
                case Bodo_CTypes::DATE:
                case Bodo_CTypes::DATETIME:
                    std::fill((uint64_t*)out_col->data1,
                              (uint64_t*)out_col->data1 + out_col->length,
                              std::numeric_limits<uint64_t>::min());
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
                case Bodo_CTypes::STRING:
                    // nothing to initialize in the case of strings
                    return;
                default:
                    Bodo_PyErr_SetString(PyExc_RuntimeError,
                                         "unsupported/not implemented");
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
        // calling this modifies arr_type and dtype
        get_groupby_output_dtype(ftype, arr_type, dtype, false);
        out_cols.push_back(alloc_array(num_groups, 1, arr_type, dtype, 0));
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
            // calling this modifies arr_type and dtype
            get_groupby_output_dtype(combine_ftype, arr_type, dtype, false);
            out_cols.push_back(alloc_array(num_groups, 1, arr_type, dtype, 0));
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
            free_array(a);
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
            alloc_array(num_groups, 1, bodo_array_type::NUMPY,
                        Bodo_CTypes::FLOAT64, 0);  // for sum and result
        array_info* c2 = alloc_array(num_groups, 1, bodo_array_type::NUMPY,
                                     Bodo_CTypes::UINT64, 0);  // for counts
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
                alloc_array(num_groups, 1, bodo_array_type::NUMPY,
                            Bodo_CTypes::FLOAT64, 0);  // for result
            out_cols.push_back(col);
            update_cols.push_back(col);
        }
        array_info* count_col = alloc_array(
            num_groups, 1, bodo_array_type::NUMPY, Bodo_CTypes::UINT64, 0);
        array_info* mean_col = alloc_array(
            num_groups, 1, bodo_array_type::NUMPY, Bodo_CTypes::FLOAT64, 0);
        array_info* m2_col = alloc_array(num_groups, 1, bodo_array_type::NUMPY,
                                         Bodo_CTypes::FLOAT64, 0);
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
        array_info* col = alloc_array(num_groups, 1, bodo_array_type::NUMPY,
                                      Bodo_CTypes::FLOAT64, 0);  // for result
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
            out_cols.push_back(alloc_array(num_groups, 1, arr_type, dtype, 0));
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
            out_cols.push_back(alloc_array(num_groups, 1, arr_type, dtype, 0));
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
        out_cols.push_back(
            alloc_array(in_col->length, 1, in_col->arr_type, in_col->dtype, 0));
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
    GroupbyPipeline(table_info* _in_table, int64_t _num_keys, bool _is_parallel,
                    int* ftypes, int* func_offsets, int* _udf_nredvars,
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
        int return_index_i = return_index;

        // NOTE cumulative operations (cumsum, cumprod, etc.) cannot be mixed
        // with non cumulative ops. This is checked at compile time in
        // aggregate.py

        for (int i = 0;
             i < func_offsets[in_table->ncols() - num_keys - return_index_i];
             i++) {
            int ftype = ftypes[i];
            if (ftype == Bodo_FTypes::nunique || ftype == Bodo_FTypes::median ||
                ftype == Bodo_FTypes::cumsum || ftype == Bodo_FTypes::cumprod ||
                ftype == Bodo_FTypes::cummin || ftype == Bodo_FTypes::cummax) {
                // these operations first require shuffling the data to
                // gather all rows with the same key in the same process
                if (is_parallel) shuffle_before_update = true;
                // these operations require extended group info
                req_extended_group_info = true;
                if (ftype == Bodo_FTypes::cumsum || ftype == Bodo_FTypes::cummin ||
                    ftype == Bodo_FTypes::cumprod || ftype == Bodo_FTypes::cummax)
                    cumulative_op = true;
                break;
            }
        }
        if (shuffle_before_update) in_table = shuffle_table(in_table, num_keys);
        // a combine operation is only necessary when data is distributed and
        // a shuffle has not been done at the start of the groupby pipeline
        do_combine = is_parallel && !shuffle_before_update;

        // construct the column sets, one for each (input_column, func) pair
        // ftypes is an array of function types received from generated code,
        // and has one ftype for each (input_column, func) pair
        int k = 0;
        for (int64_t i = num_keys; i < in_table->ncols() - return_index_i;
             i++, k++) {
            array_info* col = in_table->columns[i];
            int start = func_offsets[k];
            int end = func_offsets[k + 1];
            for (int j = start; j != end; j++) {
                col_sets.push_back(
                    makeColSet(col, ftypes[j], do_combine, skipna));
            }
        }
    }

    ~GroupbyPipeline() {
        for (auto col_set : col_sets) delete col_set;
    }

    /**
     * This is the main control flow of the Groupby pipeline.
     */
    table_info* run() {
        update();
        if (shuffle_before_update)
            // in_table was created in C++ during shuffling and not needed
            // anymore
            delete_table_free_arrays(in_table);
        if (is_parallel && !shuffle_before_update) {
            shuffle();
            combine();
        }
        eval();
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
    BasicColSet* makeColSet(array_info* in_col, int ftype, bool do_combine,
                            bool skipna) {
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
        in_table->num_keys = num_keys;
        if (req_extended_group_info) {
            bool consider_missing = cumulative_op;
            grp_info = get_group_info_iterate(in_table, consider_missing);
        } else
            get_group_info(*in_table, grp_info.row_to_group,
                           grp_info.group_to_first_row, true);
        num_groups = grp_info.group_to_first_row.size();

        update_table = cur_table = new table_info();
        if (cumulative_op)
            num_keys = 0;  // there are no key columns in output of cumulative operations
        else
            alloc_init_keys(in_table, update_table);

        for (auto col_set : col_sets) {
            col_set->alloc_update_columns(num_groups, update_table->columns);
            col_set->update(grp_info);
        }
        if (return_index)
            update_table->columns.push_back(
                copy_array(in_table->columns.back()));
        if (n_udf > 0)
            udf_info.update(in_table, update_table,
                            grp_info.row_to_group.data());
    }

    /**
     * Shuffles the update table and updates the column sets with the newly
     * shuffled table.
     */
    void shuffle() {
        table_info* shuf_table = shuffle_table(update_table, num_keys);
        delete_table_free_arrays(update_table);
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
        delete_table_free_arrays(update_table);
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
        delete cur_table;
        return out_table;
    }

    /**
     * Allocate and fill key columns, based on grouping info. It uses the
     * values of key columns from from_table to populate out_table.
     */
    void alloc_init_keys(table_info* from_table, table_info* out_table) {
        for (int64_t i = 0; i < num_keys; i++) {
            const array_info* key_col = (*from_table)[i];
            array_info* new_key_col;
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
                new_key_col = alloc_array(num_groups, n_chars,
                                          key_col->arr_type, key_col->dtype, 0);

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
            } else {
                new_key_col = alloc_array(num_groups, 1, key_col->arr_type,
                                          key_col->dtype, 0);
                int64_t dtype_size = numpy_item_size[key_col->dtype];
                for (size_t j = 0; j < num_groups; j++)
                    memcpy(new_key_col->data1 + j * dtype_size,
                           key_col->data1 +
                               grp_info.group_to_first_row[j] * dtype_size,
                           dtype_size);
            }
            out_table->columns.push_back(new_key_col);
        }
    }

   private:
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
    // when one of the groupby functions is median/nunique/cumsum/cumprod/cummin/cummax
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
};

table_info* groupby_and_aggregate(table_info* in_table, int64_t num_keys,
                                  int* ftypes, int* func_offsets,
                                  int* udf_nredvars, bool is_parallel,
                                  bool skipdropna, bool return_key,
                                  bool return_index, void* update_cb,
                                  void* combine_cb, void* eval_cb,
                                  table_info* udf_dummy_table) {
    GroupbyPipeline groupby(in_table, num_keys, is_parallel, ftypes,
                            func_offsets, udf_nredvars, udf_dummy_table,
                            (udf_table_op_fn)update_cb,
                            (udf_table_op_fn)combine_cb, (udf_eval_fn)eval_cb,
                            skipdropna, return_key, return_index);

    return groupby.run();
}
