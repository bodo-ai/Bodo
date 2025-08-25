#pragma once

#include "../_bodo_common.h"

/**
 * This file defines the various structs and typedefs used to
 * manage UDF types in groupby.
 */

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
    std::shared_ptr<table_info> udf_table_dummy;
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
 * @brief A function that computes the UDF result on a group in streaming
 * groupby.
 *
 */
using stream_udf_t = array_info*(table_info*);
