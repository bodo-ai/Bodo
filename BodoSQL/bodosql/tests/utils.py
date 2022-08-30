"""
Infrastructure used to test correctness.
"""
# Copyright (C) 2022 Bodo Inc. All rights reserved.
import os
import re
import warnings
from decimal import Decimal
from enum import Enum

import bodosql
import numba
import numpy as np
import pandas as pd
from mpi4py import MPI
from pyspark.sql.functions import col
from pyspark.sql.types import (
    ByteType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
    StructField,
    StructType,
)

import bodo


class InputDist(Enum):
    """
    Enum used to represent the various
    distributed analysis options for input
    data.
    """

    REP = 0
    OneD = 1
    OneDVar = 2


def bodo_version_older(year, month, release_number):
    """
    Helper function to check if the current Bodo version's last (mini)release
    is older than the Bodo (mini)release specified by
    (year, month, release_number). The release_number is 0 for the monthly release.

    All + values are ignored for local testing (and treated as equal to the last
    (mini)-release), so tests will be skipped locally and on CI.

    This should be used for tests that require changes on Bodo because BodoSQL CI won't
    be able to check the results until a new Bodo Minirelease.
    """
    version_string = bodo.__version__
    # Version string is always the leading digits and .
    match = re.search(r"[^(\d|\.)]", version_string)
    if match is not None:
        version_string = version_string[: match.start()]
    # Bodo version parts are split by .
    parts = version_string.split(".")
    if len(parts) == 2:
        version_tuple = (int(parts[0]), int(parts[1]), 0)
    else:
        version_tuple = (int(parts[0]), int(parts[1]), int(parts[2]))

    return version_tuple < (year, month, release_number)


def check_query(
    query,
    dataframe_dict,
    spark,
    named_params=None,
    check_names=True,
    check_dtype=True,
    sort_output=True,
    expected_output=None,
    convert_columns_bytearray=None,
    convert_columns_string=None,
    convert_columns_timedelta=None,
    convert_columns_decimal=None,
    convert_float_nan=False,
    convert_columns_bool=None,
    return_codegen=False,
    return_seq_dataframe=False,
    run_dist_tests=True,
    only_python=False,
    only_jit_seq=False,
    only_jit_1D=False,
    only_jit_1DVar=False,
    spark_dataframe_dict=None,
    equivalent_spark_query=None,
    optimize_calcite_plan=True,
    spark_input_cols_to_cast=None,
    spark_output_cols_to_cast=None,
    pyspark_schemas=None,
    named_params_timedelta_interval=False,
    convert_nullable_bodosql=True,
    use_table_format=None,
    use_dict_encoded_strings=None,
    is_out_distributed=True,
):
    """
    Evaluates the correctness of a BodoSQL query by comparing SparkSQL
    as a baseline. Correctness is determined by converting both outputs
    to Pandas dataframes and checking equality.

    This function returns a dictionary of key value pairs depending
    on the information requested to be returned. Currently the following
    key value pairs are possible:

        'pandas_code': Generated Pandas code
            - Set by return_codegen=True

        'output_df': Output DataFrame when running the query sequentially
            - Set by return_seq_dataframe=True

    @params:
        query: The SQL query to execute.

        dataframe_dict: A dictionary mapping Table names -> DataFrames.
            These tables will be placed inside a SparkSQL/BodoSQL context
            for query execution.

        spark: SparkSession used to generate expected output.

        named_params: Dictionary of mapping constant values to Bodo variable
            names used in query. These are used to support caching variable
            changes in Bodo. Spark queries need to replace these variables
            with the constant values if they exist.

        check_names: Compare BodoSQL and SparkSQL names for equality.
            This is useful for checking aliases.

        check_dtype: Compare BodoSQL and SparkSQL types for equality.

        sort_output: Compare the tables after sorting the results. Most
            queries need to be sorted because the order isn't defined.

        expected_output: The expected result of running the query. If
            this value is None the result is computed by executing the
            query in SparkSQL.

        convert_columns_bytearray: Convert the given list of
            columns to bytes types. This is because BodoSQL always
            outputs bytes types, but Spark outputs binaryarray.

        convert_columns_string: Convert the given list of string
            columns to bytes types. This is needed when the SUBSTR function
            in Spark is run on binary data and returns a string slice
            rather than a binary slice.

        convert_columns_timedelta: Convert the given list of
            columns to td64 types. This is because SparkSQL doesn't
            natively support timedelta types and converts the result
            to int64. This argument is only used if expected_output=None.

        convert_float_nan: Convert NaN values in float columns to None.
            This is used when Spark and Bodo will have different output
            types.

        convert_columns_bool: Convert NaN values to None by setting datatype
            to boolean.

        convert_columns_decimal: Convert the given list of
            decimal columns to float64 types.

        return_codegen: Return the pandas code produced.

        return_seq_dataframe: Return a sequential version of the output df.

        run_dist_tests: Should distributed tests be run. These should be skipped
            if it's not possible to distribute the input.

        only_python: Create the BodoSQL context only in Python. This is
            useful in debugging.

        only_jit_seq: Create the BodoSQL context only in a jit function.
            Input data is REP. This is useful in debugging.

        only_jit_1D: Create the BodoSQL context only in a jit function.
            Input data is 1D. This is useful in debugging.

        only_jit_1DVar: Create the BodoSQL context only in a jit function.
            Input data is 1DVar. This is useful in debugging.

        spark_dataframe_dict: A dictionary mapping Table names -> DataFrames
            used just by SparkSQL context. This is used when Spark/Bodo
            types differ.

        equivalent_spark_query: The query string to use with spark to create
            the expected output, if different from the query string used with
            BodoSQL.

        optimize_calcite_plan: Controls if the calcite plan used to construct
            the pandas code is optimized or not

        spark_input_cols_to_cast: A hashmap of dataframe --> list of tuples in the form
            (colname, typename) as strings. The specified columns in the specified
            dataframe are cast to the specified types before the expected
            output is computed. For example spark_input_cols_to_cast = {"table1": [("A", "int")]}
            would cause column A of table1 to be cast to type "int" before the SQL query
            is ran on the spark dataframe.
            This is used in certain situations when spark performs incorrect casting
            from pandas types (Specifically, all pandas integers types are converted
            to bigint, which is an invalid type for certain functions in spark,
            such as DATE_ADD, and FORMAT_NUMBER)

        spark_output_cols_to_cast: A List of tuples in the form (colname, typename).
            Casts the specified columns in the spark output to the the specified types.
            for example, a value of ["A", "Int64"] would cause the expected_ouput["A"]
            to be cast to int64. Will raise errors if the column is not present in the output,
            or if the type is not valid when used with series.astype.

        pyspark_schemas: Dictionary of Pyspark Schema for each DataFrame provided in dataframe_dict.
            If this value is None or the DataFrame is not provided in the dict, this value
            is ignored. This value is primarily used for larger examples/benchmarks that
            cannot be inferred properly by Spark (i.e. TPCxBB).

        named_params_timedelta_interval: Should Pyspark Interval literals be generated
            for timedelta named parameters.

        convert_nullable_bodosql: Should BodoSQL nullable integers be converted to Object dtype with None.

        use_table_format: flag for loading dataframes in table format for testing.
            If None, tests both formats.

        use_dict_encoded_strings: flag for loading string arrays in dictionary-encoded
            format for testing.
            If None, tests both formats if input arguments have string arrays.
        is_out_distributed: flag to whether gather the output before equality checking.
            Default True.
    """
    # Determine which bodo versions to run
    if only_python:
        run_python, run_jit_seq, run_jit_1D, run_jit_1DVar = True, False, False, False
    elif only_jit_seq:
        run_python, run_jit_seq, run_jit_1D, run_jit_1DVar = False, True, False, False
    elif only_jit_1D:
        run_python, run_jit_seq, run_jit_1D, run_jit_1DVar = False, False, True, False
    elif only_jit_1DVar:
        run_python, run_jit_seq, run_jit_1D, run_jit_1DVar = False, False, False, True
    elif not run_dist_tests:
        run_python, run_jit_seq, run_jit_1D, run_jit_1DVar = True, True, False, False
    else:
        run_python, run_jit_seq, run_jit_1D, run_jit_1DVar = True, True, True, True

    n_pes = bodo.get_size()

    if not numba.core.config.DEVELOPER_MODE:
        # avoid running sequential tests on multi-process configs to save time
        if n_pes > 1:
            run_python = False
            run_jit_seq = False
        # Avoid running parallel tests on single-process configs to save time
        elif n_pes == 1:
            run_jit_1D = False
            run_jit_1DVar = False

    # If a user sets BODOSQL_TESTING_DEBUG, we print the
    # unoptimized plan, optimized plan, and the Pandas code
    debug_mode = os.environ.get("BODOSQL_TESTING_DEBUG", False)
    if debug_mode:

        print("Query:")
        print(query)
        bc = bodosql.BodoSQLContext(dataframe_dict)
        print("Unoptimized Plan:")
        print(bc.generate_unoptimized_plan(query, named_params))
        print("Optimized Plan:")
        print(bc.generate_plan(query, named_params))
        print("Pandas Code:")
        print(bc.convert_to_pandas(query, named_params))

    # Determine the spark output.
    if expected_output is None:
        spark.catalog.clearCache()
        # If Spark specific inputs aren't provided, use the same
        # as BodoSQL
        if spark_dataframe_dict is None:
            spark_dataframe_dict = dataframe_dict

        for table_name, df in spark_dataframe_dict.items():
            spark.catalog.dropTempView(table_name)
            df = convert_nullable_object(df)

            if pyspark_schemas is None:
                schema = None
            else:
                schema = pyspark_schemas.get(table_name, None)
            spark_df = spark.createDataFrame(df, schema=schema)
            if (
                spark_input_cols_to_cast != None
                and table_name in spark_input_cols_to_cast
            ):
                for colname, typename in spark_input_cols_to_cast[table_name]:
                    spark_df = spark_df.withColumn(colname, col(colname).cast(typename))
            spark_df.createTempView(table_name)
        # Always run Spark on just 1 core for efficiency
        if bodo.get_rank() == 0:
            # If an equivalent query is provided we use that
            # instead of the original spark query
            if equivalent_spark_query is None:
                spark_query = query
            else:
                spark_query = equivalent_spark_query
            # If named params are provided we need to replace them
            # with literals in the Spark query.
            if named_params is not None:
                spark_query = replace_spark_named_params(
                    spark_query, named_params, named_params_timedelta_interval
                )
            if debug_mode:
                print("PySpark Query: ")
                print(spark_query)
            expected_output = spark.sql(spark_query).toPandas()

        comm = MPI.COMM_WORLD
        try:
            expected_output = comm.bcast(expected_output, root=0)
            errors = comm.allgather(None)
        except Exception as e:
            # If we can an exception, raise it on all processes.
            errors = comm.allgather(e)
        for e in errors:
            if isinstance(e, Exception):
                raise e

        if spark_output_cols_to_cast != None:
            for colname, typ in spark_output_cols_to_cast:
                if colname in expected_output.columns:
                    expected_output[colname] = expected_output[colname].astype(typ)
                else:
                    print("Column and typecast")
                    print(colname, typ)
                    print("expected_output")
                    print(expected_output)
                    raise Exception(
                        "Error, didn't find column to cast in expected output"
                    )
        if convert_columns_bytearray:
            expected_output = convert_spark_bytearray(
                expected_output, convert_columns_bytearray
            )
        if convert_columns_string:
            expected_output = convert_spark_string(
                expected_output, convert_columns_string
            )
        if convert_columns_timedelta:
            expected_output = convert_spark_timedelta(
                expected_output, convert_columns_timedelta
            )
        if convert_float_nan:
            expected_output = convert_spark_nan_none(expected_output)
        if convert_columns_decimal:
            expected_output = convert_spark_decimal(
                expected_output, convert_columns_decimal
            )
        if convert_columns_bool:
            expected_output = convert_spark_bool(expected_output, convert_columns_bool)

    if run_python:
        check_query_python(
            query,
            dataframe_dict,
            named_params,
            check_names,
            check_dtype,
            sort_output,
            expected_output,
            optimize_calcite_plan,
            convert_nullable_bodosql,
        )

    check_query_jit(
        run_jit_seq,
        run_jit_1D,
        run_jit_1DVar,
        query,
        dataframe_dict,
        named_params,
        check_names,
        check_dtype,
        sort_output,
        expected_output,
        optimize_calcite_plan,
        convert_nullable_bodosql,
        use_table_format,
        use_dict_encoded_strings,
        is_out_distributed=is_out_distributed,
    )

    result = dict()

    if return_codegen or return_seq_dataframe:
        bc = bodosql.BodoSQLContext(dataframe_dict)

    # Return Pandas code if requested
    if return_codegen:
        if optimize_calcite_plan:
            result["pandas_code"] = bc.convert_to_pandas(query, named_params)
        else:
            result["pandas_code"] = bc._convert_to_pandas_unoptimized(
                query, named_params
            )

    # Return sequential output if requested
    if return_seq_dataframe:
        if optimize_calcite_plan:
            result["output_df"] = bc.sql(query, named_params)
        else:
            result["output_df"] = bc._test_sql_unoptimized(query, named_params)

    return result


def check_query_jit(
    run_jit_seq,
    run_jit_1D,
    run_jit_1DVar,
    query,
    dataframe_dict,
    named_params,
    check_names,
    check_dtype,
    sort_output,
    expected_output,
    optimize_calcite_plan,
    convert_nullable_bodosql,
    use_table_format,
    use_dict_encoded_strings,
    is_out_distributed,
):
    """
    Evaluates the correctness of a BodoSQL query against expected_output.
    This function creates the BodoSQL context in a JIT function.

    @params:

        run_jit_seq: pass arguments as REP and make the function sequential

        run_jit_1D: pass arguments as 1D

        run_jit_1DVar: pass arguments as 1D_Var

        query: The SQL query to execute.

        dataframe_dict: A dictionary mapping Table names -> DataFrames.
            These tables will be placed inside a BodoSQL context
            for query execution.

        named_params: Dictionary of mapping constant values to Bodo variable
            names used in query.

        check_names: Compare BodoSQL and expected_output names for equality.

        check_dtype: Compare BodoSQL and expected_output types for equality.

        sort_output: Compare the tables after sorting the results. Most
            queries need to be sorted because the order isn't defined.

        expected_output: The expected result of running the query.

        convert_nullable_bodosql: Should BodoSQL nullable integers be converted to Object dtype with None.

        use_table_format: flag for loading dataframes in table format for testing.
            If None, tests both formats.

        use_dict_encoded_strings: flag for loading string arrays in dictionary-encoded
            format for testing.
            If None, tests both formats if input arguments have string arrays.
    """

    saved_TABLE_FORMAT_THRESHOLD = bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD
    saved_use_dict_str_type = bodo.hiframes.boxing._use_dict_str_type
    try:
        # test table format for dataframes (non-table format tested below if flag is
        # None)
        if use_table_format is None or use_table_format:
            bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD = 0

        # test dict-encoded string arrays if flag is set (dict-encoded tested below if
        # flag is None)
        if use_dict_encoded_strings:
            bodo.hiframes.boxing._use_dict_str_type = True

        if run_jit_seq:
            check_query_jit_seq(
                query,
                dataframe_dict,
                named_params,
                check_names,
                check_dtype,
                sort_output,
                expected_output,
                optimize_calcite_plan,
                convert_nullable_bodosql,
            )
        if run_jit_1D:
            check_query_jit_1D(
                query,
                dataframe_dict,
                named_params,
                check_names,
                check_dtype,
                sort_output,
                expected_output,
                optimize_calcite_plan,
                convert_nullable_bodosql,
                is_out_distributed,
            )
        if run_jit_1DVar:
            check_query_jit_1DVar(
                query,
                dataframe_dict,
                named_params,
                check_names,
                check_dtype,
                sort_output,
                expected_output,
                optimize_calcite_plan,
                convert_nullable_bodosql,
                is_out_distributed,
            )
    finally:
        bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD = saved_TABLE_FORMAT_THRESHOLD
        bodo.hiframes.boxing._use_dict_str_type = saved_use_dict_str_type

    # test non-table format case
    if use_table_format is None:
        check_query_jit(
            run_jit_seq,
            run_jit_1D,
            run_jit_1DVar,
            query,
            dataframe_dict,
            named_params,
            check_names,
            check_dtype,
            sort_output,
            expected_output,
            optimize_calcite_plan,
            convert_nullable_bodosql,
            use_table_format=False,
            use_dict_encoded_strings=use_dict_encoded_strings,
            is_out_distributed=is_out_distributed,
        )

    # test dict-encoded string type if there is any string array in input
    if use_dict_encoded_strings is None and any(
        any(t == bodo.string_array_type for t in bodo.tests.utils._typeof(df).data)
        for df in dataframe_dict.values()
    ):
        check_query_jit(
            run_jit_seq,
            run_jit_1D,
            run_jit_1DVar,
            query,
            dataframe_dict,
            named_params,
            check_names,
            check_dtype,
            sort_output,
            expected_output,
            optimize_calcite_plan,
            convert_nullable_bodosql,
            # the default case use_table_format=None already tests
            # use_table_format=False above so we just test use_table_format=True for it
            use_table_format=True if use_table_format is None else use_table_format,
            use_dict_encoded_strings=True,
            is_out_distributed=is_out_distributed,
        )


def check_query_python(
    query,
    dataframe_dict,
    named_params,
    check_names,
    check_dtype,
    sort_output,
    expected_output,
    optimize_calcite_plan,
    convert_nullable_bodosql,
):
    """
    Evaluates the correctness of a BodoSQL query against expected_output.
    This function creates the BodoSQL context in regular Python.

    @params:
        query: The SQL query to execute.

        dataframe_dict: A dictionary mapping Table names -> DataFrames.
            These tables will be placed inside a BodoSQL context
            for query execution.

        named_params: Dictionary of mapping constant values to Bodo variable
            names used in query.

        check_names: Compare BodoSQL and expected_output names for equality.

        check_dtype: Compare BodoSQL and expected_output types for equality.

        sort_output: Compare the tables after sorting the results. Most
            queries need to be sorted because the order isn't defined.

        expected_output: The expected result of running the query.

        convert_nullable_bodosql: Should BodoSQL nullable integers be converted to Object dtype with None.
    """
    bc = bodosql.BodoSQLContext(dataframe_dict)
    if optimize_calcite_plan:
        bodosql_output = bc.sql(query, named_params)
    else:
        bodosql_output = bc._test_sql_unoptimized(query, named_params)
    _check_query_equal(
        bodosql_output,
        expected_output,
        check_names,
        check_dtype,
        sort_output,
        False,
        "Sequential Python Test Failed",
        convert_nullable_bodosql,
    )


def check_query_jit_seq(
    query,
    dataframe_dict,
    named_params,
    check_names,
    check_dtype,
    sort_output,
    expected_output,
    optimize_calcite_plan,
    convert_nullable_bodosql,
):
    """
    Evaluates the correctness of a BodoSQL query against expected_output.
    This function creates the BodoSQL context in a jit function using
    code generation and keeps input data as REP.

    @params:
        query: The SQL query to execute.

        dataframe_dict: A dictionary mapping Table names -> DataFrames.
            These tables will be placed inside a BodoSQL context
            for query execution.

        named_params: Dictionary of mapping constant values to Bodo variable
            names used in query.

        check_names: Compare BodoSQL and expected_output names for equality.

        check_dtype: Compare BodoSQL and expected_output types for equality.

        sort_output: Compare the tables after sorting the results. Most
            queries need to be sorted because the order isn't defined.

        expected_output: The expected result of running the query.

        convert_nullable_bodosql: Should BodoSQL nullable integers be converted to Object dtype with None.
    """
    bodosql_output = _run_jit_query(
        query, dataframe_dict, named_params, InputDist.REP, optimize_calcite_plan, False
    )
    _check_query_equal(
        bodosql_output,
        expected_output,
        check_names,
        check_dtype,
        sort_output,
        False,
        "Sequential JIT Test Failed",
        convert_nullable_bodosql,
    )


def check_query_jit_1D(
    query,
    dataframe_dict,
    named_params,
    check_names,
    check_dtype,
    sort_output,
    expected_output,
    optimize_calcite_plan,
    convert_nullable_bodosql,
    is_out_distributed,
):
    """
    Evaluates the correctness of a BodoSQL query against expected_output.
    This function creates the BodoSQL context in a jit function using
    code generation and distributes input data as 1D.

    @params:
        query: The SQL query to execute.

        dataframe_dict: A dictionary mapping Table names -> DataFrames.
            These tables will be placed inside a BodoSQL context
            for query execution.

        named_params: Dictionary of mapping constant values to Bodo variable
            names used in query.

        check_names: Compare BodoSQL and expected_output names for equality.

        check_dtype: Compare BodoSQL and expected_output types for equality.

        sort_output: Compare the tables after sorting the results. Most
            queries need to be sorted because the order isn't defined.

        expected_output: The expected result of running the query.

        convert_nullable_bodosql: Should BodoSQL nullable integers be converted to Object dtype with None.
    """
    bodosql_output = _run_jit_query(
        query,
        dataframe_dict,
        named_params,
        InputDist.OneD,
        optimize_calcite_plan,
        is_out_distributed,
    )
    if is_out_distributed:
        bodosql_output = bodo.gatherv(bodosql_output)
    _check_query_equal(
        bodosql_output,
        expected_output,
        check_names,
        check_dtype,
        sort_output,
        is_out_distributed,
        "1D Parallel JIT Test Failed",
        convert_nullable_bodosql,
    )


def check_query_jit_1DVar(
    query,
    dataframe_dict,
    named_params,
    check_names,
    check_dtype,
    sort_output,
    expected_output,
    optimize_calcite_plan,
    convert_nullable_bodosql,
    is_out_distributed,
):
    """
    Evaluates the correctness of a BodoSQL query against expected_output.
    This function creates the BodoSQL context in a jit function using
    code generation and distributes input data as 1DVar.

    @params:
        query: The SQL query to execute.

        dataframe_dict: A dictionary mapping Table names -> DataFrames.
            These tables will be placed inside a BodoSQL context
            for query execution.

        named_params: Dictionary of mapping constant values to Bodo variable
            names used in query.

        check_names: Compare BodoSQL and expected_output names for equality.

        check_dtype: Compare BodoSQL and expected_output types for equality.

        sort_output: Compare the tables after sorting the results. Most
            queries need to be sorted because the order isn't defined.

        expected_output: The expected result of running the query.

        convert_nullable_bodosql: Should BodoSQL nullable integers be converted to Object dtype with None.
    """
    bodosql_output = _run_jit_query(
        query,
        dataframe_dict,
        named_params,
        InputDist.OneDVar,
        optimize_calcite_plan,
        is_out_distributed,
    )
    if is_out_distributed:
        bodosql_output = bodo.gatherv(bodosql_output)
    _check_query_equal(
        bodosql_output,
        expected_output,
        check_names,
        check_dtype,
        sort_output,
        is_out_distributed,
        "1DVar Parallel JIT Test Failed",
        convert_nullable_bodosql,
    )


def _run_jit_query(
    query,
    dataframe_dict,
    named_params,
    input_dist,
    optimize_calcite_plan,
    is_out_distributed,
):
    """
    Helper function to generate and run a JIT based BodoSQL query with a given
    dataframe_dict. This function distributes the input data based upon the given input_dist,
    which is either REP, OneD, or OneDVar.

    @params:
        query: The SQL query to execute.

        dataframe_dict: A dictionary mapping Table names -> DataFrames.
            These tables will be placed inside a BodoSQL context
            for query execution.

        named_params: Dictionary of mapping constant values to Bodo variable
            names used in query.

        input_dist: How the input data should be distributed. Either REP,
            1D or 1DVar. All input DataFrames are presumed to have the same
            distribution

    @returns:
        The Pandas dataframe (possibly distributed) from running the query.
    """
    # Compute named params lists if they exist.
    if named_params is not None:
        keys_list = list(named_params.keys())
        values_list = list(named_params.values())
    else:
        keys_list = []
        values_list = []

    # Generate the BodoSQLContext with func_text so we can use jit code
    params = ",".join([f"e{i}" for i in range(len(dataframe_dict))] + keys_list)
    func_text = f"def test_impl(query, {params}):\n"
    func_text += "    bc = bodosql.BodoSQLContext(\n"
    func_text += "        {\n"
    args = [query]
    for i, key in enumerate(dataframe_dict.keys()):
        if input_dist == InputDist.OneD:
            args.append(_get_dist_df(dataframe_dict[key]))
        elif input_dist == InputDist.OneDVar:
            args.append(_get_dist_df(dataframe_dict[key], var_length=True))
        else:
            args.append(dataframe_dict[key])
        func_text += f"            '{key}': e{i},\n"
    args = args + values_list
    func_text += "        }\n"
    func_text += "    )\n"
    if optimize_calcite_plan:
        func_text += f"    result = bc.sql(query"
    else:
        func_text += f"    result = bc._test_sql_unoptimized(query"
    if keys_list:
        func_text += ", {"
        for key in keys_list:
            func_text += f"'{key}': {key}, "
        func_text += "}"
    func_text += ")\n"
    func_text += "    return result\n"
    locs = {}
    exec(func_text, {"bodo": bodo, "bodosql": bodosql}, locs)
    func = locs["test_impl"]
    all_args_distributed_block = input_dist == InputDist.OneD
    all_args_distributed_varlength = input_dist == InputDist.OneDVar
    can_be_dist = input_dist != InputDist.REP
    bodosql_output = bodo.jit(
        func,
        all_args_distributed_block=all_args_distributed_block,
        all_args_distributed_varlength=all_args_distributed_varlength,
        all_returns_distributed=(is_out_distributed and can_be_dist),
        returns_maybe_distributed=(is_out_distributed and can_be_dist),
        args_maybe_distributed=can_be_dist,
    )(*args)
    return bodosql_output


def _check_query_equal(
    bodosql_output,
    expected_output,
    check_names,
    check_dtype,
    sort_output,
    is_out_distributed,
    failure_message,
    convert_nullable_bodosql,
):
    """
    Evaluates the BodoSQL output against the expected output.

    @params:
        bodosql_output: The output from bodosql.

        expected_output: The expected result of running the query.

        check_names: Compare BodoSQL and expected_output names for equality.

        check_dtype: Compare BodoSQL and expected_output types for equality.

        sort_output: Compare the tables after sorting the results. Most
            queries need to be sorted because the order isn't defined.

        is_out_distributed: Is bodosql_output possibly distributed?

        failure_message: Message used to describe the test type when a failure
            occurs.

        convert_nullable_bodosql: Should BodoSQL nullable integers be converted to Object dtype with None.

    """
    if sort_output:
        bodosql_output = bodosql_output.sort_values(
            bodosql_output.columns.tolist()
        ).reset_index(drop=True)
        expected_output = expected_output.sort_values(
            expected_output.columns.tolist()
        ).reset_index(drop=True)
    else:
        # BodoSQL doesn't maintain a matching index, so we reset the index
        # for all tests
        bodosql_output = bodosql_output.reset_index(drop=True)
        expected_output = expected_output.reset_index(drop=True)
    # check_names=False doesn't seem to work inside pd.testing.assert_frame_equal, so manually rename
    if not check_names:
        bodosql_output.columns = range(len(bodosql_output.columns))
        expected_output.columns = range(len(expected_output.columns))

    passed = 1
    n_ranks = bodo.get_size()
    # print("BODO")
    # print(bodosql_output)
    # print("SPARK")
    # print(expected_output)
    # only rank 0 should check if gatherv() called on output
    if not is_out_distributed or bodo.get_rank() == 0:
        passed = _test_equal_guard(
            bodosql_output, expected_output, check_dtype, convert_nullable_bodosql
        )
    n_passed = reduce_sum(passed)
    assert n_passed == n_ranks, failure_message


def _test_equal_guard(
    bodosql_output,
    expected_output,
    check_dtype,
    convert_nullable_bodosql,
):
    passed = 1
    try:
        # convert bodosql output to a value that can be compared with Spark
        if convert_nullable_bodosql:
            bodosql_output = convert_nullable_object(bodosql_output)
        pd.testing.assert_frame_equal(bodosql_output, expected_output, check_dtype)
    except Exception as e:
        print(e)
        passed = 0
    return passed


def check_efficient_join(pandas_code):
    """
    Checks that given pandas_code doesn't contain any joins that required
    merging the whole table on a dummy column.
    """
    assert "$__bodo_dummy__" not in pandas_code


def convert_spark_bool(df, columns):
    """
    Converts Spark Boolean object columns to boolean type to match BodoSQL.
    """
    df[columns] = df[columns].astype("boolean")
    return df


def convert_spark_bytearray(df, columns):
    """
    Converts Spark ByteArray columns to bytes to match BodoSQL.
    """
    df[columns] = df[columns].apply(
        lambda x: [bytes(y) if isinstance(y, bytearray) else y for y in x],
        axis=1,
        result_type="expand",
    )
    return df


def convert_spark_string(df, columns):
    """
    Converts Spark String columns to bytes to match BodoSQL.
    """
    df[columns] = df[columns].apply(
        lambda x: [y.encode("utf-8") if isinstance(y, str) else y for y in x],
        axis=1,
        result_type="expand",
    )
    return df


def convert_spark_decimal(df, columns):
    """
    Converts Spark DecimalArray columns to floats to match BodoSQL.
    """
    df[columns] = df[columns].apply(
        lambda x: [np.float64(y) if isinstance(y, Decimal) else y for y in x],
        axis=1,
        result_type="expand",
    )
    return df


def convert_spark_timedelta(df, columns):
    """
    Function the converts an Integer/Float DataFrame that should have been
    a timedelta column in Spark back to timedelta. This is used to compare
    Bodo results (which accepts Timedelta) with Spark (which do not).
    """
    df_proj = df[columns]
    df[columns] = (
        df_proj.fillna(0)
        .astype(np.int64)
        .where(pd.notnull(df_proj), np.timedelta64("nat"))
        .astype("timedelta64[ns]")
    )
    return df


def convert_spark_nan_none(df):
    """
    Function the converts Float NaN values to None. This is used because Spark
    may convert nullable integers to floats.
    """
    df = df.astype(object).where(pd.notnull(df), None)
    return df


def convert_nullable_object(df):
    """
    Function the converts a DataFrame with a nullable column to an
    Object Datatype replaces pd.NA with None. This is used so Spark
    can interpret the results.
    """
    if any([isinstance(x, pd.core.arrays.integer._IntegerDtype) for x in df.dtypes]):
        return df.astype(object).where(pd.notnull(df), None)
    return df


def generate_plan(query, dataframe_dict):
    """
    Return a plan for a given query with the dictionary
    for a BodoSQLContext.
    """
    bc = bodosql.BodoSQLContext(dataframe_dict)
    return bc.generate_plan(query)


def check_plan_length(query, dataframe_dict, expected_length):
    """
    Helper function that verifies a plan length for queries
    with expected plan sizes.
    """
    plan = generate_plan(query, dataframe_dict)
    assert plan.count("\n") == expected_length


@bodo.jit
def reduce_sum(val):
    """
    Reduce on an integer with Bodo.
    """
    # Copied exactly from Bodo repo (because it can't be imported).
    sum_op = np.int32(bodo.libs.distributed_api.Reduce_Type.Sum.value)
    return bodo.libs.distributed_api.dist_reduce(val, np.int32(sum_op))


def _get_dist_df(df, var_length=False, check_typing_issues=True):
    """get distributed chunk for a dataframe df on current rank (for input to test functions)"""
    # Copied from Bodo repo (because it can't be imported).
    # This code has been modified to only support the DataFrame case (because BodoSQL)
    # always works with Tables.

    start, end = get_start_end(df.shape[0])
    # for var length case to be different than regular 1D in chunk sizes, add
    # one extra element to last processor
    if var_length and bodo.get_size() >= 2:
        if bodo.get_rank() == bodo.get_size() - 2:
            end -= 1
        if bodo.get_rank() == bodo.get_size() - 1:
            start -= 1

    out_val = df.iloc[start:end]
    if check_typing_issues:
        _check_typing_issues(out_val)
    return out_val


@bodo.jit(cache=True)
def get_start_end(n):
    """
    Get the starting and ending indices for distributing data.
    """
    # Copied exactly from Bodo repo (because it can't be imported).
    rank = bodo.libs.distributed_api.get_rank()
    n_pes = bodo.libs.distributed_api.get_size()
    start = bodo.libs.distributed_api.get_start(n, n_pes, rank)
    end = bodo.libs.distributed_api.get_end(n, n_pes, rank)
    return start, end


def _check_typing_issues(val):
    """Raises an error if there is a typing issue for value 'val'.
    Runs bodo typing on value and converts warnings to errors.
    """
    # Copied exactly from Bodo repo (because it can't be imported).
    comm = MPI.COMM_WORLD
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("error")
            bodo.typeof(val)
        errors = comm.allgather(None)
    except Exception as e:
        # The typing issue typically occurs on only a subset of processes,
        # because the process got a chunk of data from Python that is empty
        # or that we cannot currently type correctly.
        # To avoid a hang, we need to notify every rank of the error.
        errors = comm.allgather(e)
    for e in errors:
        if isinstance(e, Exception):
            raise e


def replace_spark_named_params(query, named_params, use_interval):
    """
    Function that takes a query used by Spark (and possibly BodoSQL)
    and replaces any instances of named parameters with literal values
    stored in the named_parms dictionary.

    For example, if the query was:

        select A from table1 limit @a

    with named_params:

        {'a': 5, 'b': 1}

    Then this would return the string:

        select A from table1 limit 5

    This is then the query which should be run in Spark.
    """
    replace_dict = {
        "@" + key: get_pyspark_literal(value, use_interval)
        for key, value in named_params.items()
    }
    for key, value in replace_dict.items():
        query = query.replace(key, value)
    return query


def get_pyspark_literal(value, use_interval):
    """
    Takes a scalar value which is Python, Numpy, or Pandas type and returns
    a string that contains a literal value that can be used by SparkSQL.
    """
    if isinstance(value, (int, float, np.integer, np.floating)):
        return str(value)
    elif isinstance(value, str):
        return f"'{value}'"
    elif isinstance(value, pd.Timedelta):
        # TODO: Generate an interval string in Spark
        if use_interval:
            # Generate an interval string in Spark
            return f"INTERVAL {value.days} DAYS {value.seconds // 3600} HOURS {(value.seconds // 60) % 60} MINUTES {value.seconds % 60} SECONDS {value.microseconds // 1000} MILLISECONDS {value.microseconds % 1000} MICROSECONDS"
        # Spark treats the timedelta values as integers so just return value.value
        return str(value.value)
    elif isinstance(value, pd.Timestamp):
        # Generate a timestamp string in Spark, normalizing tz if necessary
        return f"TIMESTAMP '{value.tz_convert(None) if value.tz else value}'"
    elif isinstance(value, pd.DateOffset):
        # Generate an interval string in Spark with only years and months
        return f"INTERVAL {getattr(value, 'years', 0)} YEARS {getattr(value, 'months', 0)} MONTHS"
    else:
        raise ValueError(
            "Named Parameter converstion to Pyspark Literal not supported."
        )


def shrink_data(ctx, n, keys_to_shrink=None, keys_to_not_shrink=None):
    output_dict = dict()
    if keys_to_shrink is None:
        keys_to_shrink = ctx.keys()

    for key in keys_to_shrink:
        if keys_to_not_shrink != None and key in keys_to_not_shrink:
            continue
        output_dict[key] = ctx[key].head(n)
    return output_dict


def check_num_parquet_readers(bodo_func, n):
    """
    Check that there are exactly n ParquetReaders in the IR
    after typing pass.
    """
    fir = bodo_func.overloads[bodo_func.signatures[0]].metadata["preserved_ir"]
    num_pq_readers = 0
    for block in fir.blocks.values():
        for stmt in block.body:
            if isinstance(stmt, bodo.ir.parquet_ext.ParquetReader):
                num_pq_readers += 1

    assert (
        num_pq_readers == n
    ), "Number of ParquetReaders in the IR doesn't match the expectation"


def create_pyspark_schema_from_dataframe(df):
    """Constructs a Pyspark schema for an appropriately typed
    DataFrame. This is used for tests whose output depends on
    maintaining precision.
    """
    int_byte_type_map = {
        1: ByteType(),
        2: ShortType(),
        4: IntegerType(),
        8: LongType(),
    }
    float_byte_type_map = {4: FloatType(), 8: DoubleType()}

    field_list = []
    for i, col in enumerate(df.columns):
        dtype = df.dtypes[i]
        if np.issubdtype(dtype, np.integer):
            pyspark_type = int_byte_type_map[dtype.itemsize]
        elif np.issubdtype(dtype, np.floating):
            pyspark_type = float_byte_type_map[dtype.itemsize]
        else:
            raise TypeError("Type mapping to Pyspark Schema not implemented yet.")
        field_list.append(StructField(col, pyspark_type, True))
    return StructType(field_list)
