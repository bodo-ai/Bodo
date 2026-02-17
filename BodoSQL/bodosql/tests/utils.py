"""
Infrastructure used to test correctness.
"""

import os
import re
import warnings
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import duckdb
import numba
import numpy as np
import pandas as pd
import pyarrow as pa
import pyspark
import sqlglot
from mpi4py import MPI
from pyspark.sql.types import (
    ByteType,
    DayTimeIntervalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

import bodo
import bodosql
from bodo.tests.utils import (
    _convert_float_to_nullable_float,
    _get_dist_arg,
    _test_equal,
    gen_unique_table_id,
)
from bodo.tests.utils_jit import reduce_sum

if TYPE_CHECKING:
    from pyspark.sql.session import SparkSession


class InputDist(Enum):
    """
    Enum used to represent the various
    distributed analysis options for input
    data.
    """

    REP = 0
    OneD = 1
    OneDVar = 2


def check_query(
    query: str,
    dataframe_dict: dict[str, pd.DataFrame],
    spark: Optional["SparkSession"],
    named_params: dict[str, Any] | None = None,
    bind_variables: tuple[Any] | None = None,
    check_names: bool = True,
    check_dtype: bool = False,
    sort_output: bool = True,
    expected_output: pd.DataFrame | None = None,
    convert_columns_bytearray: list[str] | None = None,
    convert_columns_string: list[str] | None = None,
    convert_columns_timedelta: list[str] | None = None,
    convert_columns_decimal: list[str] | None = None,
    convert_input_to_nullable_float: bool = True,
    convert_expected_output_to_nullable_float: bool = True,
    convert_columns_bool: list[str] | None = None,
    convert_columns_tz_naive: list[str] | None = None,
    return_codegen: bool = False,
    return_seq_dataframe: bool = False,
    run_dist_tests: bool = True,
    only_python: bool = False,
    only_jit_seq: bool = False,
    only_jit_1D: bool = False,
    only_jit_1DVar: bool | None = None,
    spark_dataframe_dict: dict[str, pd.DataFrame] | None = None,
    equivalent_spark_query: str | None = None,
    spark_input_cols_to_cast: dict[str, dict[str, str]] | None = None,
    pyspark_schemas: dict[str, pyspark.sql.types.StructType] | None = None,
    named_params_timedelta_interval: bool = False,
    convert_nullable_bodosql: bool = True,
    use_table_format: bool | None = True,
    use_dict_encoded_strings: bool | None = None,
    is_out_distributed: bool = True,
    check_typing_issues: bool = True,
    use_map_arrays: bool = False,
    atol: float = 1e-08,
    rtol: float = 1e-05,
    convert_columns_to_pandas: bool = False,
    session_tz: str | None = None,
    enable_timestamp_tz: bool = False,
    use_duckdb: bool = False,
):
    """
    Evaluates the correctness of a BodoSQL query by comparing SparkSQL
    as a baseline. Correctness is determined by converting both outputs
    to Pandas DataFrames and checking equality.

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

        bind_variables: Tuple of Python variables used as bind variables in the
        query. These queries do not support Spark.

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
            outputs bytes types, but Spark outputs binary array.

        convert_columns_string: Convert the given list of string
            columns to bytes types. This is needed when the SUBSTR function
            in Spark is run on binary data and returns a string slice
            rather than a binary slice.

        convert_columns_timedelta: Convert the given list of
            columns to td64 types. This is because SparkSQL doesn't
            natively support timedelta types and converts the result
            to int64. This argument is only used if expected_output=None.

        convert_input_to_nullable_float: Convert float columns in inputs
            to nullable float if the nullable float global flag is enabled.

        convert_expected_output_to_nullable_float: Convert float columns in expected
            output to nullable float if the nullable float global flag is enabled.

        convert_columns_bool: Convert NaN values to None by setting datatype
            to boolean.

        convert_columns_tz_naive(Optional[List[str]]): List of columns
            in the input DataFrame(s) that use tz-aware timestamp data.
            Spark will produce an incorrect output for these columns, so
            we convert them to tz-naive before running the query in Spark.

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
        check_typing_issues: raise an error if there is a typing issue for input args.
        Runs bodo typing on arguments and converts warnings to errors.

        use_map_arrays: Flag for forcing all input dict-object arrays to be unboxed as map arrays.

        atol: absolute tolerance used for approximately-equal calculations

        rtol: relative tolerance used for approximately-equal calculations

        convert_columns_to_pandas: Pandas does not support well some datatype such as decimal,
        list of strings or in general arrow data type. It typically fails at sorting. In that case
        using convert_columns_to_pandas=True will convert the columns to a string format which may help.

        session_tz: the string representation of the timezone to use for TIMESTAMP_LTZ.

        enable_timestamp_tz: True if TIMESTAMP_TZ support should be enabled.

        use_duckdb: True if DuckDB should be used to generate the expected output. If so, we will either
            use the equivalent_spark_query as is or use SQLGlot to transpile the base query to DuckDB.
    """

    # We allow the environment flag BODO_TESTING_ONLY_RUN_1D_VAR to change the default
    # testing behavior, to test with only 1D_var. This environment variable is set in our
    # AWS PR CI environment
    if only_jit_1DVar is None and not (only_python or only_jit_1D or only_jit_seq):
        only_jit_1DVar = (
            os.environ.get("BODO_TESTING_ONLY_RUN_1D_VAR", None) is not None
        )

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

    # Avoid testing BodoSQLContext.sql() call in regular Python on CI to run faster.
    # It's not used in platform deployment and also not likely to fail independently.
    if not only_python and not numba.core.config.DEVELOPER_MODE:
        run_python = False

    # Avoid testing 1D on CI to run faster. It's not likely to fail independent of
    # 1D_Var.
    if not only_jit_1D and not numba.core.config.DEVELOPER_MODE:
        run_jit_1D = False

    # avoid running sequential tests on multi-process configs to save time
    # is_out_distributed=False may lead to avoiding parallel runs and seq run
    # Ideally we would like to also restrict running parallel tests when we have a single rank,
    # but this can lead to test coverage issues when running with BODO_TESTING_ONLY_RUN_1D_VAR
    # on AWS PR CI, where we only run with a single rank
    if (
        n_pes > 1
        and not numba.core.config.DEVELOPER_MODE
        and is_out_distributed is not False
    ):
        run_jit_seq = False
        run_python = False
    elif (
        n_pes == 1
        and not numba.core.config.DEVELOPER_MODE
        and os.environ.get("BODO_TESTING_PIPELINE_HAS_MULTI_RANK_TEST", False)
    ):
        # We only skip the parallel tests when running on one rank if we know that
        # there exists another worker running on multiple ranks
        run_jit_1D = False
        run_jit_1DVar = False

    # Only run Python mode when spawn testing is enabled since others aren't applicable
    if bodo.tests.utils.test_spawn_mode_enabled or bodosql.use_cpp_backend:
        run_python, run_jit_seq, run_jit_1D, run_jit_1DVar = True, False, False, False
        if only_jit_seq:
            warnings.warn(
                "check_query: no tests are being run since spawn or BodoSQL C++ backend enabled and only_jit_seq=True."
            )
            return

    if not (run_jit_seq or run_jit_1D or run_jit_1DVar or run_python):
        warnings.warn("check_query: No tests are being run.")
        return

    # If a user sets BODOSQL_TESTING_DEBUG, we print the
    # unoptimized plan, optimized plan, and the Pandas code
    debug_mode = os.environ.get("BODOSQL_TESTING_DEBUG", False)
    if debug_mode:
        print("Query:")
        print(query)
        bc = bodosql.BodoSQLContext(dataframe_dict, default_tz=session_tz)
        print("Plan:")
        print(bc.generate_plan(query, named_params, bind_variables))
        print("Pandas Code:")
        print(bc.convert_to_pandas(query, named_params, bind_variables))

    if expected_output is None and (
        bodo.get_rank() == 0 and spark is None and not use_duckdb
    ):
        raise ValueError(
            "Either `expected_output` or `spark` argument must be set to not None or use_duckdb must be True."
        )

    # Determine the Spark output.
    if expected_output is None and use_duckdb is False:
        # Only run Spark on one rank to run faster and avoid Spark issues
        if bodo.get_rank() == 0:
            spark.catalog.clearCache()
            # If Spark specific inputs aren't provided, use the same
            # as BodoSQL
            if spark_dataframe_dict is None:
                spark_dataframe_dict = dataframe_dict

            for table_name, df in spark_dataframe_dict.items():
                spark.catalog.dropTempView(table_name)
                df = convert_nullable_object(df)
                if convert_columns_tz_naive:
                    df = remove_tz_columns_spark(df, convert_columns_tz_naive)

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
                        spark_df = spark_df.withColumn(
                            colname, pyspark.sql.functions.col(colname).cast(typename)
                        )
                spark_df.createTempView(table_name)

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
        if convert_input_to_nullable_float:
            dataframe_dict = {
                c: _convert_float_to_nullable_float(df)
                for c, df in dataframe_dict.items()
            }
        if convert_expected_output_to_nullable_float:
            expected_output = _convert_float_to_nullable_float(expected_output)
        if convert_columns_bool:
            expected_output = convert_spark_bool(expected_output, convert_columns_bool)

    elif expected_output is None:
        if bodo.get_rank() == 0:
            # Use SQLGlot to transpile the query to DuckDB syntax
            # TODO: We can use this for Spark as well
            q = (
                sqlglot.transpile(query, read="snowflake", write="duckdb")[0]
                if equivalent_spark_query is None
                else equivalent_spark_query
            )

            # Start a temporary DuckDB connection and register the tables from the dataframe_dict
            conn = duckdb.connect(":memory:")
            for name, df in dataframe_dict.items():
                if convert_columns_tz_naive:
                    df = remove_tz_columns_spark(df, convert_columns_tz_naive)
                conn.register(name, df)

            # Compute the expected output
            # We need to convert to arrow and then Pandas to maintain some types (decimal, timestamp)
            expected_output = conn.sql(q).arrow().read_pandas()

        # Distribute the expected output and handle errors
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

    if convert_columns_decimal:
        expected_output = convert_spark_decimal(
            expected_output, convert_columns_decimal
        )

    if run_python:
        check_query_python(
            query,
            dataframe_dict,
            named_params,
            bind_variables,
            check_names,
            check_dtype,
            sort_output,
            expected_output,
            convert_nullable_bodosql,
            convert_columns_to_pandas,
            session_tz,
            atol=atol,
            rtol=rtol,
        )

    check_query_jit(
        run_jit_seq,
        run_jit_1D,
        run_jit_1DVar,
        query,
        dataframe_dict,
        named_params,
        bind_variables,
        check_names,
        check_dtype,
        sort_output,
        expected_output,
        convert_nullable_bodosql,
        use_table_format,
        use_dict_encoded_strings,
        session_tz=session_tz,
        enable_timestamp_tz=enable_timestamp_tz,
        is_out_distributed=is_out_distributed,
        check_typing_issues=check_typing_issues,
        convert_columns_to_pandas=convert_columns_to_pandas,
        use_map_arrays=use_map_arrays,
        atol=atol,
        rtol=rtol,
    )

    result = {}

    if return_codegen or return_seq_dataframe:
        bc = bodosql.BodoSQLContext(dataframe_dict, default_tz=session_tz)

    # Return Pandas code if requested
    if return_codegen:
        result["pandas_code"] = bc.convert_to_pandas(
            query, named_params, bind_variables
        )

    # Return sequential output if requested
    if return_seq_dataframe:
        result["output_df"] = bc.sql(query, named_params, bind_variables)
    return result


def check_query_jit(
    run_jit_seq,
    run_jit_1D,
    run_jit_1DVar,
    query,
    dataframe_dict,
    named_params,
    bind_variables,
    check_names,
    check_dtype,
    sort_output,
    expected_output,
    convert_nullable_bodosql,
    use_table_format,
    use_dict_encoded_strings,
    is_out_distributed,
    check_typing_issues,
    convert_columns_to_pandas,
    use_map_arrays,
    session_tz,
    enable_timestamp_tz,
    atol: float = 1e-08,
    rtol: float = 1e-05,
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

        bind_variables: Tuple of Python values used as bind variables in a query.

        check_names: Compare BodoSQL and expected_output names for equality.

        check_dtype: Compare BodoSQL and expected_output types for equality.

        sort_output: Compare the tables after sorting the results. Most
            queries need to be sorted because the order isn't defined.

        expected_output: The expected result of running the query.

        convert_nullable_bodosql: Should BodoSQL nullable integers be converted to Object dtype with None.

        use_table_format: flag for loading DataFrames in table format for testing.
            If None, tests both formats.

        use_dict_encoded_strings: flag for loading string arrays in dictionary-encoded
            format for testing.
            If None, tests both formats if input arguments have string arrays.

        session_tz: the string representation of the timezone to use for TIMESTAMP_LTZ.

        atol: absolute tolerance used for approximately-equal calculations

        rtol: relative tolerance used for approximately-equal calculations

        check_typing_issues: raise an error if there is a typing issue for input args.
        Runs bodo typing on arguments and converts warnings to errors.

        convert_columns_to_pandas: Pandas does not support well some datatype such as decimal,
        list of strings or in general arrow data type. It typically fails at sorting. In that case
        using convert_columns_to_pandas=True will convert the columns to a string format which may help.

        enable_timestamp_tz: true if TIMESTAMP_TZ support should be enabled
    """

    saved_TABLE_FORMAT_THRESHOLD = bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD
    saved_use_dict_str_type = bodo.hiframes.boxing._use_dict_str_type
    saved_struct_size_limit = bodo.hiframes.boxing.struct_size_limit
    saved_enable_timestamp_tz = bodo.enable_timestamp_tz
    try:
        # test table format for dataframes (non-table format tested below if flag is
        # None)
        if use_table_format is None or use_table_format:
            bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD = 0

        # test dict-encoded string arrays if flag is set (dict-encoded tested below if
        # flag is None)
        if use_dict_encoded_strings:
            bodo.hiframes.boxing._use_dict_str_type = True

        # Test all dict-like arguments as map arrays (no structs) if flag is set
        if use_map_arrays:
            bodo.hiframes.boxing.struct_size_limit = -1

        bodo.enable_timestamp_tz = enable_timestamp_tz

        if run_jit_seq:
            check_query_jit_seq(
                query,
                dataframe_dict,
                named_params,
                bind_variables,
                check_names,
                check_dtype,
                sort_output,
                expected_output,
                convert_nullable_bodosql,
                check_typing_issues,
                convert_columns_to_pandas,
                session_tz,
                atol,
                rtol,
            )
        if run_jit_1D:
            check_query_jit_1D(
                query,
                dataframe_dict,
                named_params,
                bind_variables,
                check_names,
                check_dtype,
                sort_output,
                expected_output,
                convert_nullable_bodosql,
                is_out_distributed,
                check_typing_issues,
                convert_columns_to_pandas,
                session_tz,
                atol,
                rtol,
            )
        if run_jit_1DVar:
            check_query_jit_1DVar(
                query,
                dataframe_dict,
                named_params,
                bind_variables,
                check_names,
                check_dtype,
                sort_output,
                expected_output,
                convert_nullable_bodosql,
                is_out_distributed,
                check_typing_issues,
                convert_columns_to_pandas,
                session_tz,
                atol,
                rtol,
            )
    finally:
        bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD = saved_TABLE_FORMAT_THRESHOLD
        bodo.hiframes.boxing._use_dict_str_type = saved_use_dict_str_type
        bodo.hiframes.boxing.struct_size_limit = saved_struct_size_limit
        bodo.enable_timestamp_tz = saved_enable_timestamp_tz

    # test non-table format case
    if use_table_format is None:
        check_query_jit(
            run_jit_seq,
            run_jit_1D,
            run_jit_1DVar,
            query,
            dataframe_dict,
            named_params,
            bind_variables,
            check_names,
            check_dtype,
            sort_output,
            expected_output,
            convert_nullable_bodosql,
            session_tz=session_tz,
            enable_timestamp_tz=enable_timestamp_tz,
            use_table_format=False,
            use_dict_encoded_strings=use_dict_encoded_strings,
            is_out_distributed=is_out_distributed,
            check_typing_issues=check_typing_issues,
            convert_columns_to_pandas=convert_columns_to_pandas,
            use_map_arrays=use_map_arrays,
            atol=atol,
            rtol=rtol,
        )

    # test dict-encoded string type if there is any string array in input
    if use_dict_encoded_strings is None and any(
        any(
            t == bodo.types.string_array_type for t in bodo.tests.utils._typeof(df).data
        )
        for df in dataframe_dict.values()
    ):
        check_query_jit(
            run_jit_seq,
            run_jit_1D,
            run_jit_1DVar,
            query,
            dataframe_dict,
            named_params,
            bind_variables,
            check_names,
            check_dtype,
            sort_output,
            expected_output,
            convert_nullable_bodosql,
            session_tz=session_tz,
            enable_timestamp_tz=enable_timestamp_tz,
            # the default case use_table_format=None already tests
            # use_table_format=False above so we just test use_table_format=True for it
            use_table_format=True if use_table_format is None else use_table_format,
            use_dict_encoded_strings=True,
            is_out_distributed=is_out_distributed,
            check_typing_issues=check_typing_issues,
            convert_columns_to_pandas=convert_columns_to_pandas,
            use_map_arrays=use_map_arrays,
            atol=atol,
            rtol=rtol,
        )


def check_query_python(
    query,
    dataframe_dict,
    named_params,
    bind_variables,
    check_names,
    check_dtype,
    sort_output,
    expected_output,
    convert_nullable_bodosql,
    convert_columns_to_pandas,
    session_tz,
    atol: float = 1e-08,
    rtol: float = 1e-05,
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

        bind_variables: Tuple of Python values used as bind variables in a query.

        check_names: Compare BodoSQL and expected_output names for equality.

        check_dtype: Compare BodoSQL and expected_output types for equality.

        sort_output: Compare the tables after sorting the results. Most
            queries need to be sorted because the order isn't defined.

        expected_output: The expected result of running the query.

        convert_nullable_bodosql: Should BodoSQL nullable integers be converted to Object dtype with None.

        atol: absolute tolerance used for approximately-equal calculations

        rtol: relative tolerance used for approximately-equal calculations

        convert_columns_to_pandas: Pandas does not support well some datatype such as decimal,
        list of strings or in general arrow data type. It typically fails at sorting. In that case
        using convert_columns_to_pandas=True will convert the columns to a string format which may help.

        session_tz: the string representation of the timezone to use for TIMESTAMP_LTZ.
    """
    bc = bodosql.BodoSQLContext(dataframe_dict, default_tz=session_tz)
    jit_options = {}
    if bodo.tests.utils.test_spawn_mode_enabled:
        jit_options["spawn"] = True
    bodosql_output = bc.sql(query, named_params, bind_variables, **jit_options)
    _check_query_equal(
        bodosql_output,
        expected_output,
        check_names,
        check_dtype,
        sort_output,
        False,
        "Sequential Python Test Failed",
        convert_nullable_bodosql,
        convert_columns_to_pandas=convert_columns_to_pandas,
        atol=atol,
        rtol=rtol,
    )


def check_query_jit_seq(
    query,
    dataframe_dict,
    named_params,
    bind_variables,
    check_names,
    check_dtype,
    sort_output,
    expected_output,
    convert_nullable_bodosql,
    check_typing_issues,
    convert_columns_to_pandas,
    session_tz,
    atol: float = 1e-08,
    rtol: float = 1e-05,
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

        bind_variables: Tuple of Python values used as bind variables in a query.

        check_names: Compare BodoSQL and expected_output names for equality.

        check_dtype: Compare BodoSQL and expected_output types for equality.

        sort_output: Compare the tables after sorting the results. Most
            queries need to be sorted because the order isn't defined.

        expected_output: The expected result of running the query.

        convert_nullable_bodosql: Should BodoSQL nullable integers be converted to Object dtype with None.

        check_typing_issues: raise an error if there is a typing issue for input args.
        Runs bodo typing on arguments and converts warnings to errors.

        convert_columns_to_pandas: Pandas does not support well some datatype such as decimal,
        list of strings or in general arrow data type. It typically fails at sorting. In that case
        using convert_columns_to_pandas=True will convert the columns to a string format which may help.

        session_tz: the string representation of the timezone to use for TIMESTAMP_LTZ.

        atol: absolute tolerance used for approximately-equal calculations

        rtol: relative tolerance used for approximately-equal calculations
    """
    bodosql_output = _run_jit_query(
        query,
        dataframe_dict,
        named_params,
        bind_variables,
        InputDist.REP,
        False,
        session_tz,
        check_typing_issues=check_typing_issues,
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
        convert_columns_to_pandas=convert_columns_to_pandas,
        atol=atol,
        rtol=rtol,
    )


def check_query_jit_1D(
    query,
    dataframe_dict,
    named_params,
    bind_variables,
    check_names,
    check_dtype,
    sort_output,
    expected_output,
    convert_nullable_bodosql,
    is_out_distributed,
    check_typing_issues,
    convert_columns_to_pandas,
    session_tz,
    atol: float = 1e-08,
    rtol: float = 1e-05,
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

        bind_variables: Tuple of Python values used as bind variables in a query.

        check_names: Compare BodoSQL and expected_output names for equality.

        check_dtype: Compare BodoSQL and expected_output types for equality.

        sort_output: Compare the tables after sorting the results. Most
            queries need to be sorted because the order isn't defined.

        expected_output: The expected result of running the query.

        convert_nullable_bodosql: Should BodoSQL nullable integers be converted to Object dtype with None.

        check_typing_issues: raise an error if there is a typing issue for input args.
        Runs bodo typing on arguments and converts warnings to errors.

        convert_columns_to_pandas: Pandas does not support well some datatype such as decimal,
        list of strings or in general arrow data type. It typically fails at sorting. In that case
        using convert_columns_to_pandas=True will convert the columns to a string format which may help.

        session_tz: the string representation of the timezone to use for TIMESTAMP_LTZ.

        atol: absolute tolerance used for approximately-equal calculations

        rtol: relative tolerance used for approximately-equal calculations
    """
    bodosql_output = _run_jit_query(
        query,
        dataframe_dict,
        named_params,
        bind_variables,
        InputDist.OneD,
        is_out_distributed,
        session_tz,
        check_typing_issues=check_typing_issues,
    )
    if is_out_distributed:
        try:
            bodosql_output = bodo.libs.distributed_api.gatherv(bodosql_output)
        except Exception:
            comm = MPI.COMM_WORLD
            bodosql_output_list = comm.gather(bodosql_output)
            if bodo.get_rank() == 0:
                if isinstance(bodosql_output_list[0], np.ndarray):
                    bodosql_output = np.concatenate(bodosql_output_list)
                else:
                    bodosql_output = pd.concat(bodosql_output_list)
    _check_query_equal(
        bodosql_output,
        expected_output,
        check_names,
        check_dtype,
        sort_output,
        is_out_distributed,
        "1D Parallel JIT Test Failed",
        convert_nullable_bodosql,
        convert_columns_to_pandas=convert_columns_to_pandas,
        atol=atol,
        rtol=rtol,
    )


def check_query_jit_1DVar(
    query,
    dataframe_dict,
    named_params,
    bind_variables,
    check_names,
    check_dtype,
    sort_output,
    expected_output,
    convert_nullable_bodosql,
    is_out_distributed,
    check_typing_issues,
    convert_columns_to_pandas,
    session_tz,
    atol: float = 1e-08,
    rtol: float = 1e-05,
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

        bind_variables: Tuple of Python values used as bind variables in a query.

        check_names: Compare BodoSQL and expected_output names for equality.

        check_dtype: Compare BodoSQL and expected_output types for equality.

        sort_output: Compare the tables after sorting the results. Most
            queries need to be sorted because the order isn't defined.

        expected_output: The expected result of running the query.

        convert_nullable_bodosql: Should BodoSQL nullable integers be converted to Object dtype with None.

        check_typing_issues: raise an error if there is a typing issue for input args.
        Runs bodo typing on arguments and converts warnings to errors.

        convert_columns_to_pandas: Pandas does not support well some datatype such as decimal,
        list of strings or in general arrow data type. It typically fails at sorting. In that case
        using convert_columns_to_pandas=True will convert the columns to a string format which may help.

        session_tz: the string representation of the timezone to use for TIMESTAMP_LTZ.

        atol: absolute tolerance used for approximately-equal calculations

        rtol: relative tolerance used for approximately-equal calculations
    """
    bodosql_output = _run_jit_query(
        query,
        dataframe_dict,
        named_params,
        bind_variables,
        InputDist.OneDVar,
        is_out_distributed,
        session_tz,
        check_typing_issues=check_typing_issues,
    )
    if is_out_distributed:
        try:
            bodosql_output = bodo.libs.distributed_api.gatherv(bodosql_output)
        except Exception:
            comm = MPI.COMM_WORLD
            bodosql_output_list = comm.gather(bodosql_output)
            if bodo.get_rank() == 0:
                if isinstance(bodosql_output_list[0], np.ndarray):
                    bodosql_output = np.concatenate(bodosql_output_list)
                else:
                    bodosql_output = pd.concat(bodosql_output_list)
    _check_query_equal(
        bodosql_output,
        expected_output,
        check_names,
        check_dtype,
        sort_output,
        is_out_distributed,
        "1DVar Parallel JIT Test Failed",
        convert_nullable_bodosql,
        convert_columns_to_pandas=convert_columns_to_pandas,
        atol=atol,
        rtol=rtol,
    )


def _run_jit_query(
    query,
    dataframe_dict,
    named_params,
    bind_variables,
    input_dist,
    is_out_distributed,
    session_tz,
    check_typing_issues,
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

        bind_variables: Tuple of Python values used as bind variables in a query.

        input_dist: How the input data should be distributed. Either REP,
            1D or 1DVar. All input DataFrames are presumed to have the same
            distribution

        session_tz: the string representation of the timezone to use for TIMESTAMP_LTZ.

        check_typing_issues: raise an error if there is a typing issue for input args.
        Runs bodo typing on arguments and converts warnings to errors.

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

    if bind_variables is None:
        bind_variables = ()

    # Generate the BodoSQLContext with func_text so we can use jit code
    params = ",".join(
        [f"e{i}" for i in range(len(dataframe_dict))]
        + keys_list
        + [f"f{i}" for i in range(len(bind_variables))]
    )
    func_text = f"def test_impl(query, {params}):\n"
    func_text += "    bc = bodosql.BodoSQLContext(\n"
    func_text += "        {\n"
    args = [query]
    for i, key in enumerate(dataframe_dict.keys()):
        if input_dist == InputDist.OneD:
            args.append(
                _get_dist_df(
                    dataframe_dict[key], check_typing_issues=check_typing_issues
                )
            )
        elif input_dist == InputDist.OneDVar:
            args.append(
                _get_dist_df(
                    dataframe_dict[key],
                    var_length=True,
                    check_typing_issues=check_typing_issues,
                )
            )
        else:
            args.append(dataframe_dict[key])
        func_text += f"            '{key}': e{i},\n"
    args = args + values_list + list(bind_variables)
    func_text += "        }\n"
    if session_tz is not None:
        func_text += f"        , default_tz={repr(session_tz)}\n"
    func_text += "    )\n"
    func_text += "    result = bc.sql(query, "
    if keys_list:
        func_text += "{"
        for key in keys_list:
            func_text += f"'{key}': {key}, "
        func_text += "}, "
    else:
        func_text += "None, "
    if bind_variables:
        func_text += "( "
        for j in range(len(bind_variables)):
            func_text += f"f{j}, "
        func_text += "), "
    else:
        func_text += "None, "
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
    check_names=False,
    check_dtype=False,
    sort_output=True,
    is_out_distributed=False,
    failure_message="query test failed",
    convert_nullable_bodosql=False,
    convert_columns_to_pandas: bool = False,
    atol: float = 1e-08,
    rtol: float = 1e-05,
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

        convert_columns_to_pandas: Pandas does not support well some datatype such as decimal,
        list of strings or in general arrow data type. It typically fails at sorting. In that case
        using convert_columns_to_pandas=True will convert the columns to a string format which may help.

        atol: absolute tolerance used for approximately-equal calculations

        rtol: relative tolerance used for approximately-equal calculations

    """
    # convert pyarrow string data to regular object arrays to avoid dtype errors
    new_cols = []
    new_expected_cols = []
    for i in range(len(bodosql_output.columns)):
        # pd dtype must be the first value for comparing numpy dtypes
        new_cols.append(bodosql_output.iloc[:, i])
        new_expected_cols.append(expected_output.iloc[:, i])
        if pd.StringDtype("pyarrow") == bodosql_output.dtypes.iloc[i]:
            arr = bodosql_output.iloc[:, i].values
            try:
                # Cast to regular string array to avoid dictionary issues.
                result = pa.compute.cast(arr._pa_array, pa.string()).to_pandas()
                new_cols[-1] = result.astype(object).fillna(None)
                new_expected_cols[-1] = (
                    expected_output.iloc[:, i].astype(object).fillna(None)
                )
            except Exception:
                pass

    bodosql_output = (
        pd.concat(new_cols, axis=1)
        .set_axis(bodosql_output.columns, axis=1)
        .set_index(bodosql_output.index)
    )
    expected_output = (
        pd.concat(new_expected_cols, axis=1)
        .set_axis(expected_output.columns, axis=1)
        .set_index(expected_output.index)
    )

    # Convert Time64[ns] to bodo.types.Time to avoid Pandas bugs
    bodosql_output = convert_arrow_time_to_bodo_time(bodosql_output)
    if convert_columns_to_pandas:
        bodosql_output = bodo.tests.utils.convert_non_pandas_columns(bodosql_output)
        expected_output = bodo.tests.utils.convert_non_pandas_columns(expected_output)
    # NOTE: zero length dataframe may cause errors in Arrow's sort
    if sort_output and len(bodosql_output.dropna(how="all")):
        # Avoid sorting PyArrow null columns since causes errors in Pandas
        sort_colnames = [
            c
            for i, c in enumerate(bodosql_output.columns)
            if bodosql_output.dtypes.iloc[i] != pd.ArrowDtype(pa.null())
        ]
        bodosql_output = bodosql_output.sort_values(sort_colnames).reset_index(
            drop=True
        )
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
        bodosql_output.columns = [str(i) for i in range(len(bodosql_output.columns))]
        expected_output.columns = [str(i) for i in range(len(expected_output.columns))]

    passed = 1
    n_ranks = bodo.get_size()
    # only rank 0 should check if gatherv() called on output
    if not is_out_distributed or bodo.get_rank() == 0:
        passed = _test_equal_guard(
            bodosql_output,
            expected_output,
            check_dtype,
            convert_nullable_bodosql,
            atol=atol,
            rtol=rtol,
        )
    n_passed = reduce_sum(passed)
    assert n_passed == n_ranks, failure_message


def _test_equal_guard(
    bodosql_output,
    expected_output,
    check_dtype,
    convert_nullable_bodosql,
    atol: float = 1e-08,
    rtol: float = 1e-05,
):
    # Handle binary ArrowDtype comparison
    for i, (bodo_dtype, py_dtype) in enumerate(
        zip(bodosql_output.dtypes, expected_output.dtypes)
    ):
        if isinstance(bodo_dtype, pd.ArrowDtype) and (
            pa.types.is_binary(bodo_dtype.pyarrow_dtype)
            or pa.types.is_large_binary(bodo_dtype.pyarrow_dtype)
        ):
            expected_output[expected_output.columns[i]] = pd.array(
                expected_output[expected_output.columns[i]], dtype=bodo_dtype
            )

        # Convert object columns created by Spark usually
        if py_dtype == np.object_ and bodo_dtype != np.object_:
            expected_output[expected_output.columns[i]] = expected_output[
                expected_output.columns[i]
            ].astype(bodo_dtype)

    # Convert datetime64[us] to datetime64[ns] for comparison
    for i, dtype in enumerate(expected_output.dtypes):
        if dtype == np.dtype("datetime64[us]"):
            expected_output[expected_output.columns[i]] = expected_output[
                expected_output.columns[i]
            ].astype("datetime64[ns]")

    # No need to avoid exceptions if running with a single process and hang is not
    # possible. TODO: remove _test_equal_guard in general when [BE-2223] is resolved
    if bodo.get_size() == 1:
        # convert bodosql output to a value that can be compared with Spark
        if convert_nullable_bodosql:
            bodosql_output = convert_nullable_object(bodosql_output)
        _test_equal(
            bodosql_output,
            expected_output,
            check_dtype=check_dtype,
            rtol=rtol,
            atol=atol,
        )
        return 1
    passed = 1
    try:
        # convert bodosql output to a value that can be compared with Spark
        if convert_nullable_bodosql:
            bodosql_output = convert_nullable_object(bodosql_output)
        _test_equal(
            bodosql_output,
            expected_output,
            check_dtype=check_dtype,
            rtol=rtol,
            atol=atol,
        )
    except Exception as e:
        print(e)
        passed = 0
    return passed


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

    def convert(x):
        if isinstance(x, str):
            return x.encode("utf-8")
        elif isinstance(x, float) and np.isnan(x):
            return None
        else:
            return x

    df[columns] = df[columns].apply(
        lambda x: [convert(y) for y in x],
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


def convert_nullable_object(df):
    """
    Function the converts a DataFrame with a nullable column to an
    Object Datatype replaces pd.NA with None. This is used so Spark
    can interpret the results.
    """
    if any(
        isinstance(
            x,
            (
                pd.core.arrays.integer.IntegerDtype,
                pd.core.arrays.boolean.BooleanDtype,
            ),
        )
        for x in df.dtypes
    ):
        df = df.copy()
        for i, x in enumerate(df.dtypes):
            if isinstance(
                x,
                (
                    pd.core.arrays.integer.IntegerDtype,
                    pd.core.arrays.boolean.BooleanDtype,
                ),
            ):
                S = df.iloc[:, i]
                df[df.columns[i]] = S.astype(object).where(pd.notnull(S), None)
    return df


def convert_arrow_time_to_bodo_time(df):
    """Convert a DataFrame with potentially Arrow Time64[ns] columns to bodo.types.Time objects.
    Imports the compiler for bodo.types.Time"""
    import bodo.decorators  # noqa

    if not isinstance(df, pd.DataFrame):
        return df

    for col in df.columns:
        if df[col].dtype == pd.ArrowDtype(pa.time64("ns")):
            col_as_int = pd.array(
                df[col].array._pa_array.cast(pa.int64()),
                dtype=pd.ArrowDtype(pa.int64()),
            )
            bodo_times = []
            for x in col_as_int:
                if pd.isna(x):
                    bodo_times.append(None)
                else:
                    bodo_times.append(bodo.types.Time(nanosecond=x, precision=9))
            df[col] = pd.Series(bodo_times, dtype=object)

    return df


def remove_tz_columns_spark(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Takes any DataFrame columns in df that
    are also found in cols and converts them from
    tz-aware to tz-naive. This is necessary for passing data to
    Spark because Spark will compute the incorrect timestamp when it
    converts the timezone to local time (as it doesn't handle tz information
    in a way we can use for testing).

    Args:
        df (pd.DataFrame): Data which will be an input to spark
        cols (List[str]): List of column names which if found should be
            converted from tz-aware to tz-naive.

    Returns:
        pd.DataFrame: A new DataFrame with possibly some columns changed.
    """
    new_df = pd.DataFrame({})
    naive_cols = set(cols)
    for col in df.columns:
        if col in naive_cols:
            new_df[col] = df[col].dt.tz_convert(None)
        else:
            new_df[col] = df[col]
    return new_df


def generate_plan(query, dataframe_dict, session_tz=None):
    """
    Return a plan for a given query with the dictionary
    for a BodoSQLContext.
    """
    bc = bodosql.BodoSQLContext(dataframe_dict, session_tz)
    return bc.generate_plan(query)


def _get_dist_df(df, var_length=False, check_typing_issues=True):
    """
    get distributed chunk for a dataframe df on current rank (for input to test functions).
    Wrapper around bodo's _get_dist_arg that requires a dataframe input
    """
    assert isinstance(df, pd.DataFrame), (
        "Error: _get_dist_df was passed a non dataframe object"
    )

    return _get_dist_arg(
        df, var_length=var_length, check_typing_issues=check_typing_issues
    )


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
        raise ValueError("Named Parameter conversion to Pyspark Literal not supported.")


def shrink_data(ctx, n, keys_to_shrink=None, keys_to_not_shrink=None):
    output_dict = {}
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

    assert num_pq_readers == n, (
        "Number of ParquetReaders in the IR doesn't match the expectation"
    )


def create_pyspark_schema_from_dataframe(df):
    """Constructs a Pyspark schema for an appropriately typed
    DataFrame. This is used for tests whose output depends on
    maintaining precision.
    """
    from numba.core import types

    bodo_to_pyspark_dtype_map = {
        types.int8: ByteType(),
        types.uint8: ByteType(),
        types.int16: ShortType(),
        types.uint16: ShortType(),
        types.int32: IntegerType(),
        types.uint32: IntegerType(),
        types.int64: LongType(),
        types.uint64: LongType(),
        types.float32: FloatType(),
        types.float64: DoubleType(),
        bodo.types.datetime64ns: TimestampType(),
        bodo.types.timedelta64ns: DayTimeIntervalType(),
    }

    df_type = bodo.typeof(df)

    field_list = []
    for col, arr_type in zip(df_type.columns, df_type.data):
        if (
            isinstance(
                arr_type,
                (
                    types.Array,
                    bodo.types.IntegerArrayType,
                    bodo.types.FloatingArrayType,
                ),
            )
            or arr_type == bodo.types.boolean_array_type
        ):
            pyspark_type = bodo_to_pyspark_dtype_map[arr_type.dtype]
        elif arr_type == bodo.types.string_array_type:
            pyspark_type = StringType()
        else:
            raise TypeError("Type mapping to Pyspark Schema not implemented yet.")
        field_list.append(StructField(col, pyspark_type, True))
    return StructType(field_list)


def make_tables_nullable(input_ctx):
    output_ctx = {}
    np.random.seed(42)

    for table_name, table_value in input_ctx.items():
        cond = np.random.ranf(table_value.shape) < 0.5
        nullable_table_value = table_value.mask(cond, table_value, axis=0)
        output_ctx[table_name] = nullable_table_value

    return output_ctx


def remap_spark_agg_fn_name(query):
    """
    Spark uses slightly different naming conventions for certain SQL functions
    from Snowflake, so we use the builtin str.replace() method to re-map
    SQL function names/keywords to Spark ones.

    This method is the remap function for all window/aggregation functions
    currently supported in BodoSQL.

    This method is not intended for Bodo users but for our internal testing,
    so this does not account for SQL function names/keywords appearing as
    variable names or literals in SQL queries.
    """
    spark_dict = {
        "VARIANCE_POP": "VAR_POP",
        "VARIANCE_SAMP": "VAR_SAMP",
    }

    for key in spark_dict.keys():
        query = query.replace(key, spark_dict[key])

    return query


def get_equivalent_spark_agg_query(query):
    """
    Uses the Python regex library re and remap_spark_agg_fn_name (defined above) to
    convert the input BodoSQL query into an equivalent Spark query.
    """
    spark_query = remap_spark_agg_fn_name(query)
    spark_query = re.sub(
        "MEDIAN\\(([a-zA-Z0-9-_]+)\\)", "APPROX_PERCENTILE(\\1, .5)", spark_query
    )
    spark_query = re.sub(
        "VARIANCE_POP\\(([a-zA-Z0-9-_]+)\\)", "VAR_POP(\\1)", spark_query
    )
    spark_query = re.sub(
        "VARIANCE_SAMP\\(([a-zA-Z0-9-_]+)\\)", "VAR_SAMP(\\1)", spark_query
    )
    spark_query = re.sub(
        "CAST\\(([a-zA-Z0-9-_]+) AS VARCHAR\\)", "CAST(\\1 AS STRING)", spark_query
    )

    return spark_query
    # TODO (allai5): BY CUBE, BY ROLLUP


def gen_unique_id(name_prefix: str) -> str:
    path = None
    if bodo.get_rank() == 0:
        path = gen_unique_table_id(name_prefix)
    path = MPI.COMM_WORLD.bcast(path)
    return path


def assert_equal_par(bodo_output, py_output, check_dtype=False):
    passed = _test_equal_guard(bodo_output, py_output, check_dtype, False)
    # count how many pes passed the test, since throwing exceptions directly
    # can lead to inconsistency across pes and hangs
    n_passed = reduce_sum(passed)
    assert n_passed == bodo.get_size(), "Parallel test failed"


def replace_type_varchar(output: pd.DataFrame):
    """Additionally, erase precision from VARCHAR types. This is useful for
    asserting on the output of functions like describe_table, where we just want
    to check if a column is a VARCHAR type."""
    res = output.copy()
    # replace VARCHAR(precision) with VARCHAR
    type_is_varchar = res["TYPE"].map(lambda x: x.startswith("VARCHAR"))
    res["TYPE"] = res["TYPE"].where(~type_is_varchar, "VARCHAR")
    return res
