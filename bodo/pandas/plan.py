from __future__ import annotations

import sys
import traceback

import pandas as pd
import pyarrow as pa

import bodo
from bodo.pandas.utils import (
    BODO_NONE_DUMMY,
    arrow_to_empty_df,
    cpp_table_to_df,
    cpp_table_to_series,
    get_n_index_arrays,
    wrap_plan,
)


class LazyPlan:
    """Easiest mode to use DuckDB is to generate isolated queries and try to minimize
    node re-use issues due to the frequent use of unique_ptr.  This class should be
    used when constructing all plans and holds them lazily.  On demand, generate_duckdb
    can be used to convert to an isolated set of DuckDB objects for execution.
    """

    def __init__(self, plan_class, empty_data, *args):
        self.plan_class = plan_class
        self.args = args
        assert isinstance(empty_data, (pd.DataFrame, pd.Series)), (
            "LazyPlan: empty_data must be a DataFrame or Series"
        )
        self.is_series = isinstance(empty_data, pd.Series)
        self.empty_data = empty_data
        if self.is_series:
            # None name doesn't round-trip to dataframe correctly so we use a dummy name
            # that is replaced with None in wrap_plan
            name = BODO_NONE_DUMMY if empty_data.name is None else empty_data.name
            self.empty_data = empty_data.to_frame(name=name)

        self.pa_schema = pa.Schema.from_pandas(self.empty_data)

    def __str__(self):
        args = self.args

        # Avoid duplicated plan strings by omitting data_source.
        if isinstance(self, ColRefExpression):
            col_index = args[1]
            return f"ColRefExpression({col_index})"
        elif isinstance(self, PythonScalarFuncExpression):
            func_name, col_indices = args[1][0], args[2]
            return f"PythonScalarFuncExpression({func_name}, {col_indices})"

        out = f"{self.plan_class}: \n"
        args_str = ""
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                args_str += f"{arg.columns.tolist()}\n"
            elif arg is not None:
                args_str += f"{arg}\n"

        out += "\n".join(
            f"  {arg_line}"
            for arg_line in args_str.split("\n")
            if not arg_line.isspace()
        )

        return out

    __repr__ = __str__

    def generate_duckdb(self, cache=None):
        from bodo.ext import plan_optimizer

        # Sometimes the same LazyPlan object is encountered twice during the same
        # query so  we use the cache dict to only convert it once.
        if cache is None:
            cache = {}
        # If previously converted then use the last result.
        # Don't cache expression nodes.
        # TODO - Try to eliminate caching altogether since it seems to cause
        # more problems than lack of caching.
        if not isinstance(self, Expression) and id(self) in cache:
            return cache[id(self)]

        def recursive_check(x, use_cache):
            """Recursively convert LazyPlans but return other types unmodified."""
            if isinstance(x, LazyPlan):
                return x.generate_duckdb(cache=cache if use_cache else None)
            elif isinstance(x, (tuple, list)):
                return type(x)(recursive_check(i, use_cache) for i in x)
            else:
                return x

        # NOTE: Caching is necessary to make sure source operators which have table
        # indexes and are reused in various nodes (e.g. expressions) are not re-created
        # with different table indexes.
        # Join however doesn't need this and cannot use caching since a sub-plan may
        # be reused across right and left sides (e.g. self-join) leading to unique_ptr
        # errors.
        use_cache = True
        if isinstance(self, (LogicalComparisonJoin, LogicalSetOperation)):
            use_cache = False

        # Convert any LazyPlan in the args.
        # We do this in reverse order because we expect the first arg to be
        # the source of the plan and for the node being created to take
        # ownership of that source.  If other args reference that
        # plan then if we process them after we have taken ownership then
        # we will get nullptr exceptions.  So, process the args that don't
        # claim ownership first (in the reverse direction) and finally
        # process the first arg which we expect will take ownership.
        args = [recursive_check(x, use_cache) for x in reversed(self.args)]
        args.reverse()

        # Create real duckdb class.
        ret = getattr(plan_optimizer, self.plan_class)(self.pa_schema, *args)
        # Add to cache so we don't convert it again.
        cache[id(self)] = ret
        return ret

    def replace_empty_data(self, empty_data):
        """Replace the empty_data of the plan with a new empty_data."""
        out = self.__class__(
            empty_data,
            *self.args,
        )
        out.is_series = self.is_series
        return out


class LogicalOperator(LazyPlan):
    """Base class for all logical operators in the Bodo query plan."""

    def __init__(self, empty_data, *args):
        super().__init__(self.__class__.__name__, empty_data, *args)


class Expression(LazyPlan):
    """Base class for all expressions in the Bodo query plan,
    such as column references, function calls, and arithmetic operations.
    """

    def __init__(self, empty_data, *args):
        super().__init__(self.__class__.__name__, empty_data, *args)


class LogicalProjection(LogicalOperator):
    """Logical operator for projecting columns and expressions."""

    def __init__(self, empty_data, source, exprs):
        self.source = source
        self.exprs = exprs
        super().__init__(empty_data, source, exprs)


class LogicalFilter(LogicalOperator):
    """Logical operator for filtering rows based on conditions."""

    pass


class LogicalAggregate(LogicalOperator):
    """Logical operator for aggregation operations."""

    pass


class LogicalComparisonJoin(LogicalOperator):
    """Logical operator for comparison-based joins."""

    pass


class LogicalSetOperation(LogicalOperator):
    """Logical operator for set operations like union."""

    pass


class LogicalLimit(LogicalOperator):
    """Logical operator for limiting the number of rows (e.g. df.head())."""

    pass


class LogicalOrder(LogicalOperator):
    """Logical operator for sorting data."""

    pass


class LogicalGetParquetRead(LogicalOperator):
    """Logical operator for reading Parquet files."""

    pass


class LogicalGetPandasReadSeq(LogicalOperator):
    """Logical operator for sequential read of a Pandas DataFrame."""

    pass


class LogicalGetPandasReadParallel(LogicalOperator):
    """Logical operator for parallel read of a Pandas DataFrame.\
    """

    pass


class LogicalGetIcebergRead(LogicalOperator):
    """Logical operator for reading Apache Iceberg tables."""

    def __init__(
        self,
        empty_data,
        table_identifier,
        catalog_name,
        catalog_properties,
        row_filter,
        pyiceberg_schema,
        snapshot_id,
        table_len_estimate,
        *,
        arrow_schema,
    ):
        super().__init__(
            empty_data,
            table_identifier,
            catalog_name,
            catalog_properties,
            row_filter,
            pyiceberg_schema,
            snapshot_id,
            table_len_estimate,
        )
        # Iceberg needs schema metadata
        # TODO: avoid this to support operations like renaming columns
        self.pa_schema = arrow_schema


class LogicalParquetWrite(LogicalOperator):
    """Logical operator for writing data to Parquet files."""

    pass


class LogicalIcebergWrite(LogicalOperator):
    """Logical operator for writing data to Apache Iceberg tables."""

    pass


class ColRefExpression(Expression):
    """Expression representing a column reference in the query plan."""

    def __init__(self, empty_data, source, col_index):
        self.source = source
        self.col_index = col_index
        super().__init__(empty_data, source, col_index)


class NullExpression(Expression):
    """Expression representing a null value in the query plan."""

    pass


class ConstantExpression(Expression):
    """Expression representing a constant value in the query plan."""

    pass


class AggregateExpression(Expression):
    """Expression representing an aggregate function in the query plan."""

    pass


class PythonScalarFuncExpression(Expression):
    """Expression representing a Python scalar function call in the query plan."""

    pass


class ComparisonOpExpression(Expression):
    """Expression representing a comparison operation in the query plan."""

    pass


class ConjunctionOpExpression(Expression):
    """Expression representing a conjunction (AND) operation in the query plan."""

    pass


class UnaryOpExpression(Expression):
    """Expression representing a unary operation (e.g. negation) in the query plan."""

    pass


class ArithOpExpression(Expression):
    """Expression representing an arithmetic operation (e.g. addition, subtraction)
    in the query plan.
    """

    pass


def execute_plan(plan: LazyPlan):
    """Execute a dataframe plan using Bodo's execution engine.

    Args:
        plan (LazyPlan): query plan to execute

    Returns:
        pd.DataFrame: output data
    """
    import bodo

    def _exec_plan(plan):
        import bodo
        from bodo.ext import plan_optimizer

        duckdb_plan = plan.generate_duckdb()

        if (
            bodo.dataframe_library_dump_plans
            and bodo.libs.distributed_api.get_rank() == 0
        ):
            print("Unoptimized plan")
            print(duckdb_plan.toString())

        # Print the plan before optimization
        if bodo.tracing_level >= 2 and bodo.libs.distributed_api.get_rank() == 0:
            pre_optimize_graphviz = duckdb_plan.toGraphviz()
            with open("pre_optimize" + str(id(plan)) + ".dot", "w") as f:
                print(pre_optimize_graphviz, file=f)

        optimized_plan = plan_optimizer.py_optimize_plan(duckdb_plan)

        if (
            bodo.dataframe_library_dump_plans
            and bodo.libs.distributed_api.get_rank() == 0
        ):
            print("Optimized plan")
            print(optimized_plan.toString())

        # Print the plan after optimization
        if bodo.tracing_level >= 2 and bodo.libs.distributed_api.get_rank() == 0:
            post_optimize_graphviz = optimized_plan.toGraphviz()
            with open("post_optimize" + str(id(plan)) + ".dot", "w") as f:
                print(post_optimize_graphviz, file=f)

        output_func = cpp_table_to_series if plan.is_series else cpp_table_to_df
        return plan_optimizer.py_execute_plan(
            optimized_plan, output_func, duckdb_plan.out_schema
        )

    if bodo.dataframe_library_run_parallel:
        import bodo.spawn.spawner

        # Initialize LazyPlanDistributedArg objects that may need scattering data
        # to workers before execution.
        for a in plan.args:
            _init_lazy_distributed_arg(a)

        if bodo.dataframe_library_dump_plans:
            # Sometimes when an execution is triggered it isn't expected that
            # an execution should happen at that point.  This traceback is
            # useful to identify what is triggering the execution as it may be
            # a bug or the usage of some Pandas API that calls a function that
            # triggers execution.  This traceback can help fix the bug or
            # select a different Pandas API or an internal Pandas function that
            # bypasses the issue.
            traceback.print_stack(file=sys.stdout)
            print("")  # Print on new line during tests.

        return bodo.spawn.spawner.submit_func_to_workers(_exec_plan, [], plan)

    return _exec_plan(plan)


def _init_lazy_distributed_arg(arg, visited_plans=None):
    """Initialize the LazyPlanDistributedArg objects for the given plan argument that
    may need scattering data to workers before execution.
    Has to be called right before plan execution since the dataframe state
    may change (distributed to collected) and the result ID may not be valid anymore.
    """
    if visited_plans is None:
        # Keep track of visited LazyPlans to prevent extra checks.
        visited_plans = set()

    if isinstance(arg, LazyPlan):
        if id(arg) in visited_plans:
            return
        visited_plans.add(id(arg))
        for a in arg.args:
            _init_lazy_distributed_arg(a, visited_plans=visited_plans)
    elif isinstance(arg, (tuple, list)):
        for a in arg:
            _init_lazy_distributed_arg(a, visited_plans=visited_plans)
    elif isinstance(arg, LazyPlanDistributedArg):
        arg.init()


def get_plan_cardinality(plan: LazyPlan):
    """See if we can statically know the cardinality of the result of the plan.

    Args:
        plan (LazyPlan): query plan to get cardinality of.

    Returns:
        int (if cardinality is known) or None (if not known)
    """

    duckdb_plan = plan.generate_duckdb()
    return duckdb_plan.getCardinality()


def getPlanStatistics(plan: LazyPlan):
    """Get statistics for a plan pre and post optimization.

    Args:
        plan (LazyPlan): query plan to get statistics for

    Returns:
        Number of nodes in the tree before and after optimization.
    """
    from bodo.ext import plan_optimizer

    duckdb_plan = plan.generate_duckdb()
    preOptNum = plan_optimizer.count_nodes(duckdb_plan)
    optimized_plan = plan_optimizer.py_optimize_plan(duckdb_plan)
    postOptNum = plan_optimizer.count_nodes(optimized_plan)
    return preOptNum, postOptNum


def get_proj_expr_single(proj: LazyPlan):
    """Get the single expression from a LogicalProjection node."""
    if is_single_projection(proj):
        return proj.exprs[0]
    else:
        if not proj.is_series:
            raise Exception("Got a non-Series in get_proj_expr_single")
        return make_col_ref_exprs([0], proj)[0]


def get_single_proj_source_if_present(proj: LazyPlan):
    """Get the single expression from a LogicalProjection node."""
    if is_single_projection(proj):
        return proj.source
    else:
        if not proj.is_series:
            raise Exception("Got a non-Series in get_single_proj_source_if_present")
        return proj


def is_single_projection(proj: LazyPlan):
    """Return True if plan is a projection with a single expression"""
    return isinstance(proj, LogicalProjection) and len(proj.exprs) == (
        get_n_index_arrays(proj.empty_data.index) + 1
    )


def is_single_colref_projection(proj: LazyPlan):
    """Return True if plan is a projection with a single expression that is a column reference"""
    return is_single_projection(proj) and isinstance(proj.exprs[0], ColRefExpression)


def make_col_ref_exprs(key_indices, src_plan):
    """Create column reference expressions for the given key indices for the input
    source plan.
    """

    exprs = []
    for k in key_indices:
        # Using Arrow schema instead of zero_size_self.iloc to handle Index
        # columns correctly.
        empty_data = arrow_to_empty_df(pa.schema([src_plan.pa_schema[k]]))
        p = ColRefExpression(empty_data, src_plan, k)
        exprs.append(p)

    return exprs


class LazyPlanDistributedArg:
    """
    Class to hold the arguments for a LazyPlan that are distributed on the workers.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.mgr = None
        self.res_id = None

    def init(self):
        """Initialize to make sure the result ID is set in preparation for pickling
        result ID to workers for execution.
        Should be called right before execution of the plan since the dataframe state
        may change (distributed to collected) and the result ID may not be valid
        anymore.
        """
        if getattr(self.df._mgr, "_md_result_id", None) is not None:
            # The dataframe is already distributed so we can use the existing result ID
            self.res_id = self.df._mgr._md_result_id
        elif self.mgr is not None:
            # We scattered a DataFrame already and own a manager to reuse
            self.res_id = self.mgr._md_result_id
        else:
            # The dataframe is not distributed yet so we need to scatter it
            # and create a new result ID.
            mgr = bodo.spawn.spawner.get_spawner().scatter_data(self.df)
            self.res_id = mgr._md_result_id
            self.mgr = mgr

    def __reduce__(self):
        """
        This method is used to serialize the object for distribution.
        We can't send the manager to the workers without triggering collection
        so we just send the result ID instead.
        """
        assert self.res_id is not None, (
            "LazyPlanDistributedArg: result ID is not set, call init() first"
        )
        return (str, (self.res_id,))


def count_plan(self):
    # See if we can get the cardinality statically.
    static_cardinality = get_plan_cardinality(self._plan)
    if static_cardinality is not None:
        return static_cardinality

    # Can't be known statically so create count plan on top of
    # existing plan.
    count_star_schema = pd.Series(dtype="uint64", name="count_star")
    aggregate_plan = LogicalAggregate(
        count_star_schema,
        self._plan,
        [],
        [
            AggregateExpression(
                count_star_schema,
                self._plan,
                "count_star",
                # Adding column 0 as input to avoid deleting all input by the optimizer
                # TODO: avoid materializing the input column
                [0],
                False,  # dropna
            )
        ],
    )
    projection_plan = LogicalProjection(
        count_star_schema,
        aggregate_plan,
        make_col_ref_exprs([0], aggregate_plan),
    )

    data = execute_plan(projection_plan)
    return data[0]


def _get_df_python_func_plan(df_plan, empty_data, func, args, kwargs, is_method=True):
    """Create plan for calling some function or method on a DataFrame. Creates a
    PythonScalarFuncExpression with provided arguments and a LogicalProjection.
    """
    df_len = len(df_plan.empty_data.columns)
    udf_arg = PythonScalarFuncExpression(
        empty_data,
        df_plan,
        (
            func,
            False,  # is_series
            is_method,
            args,
            kwargs,
        ),
        tuple(range(df_len + get_n_index_arrays(df_plan.empty_data.index))),
    )

    # Select Index columns explicitly for output
    index_col_refs = tuple(
        make_col_ref_exprs(
            range(df_len, df_len + get_n_index_arrays(df_plan.empty_data.index)),
            df_plan,
        )
    )
    plan = LogicalProjection(
        empty_data,
        df_plan,
        (udf_arg,) + index_col_refs,
    )
    return wrap_plan(plan=plan)


def is_col_ref(expr):
    return isinstance(expr, ColRefExpression)


def is_scalar_func(expr):
    return isinstance(expr, PythonScalarFuncExpression)


def is_arith_expr(expr):
    return isinstance(expr, ArithOpExpression)
