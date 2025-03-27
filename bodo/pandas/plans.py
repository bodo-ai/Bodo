from abc import ABC

import pandas as pd

from bodo.pandas import plan_optimizer
from bodo.pandas.parquet import get_pandas_schema


class PlanOperator(ABC):
    def __init__(self):
        self.sources = []

    def add_source(self, node):
        self.sources.append(node)

    def convert_to_duckdb(self):
        print("No explicit convert_to_duckdb for type", type(self))
        return [x.convert_to_duckdb() for x in self.sources]


class UnaryPlanOperator(PlanOperator, ABC):
    def __init__(self, source):
        super().__init__()
        self.add_source(source)


class BinaryPlanOperator(PlanOperator, ABC):
    def __init__(self, lhs_source, rhs_source):
        super().__init__()
        self.add_source(lhs_source)
        self.add_source(rhs_source)


class BinaryOpPlanOperator(BinaryPlanOperator):
    def __init__(self, lhs_source, rhs_source, binop):
        super().__init__(lhs_source, rhs_source)
        self.binop = binop


class DataSourcePlanOperator(PlanOperator, ABC):
    def __init__(self):
        super().__init__()


class LogicalJoin(BinaryPlanOperator):
    def __init__(self, lhs_source, rhs_source):
        super().__init__(lhs_source, rhs_source)


class ProjectionPlanOperator(UnaryPlanOperator):
    def __init__(self, source, select_list):
        self.select_list = select_list
        super().__init__(source)


class MaskPlanOperator(UnaryPlanOperator):
    def __init__(self, source, expr):
        self.expr = expr
        super().__init__(source)


class FilterPlanOperator(BinaryPlanOperator):
    def __init__(self, source, mask):
        super().__init__(source, mask)


class AggregatePlanOperator(PlanOperator):
    def __init__(self, group_source, aggregate_source, select_list):
        super().__init__()
        self.group_source = group_source
        self.aggregate_source = aggregate_source
        self.select_list = select_list


class DistinctPlanOperator(PlanOperator):
    def __init__(self, targets):
        super().__init__()
        self.targets = targets


class Window(UnaryPlanOperator):
    def __init__(self, source):
        super().__init__(source)


class Unnest(UnaryPlanOperator):
    def __init__(self, source):
        super().__init__(source)


class OrderByPlanOperator(PlanOperator):
    def __init__(self, orders):
        super().__init__()
        self.orders = orders


class TopNPlanOperator(PlanOperator):
    def __init__(self, orders, limit, offset):
        super().__init__()
        self.orders = orders
        self.limit = limit
        self.offset = offset


class Get(DataSourcePlanOperator):
    def __init__(
        self,
        table_index,
        func,
        bind_data,
        returned_types,
        returned_names,
        virtual_columns=None,
    ):
        super().__init__()
        self.table_index = table_index
        self.func = func
        self.bind_data = bind_data
        self.returned_types = returned_types
        self.returned_names = returned_names


class ChunkGet(DataSourcePlanOperator):
    def __init__(self, table_index, chunk_types, column_data_collection=None):
        super().__init__()
        self.table_index = table_index
        self.chunk_types = chunk_types
        self.column_data_collection = column_data_collection


class ParquetRead(DataSourcePlanOperator):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.schema = get_pandas_schema(filename)

    def convert_to_duckdb(self):
        return plan_optimizer.LogicalGetParquetRead(self.filename.encode())


class Join(LogicalJoin):
    def __init__(self, lhs_source, rhs_source, join_type):
        self.join_type = join_type
        super().__init__(lhs_source, rhs_source)


class ComparisonJoin(LogicalJoin):
    def __init__(self, lhs_source, rhs_source, conditions):
        self.conditions = conditions
        super().__init__(lhs_source, rhs_source)

    def convert_to_duckdb(self):
        [x.convert_to_duckdb() for x in self.sources]
        join_plan = plan_optimizer.LogicalComparisonJoin(plan_optimizer.CJoinType.INNER)
        """
        JoinCondition cond;
        cond.comparison = ExpressionType::COMPARE_EQUAL;
        LogicalType type(LogicalTypeId::INTEGER);
        cond.left = make_uniq<BoundColumnRefExpression>(type, ColumnBinding(0, 0));
        cond.right = make_uniq<BoundColumnRefExpression>(type, ColumnBinding(1, 0));
        comp_join->children.push_back(std::move(make_uniq<BodoLogicalDummyScan>(binder->GenerateTableIndex(), 11)));
        comp_join->children.push_back(std::move(make_uniq<BodoLogicalDummyScan>(binder->GenerateTableIndex(), 100)));
        comp_join->conditions.push_back(std::move(cond));
        """
        return join_plan


""" Unconverted DuckDB types

https://github.com/duckdb/duckdb/tree/0431406ccc5f140f591c3a80cf879e9e6521b75e/src/include/duckdb/planner/operator

//===--------------------------------------------------------------------===//
// Logical Operator Types
//===--------------------------------------------------------------------===//
enum class PlanOperatorType : uint8_t {
	LOGICAL_LIMIT = 6,
	LOGICAL_COPY_TO_FILE = 10,
	LOGICAL_SAMPLE = 12,
	LOGICAL_PIVOT = 14,
	LOGICAL_COPY_DATABASE = 15,

	// -----------------------------
	// Data sources
	// -----------------------------
	LOGICAL_DELIM_GET = 27,
	LOGICAL_EXPRESSION_GET = 28,
	LOGICAL_DUMMY_SCAN = 29,
	LOGICAL_EMPTY_RESULT = 30,
	LOGICAL_CTE_REF = 31,
	// -----------------------------
	// Joins
	// -----------------------------
	LOGICAL_DELIM_JOIN = 51,
	LOGICAL_ANY_JOIN = 53,
	LOGICAL_CROSS_PRODUCT = 54,
	LOGICAL_POSITIONAL_JOIN = 55,
	LOGICAL_ASOF_JOIN = 56,
	LOGICAL_DEPENDENT_JOIN = 57,
	// -----------------------------
	// SetOps
	// -----------------------------
	LOGICAL_UNION = 75,
	LOGICAL_EXCEPT = 76,
	LOGICAL_INTERSECT = 77,
	LOGICAL_RECURSIVE_CTE = 78,
	LOGICAL_MATERIALIZED_CTE = 79,

	// -----------------------------
	// Updates
	// -----------------------------
	LOGICAL_INSERT = 100,
	LOGICAL_DELETE = 101,
	LOGICAL_UPDATE = 102,

	// -----------------------------
	// Schema
	// -----------------------------
	LOGICAL_ALTER = 125,
	LOGICAL_CREATE_TABLE = 126,
	LOGICAL_CREATE_INDEX = 127,
	LOGICAL_CREATE_SEQUENCE = 128,
	LOGICAL_CREATE_VIEW = 129,
	LOGICAL_CREATE_SCHEMA = 130,
	LOGICAL_CREATE_MACRO = 131,
	LOGICAL_DROP = 132,
	LOGICAL_PRAGMA = 133,
	LOGICAL_TRANSACTION = 134,
	LOGICAL_CREATE_TYPE = 135,
	LOGICAL_ATTACH = 136,
	LOGICAL_DETACH = 137,

	// -----------------------------
	// Explain
	// -----------------------------
	LOGICAL_EXPLAIN = 150,

	// -----------------------------
	// Helpers
	// -----------------------------
	LOGICAL_PREPARE = 175,
	LOGICAL_EXECUTE = 176,
	LOGICAL_EXPORT = 177,
	LOGICAL_VACUUM = 178,
	LOGICAL_SET = 179,
	LOGICAL_LOAD = 180,
	LOGICAL_RESET = 181,
	LOGICAL_UPDATE_EXTENSIONS = 182,

	// -----------------------------
	// Secrets
	// -----------------------------
	LOGICAL_CREATE_SECRET = 190,
};
"""


def convert_and_execute(plan):
    orig_plan = plan.convert_to_duckdb()
    opt_plan = plan_optimizer.py_optimize_plan(orig_plan)
    print("opt_plan:", opt_plan)


def wrap_plan(schema, plan):
    from bodo.pandas.frame import BodoDataFrame
    from bodo.pandas.series import BodoSeries
    from bodo.pandas.utils import get_lazy_manager_class, get_lazy_single_manager_class

    def collect_func(plan_to_execute: PlanOperator):
        print("collect_func for plan", plan_to_execute)
        convert_and_execute(plan_to_execute)

        """
        # collect is sometimes triggered during receive (e.g. for unsupported types
        # like IntervalIndex) so we may be in the middle of function execution
        # already.
        initial_running = self._is_running
        if not initial_running:
            self._is_running = True
        self.worker_intercomm.bcast(CommandType.GATHER.value, root=root)
        self.worker_intercomm.bcast(res_id, root=root)
        res = bodo.libs.distributed_api.gatherv(
            None, root=root, comm=self.worker_intercomm
        )
        if not initial_running:
            self._is_running = False
            self._run_del_queue()
        return res
        """

    def del_func(res_id: str):
        pass  # For now.

    if isinstance(schema, dict):
        schema = {
            col: pd.Series(dtype=col_type.dtype) for col, col_type in schema.items()
        }

    if isinstance(schema, (dict, pd.DataFrame)):
        if isinstance(schema, dict):
            schema = pd.DataFrame(schema)
        lazy_mgr = get_lazy_manager_class()(
            None,
            None,
            result_id=plan,
            nrows=1,
            head=schema._mgr,
            collect_func=collect_func,
            del_func=del_func,
            index_data=None,
            plan=plan,
        )
        new_df = BodoDataFrame.from_lazy_mgr(lazy_mgr, schema)
    elif isinstance(schema, pd.Series):
        lazy_mgr = get_lazy_single_manager_class()(
            None,
            None,
            result_id=plan,
            nrows=1,
            head=schema._mgr,
            collect_func=collect_func,
            del_func=del_func,
            index_data=None,
            plan=plan,
        )
        new_df = BodoSeries.from_lazy_mgr(lazy_mgr, schema)
    else:
        assert False

    new_df.plan = plan
    return new_df
