#include "_plan.h"
#include <arrow/python/pyarrow.h>
#include <utility>

#include "_executor.h"
#include "duckdb/common/types.hpp"
#include "duckdb/common/unique_ptr.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"

#include "physical/read_pandas.h"
#include "physical/read_parquet.h"

// if status of arrow::Result is not ok, form an err msg and raise a
// runtime_error with it
#undef CHECK_ARROW
#define CHECK_ARROW(expr, msg)                                          \
    if (!(expr.ok())) {                                                 \
        std::string err_msg = std::string(msg) + " " + expr.ToString(); \
        throw std::runtime_error(err_msg);                              \
    }

// if status of arrow::Result is not ok, form an err msg and raise a
// runtime_error with it. If it is ok, get value using ValueOrDie
// and assign it to lhs using std::move
#undef CHECK_ARROW_AND_ASSIGN
#define CHECK_ARROW_AND_ASSIGN(res, msg, lhs) \
    CHECK_ARROW(res.status(), msg)            \
    lhs = std::move(res).ValueOrDie();

/**
 * @brief Convert a std::unique_ptr to the duckdb equivalent.
 *
 */
template <class T>
duckdb::unique_ptr<T> to_duckdb(std::unique_ptr<T> &val) {
    return duckdb::unique_ptr<T>(val.release());
}

duckdb::unique_ptr<duckdb::LogicalOperator> optimize_plan(
    std::unique_ptr<duckdb::LogicalOperator> plan) {
    duckdb::Optimizer &optimizer = get_duckdb_optimizer();

    // Convert std::unique_ptr to duckdb::unique_ptr
    // Input is using std since Cython supports it
    auto in_plan = to_duckdb(plan);

    duckdb::unique_ptr<duckdb::LogicalOperator> out_plan =
        optimizer.Optimize(std::move(in_plan));
    return out_plan;
}

duckdb::unique_ptr<duckdb::Expression> make_const_int_expr(int val) {
    return duckdb::make_uniq<duckdb::BoundConstantExpression>(
        duckdb::Value(val));
}

duckdb::unique_ptr<duckdb::Expression> make_col_ref_expr(
    std::unique_ptr<duckdb::LogicalOperator> &source, PyObject *field_py,
    int col_idx) {
    auto field_res = arrow::py::unwrap_field(field_py);
    std::shared_ptr<arrow::Field> field;
    CHECK_ARROW_AND_ASSIGN(field_res,
                           "make_col_ref_expr: unable to unwrap field", field);
    auto [_, ctype] = arrow_field_to_duckdb(field);

    return duckdb::make_uniq<duckdb::BoundColumnRefExpression>(
        ctype,
        duckdb::ColumnBinding(get_operator_table_index(source), col_idx));
}

duckdb::unique_ptr<duckdb::Expression> make_binop_expr(
    std::unique_ptr<duckdb::Expression> &lhs,
    std::unique_ptr<duckdb::Expression> &rhs, duckdb::ExpressionType etype) {
    // Convert std::unique_ptr to duckdb::unique_ptr.
    auto lhs_duck = to_duckdb(lhs);
    auto rhs_duck = to_duckdb(rhs);
    return duckdb::make_uniq<duckdb::BoundComparisonExpression>(
        etype, std::move(lhs_duck), std::move(rhs_duck));
}

duckdb::unique_ptr<duckdb::LogicalFilter> make_filter(
    std::unique_ptr<duckdb::LogicalOperator> &source,
    std::unique_ptr<duckdb::Expression> &filter_expr) {
    // Convert std::unique_ptr to duckdb::unique_ptr.
    auto source_duck = to_duckdb(source);
    auto filter_expr_duck = to_duckdb(filter_expr);
    auto logical_filter =
        duckdb::make_uniq<duckdb::LogicalFilter>(std::move(filter_expr_duck));

    logical_filter->children.push_back(std::move(source_duck));
    return logical_filter;
}

duckdb::unique_ptr<duckdb::LogicalProjection> make_projection(
    std::unique_ptr<duckdb::LogicalOperator> &source,
    std::vector<int> &select_vec, PyObject *out_schema_py) {
    duckdb::idx_t source_index = get_operator_table_index(source);
    // Convert std::unique_ptr to duckdb::unique_ptr.
    auto source_duck = to_duckdb(source);
    auto binder = get_duckdb_binder();
    auto table_idx = binder.get()->GenerateTableIndex();

    std::shared_ptr<arrow::Schema> out_schema = unwrap_schema(out_schema_py);
    // We only care about the types, not the field names
    auto [_, type_vec] = arrow_schema_to_duckdb(out_schema);

    assert(select_vec.size() == type_vec.size());
    std::vector<duckdb::unique_ptr<duckdb::Expression>> projection_expressions;
    for (size_t i = 0; i < select_vec.size(); ++i) {
        auto selection = select_vec[i];
        auto stype = type_vec[i];
        auto expr = duckdb::make_uniq<duckdb::BoundColumnRefExpression>(
            stype, duckdb::ColumnBinding(source_index, selection));
        projection_expressions.emplace_back(std::move(expr));
    }

    // Create projection node.
    duckdb::unique_ptr<duckdb::LogicalProjection> proj =
        duckdb::make_uniq<duckdb::LogicalProjection>(
            table_idx, std::move(projection_expressions));

    // Add the sources to be joined.
    proj->children.push_back(std::move(source_duck));

    return proj;
}

std::vector<int> get_projection_pushed_down_columns(
    std::unique_ptr<duckdb::LogicalOperator> &proj) {
    if (proj->children.size() != 1) {
        throw std::runtime_error(
            "Only one child operator expected in LogicalProjection");
    }
    duckdb::LogicalGet &get_plan =
        proj->children[0]->Cast<duckdb::LogicalGet>();

    std::vector<int> selected_columns;
    for (auto &ci : get_plan.GetColumnIds()) {
        selected_columns.push_back(ci.GetPrimaryIndex());
    }
    return selected_columns;
}

/**
 * @brief Dummy function to pass to DuckDB for UDFs. DuckDB runs some functions
 * during optimization for constant folding, but we avoid it by throwing an
 * exception.
 *
 */
static void RunFunction(duckdb::DataChunk &args, duckdb::ExpressionState &state,
                        duckdb::Vector &result) {
    throw std::runtime_error("Cannot run Bodo UDFs during optimization.");
}

duckdb::unique_ptr<duckdb::LogicalProjection>
make_projection_python_scalar_func(
    std::unique_ptr<duckdb::LogicalOperator> &source, PyObject *out_schema_py,
    PyObject *args) {
    // Output table index
    duckdb::idx_t table_idx = get_duckdb_binder()->GenerateTableIndex();

    // Get output data type (UDF output is a single column)
    std::shared_ptr<arrow::Schema> out_schema = unwrap_schema(out_schema_py);
    auto [_, out_types] = arrow_schema_to_duckdb(out_schema);
    assert(out_types.size() == 1);
    duckdb::LogicalType out_type = out_types[0];

    // Create ScalarFunction for UDF
    duckdb::ScalarFunction scalar_function = duckdb::ScalarFunction(
        "bodo_udf", source->types, out_type, RunFunction);
    duckdb::unique_ptr<duckdb::FunctionData> bind_data1 =
        duckdb::make_uniq<BodoPythonScalarFunctionData>(args);

    // Necessary before accessing source->types attribute
    source->ResolveOperatorTypes();

    // Add all input columns as UDF inputs
    std::vector<duckdb::unique_ptr<duckdb::Expression>> udf_in_exprs;
    for (size_t i = 0; i < source->types.size(); ++i) {
        auto expr = duckdb::make_uniq<duckdb::BoundColumnRefExpression>(
            source->types[i],
            duckdb::ColumnBinding(get_operator_table_index(source), i));
        udf_in_exprs.emplace_back(std::move(expr));
    }

    // Create UDF expression
    duckdb::unique_ptr<duckdb::BoundFunctionExpression> scalar_expr =
        make_uniq<duckdb::BoundFunctionExpression>(out_type, scalar_function,
                                                   std::move(udf_in_exprs),
                                                   std::move(bind_data1));

    // Create projection node
    std::vector<duckdb::unique_ptr<duckdb::Expression>> projection_expressions;
    projection_expressions.push_back(std::move(scalar_expr));
    duckdb::unique_ptr<duckdb::LogicalProjection> proj =
        duckdb::make_uniq<duckdb::LogicalProjection>(
            table_idx, std::move(projection_expressions));

    // Add the input source
    auto source_duck = to_duckdb(source);
    proj->children.push_back(std::move(source_duck));

    return proj;
}

duckdb::unique_ptr<duckdb::LogicalComparisonJoin> make_comparison_join(
    std::unique_ptr<duckdb::LogicalOperator> &lhs,
    std::unique_ptr<duckdb::LogicalOperator> &rhs, duckdb::JoinType join_type,
    std::vector<std::pair<int, int>> &cond_vec) {
    // Convert std::unique_ptr to duckdb::unique_ptr.
    auto lhs_duck = to_duckdb(lhs);
    auto rhs_duck = to_duckdb(rhs);
    // Create join node.
    auto comp_join =
        duckdb::make_uniq<duckdb::LogicalComparisonJoin>(join_type);
    // Create join condition.
    duckdb::LogicalType cbtype(duckdb::LogicalTypeId::INTEGER);
    for (std::pair<int, int> cond_pair : cond_vec) {
        duckdb::JoinCondition cond;
        cond.comparison = duckdb::ExpressionType::COMPARE_EQUAL;
        cond.left = duckdb::make_uniq<duckdb::BoundColumnRefExpression>(
            cbtype, duckdb::ColumnBinding(0, cond_pair.first));
        cond.right = duckdb::make_uniq<duckdb::BoundColumnRefExpression>(
            cbtype, duckdb::ColumnBinding(0, cond_pair.second));
        // Add the join condition to the join node.
        comp_join->conditions.push_back(std::move(cond));
    }
    // Add the sources to be joined.
    comp_join->children.push_back(std::move(lhs_duck));
    comp_join->children.push_back(std::move(rhs_duck));

    return comp_join;
}

std::pair<int64_t, PyObject *> execute_plan(
    std::unique_ptr<duckdb::LogicalOperator> plan, PyObject *out_schema_py) {
    std::shared_ptr<arrow::Schema> out_schema = unwrap_schema(out_schema_py);
    Executor executor(std::move(plan), out_schema);
    std::shared_ptr<table_info> output_table = executor.ExecutePipelines();
    PyObject *pyarrow_schema =
        arrow::py::wrap_schema(output_table->schema()->ToArrowSchema());

    return {reinterpret_cast<int64_t>(new table_info(*output_table)),
            pyarrow_schema};
}

duckdb::unique_ptr<duckdb::LogicalGet> make_parquet_get_node(
    std::string parquet_path, PyObject *pyarrow_schema,
    PyObject *storage_options) {
    duckdb::shared_ptr<duckdb::Binder> binder = get_duckdb_binder();

    BodoParquetScanFunction table_function = BodoParquetScanFunction();
    duckdb::unique_ptr<duckdb::FunctionData> bind_data1 =
        duckdb::make_uniq<BodoParquetScanFunctionData>(
            parquet_path, pyarrow_schema, storage_options);

    // Convert Arrow schema to DuckDB
    std::shared_ptr<arrow::Schema> arrow_schema = unwrap_schema(pyarrow_schema);
    auto [return_names, return_types] = arrow_schema_to_duckdb(arrow_schema);

    duckdb::virtual_column_map_t virtual_columns;

    duckdb::unique_ptr<duckdb::LogicalGet> out_get =
        duckdb::make_uniq<duckdb::LogicalGet>(
            binder->GenerateTableIndex(), table_function, std::move(bind_data1),
            return_types, return_names, virtual_columns);

    // Column ids need to be added separately.
    // DuckDB column id initialization example:
    // https://github.com/duckdb/duckdb/blob/d29a92f371179170688b4df394478f389bf7d1a6/src/catalog/catalog_entry/table_catalog_entry.cpp#L252
    for (size_t i = 0; i < return_names.size(); i++) {
        out_get->AddColumnId(i);
    }

    return out_get;
}

duckdb::unique_ptr<duckdb::LogicalGet> make_dataframe_get_seq_node(
    PyObject *df, PyObject *pyarrow_schema) {
    // See DuckDB Pandas scan code:
    // https://github.com/duckdb/duckdb/blob/d29a92f371179170688b4df394478f389bf7d1a6/tools/pythonpkg/src/include/duckdb_python/pandas/pandas_scan.hpp#L19
    // https://github.com/duckdb/duckdb/blob/d29a92f371179170688b4df394478f389bf7d1a6/tools/pythonpkg/src/include/duckdb_python/pandas/pandas_bind.hpp#L19
    // https://github.com/duckdb/duckdb/blob/d29a92f371179170688b4df394478f389bf7d1a6/tools/pythonpkg/src/pandas/scan.cpp#L185

    duckdb::shared_ptr<duckdb::Binder> binder = get_duckdb_binder();

    BodoDataFrameScanFunction table_function = BodoDataFrameScanFunction();
    duckdb::unique_ptr<duckdb::FunctionData> bind_data1 =
        duckdb::make_uniq<BodoDataFrameSeqScanFunctionData>(df);

    // Convert Arrow schema to DuckDB
    std::shared_ptr<arrow::Schema> arrow_schema = unwrap_schema(pyarrow_schema);
    auto [return_names, return_types] = arrow_schema_to_duckdb(arrow_schema);

    duckdb::virtual_column_map_t virtual_columns;

    return duckdb::make_uniq<duckdb::LogicalGet>(
        binder->GenerateTableIndex(), table_function, std::move(bind_data1),
        return_types, return_names, virtual_columns);
}

duckdb::unique_ptr<duckdb::LogicalGet> make_dataframe_get_parallel_node(
    std::string result_id, PyObject *pyarrow_schema) {
    duckdb::shared_ptr<duckdb::Binder> binder = get_duckdb_binder();

    BodoDataFrameScanFunction table_function = BodoDataFrameScanFunction();
    duckdb::unique_ptr<duckdb::FunctionData> bind_data1 =
        duckdb::make_uniq<BodoDataFrameParallelScanFunctionData>(result_id);

    // Convert Arrow schema to DuckDB
    std::shared_ptr<arrow::Schema> arrow_schema = unwrap_schema(pyarrow_schema);
    auto [return_names, return_types] = arrow_schema_to_duckdb(arrow_schema);

    duckdb::virtual_column_map_t virtual_columns;

    return duckdb::make_uniq<duckdb::LogicalGet>(
        binder->GenerateTableIndex(), table_function, std::move(bind_data1),
        return_types, return_names, virtual_columns);
}

duckdb::ClientContext &get_duckdb_context() {
    static duckdb::DuckDB db(nullptr);
    static duckdb::ClientContext context(db.instance);
    return context;
}

duckdb::shared_ptr<duckdb::Binder> get_duckdb_binder() {
    static duckdb::shared_ptr<duckdb::Binder> binder =
        duckdb::Binder::CreateBinder(get_duckdb_context());
    return binder;
}

duckdb::Optimizer &get_duckdb_optimizer() {
    static duckdb::shared_ptr<duckdb::Binder> binder = get_duckdb_binder();
    static duckdb::Optimizer optimizer(*binder, get_duckdb_context());
    return optimizer;
}

std::pair<duckdb::vector<duckdb::string>, duckdb::vector<duckdb::LogicalType>>
arrow_schema_to_duckdb(const std::shared_ptr<arrow::Schema> &arrow_schema) {
    // See Arrow type handling in DuckDB for possible cases:
    // https://github.com/duckdb/duckdb/blob/d29a92f371179170688b4df394478f389bf7d1a6/src/function/table/arrow/arrow_duck_schema.cpp#L59
    // https://github.com/duckdb/duckdb/blob/d29a92f371179170688b4df394478f389bf7d1a6/src/common/adbc/nanoarrow/schema.cpp#L73
    // Arrow types:
    // https://github.com/apache/arrow/blob/5e9fce493f21098d616f08034bc233fcc529b3ad/cpp/src/arrow/type_fwd.h#L322

    duckdb::vector<duckdb::string> return_names;
    duckdb::vector<duckdb::LogicalType> logical_types;

    for (int i = 0; i < arrow_schema->num_fields(); i++) {
        const std::shared_ptr<arrow::Field> &field = arrow_schema->field(i);
        auto [return_name, duckdb_type] = arrow_field_to_duckdb(field);

        return_names.emplace_back(field->name());
        logical_types.push_back(duckdb_type);
    }

    return {return_names, logical_types};
}

std::pair<duckdb::string, duckdb::LogicalType> arrow_field_to_duckdb(
    const std::shared_ptr<arrow::Field> &field) {
    // Convert Arrow type to DuckDB LogicalType
    // TODO: handle all types
    duckdb::LogicalType duckdb_type;
    const std::shared_ptr<arrow::DataType> &arrow_type = field->type();
    switch (arrow_type->id()) {
        case arrow::Type::STRING:
        case arrow::Type::LARGE_STRING: {
            duckdb_type = duckdb::LogicalType::VARCHAR;
            break;
        }
        case arrow::Type::INT32: {
            duckdb_type = duckdb::LogicalType::INTEGER;
            break;
        }
        case arrow::Type::INT64: {
            duckdb_type = duckdb::LogicalType::BIGINT;
            break;
        }
        case arrow::Type::FLOAT: {
            duckdb_type = duckdb::LogicalType::FLOAT;
            break;
        }
        case arrow::Type::DOUBLE: {
            duckdb_type = duckdb::LogicalType::DOUBLE;
            break;
        }
        case arrow::Type::BOOL: {
            duckdb_type = duckdb::LogicalType::BOOLEAN;
            break;
        }
        case arrow::Type::DATE32: {
            duckdb_type = duckdb::LogicalType::DATE;
            break;
        }
        case arrow::Type::TIMESTAMP: {
            auto timestamp_type =
                std::static_pointer_cast<arrow::TimestampType>(arrow_type);
            arrow::TimeUnit::type unit = timestamp_type->unit();
            std::string tz = timestamp_type->timezone();
            if (unit == arrow::TimeUnit::NANO && tz == "") {
                duckdb_type = duckdb::LogicalType::TIMESTAMP_NS;
                break;
            }
            // TODO other units and timezones
            [[fallthrough]];
        }
        case arrow::Type::DECIMAL128: {
            auto decimal_type =
                std::static_pointer_cast<arrow::DecimalType>(arrow_type);
            int32_t precision = decimal_type->precision();
            int32_t scale = decimal_type->scale();
            duckdb_type = duckdb::LogicalType::DECIMAL(precision, scale);
            break;
        }
        case arrow::Type::LIST: {
            auto list_type =
                std::static_pointer_cast<arrow::ListType>(arrow_type);
            auto [name, child_type] =
                arrow_field_to_duckdb(list_type->field(0));
            duckdb_type = duckdb::LogicalType::LIST(child_type);
            break;
        }
        default:
            throw std::runtime_error(
                "Unsupported Arrow type: " + arrow_type->ToString() +
                ". Please extend the arrow_schema_to_duckdb function to handle "
                "this type.");
    }
    return {field->name(), duckdb_type};
}

std::string plan_to_string(std::unique_ptr<duckdb::LogicalOperator> &plan) {
    return plan->ToString(duckdb::ExplainFormat::GRAPHVIZ);
}

std::shared_ptr<PhysicalSource>
BodoDataFrameParallelScanFunctionData::CreatePhysicalOperator(
    std::vector<int> &selected_columns) {
    // Read the dataframe from the result registry using
    // sys.modules["__main__"].RESULT_REGISTRY since importing
    // bodo.spawn.worker creates a new module with new empty registry.

    // Import Python sys module
    PyObject *sys_module = PyImport_ImportModule("sys");
    if (!sys_module) {
        throw std::runtime_error("Failed to import sys module");
    }

    // Get sys.modules dictionary
    PyObject *modules_dict = PyObject_GetAttrString(sys_module, "modules");
    if (!modules_dict) {
        Py_DECREF(sys_module);
        throw std::runtime_error("Failed to get sys.modules");
    }

    // Get __main__ module
    PyObject *main_module = PyDict_GetItemString(modules_dict, "__main__");
    if (!main_module) {
        Py_DECREF(modules_dict);
        Py_DECREF(sys_module);
        throw std::runtime_error("Failed to get __main__ module");
    }

    // Get RESULT_REGISTRY[result_id]
    PyObject *result_registry =
        PyObject_GetAttrString(main_module, "RESULT_REGISTRY");
    PyObject *df = PyDict_GetItemString(result_registry, result_id.c_str());
    if (!df) {
        throw std::runtime_error("Result ID not found in result registry");
    }
    Py_DECREF(result_registry);
    Py_DECREF(modules_dict);
    Py_DECREF(sys_module);

    return std::make_shared<PhysicalReadPandas>(df);
}

std::shared_ptr<PhysicalSource>
BodoDataFrameSeqScanFunctionData::CreatePhysicalOperator(
    std::vector<int> &selected_columns) {
    return std::make_shared<PhysicalReadPandas>(df);
}

std::shared_ptr<PhysicalSource>
BodoParquetScanFunctionData::CreatePhysicalOperator(
    std::vector<int> &selected_columns) {
    return std::make_shared<PhysicalReadParquet>(
        path, pyarrow_schema, storage_options, selected_columns);
}

duckdb::idx_t get_operator_table_index(
    std::unique_ptr<duckdb::LogicalOperator> &op) {
    if (op->GetTableIndex().size() != 1) {
        throw std::runtime_error("Only one table index expected in operator");
    }
    return op->GetTableIndex()[0];
}

#undef CHECK_ARROW
#undef CHECK_ARROW_AND_ASSIGN
