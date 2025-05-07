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
#include "duckdb/planner/expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_limit.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_sample.hpp"

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

duckdb::unique_ptr<duckdb::Expression> make_const_float_expr(float val) {
    return duckdb::make_uniq<duckdb::BoundConstantExpression>(
        duckdb::Value(val));
}

duckdb::unique_ptr<duckdb::Expression> make_const_string_expr(
    const std::string &val) {
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

/**
 * @brief Change the type of a constant to match the type of the other side
 *        of a binary op expr.
 *
 * params series - the non-constant part of the binary op expr
 * params constant - the constant part of the binary op expr
 * returns a BinaryConstantExpression where the internal value has been changed
 *         to match the type of the other side of the binary op
 */
duckdb::unique_ptr<duckdb::Expression> matchType(
    duckdb::unique_ptr<duckdb::Expression> &series,
    duckdb::unique_ptr<duckdb::Expression> &constant) {
    // Cast to constant to BoundConstantExpression.
    duckdb::unique_ptr<duckdb::BoundConstantExpression> bce_constant =
        dynamic_cast_unique_ptr<duckdb::BoundConstantExpression>(
            std::move(constant));

    // Get the type to convert the constant to.
    duckdb::LogicalType series_type = series->return_type;
    // Change the value to the given type.
    // Will throw an exception if such a conversion is not possible.
    bce_constant->value = bce_constant->value.DefaultCastAs(series_type);
    // Change the expression's return type to match.
    bce_constant->return_type = series_type;
    return bce_constant;
}

std::unique_ptr<duckdb::Expression> make_binop_expr(
    std::unique_ptr<duckdb::Expression> &lhs,
    std::unique_ptr<duckdb::Expression> &rhs, duckdb::ExpressionType etype) {
    // Convert std::unique_ptr to duckdb::unique_ptr.
    auto lhs_duck = to_duckdb(lhs);
    auto rhs_duck = to_duckdb(rhs);

    // If the left and right side of a binary op expression don't have
    // matching types then filter pushdown will be skipped.  Here we force
    // the constant to have the type as the other side of the binary op.
    if (lhs_duck->GetExpressionClass() ==
        duckdb::ExpressionClass::BOUND_CONSTANT) {
        lhs_duck = matchType(rhs_duck, lhs_duck);
    } else if (rhs_duck->GetExpressionClass() ==
               duckdb::ExpressionClass::BOUND_CONSTANT) {
        rhs_duck = matchType(lhs_duck, rhs_duck);
    }
    return duckdb::make_uniq<duckdb::BoundComparisonExpression>(
        etype, std::move(lhs_duck), std::move(rhs_duck));
}

duckdb::unique_ptr<duckdb::Expression> make_conjunction_expr(
    std::unique_ptr<duckdb::Expression> &lhs,
    std::unique_ptr<duckdb::Expression> &rhs, duckdb::ExpressionType etype) {
    // Convert std::unique_ptr to duckdb::unique_ptr.
    auto lhs_duck = to_duckdb(lhs);
    auto rhs_duck = to_duckdb(rhs);

    return duckdb::make_uniq<duckdb::BoundConjunctionExpression>(
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

duckdb::unique_ptr<duckdb::LogicalSample> make_sample(
    std::unique_ptr<duckdb::LogicalOperator> &source, int n) {
    // Convert std::unique_ptr to duckdb::unique_ptr.
    auto source_duck = to_duckdb(source);
    duckdb::unique_ptr<duckdb::SampleOptions> sampleOptions =
        duckdb::make_uniq<duckdb::SampleOptions>();
    sampleOptions->sample_size = duckdb::Value(n);
    sampleOptions->is_percentage = false;
    sampleOptions->method = duckdb::SampleMethod::SYSTEM_SAMPLE;
    sampleOptions->repeatable = true;  // Not sure if this is correct.
    auto logical_sample = duckdb::make_uniq<duckdb::LogicalSample>(
        std::move(sampleOptions), std::move(source_duck));

    return logical_sample;
}

duckdb::unique_ptr<duckdb::LogicalLimit> make_limit(
    std::unique_ptr<duckdb::LogicalOperator> &source, int n) {
    // Convert std::unique_ptr to duckdb::unique_ptr.
    auto source_duck = to_duckdb(source);
    auto logical_limit = duckdb::make_uniq<duckdb::LogicalLimit>(
        duckdb::BoundLimitNode::ConstantValue(n),
        duckdb::BoundLimitNode::ConstantValue(0));

    logical_limit->children.push_back(std::move(source_duck));
    return logical_limit;
}

duckdb::unique_ptr<duckdb::LogicalProjection> make_projection(
    std::unique_ptr<duckdb::LogicalOperator> &source,
    std::vector<std::unique_ptr<duckdb::Expression>> &expr_vec,
    PyObject *out_schema_py) {
    // Convert std::unique_ptr to duckdb::unique_ptr.
    auto source_duck = to_duckdb(source);
    auto binder = get_duckdb_binder();
    auto table_idx = binder.get()->GenerateTableIndex();

    std::vector<duckdb::unique_ptr<duckdb::Expression>> projection_expressions;
    for (auto &expr : expr_vec) {
        // Convert std::unique_ptr to duckdb::unique_ptr.
        auto expr_duck = to_duckdb(expr);
        projection_expressions.push_back(std::move(expr_duck));
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

duckdb::unique_ptr<duckdb::Expression> make_python_scalar_func_expr(
    std::unique_ptr<duckdb::LogicalOperator> &source, PyObject *out_schema_py,
    PyObject *args, const std::vector<int> &selected_columns) {
    // Get output data type (UDF output is a single column)
    std::shared_ptr<arrow::Schema> out_schema = unwrap_schema(out_schema_py);
    auto [_, out_types] = arrow_schema_to_duckdb(out_schema);
    // Maybe not be exactly 1 due to index column.
    assert(out_types.size() > 0);
    duckdb::LogicalType out_type = out_types[0];

    // Necessary before accessing source->types attribute
    source->ResolveOperatorTypes();

    // Create ScalarFunction for UDF
    duckdb::ScalarFunction scalar_function = duckdb::ScalarFunction(
        "bodo_udf", source->types, out_type, RunFunction);
    duckdb::unique_ptr<duckdb::FunctionData> bind_data1 =
        duckdb::make_uniq<BodoPythonScalarFunctionData>(args);

    // Add UDF input expressions for selected columns
    std::vector<duckdb::unique_ptr<duckdb::Expression>> udf_in_exprs;
    for (int col_idx : selected_columns) {
        auto expr = duckdb::make_uniq<duckdb::BoundColumnRefExpression>(
            source->types[col_idx],
            duckdb::ColumnBinding(get_operator_table_index(source), col_idx));
        udf_in_exprs.emplace_back(std::move(expr));
    }

    // Create UDF expression
    duckdb::unique_ptr<duckdb::BoundFunctionExpression> scalar_expr =
        make_uniq<duckdb::BoundFunctionExpression>(out_type, scalar_function,
                                                   std::move(udf_in_exprs),
                                                   std::move(bind_data1));

    return scalar_expr;
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

    auto out_get = duckdb::make_uniq<duckdb::LogicalGet>(
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

duckdb::unique_ptr<duckdb::LogicalGet> make_dataframe_get_parallel_node(
    std::string result_id, PyObject *pyarrow_schema) {
    duckdb::shared_ptr<duckdb::Binder> binder = get_duckdb_binder();

    BodoDataFrameScanFunction table_function = BodoDataFrameScanFunction();
    duckdb::unique_ptr<duckdb::FunctionData> bind_data1 =
        duckdb::make_uniq<BodoDataFrameParallelScanFunctionData>(result_id);

    // Convert Arrow schema to DuckDB
    std::shared_ptr<arrow::Schema> arrow_schema = unwrap_schema(pyarrow_schema);
    auto [return_names, return_types] = arrow_schema_to_duckdb(arrow_schema);

    auto out_get = duckdb::make_uniq<duckdb::LogicalGet>(
        binder->GenerateTableIndex(), table_function, std::move(bind_data1),
        return_types, return_names);

    // Column ids need to be added separately.
    // DuckDB column id initialization example:
    // https://github.com/duckdb/duckdb/blob/d29a92f371179170688b4df394478f389bf7d1a6/src/catalog/catalog_entry/table_catalog_entry.cpp#L252
    for (size_t i = 0; i < return_names.size(); i++) {
        out_get->AddColumnId(i);
    }

    return out_get;
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
        case arrow::Type::UINT8: {
            duckdb_type = duckdb::LogicalType::UTINYINT;
            break;
        }
        case arrow::Type::INT8: {
            duckdb_type = duckdb::LogicalType::TINYINT;
            break;
        }
        case arrow::Type::UINT16: {
            duckdb_type = duckdb::LogicalType::USMALLINT;
            break;
        }
        case arrow::Type::INT16: {
            duckdb_type = duckdb::LogicalType::SMALLINT;
            break;
        }
        case arrow::Type::UINT32: {
            duckdb_type = duckdb::LogicalType::UINTEGER;
            break;
        }
        case arrow::Type::INT32: {
            duckdb_type = duckdb::LogicalType::INTEGER;
            break;
        }
        case arrow::Type::UINT64: {
            duckdb_type = duckdb::LogicalType::UBIGINT;
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
        case arrow::Type::DURATION: {
            duckdb_type = duckdb::LogicalType::INTERVAL;
            break;
        }
        case arrow::Type::TIMESTAMP: {
            auto timestamp_type =
                std::static_pointer_cast<arrow::TimestampType>(arrow_type);
            arrow::TimeUnit::type unit = timestamp_type->unit();
            std::string tz = timestamp_type->timezone();
            if (tz == "") {
                switch (unit) {
                    case arrow::TimeUnit::NANO:
                        duckdb_type = duckdb::LogicalType::TIMESTAMP_NS;
                        break;
                    // TODO: Support these types in Bodo
                    case arrow::TimeUnit::MICRO:
                        // microseconds
                        duckdb_type = duckdb::LogicalType::TIMESTAMP;
                        break;
                    case arrow::TimeUnit::MILLI:
                        duckdb_type = duckdb::LogicalType::TIMESTAMP_MS;
                        break;
                    case arrow::TimeUnit::SECOND:
                        duckdb_type = duckdb::LogicalType::TIMESTAMP_S;
                        break;
                }
            } else {
                // TODO: Do we need to check units here?
                // Technically this is supposed to be in microseconds like
                // TIMESTAMP
                duckdb_type = duckdb::LogicalType::TIMESTAMP_TZ;
            }
            break;
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
                arrow_field_to_duckdb(list_type->value_field());
            duckdb_type = duckdb::LogicalType::LIST(child_type);
            break;
        }
        case arrow::Type::LARGE_LIST: {
            auto list_type =
                std::static_pointer_cast<arrow::LargeListType>(arrow_type);
            auto [name, child_type] =
                arrow_field_to_duckdb(list_type->value_field());
            duckdb_type = duckdb::LogicalType::LIST(child_type);
            break;
        }
        case arrow::Type::STRUCT: {
            duckdb::child_list_t<duckdb::LogicalType> children;
            for (std::shared_ptr<arrow::Field> field : arrow_type->fields()) {
                auto [field_name, duckdb_type] = arrow_field_to_duckdb(field);
                children.push_back({field_name, duckdb_type});
            }
            duckdb_type = duckdb::LogicalType::STRUCT(children);
            break;
        }
        case arrow::Type::MAP: {
            auto map_type =
                std::static_pointer_cast<arrow::MapType>(arrow_type);
            auto [key_name, duckdb_key_type] =
                arrow_field_to_duckdb(map_type->key_field());
            auto [item_name, duckdb_value_type] =
                arrow_field_to_duckdb(map_type->item_field());
            duckdb_type =
                duckdb::LogicalType::MAP(duckdb_key_type, duckdb_value_type);
            break;
        }
        case arrow::Type::DICTIONARY: {
            auto dict_type =
                std::static_pointer_cast<arrow::DictionaryType>(arrow_type);
            std::shared_ptr<arrow::Field> value_field =
                arrow::field("name", dict_type->value_type());
            auto [field_name, duckdb_type] = arrow_field_to_duckdb(value_field);
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

std::string plan_to_string(std::unique_ptr<duckdb::LogicalOperator> &plan,
                           bool graphviz_format) {
    return plan->ToString(graphviz_format ? duckdb::ExplainFormat::GRAPHVIZ
                                          : duckdb::ExplainFormat::TEXT);
}

std::shared_ptr<PhysicalSource>
BodoDataFrameParallelScanFunctionData::CreatePhysicalOperator(
    std::vector<int> &selected_columns, duckdb::TableFilterSet &filter_exprs,
    duckdb::unique_ptr<duckdb::BoundLimitNode> &limit_val) {
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

    return std::make_shared<PhysicalReadPandas>(df, selected_columns);
}

std::shared_ptr<PhysicalSource>
BodoDataFrameSeqScanFunctionData::CreatePhysicalOperator(
    std::vector<int> &selected_columns, duckdb::TableFilterSet &filter_exprs,
    duckdb::unique_ptr<duckdb::BoundLimitNode> &limit_val) {
    return std::make_shared<PhysicalReadPandas>(df, selected_columns);
}

std::shared_ptr<PhysicalSource>
BodoParquetScanFunctionData::CreatePhysicalOperator(
    std::vector<int> &selected_columns, duckdb::TableFilterSet &filter_exprs,
    duckdb::unique_ptr<duckdb::BoundLimitNode> &limit_val) {
    return std::make_shared<PhysicalReadParquet>(
        path, pyarrow_schema, storage_options, selected_columns, filter_exprs,
        limit_val);
}

duckdb::idx_t get_operator_table_index(
    std::unique_ptr<duckdb::LogicalOperator> &op) {
    if (op->GetTableIndex().size() != 1) {
        throw std::runtime_error("Only one table index expected in operator");
    }
    return op->GetTableIndex()[0];
}

int planCountNodes(std::unique_ptr<duckdb::LogicalOperator> &op) {
    int ret = 1;  // count yourself
    for (auto &child : op->children) {
        ret += planCountNodes(child);
    }
    return ret;
}

void set_table_meta_from_arrow(int64_t table_pointer,
                               PyObject *pyarrow_schema) {
    table_info *table = reinterpret_cast<table_info *>(table_pointer);
    std::shared_ptr<arrow::Schema> arrow_schema = unwrap_schema(pyarrow_schema);

    // Set column names if not already set
    if (table->column_names.size() == 0) {
        for (int i = 0; i < arrow_schema->num_fields(); i++) {
            table->column_names.emplace_back(arrow_schema->field(i)->name());
        }
    } else if (table->column_names.size() !=
               static_cast<size_t>(arrow_schema->num_fields())) {
        throw std::runtime_error(
            "Number of columns in Arrow schema does not match table");
    } else {
        // Check that the column names match
        for (int i = 0; i < arrow_schema->num_fields(); i++) {
            if (table->column_names[i] != arrow_schema->field(i)->name()) {
                throw std::runtime_error(
                    "Column names in Arrow schema do not match table");
            }
        }
    }

    table->metadata = std::make_shared<TableMetadata>(
        arrow_schema->metadata()->keys(), arrow_schema->metadata()->values());
}

#undef CHECK_ARROW
#undef CHECK_ARROW_AND_ASSIGN
