#include "_physical_conv.h"
#include <stdexcept>
#include <string>
#include "_bodo_scan_function.h"

#include "_bodo_write_function.h"
#include "_util.h"
#include "physical/aggregate.h"
#include "physical/filter.h"
#include "physical/join.h"
#include "physical/limit.h"
#include "physical/project.h"
#include "physical/reduce.h"
#include "physical/sample.h"
#include "physical/sort.h"
#include "physical/write_parquet.h"

void PhysicalPlanBuilder::Visit(duckdb::LogicalGet& op) {
    // Get selected columns from LogicalGet to pass to physical
    // operators
    std::vector<int> selected_columns;
    for (auto& ci : op.GetColumnIds()) {
        selected_columns.push_back(ci.GetPrimaryIndex());
    }

    auto physical_op =
        op.bind_data->Cast<BodoScanFunctionData>().CreatePhysicalOperator(
            selected_columns, op.table_filters, op.extra_info.limit_val);
    if (this->active_pipeline != nullptr) {
        throw std::runtime_error(
            "LogicalGet operator should be the first operator in the pipeline");
    }
    this->active_pipeline = std::make_shared<PipelineBuilder>(physical_op);
}

void PhysicalPlanBuilder::Visit(duckdb::LogicalProjection& op) {
    std::vector<duckdb::ColumnBinding> source_cols =
        op.children[0]->GetColumnBindings();

    // Process the source of this projection.
    this->Visit(*op.children[0]);
    std::shared_ptr<bodo::Schema> in_table_schema =
        this->active_pipeline->getPrevOpOutputSchema();

    auto physical_op = std::make_shared<PhysicalProjection>(
        source_cols, op.expressions, in_table_schema);
    this->active_pipeline->AddOperator(physical_op);
}

void PhysicalPlanBuilder::Visit(duckdb::LogicalFilter& op) {
    // Process the source of this filter.
    this->Visit(*op.children[0]);
    std::shared_ptr<bodo::Schema> in_table_schema =
        this->active_pipeline->getPrevOpOutputSchema();
    std::vector<duckdb::ColumnBinding> source_cols =
        op.children[0]->GetColumnBindings();
    std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t> col_ref_map =
        getColRefMap(source_cols);

    std::shared_ptr<PhysicalExpression> physExprTree =
        buildPhysicalExprTree(op.expressions[0], col_ref_map);
    for (size_t i = 1; i < op.expressions.size(); ++i) {
        std::shared_ptr<PhysicalExpression> subExprTree =
            buildPhysicalExprTree(op.expressions[i], col_ref_map);
        physExprTree = std::static_pointer_cast<PhysicalExpression>(
            std::make_shared<PhysicalConjunctionExpression>(
                physExprTree, subExprTree,
                duckdb::ExpressionType::CONJUNCTION_AND));
    }
    std::shared_ptr<PhysicalFilter> physical_op =
        std::make_shared<PhysicalFilter>(physExprTree, in_table_schema);
    this->active_pipeline->AddOperator(physical_op);
}

void PhysicalPlanBuilder::Visit(duckdb::LogicalAggregate& op) {
    // Process the source of this aggregate.
    this->Visit(*op.children[0]);
    std::shared_ptr<bodo::Schema> in_table_schema =
        this->active_pipeline->getPrevOpOutputSchema();

    // Single column reduction like Series.max()
    if (op.groups.empty()) {
        for (const auto& expr : op.expressions) {
            if (expr->type != duckdb::ExpressionType::BOUND_AGGREGATE) {
                throw std::runtime_error(
                    "LogicalAggregate with no groups must have BOUND_AGGREGATE "
                    "expression types for reduction.");
            }
        }
        auto& agg_expr =
            op.expressions[0]->Cast<duckdb::BoundAggregateExpression>();
        if (agg_expr.function.name == "count_star") {
            auto physical_op = std::make_shared<PhysicalCountStar>();
            // Finish the pipeline at this point so that Finalize can run
            // to reduce the number of collected rows to the desired amount.
            finished_pipelines.emplace_back(
                this->active_pipeline->Build(physical_op));
            // The same operator will exist in both pipelines.  The sink of the
            // previous pipeline and the source of the next one.
            // We record the pipeline dependency between these two pipelines.
            this->active_pipeline =
                std::make_shared<PipelineBuilder>(physical_op);
            return;
        }
        std::vector<std::string> function_names;
        std::vector<duckdb::BoundAggregateExpression*> agg_expressions;
        for (auto& expr : op.expressions) {
            auto& agg_expr = expr->Cast<duckdb::BoundAggregateExpression>();
            agg_expressions.push_back(&agg_expr);
            function_names.emplace_back(agg_expr.function.name);
        }

        auto bodo_schema = std::make_shared<bodo::Schema>();
        BodoAggFunctionData& bind_info =
            agg_expressions[0]->bind_info->Cast<BodoAggFunctionData>();
        auto col_schema = bind_info.out_schema;
        auto bodo_col_schema = bodo::Schema::FromArrowSchema(col_schema);
        for (size_t i = 0; i < bodo_col_schema->column_types.size(); i++) {
            bodo_schema->append_column(
                bodo_col_schema->column_types[i]->copy());
            bodo_schema->column_names.push_back(std::to_string(i));
        }
        bodo_schema->metadata = std::make_shared<TableMetadata>(
            std::vector<std::string>({}), std::vector<std::string>({}));
        auto physical_op =
            std::make_shared<PhysicalReduce>(bodo_schema, function_names);
        finished_pipelines.emplace_back(
            this->active_pipeline->Build(physical_op));
        this->active_pipeline = std::make_shared<PipelineBuilder>(physical_op);
        return;
    }

    // Regular groupby aggregation with groups and expressions.
    auto physical_agg =
        std::make_shared<PhysicalAggregate>(in_table_schema, op);
    // Finish the current pipeline with groupby build sink
    finished_pipelines.emplace_back(this->active_pipeline->Build(physical_agg));
    // Create a new pipeline with groupby output as source
    this->active_pipeline = std::make_shared<PipelineBuilder>(physical_agg);
}

void PhysicalPlanBuilder::Visit(duckdb::LogicalOrder& op) {
    std::vector<duckdb::ColumnBinding> source_cols =
        op.children[0]->GetColumnBindings();
    this->Visit(*op.children[0]);
    std::shared_ptr<bodo::Schema> in_table_schema =
        this->active_pipeline->getPrevOpOutputSchema();

    auto physical_sort =
        std::make_shared<PhysicalSort>(op, in_table_schema, source_cols);
    finished_pipelines.emplace_back(
        this->active_pipeline->Build(physical_sort));
    this->active_pipeline = std::make_shared<PipelineBuilder>(physical_sort);
}

void PhysicalPlanBuilder::Visit(duckdb::LogicalComparisonJoin& op) {
    // See DuckDB code for background:
    // https://github.com/duckdb/duckdb/blob/d29a92f371179170688b4df394478f389bf7d1a6/src/execution/physical_plan/plan_comparison_join.cpp#L65
    // https://github.com/duckdb/duckdb/blob/d29a92f371179170688b4df394478f389bf7d1a6/src/execution/physical_operator.cpp#L196
    // https://github.com/duckdb/duckdb/blob/d29a92f371179170688b4df394478f389bf7d1a6/src/execution/operator/join/physical_join.cpp#L31

    auto physical_join = std::make_shared<PhysicalJoin>(op, op.conditions);

    // Create pipelines for the build side of the join (right child)
    PhysicalPlanBuilder rhs_builder;
    rhs_builder.Visit(*op.children[1]);
    std::shared_ptr<bodo::Schema> build_table_schema =
        rhs_builder.active_pipeline->getPrevOpOutputSchema();
    std::vector<std::shared_ptr<Pipeline>> build_pipelines =
        std::move(rhs_builder.finished_pipelines);
    build_pipelines.push_back(
        rhs_builder.active_pipeline->Build(physical_join));
    // Build pipelines need to execute before probe pipeline (recursively
    // handles multiple joins)
    this->finished_pipelines.insert(this->finished_pipelines.begin(),
                                    build_pipelines.begin(),
                                    build_pipelines.end());

    // Create pipelines for the probe side of the join (left child)
    this->Visit(*op.children[0]);
    std::shared_ptr<bodo::Schema> probe_table_schema =
        this->active_pipeline->getPrevOpOutputSchema();

    physical_join->InitializeJoinState(build_table_schema, probe_table_schema);

    this->active_pipeline->AddOperator(physical_join);
}

void PhysicalPlanBuilder::Visit(duckdb::LogicalSample& op) {
    // Process the source of this limit.
    this->Visit(*op.children[0]);
    std::shared_ptr<bodo::Schema> in_table_schema =
        this->active_pipeline->getPrevOpOutputSchema();

    duckdb::unique_ptr<duckdb::SampleOptions>& sampleOptions =
        op.sample_options;

    if (sampleOptions->is_percentage ||
        sampleOptions->method != duckdb::SampleMethod::SYSTEM_SAMPLE) {
        throw std::runtime_error("LogicalSample unsupported offset");
    }

    std::shared_ptr<PhysicalSample> physical_op;

    std::visit(
        [&physical_op, &in_table_schema](const auto& value) {
            using T = std::decay_t<decltype(value)>;

            // Allow only types that can safely convert to int
            if constexpr (std::is_convertible_v<T, uint64_t>) {
                physical_op =
                    std::make_shared<PhysicalSample>(value, in_table_schema);
            }
        },
        extractValue(sampleOptions->sample_size));
    if (!physical_op) {
        throw std::runtime_error(
            "Cannot convert duckdb::Value to limit integer.");
    }
    this->active_pipeline->AddOperator(physical_op);
}

void PhysicalPlanBuilder::Visit(duckdb::LogicalLimit& op) {
    // Process the source of this limit.
    this->Visit(*op.children[0]);
    std::shared_ptr<bodo::Schema> in_table_schema =
        this->active_pipeline->getPrevOpOutputSchema();

    if (op.offset_val.Type() != duckdb::LimitNodeType::CONSTANT_VALUE ||
        op.offset_val.GetConstantValue() != 0) {
        throw std::runtime_error("LogicalLimit unsupported offset");
    }
    if (op.limit_val.Type() != duckdb::LimitNodeType::CONSTANT_VALUE) {
        throw std::runtime_error("LogicalLimit unsupported limit type");
    }
    duckdb::idx_t n = op.limit_val.GetConstantValue();
    auto physical_op = std::make_shared<PhysicalLimit>(n, in_table_schema);
    // Finish the pipeline at this point so that Finalize can run
    // to reduce the number of collected rows to the desired amount.
    finished_pipelines.emplace_back(this->active_pipeline->Build(physical_op));
    // The same operator will exist in both pipelines.  The sink of the
    // previous pipeline and the source of the next one.
    // We record the pipeline dependency between these two pipelines.
    this->active_pipeline = std::make_shared<PipelineBuilder>(physical_op);
}

void PhysicalPlanBuilder::Visit(duckdb::LogicalTopN& op) {
    std::vector<duckdb::ColumnBinding> source_cols =
        op.children[0]->GetColumnBindings();

    // Process the source of this TopN.
    this->Visit(*op.children[0]);
    std::shared_ptr<bodo::Schema> in_table_schema =
        this->active_pipeline->getPrevOpOutputSchema();

    auto physical_sort = std::make_shared<PhysicalSort>(
        op, in_table_schema, source_cols, op.limit, op.offset);
    finished_pipelines.emplace_back(
        this->active_pipeline->Build(physical_sort));
    this->active_pipeline = std::make_shared<PipelineBuilder>(physical_sort);
}

void PhysicalPlanBuilder::Visit(duckdb::LogicalCopyToFile& op) {
    this->Visit(*op.children[0]);
    std::shared_ptr<bodo::Schema> in_table_schema =
        this->active_pipeline->getPrevOpOutputSchema();

    BodoWriteFunctionData& write_data =
        op.bind_data->Cast<BodoWriteFunctionData>();
    auto physical_op = write_data.CreatePhysicalOperator(in_table_schema);

    finished_pipelines.emplace_back(this->active_pipeline->Build(physical_op));
    this->active_pipeline = nullptr;
}
