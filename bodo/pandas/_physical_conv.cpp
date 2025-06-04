#include "_physical_conv.h"
#include <stdexcept>
#include "_bodo_scan_function.h"

#include "_util.h"
#include "physical/aggregate.h"
#include "physical/filter.h"
#include "physical/join.h"
#include "physical/limit.h"
#include "physical/project.h"
#include "physical/sample.h"
#include "physical/sort.h"

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
    // Getting source bindings before processing the source since there
    // are destructive operations in the Visit function like
    // std::move(op.expressions).
    // TODO: avoid destructive operations in the Visit function.
    std::vector<duckdb::ColumnBinding> source_cols =
        op.children[0]->GetColumnBindings();

    // Process the source of this projection.
    this->Visit(*op.children[0]);
    std::shared_ptr<bodo::Schema> in_table_schema =
        this->active_pipeline->getPrevOpOutputSchema();

    auto physical_op = std::make_shared<PhysicalProjection>(
        source_cols, std::move(op.expressions), in_table_schema);
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

    if (op.expressions.size() == 1 &&
        op.expressions[0]
                ->Cast<duckdb::BoundAggregateExpression>()
                .function.name == "count_star") {
        auto physical_op = std::make_shared<PhysicalCountStar>();
        // Finish the pipeline at this point so that Finalize can run
        // to reduce the number of collected rows to the desired amount.
        finished_pipelines.emplace_back(
            this->active_pipeline->Build(physical_op));
        // The same operator will exist in both pipelines.  The sink of the
        // previous pipeline and the source of the next one.
        // We record the pipeline dependency between these two pipelines.
        this->active_pipeline = std::make_shared<PipelineBuilder>(physical_op);
    } else {
        auto physical_agg =
            std::make_shared<PhysicalAggregate>(in_table_schema, op);
        // Finish the current pipeline with groupby build sink
        finished_pipelines.emplace_back(
            this->active_pipeline->Build(physical_agg));
        // Create a new pipeline with groupby output as source
        this->active_pipeline = std::make_shared<PipelineBuilder>(physical_agg);
    }
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
