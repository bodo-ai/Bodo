#include "_physical_conv.h"
#include <stdexcept>
#include <string>
#include "_bodo_scan_function.h"

#include "_bodo_write_function.h"
#include "_plan.h"
#include "_util.h"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "physical/aggregate.h"
#include "physical/cte.h"
#include "physical/filter.h"
#if USE_CUDF
#include "physical/gpu_aggregate.h"
#include "physical/gpu_filter.h"
#include "physical/gpu_join.h"
#include "physical/gpu_project.h"
#endif
#include "physical/join.h"
#include "physical/join_filter.h"
#include "physical/limit.h"
#include "physical/project.h"
#include "physical/quantile.h"
#include "physical/read_empty.h"
#include "physical/reduce.h"
#include "physical/sample.h"
#include "physical/sort.h"
#include "physical/union_all.h"

void PhysicalPlanBuilder::Visit(duckdb::LogicalGet& op) {
    // Get selected columns from LogicalGet to pass to physical
    // operators
    std::vector<int> selected_columns;

    // DuckDB sets projection_ids when there are columns used in pushed down
    // filters that are not used anywhere else in the query.
    auto& column_ids = op.GetColumnIds();
    if (!op.projection_ids.empty()) {
        for (const auto& col : op.projection_ids) {
            selected_columns.push_back(column_ids[col].GetPrimaryIndex());
        }
    } else {
        for (auto& ci : column_ids) {
            selected_columns.push_back(ci.GetPrimaryIndex());
        }
    }

    // Turns out duckdb is actually generating these but they should always be
    // optional
    // TODO: implement dynamic filters for logical get nodes
    // if (op.dynamic_filters) {
    //    throw std::runtime_error(
    //        "PhysicalPlanBuilder::Visit LogicalGet: dynamic filters not "
    //        "supported");
    //}
    //
    BodoScanFunctionData& scan_data =
        op.bind_data->Cast<BodoScanFunctionData>();

    bool run_on_gpu = node_run_on_gpu(op);
    auto physical_op = scan_data.CreatePhysicalOperator(
        selected_columns, op.table_filters, op.extra_info.limit_val,
        this->join_filter_states, run_on_gpu);
    if (this->active_pipeline != nullptr) {
        throw std::runtime_error(
            "LogicalGet operator should be the first operator in the pipeline");
    }
    this->active_pipeline = std::make_shared<PipelineBuilder>(physical_op);

    // If the logical get is associated with runtime join filter states,
    // add the join filter pipelines to be run before this scan operator so the
    // stats are available. This should already be the case since filtering is
    // done on reads on the probe side, but added for completeness.
    if (scan_data.rtjf_state_map.has_value()) {
        for (const auto& [join_id, state] : scan_data.rtjf_state_map.value()) {
            this->active_pipeline->addRunBefore(
                this->join_filter_pipelines->at(join_id));
        }
    }
}

void PhysicalPlanBuilder::Visit(duckdb::LogicalEmptyResult& op) {
    auto physical_op = std::make_shared<PhysicalReadEmpty>(op.return_types);
    if (this->active_pipeline != nullptr) {
        throw std::runtime_error(
            "LogicalEmptyResult operator should be the first operator in the "
            "pipeline");
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

#ifdef USE_CUDF
    std::variant<std::shared_ptr<PhysicalProjection>,
                 std::shared_ptr<PhysicalGPUProjection>>
        physical_op;

    bool run_on_gpu = node_run_on_gpu(op);
    if (run_on_gpu) {
        physical_op = std::make_shared<PhysicalGPUProjection>(
            source_cols, op.expressions, in_table_schema);
    } else {
        physical_op = std::make_shared<PhysicalProjection>(
            source_cols, op.expressions, in_table_schema);
    }
#else   // USE_CUDF
    std::variant<std::shared_ptr<PhysicalProjection>> physical_op;
    physical_op = std::make_shared<PhysicalProjection>(
        source_cols, op.expressions, in_table_schema);
#endif  // USE_CUDF

    std::visit([&](auto& vop) { this->active_pipeline->AddOperator(vop); },
               physical_op);
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

    // Duckdb can produce empty filters.
    if (op.expressions.size() == 0) {
        return;
    }

#ifdef USE_CUDF
    std::variant<std::shared_ptr<PhysicalFilter>,
                 std::shared_ptr<PhysicalGPUFilter>>
        physical_op;

    bool run_on_gpu = node_run_on_gpu(op);
    if (run_on_gpu) {
        physical_op = std::make_shared<PhysicalGPUFilter>(
            op, op.expressions, in_table_schema, col_ref_map);
    } else {
        physical_op = std::make_shared<PhysicalFilter>(
            op, op.expressions, in_table_schema, col_ref_map);
    }
#else   // USE_CUDF
    std::variant<std::shared_ptr<PhysicalFilter>> physical_op;
    physical_op = std::make_shared<PhysicalFilter>(
        op, op.expressions, in_table_schema, col_ref_map);
#endif  // USE_CUDF

    std::visit([&](auto& vop) { this->active_pipeline->AddOperator(vop); },
               physical_op);
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
        std::vector<std::string> function_names;
        for (auto& expr : op.expressions) {
            auto& agg_expr = expr->Cast<duckdb::BoundAggregateExpression>();
            function_names.emplace_back(agg_expr.function.name);
        }

        if (function_names[0] == "count_star") {
            if (op.expressions.size() != 1) {
                throw std::runtime_error(
                    "CountStar must have exactly one "
                    "aggregate expression for reduction.");
            }
            auto physical_op = std::make_shared<PhysicalCountStar>();
            // Finish the pipeline at this point so that Finalize can run
            // to reduce the number of collected rows to the desired amount.
            // The same operator will exist in both pipelines.  The sink of the
            // previous pipeline and the source of the next one.
            // We record the pipeline dependency between these two pipelines.
            FinishPipelineOneOperator(physical_op);
            return;
        }
        // bind_info in every expression stores the same schema for the entire
        // list, formatted on the Python side; therefore we only extract
        // bind_info of first element.
        auto& agg_expr =
            op.expressions[0]->Cast<duckdb::BoundAggregateExpression>();
        BodoAggFunctionData& bind_info =
            agg_expr.bind_info->Cast<BodoAggFunctionData>();
        auto bodo_schema = std::make_shared<bodo::Schema>();
        auto col_schema = bind_info.out_schema;
        auto bodo_col_schema = bodo::Schema::FromArrowSchema(col_schema);
        for (size_t i = 0; i < bodo_col_schema->column_types.size(); i++) {
            bodo_schema->append_column(
                bodo_col_schema->column_types[i]->copy());
            bodo_schema->column_names.push_back(std::to_string(i));
        }
        bodo_schema->metadata = std::make_shared<bodo::TableMetadata>(
            std::vector<std::string>({}), std::vector<std::string>({}));

        // If function_names includes quantiles, create a PhysicalQuantile
        // operator. Function names for quantile evaluations are formatted as
        // f"quantile_{value}" where q=value (i.e. "quantile_0.5" for q=0.5).
        if (function_names[0].starts_with("quantile")) {
            std::vector<double> quantiles{};
            for (auto it : function_names) {
                if (!it.starts_with("quantile")) {
                    throw std::runtime_error(
                        "quantile functions cannot be mixed with other "
                        "aggregate operations.");
                }
                quantiles.push_back(std::stod(it.substr(9)));
            }

#define CREATE_QUANTILE(dtype)                                              \
    if (in_table_schema->column_types[0]->c_type == dtype) {                \
        auto physical_op =                                                  \
            std::make_shared<PhysicalQuantile<dtype_to_type<dtype>::type>>( \
                bodo_schema, quantiles);                                    \
        FinishPipelineOneOperator(physical_op);                             \
        return;                                                             \
    }
            // Tried to roughly order by how common the types are
            CREATE_QUANTILE(Bodo_CTypes::CTypeEnum::FLOAT64);
            CREATE_QUANTILE(Bodo_CTypes::CTypeEnum::INT64);
            CREATE_QUANTILE(Bodo_CTypes::CTypeEnum::INT32);
            CREATE_QUANTILE(Bodo_CTypes::CTypeEnum::FLOAT32);
            CREATE_QUANTILE(Bodo_CTypes::CTypeEnum::INT16);
            CREATE_QUANTILE(Bodo_CTypes::CTypeEnum::INT8);
            CREATE_QUANTILE(Bodo_CTypes::CTypeEnum::UINT16);
            CREATE_QUANTILE(Bodo_CTypes::CTypeEnum::UINT8);
            CREATE_QUANTILE(Bodo_CTypes::CTypeEnum::UINT32);
            CREATE_QUANTILE(Bodo_CTypes::CTypeEnum::UINT64);
            throw std::runtime_error(
                "quantile function is not supported for the type: " +
                bodo_schema->column_types[0]->ToString());
#undef CREATE_QUANTILE
        }

        // Otherwise, create a PhysicalReduce operator
        auto physical_op =
            std::make_shared<PhysicalReduce>(bodo_schema, function_names);
        FinishPipelineOneOperator(physical_op);
        return;
    }

    // Regular groupby aggregation with groups and expressions.
#ifdef USE_CUDF
    std::variant<std::shared_ptr<PhysicalAggregate>,
                 std::shared_ptr<PhysicalGPUAggregate>>
        physical_op;

    bool run_on_gpu = node_run_on_gpu(op);
    if (run_on_gpu) {
        physical_op =
            std::make_shared<PhysicalGPUAggregate>(in_table_schema, op);
    } else {
        physical_op = std::make_shared<PhysicalAggregate>(in_table_schema, op);
    }
#else   // USE_CUDF
    std::variant<std::shared_ptr<PhysicalAggregate>> physical_op;
    physical_op = std::make_shared<PhysicalAggregate>(in_table_schema, op);
#endif  // USE_CUDF

    // Finish the current pipeline with groupby build sink.
    // Create a new pipeline with groupby output as source.
    std::visit([&](auto& vop) { FinishPipelineOneOperator(vop); }, physical_op);
}

void PhysicalPlanBuilder::Visit(duckdb::LogicalOrder& op) {
    std::vector<duckdb::ColumnBinding> source_cols =
        op.children[0]->GetColumnBindings();
    this->Visit(*op.children[0]);
    std::shared_ptr<bodo::Schema> in_table_schema =
        this->active_pipeline->getPrevOpOutputSchema();

    auto physical_sort =
        std::make_shared<PhysicalSort>(op, in_table_schema, source_cols);
    FinishPipelineOneOperator(physical_sort);
}

void PhysicalPlanBuilder::Visit(duckdb::LogicalComparisonJoin& op) {
    // See DuckDB code for background:
    // https://github.com/duckdb/duckdb/blob/d29a92f371179170688b4df394478f389bf7d1a6/src/execution/physical_plan/plan_comparison_join.cpp#L65
    // https://github.com/duckdb/duckdb/blob/d29a92f371179170688b4df394478f389bf7d1a6/src/execution/physical_operator.cpp#L196
    // https://github.com/duckdb/duckdb/blob/d29a92f371179170688b4df394478f389bf7d1a6/src/execution/operator/join/physical_join.cpp#L31

#ifdef USE_CUDF
    std::variant<std::shared_ptr<PhysicalJoin>,
                 std::shared_ptr<PhysicalGPUJoin>>
        physical_join;
    if (node_run_on_gpu(op)) {
        physical_join = std::make_shared<PhysicalGPUJoin>(op);
        (*this->join_on_gpu).insert({op.join_id, true});
    } else {
        physical_join = std::make_shared<PhysicalJoin>(op);
        (*this->join_on_gpu).insert({op.join_id, false});
    }
#else
    std::shared_ptr<PhysicalJoin> physical_join =
        std::make_shared<PhysicalJoin>(op);
#endif

    // Create pipelines for the build side of the join (right child)
    PhysicalPlanBuilder rhs_builder(ctes, run_on_gpu, join_filter_states,
                                    join_filter_pipelines);
    rhs_builder.Visit(*op.children[1]);
    std::shared_ptr<bodo::Schema> build_table_schema =
        rhs_builder.active_pipeline->getPrevOpOutputSchema();

    /*
     * Originally in this code there was a chicken-and-egg problem.
     * We needed probe and build schemas to construct the PhysicalJoin
     * and needed the PhysicalJoin to make the build-side pipeline and
     * that pipeline needed to fill in the filter_states and the
     * filter states needed by the probe side of the join but that was
     * impossible since you had to process the probe side before you
     * could get the probe-side schema needed for the PhysicalJoin.
     *
     * So, the initialization of PhysicalJoin has been broken up into
     * the duckdb operation and then separately later the providing of
     * the schemas and all the usage of that to configure the join.
     * This later function is buildProbeSchemas below.  So now, we can
     * make the PhysicalJoin object as the first thing and then use
     * that to store the build-side pipeline in join_filter_states so
     * that it is accessible when processing the probe side.
     */
    std::shared_ptr<Pipeline> done_pipeline;
#ifdef USE_CUDF
    std::visit(
        [&](auto& vop) {
            done_pipeline = rhs_builder.active_pipeline->Build(vop);
        },
        physical_join);
#else
    done_pipeline = rhs_builder.active_pipeline->Build(physical_join);
#endif
    if (!done_pipeline) {
        throw std::runtime_error("done_pipeline null in ComparisonJoin.");
    }
    // Visit children[0] only need the pipeline portion.
    (*this->join_filter_pipelines)[op.join_id] = done_pipeline;

    // Create pipelines for the probe side of the join (left child)
    this->Visit(*op.children[0]);
    std::shared_ptr<bodo::Schema> probe_table_schema =
        this->active_pipeline->getPrevOpOutputSchema();

#ifdef USE_CUDF
    std::visit(
        [&](auto& vop) {
            vop->buildProbeSchemas(op, op.conditions, build_table_schema,
                                   probe_table_schema);
            (*this->join_filter_states)[op.join_id] = vop->getJoinStatePtr();
            this->active_pipeline->AddOperator(vop);
        },
        physical_join);
#else
    physical_join->buildProbeSchemas(op, op.conditions, build_table_schema,
                                     probe_table_schema);
    (*this->join_filter_states)[op.join_id] = physical_join->getJoinStatePtr();
    this->active_pipeline->AddOperator(physical_join);
#endif
    // Build side pipeline runs before probe side.
    this->active_pipeline->addRunBefore(done_pipeline);
}

void PhysicalPlanBuilder::Visit(bodo::LogicalJoinFilter& op) {
    // Process the source of this join filter.
    this->Visit(*op.children[0]);

#ifdef USE_CUDF
    bool all_joins_on_gpu = true;
    for (int filter_id : op.filter_ids) {
        all_joins_on_gpu = all_joins_on_gpu && !(*join_on_gpu)[filter_id];
    }
    if (all_joins_on_gpu) {
        // Don't need to add a pipeline stage if all joins will be
        // run on GPU.
        return;
    }

    bool run_on_gpu = node_run_on_gpu(op);
    if (run_on_gpu) {
        // If the planner wants this JoinFilter to run on GPU
        // then we won't do join filtering even if the join itself
        // is done on CPU.
        return;
    }
#endif

    std::shared_ptr<bodo::Schema> in_table_schema =
        this->active_pipeline->getPrevOpOutputSchema();

    std::shared_ptr<PhysicalJoinFilter> physical_op =
        std::make_shared<PhysicalJoinFilter>(op, in_table_schema,
                                             this->join_filter_states);
    this->active_pipeline->AddOperator(physical_op);

    // Make sure all filter generators used by this
    // join filter run before this pipeline.
    for (int filter_id : op.filter_ids) {
#ifdef USE_CUDF
        if ((*join_on_gpu)[filter_id]) {
            continue;
        }
#endif
        std::shared_ptr<Pipeline> filter_pipeline =
            (*join_filter_pipelines)[filter_id];
        if (!filter_pipeline) {
            throw std::runtime_error(
                "Pipeline for given filter id not found in "
                "join_filter_states.");
        }
        // Make sure generator of the filter runs before
        // this consumer of the filter.
        this->active_pipeline->addRunBefore(filter_pipeline);
    }
}

void PhysicalPlanBuilder::Visit(duckdb::LogicalMaterializedCTE& op) {
    // Create pipelines for the duplicate side of the CTE.
    this->Visit(*op.children[0]);
    std::shared_ptr<bodo::Schema> in_table_schema =
        this->active_pipeline->getPrevOpOutputSchema();
    std::shared_ptr<Pipeline> done_pipeline;
#ifdef USE_CUDF
    std::variant<std::shared_ptr<PhysicalCTE>, std::shared_ptr<PhysicalGPUCTE>>
        physical_cte;

    if (node_run_on_gpu(op)) {
        physical_cte = std::make_shared<PhysicalGPUCTE>(in_table_schema);
    } else {
        physical_cte = std::make_shared<PhysicalCTE>(in_table_schema);
    }

    std::visit(
        [&](auto& vop) {
            done_pipeline = this->active_pipeline->Build(vop);
            ctes.insert(
                {op.table_index,
                 {.physical_node = vop, .cte_pipeline_root = done_pipeline}});
        },
        physical_cte);
#else
    std::shared_ptr<PhysicalCTE> physical_cte =
        std::make_shared<PhysicalCTE>(in_table_schema);
    done_pipeline = this->active_pipeline->Build(physical_cte);
    // Save the physical_cte node away so that cte ref's on the non-duplicate
    // side can find it.
    ctes.insert(
        {op.table_index,
         {.physical_node = physical_cte, .cte_pipeline_root = done_pipeline}});
#endif

    // The active pipeline finishes after the duplicate side.
    this->active_pipeline = nullptr;
    // Create pipelines for the side that uses the duplicate side.
    this->Visit(*op.children[1]);
}

void PhysicalPlanBuilder::Visit(duckdb::LogicalCTERef& op) {
    // Match the cte_index with the CTE node table index in the global
    // structure.
    auto table_index_iter = ctes.find(op.cte_index);
    if (table_index_iter == ctes.end()) {
        throw std::runtime_error(
            "LogicalCTERef couldn't find matching table_index.");
    }
    CTEInfo& cte_index_info = table_index_iter->second;
#ifdef USE_CUDF
    std::variant<std::shared_ptr<PhysicalCTERef>,
                 std::shared_ptr<PhysicalGPUCTERef>>
        physical_cte_ref;
    std::visit(
        [&](auto& pn) {
            using U = std::decay_t<decltype(pn)>;

            if constexpr (std::is_same_v<U, std::shared_ptr<PhysicalCTE>>) {
                if (node_run_on_gpu(op)) {
                    throw std::runtime_error(
                        "Got mismatch with CPU CTE and GPU CTERef in "
                        "PhysicalPlanBuidler::Visit(LogicalCTEref).");
                } else {
                    physical_cte_ref = std::make_shared<PhysicalCTERef>(pn);
                }
            } else if constexpr (std::is_same_v<
                                     U, std::shared_ptr<PhysicalGPUCTE>>) {
                if (node_run_on_gpu(op)) {
                    physical_cte_ref = std::make_shared<PhysicalGPUCTERef>(pn);
                } else {
                    throw std::runtime_error(
                        "Got mismatch with GPU CTE and CPU CTERef in "
                        "PhysicalPlanBuidler::Visit(LogicalCTEref).");
                }
            } else {
                throw std::runtime_error(
                    "Got unknown type in "
                    "PhysicalPlanBuidler::Visit(LogicalCTEref).");
            }
        },
        cte_index_info.physical_node);
    std::visit(
        [&](auto& vop) {
            this->active_pipeline = std::make_shared<PipelineBuilder>(vop);
        },
        physical_cte_ref);
#else
    std::shared_ptr<PhysicalCTERef> physical_cte_ref =
        std::make_shared<PhysicalCTERef>(cte_index_info.physical_node);
    this->active_pipeline = std::make_shared<PipelineBuilder>(physical_cte_ref);
#endif
    this->active_pipeline->addRunBefore(cte_index_info.cte_pipeline_root);
}

void PhysicalPlanBuilder::Visit(duckdb::LogicalCrossProduct& op) {
    // Same as LogicalComparisonJoin, but without conditions.

    // Create pipelines for the build side of the join (right child)
    PhysicalPlanBuilder rhs_builder(ctes, run_on_gpu, join_filter_states,
                                    join_filter_pipelines);
    rhs_builder.Visit(*op.children[1]);
    std::shared_ptr<bodo::Schema> build_table_schema =
        rhs_builder.active_pipeline->getPrevOpOutputSchema();

    // Create pipelines for the probe side of the join (left child)
    this->Visit(*op.children[0]);
    std::shared_ptr<bodo::Schema> probe_table_schema =
        this->active_pipeline->getPrevOpOutputSchema();

    auto physical_join = std::make_shared<PhysicalJoin>(op, build_table_schema,
                                                        probe_table_schema);

    std::shared_ptr<Pipeline> done_pipeline =
        rhs_builder.active_pipeline->Build(physical_join);
    this->active_pipeline->AddOperator(physical_join);
    // Build side pipeline runs before probe side.
    this->active_pipeline->addRunBefore(done_pipeline);
}

/*
 * arrowSchemeTypeEquals
 *
 * Used to compare two arrow schema for type equality.
 * Can be used when schema may have differing column names
 * and thus the regular schema equality test is too strict.
 */
bool arrowSchemaTypeEquals(const ::arrow::Schema& s1,
                           const ::arrow::Schema& s2) {
    if (s1.num_fields() != s2.num_fields())
        return false;

    for (int i = 0; i < s1.num_fields(); ++i) {
        if (!s1.field(i)->type()->Equals(*s2.field(i)->type())) {
            return false;
        }
    }
    return true;
}

void PhysicalPlanBuilder::Visit(duckdb::LogicalSetOperation& op) {
    if (op.type == duckdb::LogicalOperatorType::LOGICAL_UNION) {
        // UNION ALL
        if (op.setop_all) {
            // Right-child will feed into a table.
            PhysicalPlanBuilder rhs_builder(
                ctes, run_on_gpu, join_filter_states, join_filter_pipelines);
            rhs_builder.Visit(*op.children[1]);
            std::shared_ptr<bodo::Schema> rhs_table_schema =
                rhs_builder.active_pipeline->getPrevOpOutputSchema();
            ::arrow::Schema rhs_arrow = *(rhs_table_schema->ToArrowSchema());

            auto physical_union_all =
                std::make_shared<PhysicalUnionAll>(rhs_table_schema);
            std::shared_ptr<Pipeline> done_pipeline =
                rhs_builder.active_pipeline->Build(physical_union_all);

            // Left-child will feed into the same table.
            this->Visit(*op.children[0]);
            std::shared_ptr<bodo::Schema> lhs_table_schema =
                this->active_pipeline->getPrevOpOutputSchema();
            ::arrow::Schema lhs_arrow = *(lhs_table_schema->ToArrowSchema());
            if (!arrowSchemaTypeEquals(rhs_arrow, lhs_arrow)) {
                throw std::runtime_error(
                    "PhysicalPlanBuilder::Visit(LogicalSetOperation lhs and "
                    "rhs schemas not identical. " +
                    lhs_arrow.ToString() + " versus " + rhs_arrow.ToString());
            }

            this->active_pipeline->AddOperator(physical_union_all);
            this->active_pipeline->addRunBefore(done_pipeline);
        } else {
            throw std::runtime_error(
                "PhysicalPlanBuilder::Visit(LogicalSetOperation non-all union "
                "unsupported");
        }
    } else {
        throw std::runtime_error(
            "PhysicalPlanBuilder::Visit(LogicalSetOperation unsupported "
            "logical operator type " +
            std::to_string(static_cast<int>(op.type)));
    }
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
    // The same operator will exist in both pipelines.  The sink of the
    // previous pipeline and the source of the next one.
    // We record the pipeline dependency between these two pipelines.
    FinishPipelineOneOperator(physical_op);
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
    FinishPipelineOneOperator(physical_sort);
}

void PhysicalPlanBuilder::Visit(duckdb::LogicalCopyToFile& op) {
    this->Visit(*op.children[0]);
    std::shared_ptr<bodo::Schema> in_table_schema =
        this->active_pipeline->getPrevOpOutputSchema();

    BodoWriteFunctionData& write_data =
        op.bind_data->Cast<BodoWriteFunctionData>();
    bool run_on_gpu = node_run_on_gpu(op);
    auto physical_op =
        write_data.CreatePhysicalOperator(in_table_schema, run_on_gpu);

    this->terminal_pipeline = this->active_pipeline->Build(physical_op);
    this->active_pipeline = nullptr;
}

void PhysicalPlanBuilder::Visit(duckdb::LogicalDistinct& op) {
    this->Visit(*op.children[0]);
    std::shared_ptr<bodo::Schema> in_table_schema =
        this->active_pipeline->getPrevOpOutputSchema();

    if (op.order_by) {
        throw std::runtime_error(
            "LogicalDistinct node with order_by field not supported");
    }

    // Regular groupby aggregation with groups and expressions.
#ifdef USE_CUDF
    std::variant<std::shared_ptr<PhysicalAggregate>,
                 std::shared_ptr<PhysicalGPUAggregate>>
        physical_op;

    bool run_on_gpu = node_run_on_gpu(op);
    if (run_on_gpu) {
        physical_op =
            std::make_shared<PhysicalGPUAggregate>(in_table_schema, op);
    } else {
        physical_op = std::make_shared<PhysicalAggregate>(in_table_schema, op);
    }
#else   // USE_CUDF
    std::variant<std::shared_ptr<PhysicalAggregate>> physical_op;
    physical_op = std::make_shared<PhysicalAggregate>(in_table_schema, op);
#endif  // USE_CUDF

    // Finish the current pipeline with groupby build sink.
    // Create a new pipeline with groupby output as source.
    std::visit([&](auto& vop) { FinishPipelineOneOperator(vop); }, physical_op);
}
