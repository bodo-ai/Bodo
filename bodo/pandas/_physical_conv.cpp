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
#include "physical/gpu_join_filter.h"
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

// Remove all non-equi join conditions from `conditions` and return them
// as JoinConditions.
// After this call, `conditions` will contain only equi-join conditions.
duckdb::vector<duckdb::JoinCondition> extractNonEquiConditions(
    std::vector<duckdb::JoinCondition>& conditions) {
    duckdb::vector<duckdb::JoinCondition> non_equi_exprs;
    size_t i = 0;
    while (i < conditions.size()) {
        auto& cond = conditions[i];

        if (cond.IsComparison() &&
            cond.GetComparisonType() == duckdb::ExpressionType::COMPARE_EQUAL) {
            ++i;
        } else {
            // Move the JoinCondition into CreateExpression to produce an
            // Expression
            non_equi_exprs.push_back(std::move(cond));

            // Remove the moved-out element from the vector efficiently by
            // swapping with back
            conditions[i] = std::move(conditions.back());
            conditions.pop_back();
            // do not increment i, because we swapped a new element into
            // position i
        }
    }

    return non_equi_exprs;
}

struct MissingBindingsResult {
    std::vector<duckdb::ColumnBinding> missing_in_probe;
    std::vector<duckdb::ColumnBinding> missing_in_build;
    std::vector<duckdb::ColumnBinding> missing_everywhere;
};

// Check whether a child-column index appears in the projection map (proj_map
// contains idx_t column indices)
static bool BindingColumnIndexInProjMap(
    const std::vector<duckdb::idx_t>& proj_map, duckdb::idx_t column_index) {
    for (auto& idx : proj_map) {
        if (idx == column_index) {
            return true;
        }
    }
    return false;
}

// Find missing bindings relative to left/right projection maps.
// probe_table_bindings and build_table_bindings are sets of table_index values
// that belong to each side. left_projection_map and right_projection_map are
// vectors of idx_t representing column indices produced by each child.
MissingBindingsResult FindMissingBindingsInProjectionMaps(
    duckdb::unique_ptr<duckdb::Expression>& expr,
    const std::vector<duckdb::idx_t>& left_projection_map,
    const std::vector<duckdb::idx_t>& right_projection_map,
    const std::unordered_set<duckdb::idx_t>& probe_table_bindings,
    const std::unordered_set<duckdb::idx_t>& build_table_bindings) {
    // collect unique ColumnBinding references from the expression
    std::unordered_set<uint64_t> seen;
    std::vector<duckdb::ColumnBinding> referenced;

    duckdb::ExpressionIterator::EnumerateExpression(
        expr, [&](duckdb::Expression& child) {
            if (child.GetExpressionClass() ==
                duckdb::ExpressionClass::BOUND_COLUMN_REF) {
                auto& colref = child.Cast<duckdb::BoundColumnRefExpression>();
                duckdb::ColumnBinding b = colref.binding;
                uint64_t key = (static_cast<uint64_t>(b.table_index) << 32) |
                               static_cast<uint64_t>(b.column_index);
                if (seen.insert(key).second) {
                    referenced.push_back(b);
                }
            }
        });

    MissingBindingsResult result;
    for (auto& b : referenced) {
        bool in_left_map =
            BindingColumnIndexInProjMap(left_projection_map, b.column_index);
        bool in_right_map =
            BindingColumnIndexInProjMap(right_projection_map, b.column_index);

        if (!in_left_map && !in_right_map) {
            // Not present in either projection map
            // Decide which side it logically belongs to using table_index sets
            bool belongs_left = probe_table_bindings.find(b.table_index) !=
                                probe_table_bindings.end();
            bool belongs_right = build_table_bindings.find(b.table_index) !=
                                 build_table_bindings.end();

            if (belongs_left) {
                result.missing_in_probe.push_back(b);
            } else if (belongs_right) {
                result.missing_in_build.push_back(b);
            } else {
                // Table index not found in either side: treat as
                // missing_everywhere
                result.missing_everywhere.push_back(b);
            }
            continue;
        }

        // If the binding's column index is present in one of the maps, but the
        // table_index indicates it belongs to the other side, treat that as
        // missing on the side it logically belongs to.
        bool belongs_left = probe_table_bindings.find(b.table_index) !=
                            probe_table_bindings.end();
        bool belongs_right = build_table_bindings.find(b.table_index) !=
                             build_table_bindings.end();

        if (belongs_left && !in_left_map) {
            result.missing_in_probe.push_back(b);
        } else if (belongs_right && !in_right_map) {
            result.missing_in_build.push_back(b);
        } else {
            // If it is present in a map consistent with its table_index,
            // nothing is missing. If ambiguous (table_index in neither set),
            // treat as missing_everywhere for safety.
            if (!belongs_left && !belongs_right) {
                result.missing_everywhere.push_back(b);
            }
        }
    }

    return result;
}

void DumpTableBindings(const std::unordered_set<duckdb::idx_t>& v,
                       const std::string& label) {
    std::cout << label << " ";
    for (auto& i : v) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

void fill_if_empty(std::vector<duckdb::idx_t>& proj_map, int num_cols) {
    if (proj_map.size() == 0) {
        std::cout << "fill_if_empty is filling " << num_cols << std::endl;
        for (int i = 0; i < num_cols; ++i) {
            proj_map.push_back(i);
        }
    }
}

duckdb::vector<duckdb::idx_t> gen_new_proj_map(
    const duckdb::vector<duckdb::idx_t>& orig_map,
    const std::vector<duckdb::ColumnBinding>& needed, int& added) {
    // start with the original projection map
    duckdb::vector<duckdb::idx_t> ret = orig_map;

    // build a fast lookup set of existing column_index values
    std::unordered_set<duckdb::idx_t> seen;
    seen.reserve(ret.size() + needed.size());
    for (auto idx : ret) {
        seen.insert(idx);
    }

    added = 0;
    // ensure every needed.column_index is present in ret
    for (const auto& b : needed) {
        duckdb::idx_t col = b.column_index;
        if (seen.find(col) == seen.end()) {
            ret.push_back(col);
            ++added;
            seen.insert(col);
        }
    }

    return ret;
}

template <typename T>
int index_of(const std::vector<T>& v, const T& value) {
    auto it = std::find(v.begin(), v.end(), value);
    return it == v.end() ? -1 : int(it - v.begin());
}

duckdb::vector<duckdb::idx_t> gen_split_filter_projection_map(
    std::vector<uint64_t> probe_orig, std::vector<uint64_t> probe_new,
    std::vector<uint64_t> build_orig, std::vector<uint64_t> build_new) {
    duckdb::vector<duckdb::idx_t> ret;
    for (auto& in_probe_orig : probe_orig) {
        ret.push_back(index_of(probe_new, in_probe_orig));
    }
    uint64_t base = ret.size();
    for (auto& in_build_orig : build_orig) {
        ret.push_back(base + index_of(build_new, in_build_orig));
    }
    return ret;
#if 0
        int probe_added, int build_added, int left_total, int right_total) {
    duckdb::vector<duckdb::idx_t> ret;
    std::cout << "todd " << left_total << " " << probe_added << " " << right_total << " " << build_added << std::endl;
    for (int i = 0; i < left_total - probe_added; ++i) {
        ret.push_back(i);
    }
    for (int i = 0; i < right_total - build_added; ++i) {
        ret.push_back(i + left_total);
    }
    return ret;
#endif
}

std::vector<uint64_t> getJoinOutputOrder(const std::vector<uint64_t>& keys,
                                         std::vector<uint64_t>& proj_map) {
    std::vector<uint64_t> ret;
    ret.reserve(proj_map.size());
    std::sort(proj_map.begin(), proj_map.end());
    return proj_map;
    // std::vector<uint64_t> sorted = proj_map;
    // std::sort(sorted.begin(), sorted.end());
    std::unordered_set<uint64_t> proj_set;
    std::unordered_set<uint64_t> key_set;
    for (auto& me : proj_map) {
        proj_set.insert(me);
    }
    for (auto& k : keys) {
        key_set.insert(k);
        if (proj_set.find(k) != proj_set.end()) {
            ret.push_back(k);
        }
    }
    for (auto& me : proj_map) {
        if (key_set.find(me) == key_set.end()) {
            ret.push_back(me);
        }
    }
    return ret;
}

std::unique_ptr<duckdb::LogicalOperator> SplitNonEquiFromComparisonJoin(
    duckdb::LogicalComparisonJoin& comp_join) {
    // Extract non-equi expressions and remove them from comp_join.conditions
    duckdb::vector<duckdb::JoinCondition> non_equi_exprs =
        extractNonEquiConditions(comp_join.conditions);

    // If there are no non-equi conditions, nothing to do
    if (non_equi_exprs.empty()) {
        return nullptr;
    }

    duckdb::unique_ptr<duckdb::Expression> combined_pred =
        duckdb::JoinCondition::CreateExpression(std::move(non_equi_exprs));

    auto probe_proj_map_copy = comp_join.left_projection_map;
    auto build_proj_map_copy = comp_join.right_projection_map;
    comp_join.ResolveOperatorTypes();
    comp_join.children[0]->ResolveOperatorTypes();
    comp_join.children[1]->ResolveOperatorTypes();
    fill_if_empty(probe_proj_map_copy, comp_join.children[0]->types.size());
    fill_if_empty(build_proj_map_copy, comp_join.children[1]->types.size());
    DumpVecIdx(probe_proj_map_copy, "probe_proj_map_copy after fill");
    DumpVecIdx(build_proj_map_copy, "build_proj_map_copy after fill");

    std::unordered_set<duckdb::idx_t> probe_table_bindings,
        build_table_bindings;
    duckdb::LogicalJoin::GetTableReferences(*(comp_join.children[0]),
                                            probe_table_bindings);
    duckdb::LogicalJoin::GetTableReferences(*(comp_join.children[1]),
                                            build_table_bindings);
    if (probe_table_bindings.size() != 1) {
        throw std::runtime_error("probe_table_bindings size is not 1.");
    }
    if (build_table_bindings.size() != 1) {
        throw std::runtime_error("build_table_bindings size is not 1.");
    }
    DumpTableBindings(probe_table_bindings, "probe_table_bindings");
    DumpTableBindings(build_table_bindings, "build_table_bindings");
    MissingBindingsResult mbr = FindMissingBindingsInProjectionMaps(
        combined_pred, probe_proj_map_copy, build_proj_map_copy,
        probe_table_bindings, build_table_bindings);
    DumpColumnBindings(mbr.missing_in_probe, "missing_in_probe");
    DumpColumnBindings(mbr.missing_in_build, "missing_in_build");
    DumpColumnBindings(mbr.missing_everywhere, "missing_everywhere");

    int probe_added, build_added;
    duckdb::vector<duckdb::idx_t> new_probe_proj_map = gen_new_proj_map(
        probe_proj_map_copy, mbr.missing_in_probe, probe_added);
    duckdb::vector<duckdb::idx_t> new_build_proj_map = gen_new_proj_map(
        build_proj_map_copy, mbr.missing_in_build, build_added);
    DumpVecIdx(new_probe_proj_map, "new_probe_proj_map");
    DumpVecIdx(new_build_proj_map, "new_build_proj_map");

    auto new_comp_join =
        duckdb::make_uniq<duckdb::LogicalComparisonJoin>(comp_join.join_type);
    new_comp_join->children.push_back(std::move(comp_join.children[0]));
    new_comp_join->children.push_back(std::move(comp_join.children[1]));
    new_comp_join->conditions = std::move(comp_join.conditions);
    new_comp_join->join_id = comp_join.join_id;
    new_comp_join->left_projection_map = new_probe_proj_map;
    new_comp_join->right_projection_map = new_build_proj_map;

    std::vector<uint64_t> probe_keys;
    std::vector<uint64_t> build_keys;
    duckdb::vector<duckdb::ColumnBinding> probe_bindings =
        new_comp_join->children[0]->GetColumnBindings();
    duckdb::vector<duckdb::ColumnBinding> build_bindings =
        new_comp_join->children[1]->GetColumnBindings();
    std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t>
        probe_col_ref_map = getColRefMap(probe_bindings);
    std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t>
        build_col_ref_map = getColRefMap(build_bindings);
    PhysicalJoin::extractKeys(new_comp_join->conditions, probe_keys, build_keys,
                              probe_col_ref_map, build_col_ref_map);

    DumpVecIdx(probe_keys, "probe_keys");
    DumpVecIdx(build_keys, "build_keys");

    std::vector<uint64_t> build_col_inds;
    std::vector<uint64_t> probe_col_inds;
    build_col_inds = getJoinOutputOrder(build_keys, new_build_proj_map);
    probe_col_inds = getJoinOutputOrder(probe_keys, new_probe_proj_map);
    DumpVecIdx(build_col_inds, "split build_col_inds");
    DumpVecIdx(probe_col_inds, "split probe_col_inds");

    std::vector<uint64_t> probe_proj_map_locations;
    std::vector<uint64_t> build_proj_map_locations;
    probe_proj_map_locations.reserve(new_probe_proj_map.size());
    build_proj_map_locations.reserve(new_build_proj_map.size());
    for (auto& col : new_probe_proj_map) {
        probe_proj_map_locations.push_back(
            index_of(probe_col_inds, static_cast<uint64_t>(col)));
    }
    for (auto& col : new_build_proj_map) {
        build_proj_map_locations.push_back(
            index_of(build_col_inds, static_cast<uint64_t>(col)));
    }
    DumpVecIdx(probe_proj_map_locations, "probe_proj_map_locations");
    DumpVecIdx(build_proj_map_locations, "build_proj_map_locations");

#if 0
    auto binder = get_duckdb_binder();
    auto project_table_idx = binder.get()->GenerateTableIndex();

    std::vector<duckdb::unique_ptr<duckdb::Expression>> projection_expressions;
#if 0
    for (auto &expr : expr_vec) {
        // Convert std::unique_ptr to duckdb::unique_ptr.
        auto expr_duck = to_duckdb(expr);
        projection_expressions.push_back(std::move(expr_duck));
    }
#endif

    // Create projection node.
    duckdb::unique_ptr<duckdb::LogicalProjection> new_project =
        duckdb::make_uniq<duckdb::LogicalProjection>(
            project_table_idx, std::move(projection_expressions));

    // Add the source of the projection.
    new_project->children.push_back(std::move(new_comp_join));
#endif

    auto filter =
        duckdb::make_uniq<duckdb::LogicalFilter>(std::move(combined_pred));
    // Move the join (which now only contains equi conditions) under the filter
    filter->children.push_back(std::move(new_comp_join));
    filter->projection_map =
        gen_split_filter_projection_map(probe_proj_map_copy, probe_col_inds,
                                        build_proj_map_copy, build_col_inds);
    // filter->projection_map = gen_split_filter_projection_map(probe_added,
    // build_added, new_probe_proj_map.size(), new_build_proj_map.size());
    DumpVecIdx(filter->projection_map, "new filter projection_map");

    // Preserve estimated cardinality / properties if needed
    if (filter->children[0]->has_estimated_cardinality) {
        filter->SetEstimatedCardinality(
            filter->children[0]->estimated_cardinality);
    }

    return std::move(filter);
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
        // Move non-equi join conditions into a filter node.
        std::unique_ptr<duckdb::LogicalOperator> split =
            SplitNonEquiFromComparisonJoin(op);
        // If there were non-equi join conditions then create the physical plan
        // based on the filter returned by the above function.
        if (split != nullptr) {
            Visit(*split);
            return;
        }

        physical_join = std::make_shared<PhysicalJoin>(op);
        (*this->join_on_gpu).insert({op.join_id, false});
    }
#else
    // Move non-equi join conditions into a filter node.
    std::unique_ptr<duckdb::LogicalOperator> split =
        SplitNonEquiFromComparisonJoin(op);
    // If there were non-equi join conditions then create the physical plan
    // based on the filter returned by the above function.
    if (split != nullptr) {
        Visit(*split);
        return;
    }

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

    std::shared_ptr<bodo::Schema> in_table_schema =
        this->active_pipeline->getPrevOpOutputSchema();

#ifdef USE_CUDF
    std::variant<std::shared_ptr<PhysicalJoinFilter>,
                 std::shared_ptr<PhysicalGPUJoinFilter>>
        physical_op;
    bool run_on_gpu = node_run_on_gpu(op);
    if (run_on_gpu) {
        physical_op = std::make_shared<PhysicalGPUJoinFilter>(
            op, in_table_schema, this->join_filter_states);
    } else {
        physical_op = std::make_shared<PhysicalJoinFilter>(
            op, in_table_schema, this->join_filter_states);
    }
#else
    std::shared_ptr<PhysicalJoinFilter> physical_op =
        std::make_shared<PhysicalJoinFilter>(op, in_table_schema,
                                             this->join_filter_states);
#endif

    bool found_join_on_same_device = false;

    // Make sure all filter generators used by this
    // join filter run before this pipeline.
    for (int filter_id : op.filter_ids) {
#ifdef USE_CUDF
        if ((*join_on_gpu)[filter_id] != run_on_gpu) {
            continue;
        }
        found_join_on_same_device = true;
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
    // Only insert join filter if both the join and the filter are on
    // the same device type.  We could change this later but it is much
    // more complicated and involves transferring and transforming the
    // bloom filters.
    if (!found_join_on_same_device) {
        return;
    }

#ifdef USE_CUDF
    std::visit([&](auto& vop) { this->active_pipeline->AddOperator(vop); },
               physical_op);
#else
    this->active_pipeline->AddOperator(physical_op);
#endif
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
