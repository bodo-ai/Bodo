
#pragma once

#include <algorithm>
#include <cstdint>
#include "../../libs/streaming/_join.h"
#include "../_util.h"
#include "duckdb/common/enums/expression_type.hpp"
#include "duckdb/planner/column_binding.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/joinside.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_cross_product.hpp"
#include "expression.h"
#include "operator.h"

#ifndef CONSUME_PROBE_BATCH
#define CONSUME_PROBE_BATCH(                                                   \
    build_table_outer, probe_table_outer, has_non_equi_cond, use_bloom_filter, \
    is_anti_join, build_table_outer_exp, probe_table_outer_exp,                \
    has_non_equi_cond_exp, use_bloom_filter_exp, is_anti_join_exp)             \
    if (build_table_outer == build_table_outer_exp &&                          \
        probe_table_outer == probe_table_outer_exp &&                          \
        has_non_equi_cond == has_non_equi_cond_exp &&                          \
        use_bloom_filter == use_bloom_filter_exp &&                            \
        is_anti_join == is_anti_join_exp) {                                    \
        is_last = join_probe_consume_batch<                                    \
            build_table_outer_exp, probe_table_outer_exp,                      \
            has_non_equi_cond_exp, use_bloom_filter_exp, is_anti_join_exp>(    \
            join_state, std::move(input_batch_reordered), build_kept_cols,     \
            probe_kept_cols, is_last);                                         \
    }
#endif

struct PhysicalJoinMetrics {
    using time_t = MetricBase::TimerValue;
    using stat_t = MetricBase::StatValue;

    time_t init_time = 0;
    time_t consume_time = 0;
    time_t process_batch_time = 0;

    stat_t output_row_count = 0;
};
/**
 * @brief Helper function to set whether column ref expressions use the left or
 * right table in the join based on the column bindings of the column ref and
 * the left table column bindings.
 */
void setExprTreeLeftRight(
    std::shared_ptr<PhysicalExpression> expr,
    const std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t>
        left_col_ref_map);

/**
 * @brief Physical node for join.
 *
 */
class PhysicalJoin : public PhysicalProcessBatch, public PhysicalSink {
   public:
    explicit PhysicalJoin(duckdb::LogicalComparisonJoin& logical_join)
        : has_non_equi_cond(false),
          is_mark_join(logical_join.join_type == duckdb::JoinType::MARK),
          is_anti_join(logical_join.join_type == duckdb::JoinType::ANTI ||
                       logical_join.join_type == duckdb::JoinType::RIGHT_ANTI) {
    }

    void buildProbeSchemas(
        duckdb::LogicalComparisonJoin& logical_join,
        duckdb::vector<duckdb::JoinCondition>& conditions,
        const std::shared_ptr<bodo::Schema> build_table_schema,
        const std::shared_ptr<bodo::Schema> probe_table_schema) {
        time_pt start_init = start_timer();
        // Probe side
        duckdb::vector<duckdb::ColumnBinding> left_bindings =
            logical_join.children[0]->GetColumnBindings();
        // Build side
        duckdb::vector<duckdb::ColumnBinding> right_bindings =
            logical_join.children[1]->GetColumnBindings();
        duckdb::vector<duckdb::ColumnBinding> join_bindings =
            logical_join.GetColumnBindings();

        std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t>
            left_col_ref_map = getColRefMap(left_bindings);
        std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t>
            right_col_ref_map = getColRefMap(right_bindings);
        std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t>
            join_col_ref_map = getColRefMap(join_bindings);

        bool is_left_anti = logical_join.join_type == duckdb::JoinType::ANTI;
        bool is_right_anti =
            logical_join.join_type == duckdb::JoinType::RIGHT_ANTI;

        // Find left/right table columns that will be in the join output.
        // Similar to DuckDB:
        // https://github.com/duckdb/duckdb/blob/d29a92f371179170688b4df394478f389bf7d1a6/src/execution/operator/join/physical_hash_join.cpp#L58
        if (!is_right_anti) {
            if (logical_join.left_projection_map.empty()) {
                for (duckdb::idx_t i = 0;
                     i < logical_join.children[0]->GetColumnBindings().size();
                     i++) {
                    this->bound_left_inds.insert(i);
                }
            } else {
                for (const auto& c : logical_join.left_projection_map) {
                    this->bound_left_inds.insert(c);
                }
            }
        }

        // Mark join does not output the build table columns
        if (!this->is_mark_join && !is_left_anti) {
            if (logical_join.right_projection_map.empty()) {
                for (duckdb::idx_t i = 0;
                     i < logical_join.children[1]->GetColumnBindings().size();
                     i++) {
                    this->bound_right_inds.insert(i);
                }
            } else {
                for (const auto& c : logical_join.right_projection_map) {
                    this->bound_right_inds.insert(c);
                }
            }
        }

        // Check conditions and add key columns
        for (const duckdb::JoinCondition& cond : conditions) {
            if (cond.IsComparison() &&
                cond.GetComparisonType() ==
                    bododuckdb::ExpressionType::COMPARE_EQUAL) {
                if (cond.GetLHS().GetExpressionClass() !=
                    duckdb::ExpressionClass::BOUND_COLUMN_REF) {
                    throw std::runtime_error(
                        "Join condition left side is not a column reference.");
                }
                if (cond.GetRHS().GetExpressionClass() !=
                    duckdb::ExpressionClass::BOUND_COLUMN_REF) {
                    throw std::runtime_error(
                        "Join condition right side is not a column reference.");
                }
                auto& left_bce =
                    cond.GetLHS().Cast<duckdb::BoundColumnRefExpression>();
                auto& right_bce =
                    cond.GetRHS().Cast<duckdb::BoundColumnRefExpression>();
                this->left_keys.push_back(
                    left_col_ref_map[{left_bce.binding.table_index,
                                      left_bce.binding.column_index}]);
                this->right_keys.push_back(
                    right_col_ref_map[{right_bce.binding.table_index,
                                       right_bce.binding.column_index}]);
            } else {
                has_non_equi_cond = true;
            }
        }

        size_t n_build_cols = build_table_schema->ncols();
        size_t n_probe_cols = probe_table_schema->ncols();

        initInputColumnMapping(build_col_inds, right_keys, n_build_cols);
        initInputColumnMapping(probe_col_inds, left_keys, n_probe_cols);
        build_col_inds_rev = std::vector<int64_t>(build_col_inds.size());
        for (size_t i = 0; i < build_col_inds.size(); ++i) {
            build_col_inds_rev[build_col_inds[i]] = i;
        }
        probe_col_inds_rev = std::vector<int64_t>(probe_col_inds.size());
        for (size_t i = 0; i < probe_col_inds.size(); ++i) {
            probe_col_inds_rev[probe_col_inds[i]] = i;
        }

        std::shared_ptr<bodo::Schema> build_table_schema_reordered =
            build_table_schema->Project(build_col_inds);
        std::shared_ptr<bodo::Schema> probe_table_schema_reordered =
            probe_table_schema->Project(probe_col_inds);

        for (duckdb::JoinCondition& cond : conditions) {
            if (cond.IsComparison() &&
                cond.GetComparisonType() ==
                    duckdb::ExpressionType::COMPARE_EQUAL) {
                // These cases are handled by the left_keys and right_keys
                // above.  Only the non-equi tests are handled here.
                continue;
            }
            std::shared_ptr<PhysicalExpression> new_phys_expr;
            duckdb::unique_ptr<duckdb::Expression> expr =
                bododuckdb::JoinCondition::CreateExpression(std::move(cond));
            // Create a col ref map for both the tables, mapping to their
            // respective positions in the reordered schemas
            std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t>
                combined_left_right_expr_col_ref_map;
            for (const auto& [k, v] : left_col_ref_map) {
                combined_left_right_expr_col_ref_map[k] = probe_col_inds_rev[v];
            }
            for (const auto& [k, v] : right_col_ref_map) {
                combined_left_right_expr_col_ref_map[k] = build_col_inds_rev[v];
            }
            new_phys_expr = buildPhysicalExprTree(
                expr, combined_left_right_expr_col_ref_map, false);
            // If we have more than one non-equi join condition then 'and'
            // them together.
            if (physExprTree) {
                physExprTree = std::static_pointer_cast<PhysicalExpression>(
                    std::make_shared<PhysicalConjunctionExpression>(
                        physExprTree, new_phys_expr,
                        duckdb::ExpressionType::CONJUNCTION_AND));
            } else {
                physExprTree = new_phys_expr;
            }
        }
        setExprTreeLeftRight(physExprTree, left_col_ref_map);

        initOutputColumnMapping(build_kept_cols, right_keys, n_build_cols,
                                bound_right_inds, build_col_inds_rev);
        initOutputColumnMapping(probe_kept_cols, left_keys, n_probe_cols,
                                bound_left_inds, probe_col_inds_rev);

        // ---------------------------------------------------

        // Implement anti-join by setting an outer flag and adding an anti-join
        // flag to avoid producing matching rows. Anti join is used for
        // ~Series.isin in dataframe library

        bool build_table_outer =
            (logical_join.join_type == duckdb::JoinType::RIGHT) ||
            (logical_join.join_type == duckdb::JoinType::OUTER) ||
            is_right_anti;
        bool probe_table_outer =
            (logical_join.join_type == duckdb::JoinType::LEFT) ||
            (logical_join.join_type == duckdb::JoinType::OUTER) || is_left_anti;

        cond_expr_fn_t join_func = nullptr;
        size_t n_equality_keys = left_keys.size();
        if (n_equality_keys == 0) {
            if (has_non_equi_cond) {
                // NestedLoopJoinState's constructor requires a cond_expr_fn_t
                // even though it uses it as a cond_expr_fn_batch_t.
                join_func = (cond_expr_fn_t)PhysicalExpression::join_expr_batch;
            }
            // No equality keys, so we do a nested loop join.
            this->join_state_ = std::make_shared<NestedLoopJoinState>(
                build_table_schema_reordered, probe_table_schema_reordered,
                build_table_outer, probe_table_outer, std::vector<int64_t>(),
                false, join_func, true, true, get_streaming_batch_size(), -1,
                getOpId());
        } else {
            if (has_non_equi_cond) {
                join_func = PhysicalExpression::join_expr;
            }
            this->join_state_ = std::make_shared<HashJoinState>(
                build_table_schema_reordered, probe_table_schema_reordered,
                this->left_keys.size(), build_table_outer, probe_table_outer,
                // TODO: support forcing broadcast by the planner
                false, join_func, true, true, get_streaming_batch_size(), -1,
                //  TODO: support query profiling
                getOpId(), -1, JOIN_MAX_PARTITION_DEPTH,
                /*is_na_equal*/ true, is_mark_join);
        }

        this->initOutputSchema(build_table_schema_reordered,
                               probe_table_schema_reordered,
                               logical_join.GetColumnBindings().size(),
                               build_table_outer, probe_table_outer);
        this->metrics.init_time += end_timer(start_init);
    }

    /**
     * @brief Initialize the output schema for the join based on input schema
     * and kept columns in output.
     */
    void initOutputSchema(
        const std::shared_ptr<bodo::Schema>& build_table_schema_reordered,
        const std::shared_ptr<bodo::Schema>& probe_table_schema_reordered,
        size_t n_op_out_cols, bool build_table_outer, bool probe_table_outer) {
        // Create the probe output schema, same as here for consistency:
        // https://github.com/bodo-ai/Bodo/blob/a2e8bb7ba455dcba7372e6e92bd8488ed2b2d5cc/bodo/libs/streaming/_join.cpp#L1138
        this->output_schema = std::make_shared<bodo::Schema>();
        std::vector<std::string> col_names;
        if (probe_table_schema_reordered->column_names.empty() ||
            build_table_schema_reordered->column_names.empty()) {
            throw std::runtime_error(
                "Join input tables must have column names.");
        }

        for (uint64_t i_col : probe_kept_cols) {
            std::unique_ptr<bodo::DataType> col_type =
                probe_table_schema_reordered->column_types[i_col]->copy();
            // In the build outer case, we need to make NUMPY arrays
            // into NULLABLE arrays. Matches the `use_nullable_arrs`
            // behavior of RetrieveTable.
            if (build_table_outer) {
                col_type = col_type->to_nullable_type();
            }
            output_schema->append_column(std::move(col_type));
            col_names.push_back(
                probe_table_schema_reordered->column_names[i_col]);
        }

        // Add the mark output column if this is a mark join.
        if (this->is_mark_join) {
            if (!build_kept_cols.empty()) {
                throw std::runtime_error(
                    "Mark join should not output build table columns.");
            }
            output_schema->append_column(std::make_unique<bodo::DataType>(
                bodo_array_type::NULLABLE_INT_BOOL, Bodo_CTypes::_BOOL));
            col_names.push_back("");
        }

        for (uint64_t i_col : build_kept_cols) {
            std::unique_ptr<bodo::DataType> col_type =
                build_table_schema_reordered->column_types[i_col]->copy();
            // In the probe outer case, we need to make NUMPY arrays
            // into NULLABLE arrays. Matches the `use_nullable_arrs`
            // behavior of RetrieveTable.
            if (probe_table_outer) {
                col_type = col_type->to_nullable_type();
            }
            output_schema->append_column(std::move(col_type));
            col_names.push_back(
                build_table_schema_reordered->column_names[i_col]);
        }
        this->output_schema->column_names = col_names;
        // Indexes are ignored in the Pandas merge if not joining on Indexes.
        // We designate empty metadata to indicate generating a trivial
        // RangeIndex.
        // TODO[BSE-4820]: support joining on Indexes
        this->output_schema->metadata = std::make_shared<bodo::TableMetadata>(
            std::vector<std::string>({}), std::vector<std::string>({}));
        if (this->output_schema->column_names.size() != n_op_out_cols) {
            throw std::runtime_error(
                "Join output schema has different number of columns than "
                "LogicalComparisonJoin");
        }

        // See
        // https://github.com/bodo-ai/Bodo/blob/546cb5a45f5bc8e3922f5060e7f778cc744a0930/bodo/libs/streaming/_join.cpp#L4062
        this->join_state_->InitOutputBuffer(this->build_kept_cols,
                                            this->probe_kept_cols);
    }

    /**
     * @brief Physical Join constructor for cross join.
     *
     */
    PhysicalJoin(duckdb::LogicalCrossProduct& logical_join,
                 const std::shared_ptr<bodo::Schema> build_table_schema,
                 const std::shared_ptr<bodo::Schema> probe_table_schema)
        : has_non_equi_cond(false) {
        time_pt start_init = start_timer();
        this->join_state_ = std::make_shared<NestedLoopJoinState>(
            build_table_schema, probe_table_schema, false, false,
            std::vector<int64_t>(),
            // TODO: support forcing broadcast by the planner
            false, nullptr, true, true, get_streaming_batch_size(), -1, -1);

        // Cross join doesn't have any keys, so we keep all columns.
        for (uint64_t i = 0; i < probe_table_schema->ncols(); i++) {
            this->probe_kept_cols.push_back(i);
        }
        for (uint64_t i = 0; i < build_table_schema->ncols(); i++) {
            this->build_kept_cols.push_back(i);
        }
        this->initOutputSchema(build_table_schema, probe_table_schema,
                               logical_join.GetColumnBindings().size(), false,
                               false);
        this->metrics.init_time += end_timer(start_init);
    }

    virtual ~PhysicalJoin() = default;

    void FinalizeSink() override {}

    void FinalizeProcessBatch() override {
        QueryProfileCollector::Default().SubmitOperatorName(getOpId(),
                                                            ToString());
        QueryProfileCollector::Default().SubmitOperatorStageTime(
            QueryProfileCollector::MakeOperatorStageID(getOpId(), 0),
            metrics.init_time);
        QueryProfileCollector::Default().SubmitOperatorStageTime(
            QueryProfileCollector::MakeOperatorStageID(getOpId(), 1),
            metrics.consume_time);
        QueryProfileCollector::Default().SubmitOperatorStageTime(
            QueryProfileCollector::MakeOperatorStageID(getOpId(), 2),
            metrics.process_batch_time);
        QueryProfileCollector::Default().SubmitOperatorStageRowCounts(
            QueryProfileCollector::MakeOperatorStageID(getOpId(), 2),
            this->metrics.output_row_count);
    }

    /**
     * @brief process input tables to build side of join (populate the hash
     * table)
     *
     * @return OperatorResult
     */
    OperatorResult ConsumeBatch(std::shared_ptr<table_info> input_batch,
                                OperatorResult prev_op_result) override {
        time_pt start_consume = start_timer();
        bool local_is_last = prev_op_result == OperatorResult::FINISHED;

        if (join_state_->IsNestedLoopJoin()) {
            bool global_is_last = nested_loop_join_build_consume_batch(
                (NestedLoopJoinState*)join_state_.get(), input_batch,
                local_is_last);
            return global_is_last ? OperatorResult::FINISHED
                                  : OperatorResult::NEED_MORE_INPUT;
        }

        HashJoinState* join_state =
            static_cast<HashJoinState*>(this->join_state_.get());

        // See
        // https://github.com/bodo-ai/Bodo/blob/967b62f1c943a3e8f8e00d5f9cdcb2865fb55cb0/bodo/libs/streaming/_join.cpp#L4018
        bool has_bloom_filter = join_state->global_bloom_filter != nullptr;

        std::shared_ptr<table_info> input_batch_reordered =
            ProjectTable(input_batch, this->build_col_inds);

        bool global_is_last = join_build_consume_batch(
            join_state, input_batch_reordered, has_bloom_filter, local_is_last);

        if (global_is_last) {
            return OperatorResult::FINISHED;
        }
        this->metrics.consume_time += end_timer(start_consume);
        return join_state->build_shuffle_state.BuffersFull()
                   ? OperatorResult::HAVE_MORE_OUTPUT
                   : OperatorResult::NEED_MORE_INPUT;
    }

    /**
     * @brief Run join probe on the input batch
     *
     * @param input_batch input batch to probe
     * @return output batch of probe and return flag
     */
    std::pair<std::shared_ptr<table_info>, OperatorResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch,
        OperatorResult prev_op_result) override {
        time_pt start_produce = start_timer();
        bool is_last = prev_op_result == OperatorResult::FINISHED;

        if (has_non_equi_cond) {
            PhysicalExpression::cur_join_expr = physExprTree.get();
        }

        bool request_input = true;

        if (join_state_->IsNestedLoopJoin()) {
            is_last = nested_loop_join_probe_consume_batch(
                (NestedLoopJoinState*)join_state_.get(), input_batch,
                build_kept_cols, probe_kept_cols, is_last);
        } else {
            HashJoinState* join_state =
                static_cast<HashJoinState*>(this->join_state_.get());

            bool has_bloom_filter = join_state->global_bloom_filter != nullptr;

            std::shared_ptr<table_info> input_batch_reordered =
                ProjectTable(input_batch, this->probe_col_inds);

            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                has_non_equi_cond, has_bloom_filter,
                                is_anti_join, true, true, true, true, false)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                has_non_equi_cond, has_bloom_filter,
                                is_anti_join, true, true, true, false, false)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                has_non_equi_cond, has_bloom_filter,
                                is_anti_join, true, true, false, true, false)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                has_non_equi_cond, has_bloom_filter,
                                is_anti_join, true, true, false, false, false)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                has_non_equi_cond, has_bloom_filter,
                                is_anti_join, true, false, true, true, false)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                has_non_equi_cond, has_bloom_filter,
                                is_anti_join, true, false, true, false, false)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                has_non_equi_cond, has_bloom_filter,
                                is_anti_join, true, false, false, true, false)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                has_non_equi_cond, has_bloom_filter,
                                is_anti_join, true, false, false, false, false)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                has_non_equi_cond, has_bloom_filter,
                                is_anti_join, false, true, true, true, false)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                has_non_equi_cond, has_bloom_filter,
                                is_anti_join, false, true, true, false, false)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                has_non_equi_cond, has_bloom_filter,
                                is_anti_join, false, true, false, true, false)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                has_non_equi_cond, has_bloom_filter,
                                is_anti_join, false, true, false, false, false)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                has_non_equi_cond, has_bloom_filter,
                                is_anti_join, false, false, true, true, false)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                has_non_equi_cond, has_bloom_filter,
                                is_anti_join, false, false, true, false, false)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                has_non_equi_cond, has_bloom_filter,
                                is_anti_join, false, false, false, true, false)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                has_non_equi_cond, has_bloom_filter,
                                is_anti_join, false, false, false, false, false)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                has_non_equi_cond, has_bloom_filter,
                                is_anti_join, true, true, true, true, true)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                has_non_equi_cond, has_bloom_filter,
                                is_anti_join, true, true, true, false, true)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                has_non_equi_cond, has_bloom_filter,
                                is_anti_join, true, true, false, true, true)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                has_non_equi_cond, has_bloom_filter,
                                is_anti_join, true, true, false, false, true)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                has_non_equi_cond, has_bloom_filter,
                                is_anti_join, true, false, true, true, true)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                has_non_equi_cond, has_bloom_filter,
                                is_anti_join, true, false, true, false, true)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                has_non_equi_cond, has_bloom_filter,
                                is_anti_join, true, false, false, true, true)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                has_non_equi_cond, has_bloom_filter,
                                is_anti_join, true, false, false, false, true)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                has_non_equi_cond, has_bloom_filter,
                                is_anti_join, false, true, true, true, true)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                has_non_equi_cond, has_bloom_filter,
                                is_anti_join, false, true, true, false, true)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                has_non_equi_cond, has_bloom_filter,
                                is_anti_join, false, true, false, true, true)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                has_non_equi_cond, has_bloom_filter,
                                is_anti_join, false, true, false, false, true)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                has_non_equi_cond, has_bloom_filter,
                                is_anti_join, false, false, true, true, true)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                has_non_equi_cond, has_bloom_filter,
                                is_anti_join, false, false, true, false, true)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                has_non_equi_cond, has_bloom_filter,
                                is_anti_join, false, false, false, true, true)
            CONSUME_PROBE_BATCH(join_state->build_table_outer,
                                join_state->probe_table_outer,
                                has_non_equi_cond, has_bloom_filter,
                                is_anti_join, false, false, false, false, true)

            if (join_state->probe_shuffle_state.BuffersFull()) {
                request_input = false;
            }
        }

        // If after emitting the next batch we'll have more than a full
        // batch left then we don't need to request input. This is to avoid
        // allocating more memory than necessary and increasing cache
        // coherence
        if (join_state_->output_buffer->total_remaining >
            (2 * join_state_->output_buffer->active_chunk_capacity)) {
            request_input = false;
        }

        auto [out_table, chunk_size] = join_state_->output_buffer->PopChunk(
            /*force_return*/ is_last);

        is_last = is_last && join_state_->output_buffer->total_remaining == 0;
        this->metrics.output_row_count += out_table->nrows();
        this->metrics.process_batch_time += end_timer(start_produce);

        out_table->column_names = this->output_schema->column_names;
        return {out_table,
                is_last ? OperatorResult::FINISHED
                        : (request_input ? OperatorResult::NEED_MORE_INPUT
                                         : OperatorResult::HAVE_MORE_OUTPUT)};
    }

    /**
     * @brief GetResult - just for API compatability but should never be called
     */
    std::variant<std::shared_ptr<table_info>, PyObject*> GetResult() override {
        // Join build doesn't return output results
        throw std::runtime_error("GetResult called on a join node.");
    }

    /**
     * @brief Get the output schema of join probe
     *
     * @return std::shared_ptr<bodo::Schema>
     */
    const std::shared_ptr<bodo::Schema> getOutputSchema() override {
        return output_schema;
    }

    std::string ToString() override { return PhysicalSink::ToString(); }

    int64_t getOpId() const { return PhysicalSink::getOpId(); }

    /**
     * @brief Get pointer to JoinState used in join filters
     */
    JoinState* getJoinStatePtr() const { return join_state_.get(); }

   private:
    /**
     * @brief  Initialize mapping of output column orders to reorder keys that
     * were moved to the beginning of of build/probe tables to match streaming
     * join APIs. See
     * https://github.com/bodo-ai/Bodo/blob/905664de2c37741d804615cdbb3fb437621ff0bd/bodo/libs/streaming/join.py#L746
     * @param col_inds output mapping to fill
     * @param keys key column indices
     * @param ncols number of columns in the table
     * @param bound_inds set of column indices that need to be produced in the
     * output according to bindings
     */
    static void initOutputColumnMapping(
        std::vector<uint64_t>& col_inds, const std::vector<uint64_t>& keys,
        uint64_t ncols, const std::set<int64_t>& bound_inds,
        const std::vector<int64_t>& col_inds_rev) {
        // Map key column index to its position in keys vector
        std::unordered_map<uint64_t, size_t> key_positions;
        for (size_t i = 0; i < keys.size(); ++i) {
            key_positions[keys[i]] = i;
        }

        for (uint64_t i = 0; i < ncols; i++) {
            if (bound_inds.find(i) == bound_inds.end()) {
                continue;
            }
            col_inds.push_back(col_inds_rev[i]);
        }
    }

    // build/probe table column indices that need to be produced in output
    std::set<int64_t> bound_left_inds;
    std::set<int64_t> bound_right_inds;

    std::shared_ptr<JoinState> join_state_;

    std::vector<uint64_t> build_kept_cols;
    std::vector<uint64_t> probe_kept_cols;
    std::vector<uint64_t> left_keys;
    std::vector<uint64_t> right_keys;
    std::shared_ptr<bodo::Schema> output_schema;

    std::vector<int64_t> build_col_inds;
    std::vector<int64_t> probe_col_inds;
    std::vector<int64_t> build_col_inds_rev;
    std::vector<int64_t> probe_col_inds_rev;

    bool has_non_equi_cond;
    std::shared_ptr<PhysicalExpression> physExprTree;

    bool is_mark_join = false;
    bool is_anti_join = false;

    PhysicalJoinMetrics metrics;
};

void setExprTreeLeftRight(
    std::shared_ptr<PhysicalExpression> expr,
    const std::map<std::pair<duckdb::idx_t, duckdb::idx_t>, size_t>
        left_col_ref_map) {
    if (!expr) {
        return;
    }
    if (expr->GetExpressionType() == PhysicalExpressionType::COLUMN_REF) {
        auto col_ref_expr =
            std::static_pointer_cast<PhysicalColumnRefExpression>(expr);
        duckdb::ColumnBinding col_binding = col_ref_expr->get_col_binding();
        if (left_col_ref_map.contains(
                {col_binding.table_index, col_binding.column_index})) {
            col_ref_expr->set_left_side(true);
        } else {
            col_ref_expr->set_left_side(false);
        }
    }
    for (auto& child : expr->GetChildren()) {
        setExprTreeLeftRight(child, left_col_ref_map);
    }
}

#undef CONSUME_PROBE_BATCH
