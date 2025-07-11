
#pragma once

#include <cstdint>
#include "../../libs/streaming/_join.h"
#include "../_util.h"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/joinside.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "operator.h"

/**
 * @brief Physical node for join.
 *
 */
class PhysicalJoin : public PhysicalSourceSink, public PhysicalSink {
   public:
    explicit PhysicalJoin(
        duckdb::LogicalComparisonJoin& logical_join,
        duckdb::vector<duckdb::JoinCondition>& conditions,
        const std::shared_ptr<bodo::Schema> build_table_schema,
        const std::shared_ptr<bodo::Schema> probe_table_schema)
        : has_non_equi_cond(false) {
        duckdb::vector<duckdb::ColumnBinding> left_bindings =
            logical_join.children[0]->GetColumnBindings();
        duckdb::vector<duckdb::ColumnBinding> right_bindings =
            logical_join.children[1]->GetColumnBindings();

        // Find left/right table columns that will be in the join output.
        // Similar to DuckDB:
        // https://github.com/duckdb/duckdb/blob/d29a92f371179170688b4df394478f389bf7d1a6/src/execution/operator/join/physical_hash_join.cpp#L58
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

        // Check conditions and add key columns
        for (const duckdb::JoinCondition& cond : conditions) {
            if (cond.comparison != duckdb::ExpressionType::COMPARE_EQUAL) {
                has_non_equi_cond = true;
            }
            if (cond.left->GetExpressionClass() !=
                duckdb::ExpressionClass::BOUND_COLUMN_REF) {
                throw std::runtime_error(
                    "Join condition left side is not a column reference.");
            }
            if (cond.right->GetExpressionClass() !=
                duckdb::ExpressionClass::BOUND_COLUMN_REF) {
                throw std::runtime_error(
                    "Join condition right side is not a column reference.");
            }
            if (cond.comparison == duckdb::ExpressionType::COMPARE_EQUAL) {
                auto& left_bce =
                    cond.left->Cast<duckdb::BoundColumnRefExpression>();
                auto& right_bce =
                    cond.right->Cast<duckdb::BoundColumnRefExpression>();
                this->left_keys.push_back(left_bce.binding.column_index);
                this->right_keys.push_back(right_bce.binding.column_index);
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

        initOutputColumnMapping(build_kept_cols, right_keys, n_build_cols,
                                bound_right_inds);
        initOutputColumnMapping(probe_kept_cols, left_keys, n_probe_cols,
                                bound_left_inds);

        std::shared_ptr<bodo::Schema> build_table_schema_reordered =
            build_table_schema->Project(build_col_inds);
        std::shared_ptr<bodo::Schema> probe_table_schema_reordered =
            probe_table_schema->Project(probe_col_inds);

        for (const duckdb::JoinCondition& cond : conditions) {
            if (cond.comparison == duckdb::ExpressionType::COMPARE_EQUAL) {
                // These cases are handled by the left_keys and right_keys
                // above.  Only the non-equi tests are handled here.
                continue;
            }
            auto& left_bce =
                cond.left->Cast<duckdb::BoundColumnRefExpression>();
            auto& right_bce =
                cond.right->Cast<duckdb::BoundColumnRefExpression>();

            std::shared_ptr<PhysicalExpression> new_phys_expr =
                std::static_pointer_cast<
                    PhysicalExpression>(std::make_shared<
                                        PhysicalComparisonExpression>(
                    std::make_shared<PhysicalColumnRefExpression>(
                        this->probe_col_inds_rev[getColRefMap(
                            left_bindings)[{left_bce.binding.table_index,
                                            left_bce.binding.column_index}]],
                        left_bce.GetName(), true),
                    std::make_shared<PhysicalColumnRefExpression>(
                        this->build_col_inds_rev[getColRefMap(
                            right_bindings)[{right_bce.binding.table_index,
                                             right_bce.binding.column_index}]],
                        right_bce.GetName(), false),
                    cond.comparison));
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

        // ---------------------------------------------------

        // TODO[BSE-4813]: handle outer joins properly
        bool build_table_outer = false;
        bool probe_table_outer = false;

        cond_expr_fn_t join_func = nullptr;
        if (has_non_equi_cond) {
            join_func = PhysicalExpression::join_expr;
        }
        this->join_state = std::make_shared<HashJoinState>(
            build_table_schema_reordered, probe_table_schema_reordered,
            this->left_keys.size(), build_table_outer, probe_table_outer,
            // TODO: support forcing broadcast by the planner
            false, join_func, true, true, get_streaming_batch_size(), -1,
            //  TODO: support query profiling
            -1, -1, JOIN_MAX_PARTITION_DEPTH, /*is_na_equal*/ true);

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
        this->output_schema->metadata = std::make_shared<TableMetadata>(
            std::vector<std::string>({}), std::vector<std::string>({}));
    }

    virtual ~PhysicalJoin() = default;

    void Finalize() override {}

    /**
     * @brief process input tables to build side of join (populate the hash
     * table)
     *
     * @return OperatorResult
     */
    OperatorResult ConsumeBatch(std::shared_ptr<table_info> input_batch,
                                OperatorResult prev_op_result) override {
        bool local_is_last = prev_op_result == OperatorResult::FINISHED;

        // See
        // https://github.com/bodo-ai/Bodo/blob/967b62f1c943a3e8f8e00d5f9cdcb2865fb55cb0/bodo/libs/streaming/_join.cpp#L4018
        bool has_bloom_filter = join_state->global_bloom_filter != nullptr;

        std::shared_ptr<table_info> input_batch_reordered =
            ProjectTable(input_batch, this->build_col_inds);

        bool global_is_last = join_build_consume_batch(
            this->join_state.get(), input_batch_reordered, has_bloom_filter,
            local_is_last);

        if (global_is_last) {
            return OperatorResult::FINISHED;
        }
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
        if (has_non_equi_cond) {
            PhysicalExpression::cur_join_expr = physExprTree.get();
        }
        // See
        // https://github.com/bodo-ai/Bodo/blob/546cb5a45f5bc8e3922f5060e7f778cc744a0930/bodo/libs/streaming/_join.cpp#L4062
        this->join_state->InitOutputBuffer(this->build_kept_cols,
                                           this->probe_kept_cols);

        bool has_bloom_filter = join_state->global_bloom_filter != nullptr;

        bool is_last = prev_op_result == OperatorResult::FINISHED;

        std::shared_ptr<table_info> input_batch_reordered =
            ProjectTable(input_batch, this->probe_col_inds);

        if (has_bloom_filter) {
            if (has_non_equi_cond) {
                is_last = join_probe_consume_batch<false, false, true, true>(
                    this->join_state.get(), input_batch_reordered,
                    build_kept_cols, probe_kept_cols, is_last);
            } else {
                is_last = join_probe_consume_batch<false, false, false, true>(
                    this->join_state.get(), input_batch_reordered,
                    build_kept_cols, probe_kept_cols, is_last);
            }
        } else {
            if (has_non_equi_cond) {
                is_last = join_probe_consume_batch<false, false, true, false>(
                    this->join_state.get(), input_batch_reordered,
                    build_kept_cols, probe_kept_cols, is_last);
            } else {
                is_last = join_probe_consume_batch<false, false, false, false>(
                    this->join_state.get(), input_batch_reordered,
                    build_kept_cols, probe_kept_cols, is_last);
            }
        }

        bool request_input = true;
        if (join_state->probe_shuffle_state.BuffersFull()) {
            request_input = false;
        }
        // If after emitting the next batch we'll have more than a full
        // batch left then we don't need to request input. This is to avoid
        // allocating more memory than necessary and increasing cache
        // coherence
        if (join_state->output_buffer->total_remaining >
            (2 * join_state->output_buffer->active_chunk_capacity)) {
            request_input = false;
        }

        auto [out_table, chunk_size] = join_state->output_buffer->PopChunk(
            /*force_return*/ is_last);

        is_last = is_last && join_state->output_buffer->total_remaining == 0;

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
    static void initOutputColumnMapping(std::vector<uint64_t>& col_inds,
                                        std::vector<uint64_t>& keys,
                                        uint64_t ncols,
                                        std::set<int64_t>& bound_inds) {
        // Map key column index to its position in keys vector
        std::unordered_map<uint64_t, size_t> key_positions;
        for (size_t i = 0; i < keys.size(); ++i) {
            key_positions[keys[i]] = i;
        }
        uint64_t data_offset = keys.size();

        for (uint64_t i = 0; i < ncols; i++) {
            if (bound_inds.find(i) == bound_inds.end()) {
                continue;
            }
            if (key_positions.find(i) != key_positions.end()) {
                col_inds.push_back(key_positions[i]);
            } else {
                col_inds.push_back(data_offset++);
            }
        }
    }

    // build/probe table column indices that need to be produced in output
    std::set<int64_t> bound_left_inds;
    std::set<int64_t> bound_right_inds;

    std::shared_ptr<HashJoinState> join_state;
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
};
