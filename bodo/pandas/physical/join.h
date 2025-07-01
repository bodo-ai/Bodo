
#pragma once

#include <cstdint>
#include "../../libs/streaming/_join.h"
#include "../_util.h"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
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
        const duckdb::vector<duckdb::JoinCondition>& conditions) {
        // Initialize column indices in join build/probe that need to be
        // produced according to join output bindings
        duckdb::idx_t left_table_index = -1;
        duckdb::idx_t right_table_index = -1;
        std::vector<duckdb::ColumnBinding> left_findings =
            logical_join.children[0]->GetColumnBindings();
        std::vector<duckdb::ColumnBinding> right_findings =
            logical_join.children[1]->GetColumnBindings();
        if (left_findings.size() > 0) {
            left_table_index = left_findings[0].table_index;
        }
        if (right_findings.size() > 0) {
            right_table_index = right_findings[0].table_index;
        }
        for (auto& c : logical_join.GetColumnBindings()) {
            if (c.table_index == left_table_index) {
                this->bound_left_inds.insert(c.column_index);
            } else if (c.table_index == right_table_index) {
                this->bound_right_inds.insert(c.column_index);
            }
        }

        // Check conditions and add key columns
        for (const duckdb::JoinCondition& cond : conditions) {
            if (cond.comparison != duckdb::ExpressionType::COMPARE_EQUAL) {
                throw std::runtime_error(
                    "Non-equi join condition not supported yet.");
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
            this->left_keys.push_back(
                cond.left->Cast<duckdb::BoundColumnRefExpression>()
                    .binding.column_index);
            this->right_keys.push_back(
                cond.right->Cast<duckdb::BoundColumnRefExpression>()
                    .binding.column_index);
        }
    }

    virtual ~PhysicalJoin() = default;

    /**
     * @brief Initialize the join state using build and probe schemas (called
     * when available).
     *
     * @param build_table_schema schema of the build table
     * @param probe_table_schema schema of the probe table
     */
    void InitializeJoinState(
        const std::shared_ptr<bodo::Schema> build_table_schema,
        const std::shared_ptr<bodo::Schema> probe_table_schema) {
        size_t n_build_cols = build_table_schema->ncols();
        size_t n_probe_cols = probe_table_schema->ncols();

        initInputColumnMapping(build_col_inds, right_keys, n_build_cols);
        initInputColumnMapping(probe_col_inds, left_keys, n_probe_cols);

        initOutputColumnMapping(build_kept_cols, right_keys, n_build_cols,
                                bound_right_inds);
        initOutputColumnMapping(probe_kept_cols, left_keys, n_probe_cols,
                                bound_left_inds);

        std::shared_ptr<bodo::Schema> build_table_schema_reordered =
            build_table_schema->Project(build_col_inds);
        std::shared_ptr<bodo::Schema> probe_table_schema_reordered =
            probe_table_schema->Project(probe_col_inds);

        // TODO[BSE-4813]: handle outer joins properly
        bool build_table_outer = false;
        bool probe_table_outer = false;

        this->join_state = std::make_shared<HashJoinState>(
            build_table_schema_reordered, probe_table_schema_reordered,
            this->left_keys.size(), build_table_outer, probe_table_outer,
            // TODO: support forcing broadcast by the planner
            false, nullptr, true, true, get_streaming_batch_size(), -1,
            // TODO: support query profiling
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
        // See
        // https://github.com/bodo-ai/Bodo/blob/546cb5a45f5bc8e3922f5060e7f778cc744a0930/bodo/libs/streaming/_join.cpp#L4062
        this->join_state->InitOutputBuffer(this->build_kept_cols,
                                           this->probe_kept_cols);

        bool contain_non_equi_cond = join_state->cond_func != nullptr;
        if (contain_non_equi_cond) {
            throw std::runtime_error(
                "Non-equi join condition not supported yet.");
        }
        bool has_bloom_filter = join_state->global_bloom_filter != nullptr;

        bool is_last = prev_op_result == OperatorResult::FINISHED;

        std::shared_ptr<table_info> input_batch_reordered =
            ProjectTable(input_batch, this->probe_col_inds);

        if (has_bloom_filter) {
            is_last = join_probe_consume_batch<false, false, false, true>(
                this->join_state.get(), input_batch_reordered, build_kept_cols,
                probe_kept_cols, is_last);
        } else {
            is_last = join_probe_consume_batch<false, false, false, false>(
                this->join_state.get(), input_batch_reordered, build_kept_cols,
                probe_kept_cols, is_last);
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
};
