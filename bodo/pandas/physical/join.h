
#pragma once

#include <cstdint>
#include "../../libs/streaming/_join.h"
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
        const duckdb::vector<duckdb::JoinCondition>& conditions) {
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
        // TODO[BSE-4813]: handle outer joins properly
        bool build_table_outer = false;
        bool probe_table_outer = false;

        this->join_state = std::make_shared<HashJoinState>(
            build_table_schema, probe_table_schema,
            // TODO[BSE-4812]: support keys that are not in the beginning of the
            // input tables
            this->left_keys.size(), build_table_outer, probe_table_outer,
            // TODO: support forcing broadcast by the planner
            false, nullptr, true, true, get_streaming_batch_size(), -1,
            // TODO: support query profiling
            -1);

        this->build_kept_cols.resize(build_table_schema->ncols());
        std::iota(this->build_kept_cols.begin(), this->build_kept_cols.end(),
                  0);
        this->probe_kept_cols.resize(probe_table_schema->ncols());
        std::iota(this->probe_kept_cols.begin(), this->probe_kept_cols.end(),
                  0);

        // Create the probe output schema, same as here for consistency:
        // https://github.com/bodo-ai/Bodo/blob/a2e8bb7ba455dcba7372e6e92bd8488ed2b2d5cc/bodo/libs/streaming/_join.cpp#L1138
        this->output_schema = std::make_shared<bodo::Schema>();

        for (uint64_t i_col : probe_kept_cols) {
            std::unique_ptr<bodo::DataType> col_type =
                probe_table_schema->column_types[i_col]->copy();
            // In the build outer case, we need to make NUMPY arrays
            // into NULLABLE arrays. Matches the `use_nullable_arrs`
            // behavior of RetrieveTable.
            if (build_table_outer) {
                col_type = col_type->to_nullable_type();
            }
            output_schema->append_column(std::move(col_type));
        }

        for (uint64_t i_col : build_kept_cols) {
            std::unique_ptr<bodo::DataType> col_type =
                build_table_schema->column_types[i_col]->copy();
            // In the probe outer case, we need to make NUMPY arrays
            // into NULLABLE arrays. Matches the `use_nullable_arrs`
            // behavior of RetrieveTable.
            if (probe_table_outer) {
                col_type = col_type->to_nullable_type();
            }
            output_schema->append_column(std::move(col_type));
        }
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

        bool global_is_last =
            join_build_consume_batch(this->join_state.get(), input_batch,
                                     has_bloom_filter, local_is_last);

        if (global_is_last) {
            return OperatorResult::FINISHED;
        }
        return !join_state->build_shuffle_state.BuffersFull()
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

        if (has_bloom_filter) {
            is_last = join_probe_consume_batch<false, false, false, true>(
                this->join_state.get(), input_batch, build_kept_cols,
                probe_kept_cols, is_last);
        } else {
            is_last = join_probe_consume_batch<false, false, false, false>(
                this->join_state.get(), input_batch, build_kept_cols,
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
    std::shared_ptr<table_info> GetResult() override {
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
    std::shared_ptr<HashJoinState> join_state;
    std::vector<uint64_t> build_kept_cols;
    std::vector<uint64_t> probe_kept_cols;
    std::vector<uint64_t> left_keys;
    std::vector<uint64_t> right_keys;
    std::shared_ptr<bodo::Schema> output_schema;
};
