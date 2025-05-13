
#pragma once

#include <cstdint>
#include "../../libs/streaming/_join.h"
#include "operator.h"

/**
 * @brief Physical node for join.
 *
 */
class PhysicalJoin : public PhysicalSourceSink, public PhysicalSink {
   public:
    explicit PhysicalJoin() {}

    virtual ~PhysicalJoin() = default;

    void InitializeJoinState(
        const std::shared_ptr<bodo::Schema> build_table_schema,
        const std::shared_ptr<bodo::Schema> probe_table_schema) {
        this->join_state = std::make_shared<HashJoinState>(
            build_table_schema, probe_table_schema,
            // TODO: handle keys properly
            1,
            // TODO: handle outer joins properly
            false, false,
            // TODO: handle broadcast join properly
            false, nullptr, true, true, get_streaming_batch_size(), -1,
            // TODO: add op_id
            -1);

        this->build_kept_cols.resize(build_table_schema->ncols());
        std::iota(this->build_kept_cols.begin(), this->build_kept_cols.end(),
                  0);
        this->probe_kept_cols.resize(probe_table_schema->ncols());
        std::iota(this->probe_kept_cols.begin(), this->probe_kept_cols.end(),
                  0);
    }

    void Finalize() override {}

    /**
     * @brief process input tables to build side of join (populate the hash
     * table)
     *
     * @return OperatorResult
     */
    OperatorResult ConsumeBatch(
        std::shared_ptr<table_info> input_batch) override {
        bool has_bloom_filter = join_state->global_bloom_filter != nullptr;
        // TODO: fix is_last
        this->join_state->global_is_last = true;
        // TODO: handle output
        bool is_last = join_build_consume_batch(this->join_state.get(),
                                                input_batch, has_bloom_filter,
                                                // TODO: set is_last properly
                                                true);

        if (is_last) {
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
        std::shared_ptr<table_info> input_batch) override {
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

        // TODO
        bool is_last = true;

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

        return {out_table, request_input ? OperatorResult::NEED_MORE_INPUT
                                         : OperatorResult::HAVE_MORE_OUTPUT};
    }

    /**
     * @brief GetResult - just for API compatability but should never be called
     */
    std::shared_ptr<table_info> GetResult() override {
        // Join build doesn't return output results
        throw std::runtime_error("GetResult called on a join node.");
    }

    std::shared_ptr<bodo::Schema> getOutputSchema() override {
        // TODO
        return nullptr;
    }

   private:
    std::shared_ptr<HashJoinState> join_state;
    std::vector<uint64_t> build_kept_cols;
    std::vector<uint64_t> probe_kept_cols;
};
