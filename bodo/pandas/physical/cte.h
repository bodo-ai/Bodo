#pragma once

#include <cstdint>
#include "../../libs/_table_builder_utils.h"
#include "../../libs/streaming/_join.h"
#include "../_util.h"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/joinside.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_cross_product.hpp"
#include "expression.h"
#include "operator.h"

class PhysicalCTERef;

/**
 * @brief Physical CTE node.
 *
 */
class PhysicalCTE : public PhysicalSink {
   public:
    explicit PhysicalCTE(const std::shared_ptr<bodo::Schema> sink_schema)
        : output_schema(sink_schema) {}

    virtual ~PhysicalCTE() = default;

    void FinalizeSink() override {}

    /**
     * @brief process input tables to build side of join (populate the hash
     * table)
     *
     * @return OperatorResult
     */
    OperatorResult ConsumeBatch(std::shared_ptr<table_info> input_batch,
                                OperatorResult prev_op_result) override {
        if (!collected_rows) {
            collected_rows = std::make_unique<ChunkedTableBuilderState>(
                input_batch->schema(), get_streaming_batch_size());
        }
        collected_rows->builder->AppendBatch(input_batch);
        return (prev_op_result == OperatorResult::FINISHED)
                   ? OperatorResult::FINISHED
                   : OperatorResult::NEED_MORE_INPUT;
    }

    /**
     * @brief GetResult - just for API compatability but should never be called
     */
    std::variant<std::shared_ptr<table_info>, PyObject*> GetResult() override {
        // CTE doesn't return output results
        throw std::runtime_error("GetResult called on a CTE node.");
    }

    std::string ToString() override { return PhysicalSink::ToString(); }

    int64_t getOpId() const { return PhysicalSink::getOpId(); }

   private:
    std::unique_ptr<ChunkedTableBuilderState> collected_rows;
    const std::shared_ptr<bodo::Schema> output_schema;
    friend class PhysicalCTERef;
};

class PhysicalCTERef : public PhysicalSource {
   public:
    explicit PhysicalCTERef(std::shared_ptr<PhysicalCTE> _cte) : cte(_cte) {}

    virtual ~PhysicalCTERef() = default;

    std::pair<std::shared_ptr<table_info>, OperatorResult> ProduceBatch()
        override {
        if (chunk_iter == std::deque<std::shared_ptr<table_info>>::iterator{}) {
            cte->collected_rows->builder->FinalizeActiveChunk();
            chunk_iter = cte->collected_rows->builder->chunks.begin();
        }
        std::shared_ptr<table_info> next_batch;
        if (chunk_iter == cte->collected_rows->builder->chunks.end()) {
            next_batch = alloc_table(cte->output_schema);
        } else {
            next_batch = *chunk_iter;
            ++chunk_iter;
        }
        bool at_end =
            (chunk_iter == cte->collected_rows->builder->chunks.end());
        return {next_batch, at_end ? OperatorResult::FINISHED
                                   : OperatorResult::HAVE_MORE_OUTPUT};
    }

    /**
     * @brief Get the output schema of join probe
     *
     * @return std::shared_ptr<bodo::Schema>
     */
    const std::shared_ptr<bodo::Schema> getOutputSchema() override {
        return cte->output_schema;
    }

    std::string ToString() override { return PhysicalSource::ToString(); }

    int64_t getOpId() const { return PhysicalSource::getOpId(); }

    void FinalizeSource() override {}

   private:
    std::shared_ptr<PhysicalCTE> cte;
    std::deque<std::shared_ptr<table_info>>::iterator chunk_iter;
};
