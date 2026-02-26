#pragma once

#include <cstdint>
#include "../../libs/streaming/_join.h"
#include "../_util.h"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/joinside.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_cross_product.hpp"
#include "gpu_expression.h"
#include "operator.h"

class PhysicalGPUCTERef;

inline bool gpu_capable(duckdb::LogicalMaterializedCTE &cte) { return true; }

inline bool gpu_capable(duckdb::LogicalCTERef &cteref) { return true; }

/**
 * @brief Physical CTE node.
 *
 */
class PhysicalGPUCTE : public PhysicalGPUSink {
   public:
    explicit PhysicalGPUCTE(const std::shared_ptr<bodo::Schema> sink_schema)
        : output_schema(sink_schema) {
        arrow_output_schema = this->output_schema->ToArrowSchema();
    }

    virtual ~PhysicalGPUCTE() = default;

    void FinalizeSink() override {}

    /**
     * @brief process input tables to build side of join (populate the hash
     * table)
     *
     * @return OperatorResult
     */
    OperatorResult ConsumeBatchGPU(
        GPU_DATA input_batch, OperatorResult prev_op_result,
        std::shared_ptr<StreamAndEvent> se) override {
        collected_rows.push_back(input_batch);
        return (prev_op_result == OperatorResult::FINISHED)
                   ? OperatorResult::FINISHED
                   : OperatorResult::NEED_MORE_INPUT;
    }

    /**
     * @brief GetResult - just for API compatability but should never be called
     */
    std::variant<std::shared_ptr<table_info>, PyObject *> GetResult() override {
        // CTE doesn't return output results
        throw std::runtime_error("GetResult called on a CTE node.");
    }

    std::string ToString() override { return PhysicalGPUSink::ToString(); }

    int64_t getOpId() const { return PhysicalGPUSink::getOpId(); }

   private:
    std::vector<GPU_DATA> collected_rows;
    const std::shared_ptr<bodo::Schema> output_schema;
    std::shared_ptr<arrow::Schema> arrow_output_schema;
    friend class PhysicalGPUCTERef;
};

class PhysicalGPUCTERef : public PhysicalGPUSource {
   public:
    explicit PhysicalGPUCTERef(std::shared_ptr<PhysicalGPUCTE> _cte)
        : cte(_cte) {}

    virtual ~PhysicalGPUCTERef() = default;

    std::pair<GPU_DATA, OperatorResult> ProduceBatchGPU(
        std::shared_ptr<StreamAndEvent> se) override {
        GPU_DATA next_batch;
        if (next_index >= cte->collected_rows.size()) {
            next_batch.table =
                empty_table_from_arrow_schema(cte->arrow_output_schema);
            next_batch.schema = cte->arrow_output_schema;
        } else {
            next_batch = cte->collected_rows[next_index];
            next_batch.stream_event->event.wait(se->stream);
            next_index++;
        }
        next_batch.stream_event = se;

        bool at_end = next_index >= cte->collected_rows.size();
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

    std::string ToString() override { return PhysicalGPUSource::ToString(); }

    int64_t getOpId() const { return PhysicalGPUSource::getOpId(); }

    void FinalizeSource() override {}

   private:
    std::shared_ptr<PhysicalGPUCTE> cte;
    size_t next_index = 0;
};
