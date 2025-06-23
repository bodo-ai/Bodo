#pragma once

#include <arrow/compute/expression.h>
#include <arrow/scalar.h>
#include <arrow/type_fwd.h>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include "../_util.h"
#include "../io/arrow_reader.h"
#include "../libs/_array_utils.h"
#include "../libs/_bodo_to_arrow.h"
#include "../libs/_utils.h"
#include "../libs/groupby/_groupby_ftypes.h"
#include "../libs/streaming/_groupby.h"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "expression.h"
#include "operator.h"

/**
 * @brief Physical node for reductions like max.
 *
 */
class PhysicalReduce : public PhysicalSource, public PhysicalSink {
   public:
    explicit PhysicalReduce(std::shared_ptr<bodo::Schema> in_table_schema,
                            std::string function_name)
        : out_schema(in_table_schema), function_name(std::move(function_name)) {
        // TODO: support reductions that out different output types from input
        // types (e.g. upcast in sum)
    }

    virtual ~PhysicalReduce() = default;

    void Finalize() override {}

    OperatorResult ConsumeBatch(std::shared_ptr<table_info> input_batch,
                                OperatorResult prev_op_result) override {
        // Convert to Arrow array
        arrow::TimeUnit::type time_unit = arrow::TimeUnit::NANO;
        std::shared_ptr<arrow::Array> in_arrow_array = bodo_array_to_arrow(
            bodo::BufferPool::DefaultPtr(), input_batch->columns[0],
            false /*convert_timedelta_to_int64*/, "", time_unit,
            false, /*downcast_time_ns_to_us*/
            bodo::default_buffer_memory_manager());

        // Reduce Arrow array using compute function
        arrow::Result<arrow::Datum> cmp_res =
            arrow::compute::CallFunction(function_name, {in_arrow_array});
        if (!cmp_res.ok()) [[unlikely]] {
            throw std::runtime_error(
                "PhysicalReduce::ConsumeBatch: Error in Arrow compute: " +
                cmp_res.status().message());
        }
        std::shared_ptr<arrow::Scalar> out_scalar_batch =
            cmp_res.ValueOrDie().scalar();

        // Update reduction result
        if (iter == 0) {
            output_scalar = out_scalar_batch;
        } else {
            arrow::Result<arrow::Datum> cmp_res_scalar =
                arrow::compute::CallFunction("greater",
                                             {out_scalar_batch, output_scalar});
            if (!cmp_res_scalar.ok()) [[unlikely]] {
                throw std::runtime_error(
                    "PhysicalReduce::ConsumeBatch: Error in Arrow compute: " +
                    cmp_res.status().message());
            }
            const std::shared_ptr<arrow::Scalar> cmp_scalar =
                cmp_res_scalar.ValueOrDie().scalar();
            if (cmp_scalar->Equals(arrow::BooleanScalar(true))) {
                output_scalar = out_scalar_batch;
            }
        }

        iter++;
        return prev_op_result == OperatorResult::FINISHED
                   ? OperatorResult::FINISHED
                   : OperatorResult::NEED_MORE_INPUT;
    }

    std::variant<std::shared_ptr<table_info>, PyObject*> GetResult() override {
        throw std::runtime_error("GetResult called on a PhysicalReduce node.");
    }

    const std::shared_ptr<bodo::Schema> getOutputSchema() override {
        return out_schema;
    }

    std::pair<std::shared_ptr<table_info>, OperatorResult> ProduceBatch()
        override {
        std::shared_ptr<arrow::Array> array = ScalarToArrowArray(output_scalar);

        std::shared_ptr<array_info> result =
            arrow_array_to_bodo(array, bodo::BufferPool::DefaultPtr());
        std::vector<std::shared_ptr<array_info>> cvec = {result};
        std::shared_ptr<table_info> next_batch =
            std::make_shared<table_info>(cvec);
        return {next_batch, OperatorResult::FINISHED};
    }

   private:
    const std::shared_ptr<bodo::Schema> out_schema;
    const std::string function_name;
    int64_t iter = 0;
    std::shared_ptr<arrow::Scalar> output_scalar;
};
