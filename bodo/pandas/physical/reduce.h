#pragma once

#include <arrow/compute/expression.h>
#include <arrow/scalar.h>
#include <arrow/type_fwd.h>
#include <memory>
#include <stdexcept>
#include <utility>
#include "../libs/_bodo_to_arrow.h"
#include "operator.h"

#undef CHECK_ARROW
#define CHECK_ARROW(expr, msg)                                                \
    if (!(expr.ok())) {                                                       \
        std::string err_msg = std::string("PhysicalReduce::ConsumeBatch: ") + \
                              msg + " " + expr.ToString();                    \
        throw std::runtime_error(err_msg);                                    \
    }

enum class ReductionType {
    COMPARISON,
    AGGREGATION,
};

/**
 * @brief Physical node for reductions like max.
 *
 */
class PhysicalReduce : public PhysicalSource, public PhysicalSink {
   public:
    explicit PhysicalReduce(std::shared_ptr<bodo::Schema> out_schema,
                            std::string function_name)
        // Drop Index columns since not necessary in output
        : out_schema(out_schema->Project(1)),
          function_name(function_name),
          scalar_cmp_name(getScalarOpName(function_name)) {}

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
        CHECK_ARROW(cmp_res.status(), "Error in Arrow compute kernel");
        std::shared_ptr<arrow::Scalar> out_scalar_batch =
            cmp_res.ValueOrDie().scalar();

        ReductionType reduction_type = getReductionType(function_name);

        // Update reduction result
        if (iter == 0) {
            output_scalar = out_scalar_batch;
        } else {
            arrow::Result<arrow::Datum> cmp_res_scalar =
                arrow::compute::CallFunction(scalar_cmp_name,
                                             {out_scalar_batch, output_scalar});
            CHECK_ARROW(cmp_res_scalar.status(),
                        "Error in Arrow compute scalar comparison");
            const std::shared_ptr<arrow::Scalar> cmp_scalar =
                cmp_res_scalar.ValueOrDie().scalar();
            if (reduction_type == ReductionType::COMPARISON) {
                if (cmp_scalar->Equals(arrow::BooleanScalar(true))) {
                    output_scalar = out_scalar_batch;
                }
            } else if (reduction_type == ReductionType::AGGREGATION) {
                output_scalar = cmp_scalar;
            } else {
                throw std::runtime_error("Unsupported reduction function: " +
                                         function_name);
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
        // Return reduction result as a single-element table
        std::shared_ptr<arrow::Array> array = ScalarToArrowArray(output_scalar);

        std::shared_ptr<array_info> result =
            arrow_array_to_bodo(array, bodo::BufferPool::DefaultPtr());
        std::vector<std::shared_ptr<array_info>> cvec = {result};
        std::shared_ptr<table_info> next_batch =
            std::make_shared<table_info>(cvec);

        return {next_batch, OperatorResult::FINISHED};
    }

   private:
    /**
     * @brief Get comparison operator for comparing scalars for the reduction
     * function
     *
     * @param func_name reduction function name (e.g., "max", "min")
     * @return std::string scalar comparison operator name like "greater" or
     * "less"
     */
    static std::string getScalarOpName(std::string func_name) {
        if (func_name == "max") {
            return "greater";
        } else if (func_name == "min") {
            return "less";
        } else if (func_name == "sum") {
            return "add";
        } else if (func_name == "product") {
            return "multiply";
        } else if (func_name == "count") {
            return "add";
        } else {
            throw std::runtime_error("Unsupported reduction function: " +
                                     func_name);
        }
    }

    static ReductionType getReductionType(std::string func_name) {
        if (func_name == "max") {
            return ReductionType::COMPARISON;
        } else if (func_name == "min") {
            return ReductionType::COMPARISON;
        } else if (func_name == "sum") {
            return ReductionType::AGGREGATION;
        } else if (func_name == "product") {
            return ReductionType::AGGREGATION;
        } else if (func_name == "count") {
            return ReductionType::AGGREGATION;
        } else {
            throw std::runtime_error("Unsupported reduction function: " +
                                     func_name);
        }
    }

    const std::shared_ptr<bodo::Schema> out_schema;
    const std::string function_name;
    const std::string scalar_cmp_name;

    int64_t iter = 0;
    std::shared_ptr<arrow::Scalar> output_scalar;
};
