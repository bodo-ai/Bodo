#pragma once

#include <memory>
#include <utility>

#include "../libs/_bodo_common.h"

/// Specifies physical operator types in the execution pipeline:
// 1. Source means the first operator of the pipeline that only produces batchs
// 2. Sink means the last operator of the pipeline that only consumes batches
// 3. SourceAndSink means operators in the middle of the pipeline that both
// produce and consume batches
enum class OperatorType : uint8_t {
    SOURCE,
    SINK,
    SOURCE_AND_SINK,
};

/// Specifies the status of the physical operator in the execution:
/// 1. NEED_MORE_INPUT means the operator is ready for additional input
/// 2. HAVE_MORE_OUTPUT means the operator can produce more output without
/// additional input.
/// 3. FINISHED means the pipeline should terminate (used only for LIMIT)
// DuckDB's description:
// https://youtu.be/MA0OsvYFGrc?t=1205
enum class OperatorResult : uint8_t {
    NEED_MORE_INPUT,
    HAVE_MORE_OUTPUT,
    FINISHED,
};

/**
 * @brief Physical operators to be used in the execution pipelines (NOTE: they
 * are Bodo classes and not using DuckDB).
 */
class PhysicalOperator {
   public:
    virtual OperatorType operator_type() const = 0;

    bool is_source() const { return operator_type() != OperatorType::SINK; }
    bool is_sink() const { return operator_type() != OperatorType::SOURCE; }

    // Constructor is always required for initialization
    // We should have a separate Finalize step that can throw an exception
    // as well as the destructor for cleanup
    virtual void Finalize() = 0;
};

class PhysicalSource : public PhysicalOperator {
   public:
    OperatorType operator_type() const override { return OperatorType::SOURCE; }

    virtual std::pair<std::shared_ptr<table_info>, OperatorResult>
    ProduceBatch() = 0;

    virtual std::shared_ptr<bodo::Schema> getOutputSchema() = 0;
};

class PhysicalSink : public PhysicalOperator {
   public:
    OperatorType operator_type() const override { return OperatorType::SINK; }

    virtual OperatorResult ConsumeBatch(std::shared_ptr<table_info> input_batch,
                                        OperatorResult prev_op_result) = 0;
    virtual std::shared_ptr<table_info> GetResult() = 0;
};

class PhysicalSourceSink : public PhysicalOperator {
   public:
    OperatorType operator_type() const override {
        return OperatorType::SOURCE_AND_SINK;
    }

    virtual std::pair<std::shared_ptr<table_info>, OperatorResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch,
        OperatorResult prev_op_result) = 0;

    virtual std::shared_ptr<bodo::Schema> getOutputSchema() = 0;
};

/**
 * @brief Get the streaming batch size from environment variable.
 * It looks up the environment variable dynamically to enable setting it
 * in tests during runtime.
 *
 * @return int batch size to be used in streaming operators
 */
int get_streaming_batch_size();
