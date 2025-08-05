#pragma once

#include <memory>
#include <typeinfo>
#include <utility>

#include "../libs/_bodo_common.h"
#include "../libs/_query_profile_collector.h"

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
/// 3. FINISHED means the operator is done executing.
// This is passed across operators and the pipeline terminates when the sink
// operator returns this status.
// DuckDB's description for background (Bodo's
// semantics is different per above): https://youtu.be/MA0OsvYFGrc?t=1205
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
    PhysicalOperator() : op_id(next_op_id++) {}

    virtual OperatorType operator_type() const = 0;

    bool is_source() const { return operator_type() != OperatorType::SINK; }
    bool is_sink() const { return operator_type() != OperatorType::SOURCE; }

    // Constructor is always required for initialization
    // We should have a separate Finalize step that can throw an exception
    // as well as the destructor for cleanup
    virtual void Finalize() = 0;

    virtual std::string ToString() {
        return typeid(*this).name();  // returns mangled name
    }

    int64_t getOpId() const { return op_id; }

   protected:
    int64_t op_id;
    static int64_t next_op_id;
};

/**
 * @brief Base class for operators that produce batches at the start of
 * pipelines.
 *
 */
class PhysicalSource : public PhysicalOperator {
   public:
    OperatorType operator_type() const override { return OperatorType::SOURCE; }

    virtual std::pair<std::shared_ptr<table_info>, OperatorResult>
    ProduceBatch() = 0;

    /**
     * @brief Get the physical schema of the source data
     *
     * @return std::shared_ptr<bodo::Schema> physical schema
     */
    virtual const std::shared_ptr<bodo::Schema> getOutputSchema() = 0;
};

/**
 * @brief Base class for operators that consume batches at the end of pipelines.
 *
 */
class PhysicalSink : public PhysicalOperator {
   public:
    OperatorType operator_type() const override { return OperatorType::SINK; }

    virtual OperatorResult ConsumeBatch(std::shared_ptr<table_info> input_batch,
                                        OperatorResult prev_op_result) = 0;
    virtual std::variant<std::shared_ptr<table_info>, PyObject*>
    GetResult() = 0;
};

/**
 * @brief Base class for operators that both consume and produce batches in the
 * middle of pipelines.
 *
 */
class PhysicalProcessBatch : public PhysicalOperator {
   public:
    OperatorType operator_type() const override {
        return OperatorType::SOURCE_AND_SINK;
    }

    virtual std::pair<std::shared_ptr<table_info>, OperatorResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch,
        OperatorResult prev_op_result) = 0;

    /**
     * @brief Get the physical schema of the output data
     *
     * @return std::shared_ptr<bodo::Schema> physical schema
     */
    virtual const std::shared_ptr<bodo::Schema> getOutputSchema() = 0;
};

/**
 * @brief Get the streaming batch size from environment variable.
 * It looks up the environment variable dynamically to enable setting it
 * in tests during runtime.
 *
 * @return int batch size to be used in streaming operators
 */
int get_streaming_batch_size();

// Maximum Parquet file size for streaming Parquet write
int64_t get_parquet_chunk_size();
