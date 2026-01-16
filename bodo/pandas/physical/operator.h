#pragma once

#include <memory>
#include <typeinfo>
#include <utility>

#include "../_util.h"
#include "../libs/_bodo_common.h"
#include "../libs/_bodo_to_arrow.h"
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
    GPU_SOURCE,
    GPU_SINK,
    GPU_SOURCE_AND_SINK,
};

/// Specifies the status of the physical operator in the execution:
/// 0. NEED_MORE_INPUT means the operator is ready for additional input
/// 1. HAVE_MORE_OUTPUT means the operator can produce more output without
/// additional input.
/// 2. FINISHED means the operator is done executing.
// This is passed across operators and the pipeline terminates when the sink
// operator returns this status.
// DuckDB's description for background (Bodo's
// semantics is different per above): https://youtu.be/MA0OsvYFGrc?t=1205
enum class OperatorResult : uint8_t {
    NEED_MORE_INPUT = 0,
    HAVE_MORE_OUTPUT = 1,
    FINISHED = 2,
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

    // Constructor is always required for initialization
    // We should have a separate Finalize step that can throw an exception
    // as well as the destructor for cleanup
    virtual void FinalizeSource() = 0;

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

    virtual OperatorResult ConsumeBatch(GPU_DATA input_batch,
                                        OperatorResult prev_op_result);

    virtual std::variant<std::shared_ptr<table_info>, PyObject*>
    GetResult() = 0;

    virtual void FinalizeSink() = 0;
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

    virtual std::pair<std::shared_ptr<table_info>, OperatorResult> ProcessBatch(
        GPU_DATA input_batch, OperatorResult prev_op_result);

    virtual void FinalizeProcessBatch() = 0;

    /**
     * @brief Get the physical schema of the output data
     *
     * @return std::shared_ptr<bodo::Schema> physical schema
     */
    virtual const std::shared_ptr<bodo::Schema> getOutputSchema() = 0;
};

/**
 * @brief Base class for operators that produce batches at the start of
 * pipelines.
 *
 */
class PhysicalGPUSource : public PhysicalOperator {
   public:
    OperatorType operator_type() const override {
        return OperatorType::GPU_SOURCE;
    }

    virtual std::pair<GPU_DATA, OperatorResult> ProduceBatch() = 0;

    // Constructor is always required for initialization
    // We should have a separate Finalize step that can throw an exception
    // as well as the destructor for cleanup
    virtual void FinalizeSource() = 0;

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
class PhysicalGPUSink : public PhysicalOperator {
   public:
    OperatorType operator_type() const override {
        return OperatorType::GPU_SINK;
    }

    virtual OperatorResult ConsumeBatch(GPU_DATA input_batch,
                                        OperatorResult prev_op_result) = 0;

    virtual OperatorResult ConsumeBatch(std::shared_ptr<table_info> input_batch,
                                        OperatorResult prev_op_result);

    virtual std::variant<GPU_DATA, PyObject*> GetResult() = 0;

    virtual void FinalizeSink() = 0;
};

/**
 * @brief Base class for operators that both consume and produce batches in the
 * middle of pipelines.
 *
 */
class PhysicalGPUProcessBatch : public PhysicalOperator {
   public:
    OperatorType operator_type() const override {
        return OperatorType::GPU_SOURCE_AND_SINK;
    }

    virtual std::pair<GPU_DATA, OperatorResult> ProcessBatch(
        GPU_DATA input_batch, OperatorResult prev_op_result) = 0;

    virtual std::pair<GPU_DATA, OperatorResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch, OperatorResult prev_op_result);

    virtual void FinalizeProcessBatch() = 0;

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

using PhysicalCpuGpuSource = std::variant<std::shared_ptr<PhysicalSource>,
                                          std::shared_ptr<PhysicalGPUSource>>;
using PhysicalCpuGpuSink = std::variant<std::shared_ptr<PhysicalSink>,
                                        std::shared_ptr<PhysicalGPUSink>>;
using PhysicalCpuGpuProcessBatch =
    std::variant<std::shared_ptr<PhysicalProcessBatch>,
                 std::shared_ptr<PhysicalGPUProcessBatch>>;
