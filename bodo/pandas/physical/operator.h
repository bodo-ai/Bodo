#pragma once

#include <arrow/c/bridge.h>
#include <arrow/table.h>
#include <cstdint>
#ifdef USE_CUDF
#include <cudf/interop.hpp>
#include <cudf/table/table.hpp>
#endif
#include <memory>
#include <typeinfo>
#include <utility>

#include "../../libs/_bodo_common.h"
#include "../../libs/_chunked_table_builder.h"
#include "../../libs/streaming/_shuffle.h"
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

extern const bool G_USE_ASYNC;

#ifdef USE_CUDF

struct cuda_event_wrapper {
    cudaEvent_t ev;

    cuda_event_wrapper() {
        cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    }

    ~cuda_event_wrapper() {
        if (ev) {
            cudaEventDestroy(ev);
        }
    }

    // Disable copy
    cuda_event_wrapper(const cuda_event_wrapper&) = delete;
    cuda_event_wrapper& operator=(const cuda_event_wrapper&) = delete;

    // Enable move
    cuda_event_wrapper(cuda_event_wrapper&& other) noexcept {
        ev = other.ev;
        other.ev = nullptr;
    }

    cuda_event_wrapper& operator=(cuda_event_wrapper&& other) noexcept {
        if (this != &other) {
            if (ev)
                cudaEventDestroy(ev);
            ev = other.ev;
            other.ev = nullptr;
        }
        return *this;
    }

    void record(rmm::cuda_stream_view stream) {
        cudaEventRecord(ev, stream.value());
    }

    void wait(rmm::cuda_stream_view stream) const {
        cudaStreamWaitEvent(stream.value(), ev, 0);
    }
};

struct StreamAndEvent {
    rmm::cuda_stream_view stream;
    cuda_event_wrapper event;

    StreamAndEvent(rmm::cuda_stream_view s, cuda_event_wrapper&& e)
        : stream(s), event(std::move(e)) {}
};

inline std::shared_ptr<StreamAndEvent> make_stream_and_event(bool use_async) {
    if (use_async) {
        // Create a new non-blocking CUDA stream
        rmm::cuda_stream_view s{rmm::cuda_stream_per_thread};

        // Create an unsignaled event (default constructor)
        cuda_event_wrapper e;

        return std::make_shared<StreamAndEvent>(s, std::move(e));
    } else {
        // Synchronous mode: use default stream
        rmm::cuda_stream_view s = rmm::cuda_stream_default;

        // Event is already completed
        cuda_event_wrapper e;
        e.record(s);

        return std::make_shared<StreamAndEvent>(s, std::move(e));
    }
}

struct GPU_DATA {
   public:
    std::shared_ptr<cudf::table> table;
    std::shared_ptr<arrow::Schema> schema;
    std::shared_ptr<StreamAndEvent> stream_event;

    GPU_DATA(std::shared_ptr<cudf::table> t, std::shared_ptr<arrow::Schema> s,
             std::shared_ptr<StreamAndEvent> se)
        : table(t), schema(s), stream_event(se) {}
};
#else

struct GPU_DATA {};
struct StreamAndEvent {};

inline std::shared_ptr<StreamAndEvent> make_stream_and_event(bool use_async) {
    return std::shared_ptr<StreamAndEvent>();
}

#endif

/**
 * @brief Base class for sending data to/from ranks with pinned resources (GPUs)
 *
 */
class RankDataExchange {
   public:
    RankDataExchange(int64_t op_id_);

    /**
     * @brief Exchange data between ranks with pinned resources (GPUs)
     *
     * @param input_batch The input batch to be exchanged.
     * @param prev_op_result The result of the previous operator, used to
     * determine if this is the last batch locally.
     * @return std::tuple<std::shared_ptr<table_info>, OperatorResult> The
     * exchanged data this will either be empty (if no data assigned to this
     * rank) or a table with data. The OperatorResult indicates if the exchange
     * is finished on all ranks.
     */
    std::tuple<std::shared_ptr<table_info>, OperatorResult> operator()(
        std::shared_ptr<table_info> input_batch, OperatorResult prev_op_result);

    ~RankDataExchange();

   protected:
    /** @brief Initialize the RankDataExchange state including output builders
     * and shuffle state.
     *
     * @param input_batch Sample input batch to determine column types.
     */
    void Initialize(table_info* input_batch);

    /**
     * @brief Initialize shuffle state for sending data between CPU and GPU
     * ranks asynchronously.
     *
     * @param input_batch
     * @param dict_builders
     */
    virtual void InitializeShuffleState(
        table_info* input_batch,
        std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders) = 0;

    virtual int64_t GetOutBatchSize() = 0;

    int64_t op_id;

    std::vector<int> gpu_ranks;
    std::vector<int> cpu_ranks;
    MPI_Comm shuffle_comm;

    // State for synchronizing after all ranks have sent their last batch
    // to/from GPU ranks.
    std::unique_ptr<IsLastState> is_last_state;
    std::unique_ptr<IncrementalShuffleState> shuffle_state;
    std::unique_ptr<ChunkedTableBuilderState> collected_rows;
};

/**
 * @brief Class for managing data exchange between CPU ranks (all ranks on the
 * node) and GPU-pinned ranks.
 *
 */
class CPUtoGPUExchange : public RankDataExchange {
   public:
    CPUtoGPUExchange(int64_t op_id_) : RankDataExchange(op_id_) {};

   private:
    int64_t GetOutBatchSize() override;
    void InitializeShuffleState(
        table_info* input_batch,
        std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders) override;
};

/**
 * @brief Class for managing data exchange between GPU-pinned ranks and CPU
 * ranks (all ranks on the node).
 *
 */
class GPUtoCPUExchange : public RankDataExchange {
   public:
    GPUtoCPUExchange(int64_t op_id_) : RankDataExchange(op_id_) {};
    int64_t GetOutBatchSize() override;
    void InitializeShuffleState(
        table_info* input_batch,
        std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders) override;
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

#ifdef USE_CUDF
    PhysicalSink() : gpu_to_cpu_exchange(this->op_id) {}

   protected:
    GPUtoCPUExchange gpu_to_cpu_exchange;
#endif
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

#ifdef USE_CUDF
    PhysicalProcessBatch() : gpu_to_cpu_exchange(this->op_id) {}

   protected:
    GPUtoCPUExchange gpu_to_cpu_exchange;
#endif
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

    std::pair<GPU_DATA, OperatorResult> ProduceBatch() {
        std::shared_ptr<StreamAndEvent> se = make_stream_and_event(G_USE_ASYNC);
        return ProduceBatchGPU(se);
    }

    virtual std::pair<GPU_DATA, OperatorResult> ProduceBatchGPU(
        std::shared_ptr<StreamAndEvent> se) = 0;

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

    OperatorResult ConsumeBatch(GPU_DATA input_batch,
                                OperatorResult prev_op_result);

    OperatorResult ConsumeBatch(std::shared_ptr<table_info> input_batch,
                                OperatorResult prev_op_result);

    virtual OperatorResult ConsumeBatchGPU(
        GPU_DATA input_batch, OperatorResult prev_op_result,
        std::shared_ptr<StreamAndEvent> se) = 0;

    virtual std::variant<std::shared_ptr<table_info>, PyObject*>
    GetResult() = 0;

    virtual void FinalizeSink() = 0;

#ifdef USE_CUDF
    PhysicalGPUSink() : cpu_to_gpu_exchange(this->op_id) {}

   protected:
    CPUtoGPUExchange cpu_to_gpu_exchange;
#endif
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

    std::pair<GPU_DATA, OperatorResult> ProcessBatch(
        GPU_DATA input_batch, OperatorResult prev_op_result);

    std::pair<GPU_DATA, OperatorResult> ProcessBatch(
        std::shared_ptr<table_info> input_batch, OperatorResult prev_op_result);

    virtual std::pair<GPU_DATA, OperatorResult> ProcessBatchGPU(
        GPU_DATA input_batch, OperatorResult prev_op_result,
        std::shared_ptr<StreamAndEvent> se) = 0;

    virtual void FinalizeProcessBatch() = 0;

    /**
     * @brief Get the physical schema of the output data
     *
     * @return std::shared_ptr<bodo::Schema> physical schema
     */
    virtual const std::shared_ptr<bodo::Schema> getOutputSchema() = 0;

#ifdef USE_CUDF
    PhysicalGPUProcessBatch() : cpu_to_gpu_exchange(this->op_id) {}

   protected:
    CPUtoGPUExchange cpu_to_gpu_exchange;
#endif
};
/**
 * @brief Get the streaming batch size from environment variable.
 * It looks up the environment variable dynamically to enable setting it
 * in tests during runtime.
 *
 * @return int batch size to be used in streaming operators
 */
int get_streaming_batch_size();

/**
 * @brief Get the streaming batch size from environment variable.
 *
 * @return int batch size to be used in gpu streaming operators
 */
int get_gpu_streaming_batch_size();

// Maximum Parquet file size for streaming Parquet write
int64_t get_parquet_chunk_size();

using PhysicalCpuGpuSource = std::variant<std::shared_ptr<PhysicalSource>,
                                          std::shared_ptr<PhysicalGPUSource>>;
using PhysicalCpuGpuSink = std::variant<std::shared_ptr<PhysicalSink>,
                                        std::shared_ptr<PhysicalGPUSink>>;
using PhysicalCpuGpuProcessBatch =
    std::variant<std::shared_ptr<PhysicalProcessBatch>,
                 std::shared_ptr<PhysicalGPUProcessBatch>>;

#ifdef USE_CUDF
GPU_DATA convertTableToGPU(std::shared_ptr<table_info> batch);
std::shared_ptr<table_info> convertGPUToTable(GPU_DATA batch);
std::shared_ptr<arrow::Table> convertGPUToArrow(GPU_DATA batch);
#endif
