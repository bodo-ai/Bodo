#include "operator.h"

int64_t PhysicalOperator::next_op_id = 1;

int get_streaming_batch_size() {
    char *env_str = std::getenv("BODO_STREAMING_BATCH_SIZE");
    return (env_str != nullptr) ? std::stoi(env_str) : 32768;
}

// Maximum Parquet file size for streaming Parquet write
int64_t get_parquet_chunk_size() {
    char *env_str = std::getenv("BODO_PARQUET_WRITE_CHUNK_SIZE");
    return (env_str != nullptr) ? std::stoll(env_str)
                                : 256e6;  // Default to 256 MiB
}

OperatorResult PhysicalSink::ConsumeBatch(GPU_DATA input_batch,
                                          OperatorResult prev_op_result) {
    auto cpu_batch = convertGPUToTable(input_batch);
    return ConsumeBatch(cpu_batch, prev_op_result);
}

std::pair<std::shared_ptr<table_info>, OperatorResult>
PhysicalProcessBatch::ProcessBatch(GPU_DATA input_batch,
                                   OperatorResult prev_op_result) {
    auto cpu_batch = convertGPUToTable(input_batch);
    return ProcessBatch(cpu_batch, prev_op_result);
}

OperatorResult PhysicalGPUSink::ConsumeBatch(
    std::shared_ptr<table_info> input_batch, OperatorResult prev_op_result) {
    auto gpu_batch = convertTableToGPU(input_batch);
    return ConsumeBatch(gpu_batch, prev_op_result);
}

std::pair<GPU_DATA, OperatorResult> PhysicalGPUProcessBatch::ProcessBatch(
    std::shared_ptr<table_info> input_batch, OperatorResult prev_op_result) {
    auto gpu_batch = convertTableToGPU(input_batch);
    return ProcessBatch(gpu_batch, prev_op_result);
}
