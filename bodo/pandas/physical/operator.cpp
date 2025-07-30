#include "operator.h"

int64_t PhysicalOperator::next_op_id = 1;

int get_streaming_batch_size() {
    char* env_str = std::getenv("BODO_STREAMING_BATCH_SIZE");
    return (env_str != nullptr) ? std::stoi(env_str) : 32768;
}

// Maximum Parquet file size for streaming Parquet write
int64_t get_parquet_chunk_size() {
    char* env_str = std::getenv("BODO_PARQUET_WRITE_CHUNK_SIZE");
    return (env_str != nullptr) ? std::stoll(env_str)
                                : 256e6;  // Default to 256 MiB
}
