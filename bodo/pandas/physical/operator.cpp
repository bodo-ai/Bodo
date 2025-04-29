#include "operator.h"

int get_streaming_batch_size() {
    char* env_str = std::getenv("BODO_STREAMING_BATCH_SIZE");
    return (env_str != nullptr) ? std::stoi(env_str) : 4096;
}
