#include "_executor.h"
#include <arrow/python/api.h>
#include <arrow/python/pyarrow.h>
#include <arrow/status.h>
#include <cstdint>

std::pair<int64_t, PyObject*> Executor::ExecutePipelines() {
    // TODO: support multiple pipelines
    pipelines[0].Execute();
    auto output = pipelines[0].GetResult();
}
