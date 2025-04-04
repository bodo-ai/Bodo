#include "_executor.h"
#include <arrow/python/api.h>
#include <arrow/python/pyarrow.h>
#include <arrow/status.h>
#include <cstdint>
#include <memory>

#include "_plan.h"

Executor::Executor(std::unique_ptr<duckdb::LogicalOperator> plan) {
    // Convert logical plan to physical plan and create query pipelines

    // TODO: support nodes other than read parquet
    duckdb::LogicalGet& get_plan = plan->Cast<duckdb::LogicalGet>();

    std::shared_ptr<PhysicalOperator> physical_op =
        get_plan.bind_data->Cast<BodoScanFunctionData>()
            .CreatePhysicalOperator();

    pipelines.emplace_back(
        std::vector<std::shared_ptr<PhysicalOperator>>({physical_op}));
}

std::pair<int64_t, PyObject*> Executor::execute() {
    // TODO: support multiple pipelines
    return pipelines[0].execute();
}

std::pair<int64_t, PyObject*> Pipeline::execute() {
    // TODO: support multiple operators
    return operators[0]->execute();
}
