// Executor backend for running DataFrame library plans in DuckDB format

#pragma once

#include <Python.h>
#include "../libs/_query_profile_collector.h"
#include "_physical_conv.h"
#include "_pipeline.h"
#include "duckdb/planner/logical_operator.hpp"

#ifdef DEBUG_PIPELINE
#define DEBUG_PIPELINE_CONTENTS(rank, pipeliens)                     \
    do {                                                             \
        std::cout << "Rank " << rank << " ExecutePipelines with "    \
                  << pipelines.size() << " pipelines." << std::endl; \
        for (size_t i = 0; i < pipelines.size(); ++i) {              \
            std::cout << "Rank " << rank << " ------ Pipeline " << i \
                      << " ------" << std::endl;                     \
            pipelines[i]->printPipeline();                           \
            std::cout << "Rank " << rank << " ------ Pipeline " << i \
                      << " ------" << std::endl;                     \
        }                                                            \
    } while (0)
#else
#define DEBUG_PIPELINE_CONTENTS(rank, pipeliens) \
    do {                                         \
    } while (0)
#endif

#ifdef DEBUG_PIPELINE
#define DEBUG_PIPELINE_PRE_EXECUTE(rank)                                 \
    do {                                                                 \
        std::cout << "Rank " << rank << " Before execute pipeline " << i \
                  << std::endl;                                          \
    } while (0)
#else
#define DEBUG_PIPELINE_PRE_EXECUTE(rank) \
    do {                                 \
    } while (0)
#endif

#ifdef DEBUG_PIPELINE
#define DEBUG_PIPELINE_POST_EXECUTE(rank)                               \
    do {                                                                \
        std::cout << "Rank " << rank << " After execute pipeline " << i \
                  << std::endl;                                         \
    } while (0)
#else
#define DEBUG_PIPELINE_POST_EXECUTE(rank) \
    do {                                  \
    } while (0)
#endif

/**
 * @brief Executor class for executing a DuckDB logical plan in streaming
 * fashion (push-based approach).
 *
 */
class Executor {
   private:
    std::vector<std::shared_ptr<Pipeline>> pipelines;

   public:
    // Holds table_index to PhysicalCTE mapping during physical plan
    // construction. Executor only active for one plan execution so ctes cleaned
    // up by destructor.
    std::map<duckdb::idx_t, std::shared_ptr<PhysicalCTE>> ctes;

   public:
    explicit Executor(std::unique_ptr<duckdb::LogicalOperator> plan,
                      std::shared_ptr<arrow::Schema> out_schema) {
        QueryProfileCollector::Default().Init();
        // Convert the logical plan to a physical plan
        PhysicalPlanBuilder builder(ctes);
        builder.Visit(*plan);

        // Move frozen/locked pipelines from builder to Executor pipelines.
        pipelines.insert(
            pipelines.end(),
            std::make_move_iterator(builder.locked_pipelines.begin()),
            std::make_move_iterator(builder.locked_pipelines.end()));

        // Move normal pipelines from builder to Executor pipelines.
        pipelines.insert(
            pipelines.end(),
            std::make_move_iterator(builder.finished_pipelines.begin()),
            std::make_move_iterator(builder.finished_pipelines.end()));

        // Write finalizes the active pipeline but others need result collection
        if (builder.active_pipeline != nullptr) {
            std::shared_ptr<bodo::Schema> in_schema =
                builder.active_pipeline->getPrevOpOutputSchema();
            pipelines.push_back(builder.active_pipeline->BuildEnd(
                in_schema, bodo::Schema::FromArrowSchema(out_schema)));
        }
    }

    /**
     * @brief Execute the plan and return the result.
     */
    std::variant<std::shared_ptr<table_info>, PyObject*> ExecutePipelines() {
        // Pipelines generation ensures that pipelines are in the right
        // order and that the dependencies are satisfied (e.g. join build
        // pipeline is before probe).
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        DEBUG_PIPELINE_CONTENTS(rank, pipelines);

        for (size_t i = 0; i < pipelines.size(); ++i) {
            QueryProfileCollector::Default().StartPipeline(i);
            DEBUG_PIPELINE_PRE_EXECUTE(rank);
            uint64_t batches_processed = pipelines[i]->Execute();

            // Free pipeline resources as early as possible to reduce memory
            // pressure.
            if (i < pipelines.size() - 1) {
                pipelines[i].reset();
            }

            DEBUG_PIPELINE_POST_EXECUTE(rank);
            QueryProfileCollector::Default().EndPipeline(i, batches_processed);
        }
        QueryProfileCollector::Default().Finalize(0);
        return pipelines.back()->GetResult();
    }
};
