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
    /*
     * @brief Do topological sort and fill in the pipelines vector.
     * https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
     * with the modification that when a dependency is removed it is only
     * removed from one of its parents' run_before list since we don't have an
     * easy reference to remove it from all parents. As a result we have to
     * check that a pipeline hasn't already been inserted since it can be
     * visited from multiple parents (e.g. Join Filters).
     *
     * @param cur - the current Pipeline to examine for inclusion in pipelines.
     * @param seen - used for recursion stack cycle detection
     * @return true if cur was added to pipelines, false otherwise.
     */
    bool fillPipelinesTopoSort(std::shared_ptr<Pipeline> cur,
                               std::set<std::shared_ptr<Pipeline>> seen =
                                   std::set<std::shared_ptr<Pipeline>>()) {
        // Check if cur is already in seen
        if (seen.find(cur) != seen.end()) {
            throw std::runtime_error(
                "Cycle detected during fillPipelinesTopoSort.");
        }

        // Otherwise, mark it as seen
        seen.insert(cur);

        std::vector<std::shared_ptr<Pipeline>> inserted_dependents;
        for (auto it = cur->run_before_begin(); it != cur->run_before_end();
             ++it) {
            if (fillPipelinesTopoSort(*it, seen)) {
                // Dependent was added so remove it from run_before
                inserted_dependents.emplace_back(*it);
            }
        }
        for (auto &dep : inserted_dependents) {
            cur->removeRunBefore(dep);
        }

        // Remove it from seen so if it occurs in other parts of the tree
        // (which can happen for CTE pipelines) that it won't falsely
        // think there is a cycle and throw an exception.
        seen.erase(cur);
        // If all dependents have been added, add this one
        if (cur->run_before_begin() == cur->run_before_end()) {
            // No dependents so leaf node
            // insert it if it isn't already present
            if (std::ranges::find(pipelines, cur) == pipelines.end()) {
                pipelines.emplace_back(cur);
            }
            return true;
        }
        return false;
    }

   public:
    // Holds table_index to PhysicalCTE mapping during physical plan
    // construction. Executor only active for one plan execution so ctes cleaned
    // up by destructor.
    std::map<duckdb::idx_t, CTEInfo> ctes;
    std::map<void *, bool> run_on_gpu;

   private:
    void partition_internal(duckdb::LogicalOperator &op,
                            std::map<void *, bool> &run_on_gpu);

   public:
    std::map<void *, bool> partition_to_gpu(
        std::unique_ptr<duckdb::LogicalOperator> &plan) {
        std::map<void *, bool> run_on_gpu;
        partition_internal(*plan, run_on_gpu);
        return run_on_gpu;
    }

    explicit Executor(std::unique_ptr<duckdb::LogicalOperator> plan,
                      std::shared_ptr<arrow::Schema> out_schema) {
        QueryProfileCollector::Default().Init();
        // Partition between CPU and GPU.
        run_on_gpu = partition_to_gpu(plan);
        // Convert the logical plan to a physical plan
        PhysicalPlanBuilder builder(ctes, run_on_gpu);
        builder.Visit(*plan);

        // Write finalizes the active pipeline but others need result collection
        std::shared_ptr<Pipeline> root_pipeline;
        if (builder.active_pipeline != nullptr) {
            std::shared_ptr<bodo::Schema> in_schema =
                builder.active_pipeline->getPrevOpOutputSchema();
            root_pipeline = builder.active_pipeline->BuildEnd(
                in_schema, bodo::Schema::FromArrowSchema(out_schema));
        } else {
            assert(builder.terminal_pipeline);
            root_pipeline = builder.terminal_pipeline;
        }
        fillPipelinesTopoSort(root_pipeline);
    }

    /**
     * @brief Execute the plan and return the result.
     */
    std::variant<std::shared_ptr<table_info>, PyObject *> ExecutePipelines() {
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
        std::variant<std::variant<std::shared_ptr<table_info>, PyObject *>,
                     std::variant<GPU_DATA, PyObject *>>
            gr_res = pipelines.back()->GetResult();
        std::variant<std::shared_ptr<table_info>, PyObject *> ret;
        std::visit(
            [&](auto &vres) {
                using T = std::decay_t<decltype(vres)>;
                if constexpr (std::is_same_v<
                                  T, std::variant<GPU_DATA, PyObject *>>) {
#ifdef USE_CUDF
                    std::visit(
                        [&](auto &gpu_var) {
                            using U = std::decay_t<decltype(gpu_var)>;
                            if constexpr (std::is_same_v<U, GPU_DATA>) {
                                ret = convertGPUToTable(gpu_var);
                            } else {
                                ret = gpu_var;
                            }
                        },
                        vres);
#endif
                } else if constexpr (std::is_same_v<
                                         T, std::variant<
                                                std::shared_ptr<table_info>,
                                                PyObject *>>) {
                    std::visit(
                        [&](auto &table_or_pyobject_var) {
                            ret = table_or_pyobject_var;
                        },
                        vres);
                } else {
                    static_assert(sizeof(T) == 0,
                                  "Unexpected GetResult() alternative.");
                }
            },
            gr_res);
        return ret;
    }
};
