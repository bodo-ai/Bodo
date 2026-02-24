#pragma once

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>
#include "_util.h"

#include "physical/operator.h"

// enable and build to print debug info on the pipeline
// #define DEBUG_PIPELINE 1  // 1 for control flow, 2 adds data
#ifdef DEBUG_PIPELINE
#include <iostream>
#endif

template <class T>
std::string getNodeString(T &t) {
    std::string ret;
    std::visit([&](auto &vt) { ret = vt->ToString(); }, t);
    return ret;
}

template <class T>
uint64_t getBatchRows(T &t) {
    uint64_t ret;
    std::visit(
        [&](auto &vt) {
            using U = std::decay_t<decltype(vt)>;
            if constexpr (std::is_same_v<U, std::shared_ptr<table_info>>) {
                ret = vt->nrows();
            } else if constexpr (std::is_same_v<U, GPU_DATA>) {
#ifdef USE_CUDF
                ret = vt.table->num_rows();
#endif
            } else {
                throw std::runtime_error("Unexpected type in getBatchRows");
            }
        },
        t);
    return ret;
}

/// @brief Pipeline class for executing a sequence of physical operators.
class Pipeline {
   private:
    PhysicalCpuGpuSource source;
    std::vector<PhysicalCpuGpuProcessBatch> between_ops;
    PhysicalCpuGpuSink sink;
    bool executed;
    // A vector of pipelines that needs to run before the current pipeline.
    std::vector<std::shared_ptr<Pipeline>> run_before;

    /**
     * @brief Execute the pipeline starting at a certain point.
     *
     * @param idx - the operator index in between_ops to start at
     * @param batch - the output of the previous operator in the pipeline
     * @param prev_op_result - the result flag of the previous operator in the
     * pipeline
     * @return - bool that is True if some operator in the pipeline has
     * indicated that no more output needs to be generated.
     */
    bool midPipelineExecute(
        unsigned idx, std::variant<std::shared_ptr<table_info>, GPU_DATA> batch,
        OperatorResult prev_op_result, int rank);

    friend class PipelineBuilder;

   public:
    /**
     * @brief Execute the pipeline and return the result (placeholder for now).
     * @return - number of batches processed
     */
    uint64_t Execute();

    /// @brief Get the final result. Result collector returns table_info,
    // Parquet write returns null table_info pointer, and Iceberg write
    // returns a PyObject* of Iceberg files infos.
    std::variant<std::variant<std::shared_ptr<table_info>, PyObject *>,
                 std::variant<GPU_DATA, PyObject *>>
    GetResult();

    // Const iterator accessors
    std::vector<std::shared_ptr<Pipeline>>::const_iterator run_before_begin()
        const {
        return run_before.begin();
    }
    std::vector<std::shared_ptr<Pipeline>>::const_iterator run_before_end()
        const {
        return run_before.end();
    }

#ifdef DEBUG_PIPELINE
    void printPipeline(void) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        std::cout << "Rank " << rank << " " << getNodeString(source)
                  << std::endl;
        for (auto &op : between_ops) {
            std::cout << "Rank " << rank << " " << getNodeString(op)
                      << std::endl;
        }
        std::cout << "Rank " << rank << " " << getNodeString(sink) << std::endl;
    }
#endif

    void removeRunBefore(std::shared_ptr<Pipeline> pipeline) {
        auto it = std::ranges::find(run_before, pipeline);
        if (it != run_before.end()) {
            run_before.erase(it);
        }
    }
};

class PipelineBuilder {
   private:
    std::vector<std::shared_ptr<Pipeline>> run_before;

    PhysicalCpuGpuSource source;
    std::vector<PhysicalCpuGpuProcessBatch> between_ops;

   public:
    explicit PipelineBuilder(PhysicalCpuGpuSource _source)
        : source(std::move(_source)) {}

    // Add a physical operator to the pipeline
    void AddOperator(PhysicalCpuGpuProcessBatch op) {
        between_ops.emplace_back(op);
    }

    void addRunBefore(std::shared_ptr<Pipeline> pipeline) {
        if (!pipeline) {
            throw std::runtime_error("Adding null pipeline to run before.");
        }
        run_before.push_back(pipeline);
    }

    /// @brief Build the pipeline and return it
    std::shared_ptr<Pipeline> Build(PhysicalCpuGpuSink sink);

    /**
     * @brief Build the last pipeline for a plan, using a result collector as
     * the sink.
     *
     * @param in_schema Schema of input data to the sink from the previous
     * operator.
     * @param out_schema Schema of output data from the sink expected by Python.
     * Only column orders may be different from the input schema due to DuckDB
     * optimizers changes (e.g. reorder build/probe sides in join).
     * @return std::shared_ptr<Pipeline> finalized pipeline
     */
    std::shared_ptr<Pipeline> BuildEnd(
        std::shared_ptr<bodo::Schema> in_schema,
        std::shared_ptr<bodo::Schema> out_schema);

    /**
     * @brief Get the physical schema of the output of the last operator in the
     pipeline (same logical schema may have different physical schema such as
     regular string arrays and dictionary-encoded ones).
     *
     * @return std::shared_ptr<bodo::Schema> physical schema
     */
    std::shared_ptr<bodo::Schema> getPrevOpOutputSchema() {
        std::shared_ptr<bodo::Schema> ret;
        if (this->between_ops.empty()) {
            std::visit([&](auto &vop) { ret = vop->getOutputSchema(); },
                       this->source);
            return ret;
        }
        std::visit([&](auto &vop) { ret = vop->getOutputSchema(); },
                   this->between_ops.back());
        return ret;
    }
};
