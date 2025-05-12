#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "physical/operator.h"

/// @brief Pipeline class for executing a sequence of physical operators.
class Pipeline {
   private:
    std::shared_ptr<PhysicalSource> source;
    std::vector<std::shared_ptr<PhysicalSourceSink>> between_ops;
    std::shared_ptr<PhysicalSink> sink;
    bool executed;
    std::vector<std::shared_ptr<Pipeline>> dependencies;

    /**
     * @brief Execute the pipeline starting at a certain point.
     *
     * param idx - the operator index in between_ops to start at
     * param batch - the output of the previous operator in the pipeline
     * returns - bool that is True if some operator in the pipeline has
     * indicated that no more output needs to be generated.
     */
    bool midPipelineExecute(unsigned idx, std::shared_ptr<table_info> batch);

    friend class PipelineBuilder;

   public:
    /**
     * @brief Execute the pipeline and return the result (placeholder for now).
     */
    void Execute();

    /// @brief Get the final result. Should be anything because of write, but
    /// stick to table_info for now
    std::shared_ptr<table_info> GetResult();

    /**
     * @brief Check all dependent pipelines to see if they have completed.
     *
     * returns true if all dependencies are completed, false otherwise.
     */
    bool dependencies_finished() const {
        for (size_t i = 0; i < dependencies.size(); ++i) {
            if (!dependencies[i]->executed) {
                return false;
            }
        }
        return true;
    }
};

class PipelineBuilder {
   private:
    std::shared_ptr<PhysicalSource> source;
    std::vector<std::shared_ptr<PhysicalSourceSink>> between_ops;

   public:
    explicit PipelineBuilder(std::shared_ptr<PhysicalSource> _source)
        : source(std::move(_source)) {};

    // Add a physical operator to the pipeline
    void AddOperator(std::shared_ptr<PhysicalSourceSink> op) {
        between_ops.emplace_back(op);
    }

    /// @brief Build the pipeline and return it
    std::shared_ptr<Pipeline> Build(std::shared_ptr<PhysicalSink> sink);

    /// @brief Build the last pipeline for a plan, using a result collector as
    /// the sink.
    std::shared_ptr<Pipeline> BuildEnd(
        std::shared_ptr<arrow::Schema> out_schema);
};
