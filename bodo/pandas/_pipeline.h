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
    bool midPipelineExecute(unsigned idx, std::shared_ptr<table_info> batch,
                            OperatorResult prev_op_result);

    friend class PipelineBuilder;

   public:
    /**
     * @brief Execute the pipeline and return the result (placeholder for now).
     */
    void Execute();

    /// @brief Get the final result. Should be anything because of write, but
    /// stick to table_info for now
    std::shared_ptr<table_info> GetResult();
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

    std::shared_ptr<bodo::Schema> getPrevOpOutputSchema() {
        if (this->between_ops.empty()) {
            return this->source->getOutputSchema();
        }
        return this->between_ops.back()->getOutputSchema();
    }
};
