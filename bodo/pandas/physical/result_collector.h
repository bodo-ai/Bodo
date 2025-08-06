#pragma once

#include "../libs/_table_builder.h"

#include "physical/operator.h"

class PhysicalResultCollector : public PhysicalSink {
   private:
    std::shared_ptr<TableBuildBuffer> buffer;
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;
    const std::shared_ptr<bodo::Schema> in_schema;
    const std::shared_ptr<bodo::Schema> out_schema;

   public:
    explicit PhysicalResultCollector(std::shared_ptr<bodo::Schema> in_schema,
                                     std::shared_ptr<bodo::Schema> out_schema)
        : in_schema(in_schema), out_schema(out_schema) {
        // TODO: check that the input schema is compatible with the output
        // schema
        if (in_schema->ncols() != out_schema->ncols()) {
            throw std::runtime_error(
                "Input and output schemas must have the same number of "
                "columns. Input schema has " +
                std::to_string(in_schema->ncols()) + " and output schema has " +
                std::to_string(out_schema->ncols()) + " columns.");
        }
        for (auto& col : in_schema->column_types) {
            // Note that none of the columns are "keys" from the perspective of
            // the dictionary builder, which is referring to keys for
            // hashing/join
            dict_builders.emplace_back(
                create_dict_builder_for_array(col->copy(), false));
        }
        buffer = std::make_shared<TableBuildBuffer>(in_schema, dict_builders);
    }

    virtual ~PhysicalResultCollector() = default;

    OperatorResult ConsumeBatch(std::shared_ptr<table_info> input_batch,
                                OperatorResult prev_op_result) override {
        buffer->UnifyTablesAndAppend(input_batch, dict_builders);

        return prev_op_result == OperatorResult::FINISHED
                   ? OperatorResult::FINISHED
                   : OperatorResult::NEED_MORE_INPUT;
    }

    void FinalizeSink() override {}

    std::variant<std::shared_ptr<table_info>, PyObject*> GetResult() override {
        return buffer->data_table;
    }
};
