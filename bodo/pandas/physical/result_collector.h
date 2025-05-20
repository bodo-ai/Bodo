#pragma once

#include "../libs/_table_builder.h"

#include "physical/operator.h"

class PhysicalResultCollector : public PhysicalSink {
   private:
    std::shared_ptr<TableBuildBuffer> buffer;
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;

   public:
    explicit PhysicalResultCollector(std::shared_ptr<bodo::Schema> in_schema,
                                     std::shared_ptr<bodo::Schema> out_schema)
        : in_schema(in_schema), out_schema(out_schema) {
        if (in_schema->column_names.empty() ||
            out_schema->column_names.empty()) {
            throw std::runtime_error(
                "PhysicalResultCollector::GetResult: Input/output schema must "
                "have column names.");
        }
        if (in_schema->ncols() != out_schema->ncols()) {
            throw std::runtime_error(
                "Input and output schemas must have the same number of "
                "columns.");
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

    void Finalize() override {}

    std::shared_ptr<table_info> GetResult() override {
        std::shared_ptr<table_info> in_table = buffer->data_table;

        // Reorder the columns of the table according to the output schema
        // (expected by Python). Necessary since DuckDB's optimizer may change
        // the order of columns (e.g. reorder build/probe sides in join).
        std::unordered_map<std::string, size_t> col_name_to_idx;
        for (size_t i = 0; i < in_schema->ncols(); i++) {
            col_name_to_idx[in_schema->column_names[i]] = i;
        }
        std::vector<std::shared_ptr<array_info>> out_columns;
        for (std::string& col_name : out_schema->column_names) {
            if (col_name_to_idx.find(col_name) == col_name_to_idx.end()) {
                throw std::runtime_error(
                    "PhysicalResultCollector::GetResult(): Column " + col_name +
                    " not found in input schema");
            }
            auto out_col = in_table->columns[col_name_to_idx[col_name]];
            out_columns.push_back(out_col);
        }
        std::shared_ptr<table_info> out_table = std::make_shared<table_info>(
            out_columns, in_table->nrows(), out_schema->column_names,
            out_schema->metadata);
        return out_table;
    }

   private:
    const std::shared_ptr<bodo::Schema> in_schema;
    const std::shared_ptr<bodo::Schema> out_schema;
};
