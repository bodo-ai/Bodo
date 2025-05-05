#pragma once

#include <iostream>
#include <sstream>
#include "../libs/_table_builder.h"

#include "physical/operator.h"

class PhysicalResultCollector : public PhysicalSink {
   private:
    std::shared_ptr<TableBuildBuffer> buffer;
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;

   public:
    explicit PhysicalResultCollector(std::shared_ptr<bodo::Schema> schema) {
        for (auto& col : schema->column_types) {
            // Note that none of the columns are "keys" from the perspective of
            // the dictionary builder, which is referring to keys for
            // hashing/join
            dict_builders.emplace_back(
                create_dict_builder_for_array(col->copy(), false));
        }
        buffer = std::make_shared<TableBuildBuffer>(schema, dict_builders);
    }

    virtual ~PhysicalResultCollector() = default;

    void ConsumeBatch(std::shared_ptr<table_info> input_batch) override {
        std::stringstream ss, ss2;
        DEBUG_PrintTable(ss, buffer->data_table);
        std::cout << "got schema: " << ss.str() << std::endl;

        DEBUG_PrintTable(ss2, input_batch);
        std::cout << "got schema: " << ss2.str() << std::endl;

        buffer->UnifyTablesAndAppend(input_batch, dict_builders);
    }

    void Finalize() override {}

    std::shared_ptr<table_info> GetResult() override {
        return buffer->data_table;
    }
};
