#pragma once

#include <memory>
#include <utility>

#include <arrow/python/pyarrow.h>
#include <arrow/table.h>

#include "../libs/_bodo_to_arrow.h"
#include "../libs/_table_builder_utils.h"
#include "operator.h"

/// @brief Physical node for reading Parquet files in pipelines.
class PhysicalReadEmpty : public PhysicalSource {
   private:
    const std::shared_ptr<bodo::Schema> output_schema;

   public:
    explicit PhysicalReadEmpty(std::vector<duckdb::LogicalType> return_types)
        : output_schema(initOutputSchema(return_types)) {}

    virtual ~PhysicalReadEmpty() = default;

    /**
     * @brief Initialize the output schema based on the selected columns and
     * Arrow schema.
     *
     * @param selected_columns The selected columns to project.
     * @param arrow_schema The Arrow schema of the DataFrame.
     * @return std::shared_ptr<bodo::Schema> The initialized output schema.
     */
    static std::shared_ptr<bodo::Schema> initOutputSchema(
        std::vector<duckdb::LogicalType> return_types) {
        return bodo::Schema::FromArrowSchema(
            duckdb_to_arrow_schema(return_types));
    }

    void FinalizeSource() override {}

    std::pair<std::shared_ptr<table_info>, OperatorResult> ProduceBatch()
        override {
        std::shared_ptr<table_info> out_table = alloc_table(output_schema);
        return {out_table, OperatorResult::FINISHED};
    }

    /**
     * @brief Get the physical schema of the dataframe/series data
     *
     * @return std::shared_ptr<bodo::Schema> physical schema
     */
    const std::shared_ptr<bodo::Schema> getOutputSchema() override {
        return output_schema;
    }
};
