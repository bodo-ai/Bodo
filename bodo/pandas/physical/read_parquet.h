#pragma once

#include <memory>
#include <utility>
#include "../io/parquet_reader.h"
#include "operator.h"

/// @brief Physical node for reading Parquet files in pipelines.
class PhysicalReadParquet : public PhysicalSource {
   private:
    ParquetReader internal_reader;

   public:
    // TODO: Fill in the contents with info from the logical operator
    PhysicalReadParquet()
        : internal_reader(nullptr, false, nullptr, nullptr, nullptr, 0, {}, {},
                          false, 0) {
        internal_reader.init_pq_reader({}, nullptr, nullptr, 0);
    }

    void Finalize() override {}

    std::pair<std::shared_ptr<table_info>, ProducerResult> ProduceBatch()
        override {
        uint64_t total_rows;
        bool is_last;

        auto batch = internal_reader.read_batch(is_last, total_rows, true);
        auto result = is_last ? ProducerResult::FINISHED
                              : ProducerResult::HAVE_MORE_OUTPUT;

        return {std::shared_ptr<table_info>(batch), result};
    }
};
