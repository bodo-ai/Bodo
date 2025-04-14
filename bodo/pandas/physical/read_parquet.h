#pragma once

#include <memory>
#include <utility>
#include "../io/parquet_reader.h"
#include "operator.h"

/// @brief Physical node for reading Parquet files in pipelines.
class PhysicalReadParquet : public PhysicalSource {
   private:
    std::shared_ptr<ParquetReader> internal_reader;

   public:
    // TODO: Fill in the contents with info from the logical operator
    explicit PhysicalReadParquet(std::string _path, PyObject *pyarrow_schema,
                                 PyObject *storage_options,
                                 std::vector<int> &selected_columns) {
        PyObject *py_path = PyUnicode_FromString(_path.c_str());

        std::vector<bool> is_nullable(selected_columns.size(), true);

        internal_reader = std::make_shared<ParquetReader>(
            py_path, true, Py_None, storage_options, pyarrow_schema, -1,
            selected_columns, is_nullable, false, -1);
        internal_reader->init_pq_reader({}, nullptr, nullptr, 0);
    }
    virtual ~PhysicalReadParquet() = default;

    void Finalize() override {}

    std::pair<std::shared_ptr<table_info>, ProducerResult> ProduceBatch()
        override {
        uint64_t total_rows;
        bool is_last;

        auto batch = internal_reader->read_batch(is_last, total_rows, true);
        auto result = is_last ? ProducerResult::FINISHED
                              : ProducerResult::HAVE_MORE_OUTPUT;

        return std::make_pair(std::shared_ptr<table_info>(batch), result);
    }
};
