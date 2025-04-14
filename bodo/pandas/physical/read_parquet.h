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

        // Extract column names from pyarrow schema using selected columns
        PyObject *schema_fields =
            PyObject_GetAttrString(pyarrow_schema, "names");
        if (!schema_fields || !PyList_Check(schema_fields)) {
            throw std::runtime_error(
                "PhysicalReadParquet(): failed to get schema fields from "
                "pyarrow schema");
        }
        int num_fields = PyList_Size(schema_fields);
        out_column_names.reserve(selected_columns.size());

        for (int col_idx : selected_columns) {
            if (!(col_idx >= 0 && col_idx < num_fields)) {
                throw std::runtime_error(
                    "PhysicalReadParquet(): invalid column index " +
                    std::to_string(col_idx) + " for schema with " +
                    std::to_string(num_fields) + " fields");
            }
            PyObject *name = PyList_GetItem(schema_fields, col_idx);
            if (name && PyUnicode_Check(name)) {
                out_column_names.push_back(PyUnicode_AsUTF8(name));
            } else {
                out_column_names.push_back("column_" + std::to_string(col_idx));
            }
        }

        Py_DECREF(schema_fields);
    }
    virtual ~PhysicalReadParquet() = default;

    void Finalize() override {}

    std::pair<std::shared_ptr<table_info>, ProducerResult> ProduceBatch()
        override {
        uint64_t total_rows;
        bool is_last;

        table_info *batch =
            internal_reader->read_batch(is_last, total_rows, true);
        auto result = is_last ? ProducerResult::FINISHED
                              : ProducerResult::HAVE_MORE_OUTPUT;

        batch->column_names = out_column_names;
        return std::make_pair(std::shared_ptr<table_info>(batch), result);
    }

    std::vector<std::string> out_column_names;
};
