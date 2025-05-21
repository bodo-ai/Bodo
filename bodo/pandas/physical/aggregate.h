#pragma once

#include <memory>
#include <utility>
#include "../libs/_array_utils.h"
#include "../libs/_utils.h"
#include "expression.h"
#include "operator.h"
#include "../io/arrow_reader.h"

/**
 * @brief Physical node for count_star().
 *
 */
class PhysicalCountStar : public PhysicalSource, public PhysicalSink {
   public:
    explicit PhysicalCountStar() : local_count(0), global_count(0)
{
        std::vector<std::unique_ptr<bodo::DataType>> types;
        types.emplace_back(std::make_unique<bodo::DataType>(
            bodo_array_type::arr_type_enum::NULLABLE_INT_BOOL,
            Bodo_CTypes::CTypeEnum::UINT64));
        std::vector<std::string> names = {std::string("count_star()")};
        out_schema = std::make_shared<bodo::Schema>(std::move(types), names);

        // Import pandas and pyarrow
        PyObject *pandas = PyImport_ImportModule("pandas");
        PyObject *pyarrow = PyImport_ImportModule("pyarrow");

        // Create an empty DataFrame with a uint64 column
        PyObject *dict = PyDict_New();
        PyObject *series_class = PyObject_GetAttrString(pandas, "Series");
        PyObject *empty_series = PyObject_CallFunction(series_class, "(O)", PyUnicode_FromString("uint64"));
        PyDict_SetItemString(dict, "count_star()", empty_series);

        PyObject *df_class = PyObject_GetAttrString(pandas, "DataFrame");
        PyObject *df = PyObject_CallFunctionObjArgs(df_class, dict, NULL);

        // Convert to PyArrow Table
        PyObject *table_class = PyObject_GetAttrString(pyarrow, "Table");
        PyObject *from_pandas = PyObject_GetAttrString(table_class, "from_pandas");
        PyObject *table = PyObject_CallFunctionObjArgs(from_pandas, df, NULL);

        PyObject *pyarrow_schema = PyObject_GetAttrString(table, "schema");
        std::shared_ptr<arrow::Schema> arrow_schema =
            unwrap_schema(pyarrow_schema);

        out_schema->metadata =
            std::make_shared<TableMetadata>(arrow_schema->metadata()->keys(),
                                            arrow_schema->metadata()->values());
        Py_DECREF(pandas);
        Py_DECREF(pyarrow);
        Py_DECREF(dict);
        Py_DECREF(series_class);
        Py_DECREF(empty_series);
        Py_DECREF(df_class);
        Py_DECREF(df);
        Py_DECREF(table_class);
        Py_DECREF(from_pandas);
        Py_DECREF(pyarrow_schema);
    }

    virtual ~PhysicalCountStar() = default;

    void Finalize() override {
        int result =
            MPI_Allreduce(&local_count, &global_count, 1,
                          MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        if (result != MPI_SUCCESS) {
            throw std::runtime_error(
                "PhysicalCountStar::Finalize MPI_Allreduce failed.");
        }
    }

    OperatorResult ConsumeBatch(std::shared_ptr<table_info> input_batch,
                                OperatorResult prev_op_result) override {
        local_count += input_batch->nrows();
        return prev_op_result == OperatorResult::FINISHED
                   ? OperatorResult::FINISHED
                   : OperatorResult::NEED_MORE_INPUT;
    }

    std::shared_ptr<table_info> GetResult() override {
        throw std::runtime_error(
            "GetResult called on a PhysicalCountStar node.");
    }

    const std::shared_ptr<bodo::Schema> getOutputSchema() override {
        return out_schema;
    }

    std::pair<std::shared_ptr<table_info>, OperatorResult> ProduceBatch()
        override {
        std::shared_ptr<arrow::Array> array =
            CreateOneElementArrowArray(global_count);

        std::shared_ptr<array_info> result =
            arrow_array_to_bodo(array, bodo::BufferPool::DefaultPtr());
        std::vector<std::shared_ptr<array_info>> cvec = {result};
        std::shared_ptr<table_info> next_batch =
            std::make_shared<table_info>(cvec);
        next_batch->metadata = out_schema->metadata;
        return {next_batch, OperatorResult::FINISHED};
    }

   private:
    uint64_t local_count, global_count;
    std::shared_ptr<bodo::Schema> out_schema;
};
