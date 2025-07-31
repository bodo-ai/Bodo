#pragma once

#include <string>
#include "../../libs/streaming/_shuffle.h"
#include "_bodo_write_function.h"
#include "physical/operator.h"

class PhysicalWriteS3Vectors : public PhysicalSink {
   public:
    explicit PhysicalWriteS3Vectors(
        std::shared_ptr<bodo::Schema> in_bodo_schema,
        S3VectorsWriteFunctionData& bind_data)
        : vector_bucket_name(std::move(bind_data.vector_bucket_name)),
          index_name(std::move(bind_data.index_name)),
          region(bind_data.region),
          is_last_state(std::make_shared<IsLastState>()),
          finished(false) {}

    virtual ~PhysicalWriteS3Vectors() = default;

    OperatorResult ConsumeBatch(std::shared_ptr<table_info> input_batch,
                                OperatorResult prev_op_result) override {
        if (finished) {
            return OperatorResult::FINISHED;
        }

        this->write_batch(input_batch);

        // Sync is_last flag
        bool is_last = prev_op_result == OperatorResult::FINISHED;
        is_last = static_cast<bool>(sync_is_last_non_blocking(
            is_last_state.get(), static_cast<int32_t>(is_last)));

        if (is_last) {
            finished = true;
        }

        iter++;
        return is_last ? OperatorResult::FINISHED
                       : OperatorResult::NEED_MORE_INPUT;
    }

    void Finalize() override {}

    std::variant<std::shared_ptr<table_info>, PyObject*> GetResult() override {
        return std::shared_ptr<table_info>(nullptr);
    }

   private:
    /**
     * @brief Write a batch of vectors to S3 using the
     * bodo.pandas.utils.write_s3_vectors_helper function.
     *
     * @param input_batch input data to write to S3 Vectors.
     */
    void write_batch(std::shared_ptr<table_info> input_batch) {
        // Call bodo.pandas.utils.write_s3_vectors_helper()

        // Import the bodo.pandas.utils module
        PyObject* bodo_module = PyImport_ImportModule("bodo.pandas.utils");
        if (!bodo_module) {
            PyErr_Print();
            throw std::runtime_error(
                "Failed to import bodo.pandas.utils module");
        }

        // Call the write_s3_vectors_helper() with the table_info pointer and
        // vector/index names
        PyObject* vector_bucket_name_py =
            PyUnicode_FromString(vector_bucket_name.c_str());
        PyObject* index_name_py = PyUnicode_FromString(index_name.c_str());
        if (!vector_bucket_name_py || !index_name_py) {
            Py_DECREF(bodo_module);
            throw std::runtime_error(
                "Failed to create Python strings for bucket and index names");
        }
        PyObject* result = PyObject_CallMethod(
            bodo_module, "write_s3_vectors_helper", "LOOO",
            reinterpret_cast<int64_t>(new table_info(*input_batch)),
            vector_bucket_name_py, index_name_py, region);
        if (!result) {
            PyErr_Print();
            Py_DECREF(bodo_module);
            throw std::runtime_error("Error calling write_s3_vectors_helper");
        }

        // Clean up Python objects
        Py_DECREF(vector_bucket_name_py);
        Py_DECREF(index_name_py);
        Py_DECREF(bodo_module);
        Py_DECREF(result);
    }

    std::string vector_bucket_name;
    std::string index_name;
    PyObject* region;

    const std::shared_ptr<IsLastState> is_last_state;
    bool finished = false;
    int64_t iter = 0;
};
