#pragma once

#include "../libs/_table_builder.h"

#include <arrow/filesystem/filesystem.h>
#include <arrow/python/api.h>
#include "../../io/iceberg_parquet_write.h"
#include "../../libs/streaming/_shuffle.h"
#include "_bodo_write_function.h"
#include "physical/operator.h"

struct PhysicalWriteIcebergMetrics {
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;

    stat_t max_buffer_size = 0;
    stat_t n_files_written = 0;

    time_t accumulate_time = 0;
    time_t file_write_time = 0;
    time_t finalize_time = 0;
};

class PhysicalWriteIceberg : public PhysicalSink {
   public:
    explicit PhysicalWriteIceberg(std::shared_ptr<bodo::Schema> in_bodo_schema,
                                  IcebergWriteFunctionData& bind_data)
        : in_schema(std::move(bind_data.in_schema)),
          table_loc(std::move(bind_data.table_loc)),
          bucket_region(std::move(bind_data.bucket_region)),
          max_pq_chunksize(bind_data.max_pq_chunksize),
          compression(std::move(bind_data.compression)),
          partition_tuples(bind_data.partition_tuples),
          sort_tuples(bind_data.sort_tuples),
          iceberg_schema_str(std::move(bind_data.iceberg_schema_str)),
          iceberg_schema(std::move(bind_data.iceberg_schema)),
          fs(std::move(bind_data.fs)),
          is_last_state(std::make_shared<IsLastState>()),
          finished(false) {
        // Similar to streaming Iceberg write in Bodo JIT
        // https://github.com/bodo-ai/Bodo/blob/71cbb2db57a9f9c67e13eb3e49222b3ca90ede83/bodo/io/iceberg/stream_iceberg_write.py#L558
        Py_INCREF(partition_tuples);
        Py_INCREF(sort_tuples);

        // Initialize the buffer and dictionary builders
        for (auto& col : in_bodo_schema->column_types) {
            dict_builders.emplace_back(
                create_dict_builder_for_array(col->copy(), false));
        }
        buffer =
            std::make_shared<TableBuildBuffer>(in_bodo_schema, dict_builders);
    }

    virtual ~PhysicalWriteIceberg() {
        Py_DECREF(partition_tuples);
        Py_DECREF(sort_tuples);
    };

    OperatorResult ConsumeBatch(std::shared_ptr<table_info> input_batch,
                                OperatorResult prev_op_result) override {
        // Similar to streaming parquet write in Bodo JIT
        // https://github.com/bodo-ai/Bodo/blob/71cbb2db57a9f9c67e13eb3e49222b3ca90ede83/bodo/io/iceberg/stream_iceberg_write.py#L744

        if (finished) {
            return OperatorResult::FINISHED;
        }

        // ===== Part 1: Accumulate batch in writer and compute total size =====
        time_pt start_accumulate_time = start_timer();
        buffer->UnifyTablesAndAppend(input_batch, dict_builders);
        int64_t buffer_nbytes =
            table_local_memory_size(buffer->data_table, true);
        this->metrics.accumulate_time += end_timer(start_accumulate_time);
        this->metrics.max_buffer_size =
            std::max(this->metrics.max_buffer_size, buffer_nbytes);

        // Sync is_last flag
        bool is_last = prev_op_result == OperatorResult::FINISHED;
        is_last = static_cast<bool>(sync_is_last_non_blocking(
            is_last_state.get(), static_cast<int32_t>(is_last)));

        // === Part 2: Write Parquet file if file size threshold is exceeded ===
        time_pt start_file_write_time = start_timer();
        if (is_last || buffer_nbytes >= this->max_pq_chunksize) {
            std::shared_ptr<table_info> data = buffer->data_table;

            if (data->nrows() > 0) {
                iceberg_pq_write(
                    table_loc.c_str(), data, in_schema->field_names(),
                    partition_tuples, sort_tuples, compression.c_str(), false,
                    bucket_region.c_str(), -1, iceberg_schema_str.c_str(),
                    iceberg_files_info_py, iceberg_schema, fs, nullptr);
                this->metrics.n_files_written++;
            }
            // Reset the buffer for the next batch
            buffer->Reset();
        }
        this->metrics.file_write_time += end_timer(start_file_write_time);

        if (is_last) {
            finished = true;
        }

        iter++;
        return is_last ? OperatorResult::FINISHED
                       : OperatorResult::NEED_MORE_INPUT;
    }

    void FinalizeSink() override {
        // Gather iceberg_files_info from all ranks using MPI
        // Equivalent to all_infos = comm.gather(iceberg_files_info_py)

        time_pt start_finalize_time = start_timer();
        PyObject* mpi4py_module = PyImport_ImportModule("mpi4py");
        if (!mpi4py_module) {
            throw std::runtime_error(
                "PhysicalWriteIceberg::Finalize: Failed to import "
                "mpi4py");
        }
        PyObject* mpi_module = PyObject_GetAttrString(mpi4py_module, "MPI");
        if (!mpi_module) {
            Py_DECREF(mpi4py_module);
            throw std::runtime_error(
                "PhysicalWriteIceberg::Finalize: Failed to get MPI module");
        }

        PyObject* comm_world = PyObject_GetAttrString(mpi_module, "COMM_WORLD");
        if (!comm_world) {
            Py_DECREF(mpi_module);
            throw std::runtime_error(
                "PhysicalWriteIceberg::Finalize: Failed to get COMM_WORLD");
        }

        PyObject* gather_method = PyObject_GetAttrString(comm_world, "gather");
        if (!gather_method) {
            Py_DECREF(comm_world);
            Py_DECREF(mpi_module);
            throw std::runtime_error(
                "PhysicalWriteIceberg::Finalize: Failed to get gather method");
        }

        PyObject* args = PyTuple_New(1);
        PyTuple_SetItem(args, 0, iceberg_files_info_py);

        PyObject* all_infos = PyObject_CallObject(gather_method, args);

        Py_DECREF(args);
        Py_DECREF(gather_method);
        Py_DECREF(comm_world);
        Py_DECREF(mpi4py_module);
        Py_DECREF(mpi_module);

        if (!all_infos) {
            throw std::runtime_error(
                "PhysicalWriteIceberg::Finalize: Iceberg write: MPI gather "
                "operation failed");
        }
        this->metrics.finalize_time = end_timer(start_finalize_time);

        // NOTE: PyTuple_SetItem stole the reference to iceberg_files_info_py
        iceberg_files_info_py = all_infos;

        // Report metrics
        std::vector<MetricBase> metrics_out;
        this->ReportMetrics(metrics_out);
        QueryProfileCollector::Default().RegisterOperatorStageMetrics(
            QueryProfileCollector::MakeOperatorStageID(getOpId(), 1),
            std::move(metrics_out));
        // Write doesn't produce rows
        QueryProfileCollector::Default().SubmitOperatorStageRowCounts(
            QueryProfileCollector::MakeOperatorStageID(getOpId(), 1), 0);
    }

    std::variant<std::shared_ptr<table_info>, PyObject*> GetResult() override {
        return iceberg_files_info_py;
    }

   private:
    // State similar to streaming parquet write in Bodo JIT
    // https://github.com/bodo-ai/Bodo/blob/1ab0dedf37f7b2ae8551bb95a1e3d4cfc70c553f/bodo/io/stream_parquet_write.py#L289
    const std::shared_ptr<arrow::Schema> in_schema;
    const std::string table_loc;
    const std::string bucket_region;
    const int64_t max_pq_chunksize;
    const std::string compression;
    PyObject* partition_tuples;
    PyObject* sort_tuples;
    const std::string iceberg_schema_str;
    const std::shared_ptr<arrow::Schema> iceberg_schema;
    std::shared_ptr<arrow::fs::FileSystem> fs;

    PyObject* iceberg_files_info_py = PyList_New(0);

    std::shared_ptr<TableBuildBuffer> buffer;
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;
    const std::shared_ptr<IsLastState> is_last_state;
    bool finished = false;
    int64_t iter = 0;
    PhysicalWriteIcebergMetrics metrics;

    void ReportMetrics(std::vector<MetricBase>& metrics_out) {
        metrics_out.emplace_back(
            StatMetric("max_buffer_size", this->metrics.max_buffer_size));
        metrics_out.emplace_back(
            StatMetric("n_files_written", this->metrics.n_files_written));
        metrics_out.emplace_back(
            TimerMetric("accumulate_time", this->metrics.accumulate_time));
        metrics_out.emplace_back(
            TimerMetric("file_write_time", this->metrics.file_write_time));
        metrics_out.emplace_back(
            TimerMetric("finalize_time", this->metrics.finalize_time));

        // Add the dict builder metrics if they exist
        for (size_t i = 0; i < this->dict_builders.size(); ++i) {
            auto dict_builder = this->dict_builders[i];
            if (dict_builder) {
                dict_builder->GetMetrics().add_to_metrics(
                    metrics_out, fmt::format("dict_builder_{}", i));
            }
        }
    }
};
