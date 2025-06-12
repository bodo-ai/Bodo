
#pragma once

#include <utility>
#include "_util.h"

#include "duckdb/function/function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/planner/bound_result_modifier.hpp"
#include "fmt/core.h"
#include "physical/operator.h"

/**
 * @brief Superclass for Bodo's DuckDB FunctionData classes for write.
 *
 */
class BodoWriteFunctionData : public duckdb::FunctionData {
   public:
    BodoWriteFunctionData() = default;
    /**
     * @brief Create a PhysicalOperator for writing data.
     *
     * @return std::shared_ptr<PhysicalSink> write operator
     */
    virtual std::shared_ptr<PhysicalSink> CreatePhysicalOperator(
        std::shared_ptr<bodo::Schema> in_table_schema) = 0;
};

/**
 * @brief Data for writing Parquet datasets
 *
 */
struct ParquetWriteFunctionData : public BodoWriteFunctionData {
    ParquetWriteFunctionData(std::string path,
                             std::shared_ptr<arrow::Schema> arrow_schema,
                             std::string compression, std::string bucket_region,
                             int64_t row_group_size)
        : path(std::move(path)),
          arrow_schema(std::move(arrow_schema)),
          compression(std::move(compression)),
          bucket_region(std::move(bucket_region)),
          row_group_size(row_group_size) {}

    bool Equals(const FunctionData &other_p) const override {
        const ParquetWriteFunctionData &other =
            other_p.Cast<ParquetWriteFunctionData>();
        return (other.path == this->path &&
                other.arrow_schema->Equals(this->arrow_schema) &&
                other.compression == this->compression &&
                other.bucket_region == this->bucket_region &&
                other.row_group_size == this->row_group_size);
    }

    duckdb::unique_ptr<duckdb::FunctionData> Copy() const override {
        return duckdb::make_uniq<ParquetWriteFunctionData>(
            this->path, this->arrow_schema, this->compression,
            this->bucket_region, this->row_group_size);
    }

    std::shared_ptr<PhysicalSink> CreatePhysicalOperator(
        std::shared_ptr<bodo::Schema> in_table_schema) override {
        return nullptr;  // std::make_shared<PhysicalWriteParquet>(in_table_schema,
                         // *this);
    }

    std::string path;
    std::shared_ptr<arrow::Schema> arrow_schema;
    std::string compression;
    std::string bucket_region;
    int64_t row_group_size;
};

/**
 * @brief Data for writing Iceberg datasets
 *
 */
struct IcebergWriteFunctionData : public BodoWriteFunctionData {
    IcebergWriteFunctionData(std::string table_loc, std::string bucket_region,
                             int64_t max_pq_chunksize, std::string compression,
                             PyObject *partition_tuples, PyObject *sort_tuples,
                             std::string iceberg_schema_str,
                             PyObject *output_pa_schema, PyObject *fs)
        : table_loc(std::move(table_loc)),
          bucket_region(std::move(bucket_region)),
          max_pq_chunksize(max_pq_chunksize),
          compression(std::move(compression)),
          partition_tuples(partition_tuples),
          sort_tuples(sort_tuples),
          iceberg_schema_str(std::move(iceberg_schema_str)),
          output_pa_schema(output_pa_schema),
          fs(fs) {
        Py_INCREF(partition_tuples);
        Py_INCREF(sort_tuples);
        Py_INCREF(output_pa_schema);
        Py_INCREF(fs);
    }

    ~IcebergWriteFunctionData() override {
        Py_DECREF(partition_tuples);
        Py_DECREF(sort_tuples);
        Py_DECREF(output_pa_schema);
        Py_DECREF(fs);
    }

    bool Equals(const FunctionData &other_p) const override {
        const IcebergWriteFunctionData &other =
            other_p.Cast<IcebergWriteFunctionData>();
        return (other.table_loc == this->table_loc &&
                other.bucket_region == this->bucket_region &&
                other.max_pq_chunksize == this->max_pq_chunksize &&
                other.compression == this->compression &&
                PyObject_RichCompareBool(other.partition_tuples,
                                         this->partition_tuples, Py_EQ) &&
                PyObject_RichCompareBool(other.sort_tuples, this->sort_tuples,
                                         Py_EQ) &&
                other.iceberg_schema_str == this->iceberg_schema_str &&
                PyObject_RichCompareBool(other.output_pa_schema,
                                         this->output_pa_schema, Py_EQ) &&
                PyObject_RichCompareBool(other.fs, this->fs, Py_EQ));
    }

    duckdb::unique_ptr<duckdb::FunctionData> Copy() const override {
        return duckdb::make_uniq<IcebergWriteFunctionData>(
            this->table_loc, this->bucket_region, this->max_pq_chunksize,
            this->compression, this->partition_tuples, this->sort_tuples,
            this->iceberg_schema_str, this->output_pa_schema, this->fs);
    }

    std::shared_ptr<PhysicalSink> CreatePhysicalOperator(
        std::shared_ptr<bodo::Schema> in_table_schema) override {
        return nullptr;  // std::make_shared<PhysicalWriteIceberg>(in_table_schema,
                         // *this);
    }

    std::string table_loc;
    std::string bucket_region;
    int64_t max_pq_chunksize;
    std::string compression;
    PyObject *partition_tuples;
    PyObject *sort_tuples;
    std::string iceberg_schema_str;
    PyObject *output_pa_schema;
    PyObject *fs;
};
