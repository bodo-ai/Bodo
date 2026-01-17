
#pragma once

#include <utility>
#include "_util.h"

#include <arrow/filesystem/filesystem.h>
#include <arrow/python/api.h>
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
    virtual std::variant<std::shared_ptr<PhysicalSink>,
                         std::shared_ptr<PhysicalGPUSink>>
    CreatePhysicalOperator(std::shared_ptr<bodo::Schema> in_table_schema,
                           bool run_on_gpu) = 0;
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

    std::variant<std::shared_ptr<PhysicalSink>,
                 std::shared_ptr<PhysicalGPUSink>>
    CreatePhysicalOperator(std::shared_ptr<bodo::Schema> in_table_schema,
                           bool run_on_gpu) override;

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
    IcebergWriteFunctionData(std::shared_ptr<arrow::Schema> arrow_schema,
                             std::string table_loc, std::string bucket_region,
                             int64_t max_pq_chunksize, std::string compression,
                             PyObject *partition_tuples, PyObject *sort_tuples,
                             std::string iceberg_schema_str,
                             std::shared_ptr<arrow::Schema> iceberg_schema,
                             std::shared_ptr<arrow::fs::FileSystem> fs)
        : in_schema(arrow_schema),
          table_loc(std::move(table_loc)),
          bucket_region(std::move(bucket_region)),
          max_pq_chunksize(max_pq_chunksize),
          compression(std::move(compression)),
          partition_tuples(partition_tuples),
          sort_tuples(sort_tuples),
          iceberg_schema_str(std::move(iceberg_schema_str)),
          iceberg_schema(iceberg_schema),
          fs(fs) {
        Py_INCREF(partition_tuples);
        Py_INCREF(sort_tuples);
    }

    ~IcebergWriteFunctionData() override {
        Py_DECREF(partition_tuples);
        Py_DECREF(sort_tuples);
    }

    bool Equals(const FunctionData &other_p) const override {
        const IcebergWriteFunctionData &other =
            other_p.Cast<IcebergWriteFunctionData>();
        return (other.in_schema->Equals(this->in_schema) &&
                other.table_loc == this->table_loc &&
                other.bucket_region == this->bucket_region &&
                other.max_pq_chunksize == this->max_pq_chunksize &&
                other.compression == this->compression &&
                PyObject_RichCompareBool(other.partition_tuples,
                                         this->partition_tuples, Py_EQ) &&
                PyObject_RichCompareBool(other.sort_tuples, this->sort_tuples,
                                         Py_EQ) &&
                other.iceberg_schema_str == this->iceberg_schema_str &&
                other.iceberg_schema->Equals(this->iceberg_schema) &&
                other.fs->Equals(this->fs));
    }

    duckdb::unique_ptr<duckdb::FunctionData> Copy() const override {
        return duckdb::make_uniq<IcebergWriteFunctionData>(
            this->in_schema, this->table_loc, this->bucket_region,
            this->max_pq_chunksize, this->compression, this->partition_tuples,
            this->sort_tuples, this->iceberg_schema_str, this->iceberg_schema,
            this->fs);
    }

    std::variant<std::shared_ptr<PhysicalSink>,
                 std::shared_ptr<PhysicalGPUSink>>
    CreatePhysicalOperator(std::shared_ptr<bodo::Schema> in_table_schema,
                           bool run_on_gpu) override;

    std::shared_ptr<arrow::Schema> in_schema;
    std::string table_loc;
    std::string bucket_region;
    int64_t max_pq_chunksize;
    std::string compression;
    PyObject *partition_tuples;
    PyObject *sort_tuples;
    std::string iceberg_schema_str;
    std::shared_ptr<arrow::Schema> iceberg_schema;
    std::shared_ptr<arrow::fs::FileSystem> fs;
};

/**
 * @brief Data for writing S3 Vectors
 *
 */
struct S3VectorsWriteFunctionData : public BodoWriteFunctionData {
    S3VectorsWriteFunctionData(std::string vector_bucket_name,
                               std::string index_name, PyObject *region)
        : vector_bucket_name(std::move(vector_bucket_name)),
          index_name(std::move(index_name)),
          region(region) {}

    ~S3VectorsWriteFunctionData() override = default;

    bool Equals(const FunctionData &other_p) const override {
        const S3VectorsWriteFunctionData &other =
            other_p.Cast<S3VectorsWriteFunctionData>();
        return (other.vector_bucket_name == this->vector_bucket_name &&
                other.index_name == this->index_name &&
                other.region == this->region);
    }

    duckdb::unique_ptr<duckdb::FunctionData> Copy() const override {
        return duckdb::make_uniq<S3VectorsWriteFunctionData>(
            this->vector_bucket_name, this->index_name, this->region);
    }

    std::variant<std::shared_ptr<PhysicalSink>,
                 std::shared_ptr<PhysicalGPUSink>>
    CreatePhysicalOperator(std::shared_ptr<bodo::Schema> in_table_schema,
                           bool run_on_gpu) override;

    std::string vector_bucket_name;
    std::string index_name;
    PyObject *region;
};
