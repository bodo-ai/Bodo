#pragma once

#include <utility>
#include "_util.h"

#include "duckdb/function/function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/planner/bound_result_modifier.hpp"
#include "fmt/core.h"
#include "physical/operator.h"

/**
 * @brief Superclass for Bodo's DuckDB TableFunction classes.
 *
 */
class BodoScanFunction : public duckdb::TableFunction {
   public:
    BodoScanFunction(std::string name)
        : TableFunction(name, {}, nullptr, nullptr, nullptr, nullptr) {}
};

/**
 * @brief Superclass for Bodo's DuckDB TableFunctionData classes.
 *
 */
class BodoScanFunctionData : public duckdb::TableFunctionData {
   public:
    BodoScanFunctionData() = default;
    /**
     * @brief Create a PhysicalOperator for reading data from this source.
     *
     * @return std::shared_ptr<PhysicalSource> read operator
     */
    virtual std::shared_ptr<PhysicalSource> CreatePhysicalOperator(
        std::vector<int> &selected_columns,
        duckdb::TableFilterSet &filter_exprs,
        duckdb::unique_ptr<duckdb::BoundLimitNode> &limit_val) = 0;
};

/**
 * @brief Bodo's DuckDB TableFunction for reading Parquet datasets with Bodo
 * metadata (used in LogicalGet).
 *
 */
class BodoParquetScanFunction : public BodoScanFunction {
   public:
    BodoParquetScanFunction(const std::shared_ptr<arrow::Schema> arrow_schema)

        : BodoScanFunction(
              fmt::format("bodo_read_parquet({})",
                          schemaColumnNamesToString(arrow_schema))) {
        filter_pushdown = true;
        filter_prune = true;
        projection_pushdown = true;
        limit_pushdown = true;
        // TODO: set statistics and other optimization flags as needed
        // https://github.com/duckdb/duckdb/blob/d29a92f371179170688b4df394478f389bf7d1a6/src/include/duckdb/function/table_function.hpp#L357
    }
};

/**
 * @brief Data for Bodo's DuckDB TableFunction for reading Parquet datasets.
 *
 */
class BodoParquetScanFunctionData : public BodoScanFunctionData {
   public:
    BodoParquetScanFunctionData(PyObject *path, PyObject *pyarrow_schema,
                                PyObject *storage_options)
        : path(path),
          pyarrow_schema(pyarrow_schema),
          storage_options(storage_options) {
        Py_INCREF(pyarrow_schema);
        Py_INCREF(storage_options);
        Py_INCREF(path);
    }

    ~BodoParquetScanFunctionData() override {
        Py_DECREF(pyarrow_schema);
        Py_DECREF(storage_options);
        Py_DECREF(path);
    }

    std::shared_ptr<PhysicalSource> CreatePhysicalOperator(
        std::vector<int> &selected_columns,
        duckdb::TableFilterSet &filter_exprs,
        duckdb::unique_ptr<duckdb::BoundLimitNode> &limit_val) override;

    // Parquet dataset path
    PyObject *path;
    PyObject *pyarrow_schema;
    PyObject *storage_options;
};

/**
 * @brief Bodo's DuckDB TableFunction for reading dataframe rows
 * (used in LogicalGet).
 *
 */

class BodoDataFrameScanFunction : public BodoScanFunction {
   public:
    BodoDataFrameScanFunction(const std::shared_ptr<arrow::Schema> arrow_schema)
        : BodoScanFunction(fmt::format(

              "bodo_read_df({})", schemaColumnNamesToString(arrow_schema))) {
        projection_pushdown = true;
    }
};

/**
 * @brief Data for Bodo's DuckDB TableFunction for reading dataframe rows on
 * spawner sequentially.
 *
 */
class BodoDataFrameSeqScanFunctionData : public BodoScanFunctionData {
   public:
    BodoDataFrameSeqScanFunctionData(
        PyObject *df, std::shared_ptr<arrow::Schema> arrow_schema)
        : df(df), arrow_schema(std::move(arrow_schema)) {
        Py_INCREF(df);
    }
    ~BodoDataFrameSeqScanFunctionData() override { Py_DECREF(df); }
    /**
     * @brief Create a PhysicalOperator for reading from the dataframe.
     *
     * @return std::shared_ptr<PhysicalOperator> dataframe read operator
     */
    std::shared_ptr<PhysicalSource> CreatePhysicalOperator(
        std::vector<int> &selected_columns,
        duckdb::TableFilterSet &filter_exprs,
        duckdb::unique_ptr<duckdb::BoundLimitNode> &limit_val) override;

    PyObject *df;
    const std::shared_ptr<arrow::Schema> arrow_schema;
};

/**
 * @brief Data for Bodo's DuckDB TableFunction for reading dataframe rows on
 * workers in parallel.
 *
 */
class BodoDataFrameParallelScanFunctionData : public BodoScanFunctionData {
   public:
    BodoDataFrameParallelScanFunctionData(
        std::string result_id, std::shared_ptr<arrow::Schema> arrow_schema)
        : result_id(std::move(result_id)),
          arrow_schema(std::move(arrow_schema)) {}
    ~BodoDataFrameParallelScanFunctionData() override = default;
    /**
     * @brief Create a PhysicalOperator for reading from the dataframe.
     *
     * @return std::shared_ptr<PhysicalOperator> dataframe read operator
     */
    std::shared_ptr<PhysicalSource> CreatePhysicalOperator(
        std::vector<int> &selected_columns,
        duckdb::TableFilterSet &filter_exprs,
        duckdb::unique_ptr<duckdb::BoundLimitNode> &limit_val) override;
    std::string result_id;
    const std::shared_ptr<arrow::Schema> arrow_schema;
};

/**
 * @brief Bodo's DuckDB TableFunction for reading Iceberg datasets with Bodo
 * metadata (used in LogicalGet).
 *
 */
class BodoIcebergScanFunction : public BodoScanFunction {
   public:
    BodoIcebergScanFunction(const std::shared_ptr<arrow::Schema> arrow_schema)
        : BodoScanFunction(
              fmt::format("bodo_read_iceberg({})",
                          schemaColumnNamesToString(arrow_schema))) {
        filter_pushdown = true;
        filter_prune = true;
        projection_pushdown = true;
        limit_pushdown = true;
        // TODO: set statistics and other optimization flags as needed
        // https://github.com/duckdb/duckdb/blob/d29a92f371179170688b4df394478f389bf7d1a6/src/include/duckdb/function/table_function.hpp#L357
    }
};

/**
 * @brief Data for Bodo's DuckDB TableFunction for reading Iceberg datasets.
 *
 */
class BodoIcebergScanFunctionData : public BodoScanFunctionData {
   public:
    BodoIcebergScanFunctionData(std::shared_ptr<arrow::Schema> _arrow_schema,
                                PyObject *_catalog, const std::string _table_id,
                                PyObject *_iceberg_filter,
                                PyObject *_iceberg_schema, int64_t _snapshot_id)
        : arrow_schema(std::move(_arrow_schema)),
          catalog(_catalog),
          iceberg_filter(_iceberg_filter),
          iceberg_schema(_iceberg_schema),
          table_id(_table_id),
          snapshot_id(_snapshot_id) {
        Py_INCREF(this->catalog);
        Py_INCREF(this->iceberg_filter);
        Py_INCREF(this->iceberg_schema);
    };

    ~BodoIcebergScanFunctionData() override {
        Py_DECREF(this->catalog);
        Py_DECREF(this->iceberg_filter);
        Py_DECREF(this->iceberg_schema);
    };

    std::shared_ptr<PhysicalSource> CreatePhysicalOperator(
        std::vector<int> &selected_columns,
        duckdb::TableFilterSet &filter_exprs,
        duckdb::unique_ptr<duckdb::BoundLimitNode> &limit_val) override;
    const std::shared_ptr<arrow::Schema> arrow_schema;
    PyObject *catalog;
    PyObject *iceberg_filter;
    PyObject *iceberg_schema;
    const std::string table_id;
    const int64_t snapshot_id;
};

/**
 * @brief Data for writing Parquet datasets
 *
 */
struct ParquetWriteFunctionData : public duckdb::FunctionData {
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
struct IcebergWriteFunctionData : public duckdb::FunctionData {
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
