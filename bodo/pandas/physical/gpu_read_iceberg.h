#pragma once

#include <Python.h>
#include <arrow/filesystem/filesystem.h>
#include <mpi.h>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../../libs/_utils.h"
#include "../../libs/gpu_utils.h"
#include "../_util.h"
#include "duckdb/planner/bound_result_modifier.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "gpu_expression.h"
#include "operator.h"

/**
 * @brief Batch generator for Iceberg on GPU.
 */
class GPUIcebergRankBatchGenerator {
   public:
    /**
     * @brief Initialize the generator by fetching the Iceberg dataset,
     * distributing pieces across GPU ranks, and initializing per-schema-group
     * read schemas and field-id mappings.
     *
     * On non-GPU ranks, only the dataset is fetched (to participate in
     * collective MPI operations); pieces and scanners are skipped.
     */
    GPUIcebergRankBatchGenerator(
        PyObject* catalog, const std::string& table_id,
        PyObject* iceberg_filter, PyObject* iceberg_schema,
        const std::shared_ptr<arrow::Schema>& arrow_schema,
        const int64_t snapshot_id, const std::vector<int>& selected_columns,
        duckdb::unique_ptr<duckdb::TableFilterSet> filter_exprs,
        std::size_t target_rows, MPI_Comm comm,
        std::shared_ptr<arrow::Schema> output_arrow_schema);

    /**
     * @brief Produce the next batch of up to target_rows_ rows.
     *
     * Reads from individual Parquet files via chunked readers, applies
     * DuckDB and Iceberg filters, evolves columns to match the output
     * schema, and concatenates chunks to hit the target row count.
     *
     * @return A pair of (table, finished). If no more data is available,
     *         returns an empty table with the output schema and finished=true.
     */
    std::pair<std::unique_ptr<cudf::table>, bool> next(
        std::shared_ptr<StreamAndEvent> se);

    /**
     * @brief Report per-operator metrics.
     */
    void ReportMetrics(std::vector<MetricBase>& metrics_out);

   private:
    void init_next_reader(rmm::cuda_stream_view stream);

    /**
     * @brief Reconstruct a column from released contents after child
     * evolution. Shares the null-mask + column-construction boilerplate
     * between list and map evolution paths.
     */
    static std::unique_ptr<cudf::column> make_column_from_contents(
        cudf::column::contents&& col_data, cudf::size_type num_row,
        cudf::size_type null_count, cudf::data_type type,
        std::vector<std::unique_ptr<cudf::column>>&& children,
        rmm::cuda_stream_view stream);

    /**
     * @brief Evolve a single column from the file's read schema to the
     * output schema.
     *
     * Handles lists, maps, and structs recursively. Primitive types are
     * cast to the target type when needed. Columns missing in the file
     * are filled with nulls by the caller (evolve_table).
     */
    std::unique_ptr<cudf::column> evolve_column(
        std::unique_ptr<cudf::column> col,
        const std::shared_ptr<arrow::Field>& source_field,
        const std::shared_ptr<arrow::Field>& target_field,
        rmm::cuda_stream_view stream);

    std::unique_ptr<cudf::table> evolve_table(std::unique_ptr<cudf::table> tbl,
                                              rmm::cuda_stream_view stream);

    static void push_cudf_identity(
        bool value, cudf::ast::tree& filter_ast_tree,
        std::vector<std::unique_ptr<cudf::scalar>>& filter_scalars);

    static void push_literal_to_cudf_ast(
        PyObject* value_py, const std::shared_ptr<arrow::DataType>& type,
        cudf::ast::tree& filter_ast_tree,
        std::vector<std::unique_ptr<cudf::scalar>>& filter_scalars);

    /**
     * @brief Recursively walk a pyiceberg cuDF AST filter tree (nested
     * Python tuples) and push cuDF AST nodes onto filter_ast_tree.
     *
     * Supported operators: true, false, is_null, is_not_null, eq, neq,
     * gt, gte, lt, lte, in, not_in, not, and, or.
     */
    void build_pyiceberg_cudf_ast_node(
        PyObject* node, const std::map<int, int>& field_id_to_col_idx,
        const std::shared_ptr<arrow::Schema>& read_schema,
        cudf::ast::tree& filter_ast_tree,
        std::vector<std::unique_ptr<cudf::scalar>>& filter_scalars,
        rmm::cuda_stream_view stream);

    /**
     * @brief Fetch the Iceberg dataset from Python (all ranks participate
     * in the collective call), unwrap the Arrow filesystem, and store
     * pieces and schema groups for later distribution.
     *
     * Also converts the combined DuckDB+Iceberg filter to both a cuDF AST
     * (for GPU row-level filtering) and an Arrow format string (for
     * file-level filtering via pyiceberg).
     */
    void get_dataset(PyObject* catalog, const std::string& table_id,
                     const std::shared_ptr<arrow::Schema>& arrow_schema,
                     PyObject* iceberg_filter, PyObject* iceberg_schema,
                     int64_t snapshot_id, int64_t tot_rows_to_read,
                     MPI_Comm comm);

    void distribute_pieces();

    void init_scanners(MPI_Comm comm);

    struct IcebergPieceInfo {
        std::string path;
        int64_t num_rows;
        int64_t schema_group_idx;
    };

    std::size_t target_rows_;
    int rank_{0}, size_{1};

    duckdb::unique_ptr<duckdb::TableFilterSet> filter_exprs_;
    cudf::ast::tree filter_ast_tree;
    std::vector<std::unique_ptr<cudf::scalar>> filter_scalars;

    std::shared_ptr<arrow::fs::FileSystem> filesystem_;
    PyObjectPtr py_pieces;
    PyObjectPtr py_schema_groups;
    PyObjectPtr pyarrow_schema;
    PyObjectPtr py_filesystem;

    std::vector<IcebergPieceInfo> pieces_;
    std::vector<std::shared_ptr<arrow::Schema>> scanner_read_schemas;
    int64_t rows_to_skip = 0;
    const std::vector<int>& selected_columns_;
    std::shared_ptr<arrow::Schema> output_arrow_schema;

    size_t curr_piece_idx = 0;
    std::unique_ptr<cudf::io::chunked_parquet_reader> curr_reader;
    std::shared_ptr<arrow::Schema> curr_read_schema;
    std::unique_ptr<cudf::table> leftover_tbl;
    std::vector<std::pair<int, int>> curr_col_mapping;
    int64_t last_filter_schema_group_idx = -1;
    PyObjectPtr iceberg_filter_cudf_ast = nullptr;
    std::vector<PyObjectPtr> scanner_field_ids;
    MetricBase::TimerValue evolve_time_ = 0;
};

struct PhysicalGPUReadIcebergMetrics {
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;

    stat_t rows_read = 0;
    time_t init_time = 0;
    time_t produce_time = 0;
};

/**
 * @brief Physical operator for reading Iceberg tables on GPU.
 *
 * Uses GPUIcebergRankBatchGenerator to produce GPU-side cudf::table
 * batches. Filters are applied at both the file level (via pyiceberg)
 * and the row level (via cuDF AST). Non-GPU ranks produce empty results.
 */
class PhysicalGPUReadIceberg : public PhysicalGPUSource {
   private:
    PyObject* catalog;
    const std::string table_id;
    PyObject* iceberg_filter;
    PyObject* iceberg_schema;
    const int64_t snapshot_id;
    duckdb::unique_ptr<duckdb::TableFilterSet> filter_exprs;
    const std::shared_ptr<arrow::Schema> arrow_schema;
    const std::vector<int> selected_columns;

    std::shared_ptr<bodo::Schema> output_schema;
    std::shared_ptr<arrow::Schema> output_arrow_schema;

    PhysicalGPUReadIcebergMetrics metrics;
    std::shared_ptr<GPUIcebergRankBatchGenerator> batch_gen;
    MPI_Comm comm;

    JoinFilterColStats join_filter_col_stats;

   public:
    explicit PhysicalGPUReadIceberg(
        PyObject* catalog, const std::string table_id, PyObject* iceberg_filter,
        PyObject* iceberg_schema,
        const std::shared_ptr<arrow::Schema> arrow_schema,
        const int64_t snapshot_id, const std::vector<int>& selected_columns,
        duckdb::TableFilterSet& filter_exprs,
        duckdb::unique_ptr<duckdb::BoundLimitNode>& limit_val,
        JoinFilterColStats join_filter_col_stats);

    virtual ~PhysicalGPUReadIceberg();

    void FinalizeSource() override;

    std::pair<GPU_DATA, OperatorResult> ProduceBatchGPU(
        std::shared_ptr<StreamAndEvent> se) override;

    const std::shared_ptr<bodo::Schema> getOutputSchemaInternal() override;

   private:
    void init_batch_gen();
};
