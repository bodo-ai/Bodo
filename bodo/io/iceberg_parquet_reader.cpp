
// Implementation of IcebergParquetReader (subclass of ArrowReader) with
// functionality that is specific to reading iceberg datasets (made up of
// parquet files)

#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <arrow/compute/expression.h>
#include <arrow/compute/type_fwd.h>
#include <arrow/dataset/dataset.h>
#include <arrow/dataset/scanner.h>
#include <arrow/python/pyarrow.h>
#include <arrow/record_batch.h>
#include <arrow/util/io_util.h>
#include <arrow/util/thread_pool.h>
#include <fmt/format.h>
#include <object.h>

#include "../libs/_bodo_to_arrow.h"
#include "../libs/_distributed.h"
#include "arrow_compat.h"
#include "arrow_reader.h"
#include "iceberg_helpers.h"
#include "iceberg_parquet_reader.h"

// Helper to ensure that the pyarrow wrappers have been imported.
// We use a static variable to make sure we only do the import once.
static bool imported_pyarrow_wrappers = false;
static void ensure_pa_wrappers_imported() {
#define CHECK(expr, msg)                                                    \
    if (expr) {                                                             \
        throw std::runtime_error(std::string("scanner_from_py_dataset: ") + \
                                 msg);                                      \
    }
    if (imported_pyarrow_wrappers) {
        return;
    }
    CHECK(arrow::py::import_pyarrow_wrappers(),
          "importing pyarrow_wrappers failed!");
    imported_pyarrow_wrappers = true;

#undef CHECK
}

/**
 * @brief Recursive helper for EvolveRecordBatch to handle the evolution of
 * columns, including nested columns by calling this function recursively on the
 * sub-fields.
 *
 * @param column Column from the RecordBatch to evolve.
 * @param source_field Expected field of the column. This is the field from the
 * read_schema of the IcebergSchemaGroup that the record batch belongs to.
 * @param target_field Target field to evolve the column to.
 * @return std::shared_ptr<::arrow::Array> Evolved column.
 */
std::shared_ptr<::arrow::Array> EvolveArray(
    std::shared_ptr<::arrow::Array> column,
    const std::shared_ptr<::arrow::Field>& source_field,
    const std::shared_ptr<::arrow::Field>& target_field) {
    // Verify that the Iceberg field ID is the same.
    int source_field_iceberg_field_id = get_iceberg_field_id(source_field);
    int target_field_iceberg_field_id = get_iceberg_field_id(target_field);
    if (source_field_iceberg_field_id != target_field_iceberg_field_id) {
        throw std::runtime_error(fmt::format(
            "IcebergParquetReader::EvolveArray: Iceberg field ID of the source "
            "({}) and target ({}) fields do not match!",
            source_field_iceberg_field_id, target_field_iceberg_field_id));
    }
    // Verify that the source field is the same type as the column (including
    // dict-encoding).
    if (column->type()->id() != source_field->type()->id()) {
        throw std::runtime_error(fmt::format(
            "IcebergParquetReader::EvolveArray: Column field type ({}) "
            "does not match expected field type ({})!",
            column->type()->ToString(), source_field->type()->ToString()));
    }
    // Verify that the overarching type is the same between the
    // source and target fields.
    if (target_field->type()->id() != source_field->type()->id()) {
        // Source field being a dict-encoded string while the target field
        // being a string is fine. If that's not the case, then raise
        // an exception.
        if (!(arrow::is_dictionary(source_field->type()->id()) &&
              (std::dynamic_pointer_cast<arrow::DictionaryType>(
                   source_field->type())
                   ->value_type()
                   ->id() == target_field->type()->id()) &&
              (arrow::is_base_binary_like(target_field->type()->id())))) {
            throw std::runtime_error(fmt::format(
                "IcebergParquetReader::EvolveArray: Source ({}) and "
                "target ({}) field types do not match!",
                source_field->type()->ToString(),
                target_field->type()->ToString()));
        }
    }

    if (::arrow::is_list(source_field->type()->id())) {
        // Just recurse on the value field, put the
        // outputs back together and return.

        // Get the values array out from the column. Call
        // this function recursively on it with target field.
        const std::shared_ptr<::arrow::Field>& source_value_field =
            std::dynamic_pointer_cast<arrow::BaseListType>(source_field->type())
                ->value_field();
        const std::shared_ptr<::arrow::Field>& target_value_field =
            std::dynamic_pointer_cast<arrow::BaseListType>(target_field->type())
                ->value_field();

        if (source_field->type()->id() == ::arrow::Type::LIST) {
            std::shared_ptr<::arrow::ListArray> column_ =
                std::dynamic_pointer_cast<::arrow::ListArray>(column);
            std::shared_ptr<::arrow::Array> values_column = column_->values();
            std::shared_ptr<::arrow::Array> evolved_values_column = EvolveArray(
                values_column, source_value_field, target_value_field);
            // Create a new ListArray using this as the new values array (use
            // existing offsets, null-bitmask, etc.) and return it.
            return std::make_shared<::arrow::ListArray>(
                target_field->type(), column_->length(),
                column_->value_offsets(), evolved_values_column,
                column_->null_bitmap(), column_->null_count(),
                column_->offset());
        } else if (source_field->type()->id() == ::arrow::Type::LARGE_LIST) {
            // Get the values array out from the column. Call
            // this function recursively on it with target field.
            std::shared_ptr<::arrow::LargeListArray> column_ =
                std::dynamic_pointer_cast<::arrow::LargeListArray>(column);
            std::shared_ptr<::arrow::Array> values_column = column_->values();
            std::shared_ptr<::arrow::Array> evolved_values_column = EvolveArray(
                values_column, source_value_field, target_value_field);
            // Create a new ListArray using this as the new values array (use
            // existing offsets, null-bitmask, etc.) and return it.
            return std::make_shared<::arrow::LargeListArray>(
                target_field->type(), column_->length(),
                column_->value_offsets(), evolved_values_column,
                column_->null_bitmap(), column_->null_count(),
                column_->offset());
        } else {
            throw std::runtime_error(fmt::format(
                "IcebergParquetReader::EvolveArray: Unsupported "
                "list type ({})! This is most likely a bug in Bodo.",
                source_field->type()->ToString()));
        }

    } else if (source_field->type()->id() == ::arrow::Type::MAP) {
        // Just recurse on the key and item fields, put the
        // outputs back together and return.
        const std::shared_ptr<::arrow::Field>& source_key_field =
            std::dynamic_pointer_cast<arrow::MapType>(source_field->type())
                ->key_field();
        const std::shared_ptr<::arrow::Field>& target_key_field =
            std::dynamic_pointer_cast<arrow::MapType>(target_field->type())
                ->key_field();
        const std::shared_ptr<::arrow::Field>& source_item_field =
            std::dynamic_pointer_cast<arrow::MapType>(source_field->type())
                ->item_field();
        const std::shared_ptr<::arrow::Field>& target_item_field =
            std::dynamic_pointer_cast<arrow::MapType>(target_field->type())
                ->item_field();
        std::shared_ptr<::arrow::MapArray> column_ =
            std::dynamic_pointer_cast<::arrow::MapArray>(column);
        std::shared_ptr<::arrow::Array> keys_column = column_->keys();
        std::shared_ptr<::arrow::Array> items_column = column_->items();

        std::shared_ptr<::arrow::Array> evolved_keys_column =
            EvolveArray(keys_column, source_key_field, target_key_field);
        std::shared_ptr<::arrow::Array> evolved_items_column =
            EvolveArray(items_column, source_item_field, target_item_field);

        return std::make_shared<::arrow::MapArray>(
            target_field->type(), column_->length(), column_->value_offsets(),
            evolved_keys_column, evolved_items_column, column_->null_bitmap(),
            column_->null_count(), column_->offset());

    } else if (source_field->type()->id() == ::arrow::Type::STRUCT) {
        // This is the only real case where we perform any real evolution
        // since Arrow doesn't do it for us.
        // In particular, we need to re-order and rename the sub-fields to match
        // the order and their names in the target field. This is done based on
        // the sub-fields' Iceberg Field IDs.
        // In case a sub-field from the target field does not exist, we need
        // to insert an all-null sub-field.

        const std::shared_ptr<arrow::StructType>& source_field_type =
            std::dynamic_pointer_cast<arrow::StructType>(source_field->type());
        const std::shared_ptr<arrow::StructType>& target_field_type =
            std::dynamic_pointer_cast<arrow::StructType>(target_field->type());
        const std::shared_ptr<arrow::StructType>& column_type =
            std::dynamic_pointer_cast<arrow::StructType>(column->type());
        if (source_field_type->num_fields() != column_type->num_fields()) {
            throw std::runtime_error(fmt::format(
                "IcebergParquetReader::EvolveArray: Number of struct "
                "sub-fields in the source ({}) and actual column ({}) do "
                "not match!",
                source_field_type->num_fields(), column_type->num_fields()));
        }

        std::shared_ptr<::arrow::StructArray> column_ =
            std::dynamic_pointer_cast<::arrow::StructArray>(column);

        // Loop over the source field's sub-fields.
        // Ensure they match the sub-fields in the column and create a map that
        // maps the Iceberg Field ID to the index of the sub-field in the
        // column.
        std::unordered_map<int, int>
            source_sub_field_iceberg_field_id_to_field_idx;
        for (int i = 0; i < source_field_type->num_fields(); i++) {
            if (source_field_type->field(i)->name() !=
                column_type->field(i)->name()) {
                throw std::runtime_error(fmt::format(
                    "IcebergParquetReader::EvolveArray: Name of struct "
                    "sub-field at index {} does not match! Expected '{}', but "
                    "got '{}' instead.",
                    i, source_field_type->field(i)->name(),
                    column_type->field(i)->name()));
            }
            int iceberg_field_id =
                get_iceberg_field_id(source_field_type->field(i));
            source_sub_field_iceberg_field_id_to_field_idx[iceberg_field_id] =
                i;
        }

        std::vector<std::shared_ptr<arrow::Array>> new_child_arrays;
        new_child_arrays.reserve(target_field_type->num_fields());
        // Now, loop over the sub-fields in the target field.
        // Get the Iceberg Field ID of this sub-field. Check
        // if the Iceberg Field ID exists in the map we created
        // above.
        for (int i = 0; i < target_field_type->num_fields(); i++) {
            int iceberg_field_id =
                get_iceberg_field_id(target_field_type->field(i));
            if (source_sub_field_iceberg_field_id_to_field_idx.contains(
                    iceberg_field_id)) {
                // If it does, use that to get the sub-field
                // from the column, evolve it and add it to the final
                // evolved column.
                int source_sub_field_idx =
                    source_sub_field_iceberg_field_id_to_field_idx
                        [iceberg_field_id];
                new_child_arrays.emplace_back(
                    EvolveArray(column_->field(source_sub_field_idx),
                                source_field_type->field(source_sub_field_idx),
                                target_field_type->field(i)));
            } else {
                // If it doesn't exist, create an all-null column of the
                // required type using Arrow's MakeArrayOfNull.
                arrow::Result<std::shared_ptr<arrow::Array>> null_arr_res =
                    ::arrow::MakeArrayOfNull(
                        target_field_type->field(i)->type(), column_->length(),
                        bodo::BufferPool::DefaultPtr());
                std::shared_ptr<arrow::Array> null_arr;
                CHECK_ARROW_READER_AND_ASSIGN(
                    null_arr_res,
                    fmt::format(
                        "IcebergParquetReader::EvolveArray: Failed to "
                        "create null array of type {}:",
                        target_field_type->field(i)->type()->ToString()),
                    null_arr);
                new_child_arrays.push_back(null_arr);
            }
        }

        return std::make_shared<arrow::StructArray>(
            target_field_type, column_->length(), new_child_arrays,
            column_->null_bitmap(), column_->null_count(), column_->offset());

    } else if (::arrow::is_dictionary(source_field->type()->id())) {
        // If it's a dict-encoded column, ensure that the value types match
        // and that it is (large)_string/binary.
        std::shared_ptr<arrow::DictionaryType> source_field_type =
            std::dynamic_pointer_cast<arrow::DictionaryType>(
                source_field->type());
        std::shared_ptr<arrow::DictionaryType> column_type =
            std::dynamic_pointer_cast<arrow::DictionaryType>(column->type());
        if (source_field_type->value_type()->id() !=
            column_type->value_type()->id()) {
            throw std::runtime_error(
                fmt::format("IcebergParquetReader::EvolveArray: Value type of "
                            "dictionary-encoded column ({}) does not match "
                            "expected type ({}).",
                            column_type->value_type()->ToString(),
                            source_field_type->value_type()->ToString()));
        }
        if (!arrow::is_base_binary_like(
                source_field_type->value_type()->id())) {
            throw std::runtime_error(
                fmt::format("IcebergParquetReader::EvolveArray: Value type of "
                            "a dictionary-encoded column must be "
                            "string/binary. Got '{}' instead.",
                            source_field_type->value_type()->ToString()));
        }
        // If everything is as expected, return the column as is.
        return column;
    } else {
        // Primitive type. No evolution required.
        return column;
    }
}

/**
 * @brief Helper function to evolve an Arrow Record Batch to the target schema.
 * No data is copied since we expect all data to be of the right type already.
 * We may add some all-null columns in the struct case. These will be allocated
 * from the default buffer pool.
 * We traverse recursively over the target schema. The source schema and the
 * schema of the record-batch is expected to match exactly (verified by this
 * function).
 * Most of the evolution is already handled by Arrow. The only evolution we
 * handle ourselves is related to struct fields where we need to re-order and
 * rename the sub-fields. We may also need to insert all-null sub-fields to a
 * struct when they don't exist in the record-batch.
 *
 * @param batch Record Batch to evolve.
 * @param source_schema Expected schema of the record batch. This is the
 * "read_schema" of the IcebergSchemaGroup that the record batch belongs to.
 * @param target_schema Target schema to evolve the record batch to. This is the
 * final schema of the table.
 * @param selected_fields Since we only load the required fields, this lists the
 * indices of the fields that are expected to have been loaded. These indices
 * correspond to both the target and source schemas.
 * @return std::shared_ptr<::arrow::RecordBatch> Evolved Record Batch.
 */
std::shared_ptr<::arrow::RecordBatch> EvolveRecordBatch(
    std::shared_ptr<::arrow::RecordBatch> batch,
    std::shared_ptr<::arrow::Schema> source_schema,
    std::shared_ptr<::arrow::Schema> target_schema,
    const std::vector<int>& selected_fields) {
    std::vector<std::shared_ptr<::arrow::Array>> out_batch_columns;
    out_batch_columns.reserve(selected_fields.size());
    std::vector<std::shared_ptr<arrow::Field>> output_schema_fields;
    output_schema_fields.reserve(selected_fields.size());
    int j = 0;
    for (int field_idx : selected_fields) {
        std::shared_ptr<::arrow::Array> orig_col = batch->column(j);
        const std::shared_ptr<::arrow::Field>& source_field =
            source_schema->field(field_idx);
        const std::shared_ptr<::arrow::Field>& target_field =
            target_schema->field(field_idx);
        out_batch_columns.push_back(
            EvolveArray(orig_col, source_field, target_field));
        output_schema_fields.push_back(target_field);
        j++;
    }
    std::shared_ptr<arrow::Schema> output_schema =
        std::make_shared<arrow::Schema>(output_schema_fields,
                                        target_schema->metadata());

    return ::arrow::RecordBatch::Make(output_schema, batch->num_rows(),
                                      out_batch_columns);
}

// ------------ Arrow Helpers -----------

namespace bodo::arrow_py_compat {

/**
 * @brief Equivalent to PyArrow's Scanner._make_scan_options. The main change is
 * that we have added 'cpu_executor' as a parameter. See that function's
 * docstring for a more detailed description of the arguments.
 *
 * @param dataset
 * @param expr_filter
 * @param column_names
 * @param batch_size
 * @param use_threads
 * @param batch_readahead
 * @param fragment_readahead
 * @param cpu_executor The executor to use for CPU intensive operations like
 * decoding, decompression, applying filters, etc.
 * @return std::shared_ptr<::arrow::dataset::ScanOptions>
 */
static std::shared_ptr<::arrow::dataset::ScanOptions> make_scan_options(
    std::shared_ptr<arrow::dataset::Dataset> dataset,
    arrow::compute::Expression expr_filter,
    const std::vector<std::string>& column_names, int64_t batch_size,
    bool use_threads, int32_t batch_readahead, int32_t fragment_readahead,
    arrow::internal::Executor* cpu_executor) {
    std::shared_ptr<arrow::dataset::ScannerBuilder> builder =
        std::make_shared<arrow::dataset::ScannerBuilder>(dataset);

    auto bind_res = expr_filter.Bind(*(builder->schema()));
    CHECK_ARROW_READER_AND_ASSIGN(bind_res,
                                  "Error while binding schema to the filter",
                                  auto bound_expr_filter);
    auto filter_res = builder->Filter(bound_expr_filter);
    CHECK_ARROW_READER(filter_res, "Error during filter");
    auto project_res = builder->Project(column_names);
    CHECK_ARROW_READER(project_res, "Error during projection ");
    auto batch_size_res = builder->BatchSize(batch_size);
    CHECK_ARROW_READER(batch_size_res, "Error setting batch_size");
    auto batch_readahead_res = builder->BatchReadahead(batch_readahead);
    CHECK_ARROW_READER(builder->BatchReadahead(batch_readahead),
                       "Error setting batch_readahead");
    auto fragment_readahead_res =
        builder->FragmentReadahead(fragment_readahead);
    CHECK_ARROW_READER(builder->FragmentReadahead(fragment_readahead),
                       "Error setting fragment_readahead");
    auto use_threads_res = builder->UseThreads(use_threads);
    CHECK_ARROW_READER(use_threads_res, "Error setting use_threads");
    auto pool_res = builder->Pool(bodo::BufferPool::DefaultPtr());
    CHECK_ARROW_READER(pool_res, "Error setting pool");
    // XXX Could set FragmentScanOptions similarly

    CHECK_ARROW_READER_AND_ASSIGN(
        builder->GetScanOptions(), "Error during GetScanOptions",
        std::shared_ptr<::arrow::dataset::ScanOptions> scan_options);

#if ARROW_VERSION_MAJOR >= 22
    scan_options->cpu_executor = cpu_executor;
#endif

    return scan_options;
}

/**
 * @brief Create an Arrow Scanner from a PyArrow Dataset Object. This is roughly
 * what PyArrow's `Scanner.from_dataset` function
 * (https://github.com/apache/arrow/blob/apache-arrow-17.0.0/python/pyarrow/_dataset.pyx#L3491)
 * does. See that function's docstring for a more detailed description of the
 * arguments.
 *
 * @param dataset_py The PyArrow Dataset Object (pyarrow.Dataset) to create the
 * Scanner from.
 * @param expr_filter_py The filter to apply during the scan. This must be a
 * pyarrow.compute.Expression object.
 * @param selected_fields List of field indices to keep (based on the dataset's
 * schema).
 * @param batch_size
 * @param use_threads
 * @param batch_readahead
 * @param fragment_readahead
 * @param cpu_executor The executor to use for CPU intensive operations like
 * decoding, decompression, applying filters, etc.
 * @return std::shared_ptr<::arrow::dataset::Scanner>
 */
std::shared_ptr<::arrow::dataset::Scanner> scanner_from_py_dataset(
    PyObject* dataset_py, PyObject* expr_filter_py,
    const std::vector<int>& selected_fields,
    int64_t batch_size = arrow::dataset::kDefaultBatchSize,
    bool use_threads = true,
    int32_t batch_readahead = arrow::dataset::kDefaultBatchReadahead,
    int32_t fragment_readahead = arrow::dataset::kDefaultFragmentReadahead,
    arrow::internal::Executor* cpu_executor =
        arrow::internal::GetCpuThreadPool()) {
    ensure_pa_wrappers_imported();
    // Unwrap Python objects into C++
    auto unwrap_expr_res = arrow::py::unwrap_expression(expr_filter_py);
    CHECK_ARROW_READER_AND_ASSIGN(unwrap_expr_res, "Error unwrapping filter",
                                  arrow::compute::Expression expr_filter);
    auto unwrap_dataset_res = arrow::py::unwrap_dataset(dataset_py);
    CHECK_ARROW_READER_AND_ASSIGN(
        unwrap_dataset_res, "Error unwrapping dataset",
        std::shared_ptr<arrow::dataset::Dataset> dataset);

    std::vector<std::string> column_names;
    for (int field_num : selected_fields) {
        column_names.push_back(dataset->schema()->field_names()[field_num]);
    }

    std::shared_ptr<::arrow::dataset::ScanOptions> scan_options =
        make_scan_options(dataset, expr_filter, column_names, batch_size,
                          use_threads, batch_readahead, fragment_readahead,
                          cpu_executor);

    std::shared_ptr<arrow::dataset::ScannerBuilder> scanner_builder =
        std::make_shared<arrow::dataset::ScannerBuilder>(dataset, scan_options);

    auto builder_res = scanner_builder->Finish();
    CHECK_ARROW_READER_AND_ASSIGN(
        builder_res, "Error finalizing ScannerBuilder!",
        std::shared_ptr<arrow::dataset::Scanner> scanner);
    return scanner;
}

/**
 * @brief Equivalent to PyArrow's Scanner.to_reader.
 *
 * @param scanner
 * @return std::shared_ptr<arrow::RecordBatchReader>
 */
static std::shared_ptr<arrow::RecordBatchReader> scanner_to_rb_reader(
    std::shared_ptr<::arrow::dataset::Scanner> scanner) {
    std::shared_ptr<arrow::RecordBatchReader> reader;
    auto to_reader_res = scanner->ToRecordBatchReader();
    CHECK_ARROW_READER_AND_ASSIGN(
        to_reader_res,
        "scanner_to_rb_reader: Error creating RecordBatchReader from Scanner!",
        reader)
    return reader;
}

}  // namespace bodo::arrow_py_compat

IcebergParquetReader::IcebergParquetReader(
    PyObject* _catalog, const char* _table_id, bool _parallel,
    int64_t tot_rows_to_read, PyObject* _iceberg_filter,
    std::string _expr_filter_f_str, PyObject* _filter_scalars,
    std::vector<int> _selected_fields, std::vector<bool> is_nullable,
    PyObject* _pyarrow_schema, int64_t batch_size, int64_t op_id,
    int64_t _snapshot_id)
    : ArrowReader(_parallel, _pyarrow_schema, tot_rows_to_read,
                  _selected_fields, is_nullable, batch_size, op_id),
      catalog(_catalog),
      table_id(_table_id),
      iceberg_filter(_iceberg_filter),
      expr_filter_f_str(std::move(_expr_filter_f_str)),
      filter_scalars(_filter_scalars),
      snapshot_id(_snapshot_id) {
    // Unless explicitly disabled, use our own SingleThreadedCpuThreadPool
    // for streaming read.
    char* use_st_pool_env_ =
        std::getenv("BODO_STREAM_ICEBERG_READER_DISABLE_ST_THREAD_POOL");
    bool use_st_pool =
        (batch_size != -1) &&
        !(use_st_pool_env_ && (std::strcmp(use_st_pool_env_, "1") == 0));
    if (use_st_pool) {
        this->st_cpu_executor_.emplace(
            bodo::SingleThreadedCpuThreadPool::Default());
    }
}

IcebergParquetReader::~IcebergParquetReader() {
    // When an unsupported schema evolution is detected in
    // `bodo.io.iceberg.get_iceberg_pq_dataset`, a Python exception
    // is thrown. That exception is detected and converted to a C++
    // exception in `IcebergParquetReader::get_dataset`. That exception
    // will be caught in `get_iceberg_pq_dataset` so this class to be
    // destructed, calling this function aka the destructor.

    // Py_XDECREF checks if the input is null,
    // while Py_DECREF doesn't and would just segfault
    Py_XDECREF(this->file_list);
}

void IcebergParquetReader::init_iceberg_reader(
    std::span<int32_t> str_as_dict_cols, bool create_dict_from_string) {
    ArrowReader::init_arrow_reader(str_as_dict_cols, create_dict_from_string);
    // Initialize the scanners.
    if (this->parallel) {
        // Get the average number of pieces per rank. This is used to
        // increase the number of threads of the Arrow batch reader
        // for ranks that have to read many more files than others.
        int num_ranks;
        MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
        uint64_t num_pieces = static_cast<uint64_t>(get_num_pieces());
        CHECK_MPI(MPI_Allreduce(MPI_IN_PLACE, &num_pieces, 1, MPI_UINT64_T,
                                MPI_SUM, MPI_COMM_WORLD),
                  "IcebergParquetReader::init_iceberg_reader: MPI error on "
                  "MPI_Allreduce:");
        this->avg_num_pieces = num_pieces / static_cast<double>(num_ranks);
    }
    // Initialize the Arrow Dataset Scanners for reading the file segments
    // assigned to this rank. This will create as many scanners as the
    // number of unique SchemaGroups that these files belong to.
    this->init_scanners();
    // Construct ChunkedTableBuilder for output in the streaming case.
    if (this->batch_size != -1) {
        this->dict_builders = std::vector<std::shared_ptr<DictionaryBuilder>>(
            selected_fields.size());

        // This will create dict-builders for the nested types. The target
        // schema doesn't have dict-encoding information, so this will
        // essentially create nullptrs.
        for (size_t i = 0; i < selected_fields.size(); i++) {
            const std::shared_ptr<arrow::Field>& field =
                this->schema->field(selected_fields[i]);
            this->dict_builders[i] = create_dict_builder_for_array(
                arrow_type_to_bodo_data_type(field->type()), false);
        }

        // Generate a mapping from schema index to selected fields for the
        // str_as_dict_cols.
        std::vector<int32_t> str_as_dict_cols_map(schema->num_fields(), -1);
        for (size_t i = 0; i < selected_fields.size(); i++) {
            str_as_dict_cols_map[selected_fields[i]] = i;
        }

        // Create dict builders for columns we will be reading
        // with dict-encoding (either directly from Arrow or doing the
        // dict-encoding ourselves).
        for (int str_as_dict_col : str_as_dict_cols) {
            int32_t index = str_as_dict_cols_map[str_as_dict_col];
            this->dict_builders[index] = create_dict_builder_for_array(
                std::make_unique<bodo::DataType>(bodo_array_type::DICT,
                                                 Bodo_CTypes::STRING),
                false);
        }
        auto empty_table = get_empty_out_table();
        this->out_batches = std::make_shared<ChunkedTableBuilder>(
            empty_table->schema(), this->dict_builders, (size_t)batch_size);
    }

    std::vector<MetricBase> metrics;
    this->ReportInitStageMetrics(metrics);

    QueryProfileCollector::Default().RegisterOperatorStageMetrics(
        QueryProfileCollector::MakeOperatorStageID(this->op_id,
                                                   QUERY_PROFILE_INIT_STAGE_ID),
        std::move(metrics));
}

// Return and incref the file list.
PyObject* IcebergParquetReader::get_file_list() {
    Py_INCREF(this->file_list);
    return this->file_list;
}

bool IcebergParquetReader::force_row_level_read() const {
    // Env var to explicitly disable piece level read. This is useful for
    // performance testing, etc.
    char* disable_piece_level_env_ =
        std::getenv("BODO_ICEBERG_DISABLE_PIECE_LEVEL_READ");
    if (disable_piece_level_env_ &&
        !std::strcmp(disable_piece_level_env_, "1")) {
        return true;
    }
    // Piece level read is only supported in distributed mode
    // when there's no LIMIT (this->tot_rows_to_read == -1).
    // TODO Add support for piece level read when there's a LIMIT
    // clause, the file planning itself could be optimized to only look at a
    // few files instead of all files.
    // TODO Add support for piece-level read in the replicated case. This
    // can be supported pretty easily by simply having all ranks read
    // all the pieces.
    return (this->tot_rows_to_read != -1) || (!this->parallel);
}

PyObject* IcebergParquetReader::get_dataset() {
    // import bodo.io.iceberg
    PyObject* iceberg_mod = PyImport_ImportModule("bodo.io.iceberg");
    if (PyErr_Occurred()) {
        throw std::runtime_error("python");
    }

    // If we will be doing the conversion to dict-encoded strings ourselves
    // (e.g. when reading Snowflake-managed Iceberg tables), then we can
    // simply pass str_as_dict_cols_py as an empty list for the purposes of
    // getting the dataset object.
    PyObject* str_as_dict_cols_py;
    if (this->create_dict_encoding_from_strings) {
        str_as_dict_cols_py = PyList_New(0);
    } else {
        str_as_dict_cols_py = PyList_New(this->str_as_dict_colnames.size());
        size_t i = 0;
        for (auto field_name : this->str_as_dict_colnames) {
            // PyList_SetItem steals the reference created by
            // PyUnicode_FromString.
            PyList_SetItem(str_as_dict_cols_py, i++,
                           PyUnicode_FromString(field_name.c_str()));
        }
    }

    PyObject* force_row_level_py =
        this->force_row_level_read() ? Py_True : Py_False;

    // ds = bodo.io.iceberg.get_iceberg_pq_dataset(
    //          catalog, table_id, pyarrow_schema, iceberg_filter,
    //          expr_filter_f_str, filter_scalars, snapshot_id,
    //      )
    PyObject* ds = PyObject_CallMethod(
        iceberg_mod, "get_iceberg_pq_dataset", "OsOOOsOOLL", this->catalog,
        this->table_id, this->pyarrow_schema, str_as_dict_cols_py,
        this->iceberg_filter, this->expr_filter_f_str.c_str(),
        this->filter_scalars, force_row_level_py,
        static_cast<long long>(this->snapshot_id),
        static_cast<long long>(this->tot_rows_to_read));
    if (ds == nullptr && PyErr_Occurred()) {
        throw std::runtime_error("python");
    }

    // The iceberg reader owns these references and doesn't need them after this
    Py_DECREF(this->catalog);
    Py_XDECREF(this->iceberg_filter);
    if (this->filter_scalars != Py_None) {
        Py_XDECREF(this->filter_scalars);
    }

    Py_DECREF(str_as_dict_cols_py);
    Py_DECREF(iceberg_mod);

    this->filesystem = PyObject_GetAttrString(ds, "filesystem");
    // Save the file list and snapshot id for use later
    this->file_list = PyObject_GetAttrString(ds, "file_list");
    PyObject* py_snapshot_id = PyObject_GetAttrString(ds, "snapshot_id");
    // The snapshot Id is just an integer so store in native code.
    this->snapshot_id = PyLong_AsLongLong(py_snapshot_id);
    Py_DECREF(py_snapshot_id);
    // Returns a new reference.
    this->schema_groups_py = PyObject_GetAttrString(ds, "schema_groups");
    this->iceberg_reader_metrics.ds_nunique_schema_groups =
        PyObject_Length(this->schema_groups_py);

    // Copy over metrics. This object is of type IcebergPqDatasetMetrics.
    PyObject* ds_metrics_py = PyObject_GetAttrString(ds, "metrics");
    if (ds_metrics_py == nullptr) {
        throw std::runtime_error(
            "IcebergParquetReader::get_dataset: metrics attribute not in "
            "dataset!");
    }

#define COPY_NUMERIC_METRIC(field)                                             \
    {                                                                          \
        PyObject* metric_py = PyObject_GetAttrString(ds_metrics_py, #field);   \
        if (metric_py == NULL) {                                               \
            throw std::runtime_error(                                          \
                fmt::format("IcebergParquetReader::get_dataset: {} attribute " \
                            "not in dataset metrics!",                         \
                            #field));                                          \
        }                                                                      \
        this->iceberg_reader_metrics.get_ds_##field =                          \
            PyLong_AsUnsignedLongLong(metric_py);                              \
        Py_DECREF(metric_py);                                                  \
    }

#define COPY_BOOL_METRIC(field)                                                \
    {                                                                          \
        PyObject* metric_py = PyObject_GetAttrString(ds_metrics_py, #field);   \
        if (metric_py == NULL) {                                               \
            throw std::runtime_error(                                          \
                fmt::format("IcebergParquetReader::get_dataset: {} attribute " \
                            "not in dataset metrics!",                         \
                            #field));                                          \
        }                                                                      \
        this->iceberg_reader_metrics.get_ds_##field =                          \
            PyObject_IsTrue(metric_py) ? 1 : 0;                                \
        Py_DECREF(metric_py);                                                  \
    }

    COPY_NUMERIC_METRIC(file_list_time);
    COPY_NUMERIC_METRIC(file_to_schema_time_us);
    COPY_NUMERIC_METRIC(get_fs_time);
    COPY_NUMERIC_METRIC(n_files_analyzed);
    COPY_NUMERIC_METRIC(file_frags_creation_time);
    COPY_NUMERIC_METRIC(get_sg_id_time);
    COPY_NUMERIC_METRIC(sort_by_sg_id_time);
    COPY_NUMERIC_METRIC(nunique_sgs_seen);
    COPY_NUMERIC_METRIC(exact_row_counts_time);
    COPY_NUMERIC_METRIC(get_row_counts_nrgs);
    COPY_NUMERIC_METRIC(get_row_counts_nrows);
    COPY_NUMERIC_METRIC(get_row_counts_total_bytes);
    COPY_NUMERIC_METRIC(pieces_allgather_time);
    COPY_NUMERIC_METRIC(sort_all_pieces_time);
    COPY_NUMERIC_METRIC(assemble_ds_time);

    Py_DECREF(ds_metrics_py);

    return ds;

#undef COPY_BOOL_METRIC
#undef COPY_NUMERIC_METRIC
}

void IcebergParquetReader::distribute_pieces(PyObject* pieces_py) {
    assert(!this->row_level);
    assert(this->parallel);
    assert(this->tot_rows_to_read == -1);

    // import bodo.io.iceberg
    PyObject* iceberg_mod =
        PyImport_ImportModule("bodo.io.iceberg.read_parquet");
    if (PyErr_Occurred()) {
        throw std::runtime_error("python");
    }

    // pieces_myrank_py =
    // bodo.io.iceberg.read_parquet.distribute_pieces(pieces_py)
    PyObject* pieces_myrank_py =
        PyObject_CallMethod(iceberg_mod, "distribute_pieces", "O", pieces_py);
    if (pieces_myrank_py == nullptr && PyErr_Occurred()) {
        throw std::runtime_error("python");
    }

    PyObject* piece;
    PyObject* iterator = PyObject_GetIter(pieces_myrank_py);
    if (iterator == nullptr) {
        throw std::runtime_error(
            "ArrowReader::distribute_pieces(): error getting "
            "pieces iterator");
    }

    while ((piece = PyIter_Next(iterator))) {
        PyObject* num_rows_piece_py =
            PyObject_GetAttrString(piece, "_bodo_num_rows");
        if (num_rows_piece_py == nullptr) {
            throw std::runtime_error(
                "ArrowReader::distribute_pieces(): _bodo_num_rows "
                "attribute not in piece");
        }
        int64_t num_rows_piece = PyLong_AsLongLong(num_rows_piece_py);
        Py_DECREF(num_rows_piece_py);
        this->add_piece(piece, num_rows_piece);
        this->metrics.local_rows_to_read += num_rows_piece;
        this->metrics.local_n_pieces_to_read_from++;
        Py_DECREF(piece);
    }

    Py_DECREF(iterator);
    Py_DECREF(pieces_myrank_py);
    Py_DECREF(iceberg_mod);
}

void IcebergParquetReader::add_piece(PyObject* piece, int64_t num_rows) {
    // p = piece.path
    PyObject* p = PyObject_GetAttrString(piece, "path");
    const char* c_path = PyUnicode_AsUTF8(p);
    this->file_paths.emplace_back(c_path);
    Py_DECREF(p);

    // Number of rows to read from this file.
    this->pieces_nrows.push_back(num_rows);

    // Get the index of the schema group to use for this
    // file.
    PyObject* schema_group_idx_py =
        PyObject_GetAttrString(piece, "schema_group_idx");
    if (schema_group_idx_py == nullptr) {
        throw std::runtime_error("schema_group_idx_py attribute not in piece");
    }
    int64_t schema_group_idx = PyLong_AsLongLong(schema_group_idx_py);
    this->pieces_schema_group_idx.push_back(schema_group_idx);
    Py_DECREF(schema_group_idx_py);
}

std::shared_ptr<table_info> IcebergParquetReader::get_empty_out_table() {
    if (this->empty_out_table != nullptr) {
        return this->empty_out_table;
    }

    TableBuilder builder(this->schema, this->selected_fields, 0,
                         this->is_nullable, this->str_as_dict_colnames,
                         this->create_dict_encoding_from_strings);
    std::shared_ptr<table_info> out_table =
        std::shared_ptr<table_info>(builder.get_table());

    this->empty_out_table = out_table;

    return out_table;
}
std::tuple<table_info*, bool, uint64_t>
IcebergParquetReader::read_inner_row_level() {
    // If batch_size is set, then we need to iteratively read a
    // batch at a time.
    if (this->batch_size != -1) {
        // TODO: Match SnowflakeReader for the zero-col case.
        // Fetch a new batch if we don't have a full batch yet and have more
        // rows to read.
        while (this->rows_left_to_read > 0 &&
               this->out_batches->total_remaining <
                   static_cast<size_t>(this->batch_size)) {
            this->iceberg_reader_metrics.n_batches++;
            time_pt get_batch_start = start_timer();
            // Get the next batch.
            std::shared_ptr<arrow::RecordBatch> batch = nullptr;
            do {
                bool out_of_pieces = false;
                std::tie(out_of_pieces, batch) = this->get_next_batch();
                if (out_of_pieces) {
                    // 'this->rows_left_to_emit'/'this->rows_left_to_read'
                    // should be 0 by now!
                    throw std::runtime_error(
                        "IcebergParquetReader::read_inner_row_level: Out "
                        "of pieces! This is most likely a bug.");
                }
            } while (batch == nullptr);
            this->iceberg_reader_metrics.get_batch_time +=
                end_timer(get_batch_start);

            // Transform the batch to the required target schema.
            time_pt evolve_start = start_timer();
            batch = EvolveRecordBatch(std::move(batch), this->curr_read_schema,
                                      this->schema, this->selected_fields);
            this->iceberg_reader_metrics.evolve_time += end_timer(evolve_start);

            int64_t batch_offset =
                std::min(this->rows_to_skip, batch->num_rows());
            int64_t length = std::min(this->rows_left_to_read,
                                      batch->num_rows() - batch_offset);
            // Update stats
            this->rows_left_to_read -= length;
            this->rows_to_skip -= batch_offset;
            // This is zero-copy slice.
            batch = batch->Slice(batch_offset, length);

            time_pt start_rb_to_bodo = start_timer();
            // TODO Pass BufferPool as the source pool!
            std::shared_ptr<table_info> bodo_table =
                arrow_recordbatch_to_bodo(batch, length);
            this->iceberg_reader_metrics.arrow_rb_to_bodo_time +=
                end_timer(start_rb_to_bodo);
            if (length == this->batch_size) {
                // We have read a batch of the exact size. Just reuse this
                // batch for the next read and unify the dictionaries.
                time_pt unify_start = start_timer();
                table_info* out_table =
                    this->unify_table_with_dictionary_builders(
                        std::move(bodo_table));
                this->iceberg_reader_metrics.unify_time +=
                    end_timer(unify_start);
                this->rows_left_to_emit -= length;
                bool is_last = (this->rows_left_to_emit <= 0) &&
                               (this->out_batches->total_remaining <= 0);
                return std::make_tuple(out_table, is_last, length);
            } else {
                // We are reading less than a full batch. We need to append
                // to the chunked table builder. Assuming we are prefetching
                // then its likely more efficient to read the next full
                // batch then to output a partial batch that could be
                // extremely small.
                this->iceberg_reader_metrics.n_small_batches++;
                time_pt unify_start = start_timer();
                this->out_batches->UnifyDictionariesAndAppend(bodo_table);
                this->iceberg_reader_metrics.unify_append_small_time +=
                    end_timer(unify_start);
            }
        }
        // Emit from the chunked table builder.
        time_pt start_pop = start_timer();
        auto [next_batch, out_batch_size] = out_batches->PopChunk(true);
        this->iceberg_reader_metrics.output_pop_chunk_time +=
            end_timer(start_pop);
        rows_left_to_emit -= out_batch_size;
        bool is_last = (this->rows_left_to_emit <= 0) &&
                       (this->out_batches->total_remaining <= 0);
        return std::make_tuple(new table_info(*next_batch), is_last,
                               out_batch_size);
    }

    TableBuilder builder(this->schema, this->selected_fields, this->count,
                         this->is_nullable, this->str_as_dict_colnames,
                         this->create_dict_encoding_from_strings);

    if (get_num_pieces() == 0) {
        table_info* table = builder.get_table();
        return std::make_tuple(table, true, 0);
    }

    while (this->rows_left_to_read > 0) {
        std::shared_ptr<arrow::RecordBatch> batch = nullptr;
        do {
            bool out_of_pieces = false;
            std::tie(out_of_pieces, batch) = this->get_next_batch();
            if (out_of_pieces) {
                // 'this->rows_left_to_emit'/'this->rows_left_to_read'
                // should be 0 by now!
                throw std::runtime_error(
                    "IcebergParquetReader::read_inner_row_level: Out "
                    "of pieces! This is most likely a bug.");
            }
        } while (batch == nullptr);
        // Transform the batch to the required target schema.
        batch = EvolveRecordBatch(std::move(batch), this->curr_read_schema,
                                  this->schema, this->selected_fields);
        int64_t batch_offset = std::min(this->rows_to_skip, batch->num_rows());
        int64_t length =
            std::min(this->rows_left_to_read, batch->num_rows() - batch_offset);
        if (length > 0) {
            // This is zero-copy slice.
            std::shared_ptr<::arrow::Table> table = arrow::Table::Make(
                this->schema, batch->Slice(batch_offset, length)->columns());
            builder.append(table);
            this->rows_left_to_read -= length;
        }
        this->rows_to_skip -= batch_offset;
    }

    table_info* table = builder.get_table();

    // rows_left_to_emit is unnecessary for non-streaming
    // so just set equal to rows_left_to_read
    this->rows_left_to_emit = this->rows_left_to_read;
    assert(this->rows_left_to_read == 0);
    return std::make_tuple(table, true, table->nrows());
}

std::tuple<table_info*, bool, uint64_t>
IcebergParquetReader::read_inner_piece_level() {
    if (this->batch_size != -1) {
        while (this->out_batches->total_remaining <
                   static_cast<size_t>(this->batch_size) &&
               !this->done_reading_pieces) {
            time_pt get_batch_start = start_timer();
            std::shared_ptr<arrow::RecordBatch> batch = nullptr;
            do {
                std::tie(this->done_reading_pieces, batch) =
                    this->get_next_batch();
            } while (batch == nullptr && !this->done_reading_pieces);
            if (this->done_reading_pieces) {
                assert(batch == nullptr);
                break;
            }
            this->iceberg_reader_metrics.get_batch_time +=
                end_timer(get_batch_start);
            this->iceberg_reader_metrics.n_batches++;
            time_pt evolve_start = start_timer();
            batch = EvolveRecordBatch(std::move(batch), this->curr_read_schema,
                                      this->schema, this->selected_fields);
            this->iceberg_reader_metrics.evolve_time += end_timer(evolve_start);
            int64_t nrows = batch->num_rows();
            time_pt start_rb_to_bodo = start_timer();
            // TODO Pass BufferPool as the source pool!
            std::shared_ptr<table_info> bodo_table =
                arrow_recordbatch_to_bodo(std::move(batch), nrows);
            this->iceberg_reader_metrics.arrow_rb_to_bodo_time +=
                end_timer(start_rb_to_bodo);
            if (nrows == this->batch_size) {
                time_pt unify_start = start_timer();
                table_info* out_table =
                    this->unify_table_with_dictionary_builders(
                        std::move(bodo_table));
                this->iceberg_reader_metrics.unify_time +=
                    end_timer(unify_start);
                return std::make_tuple(out_table, false, nrows);
            } else {
                // We are reading less than a full batch. We need to append
                // to the chunked table builder. Assuming we are prefetching
                // then its likely more efficient to read the next full
                // batch then to output a partial batch that could be
                // extremely small.
                this->iceberg_reader_metrics.n_small_batches++;
                time_pt unify_start = start_timer();
                this->out_batches->UnifyDictionariesAndAppend(bodo_table);
                this->iceberg_reader_metrics.unify_append_small_time +=
                    end_timer(unify_start);
            }
        }

        time_pt start_pop = start_timer();
        auto [next_batch, out_batch_size] = this->out_batches->PopChunk(true);
        this->iceberg_reader_metrics.output_pop_chunk_time +=
            end_timer(start_pop);
        this->emitted_all_output = (this->done_reading_pieces) &&
                                   (this->out_batches->total_remaining <= 0);
        return std::make_tuple(new table_info(*next_batch),
                               /*is_last*/ this->emitted_all_output,
                               out_batch_size);
    }

    /// Non-streaming case:

    // Read all batches. We cannot create the TableBuilder before reading
    // since we don't know the total row count yet.
    std::vector<std::shared_ptr<::arrow::Table>> batches;
    int64_t total_nrows = 0;
    while (!this->done_reading_pieces) {
        std::shared_ptr<arrow::RecordBatch> batch = nullptr;
        do {
            std::tie(this->done_reading_pieces, batch) = this->get_next_batch();
        } while (batch == nullptr && !this->done_reading_pieces);

        if (this->done_reading_pieces) {
            assert(batch == nullptr);
            this->emitted_all_output = true;
            break;
        }
        batch = EvolveRecordBatch(std::move(batch), this->curr_read_schema,
                                  this->schema, this->selected_fields);
        int64_t nrows = batch->num_rows();
        if (nrows > 0) {
            batches.push_back(
                arrow::Table::Make(this->schema, batch->columns()));
            total_nrows += nrows;
        }
    }

    TableBuilder builder(this->schema, this->selected_fields, total_nrows,
                         this->is_nullable, this->str_as_dict_colnames,
                         this->create_dict_encoding_from_strings);

    for (auto& batch : batches) {
        builder.append(std::move(batch));
    }
    batches.clear();

    table_info* table = builder.get_table();
    return std::make_tuple(table, true, total_nrows);
}
void IcebergParquetReader::init_scanners() {
    tracing::Event ev_scanner("init_scanners", this->parallel);
    if (get_num_pieces() == 0) {
        return;
    }

    // Construct Python lists from C++ vectors for values used in
    // get_pyarrow_datasets
    assert(this->file_paths.size() == this->pieces_nrows.size());
    assert(this->file_paths.size() == this->pieces_schema_group_idx.size());
    PyObject* fpaths_py = PyList_New(this->file_paths.size());
    PyObject* file_nrows_to_read_py = PyList_New(this->pieces_nrows.size());
    PyObject* file_schema_group_idxs_py =
        PyList_New(this->pieces_schema_group_idx.size());
    for (size_t i = 0; i < this->file_paths.size(); i++) {
        // PyList_SetItem steals the reference created by
        // PyUnicode_FromString and PyLong_FromLong.
        PyList_SetItem(fpaths_py, i,
                       PyUnicode_FromString(this->file_paths[i].c_str()));
        PyList_SetItem(file_nrows_to_read_py, i,
                       PyLong_FromLong(this->pieces_nrows[i]));
        PyList_SetItem(file_schema_group_idxs_py, i,
                       PyLong_FromLong(this->pieces_schema_group_idx[i]));
    }

    // If we will be doing the conversion to dict-encoded strings ourselves
    // (e.g. when reading Snowflake-managed Iceberg tables), then we can
    // simply pass str_as_dict_cols_py as an empty list for the purposes of
    // getting the Arrow scanners.
    PyObject* str_as_dict_cols_py;
    if (this->create_dict_encoding_from_strings) {
        str_as_dict_cols_py = PyList_New(0);
    } else {
        str_as_dict_cols_py = PyList_New(this->str_as_dict_colnames.size());
        size_t i = 0;
        for (auto field_name : this->str_as_dict_colnames) {
            // PyList_SetItem steals the reference created by
            // PyUnicode_FromString.
            PyList_SetItem(str_as_dict_cols_py, i++,
                           PyUnicode_FromString(field_name.c_str()));
        }
    }

    PyObject* iceberg_mod =
        PyImport_ImportModule("bodo.io.iceberg.read_parquet");

    // XXX TODO Use something like this to set the IO thread pool size and
    // then reset at the end.
    // uint32_t cpu_count = std::thread::hardware_concurrency();
    // if (cpu_count == 0) {
    //     cpu_count = 2;
    // }
    // CHECK_ARROW_READER(arrow::io::SetIOThreadPoolCapacity(4),
    //             "Error setting IO thread pool capacity!");

    // get_pyarrow_datasets returns a tuple with the a list of PyArrow
    // Datasets, a list of the corresponding read_schemas and the updated
    // offset for the first batch.
    time_pt start = start_timer();
    PyObject* datasets_updated_offset_tup = PyObject_CallMethod(
        iceberg_mod, "get_pyarrow_datasets", "OOOOdiOOLO", fpaths_py,
        file_nrows_to_read_py, file_schema_group_idxs_py,
        this->schema_groups_py, this->avg_num_pieces, int(this->parallel),
        this->filesystem, str_as_dict_cols_py, this->start_row_first_piece,
        this->pyarrow_schema);
    iceberg_reader_metrics.init_scanners_get_pa_datasets_time +=
        end_timer(start);
    if (datasets_updated_offset_tup == nullptr && PyErr_Occurred()) {
        throw std::runtime_error("python");
    }

    // PyTuple_GetItem returns a borrowed reference, so we don't need to
    // DECREF it explicitly.
    PyObject* datasets_py = PyTuple_GetItem(datasets_updated_offset_tup, 0);
    size_t n_datasets = PyList_Size(datasets_py);
    this->scanners.reserve(n_datasets);
    this->iceberg_reader_metrics.n_scanners = n_datasets;

    // PyTuple_GetItem returns a borrowed reference, so we don't need to
    // DECREF it explicitly.
    PyObject* dataset_expr_filters_py =
        PyTuple_GetItem(datasets_updated_offset_tup, 2);
    size_t n_dataset_expr_filters = PyList_Size(dataset_expr_filters_py);
    if (n_datasets != n_dataset_expr_filters) {
        throw std::runtime_error(
            "IcebergParquetReader::init_scanners: Number of datasets and "
            "expr_filters don't match! This is likely a bug.");
    }

    start = start_timer();
    for (size_t i = 0; i < n_datasets; i++) {
        // PyList_GetItem returns a borrowed reference, so we don't need to
        // DECREF it explicitly.
        PyObject* dataset_py = PyList_GetItem(datasets_py, i);
        PyObject* dataset_expr_filter_py =
            PyList_GetItem(dataset_expr_filters_py, i);
        this->scanners.push_back(bodo::arrow_py_compat::scanner_from_py_dataset(
            dataset_py, dataset_expr_filter_py, this->selected_fields,
            this->batch_size != -1 ? this->batch_size
                                   : arrow::dataset::kDefaultBatchSize,
            /*use_threads*/ true, this->batch_readahead, this->frag_readahead,
            this->cpu_executor()));
    }
    this->iceberg_reader_metrics.create_scanners_time += end_timer(start);

    // PyTuple_GetItem returns a borrowed reference, so we don't need to
    // DECREF it explicitly.
    PyObject* scanner_read_schemas_py =
        PyTuple_GetItem(datasets_updated_offset_tup, 1);
    size_t n_read_schemas = PyList_Size(scanner_read_schemas_py);
    if (n_datasets != n_read_schemas) {
        throw std::runtime_error(
            "IcebergParquetReader::init_scanners: Number of datasets and "
            "read_schemas don't match! This is likely a bug.");
    }

    this->scanner_read_schemas.reserve(n_read_schemas);
    for (size_t i = 0; i < n_read_schemas; i++) {
        // This returns a borrowed reference, so we don't need to
        // decref it.
        PyObject* read_schema_py = PyList_GetItem(scanner_read_schemas_py, i);
        std::shared_ptr<arrow::Schema> read_schema;
        CHECK_ARROW_READER_AND_ASSIGN(
            arrow::py::unwrap_schema(read_schema_py),
            "Iceberg Scanner Read Schema Couldn't Unwrap from Python",
            read_schema);
        this->scanner_read_schemas.push_back(read_schema);
    }

    this->rows_to_skip =
        PyLong_AsLongLong(PyTuple_GetItem(datasets_updated_offset_tup, 3));

    Py_DECREF(iceberg_mod);
    Py_DECREF(fpaths_py);
    Py_DECREF(file_nrows_to_read_py);
    Py_DECREF(file_schema_group_idxs_py);
    Py_DECREF(str_as_dict_cols_py);
    Py_DECREF(datasets_updated_offset_tup);
    Py_XDECREF(this->filesystem);
    Py_XDECREF(this->pyarrow_schema);
    Py_XDECREF(this->schema_groups_py);
}
void IcebergParquetReader::ReportInitStageMetrics(
    std::vector<MetricBase>& metrics_out) {
    if ((this->op_id == -1) || this->reported_init_stage_metrics) {
        return;
    }
    metrics_out.reserve(metrics_out.size() + 32);

#define APPEND_TIMER_METRIC(field) \
    metrics_out.push_back(         \
        TimerMetric(#field, this->iceberg_reader_metrics.field));

#define APPEND_STAT_METRIC(field) \
    metrics_out.push_back(        \
        StatMetric(#field, this->iceberg_reader_metrics.field));

#define APPEND_GLOBAL_STAT_METRIC(field) \
    metrics_out.push_back(               \
        StatMetric(#field, this->iceberg_reader_metrics.field, true));

    APPEND_TIMER_METRIC(get_ds_file_list_time);
    APPEND_TIMER_METRIC(get_ds_file_to_schema_time_us);
    APPEND_TIMER_METRIC(get_ds_get_fs_time);
    APPEND_STAT_METRIC(get_ds_n_files_analyzed);
    APPEND_TIMER_METRIC(get_ds_file_frags_creation_time);
    APPEND_TIMER_METRIC(get_ds_get_sg_id_time);
    APPEND_TIMER_METRIC(get_ds_sort_by_sg_id_time);
    APPEND_STAT_METRIC(get_ds_nunique_sgs_seen);
    APPEND_TIMER_METRIC(get_ds_exact_row_counts_time);
    APPEND_STAT_METRIC(get_ds_get_row_counts_nrgs);
    APPEND_STAT_METRIC(get_ds_get_row_counts_nrows);
    APPEND_STAT_METRIC(get_ds_get_row_counts_total_bytes);
    APPEND_TIMER_METRIC(get_ds_pieces_allgather_time);
    APPEND_TIMER_METRIC(get_ds_sort_all_pieces_time);
    APPEND_TIMER_METRIC(get_ds_assemble_ds_time);
    APPEND_GLOBAL_STAT_METRIC(ds_nunique_schema_groups);
    APPEND_TIMER_METRIC(init_scanners_get_pa_datasets_time);
    APPEND_STAT_METRIC(n_scanners);
    APPEND_TIMER_METRIC(create_scanners_time);

    // Report the ArrowReader Init Stage metrics.
    ArrowReader::ReportInitStageMetrics(metrics_out);

#undef APPEND_GLOBAL_STAT_METRIC
#undef APPEND_STAT_METRIC
#undef APPEND_TIMER_METRIC
}

void IcebergParquetReader::ReportReadStageMetrics(
    std::vector<MetricBase>& metrics_out) {
    if ((this->op_id == -1) || this->reported_read_stage_metrics) {
        return;
    }
    metrics_out.reserve(metrics_out.size() + 8);

#define APPEND_TIMER_METRIC(field) \
    metrics_out.push_back(         \
        TimerMetric(#field, this->iceberg_reader_metrics.field));

#define APPEND_STAT_METRIC(field) \
    metrics_out.push_back(        \
        StatMetric(#field, this->iceberg_reader_metrics.field));

    APPEND_TIMER_METRIC(get_batch_time);
    APPEND_TIMER_METRIC(evolve_time);
    APPEND_TIMER_METRIC(arrow_rb_to_bodo_time);
    APPEND_TIMER_METRIC(unify_time);
    APPEND_TIMER_METRIC(unify_append_small_time);
    APPEND_STAT_METRIC(n_batches);
    APPEND_STAT_METRIC(n_small_batches);
    APPEND_TIMER_METRIC(output_pop_chunk_time);

    // Report the ArrowReader Read Stage metrics.
    ArrowReader::ReportReadStageMetrics(metrics_out);

#undef APPEND_STAT_METRIC
#undef APPEND_TIMER_METRIC
}

std::tuple<bool, std::shared_ptr<arrow::RecordBatch>>
IcebergParquetReader::get_next_batch() {
    if (this->st_cpu_executor_.has_value()) {
        // Resume our single threaded executor so that tasks required for
        // producing the next batch can be executed.
        this->st_cpu_executor_.value()->ResumeExecutingTasks();
    }
    // Create the next reader:
    if (this->curr_reader == nullptr) {
        if (this->next_scanner_idx < this->scanners.size()) {
            this->curr_reader = bodo::arrow_py_compat::scanner_to_rb_reader(
                this->scanners[this->next_scanner_idx]);
            this->curr_read_schema =
                this->scanner_read_schemas[this->next_scanner_idx];
            // Release the corresponding scanner since we don't
            // need it anymore.
            this->scanners[this->next_scanner_idx].reset();
            this->next_scanner_idx++;
        } else {
            // Out of scanners, so return done=true.
            return {true, nullptr};
        }
    }
    std::shared_ptr<arrow::RecordBatch> batch = nullptr;
    auto batch_res = this->curr_reader->ReadNext(&batch);
    CHECK_ARROW_READER(batch_res,
                       "IcebergParquetReader::get_next_batch_: Error reading "
                       "next batch!");
    if (batch == nullptr) {
        // Reset the reader to free it. This will
        // prompt the next invocation to create a reader from the
        // next scanner.
        this->curr_reader = nullptr;
    }
    if (this->st_cpu_executor_.has_value()) {
        // Pause our single threaded executor so that the worker thread
        // doesn't interfere with the rest of the pipeline.
        this->st_cpu_executor_.value()->WaitForIdle();
    }
    return {false, batch};
}
/**
 * Read a Iceberg table made up of parquet files.
 *
 * @param catalog: Pyiceberg Catalog to read iceberg metadata
 * @param table_id : Table identifier of the iceberg table to read
 * @param parallel : true if reading in parallel
 * @param tot_rows_to_read : limit of rows to read or -1 if not limited
 * @param iceberg_filter : filters passed to iceberg for filter pushdown
 * @param expr_filter_f_str : Format string to use to generate the Arrow
 *  expression filter dynamically.
 * @param filter_scalars : Python list of tuples of the form
 * (var_name: str, var_value: Any). These are the scalars to plug into
 * the Arrow expression filters.
 * @param _selected_fields : Fields to select from the parquet dataset,
 * using the field ID of Arrow schema. Note that this *DOES* include
 * partition columns (Iceberg has hidden partitioning, so they *ARE* part of the
 * parquet files)
 * NOTE: selected_fields must be sorted
 * @param num_selected_fields : length of selected_fields array
 * @param _is_nullable : array of booleans that indicates which of the
 * selected fields is nullable. Same length and order as selected_fields.
 * @param pyarrow_schema : PyArrow schema (instance of pyarrow.lib.Schema)
 * determined at compile time. Used for schema evolution detection, and for
 * evaluating transformations in the future.
 * @param _str_as_dict_cols Indices of fields that must be read as dict-encoded
 * strings.
 * @param num_str_as_dict_cols Number of fields to read as dict-encoded strings.
 * @param create_dict_from_string Whether we should read the fields as strings
 * and then dict-encode them ourselves in Bodo (e.g. similar to the
 * SnowflakeReader) or should we let Arrow return dict-encoded arrays directly.
 * The former is useful when reading Snowflake-managed Iceberg tables since the
 * parquet files written by Snowflake are sometimes encoded in a way that Arrow
 * cannot convert to dict-encoded columns and can error out.
 * @param is_merge_into_cow : Is this table loaded as the target table for merge
 * into with COW. If True we will only apply filters that limit the number of
 * files and cannot filter rows within a file.
 * @param[out] total_rows_out Total number of rows read (globally).
 * @param[out] file_list_ptr : Additional output of the Python list of read-in
 * files. This is currently only used for MERGE INTO COW
 * @param[out] snapshot_id_ptr : Additional output of current snapshot id
 * This is currently only used for MERGE INTO COW
 * @return Table containing all the read data
 */
table_info* iceberg_pq_read_py_entry(
    PyObject* catalog, const char* table_id, bool parallel,
    int64_t tot_rows_to_read, PyObject* iceberg_filter,
    const char* expr_filter_f_str_, PyObject* filter_scalars,
    int32_t* _selected_fields, int32_t num_selected_fields,
    int32_t* _is_nullable, PyObject* pyarrow_schema, int32_t* _str_as_dict_cols,
    int32_t num_str_as_dict_cols, bool create_dict_from_string,
    bool is_merge_into_cow, int64_t* snapshot_id_ptr, int64_t* total_rows_out,
    PyObject** file_list_ptr) {
    try {
        std::vector<int> selected_fields(
            {_selected_fields, _selected_fields + num_selected_fields});
        std::vector<bool> is_nullable(_is_nullable,
                                      _is_nullable + num_selected_fields);
        std::span<int32_t> str_as_dict_cols(_str_as_dict_cols,
                                            num_str_as_dict_cols);

        std::string expr_filter_f_str(expr_filter_f_str_);
        if (is_merge_into_cow) {
            // If is_merge_into=True then we don't want to use any expr_filters
            // as we must load the whole file.
            expr_filter_f_str = "";
        }
        IcebergParquetReader reader(
            catalog, table_id, parallel, tot_rows_to_read, iceberg_filter,
            expr_filter_f_str, filter_scalars, selected_fields, is_nullable,
            pyarrow_schema, -1, -1, *snapshot_id_ptr);

        // Initialize reader
        reader.init_iceberg_reader(str_as_dict_cols, create_dict_from_string);

        // MERGE INTO COW Output Handling
        if (is_merge_into_cow) {
            *file_list_ptr = reader.get_file_list();
            *snapshot_id_ptr = reader.get_snapshot_id();
        } else {
            *file_list_ptr = Py_None;
            *snapshot_id_ptr = -1;
        }

        table_info* read_output = reader.read_all();
        uint64_t local_nrows = read_output->nrows();
        uint64_t global_nrows = local_nrows;
        if (parallel) {
            CHECK_MPI(MPI_Allreduce(&local_nrows, &global_nrows, 1,
                                    MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD),
                      "iceberg_pq_read_py_entry: MPI error on MPI_Allreduce:");
        }
        *total_rows_out = global_nrows;

        // Append the index column to the output table used for MERGE INTO COW
        // Since the MERGE INTO flag is internal, we assume that this column
        // is never dead for simplicity sake.
        if (is_merge_into_cow) {
            int64_t num_local_rows = read_output->nrows();
            std::shared_ptr<array_info> row_id_col_arr =
                alloc_numpy(num_local_rows, Bodo_CTypes::INT64);

            // Create the initial value on this rank
            // TODO: Replace with start_idx from ArrowReader
            int64_t init_val = 0;
            if (parallel) {
                CHECK_MPI(
                    MPI_Exscan(&num_local_rows, &init_val, 1, MPI_LONG_LONG_INT,
                               MPI_SUM, MPI_COMM_WORLD),
                    "iceberg_pq_read_py_entry: MPI error on MPI_Exscan:");
            }

            // Equivalent to np.arange(*total_rows_out, dtype=np.int64)
            std::iota(
                (int64_t*)row_id_col_arr->data1<bodo_array_type::NUMPY>(),
                (int64_t*)row_id_col_arr->data1<bodo_array_type::NUMPY>() +
                    num_local_rows,
                init_val);

            read_output->columns.push_back(row_id_col_arr);
        }

        return read_output;

    } catch (const std::exception& e) {
        // if the error string is "python" this means the C++ exception is
        // a result of a Python exception, so we don't call PyErr_SetString
        // because we don't want to replace the original Python error
        if (std::string(e.what()) != "python") {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
        return nullptr;
    }
}

/**
 * Construct an Iceberg Parquet-based ArrowReader
 *
 * @param catalog: Pyiceberg Catalog to read iceberg metadata
 * @param table_id : Table Identifier of the iceberg table to read
 * @param parallel : true if reading in parallel
 * @param tot_rows_to_read : limit of rows to read or -1 if not limited
 * @param iceberg_filter : filters passed to iceberg for filter pushdown
 * @param expr_filter_f_str : Format string to use to generate the Arrow
 *  expression filter dynamically.
 * @param filter_scalars : Python list of tuples of the form
 * (var_name: str, var_value: Any). These are the scalars to plug into
 * the Arrow expression filters.
 * @param _selected_fields : Fields to select from the parquet dataset,
 * using the field ID of Arrow schema. Note that this *DOES* include
 * partition columns (Iceberg has hidden partitioning, so they *ARE* part of the
 * parquet files)
 * NOTE: selected_fields must be sorted
 * @param num_selected_fields : length of selected_fields array
 * @param is_nullable : array of booleans that indicates which of the
 * selected fields is nullable. Same length and order as selected_fields.
 * @param pyarrow_schema : PyArrow schema (instance of pyarrow.lib.Schema)
 * determined at compile time. Used for schema evolution detection, and for
 * evaluating transformations in the future.
 * @param batch_size Reading batch size
 * @param op_id Operator ID generated by the planner for query profile.
 * @return ArrowReader to read the table's data
 */
ArrowReader* iceberg_pq_reader_init_py_entry(
    PyObject* catalog, const char* table_id, bool parallel,
    int64_t tot_rows_to_read, PyObject* iceberg_filter,
    const char* expr_filter_f_str, PyObject* filter_scalars,
    int32_t* _selected_fields, int32_t num_selected_fields,
    int32_t* _is_nullable, PyObject* pyarrow_schema, int32_t* _str_as_dict_cols,
    int32_t num_str_as_dict_cols, bool create_dict_from_string,
    int64_t batch_size, int64_t op_id) {
    try {
        std::vector<int> selected_fields(
            {_selected_fields, _selected_fields + num_selected_fields});
        std::vector<bool> is_nullable(_is_nullable,
                                      _is_nullable + num_selected_fields);
        std::span<int32_t> str_as_dict_cols(_str_as_dict_cols,
                                            num_str_as_dict_cols);

        IcebergParquetReader* reader = new IcebergParquetReader(
            catalog, table_id, parallel, tot_rows_to_read, iceberg_filter,
            std::string(expr_filter_f_str), filter_scalars, selected_fields,
            is_nullable, pyarrow_schema, batch_size, op_id, -1);

        // Initialize reader
        reader->init_iceberg_reader(str_as_dict_cols, create_dict_from_string);

        return reinterpret_cast<ArrowReader*>(reader);

    } catch (const std::exception& e) {
        // if the error string is "python" this means the C++ exception is
        // a result of a Python exception, so we don't call PyErr_SetString
        // because we don't want to replace the original Python error
        if (std::string(e.what()) != "python") {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
        return nullptr;
    }
}
