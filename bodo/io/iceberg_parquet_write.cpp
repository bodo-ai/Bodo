// Functions to write Bodo arrays to Iceberg table (parquet format)

#include <algorithm>
#include <memory>
#include <stdexcept>
#if _MSC_VER >= 1900
#undef timezone
#endif

#include <arrow/array.h>
#include <arrow/compute/cast.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/python/api.h>
#include <arrow/table.h>
#include <arrow/util/key_value_metadata.h>

#include <boost/uuid/uuid.hpp>             // uuid class
#include <boost/uuid/uuid_generators.hpp>  // generators
#include <boost/uuid/uuid_io.hpp>          // streaming operators etc.

#include <fmt/format.h>

#include <Python.h>

#include "../libs/_array_hash.h"
#include "../libs/_array_operations.h"
#include "../libs/_array_utils.h"
#include "../libs/_bodo_to_arrow.h"
#include "../libs/_dict_builder.h"
#include "../libs/_theta_sketches.h"
#include "../libs/iceberg_transforms.h"
#include "_s3_reader.h"
#include "arrow_compat.h"
#include "iceberg_helpers.h"
#include "parquet_write.h"

// if status of arrow::Result is not ok, form an err msg and raise a
// runtime_error with it
#define CHECK_ARROW(expr, msg)                                              \
    if (!(expr.ok())) {                                                     \
        std::string err_msg = std::string("Error in arrow parquet I/O: ") + \
                              msg + " " + expr.ToString();                  \
        throw std::runtime_error(err_msg);                                  \
    }

// if status of arrow::Result is not ok, form an err msg and raise a
// runtime_error with it. If it is ok, get value using ValueOrDie
// and assign it to lhs using std::move
#undef CHECK_ARROW_AND_ASSIGN
#define CHECK_ARROW_AND_ASSIGN(res, msg, lhs) \
    CHECK_ARROW(res.status(), msg)            \
    lhs = std::move(res).ValueOrDie();

constexpr int NUM_ICEBERG_DATA_FILE_STATS = 6;
constexpr int NUM_ICEBERG_FIELDS_WITH_FILENAME =
    NUM_ICEBERG_DATA_FILE_STATS + 1;

/**
 * @brief Converts a value to a little endian bytes object.
 *
 * @tparam T The C data type of the value to convert.
 * @param value The value to convert.
 * @return PyObject* A Python bytes object containing the little endian
 * encoding of the value.
 */
template <typename T>
PyObject *buffer_to_little_endian_bytes(T value) {
    const char *initial_bytes = reinterpret_cast<const char *>(&value);
    if constexpr (std::endian::native == std::endian::little) {
        return PyBytes_FromStringAndSize(initial_bytes, sizeof(T));
    } else {
        // Convert to little endian
        std::string str = std::string(initial_bytes, sizeof(T));
        std::ranges::reverse(str);
        const char *bytes = str.c_str();
        return PyBytes_FromStringAndSize(bytes, sizeof(T));
    }
}

/**
 * @brief Converts an Arrow scalar to a Python bytes object
 * that matches the scalar encoding from the Iceberg spec.
 * https://iceberg.apache.org/spec/#appendix-d-single-value-serialization
 *
 * @return PyObject* A Python bytes object containing the encoding of the
 * value according to the serialization rules in the Iceberg spec.
 */
PyObject *arrow_scalar_to_iceberg_bytes(std::shared_ptr<arrow::Scalar> scalar) {
    switch (scalar->type->id()) {
        case (arrow::Type::BOOL): {
            auto bool_scalar =
                std::static_pointer_cast<arrow::BooleanScalar>(scalar);
            bool value = bool_scalar->value;
            char byte_value = value ? 0x1 : 0x0;
            return PyBytes_FromStringAndSize(&byte_value, 1);
        }
        case (arrow::Type::INT32): {
            auto int32_scalar =
                std::static_pointer_cast<arrow::Int32Scalar>(scalar);
            int32_t value = int32_scalar->value;
            return buffer_to_little_endian_bytes<int32_t>(value);
        }
        case (arrow::Type::INT64): {
            auto int64_scalar =
                std::static_pointer_cast<arrow::Int64Scalar>(scalar);
            int64_t value = int64_scalar->value;
            return buffer_to_little_endian_bytes<int64_t>(value);
        }
        case (arrow::Type::FLOAT): {
            auto float_scalar =
                std::static_pointer_cast<arrow::FloatScalar>(scalar);
            float value = float_scalar->value;
            return buffer_to_little_endian_bytes<float>(value);
        }
        case (arrow::Type::DOUBLE): {
            auto double_scalar =
                std::static_pointer_cast<arrow::DoubleScalar>(scalar);
            double value = double_scalar->value;
            return buffer_to_little_endian_bytes<double>(value);
        }
        case (arrow::Type::DATE32): {
            auto date_scalar =
                std::static_pointer_cast<arrow::Date32Scalar>(scalar);
            int32_t value = date_scalar->value;
            return buffer_to_little_endian_bytes<int32_t>(value);
        }
        case (arrow::Type::TIME64): {
            auto time_scalar =
                std::static_pointer_cast<arrow::Time64Scalar>(scalar);
            int64_t value = time_scalar->value;
            return buffer_to_little_endian_bytes<int64_t>(value);
        }
        case (arrow::Type::TIMESTAMP): {
            auto timestamp_scalar =
                std::static_pointer_cast<arrow::TimestampScalar>(scalar);
            int64_t value = timestamp_scalar->value;
            return buffer_to_little_endian_bytes<int64_t>(value);
        }
        case (arrow::Type::STRING): {
            auto string_scalar =
                std::static_pointer_cast<arrow::StringScalar>(scalar);
            // Note: The string should already be in UTF-8 format
            return PyBytes_FromStringAndSize(
                reinterpret_cast<const char *>(string_scalar->value->data()),
                string_scalar->value->size());
        }
        case (arrow::Type::LARGE_STRING): {
            auto string_scalar =
                std::static_pointer_cast<arrow::LargeStringScalar>(scalar);
            // Note: The string should already be in UTF-8 format
            return PyBytes_FromStringAndSize(
                reinterpret_cast<const char *>(string_scalar->value->data()),
                string_scalar->value->size());
        }
        case (arrow::Type::BINARY): {
            auto binary_scalar =
                std::static_pointer_cast<arrow::BinaryScalar>(scalar);
            return PyBytes_FromStringAndSize(
                reinterpret_cast<const char *>(binary_scalar->value->data()),
                binary_scalar->value->size());
        }
        case (arrow::Type::LARGE_BINARY): {
            auto binary_scalar =
                std::static_pointer_cast<arrow::LargeBinaryScalar>(scalar);
            return PyBytes_FromStringAndSize(
                reinterpret_cast<const char *>(binary_scalar->value->data()),
                binary_scalar->value->size());
        }
        case (arrow::Type::DECIMAL128): {
            auto decimal_scalar =
                std::static_pointer_cast<arrow::Decimal128Scalar>(scalar);
            auto decimal_type =
                static_pointer_cast<arrow::DecimalType>(scalar->type);
            auto precision = decimal_type->precision();
            auto n_bytes = decimal_precision_to_integer_bytes(precision);
            auto skip_bytes = 16 - n_bytes;
            arrow::Decimal128 value = decimal_scalar->value;
            const char *initial_bytes = reinterpret_cast<const char *>(&value);
            if constexpr (std::endian::native == std::endian::big) {
                // For big endian, the bytes can be copied as is, but we should
                // skip the leading zeros and use the bytes starting at the
                // specified integer byte size based on the integer.
                return PyBytes_FromStringAndSize(initial_bytes + skip_bytes,
                                                 n_bytes);
            } else {
                // Otherwise, get the leading bytes and reverse to convert
                // to big endian.
                std::string str = std::string(initial_bytes, n_bytes);
                std::ranges::reverse(str);
                const char *bytes = str.c_str();
                return PyBytes_FromStringAndSize(bytes, n_bytes);
            }
        }
        default: {
            std::string err_msg = fmt::format(
                "Iceberg Parquet Write: Unsupported scalar type for "
                "lower_bound or upper_bound: {}",
                scalar->type->ToString());
            throw std::runtime_error(err_msg);
        }
    }
}

/**
 * @brief Convert a dictionary array into a format that can be used
 * to compute metrics like min and max. Dictionary arrays cannot directly
 * call compute functions because they are not supported in arrow and may
 * have value in the dictionary that are not used by any row in the
 * column. This function extracts the underlying dictionary and converts
 * any elements without matching index to null.
 *
 * @param dict_array The dictionary array from which to extract the data.
 * @return std::shared_ptr<arrow::ChunkedArray> A string array that can
 * be used to compute metrics like min and max.
 * @param dict_hits: A bitmaks indicating which indices are used.
 */
std::shared_ptr<arrow::ChunkedArray>
get_metrics_computable_array_for_dictionary(
    const std::shared_ptr<arrow::ChunkedArray> &dict_array,
    const std::shared_ptr<arrow::Buffer> &dict_hits) {
    // Dictionary arrays cannot directly call MinMax because
    // they are not supported. We need to convert them to
    // get the underlying data array and skip any values we
    // don't actually have in our table.
    // Unify all of the dictionaries. This should be
    // unnecessary, but we need to ensure we can prove this.
    arrow::Result<std::shared_ptr<::arrow::ChunkedArray>> unified_arr_res =
        arrow::DictionaryUnifier::UnifyChunkedArray(dict_array);
    std::shared_ptr<::arrow::ChunkedArray> unified_arr;
    CHECK_ARROW_AND_ASSIGN(unified_arr_res,
                           "Runtime error in chunk array "
                           "dictionary unification...",
                           unified_arr)
    // After unification, all chunks should have the same
    // dictionary
    // TODO Verify
    std::shared_ptr<arrow::Array> first_chunk = (unified_arr->chunk(0));
    std::shared_ptr<arrow::Array> dictionary =
        reinterpret_cast<arrow::DictionaryArray *>(first_chunk.get())
            ->dictionary();
    // We mark elements as seen/not seen by marking them
    // as null in the values we pass to arrow. The passed
    // in dict_hits bitmask serves this purpose.
    // Dictionary can be either string or large string.
    auto dict_type_id = dictionary->type()->id();
    std::shared_ptr<arrow::Array> new_array;
    if (dict_type_id == arrow::Type::STRING) {
        auto dict_string_array =
            std::dynamic_pointer_cast<arrow::StringArray>(dictionary);
        new_array = make_shared<arrow::StringArray>(
            dict_string_array->length(), dict_string_array->value_offsets(),
            dict_string_array->value_data(), dict_hits);
    } else {
        assert(dict_type_id == arrow::Type::LARGE_STRING);
        auto dict_string_array =
            std::dynamic_pointer_cast<arrow::LargeStringArray>(dictionary);
        new_array = make_shared<arrow::LargeStringArray>(
            dict_string_array->length(), dict_string_array->value_offsets(),
            dict_string_array->value_data(), dict_hits);
    }
    return make_shared<arrow::ChunkedArray>(new_array);
}

/**
 * @brief Compute the min and max for an iceberg field and update the
 * lower_bound_dict_py and upper_bound_dict_py dictionaries.
 *
 * @param[in] column Arrow chunked array for the field.
 * @param[in] field_id_py Python integer object for the field id.
 * @param[out] lower_bound_dict_py Python dictionary object for writing the
 * lower bound information for the field.
 * @param[out] upper_bound_dict_py Python dictionary object for writing the
 * upper bound information for the field.
 * @param[in] dict_hits: for dictionary encoded columns, which indices are used.
 */
void compute_min_max_iceberg_field(
    const std::shared_ptr<arrow::ChunkedArray> &column, PyObject *field_id_py,
    PyObject *lower_bound_dict_py, PyObject *upper_bound_dict_py,
    std::optional<std::shared_ptr<arrow::Buffer>> dict_hits = std::nullopt) {
    std::shared_ptr<arrow::ChunkedArray> chunked_array;
    if (column->type()->id() == arrow::Type::DICTIONARY) {
        chunked_array = get_metrics_computable_array_for_dictionary(
            column, dict_hits.value());
    } else {
        chunked_array = column;
    }
    auto minMaxRes = arrow::compute::MinMax(chunked_array);
    CHECK_ARROW(minMaxRes.status(), "Error in computing min/max");
    std::shared_ptr<arrow::Scalar> minMax =
        std::move(minMaxRes.ValueOrDie()).scalar();
    std::shared_ptr<arrow::StructScalar> minMaxStruct =
        std::static_pointer_cast<arrow::StructScalar>(minMax);
    std::shared_ptr<arrow::Scalar> min =
        minMaxStruct->field(arrow::FieldRef("min")).ValueOrDie();
    std::shared_ptr<arrow::Scalar> max =
        minMaxStruct->field(arrow::FieldRef("max")).ValueOrDie();
    // Convert to Python objects
    PyObject *min_object = arrow_scalar_to_iceberg_bytes(min);
    PyObject *max_object = arrow_scalar_to_iceberg_bytes(max);
    PyDict_SetItem(lower_bound_dict_py, field_id_py, min_object);
    PyDict_SetItem(upper_bound_dict_py, field_id_py, max_object);
    Py_DECREF(min_object);
    Py_DECREF(max_object);
}

/**
 * @brief Generate all metrics associated with an Iceberg field.
 * For types with nested data this may generate multiple metrics
 * for the children fields.
 *
 * @param[in] column Arrow chunked array for the field
 * @param[in] field Arrow field used to generate the field id information.
 * @param compute_min_and_max Whether to compute the min and max. This gets
 * modified for nested data.
 * @param[out] value_counts_dict_py Python dictionary object
 * for writing the value counts information for the field(s).
 * @param[out] null_count_dict_py Python dictionary object
 * for writing the null count information for the field(s).
 * @param[out] lower_bound_dict_py Python dictionary object
 * for writing the lower bound information for the field(s).
 * @param[out] upper_bound_dict_py Python dictionary object
 * for writing the upper bound information for the field(s).
 * @param[in] col_idx: the original column index from the table column
 * was fetched from.
 * @param[in] sketches: the theta sketches to update
 */
void generate_iceberg_field_metrics(
    const std::shared_ptr<arrow::ChunkedArray> &column,
    const std::shared_ptr<arrow::Field> &field, bool compute_min_and_max,
    PyObject *value_counts_dict_py, PyObject *null_count_dict_py,
    PyObject *lower_bound_dict_py, PyObject *upper_bound_dict_py,
    size_t col_idx, UpdateSketchCollection *sketches) {
    // Compute the value counts and null count for all columns. This differs
    // from Spark (which doesn't compute any statistics for anything but the
    // bottom-most level of nested arrays), but the null values may be useful
    // to us and it should be cheap to compute.
    int field_id = get_iceberg_field_id(field);
    PyObject *field_id_py = PyLong_FromLongLong(field_id);
    int64_t value_counts = column->length();
    PyObject *value_counts_py = PyLong_FromLongLong(value_counts);
    PyDict_SetItem(value_counts_dict_py, field_id_py, value_counts_py);
    Py_DECREF(value_counts_py);
    int64_t null_count = column->null_count();
    PyObject *null_count_py = PyLong_FromLongLong(null_count);
    PyDict_SetItem(null_count_dict_py, field_id_py, null_count_py);
    Py_DECREF(null_count_py);
    auto type_id = field->type()->id();
    if (type_id == arrow::Type::LIST) {
        // If we have a list we want to recurse onto the child values.
        std::vector<std::shared_ptr<arrow::Array>> chunks;
        std::transform(
            column->chunks().begin(), column->chunks().end(),
            std::back_inserter(chunks),
            [](std::shared_ptr<arrow::Array> chunk) {
                return std::static_pointer_cast<arrow::ListArray>(chunk)
                    ->values();
            });
        std::shared_ptr<arrow::ChunkedArray> child_column =
            std::make_shared<arrow::ChunkedArray>(chunks);
        std::shared_ptr<arrow::BaseListType> list_type =
            std::static_pointer_cast<arrow::BaseListType>(field->type());
        std::shared_ptr<arrow::Field> child_field = list_type->value_field();
        generate_iceberg_field_metrics(child_column, child_field, false,
                                       value_counts_dict_py, null_count_dict_py,
                                       lower_bound_dict_py, upper_bound_dict_py,
                                       -1, nullptr);
    } else if (type_id == arrow::Type::LARGE_LIST) {
        // If we have a large list we want to recurse onto the child values.
        std::vector<std::shared_ptr<arrow::Array>> chunks;
        std::transform(
            column->chunks().begin(), column->chunks().end(),
            std::back_inserter(chunks),
            [](std::shared_ptr<arrow::Array> chunk) {
                return std::static_pointer_cast<arrow::LargeListArray>(chunk)
                    ->values();
            });
        std::shared_ptr<arrow::ChunkedArray> child_column =
            std::make_shared<arrow::ChunkedArray>(chunks);
        std::shared_ptr<arrow::BaseListType> list_type =
            std::static_pointer_cast<arrow::BaseListType>(field->type());
        std::shared_ptr<arrow::Field> child_field = list_type->value_field();
        generate_iceberg_field_metrics(child_column, child_field, false,
                                       value_counts_dict_py, null_count_dict_py,
                                       lower_bound_dict_py, upper_bound_dict_py,
                                       -1, nullptr);
    } else if (type_id == arrow::Type::MAP) {
        // If we have a map we recurse onto the key and value.
        std::vector<std::shared_ptr<arrow::Array>> key_chunks;
        std::vector<std::shared_ptr<arrow::Array>> value_chunks;
        for (auto chunk : column->chunks()) {
            auto map_chunk = std::static_pointer_cast<arrow::MapArray>(chunk);
            key_chunks.push_back(map_chunk->keys());
            value_chunks.push_back(map_chunk->items());
        }
        std::shared_ptr<arrow::ChunkedArray> key_column =
            std::make_shared<arrow::ChunkedArray>(key_chunks);
        std::shared_ptr<arrow::MapType> map_type =
            std::static_pointer_cast<arrow::MapType>(field->type());

        std::shared_ptr<arrow::Field> key_field = map_type->key_field();
        generate_iceberg_field_metrics(key_column, key_field, false,
                                       value_counts_dict_py, null_count_dict_py,
                                       lower_bound_dict_py, upper_bound_dict_py,
                                       -1, nullptr);
        std::shared_ptr<arrow::ChunkedArray> value_column =
            std::make_shared<arrow::ChunkedArray>(value_chunks);
        std::shared_ptr<arrow::Field> value_field = map_type->item_field();
        generate_iceberg_field_metrics(value_column, value_field, false,
                                       value_counts_dict_py, null_count_dict_py,
                                       lower_bound_dict_py, upper_bound_dict_py,
                                       -1, nullptr);
    } else if (type_id == arrow::Type::STRUCT) {
        // If we have a struct we recurse onto the children.
        std::shared_ptr<arrow::StructType> struct_type =
            std::static_pointer_cast<arrow::StructType>(field->type());
        std::vector<std::vector<std::shared_ptr<arrow::Array>>> child_chunks(
            struct_type->num_fields());
        for (auto chunk : column->chunks()) {
            auto struct_chunk =
                std::static_pointer_cast<arrow::StructArray>(chunk);
            for (int i = 0; i < struct_chunk->num_fields(); i++) {
                child_chunks[i].push_back(struct_chunk->field(i));
            }
        }
        for (int i = 0; i < struct_type->num_fields(); i++) {
            std::shared_ptr<arrow::Field> child_field = struct_type->field(i);
            std::shared_ptr<arrow::ChunkedArray> child_column =
                std::make_shared<arrow::ChunkedArray>(child_chunks[i]);
            // Structs should still compute the min and max.
            generate_iceberg_field_metrics(
                child_column, child_field, compute_min_and_max,
                value_counts_dict_py, null_count_dict_py, lower_bound_dict_py,
                upper_bound_dict_py, -1, nullptr);
        }
    } else if (type_id == arrow::Type::DICTIONARY) {
        // Pre-compute the bitmask indicating which dict
        // indices are used.
        arrow::Result<std::shared_ptr<::arrow::ChunkedArray>> unified_arr_res =
            arrow::DictionaryUnifier::UnifyChunkedArray(column);
        std::shared_ptr<::arrow::ChunkedArray> unified_arr;
        CHECK_ARROW_AND_ASSIGN(unified_arr_res, "UnifyChunkedArray",
                               unified_arr);
        auto dict_hits = get_dictionary_hits(unified_arr);
        if (sketches != nullptr) {
            sketches->update_sketch(unified_arr, col_idx, dict_hits);
        }
        // Generate max and min if they exist.
        if (compute_min_and_max && null_count != value_counts) {
            compute_min_max_iceberg_field(unified_arr, field_id_py,
                                          lower_bound_dict_py,
                                          upper_bound_dict_py, dict_hits);
        }
    } else {
        if (sketches != nullptr) {
            sketches->update_sketch(column, col_idx);
        }
        // Generate max and min if they exist.
        if (compute_min_and_max && null_count != value_counts) {
            compute_min_max_iceberg_field(
                column, field_id_py, lower_bound_dict_py, upper_bound_dict_py);
        }
    }
    // Decref the field IDs so there is 1 reference per dictionary.
    Py_DECREF(field_id_py);
}

/**
 * @brief Generate a random file name for Iceberg Table write.
 * @return std::string (a random filename of form {rank:05}-rank-{uuid}.parquet)
 */
std::string generate_iceberg_file_name() {
    int rank = dist_get_rank();
    boost::uuids::uuid _uuid = boost::uuids::random_generator()();
    std::string uuid = boost::uuids::to_string(_uuid);
    int check;
    std::vector<char> fname;
    // 5+1+5+1+uuid+8
    fname.resize(20 + uuid.length());
    // The file name format is based on Spark (hence the double usage of rank in
    // the name)
    check = snprintf(fname.data(), fname.size(), "%05d-%d-%s.parquet", rank,
                     rank, uuid.c_str());
    if (size_t(check + 1) > fname.size()) {
        throw std::runtime_error(
            "Fatal error: number of written char for iceberg file name is "
            "greater than fname size");
    }
    return std::string(fname.data());
}

/**
 * @brief Update an arrow table derived based on the expected Iceberg
 * table schema.
 * @param table Arrow table to be updated
 * @param iceberg_schema Expected Arrow schema of files written to Iceberg.
 * @return A new arrow table with the updated schema.
 */
std::shared_ptr<arrow::Table> cast_arrow_table_to_iceberg_schema(
    std::shared_ptr<arrow::Table> input_table,
    std::shared_ptr<arrow::Schema> iceberg_schema) {
    std::vector<std::shared_ptr<arrow::ChunkedArray>> arrow_columns;
    std::vector<std::shared_ptr<arrow::Field>> schema_vector;
    for (int i = 0; i < input_table->num_columns(); i++) {
        auto expected_field = iceberg_schema->field(i);
        auto arrow_field = input_table->field(i);
        auto column = input_table->column(i);
        if (!expected_field->nullable() && column->null_count() != 0) {
            std::string err_msg =
                std::string("Iceberg Parquet Write: Column ") +
                arrow_field->name() +
                " contains nulls but is expected to be non-nullable";
            throw std::runtime_error(err_msg);
        }

        auto expected_type = iceberg_schema->field(i)->type();
        auto arrow_type = arrow_field->type();
        auto dest_type = arrow_type;
        if (expected_type->id() != arrow::Type::LARGE_STRING) {
            // String types must keep the same type to avoid dictionary encoding
            // issues.
            dest_type = expected_type;
        }

        // Check if the arrow field type matches the expected type. This
        // is necessary to ensure we perform downcasting.
        //
        // Note: We may be able to skip upcasts in the future.
        if (!arrow_type->Equals(dest_type)) {
            std::vector<std::shared_ptr<arrow::Array>> chunks;
            std::transform(
                column->chunks().begin(), column->chunks().end(),
                std::back_inserter(chunks),
                [arrow_field, arrow_type,
                 dest_type](std::shared_ptr<arrow::Array> chunk) {
                    auto res = arrow::compute::Cast(
                        *chunk.get(), dest_type,
                        arrow::compute::CastOptions::Safe(),
                        bodo::default_buffer_exec_context());
                    if (!res.ok()) {
                        std::string err_msg =
                            std::string("Iceberg Parquet Write: Column ") +
                            arrow_field->name() + " is type " +
                            arrow_type->ToString() +
                            " but is expected to be type " +
                            dest_type->ToString();
                        throw std::runtime_error(err_msg);
                    }
                    return res.ValueOrDie();
                });
            column = std::make_shared<arrow::ChunkedArray>(chunks);
        }
        arrow_columns.emplace_back(column);
        schema_vector.emplace_back(arrow::field(arrow_field->name(), dest_type,
                                                expected_field->nullable(),
                                                expected_field->metadata()));
    }
    return arrow::Table::Make(
        arrow::schema(schema_vector, input_table->schema()->metadata()),
        arrow_columns, input_table->num_rows());
}

/**
 * Write the Bodo table (the chunk in this process) to a parquet file
 * as part of an Iceberg table
 * @param fpath full path of the parquet file to write
 * @param table table to write to parquet file
 * @param col_names_arr array containing the table's column names (index not
 * included)
 * @param compression compression scheme to use
 * @param is_parallel true if the table is part of a distributed table
 * @param bucket_region in case of S3, this is the region the bucket is in
 * @param row_group_size Row group size in number of rows
 * @param iceberg_metadata Iceberg metadata string to be added to the parquet
 * schema's metadata with key 'iceberg.schema'
 * @param iceberg_arrow_schema Iceberg schema in Arrow format that written
 * Parquet files should match
 * @param sketches collection of theta sketches accumulating ndv for the table
 * as it is being written.
 * @param[out] record_count Number of records in this file
 * @param[out] file_size_in_bytes Size of the file in bytes
 * @return A vector of Python Objects representing the metadata information
 * that needs to be written back to the data file. If there is no metadata
 * this returns an empty vector.
 */
std::vector<PyObject *> iceberg_pq_write_helper(
    const char *fpath, const std::shared_ptr<table_info> table,
    const std::vector<std::string> col_names, const char *compression,
    bool is_parallel, const char *bucket_region, int64_t row_group_size,
    const char *iceberg_metadata,
    std::shared_ptr<arrow::Schema> iceberg_arrow_schema,
    std::shared_ptr<arrow::fs::FileSystem> arrow_fs,
    UpdateSketchCollection *sketches) {
    std::unordered_map<std::string, std::string> md = {
        {"iceberg.schema", std::string(iceberg_metadata)}};

    // For Iceberg, all timestamp data needs to be written
    // as microseconds, so that's the type we
    // specify. `pq_write` will convert the nanoseconds to
    // microseconds during `bodo_array_to_arrow`.
    // See https://iceberg.apache.org/spec/#primitive-types,
    // https://iceberg.apache.org/spec/#parquet.
    // We've also made the decision to always
    // write the `timestamptz` type when writing
    // Iceberg data, similar to Spark.
    // The underlying already is in UTC already
    // for timezone aware types, and for timezone
    // naive, it won't matter.

    std::shared_ptr<arrow::KeyValueMetadata> schema_metadata =
        ::arrow::key_value_metadata(md);
    std::shared_ptr<arrow::Table> arrow_table = bodo_table_to_arrow(
        table, std::move(col_names), schema_metadata,
        false /*convert_timedelta_to_int64*/, "UTC", arrow::TimeUnit::MICRO,
        false /*downcast_time_ns_to_us*/
    );
    std::shared_ptr<arrow::Table> iceberg_table =
        cast_arrow_table_to_iceberg_schema(std::move(arrow_table),
                                           std::move(iceberg_arrow_schema));
    std::vector<bodo_array_type::arr_type_enum> bodo_array_types;
    for (auto col : table->columns) {
        bodo_array_types.push_back(col->arr_type);
    }
    // Convert the arrow table to use the iceberg schema.
    int64_t file_size_in_bytes =
        pq_write(fpath, iceberg_table, compression, is_parallel, bucket_region,
                 row_group_size, "", bodo_array_types, true, std::string(fpath),
                 arrow_fs.get());
    int64_t record_count = iceberg_table->num_rows();
    if (record_count == 0) {
        return {};
    } else {
        std::vector<PyObject *> iceberg_py_objs(NUM_ICEBERG_DATA_FILE_STATS);
        PyObject *record_count_py = PyLong_FromLongLong(record_count);
        iceberg_py_objs[0] = record_count_py;
        PyObject *file_size_in_bytes_py =
            PyLong_FromLongLong(file_size_in_bytes);
        iceberg_py_objs[1] = file_size_in_bytes_py;
        // Generate per column stats.
        PyObject *value_counts_dict_py = PyDict_New();
        iceberg_py_objs[2] = value_counts_dict_py;
        PyObject *null_count_dict_py = PyDict_New();
        iceberg_py_objs[3] = null_count_dict_py;
        PyObject *lower_bound_dict_py = PyDict_New();
        iceberg_py_objs[4] = lower_bound_dict_py;
        PyObject *upper_bound_dict_py = PyDict_New();
        iceberg_py_objs[5] = upper_bound_dict_py;
        for (int i = 0; i < iceberg_table->num_columns(); i++) {
            auto column = iceberg_table->column(i);
            auto field = iceberg_table->schema()->field(i);
            generate_iceberg_field_metrics(
                column, field, true, value_counts_dict_py, null_count_dict_py,
                lower_bound_dict_py, upper_bound_dict_py, i, sketches);
        }
        return iceberg_py_objs;
    }
}

/**
 * @brief Main function for Iceberg write which can handle partition-spec
 * and sort-orders.
 *
 * @param table_data_loc Location of the Iceberg warehouse (the data folder)
 * @param table Bodo table to write
 * @param col_names_arr array containing the table's column names (index not
 * included)
 * @param partition_spec Python list of tuples containing description of the
 * partition fields (in the order that the partitions should be applied)
 * @param sort_order Python list of tuples containing description of the
 * sort fields (in the order that the sort should be performed)
 * @param compression compression scheme to use
 * @param is_parallel true if the table is part of a distributed table
 * @param bucket_region in case of S3, this is the region the bucket is in
 * @param row_group_size Row group size in number of rows
 * @param iceberg_metadata Iceberg metadata string to be added to the parquet
 * schema's metadata with key 'iceberg.schema'
 * @param[out] iceberg_files_info_py List of tuples for each of the files
 consisting of (file_name, record_count, file_size, *partition_values). Should
 be passed in as an empty list which will be filled during execution.
 * @param iceberg_schema Expected Arrow schema of files written to Iceberg
 table, if given.
* @param arrow_fs Arrow file system object for writing the data, can be nullptr
in which case the filesystem is inferred from the path.
 * @param sketches collection of theta sketches accumulating ndv for the table
 * as it is being written.
 */
void iceberg_pq_write(
    const char *table_data_loc, std::shared_ptr<table_info> table,
    const std::vector<std::string> col_names_arr, PyObject *partition_spec,
    PyObject *sort_order, const char *compression, bool is_parallel,
    const char *bucket_region, int64_t row_group_size,
    const char *iceberg_metadata, PyObject *iceberg_files_info_py,
    std::shared_ptr<arrow::Schema> iceberg_schema,
    std::shared_ptr<arrow::fs::FileSystem> arrow_fs, void *sketches_ptr) {
    tracing::Event ev("iceberg_pq_write", is_parallel);
    ev.add_attribute("table_data_loc", table_data_loc);
    ev.add_attribute("iceberg_metadata", iceberg_metadata);
    if (!PyList_Check(iceberg_files_info_py)) {
        throw std::runtime_error(
            "IcebergParquetWrite: iceberg_files_info_py is not a list");
    } else if (!PyList_Check(sort_order)) {
        throw std::runtime_error(
            "IcebergParquetWrite: sort_order is not a list");
    } else if (!PyList_Check(partition_spec)) {
        throw std::runtime_error(
            "IcebergParquetWrite: partition_spec is not a list");
    }

    // Create dummy sketches if not provided.
    UpdateSketchCollection *sketches =
        (sketches_ptr == nullptr)
            ? new UpdateSketchCollection(
                  std::vector<bool>(table->ncols(), false))
            : static_cast<UpdateSketchCollection *>(sketches_ptr);

    std::shared_ptr<table_info> working_table = table;

    // If sort order, then iterate over and create the transforms.
    // Sort Order should be a list of tuples.
    // Each tuple is of the form:
    // (int64_t column_idx, string transform_name,
    //  int64_t arg, int64_t asc, int64_t null_last).
    // column_idx is the position in table.
    // transform name is one of 'identity', 'bucket', 'truncate'
    // 'year', 'month', 'day', 'hour', 'void'.
    // arg is N (number of buckets) for bucket transform,
    // W (width) for truncate transform, and 0 otherwise.
    // asc when sort direction is ascending.
    // na_last when nulls should be last.
    if (PyList_Size(sort_order) > 0) {
        int64_t sort_order_len = PyList_Size(sort_order);
        tracing::Event ev_sort("iceberg_pq_write_sort", is_parallel);
        ev_sort.add_attribute("sort_order_len", sort_order_len);
        // Vector to collect the transformed columns
        std::vector<std::shared_ptr<array_info>> transform_cols;
        transform_cols.reserve(sort_order_len);
        // Vector of ints (booleans essentially) to be
        // eventually passed to sort_values_table
        std::vector<int64_t> vect_ascending;
        vect_ascending.reserve(sort_order_len);
        std::vector<int64_t> na_position;
        na_position.reserve(sort_order_len);

        // Iterate over the python list describing the sort order
        // and create the transform columns.
        PyObject *sort_order_iter = PyObject_GetIter(sort_order);
        PyObject *sort_order_field_tuple;
        while ((sort_order_field_tuple = PyIter_Next(sort_order_iter))) {
            // PyTuple_GetItem returns borrowed reference, no need to decref
            PyObject *col_idx_py = PyTuple_GetItem(sort_order_field_tuple, 0);
            int64_t col_idx = PyLong_AsLongLong(col_idx_py);

            // PyTuple_GetItem returns borrowed reference, no need to decref
            PyObject *transform_name_py =
                PyTuple_GetItem(sort_order_field_tuple, 1);
            PyObject *transform_name_str =
                PyUnicode_AsUTF8String(transform_name_py);
            const char *transform_name_ = PyBytes_AS_STRING(transform_name_str);
            std::string transform_name(transform_name_);
            // PyBytes_AS_STRING returns the internal buffer without copy, so
            // decref should happen after the data is copied to std::string.
            // https:// docs.python.org/3/c-api/bytes.html#c.PyBytes_AsString
            Py_DECREF(transform_name_str);

            // PyTuple_GetItem returns borrowed reference, no need to decref
            PyObject *arg_py = PyTuple_GetItem(sort_order_field_tuple, 2);
            int64_t arg = PyLong_AsLongLong(arg_py);

            // PyTuple_GetItem returns borrowed reference, no need to decref
            PyObject *asc_py = PyTuple_GetItem(sort_order_field_tuple, 3);
            int64_t asc = PyLong_AsLongLong(asc_py);
            vect_ascending.push_back(asc);

            // PyTuple_GetItem returns borrowed reference, no need to decref
            PyObject *null_last_py = PyTuple_GetItem(sort_order_field_tuple, 4);
            int64_t null_last = PyLong_AsLongLong(null_last_py);
            na_position.push_back(null_last);

            std::shared_ptr<array_info> transform_col;
            if (transform_name == "identity") {
                transform_col = iceberg_identity_transform(
                    working_table->columns[col_idx], is_parallel);
            } else {
                transform_col =
                    iceberg_transform(working_table->columns[col_idx],
                                      transform_name, arg, is_parallel);
            }
            transform_cols.push_back(transform_col);

            Py_DECREF(sort_order_field_tuple);
        }
        Py_DECREF(sort_order_iter);

        // Create a new table (a copy) with the transforms and then sort it.
        // The transformed columns go in front since they are the keys.
        std::vector<std::shared_ptr<array_info>> new_cols;
        new_cols.reserve(transform_cols.size() + working_table->columns.size());
        new_cols.insert(new_cols.end(), transform_cols.begin(),
                        transform_cols.end());
        new_cols.insert(new_cols.end(), working_table->columns.begin(),
                        working_table->columns.end());
        std::shared_ptr<table_info> new_table =
            std::make_shared<table_info>(new_cols);

        // Make all local dictionaries unique
        // NOTE: can't make them global since streaming write isn't bulk
        // synchronous
        for (auto a : new_cols) {
            if (a->arr_type == bodo_array_type::DICT) {
                // Note: Sorting is an optimization but not required. This
                // is equivalent to the parquet_write
                drop_duplicates_local_dictionary(a, true);
            }
        }

        // NOTE: we can't sort in parallel since streaming write is not bulk
        // synchronous
        // TODO[BSE-2614]: explore adding a global sort in the planner if table
        // has sort order
        std::shared_ptr<table_info> sorted_new_table = sort_values_table(
            new_table, transform_cols.size(), vect_ascending.data(),
            na_position.data(), /*dead_keys=*/nullptr, /*out_n_rows=*/nullptr,
            /*bounds=*/nullptr, false);

        // Remove the unused transform columns
        // TODO Optimize to not remove if they can be reused for partition spec
        sorted_new_table->columns.erase(
            sorted_new_table->columns.begin(),
            sorted_new_table->columns.begin() + transform_cols.size());

        transform_cols.clear();
        // Set working table for the subsequent steps
        working_table = sorted_new_table;
        ev_sort.finalize();
    }

    // If partition spec, iterate over and create the transforms
    // Partition Spec should be a list of tuples.
    // Each tuple is of the form:
    // (int64_t column_idx, string transform_name,
    //  int64_t arg, string partition_name).
    // column_idx is the position in table.
    // transform name is one of 'identity', 'bucket', 'truncate'
    // 'year', 'month', 'day', 'hour', 'void'.
    // arg is N (number of buckets) for bucket transform,
    // W (width) for truncate transform, and 0 otherwise.
    // partition_name is the name to set for folder (e.g.
    // /<partition_name>=<value>/)
    if (PyList_Size(partition_spec) > 0) {
        int64_t partition_spec_len = PyList_Size(partition_spec);
        tracing::Event ev_part("iceberg_pq_write_partition_spec", is_parallel);
        ev_part.add_attribute("partition_spec_len", partition_spec_len);
        std::vector<std::string> partition_names;
        partition_names.reserve(partition_spec_len);
        std::vector<int64_t> part_col_indices;
        part_col_indices.reserve(partition_spec_len);
        // Similar to the ones in sort-order handling
        std::vector<std::string> transform_names;
        transform_names.reserve(partition_spec_len);
        std::vector<std::shared_ptr<array_info>> transform_cols;
        transform_cols.reserve(partition_spec_len);

        // Iterate over the partition fields and create the transform
        // columns
        PyObject *partition_spec_iter = PyObject_GetIter(partition_spec);
        PyObject *partition_spec_field_tuple;
        while (
            (partition_spec_field_tuple = PyIter_Next(partition_spec_iter))) {
            // PyTuple_GetItem returns borrowed reference, no need to decref
            PyObject *col_idx_py =
                PyTuple_GetItem(partition_spec_field_tuple, 0);
            int64_t col_idx = PyLong_AsLongLong(col_idx_py);

            // PyTuple_GetItem returns borrowed reference, no need to decref
            PyObject *transform_name_py =
                PyTuple_GetItem(partition_spec_field_tuple, 1);
            PyObject *transform_name_str =
                PyUnicode_AsUTF8String(transform_name_py);
            const char *transform_name_ = PyBytes_AS_STRING(transform_name_str);
            std::string transform_name(transform_name_);
            // PyBytes_AS_STRING returns the internal buffer without copy, so
            // decref should happen after the data is copied to std::string.
            // https:// docs.python.org/3/c-api/bytes.html#c.PyBytes_AsString
            Py_DECREF(transform_name_str);
            transform_names.push_back(transform_name);

            // PyTuple_GetItem returns borrowed reference, no
            // need to decref
            PyObject *arg_py = PyTuple_GetItem(partition_spec_field_tuple, 2);
            int64_t arg = PyLong_AsLongLong(arg_py);

            // PyTuple_GetItem returns borrowed reference, no need to decref
            PyObject *partition_name_py =
                PyTuple_GetItem(partition_spec_field_tuple, 3);
            PyObject *partition_name_str =
                PyUnicode_AsUTF8String(partition_name_py);
            const char *partition_name_ = PyBytes_AS_STRING(partition_name_str);
            std::string partition_name(partition_name_);
            // PyBytes_AS_STRING returns the internal buffer without copy, so
            // decref should happen after the data is copied to std::string.
            // https:// docs.python.org/3/c-api/bytes.html#c.PyBytes_AsString
            Py_DECREF(partition_name_str);
            partition_names.push_back(partition_name);

            std::shared_ptr<array_info> transform_col;
            if (transform_name == "identity") {
                transform_col = iceberg_identity_transform(
                    working_table->columns[col_idx], is_parallel);
            } else {
                transform_col =
                    iceberg_transform(working_table->columns[col_idx],
                                      transform_name, arg, is_parallel);
                // Always free in case of non identity transform
            }
            transform_cols.push_back(transform_col);
            part_col_indices.push_back(col_idx);

            Py_DECREF(partition_spec_field_tuple);
        }
        Py_DECREF(partition_spec_iter);

        // Make all local dictionaries unique to enable hashing.
        // This can only happen in the case of identity transform or
        // truncate transform on DICT arrays.
        // NOTE: can't make them global since streaming write isn't bulk
        // synchronous
        for (auto a : transform_cols) {
            if (a->arr_type == bodo_array_type::DICT) {
                drop_duplicates_local_dictionary(a);
            }
        }

        // Create a new table (a copy) with the transforms.
        // new_table will have transformed partition columns at the beginning
        // and the rest after (to use multi_col_key for hashing which assumes
        // that keys are at the beginning), and we will then drop the
        // transformed columns from it for writing
        std::vector<std::shared_ptr<array_info>> new_cols;
        new_cols.reserve(transform_cols.size() + working_table->columns.size());
        new_cols.insert(new_cols.end(), transform_cols.begin(),
                        transform_cols.end());
        new_cols.insert(new_cols.end(), working_table->columns.begin(),
                        working_table->columns.end());
        std::shared_ptr<table_info> new_table =
            std::make_shared<table_info>(new_cols);

        // XXX Some of the code below is similar to that in
        // pq_write_partitioned_py_entry, and should be refactored.

        // Create partition keys and "parts", populate partition_write_info
        // structs
        tracing::Event ev_part_gen("iceberg_pq_write_partition_spec_gen_parts",
                                   is_parallel);
        const uint32_t seed = SEED_HASH_PARTITION;
        std::shared_ptr<uint32_t[]> hashes =
            hash_keys(transform_cols, seed, is_parallel, false);
        bodo::unord_map_container<multi_col_key, partition_write_info,
                                  multi_col_key_hash>
            key_to_partition;

        const int64_t num_keys = transform_cols.size();
        for (uint64_t i = 0; i < new_table->nrows(); i++) {
            multi_col_key key(hashes[i], new_table, i, num_keys);
            partition_write_info &p = key_to_partition[key];
            if (p.rows.size() == 0) {
                // This is the path after the table_loc that will
                // be returned to Python.
                std::string inner_path = "";

                // We store the iceberg-file-info as part of the
                // partition_write_info struct. It's a tuple of length
                // 3 (for file_name, record_count and file_size) + number of
                // partition fields (same as number of transform cols).
                // See function description string for more details.
                p.iceberg_file_info_py =
                    PyTuple_New(NUM_ICEBERG_FIELDS_WITH_FILENAME + num_keys);

                for (int64_t j = 0; j < num_keys; j++) {
                    auto transformed_part_col = transform_cols[j];
                    // convert transformed partition value to string
                    std::string value_str = transform_val_to_str(
                        transform_names[j],
                        new_table->columns[part_col_indices[j] +
                                           transform_names.size()],
                        transformed_part_col, i);
                    inner_path += partition_names[j] + "=" + value_str + "/";
                    // Get python representation of the partition value
                    // and then add it to the iceberg-file-info tuple.
                    PyObject *partition_val_py =
                        iceberg_transformed_val_to_py(transformed_part_col, i);
                    PyTuple_SET_ITEM(p.iceberg_file_info_py,
                                     NUM_ICEBERG_FIELDS_WITH_FILENAME + j,
                                     partition_val_py);
                }
                // create a random file name
                inner_path += generate_iceberg_file_name();
                PyObject *file_name_py =
                    PyUnicode_FromString(inner_path.c_str());
                PyTuple_SET_ITEM(p.iceberg_file_info_py, 0, file_name_py);

                // Generate output file name
                // TODO: Make our path handling more consistent between C++ and
                // Java
                p.fpath = std::string(table_data_loc);
                if (p.fpath.back() != '/') {
                    p.fpath += "/";
                }
                p.fpath += inner_path;
            }
            p.rows.push_back(i);
        }
        hashes.reset();
        ev_part_gen.finalize();

        // Remove the unused transform columns
        // Note that we can remove even the identity transforms
        // since they were just copies (pointer copy)
        new_table->columns.erase(
            new_table->columns.begin(),
            new_table->columns.begin() + transform_cols.size());

        transform_cols.clear();

        tracing::Event ev_part_write(
            "iceberg_pq_write_partition_spec_write_parts", is_parallel);
        // Write the file for each partition key
        for (auto &it : key_to_partition) {
            const partition_write_info &p = it.second;
            std::shared_ptr<table_info> part_table =
                RetrieveTable(new_table, p.rows, new_table->ncols());
            // NOTE: we pass is_parallel=False because we already took care of
            // is_parallel here
            Bodo_Fs::FsEnum fs_type = filesystem_type(p.fpath.c_str());
            if (fs_type == Bodo_Fs::FsEnum::posix) {
                // s3 and hdfs create parent directories automatically when
                // writing partitioned columns
                std::filesystem::path path = p.fpath;
                std::filesystem::create_directories(path.parent_path());
            }

            // Write the file and then attach the relevant information
            // to the iceberg-file-info tuple for this file.
            // We don't need to check if record count is zero in this case.
            std::vector<PyObject *> iceberg_py_objs = iceberg_pq_write_helper(
                p.fpath.c_str(), part_table, col_names_arr, compression, false,
                bucket_region, row_group_size, iceberg_metadata, iceberg_schema,
                arrow_fs, sketches);

            for (int i = 0; i < NUM_ICEBERG_DATA_FILE_STATS; i++) {
                PyObject *obj = iceberg_py_objs[i];
                PyTuple_SET_ITEM(p.iceberg_file_info_py, i + 1, obj);
            }
            PyList_Append(iceberg_files_info_py, p.iceberg_file_info_py);
            Py_DECREF(p.iceberg_file_info_py);
        }
        ev_part_write.finalize();
        ev_part.finalize();
    } else {
        tracing::Event ev_general("iceberg_pq_write_general", is_parallel);
        // If no partition spec, then write the working table (after sort if
        // there was a sort order else the table as is)
        std::string fname = generate_iceberg_file_name();
        std::string fpath(table_data_loc);
        if (fpath.back() != '/') {
            fpath += "/";
        }
        fpath += fname;
        Bodo_Fs::FsEnum fs_type = filesystem_type(fpath.c_str());
        if (fs_type == Bodo_Fs::FsEnum::posix) {
            // s3 and hdfs create parent directories automatically when
            // writing partitioned columns
            std::filesystem::path path = fpath;
            std::filesystem::create_directories(path.parent_path());
        }
        // XXX Should we pass is_parallel instead of always false?
        // We can't at the moment due to how pq_write works. Once we
        // refactor that a little, we should be able to handle this
        // more elegantly.
        std::vector<PyObject *> iceberg_py_objs = iceberg_pq_write_helper(
            fpath.c_str(), working_table, col_names_arr, compression, false,
            bucket_region, row_group_size, iceberg_metadata, iceberg_schema,
            arrow_fs, sketches);

        if (iceberg_py_objs.size() > 0) {
            PyObject *file_name_py = PyUnicode_FromString(fname.c_str());

            PyObject *iceberg_file_info_py =
                PyTuple_New(NUM_ICEBERG_FIELDS_WITH_FILENAME);
            PyTuple_SET_ITEM(iceberg_file_info_py, 0, file_name_py);
            for (int i = 0; i < NUM_ICEBERG_DATA_FILE_STATS; i++) {
                PyObject *obj = iceberg_py_objs[i];
                PyTuple_SET_ITEM(iceberg_file_info_py, i + 1, obj);
            }
            PyList_Append(iceberg_files_info_py, iceberg_file_info_py);
            Py_DECREF(iceberg_file_info_py);
        }
        ev_general.finalize();
    }

    // Clean up the dummy sketches if we created them
    if (sketches_ptr == nullptr) {
        delete sketches;
    }
}

/**
 * @brief Python entrypoint for the iceberg write function
 * with error handling. Since write occurs in parallel, exceptions are
 * captured and propagated to ensure all ranks raise.
 */
PyObject *iceberg_pq_write_py_entry(
    const char *table_data_loc, table_info *in_table,
    array_info *in_col_names_arr, PyObject *partition_spec,
    PyObject *sort_order, const char *compression, bool is_parallel,
    const char *bucket_region, int64_t row_group_size,
    const char *iceberg_metadata, PyObject *iceberg_arrow_schema_py,
    PyObject *arrow_fs, void *sketches_ptr) {
    try {
        std::shared_ptr<table_info> table =
            std::shared_ptr<table_info>(in_table);
        std::shared_ptr<array_info> col_names_arr =
            std::shared_ptr<array_info>(in_col_names_arr);

        // Python list of tuples describing the data files written.
        // iceberg_pq_write will append to the list and then this will be
        // returned to the Python.
        PyObject *iceberg_files_info_py = PyList_New(0);

        if (arrow::py::import_pyarrow_wrappers()) {
            throw std::runtime_error("Importing pyarrow_wrappers failed!");
        }

        std::shared_ptr<arrow::Schema> iceberg_schema;
        CHECK_ARROW_AND_ASSIGN(
            arrow::py::unwrap_schema(iceberg_arrow_schema_py),
            "Iceberg Schema Couldn't Unwrap from Python", iceberg_schema);

        std::shared_ptr<arrow::fs::FileSystem> fs;
        CHECK_ARROW_AND_ASSIGN(
            arrow::py::unwrap_filesystem(arrow_fs),
            "Error during Iceberg write: Failed to unwrap Arrow filesystem",
            fs);

        iceberg_pq_write(
            table_data_loc, table, array_to_string_vector(col_names_arr),
            partition_spec, sort_order, compression, is_parallel, bucket_region,
            row_group_size, iceberg_metadata, iceberg_files_info_py,
            iceberg_schema, fs, sketches_ptr);

        return iceberg_files_info_py;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}
