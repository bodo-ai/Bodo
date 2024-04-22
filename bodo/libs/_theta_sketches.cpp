#include "_theta_sketches.h"
#include <arrow/python/api.h>
#include <mpi.h>
#include "_shuffle.h"

theta_sketch_collection_t init_theta_sketches(
    const std::vector<bool> &ndv_cols) {
    size_t n_cols = ndv_cols.size();
    theta_sketch_collection_t result =
        new std::optional<datasketches::update_theta_sketch>[n_cols];
    for (size_t col_idx = 0; col_idx < n_cols; col_idx++) {
        if (ndv_cols[col_idx]) {
            result[col_idx] =
                datasketches::update_theta_sketch::builder().build();
        } else {
            result[col_idx] = std::nullopt;
        }
    }
    return result;
}

inline void insert_numeric_buffer_theta_sketch(
    datasketches::update_theta_sketch &sketch, const char *initial_bytes,
    size_t n_bytes) {
    if constexpr (std::endian::native == std::endian::little) {
        // If already little endian, insert those bytes
        sketch.update(initial_bytes, n_bytes);
    } else {
        // Convert to little endian
        std::string str = std::string(initial_bytes, n_bytes);
        std::reverse(std::begin(str), std::end(str));
        const char *bytes = str.c_str();
        sketch.update(bytes, n_bytes);
    }
}

template <arrow::Type::type ArrowType>
    requires(ArrowType == arrow::Type::INT32)
void insert_value_theta_sketch(const std::shared_ptr<arrow::Scalar> &scalar,
                               datasketches::update_theta_sketch &sketch) {
    auto int32_scalar = std::static_pointer_cast<arrow::Int32Scalar>(scalar);
    int32_t value = int32_scalar->value;
    const char *initial_bytes = reinterpret_cast<const char *>(&value);
    insert_numeric_buffer_theta_sketch(sketch, initial_bytes, sizeof(int32_t));
}

template <arrow::Type::type ArrowType>
    requires(ArrowType == arrow::Type::INT64)
void insert_value_theta_sketch(const std::shared_ptr<arrow::Scalar> &scalar,
                               datasketches::update_theta_sketch &sketch) {
    auto int64_scalar = std::static_pointer_cast<arrow::Int64Scalar>(scalar);
    int64_t value = int64_scalar->value;
    const char *initial_bytes = reinterpret_cast<const char *>(&value);
    insert_numeric_buffer_theta_sketch(sketch, initial_bytes, sizeof(int64_t));
}

template <arrow::Type::type ArrowType>
    requires(ArrowType == arrow::Type::FLOAT)
void insert_value_theta_sketch(const std::shared_ptr<arrow::Scalar> &scalar,
                               datasketches::update_theta_sketch &sketch) {
    auto float_scalar = std::static_pointer_cast<arrow::FloatScalar>(scalar);
    float value = float_scalar->value;
    const char *initial_bytes = reinterpret_cast<const char *>(&value);
    insert_numeric_buffer_theta_sketch(sketch, initial_bytes, sizeof(float));
}

template <arrow::Type::type ArrowType>
    requires(ArrowType == arrow::Type::DOUBLE)
void insert_value_theta_sketch(const std::shared_ptr<arrow::Scalar> &scalar,
                               datasketches::update_theta_sketch &sketch) {
    auto double_scalar = std::static_pointer_cast<arrow::DoubleScalar>(scalar);
    double value = double_scalar->value;
    const char *initial_bytes = reinterpret_cast<const char *>(&value);
    insert_numeric_buffer_theta_sketch(sketch, initial_bytes, sizeof(double));
}

template <arrow::Type::type ArrowType>
    requires(ArrowType == arrow::Type::DATE32)
void insert_value_theta_sketch(const std::shared_ptr<arrow::Scalar> &scalar,
                               datasketches::update_theta_sketch &sketch) {
    auto date32_scalar = std::static_pointer_cast<arrow::Date32Scalar>(scalar);
    int32_t value = date32_scalar->value;
    const char *initial_bytes = reinterpret_cast<const char *>(&value);
    insert_numeric_buffer_theta_sketch(sketch, initial_bytes, sizeof(int32_t));
}

template <arrow::Type::type ArrowType>
    requires(ArrowType == arrow::Type::TIME64)
void insert_value_theta_sketch(const std::shared_ptr<arrow::Scalar> &scalar,
                               datasketches::update_theta_sketch &sketch) {
    auto time64_scalar = std::static_pointer_cast<arrow::Time64Scalar>(scalar);
    int64_t value = time64_scalar->value;
    const char *initial_bytes = reinterpret_cast<const char *>(&value);
    insert_numeric_buffer_theta_sketch(sketch, initial_bytes, sizeof(int64_t));
}

template <arrow::Type::type ArrowType>
    requires(ArrowType == arrow::Type::TIMESTAMP)
void insert_value_theta_sketch(const std::shared_ptr<arrow::Scalar> &scalar,
                               datasketches::update_theta_sketch &sketch) {
    auto timestamp_scalar =
        std::static_pointer_cast<arrow::TimestampScalar>(scalar);
    int64_t value = timestamp_scalar->value;
    const char *initial_bytes = reinterpret_cast<const char *>(&value);
    insert_numeric_buffer_theta_sketch(sketch, initial_bytes, sizeof(int64_t));
}

template <arrow::Type::type ArrowType>
    requires(ArrowType == arrow::Type::LARGE_STRING)
void insert_value_theta_sketch(const std::shared_ptr<arrow::Scalar> &scalar,
                               datasketches::update_theta_sketch &sketch) {
    auto string_scalar =
        std::static_pointer_cast<arrow::LargeStringScalar>(scalar);
    sketch.update(string_scalar->value->data(), string_scalar->value->size());
}

template <arrow::Type::type ArrowType>
    requires(ArrowType == arrow::Type::LARGE_BINARY)
void insert_value_theta_sketch(const std::shared_ptr<arrow::Scalar> &scalar,
                               datasketches::update_theta_sketch &sketch) {
    auto binary_scalar =
        std::static_pointer_cast<arrow::LargeBinaryScalar>(scalar);
    sketch.update(binary_scalar->value->data(), binary_scalar->value->size());
}

template <arrow::Type::type ArrowType>
    requires(ArrowType == arrow::Type::DICTIONARY)
void insert_value_theta_sketch(const std::shared_ptr<arrow::Scalar> &scalar,
                               datasketches::update_theta_sketch &sketch) {
    auto dict_scalar =
        std::static_pointer_cast<arrow::DictionaryScalar>(scalar);
    std::shared_ptr<arrow::Scalar> string_scalar =
        dict_scalar->GetEncodedValue().ValueOrDie();
    insert_value_theta_sketch<arrow::Type::LARGE_STRING>(string_scalar, sketch);
}

void update_theta_sketches(theta_sketch_collection_t sketches,
                           const std::shared_ptr<arrow::Table> &in_table) {
    if (sketches == nullptr) {
        return;
    }
#define add_column_case(arrow_type)                                  \
    case arrow_type: {                                               \
        int n_chunks = col->num_chunks();                            \
        for (int cur_chunk = 0; cur_chunk < n_chunks; cur_chunk++) { \
            const std::shared_ptr<arrow::Array> &chunk =             \
                col->chunk(cur_chunk);                               \
            size_t n_rows = chunk->length();                         \
            for (size_t row = 0; row < n_rows; row++) {              \
                if (chunk->IsNull((int64_t)row))                     \
                    continue;                                        \
                std::shared_ptr<arrow::Scalar> current_row =         \
                    chunk->GetScalar((int64_t)row).ValueOrDie();     \
                insert_value_theta_sketch<arrow_type>(               \
                    current_row, sketches[col_idx].value());         \
            }                                                        \
        }                                                            \
        break;                                                       \
    }
    // Loop over every column in the table, skipping columns where the theta
    // sketch is absent.
    size_t n_columns = in_table->columns().size();
    for (size_t col_idx = 0; col_idx < n_columns; col_idx++) {
        if (sketches[col_idx].has_value()) {
            std::shared_ptr<arrow::ChunkedArray> col =
                in_table->column(col_idx);
            switch (col->type()->id()) {
                add_column_case(arrow::Type::INT32);
                add_column_case(arrow::Type::INT64);
                add_column_case(arrow::Type::FLOAT);
                add_column_case(arrow::Type::DOUBLE);
                add_column_case(arrow::Type::DATE32);
                add_column_case(arrow::Type::TIME64);
                add_column_case(arrow::Type::TIMESTAMP);
                add_column_case(arrow::Type::LARGE_STRING);
                add_column_case(arrow::Type::LARGE_BINARY);
                add_column_case(arrow::Type::DICTIONARY);
                default: {
                    throw std::runtime_error(
                        "update_theta_sketches: unsupported arrow type " +
                        col->type()->name());
                }
            }
        }
    }
#undef add_column_case
}

immutable_theta_sketch_collection_t compact_theta_sketches(
    theta_sketch_collection_t sketches, size_t n_sketches) {
    immutable_theta_sketch_collection_t result(n_sketches);
    // Loop over every column in the original table, skipping indices
    // without a theta sketch.
    for (size_t col_idx = 0; col_idx < n_sketches; col_idx++) {
        if (sketches[col_idx].has_value()) {
            result[col_idx] = sketches[col_idx].value().compact();
        } else {
            result[col_idx] = std::nullopt;
        }
    }
    return result;
}

immutable_theta_sketch_collection_t merge_parallel_theta_sketches(
    immutable_theta_sketch_collection_t sketches) {
    size_t n_sketches = sketches.size();
    int cp, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &cp);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    auto current_rank = (size_t)cp;
    auto num_ranks = (size_t)np;
    // Serialize the theta sketches and dump into a string array
    auto serialized = serialize_theta_sketches(sketches);
    bodo::vector<std::string> strings(n_sketches);
    bodo::vector<uint8_t> nulls((n_sketches + 7) >> 3);
    for (size_t col_idx = 0; col_idx < n_sketches; col_idx++) {
        if (serialized[col_idx].has_value()) {
            strings[col_idx] = serialized[col_idx].value();
            SetBitTo(nulls.data(), col_idx, true);
        } else {
            strings[col_idx] = "";
            SetBitTo(nulls.data(), col_idx, false);
        }
    }
    std::shared_ptr<array_info> as_string_array =
        create_string_array(Bodo_CTypes::STRING, nulls, strings, -1);

    // Gather the string arrays onto rank zero
    auto combined_string_array = gather_array(
        as_string_array, false, num_ranks > 1, 0, num_ranks, current_rank);

    if (current_rank == 0) {
        // On rank zero, convert the combined array into a vector of theta
        // sketch collecitons
        std::vector<immutable_theta_sketch_collection_t> collections;
        char *raw_data_ptr =
            combined_string_array->data1<bodo_array_type::STRING>();
        offset_t *offsets =
            combined_string_array->data2<bodo_array_type::STRING, offset_t>();
        size_t combined_idx = 0;
        for (size_t rank = 0; rank < (size_t)num_ranks; rank++) {
            std::vector<std::optional<std::string>> serialized_collection;
            for (size_t col_idx = 0; col_idx < n_sketches; col_idx++) {
                if (combined_string_array->get_null_bit(combined_idx)) {
                    offset_t start_offset = offsets[combined_idx];
                    offset_t end_offset = offsets[combined_idx + 1];
                    offset_t length = end_offset - start_offset;
                    std::string serialized_str(&raw_data_ptr[start_offset],
                                               length);
                    serialized_collection.push_back(serialized_str);
                } else {
                    serialized_collection.push_back(std::nullopt);
                }
                combined_idx++;
            }
            collections.push_back(
                deserialize_theta_sketches(serialized_collection));
        }
        // Finally, merge the collections
        return merge_theta_sketches(collections);
    } else {
        // On all other ranks, the result doesn't matter
        // so just return an empty vector
        immutable_theta_sketch_collection_t dummy(0);
        return dummy;
    }
}

immutable_theta_sketch_collection_t merge_theta_sketches(
    std::vector<immutable_theta_sketch_collection_t> sketch_collections) {
    size_t n_sketches = sketch_collections[0].size();
    immutable_theta_sketch_collection_t result(n_sketches);
    // Loop over every column in the original table, skipping indices
    // without a theta sketch.
    for (size_t col_idx = 0; col_idx < n_sketches; col_idx++) {
        if (sketch_collections[0][col_idx].has_value()) {
            // Create a union builder and iterate across all of the collections,
            // adding the current column's theta sketch from all the collections
            // into the builder
            auto combined = datasketches::theta_union::builder().build();
            for (immutable_theta_sketch_collection_t collection :
                 sketch_collections) {
                combined.update(collection[col_idx].value());
            }
            // Replace the first collection's theta sketch in the current
            // column with the combined result.
            result[col_idx] = combined.get_result();
        } else {
            result[col_idx] = std::nullopt;
        }
    }
    return result;
}

std::vector<std::optional<std::string>> serialize_theta_sketches(
    immutable_theta_sketch_collection_t sketches) {
    size_t n_sketches = sketches.size();
    std::vector<std::optional<std::string>> result;
    for (size_t col_idx = 0; col_idx < n_sketches; col_idx++) {
        if (sketches[col_idx].has_value()) {
            std::stringstream serialized;
            sketches[col_idx].value().serialize(serialized);
            result.push_back(serialized.str());
        } else {
            result.push_back(std::nullopt);
        }
    }
    return result;
}

immutable_theta_sketch_collection_t deserialize_theta_sketches(
    std::vector<std::optional<std::string>> strings) {
    size_t n_sketches = strings.size();
    immutable_theta_sketch_collection_t result(n_sketches);
    // Loop over every string in the collection, skipping ones where
    // the option is a nullopt
    for (size_t col_idx = 0; col_idx < n_sketches; col_idx++) {
        if (strings[col_idx].has_value()) {
            result[col_idx] = datasketches::compact_theta_sketch::deserialize(
                strings[col_idx].value().data(),
                strings[col_idx].value().length());
        } else {
            result[col_idx] = std::nullopt;
        }
    }
    return result;
}

// if status of arrow::Result is not ok, form an err msg and raise a
// runtime_error with it
#undef CHECK_ARROW
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

/** Python entrypoint to create a new theta sketch collection.
 * @param iceberg_arrow_schema_py: pointer to the python object representing the
 * iceberg arrow schema of the table.
 * @param already_exists: boolean indicating whether the table already exists or
 * is being created for the first time.
 * @param enable_theta_sketches: boolean indicating whether theta sketches are
 * turned on or not.
 */
theta_sketch_collection_t init_theta_sketches_py_entrypt(
    PyObject *iceberg_arrow_schema_py, bool already_exists,
    bool enable_theta_sketches) {
    std::shared_ptr<arrow::Schema> iceberg_schema;
    CHECK_ARROW_AND_ASSIGN(arrow::py::unwrap_schema(iceberg_arrow_schema_py),
                           "Iceberg Schema Couldn't Unwrap from Python",
                           iceberg_schema);
    size_t n_fields = iceberg_schema->num_fields();
    // Turn on theta sketches for a column if the following conditions are true:
    // 1. The column's type is one of the supported ones for theta sketches.
    // 2. The boolean was passed in to indicate that theta sketches are enabled.
    // 3. The boolean was passed in to indicate that this is NOT an insert into
    // operation.
    std::vector<bool> ndv_cols(n_fields);
    for (size_t col_idx = 0; col_idx < n_fields; col_idx++) {
        bool valid_theta_type;
        switch (iceberg_schema->field(col_idx)->type()->id()) {
            case arrow::Type::INT32:
            case arrow::Type::INT64:
            case arrow::Type::DATE32:
            case arrow::Type::TIME64:
            case arrow::Type::TIMESTAMP:
            case arrow::Type::LARGE_STRING:
            case arrow::Type::LARGE_BINARY:
            case arrow::Type::DICTIONARY:
                valid_theta_type = true;
                break;
            case arrow::Type::FLOAT:
            case arrow::Type::DOUBLE:
            default:
                valid_theta_type = false;
                break;
        }
        ndv_cols[col_idx] =
            valid_theta_type && enable_theta_sketches && !already_exists;
    }
    return init_theta_sketches(ndv_cols);
}

/** Python entrypoint to fetch the NDV approximations from a theta sketch
 * collection, largely for testing purposes.
 * @param sketches: the theta sketches being used to extract the approximation.
 * @param iceberg_arrow_schema_py: pointer to the python object representing the
 * iceberg arrow schema of the table.
 * @param parallel: boolean indicating whether the test is being done in
 * parallel.
 */
array_info *fetch_ndv_approximations_py_entrypt(
    theta_sketch_collection_t sketches, PyObject *iceberg_arrow_schema_py) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::shared_ptr<arrow::Schema> iceberg_schema;
    CHECK_ARROW_AND_ASSIGN(arrow::py::unwrap_schema(iceberg_arrow_schema_py),
                           "Iceberg Schema Couldn't Unwrap from Python",
                           iceberg_schema);
    size_t n_fields = iceberg_schema->num_fields();
    // Gather the theta sketches onto rank 0
    auto immutable_collection = compact_theta_sketches(sketches, n_fields);
    auto merged_collection =
        merge_parallel_theta_sketches(immutable_collection);
    std::vector<double> local_estimates(n_fields);
    if (rank == 0) {
        // Populate the vector on rank 0
        for (size_t col_idx = 0; col_idx < n_fields; col_idx++) {
            if (merged_collection[col_idx].has_value()) {
                double estimate =
                    merged_collection[col_idx].value().get_estimate();
                local_estimates[col_idx] = estimate;
            } else {
                local_estimates[col_idx] = -1.0;
            }
        }
    }
    // Broadcast the vector onto all ranks);
    MPI_Bcast(local_estimates.data(), n_fields, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Convert to an array where -1 is a sentinel for null
    std::unique_ptr<array_info> arr =
        alloc_nullable_array_no_nulls(n_fields, Bodo_CTypes::FLOAT64, 0);
    for (size_t col_idx = 0; col_idx < n_fields; col_idx++) {
        if (local_estimates[col_idx] < 0) {
            arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(col_idx, 0);
        } else {
            arr->data1<bodo_array_type::NULLABLE_INT_BOOL, double>()[col_idx] =
                local_estimates[col_idx];
        }
    }
    return arr.release();
}

PyMODINIT_FUNC PyInit_theta_sketches(void) {
    PyObject *m;
    MOD_DEF(m, "theta_sketches", "No docs", NULL);
    if (m == NULL) {
        return NULL;
    }

    bodo_common_init();

    SetAttrStringFromVoidPtr(m, init_theta_sketches_py_entrypt);
    SetAttrStringFromVoidPtr(m, fetch_ndv_approximations_py_entrypt);

    return m;
}
#undef CHECK_ARROW
#undef CHECK_ARROW_AND_ASSIGN
