#include "_theta_sketches.h"
#include <arrow/python/api.h>
#include <algorithm>
#include <theta_union.hpp>

#include "_array_utils.h"
#include "_distributed.h"
#include "_mpi.h"

// if status of arrow::Result is not ok, form an err msg and raise a
// runtime_error with it
#undef CHECK_ARROW
#define CHECK_ARROW(expr, msg)                                                 \
    if (!(expr.ok())) {                                                        \
        std::string err_msg = std::string("Error in theta sketches: ") + msg + \
                              " " + expr.ToString();                           \
        throw std::runtime_error(err_msg);                                     \
    }

// if status of arrow::Result is not ok, form an err msg and raise a
// runtime_error with it. If it is ok, get value using ValueOrDie
// and assign it to lhs using std::move
#undef CHECK_ARROW_AND_ASSIGN
#define CHECK_ARROW_AND_ASSIGN(res, msg, lhs) \
    CHECK_ARROW(res.status(), msg)            \
    lhs = std::move(res).ValueOrDie();

std::shared_ptr<CompactSketchCollection>
CompactSketchCollection::merge_parallel_sketches() {
    size_t n_sketches = this->sketches.size();
    int rank, n_pes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
    // Serialize the theta sketches and dump into a string array
    std::vector<std::optional<std::string>> serialized =
        this->serialize_sketches();
    bodo::vector<std::string> strings(n_sketches);
    bodo::vector<uint8_t> nulls(arrow::bit_util::BytesForBits(n_sketches));
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
    auto combined_string_array =
        gather_array(as_string_array, false, n_pes > 1, 0, n_pes, rank);

    if (rank == 0) {
        // On rank zero, convert the combined array into a vector of theta
        // sketch collections
        char *raw_data_ptr =
            combined_string_array->data1<bodo_array_type::STRING>();
        offset_t *offsets =
            combined_string_array->data2<bodo_array_type::STRING, offset_t>();
        size_t combined_idx = 0;
        std::vector<std::shared_ptr<CompactSketchCollection>> collections;
        for (int i = 0; i < n_pes; i++) {
            std::vector<std::optional<std::string>> serialized_collection;
            for (size_t col_idx = 0; col_idx < n_sketches; col_idx++) {
                if (combined_string_array->get_null_bit(combined_idx)) {
                    offset_t start_offset = offsets[combined_idx];
                    offset_t end_offset = offsets[combined_idx + 1];
                    offset_t length = end_offset - start_offset;
                    std::string serialized_str(&raw_data_ptr[start_offset],
                                               length);
                    serialized_collection.emplace_back(serialized_str);
                } else {
                    serialized_collection.emplace_back(std::nullopt);
                }
                combined_idx++;
            }
            collections.push_back(CompactSketchCollection::deserialize_sketches(
                serialized_collection));
        }
        // Finally, merge the collections
        return CompactSketchCollection::merge_sketches(std::move(collections));
    } else {
        // On all other ranks, the result doesn't matter
        // so just return an empty vector
        std::vector<std::optional<datasketches::compact_theta_sketch>> empty;
        return std::make_unique<CompactSketchCollection>(std::move(empty));
    }
}

std::shared_ptr<CompactSketchCollection>
CompactSketchCollection::merge_sketches(
    std::vector<std::shared_ptr<CompactSketchCollection>> sketch_collections) {
    size_t n_sketches = sketch_collections[0]->max_num_sketches();
    std::vector<std::optional<datasketches::compact_theta_sketch>> result(
        n_sketches);
    // Loop over every column in the original table, skipping indices
    // without a theta sketch.
    for (size_t col_idx = 0; col_idx < n_sketches; col_idx++) {
        if (sketch_collections[0]->sketches[col_idx].has_value()) {
            // Create a union builder and iterate across all of the collections,
            // adding the current column's theta sketch from all the collections
            // into the builder
            auto combined = datasketches::theta_union::builder().build();
            for (std::shared_ptr<CompactSketchCollection> &collection :
                 sketch_collections) {
                combined.update(collection->sketches[col_idx].value());
            }
            // Replace the first collection's theta sketch in the current
            // column with the combined result.
            result[col_idx] = combined.get_result();
        } else {
            result[col_idx] = std::nullopt;
        }
    }
    return std::make_unique<CompactSketchCollection>(std::move(result));
}

std::vector<std::optional<std::string>>
CompactSketchCollection::serialize_sketches() {
    size_t n_sketches = this->sketches.size();
    std::vector<std::optional<std::string>> result;
    for (size_t col_idx = 0; col_idx < n_sketches; col_idx++) {
        if (this->sketches[col_idx].has_value()) {
            std::stringstream serialized;
            this->sketches[col_idx].value().serialize(serialized);
            result.emplace_back(serialized.str());
        } else {
            result.emplace_back(std::nullopt);
        }
    }
    return result;
}

std::shared_ptr<CompactSketchCollection>
CompactSketchCollection::deserialize_sketches(
    std::vector<std::optional<std::string>> strings) {
    size_t n_sketches = strings.size();
    std::vector<std::optional<datasketches::compact_theta_sketch>> result(
        n_sketches);
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
    return std::make_unique<CompactSketchCollection>(std::move(result));
}

std::unique_ptr<array_info> CompactSketchCollection::compute_ndv() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    size_t n_fields = this->sketches.size();
    std::vector<double> local_estimates(n_fields);
    if (rank == 0) {
        // Populate the vector on rank 0
        for (size_t col_idx = 0; col_idx < n_fields; col_idx++) {
            if (this->sketches[col_idx].has_value()) {
                double estimate =
                    this->sketches[col_idx].value().get_estimate();
                local_estimates[col_idx] = estimate;
            } else {
                local_estimates[col_idx] = -1.0;
            }
        }
    }
    // Broadcast the vector onto all ranks);
    CHECK_MPI(MPI_Bcast(local_estimates.data(), n_fields, MPI_DOUBLE, 0,
                        MPI_COMM_WORLD),
              "CompactSketchCollection::compute_ndv: MPI error on MPI_Bcast:");

    // Convert to an array where -1 is a sentinel for null
    std::unique_ptr<array_info> arr =
        alloc_nullable_array_no_nulls(n_fields, Bodo_CTypes::FLOAT64);
    for (size_t col_idx = 0; col_idx < n_fields; col_idx++) {
        if (local_estimates[col_idx] < 0) {
            arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(col_idx,
                                                                  false);
        } else {
            arr->data1<bodo_array_type::NULLABLE_INT_BOOL, double>()[col_idx] =
                local_estimates[col_idx];
        }
    }
    return arr;
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
        std::ranges::reverse(str);
        const char *bytes = str.c_str();
        sketch.update(bytes, n_bytes);
    }
}

template <arrow::Type::type ArrowType>
    requires(ArrowType == arrow::Type::INT32 ||
             ArrowType == arrow::Type::INT64 ||
             ArrowType == arrow::Type::FLOAT ||
             ArrowType == arrow::Type::DOUBLE ||
             ArrowType == arrow::Type::DATE32 ||
             ArrowType == arrow::Type::TIME64 ||
             ArrowType == arrow::Type::TIMESTAMP)
void insert_value_theta_sketch(const char *data, int64_t num_bytes,
                               datasketches::update_theta_sketch &sketch) {
    const char *initial_bytes = reinterpret_cast<const char *>(data);
    insert_numeric_buffer_theta_sketch(sketch, initial_bytes, num_bytes);
}

template <arrow::Type::type ArrowType>
    requires(ArrowType == arrow::Type::LARGE_STRING ||
             ArrowType == arrow::Type::LARGE_BINARY)
void insert_value_theta_sketch(const char *data, int64_t num_bytes,
                               datasketches::update_theta_sketch &sketch) {
    sketch.update(data, num_bytes);
}

void insert_theta_sketch_dict_column(
    const std::shared_ptr<arrow::ChunkedArray> &col,
    datasketches::update_theta_sketch &sketch,
    const std::shared_ptr<arrow::Buffer> &dict_hits) {
    for (int64_t chunk_idx = 0; chunk_idx < col->num_chunks(); chunk_idx++) {
        const auto &chunk = col->chunks()[chunk_idx];
        const auto &dict_array_chunk =
            std::dynamic_pointer_cast<arrow::DictionaryArray>(chunk);
        const auto &str_arr =
            std::dynamic_pointer_cast<arrow::LargeStringArray>(
                dict_array_chunk->dictionary());
        int64_t dict_size = str_arr->length();
        for (int64_t row = 0; row < dict_size; row++) {
            if (arrow::bit_util::GetBit(dict_hits->mutable_data(), row)) {
                arrow::LargeStringType::offset_type num_bytes;
                const char *buffer = reinterpret_cast<const char *>(
                    str_arr->GetValue(row, &num_bytes));
                sketch.update(buffer, num_bytes);
            }
        }
    }
}

template <size_t n_bytes>
void insert_theta_sketch_decimal_column_helper(
    const std::shared_ptr<arrow::ChunkedArray> &col,
    datasketches::update_theta_sketch &sketch) {
    const size_t skip_bytes = 16 - n_bytes;
    for (int64_t chunk_idx = 0; chunk_idx < col->num_chunks(); chunk_idx++) {
        const auto &chunk = col->chunks()[chunk_idx];
        const auto &decimal_array_chunk =
            std::dynamic_pointer_cast<arrow::Decimal128Array>(chunk);
        int64_t n_rows = decimal_array_chunk->length();
        const char *raw_values =
            reinterpret_cast<const char *>(decimal_array_chunk->raw_values());
        for (int64_t row = 0; row < n_rows; row++) {
            if (!decimal_array_chunk->IsNull(row)) {
                if constexpr (std::endian::native == std::endian::big) {
                    // If already big endian, insert those bytes (skipping
                    // any of the 16 bytes that are before the relevent section)
                    const char *initial_bytes =
                        (raw_values + (16 * row) + skip_bytes);
                    sketch.update(initial_bytes, n_bytes);
                } else {
                    // Convert to big endian (no skipping required because the
                    // relevant n_bytes are at the begining)
                    const char *initial_bytes = (raw_values + (16 * row));
                    std::string str = std::string(initial_bytes, n_bytes);
                    std::ranges::reverse(str);
                    const char *bytes = str.c_str();
                    sketch.update(bytes, n_bytes);
                }
            }
        }
    }
}

void insert_theta_sketch_decimal_column(
    const std::shared_ptr<arrow::ChunkedArray> &col,
    datasketches::update_theta_sketch &sketch, size_t prec, size_t scale) {
    int32_t n_bytes = decimal_precision_to_integer_bytes((int32_t)prec);
    switch (n_bytes) {
        case 1: {
            insert_theta_sketch_decimal_column_helper<1>(col, sketch);
            break;
        }
        case 2: {
            insert_theta_sketch_decimal_column_helper<2>(col, sketch);
            break;
        }
        case 4: {
            insert_theta_sketch_decimal_column_helper<4>(col, sketch);
            break;
        }
        case 8: {
            insert_theta_sketch_decimal_column_helper<8>(col, sketch);
            break;
        }
        case 16: {
            insert_theta_sketch_decimal_column_helper<16>(col, sketch);
            break;
        }
        default: {
            throw std::runtime_error(
                "update_sketches: decimal128 column requires "
                "a precision of at most 38");
        }
    }
}

void UpdateSketchCollection::update_sketch(
    const std::shared_ptr<arrow::ChunkedArray> &col, size_t col_idx,
    std::optional<std::shared_ptr<arrow::Buffer>> dict_hits) {
#define add_numeric_column_case(arrow_type)                               \
    case (arrow_type::type_id): {                                         \
        datasketches::update_theta_sketch &sketch =                       \
            this->sketches[col_idx].value();                              \
        int n_chunks = col->num_chunks();                                 \
        for (int cur_chunk = 0; cur_chunk < n_chunks; cur_chunk++) {      \
            const std::shared_ptr<arrow::Array> &chunk =                  \
                col->chunk(cur_chunk);                                    \
            size_t n_rows = chunk->length();                              \
            std::shared_ptr<arrow::NumericArray<arrow_type>> base_array = \
                std::reinterpret_pointer_cast<                            \
                    arrow::NumericArray<arrow_type>>(chunk);              \
            using T = typename arrow_type::c_type;                        \
            const T *raw_values = base_array->raw_values();               \
            for (size_t row = 0; row < n_rows; row++) {                   \
                if (chunk->IsNull(row)) {                                 \
                    continue;                                             \
                }                                                         \
                int64_t num_bytes = sizeof(T);                            \
                const char *data =                                        \
                    reinterpret_cast<const char *>(raw_values + row);     \
                insert_value_theta_sketch<arrow_type::type_id>(           \
                    data, num_bytes, sketch);                             \
            }                                                             \
        }                                                                 \
        break;                                                            \
    }
#define add_binary_column_case(arrow_type)                                     \
    case (arrow_type::type_id): {                                              \
        datasketches::update_theta_sketch &sketch =                            \
            this->sketches[col_idx].value();                                   \
        int n_chunks = col->num_chunks();                                      \
        for (int cur_chunk = 0; cur_chunk < n_chunks; cur_chunk++) {           \
            const std::shared_ptr<arrow::Array> &chunk =                       \
                col->chunk(cur_chunk);                                         \
            size_t n_rows = chunk->length();                                   \
            std::shared_ptr<arrow::LargeBinaryArray> base_array =              \
                std::reinterpret_pointer_cast<arrow::LargeBinaryArray>(chunk); \
            for (size_t row = 0; row < n_rows; row++) {                        \
                if (chunk->IsNull(row)) {                                      \
                    continue;                                                  \
                }                                                              \
                arrow_type::offset_type num_bytes;                             \
                const char *data = reinterpret_cast<const char *>(             \
                    base_array->GetValue(row, &num_bytes));                    \
                insert_value_theta_sketch<arrow_type::type_id>(                \
                    data, num_bytes, sketch);                                  \
            }                                                                  \
        }                                                                      \
        break;                                                                 \
    }
    // Skipping the column if the theta sketch is absent.
    if (this->sketches[col_idx].has_value()) {
        switch (col->type()->id()) {
            add_numeric_column_case(arrow::Int32Type);
            add_numeric_column_case(arrow::Int64Type);
            add_numeric_column_case(arrow::FloatType);
            add_numeric_column_case(arrow::DoubleType);
            add_numeric_column_case(arrow::Date32Type);
            add_numeric_column_case(arrow::Time64Type);
            add_numeric_column_case(arrow::TimestampType);
            add_binary_column_case(arrow::LargeStringType);
            add_binary_column_case(arrow::LargeBinaryType);
            case arrow::Type::DECIMAL128: {
                auto typ =
                    reinterpret_cast<arrow::DecimalType *>((col->type()).get());
                size_t prec = typ->precision();
                size_t scale = typ->scale();
                insert_theta_sketch_decimal_column(
                    col, this->sketches[col_idx].value(), prec, scale);
                break;
            }
            case arrow::Type::DICTIONARY: {
                if (dict_hits.has_value()) {
                    insert_theta_sketch_dict_column(
                        col, this->sketches[col_idx].value(),
                        dict_hits.value());
                } else {
                    throw std::runtime_error(
                        "update_sketches: dictionary encoded column requires "
                        "non-nullopt value for dict-hits");
                }
                break;
            }
            default: {
                throw std::runtime_error(
                    "update_sketches: unsupported arrow type " +
                    col->type()->name());
            }
        }
    }
#undef add_numeric_column_case
#undef add_binary_column_case
}

std::shared_ptr<CompactSketchCollection>
UpdateSketchCollection::compact_sketches() {
    // Loop over every column in the original table, skipping indices
    // without a theta sketch.
    std::vector<std::optional<datasketches::compact_theta_sketch>> result(
        this->sketches.size());
    for (size_t col_idx = 0; col_idx < this->sketches.size(); col_idx++) {
        if (sketches[col_idx].has_value()) {
            result[col_idx] = this->sketches[col_idx].value().compact();
        } else {
            result[col_idx] = std::nullopt;
        }
    }
    return std::make_unique<CompactSketchCollection>(std::move(result));
}

/**
 * @brief Python entrypoint to create a new theta sketch collection.
 * @param theta_columns_py: Array info indicating which columns have theta
 * sketches enabled.
 * @return UpdateSketchCollection* The new theta sketch collection.
 */
UpdateSketchCollection *init_theta_sketches_py_entrypt(
    array_info *theta_columns_py) {
    try {
        std::shared_ptr<array_info> theta_columns =
            std::shared_ptr<array_info>(theta_columns_py);
        return new UpdateSketchCollection(
            array_to_boolean_vector(theta_columns));
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

/**
 * @brief Python entrypoint to fetch the NDV approximations from a theta sketch
 * collection, largely for testing purposes.
 * @param sketches: the theta sketches being used to extract the approximation.
 * @param iceberg_arrow_schema_py: pointer to the python object representing the
 * iceberg arrow schema of the table.
 * @param parallel: boolean indicating whether the test is being done in
 * parallel.
 */
array_info *fetch_ndv_approximations_py_entrypt(
    UpdateSketchCollection *sketches) {
    try {
        std::shared_ptr<CompactSketchCollection> compact_sketches =
            sketches->compact_sketches();
        std::shared_ptr<CompactSketchCollection> merged_sketches =
            compact_sketches->merge_parallel_sketches();
        return merged_sketches->compute_ndv().release();
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

/**
 * @brief Python entrypoint to output a boolean array indicating which columns
 * have theta sketches enabled based on types we can support.
 *
 * @param iceberg_arrow_schema_py The iceberg arrow schema of the table.
 * @return array_info* The boolean array indicating which columns have theta
 * sketches.
 */
array_info *get_supported_theta_sketch_columns_py_entrypt(
    PyObject *iceberg_arrow_schema_py) {
    try {
        std::shared_ptr<arrow::Schema> iceberg_schema;
        CHECK_ARROW_AND_ASSIGN(
            arrow::py::unwrap_schema(iceberg_arrow_schema_py),
            "Iceberg Schema Couldn't Unwrap from Python", iceberg_schema);
        size_t n_fields = iceberg_schema->num_fields();
        std::shared_ptr<array_info> supported_columns =
            alloc_nullable_array_no_nulls(n_fields, Bodo_CTypes::_BOOL);
        uint8_t *out_arr = reinterpret_cast<uint8_t *>(
            supported_columns->data1<bodo_array_type::NULLABLE_INT_BOOL>());
        for (size_t col_idx = 0; col_idx < n_fields; col_idx++) {
            bool valid_theta_type = type_supports_theta_sketch(
                iceberg_schema->field(col_idx)->type());
            SetBitTo(out_arr, col_idx, valid_theta_type);
        }
        return new array_info(*supported_columns);

    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

/**
 * @brief Python entrypoint to output a boolean array indicating which columns
 * have theta sketches enabled based on types we can support.
 *
 * @param iceberg_arrow_schema_py The iceberg arrow schema of the table.
 * @return array_info* The boolean array indicating which columns have theta
 * sketches.
 */
array_info *get_default_theta_sketch_columns_py_entrypt(
    PyObject *iceberg_arrow_schema_py) {
    try {
        std::shared_ptr<arrow::Schema> iceberg_schema;
        CHECK_ARROW_AND_ASSIGN(
            arrow::py::unwrap_schema(iceberg_arrow_schema_py),
            "Iceberg Schema Couldn't Unwrap from Python", iceberg_schema);
        size_t n_fields = iceberg_schema->num_fields();
        std::shared_ptr<array_info> supported_columns =
            alloc_nullable_array_no_nulls(n_fields, Bodo_CTypes::_BOOL);
        uint8_t *out_arr = reinterpret_cast<uint8_t *>(
            supported_columns->data1<bodo_array_type::NULLABLE_INT_BOOL>());
        for (size_t col_idx = 0; col_idx < n_fields; col_idx++) {
            bool valid_theta_type = is_default_theta_sketch_type(
                iceberg_schema->field(col_idx)->type());
            SetBitTo(out_arr, col_idx, valid_theta_type);
        }
        return new array_info(*supported_columns);

    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

/**
 * @brief Python entrypoint to delete the theta sketch object.
 * This is called by Python once the object is no longer needed.
 *
 * @param sketches The sketches to delete.
 */
void delete_theta_sketches_py_entrypt(UpdateSketchCollection *sketches) {
    try {
        delete sketches;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

PyMODINIT_FUNC PyInit_theta_sketches(void) {
    PyObject *m;
    MOD_DEF(m, "theta_sketches", "No docs", nullptr);
    if (m == nullptr) {
        return nullptr;
    }

    bodo_common_init();

    SetAttrStringFromVoidPtr(m, init_theta_sketches_py_entrypt);
    SetAttrStringFromVoidPtr(m, fetch_ndv_approximations_py_entrypt);
    SetAttrStringFromVoidPtr(m, get_supported_theta_sketch_columns_py_entrypt);
    SetAttrStringFromVoidPtr(m, get_default_theta_sketch_columns_py_entrypt);
    SetAttrStringFromVoidPtr(m, delete_theta_sketches_py_entrypt);

    return m;
}
#undef CHECK_ARROW
#undef CHECK_ARROW_AND_ASSIGN
