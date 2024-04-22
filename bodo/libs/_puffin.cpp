#include "_puffin.h"
#include <arrow/python/api.h>

static char puffin_magic[4] = {0x50, 0x46, 0x41, 0x31};

// Verify that the input string contains the expected magic 4-byte sequence for
// puffin files at the index specified.
inline void verify_magic(std::string src, size_t idx) {
    if (src.substr(idx, 4) != std::string(puffin_magic, 4)) {
        throw std::runtime_error(
            "Malformed magic bytes for Puffin file (index " +
            std::to_string(idx) + "): " + src.substr(idx, 4));
    }
}

#define invalid_blob_metadata(errmsg)                             \
    throw std::runtime_error("Malformed blob metadata object: " + \
                             std::string(errmsg));

// Fetches an integer field from a json object representing a BlobMetadata.
long fetch_numeric_field(boost::json::object obj, std::string field_name) {
    boost::json::value *as_val = obj.if_contains(field_name);
    if (as_val == nullptr) {
        invalid_blob_metadata("missing required field '" + field_name + "'");
    }
    int64_t *as_int = as_val->if_int64();
    if (as_int == nullptr) {
        invalid_blob_metadata("field '" + field_name + "' must be an integer");
    }
    return (long)(*as_int);
}

BlobMetadata BlobMetadata::from_json(boost::json::object obj) {
    // Extract the required 'type' field
    boost::json::value *type_val = obj.if_contains("type");
    if (type_val == nullptr) {
        invalid_blob_metadata("missing required field 'type'");
    }
    boost::json::string *type_string = type_val->if_string();
    if (type_string == nullptr) {
        invalid_blob_metadata("field 'type' must be a string");
    }
    std::string type(type_string->data(), type_string->size());

    // Extract the required 'fields' field
    std::vector<int64_t> fields;
    boost::json::value *fields_val = obj.if_contains("fields");
    if (type_val == nullptr) {
        invalid_blob_metadata("missing required field 'fields'");
    }
    boost::json::array *fields_array = fields_val->if_array();
    if (fields_array == nullptr) {
        invalid_blob_metadata("field 'fields' must be an array");
    }
    for (auto field_val : *fields_array) {
        int64_t *fields_int = field_val.if_int64();
        if (fields_int == nullptr) {
            invalid_blob_metadata("field 'fields' must only contain integers");
        }
        fields.push_back(*fields_int);
    }

    // Extract the required fields snapshot-id, sequence-number, offset, length
    long snapshot_id = fetch_numeric_field(obj, "snapshot-id");
    long sequence_number = fetch_numeric_field(obj, "sequence-number");
    long offset = fetch_numeric_field(obj, "offset");
    long length = fetch_numeric_field(obj, "length");

    // Extract the optional 'compression_codec' field
    std::optional<std::string> compression_codec = std::nullopt;
    boost::json::value *codec_val = obj.if_contains("compression-codec");
    if (codec_val != nullptr) {
        boost::json::string *codec_string = codec_val->if_string();
        if (codec_string == nullptr) {
            invalid_blob_metadata("field 'compression-codec' must be a string");
        }
        std::string codec(codec_string->data(), codec_string->size());
        if (codec != "zstd") {
            invalid_blob_metadata("unsupported compression codec: " + codec);
        }
        compression_codec = codec;
    }

    // Extract the optional 'properties' field
    std::optional<std::unordered_map<std::string, std::string>> properties =
        std::nullopt;
    boost::json::value *properties_val = obj.if_contains("properties");
    if (properties_val != nullptr) {
        std::unordered_map<std::string, std::string> new_properties;
        boost::json::object *properties_object = properties_val->if_object();
        if (properties_object == nullptr) {
            invalid_blob_metadata("field 'properties' must be an object");
        }
        for (auto it : *properties_object) {
            boost::json::string *property_val = it.value().if_string();
            if (property_val == nullptr) {
                invalid_blob_metadata(
                    "field 'properties' must be an object with string values");
            }
            new_properties[it.key()] =
                std::string(property_val->data(), property_val->size());
        }
        properties = new_properties;
    }

    // Return the new object
    return BlobMetadata(type, fields, snapshot_id, sequence_number, offset,
                        length, compression_codec, properties);
}

#undef invalid_blob_metadata

boost::json::object BlobMetadata::to_json() {
    boost::json::object obj;
    obj["type"] = get_type();
    boost::json::array fields_array;
    for (auto field : get_fields()) {
        fields_array.push_back(boost::json::value(field));
    }
    obj["fields"] = fields_array;
    obj["snapshot-id"] = get_snapshot_id();
    obj["sequence-number"] = get_sequence_number();
    obj["offset"] = get_offset();
    obj["length"] = get_length();
    if (has_compression_codec()) {
        obj["compression-codec"] = get_compression_codec();
    }
    if (has_properties()) {
        boost::json::object blob_properties;
        for (auto it : get_properties()) {
            blob_properties[it.first] = it.second;
        }
        obj["properties"] = blob_properties;
    }
    return obj;
}

std::string PuffinFile::serialize() {
    std::stringstream ss;
    // Start with the magic 4 bytes
    ss.write(puffin_magic, 4);

    // Add each of the blobs
    for (auto blob : blobs) {
        ss.write(blob.data(), blob.size());
    }

    // Add the magic 4 bytes again
    ss.write(puffin_magic, 4);

    // Convert the footer back to a single JSON object,
    // serialize it as a string, and add it to the stream.
    boost::json::object footer_object;
    boost::json::array footer_blobs;
    for (size_t blob_idx = 0; blob_idx < num_blobs(); blob_idx++) {
        footer_blobs.push_back(get_blob_metadata(blob_idx).to_json());
    }
    footer_object["blobs"] = footer_blobs;
    if (has_properties()) {
        boost::json::object footer_properties;
        for (auto it : get_properties()) {
            footer_properties[it.first] = it.second;
        }
        footer_object["properties"] = footer_properties;
    }
    std::stringstream fs;
    fs << footer_object;
    std::string footer_payload = fs.str();
    int32_t payload_size = (int32_t)(footer_payload.size());
    ss.write(footer_payload.data(), payload_size);

    // Add the size of the footer payload.
    ss.write((const char *)(&payload_size), 4);

    // Add the 4 flag bytes (all zero)
    char flags[4] = {0x0, 0x0, 0x0, 0x0};
    ss.write(flags, 4);

    // End with the magic 4 bytes
    ss.write(puffin_magic, 4);

    // Finalize and return the result
    return ss.str();
}

#define invalid_footer(footer, errmsg)                                    \
    throw std::runtime_error("Malformed puffin file footer: '" + footer + \
                             "' " + errmsg);

std::unique_ptr<PuffinFile> PuffinFile::deserialize(std::string src) {
    // Verify that the file is long enough to contain the header and footer
    size_t file_size = src.length();
    if (file_size < 20) {
        throw std::runtime_error(
            "Malformed puffin file: must be at least 20 bytes");
    }

    // Verify that the file starts and ends with the magic.
    verify_magic(src, 0);
    verify_magic(src, file_size - 4);

    // Extract the 4 bytes of flag data (currently all unsupported if set)
    int32_t flags;
    memcpy(&flags, src.data() + (file_size - 8), sizeof(int32_t));
    if (flags != 0) {
        throw std::runtime_error(
            "Bodo does not currently support reading Puffin files with any "
            "flag bits set");
    }

    // Extract the size of the payload footer and verify that the 4 bytes
    // preceding it are the magic byte sequence.
    int32_t footer_payload_size;
    std::memcpy(&footer_payload_size, src.substr(file_size - 12).data(),
                sizeof(int32_t));
    if (file_size < (size_t)footer_payload_size + 20) {
        throw std::runtime_error(
            "Malformed puffin file with footer payload size of " +
            std::to_string(footer_payload_size) + ": file must be at least " +
            std::to_string(footer_payload_size + 20) + " bytes long");
    }
    verify_magic(src, file_size - footer_payload_size - 16);

    // Extract the footer payload and parse it as a JSON string
    std::string footer_payload =
        src.substr(file_size - footer_payload_size - 12, footer_payload_size);
    auto parsed = boost::json::parse(footer_payload);
    boost::json::object *obj = parsed.if_object();
    if (obj == nullptr) {
        invalid_footer(footer_payload, "expected an object");
    }

    // Extract the properties field of the json object, if present.
    std::optional<std::unordered_map<std::string, std::string>>
        file_properties = std::nullopt;
    boost::json::value *properties_val = obj->if_contains("properties");
    if (properties_val != nullptr) {
        std::unordered_map<std::string, std::string> new_properties;
        boost::json::object *properties_obj = properties_val->if_object();
        if (properties_obj == nullptr) {
            invalid_footer(footer_payload,
                           "optional field 'properties' must be an object");
        }
        for (auto it : *properties_obj) {
            boost::json::string *property_val = it.value().if_string();
            if (property_val == nullptr) {
                invalid_footer(footer_payload,
                               "optional field 'properties' must only contain "
                               "string values");
            }
            new_properties[it.key()] =
                std::string(property_val->data(), property_val->size());
        }
        file_properties = new_properties;
    }

    // Extract the blobs field of the json object and turn each entry
    // in the array into a BlobMetadata object
    std::vector<BlobMetadata> metadata;
    if (obj->contains("blobs")) {
        boost::json::value *blobs_val = obj->if_contains("blobs");
        boost::json::array *blobs_arr = blobs_val->if_array();
        if (blobs_arr == nullptr) {
            invalid_footer(footer_payload, "field 'blobs' must be an array");
        }
        for (auto arr_val : *blobs_arr) {
            boost::json::object *blob_obj = arr_val.if_object();
            if (blob_obj == nullptr) {
                invalid_footer(footer_payload,
                               "all elements in 'blobs' array must be objects");
            }
            metadata.push_back(BlobMetadata::from_json(*blob_obj));
        }
    } else {
        invalid_footer(footer_payload, "missing required field 'blobs'");
    }

    // Using the metadatas, extract the string components from the original raw
    // string corresponding to each of the blobs.
    std::vector<std::string> file_blobs;
    for (size_t blob_idx = 0; blob_idx < metadata.size(); blob_idx++) {
        file_blobs.push_back(src.substr(metadata[blob_idx].get_offset(),
                                        metadata[blob_idx].get_length()));
    }

    return std::make_unique<PuffinFile>(file_blobs, metadata, file_properties);
}

#undef invalid_footer

std::unique_ptr<PuffinFile> PuffinFile::from_theta_sketches(
    immutable_theta_sketch_collection_t sketches, int64_t snapshot_id,
    int64_t sequence_number) {
    size_t n_sketches = sketches.size();
    // For each theta sketch, add the serialized sketch to blobs, then
    // populate the metadata.
    std::vector<std::string> blobs;
    std::vector<BlobMetadata> metadata;
    std::vector<std::optional<std::string>> serialized_sketches =
        serialize_theta_sketches(sketches);
    size_t curr_offset = 4;
    for (size_t sketch_idx = 0; sketch_idx < n_sketches; sketch_idx++) {
        if (sketches[sketch_idx] != std::nullopt) {
            std::string blob = serialized_sketches[sketch_idx].value();
            blobs.push_back(blob);
            std::unordered_map<std::string, std::string> properties;
            properties["ndv"] = std::to_string(
                (size_t)(sketches[sketch_idx].value().get_estimate()));
            metadata.push_back({
                "apache-datasketches-theta-v1",  // type
                {(int64_t)sketch_idx + 1},       // fields (1-indexed)
                snapshot_id,                     // snapshot-id
                sequence_number,                 // sequence-number
                (int64_t)curr_offset,            // offset
                (int64_t)blob.size(),            // length
                std::nullopt,                    // codec (absent)
                properties                       // properties (ndv estimate)
            });
            curr_offset += blob.size();
        }
    }

    // Populate the properties with a single mapping:
    // "created-by" -> the Bodo version
    std::unordered_map<std::string, std::string> properties_inner;
    properties_inner["created-by"] = "Bodo version " + get_bodo_version();
    std::optional<std::unordered_map<std::string, std::string>> properties =
        properties_inner;

    // Return the new PuffinFile object
    return std::make_unique<PuffinFile>(blobs, metadata, properties);
}

std::string PuffinFile::get_blob(size_t idx) {
    std::string blob = blobs[idx];
    if (!get_blob_metadata(idx).has_compression_codec())
        return blob;
    std::string compression_codec =
        get_blob_metadata(idx).get_compression_codec();
    if (compression_codec == "zstd") {
        return decode_zstd(blob);
    } else {
        throw std::runtime_error(
            "Unsupported compression codec for PuffinFile blob: " +
            compression_codec);
    }
}

immutable_theta_sketch_collection_t PuffinFile::to_theta_sketches(
    size_t n_columns) {
    std::vector<std::optional<std::string>> to_deserialize(n_columns,
                                                           std::nullopt);
    // Iterate across all the blobs and add the theta sketches to the
    // collection.
    for (size_t blob_idx = 0; blob_idx < num_blobs(); blob_idx++) {
        if (get_blob_metadata(blob_idx).get_type() ==
            "apache-datasketches-theta-v1") {
            // Fetch the column index and decrement by 1 so it is zero-indexed.
            if (get_blob_metadata(blob_idx).get_fields().size() != 1) {
                throw std::runtime_error(
                    "Expected theta sketch blob metadata to refer to exactly 1 "
                    "column");
            }
            int64_t field = get_blob_metadata(blob_idx).get_fields()[0] - 1;
            if (field < 0 || field >= (int64_t)n_columns) {
                throw std::runtime_error(
                    "Expected theta sketch blob metadata to refer to a valid "
                    "column index");
            }
            // Fetch the corresponding string and add it to the collection.
            std::string blob = get_blob(blob_idx);
            to_deserialize[field] = blob;
        }
    }
    return deserialize_theta_sketches(to_deserialize);
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

/**
 * Entrypoint from Python to initialize a PuffinFile write.
 * @param table_loc_buffer: buffer storing the table_loc string.
 * @param table_loc_size: number of characters in table_loc_buffer.
 * @param snapshot_id: the snapshot_id to use for the PuffinFile metadata.
 * @param snapshot_id: the sequence_number to use for the PuffinFile metadata.
 * @param sketches: the collection of theta sketches to write.
 * @param iceberg_arrow_schema_py: the schema of the table being written.
 * @param already_exists: whether the table being written to already exists.
 */
void write_puffin_file_py_entrypt(char *table_loc_buffer,
                                  int64_t table_loc_size, int64_t snapshot_id,
                                  int64_t sequence_number,
                                  theta_sketch_collection_t sketches,
                                  PyObject *iceberg_arrow_schema_py,
                                  bool already_exists) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::shared_ptr<arrow::Schema> iceberg_schema;
    CHECK_ARROW_AND_ASSIGN(arrow::py::unwrap_schema(iceberg_arrow_schema_py),
                           "Iceberg Schema Couldn't Unwrap from Python",
                           iceberg_schema);
    size_t n_cols = iceberg_schema->num_fields();
    // Gather the theta sketches onto rank 0
    auto immutable_collection = compact_theta_sketches(sketches, n_cols);
    auto merged_collection =
        merge_parallel_theta_sketches(immutable_collection);
    // TODO: if already exists, combine merged_collections with the existing
    // theta sketches
    if (rank == 0) {
        std::string table_loc(table_loc_buffer, table_loc_size);
        auto puff = PuffinFile::from_theta_sketches(
            merged_collection, snapshot_id, sequence_number);
        std::string serialized = puff->serialize();
        // TODO: write serialized
    }
}

PyMODINIT_FUNC PyInit_puffin_file(void) {
    PyObject *m;
    MOD_DEF(m, "puffin_file", "No docs", NULL);
    if (m == NULL) {
        return NULL;
    }

    bodo_common_init();

    SetAttrStringFromVoidPtr(m, write_puffin_file_py_entrypt);

    return m;
}
#undef CHECK_ARROW
#undef CHECK_ARROW_AND_ASSIGN
