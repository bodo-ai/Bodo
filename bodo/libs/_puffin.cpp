#include "_puffin.h"

#include <arrow/python/api.h>
#include <listobject.h>
#include <object.h>
#include <zstd.h>
#include <algorithm>

#include "../io/_fs_io.h"
#include "../io/arrow_compat.h"
#include "../io/iceberg_helpers.h"

// Throw an error if there is a null pointer returned
// by a CPython API call.
#undef CHECK
#define CHECK(expr, msg)               \
    if (!(expr)) {                     \
        throw std::runtime_error(msg); \
    }

// if status of arrow::Result is not ok, form an err msg and raise a
// runtime_error with it
#undef CHECK_ARROW
#define CHECK_ARROW(expr, msg)                                             \
    if (!(expr.ok())) {                                                    \
        std::string err_msg = std::string("Error in puffin I/O: ") + msg + \
                              " " + expr.ToString();                       \
        throw std::runtime_error(err_msg);                                 \
    }

// if status of arrow::Result is not ok, form an err msg and raise a
// runtime_error with it. If it is ok, get value using ValueOrDie
// and assign it to lhs using std::move
#undef CHECK_ARROW_AND_ASSIGN
#define CHECK_ARROW_AND_ASSIGN(res, msg, lhs) \
    CHECK_ARROW(res.status(), msg)            \
    lhs = std::move(res).ValueOrDie();

static char puffin_magic[4] = {0x50, 0x46, 0x41, 0x31};

// Verify that the input string contains the expected magic 4-byte sequence for
// puffin files at the index specified.
inline void verify_magic(const std::string &src, size_t idx) {
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
int64_t fetch_numeric_field(boost::json::object obj, std::string field_name) {
    boost::json::value *as_val = obj.if_contains(field_name);
    if (as_val == nullptr) {
        invalid_blob_metadata("missing required field '" + field_name + "'");
    }
    int64_t *as_int = as_val->if_int64();
    if (as_int == nullptr) {
        invalid_blob_metadata("field '" + field_name + "' must be an integer");
    }
    return *as_int;
}

std::string decode_zstd(std::string blob) {
    auto const est_decomp_size =
        ZSTD_getFrameContentSize(blob.data(), blob.size());
    std::string decomp_buffer{};
    decomp_buffer.resize(est_decomp_size);
    size_t const decomp_size =
        ZSTD_decompress((void *)decomp_buffer.data(), est_decomp_size,
                        blob.data(), blob.size());
    if (decomp_size == ZSTD_CONTENTSIZE_UNKNOWN ||
        decomp_size == ZSTD_CONTENTSIZE_ERROR) {
        throw std::runtime_error("Malformed ZSTD decompression");
    }
    return decomp_buffer;
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
    int64_t snapshot_id = fetch_numeric_field(obj, "snapshot-id");
    int64_t sequence_number = fetch_numeric_field(obj, "sequence-number");
    int64_t offset = fetch_numeric_field(obj, "offset");
    int64_t length = fetch_numeric_field(obj, "length");

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

std::pair<std::string, int32_t> PuffinFile::serialize() {
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
    const char *payload_size_bytes =
        reinterpret_cast<const char *>(&payload_size);
    if constexpr (std::endian::native == std::endian::little) {
        ss.write(payload_size_bytes, 4);
    } else {
        // Convert to little endian
        std::string str = std::string(payload_size_bytes, 4);
        std::ranges::reverse(str);
        const char *little_endian_bytes = str.c_str();
        ss.write(little_endian_bytes, sizeof(int32_t));
    }

    // Add the 4 flag bytes (all zero)
    char flags[4] = {0x0, 0x0, 0x0, 0x0};
    ss.write(flags, 4);

    // End with the magic 4 bytes
    ss.write(puffin_magic, 4);

    // Finalize and return the result
    return std::pair(ss.str(), payload_size);
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
    if (std::endian::native == std::endian::big) {
        // Reverse the string if our architecture is big endian
        const char *bytes =
            reinterpret_cast<const char *>(&footer_payload_size);
        std::string str = std::string(bytes, 4);
        std::ranges::reverse(str);
        footer_payload_size = *reinterpret_cast<const int32_t *>(str.c_str());
    }
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

    // Using the metadata, extract the string components from the original raw
    // string corresponding to each of the blobs.
    std::vector<std::string> file_blobs;
    for (auto &blob_idx : metadata) {
        file_blobs.push_back(
            src.substr(blob_idx.get_offset(), blob_idx.get_length()));
    }

    return std::make_unique<PuffinFile>(file_blobs, metadata, file_properties);
}

#undef invalid_footer

std::unique_ptr<PuffinFile> PuffinFile::from_theta_sketches(
    std::shared_ptr<CompactSketchCollection> sketches,
    std::shared_ptr<arrow::Schema> iceberg_schema, int64_t snapshot_id,
    int64_t sequence_number) {
    size_t n_sketches = sketches->max_num_sketches();
    // For each theta sketch, add the serialized sketch to blobs, then
    // populate the metadata.
    std::vector<std::string> blobs;
    std::vector<BlobMetadata> metadata;
    std::vector<std::optional<std::string>> serialized_sketches =
        sketches->serialize_sketches();
    size_t curr_offset = 4;
    for (size_t sketch_idx = 0; sketch_idx < n_sketches; sketch_idx++) {
        if (sketches->column_has_sketch(sketch_idx)) {
            std::string blob = serialized_sketches[sketch_idx].value();
            blobs.push_back(blob);
            std::unordered_map<std::string, std::string> properties;
            properties["ndv"] = std::to_string(
                (size_t)(sketches->get_value(sketch_idx).get_estimate()));
            int field_id =
                get_iceberg_field_id(iceberg_schema->field(sketch_idx));
            metadata.push_back({
                "apache-datasketches-theta-v1",  // type
                {field_id},                      // field ids
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
    if (!get_blob_metadata(idx).has_compression_codec()) {
        return blob;
    }
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

std::shared_ptr<CompactSketchCollection> PuffinFile::to_theta_sketches(
    std::shared_ptr<arrow::Schema> iceberg_schema) {
    size_t n_columns = iceberg_schema->num_fields();
    std::vector<std::optional<std::string>> to_deserialize(n_columns,
                                                           std::nullopt);
    // Create a map from to field id to field index.
    std::unordered_map<int, int> field_id_to_index;
    for (size_t i = 0; i < n_columns; i++) {
        field_id_to_index[get_iceberg_field_id(iceberg_schema->field(i))] = i;
    }
    // Iterate across all the blobs and add the theta sketches to the
    // collection.
    for (size_t blob_idx = 0; blob_idx < num_blobs(); blob_idx++) {
        if (get_blob_metadata(blob_idx).get_type() ==
            "apache-datasketches-theta-v1") {
            // Fetch the column index and decrement by 1 so it is zero-indexed.
            if (get_blob_metadata(blob_idx).get_fields().size() != 1) {
                // Bodo can only support theta sketches on a single field.
                continue;
            }
            int64_t field = get_blob_metadata(blob_idx).get_fields()[0];
            auto field_it = field_id_to_index.find(field);
            if (field_it == field_id_to_index.end()) {
                // The field id does not correspond to any column in the schema.
                // This can occur if schema evolution has caused a column to be
                // dropped.
                continue;
            }
            int64_t field_idx = field_it->second;
            // To ensure consistent with the sketches we write ensure we support
            // the type.
            if (type_supports_theta_sketch(
                    iceberg_schema->field(field_idx)->type())) {
                // Fetch the corresponding string and add it to the collection.
                std::string blob = get_blob(blob_idx);
                to_deserialize[field_idx] = blob;
            }
        }
    }
    return CompactSketchCollection::deserialize_sketches(to_deserialize);
}

/**
 * @brief Get Python statistics file object from C++ Puffin file
 * information.
 *
 * @param puffin The puffin information we use to create the statistics file.
 * @param puffin_loc The location of the puffin file.
 * @param snapshot_id The snapshot id of the puffin file's target data.
 * @param file_size_in_bytes The size of the puffin file in bytes.
 * @param footer_size The size of the footer in the puffin file in bytes.
 * @return PyObject* The Python statistics file object.
 */
PyObject *get_statistics_file_metadata(
    const std::unique_ptr<PuffinFile> &puffin, std::string puffin_loc,
    int64_t snapshot_id, size_t file_size_in_bytes, int32_t footer_size) {
    PyObject *ice = PyImport_ImportModule("pyiceberg.table.statistics");
    CHECK(ice,
          "importing PyIceberg submodule (pyiceberg.table.statistics) module "
          "failed");
    // Create the list of BlobMetadata objects
    PyObject *blob_metadata_class = PyObject_GetAttrString(ice, "BlobMetadata");
    CHECK(blob_metadata_class,
          "getting pyiceberg.table.statistics.BlobMetadata failed");
    size_t n_blobs = puffin->num_blobs();
    PyObject *blob_list = PyList_New(n_blobs);
    CHECK(blob_list, "creating blob list failed");

    PyObject *args = PyTuple_New(0);
    CHECK(args, "Creating empty args tuple failed");

    for (size_t i = 0; i < n_blobs; i++) {
        BlobMetadata blob_metadata = puffin->get_blob_metadata(i);
        // Get the fields
        std::vector<int64_t> fields = blob_metadata.get_fields();
        PyObject *fields_list = PyList_New(fields.size());
        CHECK(fields_list, "creating fields list failed");
        for (size_t j = 0; j < fields.size(); j++) {
            // PyList_SetItem steals the reference created by
            // PyLong_FromLong.
            CHECK(
                PyList_SetItem(fields_list, j, PyLong_FromLong(fields[j])) == 0,
                "setting fields list item failed");
        }
        // Get the properties
        PyObject *properties_dict = PyDict_New();
        CHECK(properties_dict, "creating properties dict failed");
        if (blob_metadata.has_properties()) {
            for (auto it : blob_metadata.get_properties()) {
                PyObject *key = PyUnicode_FromString(it.first.c_str());
                CHECK(key, "creating key string failed");
                PyObject *value = PyUnicode_FromString(it.second.c_str());
                CHECK(value, "creating value string failed");
                CHECK(PyDict_SetItem(properties_dict, key, value) == 0,
                      "setting properties dict item failed");
                Py_DECREF(key);
                Py_DECREF(value);
            }
        }

        // Call BlobMetadata(type, snap_id, seq_num, fields, properties)
        // Note, all args need to be kwargs
        PyObject *kwargs = Py_BuildValue(
            "{s:s,s:L,s:L,s:O,s:O}", "type", blob_metadata.get_type().c_str(),
            "snapshot-id", blob_metadata.get_snapshot_id(), "sequence-number",
            blob_metadata.get_sequence_number(), "fields", fields_list,
            "properties", properties_dict);
        CHECK(kwargs, "Creating args for BlobMetadata failed");
        PyObject *blob_metadata_obj =
            PyObject_Call(blob_metadata_class, args, kwargs);
        Py_DECREF(kwargs);

        CHECK(blob_metadata_obj, "creating BlobMetadata object failed");
        // PyList_SetItem steals the reference created by
        // the constructor.
        CHECK(PyList_SetItem(blob_list, i, blob_metadata_obj) == 0,
              "setting blob list item failed");
        Py_DECREF(fields_list);
        Py_DECREF(properties_dict);
    }
    Py_DECREF(blob_metadata_class);
    // Create the statistics file object
    PyObject *statistics_file_class =
        PyObject_GetAttrString(ice, "StatisticsFile");
    CHECK(statistics_file_class,
          "getting pyiceberg.table.statistics.StatisticsFile failed");

    PyObject *kwargs = Py_BuildValue(
        "{s:L,s:s,s:L,s:i,s:O}", "snapshot-id", snapshot_id, "statistics-path",
        puffin_loc.c_str(), "file-size-in-bytes", file_size_in_bytes,
        "file-footer-size-in-bytes", footer_size, "blob-metadata", blob_list);
    CHECK(kwargs, "Creating args for StatisticsFile failed");

    PyObject *statistics_file_obj =
        PyObject_Call(statistics_file_class, args, kwargs);

    CHECK(statistics_file_obj, "creating StatisticsFile object failed");
    Py_DECREF(kwargs);
    Py_DECREF(args);
    Py_DECREF(blob_list);
    Py_DECREF(statistics_file_class);
    Py_DECREF(ice);
    return statistics_file_obj;
}

/**
 * @brief Generate an equivalent empty pyiceberg.table.statistics.StatisticsFile
 * object for when the rank is not 0. This is used for type stability when
 * interacting with object mode.
 *
 * @return PyObject*
 */
PyObject *get_empty_statistics_file_metadata() {
    PyObject *ice = PyImport_ImportModule("pyiceberg.table.statistics");
    CHECK(ice, "importing pyiceberg.table.statistics module failed");
    PyObject *statistics_file_class =
        PyObject_GetAttrString(ice, "StatisticsFile");
    CHECK(statistics_file_class,
          "getting pyiceberg.table.statistics.StatisticsFile failed");

    PyObject *args = PyTuple_New(0);
    CHECK(args, "Creating empty args tuple failed");

    PyObject *blob_list = PyList_New(0);
    CHECK(blob_list, "Creating empty list failed");

    PyObject *kwargs = Py_BuildValue(
        "{s:L,s:s,s:L,s:i,s:O}", "snapshot-id", -1, "statistics-path", "",
        "file-size-in-bytes", -1, "file-footer-size-in-bytes", -1,
        "blob-metadata", blob_list);
    CHECK(kwargs, "Creating empty args for StatisticsFile failed");

    PyObject *statistics_file_obj =
        PyObject_Call(statistics_file_class, args, kwargs);

    CHECK(statistics_file_obj, "creating empty StatisticsFile object failed");
    Py_DECREF(blob_list);
    Py_DECREF(args);
    Py_DECREF(kwargs);
    Py_DECREF(statistics_file_class);
    Py_DECREF(ice);
    return statistics_file_obj;
}

/**
 * @brief Read a puffin file using an arrow file system and return the
 * result as a string.
 *
 * @param puffin_loc
 * @param bucket_region
 * @return std::unique_ptr<PuffinFile>
 */
std::unique_ptr<PuffinFile> read_puffin_file(std::string puffin_loc,
                                             std::string bucket_region) {
    auto info = get_reader_file_system(puffin_loc, bucket_region, false);
    std::shared_ptr<arrow::fs::FileSystem> &fs = info.first;
    const std::string &updated_puffin_loc = info.second;
    arrow::fs::FileInfo puffin_info;
    CHECK_ARROW_AND_ASSIGN(fs->GetFileInfo(updated_puffin_loc),
                           "Failed to get puffin file info", puffin_info);
    // Fetch the file size
    int64_t file_size = puffin_info.size();
    // TODO: Can this ever be < 0. The arrow source code mentions
    // "regular file", which should mean non-dictionary.
    CHECK(file_size >= 0, "Invalid file size");
    std::shared_ptr<arrow::io::InputStream> file_stream;
    CHECK_ARROW_AND_ASSIGN(fs->OpenInputStream(puffin_info),
                           "Failed to open puffin file", file_stream);
    std::string puffin_data(file_size, 0);
    CHECK_ARROW(file_stream->Read(file_size, puffin_data.data()).status(),
                "Failed to read puffin file");
    CHECK_ARROW(file_stream->Close(), "Failed to close puffin file");
    return PuffinFile::deserialize(puffin_data);
}

/**
 * Entrypoint from Python to initialize a PuffinFile write.
 * @param puffin_file_loc: Where to write the puffin file.
 * @param bucket_region: AWS bucket region.
 * @param snapshot_id: the snapshot_id to use for the PuffinFile metadata.
 * @param snapshot_id: the sequence_number to use for the PuffinFile metadata.
 * @param sketches: the collection of theta sketches to write.
 * @param iceberg_arrow_schema_py: the schema of the table being written.
 *        This is needed to determine the field_id for each column.
 * @param existing_puffin_file_loc: The location of an existing puffin file
 *        to combine with the new sketches when doing an insert into. If this
 *        is a create table it will be "".
 * @return: A StatisticsFile object with the information to forward to the
 *          Iceberg connector.
 */
PyObject *write_puffin_file_py_entrypt(
    const char *puffin_file_loc, const char *bucket_region, int64_t snapshot_id,
    int64_t sequence_number, UpdateSketchCollection *sketches,
    PyObject *iceberg_arrow_schema_py, PyObject *pyarrow_fs,
    const char *existing_puffin_file_loc) {
    try {
        std::shared_ptr<arrow::Schema> iceberg_schema;
        CHECK_ARROW_AND_ASSIGN(
            arrow::py::unwrap_schema(iceberg_arrow_schema_py),
            "Iceberg Schema Couldn't Unwrap from Python", iceberg_schema);

        // Gather the theta sketches onto rank 0
        std::shared_ptr<CompactSketchCollection> compact_sketches =
            sketches->compact_sketches();
        std::shared_ptr<CompactSketchCollection> merged_sketches =
            compact_sketches->merge_parallel_sketches();
        std::string existing_puffin_path(existing_puffin_file_loc);
        int rank;
        int n_pes;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        if (rank == 0) {
            if (!existing_puffin_path.empty()) {
                auto existing_puffin =
                    read_puffin_file(existing_puffin_path, bucket_region);
                merged_sketches = CompactSketchCollection::merge_sketches(
                    {existing_puffin->to_theta_sketches(iceberg_schema),
                     std::move(merged_sketches)});
            }
            std::string puffin_loc(puffin_file_loc);
            auto puff = PuffinFile::from_theta_sketches(
                merged_sketches, iceberg_schema, snapshot_id, sequence_number);
            // TODO: Should we write directly to the output buffer instead?
            auto serialized_result = puff->serialize();
            std::string &serialized = serialized_result.first;
            int32_t footer_size = serialized_result.second;
            // TODO: Refactor this old code into simpler/logical APIs.
            // This function does too much.
            std::string path_name;
            // Unused but we need to pass the directory.
            std::string dirname = "";
            std::string fname = "";
            Bodo_Fs::FsEnum fs_option;
            extract_fs_dir_path(puffin_file_loc, false, "", "", 0, n_pes,
                                &fs_option, &dirname, &fname, &puffin_loc,
                                &path_name);
            std::shared_ptr<::arrow::io::OutputStream> out_stream;

            if (arrow::py::import_pyarrow_wrappers()) {
                throw std::runtime_error("Importing pyarrow_wrappers failed!");
            }
            std::shared_ptr<arrow::fs::FileSystem> arrow_fs;
            CHECK_ARROW_AND_ASSIGN(
                arrow::py::unwrap_filesystem(pyarrow_fs),
                "Error during Iceberg write: Failed to unwrap Arrow filesystem",
                arrow_fs);

            std::filesystem::path out_path(dirname);
            out_path /= fname;  // append file name to output path
            arrow::Result<std::shared_ptr<arrow::io::OutputStream>> result =
                arrow_fs->OpenOutputStream(out_path.string());
            CHECK_ARROW_AND_ASSIGN(result, "FileOutputStream::Open",
                                   out_stream);

            CHECK_ARROW(out_stream->Write(serialized.data(), serialized.size()),
                        "Failed to write puffin data");
            CHECK_ARROW(out_stream->Close(), "Failed to close puffin files");
            return get_statistics_file_metadata(puff, puffin_loc, snapshot_id,
                                                serialized.size(), footer_size);
        } else {
            return get_empty_statistics_file_metadata();
        }
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

/**
 * @brief Python entrypoint for reading a puffin file
 * and returning a the ndv calculations. This is just
 * used for testing.
 * @param puffin_file_loc The location of the puffin file.
 * @param bucket_region_info The bucket region info.
 * @param iceberg_arrow_schema_py The schema of the iceberg table.
 */
array_info *read_puffin_file_ndvs_py_entrypt(
    const char *puffin_file_loc, const char *bucket_region_info,
    PyObject *iceberg_arrow_schema_py) {
    try {
        std::shared_ptr<arrow::Schema> iceberg_schema;
        CHECK_ARROW_AND_ASSIGN(
            arrow::py::unwrap_schema(iceberg_arrow_schema_py),
            "Iceberg Schema Couldn't Unwrap from Python", iceberg_schema);
        int rank;
        int n_pes;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        std::shared_ptr<CompactSketchCollection> result;
        if (rank == 0) {
            std::string puffin_loc(puffin_file_loc);
            std::string bucket_region(bucket_region_info);
            std::unique_ptr<PuffinFile> puffin =
                read_puffin_file(puffin_loc, bucket_region);
            result = puffin->to_theta_sketches(iceberg_schema);
        } else {
            std::vector<bool> ndv(iceberg_schema->num_fields(), false);
            result = std::make_unique<UpdateSketchCollection>(ndv)
                         ->compact_sketches();
        }
        return result->compute_ndv().release();
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

PyMODINIT_FUNC PyInit_puffin_file(void) {
    PyObject *m;
    MOD_DEF(m, "puffin_file", "No docs", nullptr);
    if (m == nullptr) {
        return nullptr;
    }

    bodo_common_init();

    SetAttrStringFromVoidPtr(m, write_puffin_file_py_entrypt);
    SetAttrStringFromVoidPtr(m, read_puffin_file_ndvs_py_entrypt);

    return m;
}
#undef CHECK
#undef CHECK_ARROW
#undef CHECK_ARROW_AND_ASSIGN
