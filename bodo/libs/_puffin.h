#pragma once

#include <Python.h>
#include <boost/json.hpp>

#include "_theta_sketches.h"

// Use libzstd to decompress a blob string.
// It is in this file for now because we don't have enough zstd functionality
// to give it its own file.
std::string decode_zstd(std::string blob);

/**
 * Class used to describe one of the blocks of metadata in the footer of the
 * Puffin file corresponding to one of the blobs.
 *
 * Confluence Doc:
 * https://bodo.atlassian.net/wiki/spaces/B/pages/1665368065/Puffin+File+C+Library
 */
class BlobMetadata {
   public:
    /**
     * @param _type: the type string of the metadata kind (currently only
     * supports theta sketch).
     * @param _fields: the list of column indices that the metadata refers to.
     * @param _snapshot_id: the Iceberg snapshot that the result was computed
     * from.
     * @param _sequence_number: the sequence number of the Iceberg table's
     * snapshot the blob was computed from.
     * @param _offset: the byte offset from the start of the file that this blob
     * starts at.
     * @param _length: the length of the blob this metadata refers to in bytes.
     * @param _compression_codec: optional string describing compression of the
     * blob.
     * @param _properties: optional [str,str] mapping describing properties of
     * the metadata, such as the NDV value for a theta sketch.
     */
    BlobMetadata(
        std::string _type, std::vector<int64_t> _fields, int64_t _snapshot_id,
        int64_t _sequence_number, int64_t _offset, int64_t _length,
        std::optional<std::string> _compression_codec,
        std::optional<std::unordered_map<std::string, std::string>> _properties)
        : type(_type),
          fields(_fields),
          snapshot_id(_snapshot_id),
          sequence_number(_sequence_number),
          offset(_offset),
          length(_length),
          compression_codec(_compression_codec),
          properties(_properties) {}

    /**
     * Retrieves the type string of the BlobMetadata.
     */
    std::string get_type() { return type; }

    /**
     * Retrieves the column fields list of the BlobMetadata.
     */
    std::vector<int64_t> &get_fields() { return fields; }

    /**
     * Retrieves the Iceberg snapshot ID of the BlobMetadata.
     */
    int64_t get_snapshot_id() { return snapshot_id; }

    /**
     * Retrieves the sequence number of the BlobMetadata.
     */
    int64_t get_sequence_number() { return sequence_number; }

    /**
     * Retrieves the byte offset of the blob that the metadata refers to.
     */
    int64_t get_offset() { return offset; }

    /**
     * Retrieves the byte length of the blob that the metadata refers to.
     */
    int64_t get_length() { return length; }

    /**
     * Returns whether the blob that the metadata refers to has a compression
     * string.
     */
    bool has_compression_codec() { return compression_codec.has_value(); }

    /**
     * Returns the compression string of the blob that the metadata refers to.
     */
    std::string get_compression_codec() { return compression_codec.value(); }

    /**
     * Returns whether the blob that the metadata refers to has a properties
     * map.
     */
    bool has_properties() { return properties.has_value(); }

    /**
     * Returns the properties map of the blob that the metadata refers to.
     */
    std::unordered_map<std::string, std::string> &get_properties() {
        return properties.value();
    }

    /**
     * Constructs a BlobMetadata object from a JSON object with the
     * following schema:
     * {
     *   "type": <string>,
     *   "fields": <integer list>,
     *   "snapshot-id": <integer>,
     *   "sequence_number": <integer>,
     *   "offset": <integer>,
     *   "length": <integer>,
     *   "compression_codec": <string>, // OPTIONAL
     *   "properties": <object>, // OPTIONAL
     * }
     */
    static BlobMetadata from_json(boost::json::object obj);

    /**
     * Converts the BlobMetadata back to the same json format that
     * from_json expects as an input.
     */
    boost::json::object to_json();

   private:
    std::string type;
    std::vector<int64_t> fields;
    int64_t snapshot_id;
    int64_t sequence_number;
    int64_t offset;
    int64_t length;
    std::optional<std::string> compression_codec;
    std::optional<std::unordered_map<std::string, std::string>> properties;
};

/**
 * Class used to describe a Puffin file, which has the following format:
 *
 *  magic
 *      blob_1
 *      blob_2
 *      ...
 *      blob_n
 *  magic
 *      footer
 *      footer_length
 *      flags
 *  magic
 *
 * Where the footer is a JSON object with the following format:
 *
 *  {
 *      "blobs": [blob_meta_1, blob_meta_2, ..., blob_meta_n]
 *      "properties": <object> // OPTIONAL
 *  }
 *
 * See BlobMetadata for description of the blob_metadata objects.
 */
class PuffinFile {
   public:
    /**
     * @param _blobs a collection of strings representing the blobs in the
     * Puffin file.
     * @param _blob_metadatas a collection of BlobMetadata objects corresponding
     * to each of the blobs.
     * @param _properites an optional [str,str] mapping of file-level
     * properties.
     */
    PuffinFile(
        std::vector<std::string> _blobs,
        std::vector<BlobMetadata> _blob_metadatas,
        std::optional<std::unordered_map<std::string, std::string>> _properties)
        : blobs(_blobs),
          blob_metadatas(_blob_metadatas),
          properties(_properties) {
        // We expect blobs and blob_metadatas to be the same length
        if (blobs.size() != blob_metadatas.size()) {
            throw std::runtime_error(
                "PuffinFile: blobs and blob_metadatas size mismatch");
        }
    }

    /**
     * Returns the number of blobs in the PuffinFile.
     */
    size_t num_blobs() { return blobs.size(); }

    /**
     * Takes in an index and returns the corresponding raw blob string.
     */
    std::string get_blob(size_t idx);

    /**
     * Takes in an index and returns the metadata of the corresponding blob.
     */
    BlobMetadata &get_blob_metadata(size_t idx) { return blob_metadatas[idx]; }

    /**
     * Returns whether the entire PuffinFile has a properties map.
     */
    bool has_properties() { return properties.has_value(); }

    /**
     * Retrieves the properties map for the entire PuffinFile.
     */
    std::unordered_map<std::string, std::string> &get_properties() {
        return properties.value();
    }

    /**
     * Converts the entire PuffinFile to a string such that if
     * deserialized, it would return the same PuffinFile object.
     * @return A pair of the serialized string and the length of the
     *         footer as this is needed for writing the metadata.
     */
    std::pair<std::string, int32_t> serialize();

    /**
     * Create a PuffinFile from parsing a string. The idea is that
     * deserialize(serialize(P)) is equivalent to puffin file P, and
     * serialize(deserialize(S)) is equivalent to string S.
     */
    static std::unique_ptr<PuffinFile> deserialize(std::string src);

    /**
     * Creates a PuffinFile from a collection of theta sketches.
     * Certain fields are not filled out / are filled out in a
     * manner that is potentially different versus the input:
     *
     * - PuffinFile properties: has "created by" -> "Bodo version X.X.X".
     * - BlobMetadata snapshot_id: replaced with a passed in value.
     * - BlobMetadata sequence_number: replaced with a passed in value.
     * - BlobMetadata compression_codec: always uses nothing (for now).
     * - BlobMetadata properties: always has "ndv" field.
     *
     * @param sketches: the collection of theta sketches to convert.
     * @param iceberg_schema: The schema of the Iceberg table that is used
     *     to extract the corresponding field_ids for each sketch.
     * @param snapshot_id: the Iceberg snapshot that the sketches were computed
     * from.
     * @param sequence_number: the sequence number of the Iceberg table's
     * snapshot.
     */
    static std::unique_ptr<PuffinFile> from_theta_sketches(
        std::shared_ptr<CompactSketchCollection> sketches,
        std::shared_ptr<arrow::Schema> iceberg_schema, int64_t snapshot_id,
        int64_t sequence_number);

    /**
     * Converts a PuffinFile into a collection of theta sketches.
     * @param iceberg_schema: The iceberg schema. This is used to map
     *    the field_ids in the BlobMetadata to the corresponding column
     *    locations.
     */
    std::shared_ptr<CompactSketchCollection> to_theta_sketches(
        std::shared_ptr<arrow::Schema> iceberg_schema);

   private:
    std::vector<std::string> blobs;
    std::vector<BlobMetadata> blob_metadatas;
    std::optional<std::unordered_map<std::string, std::string>> properties;
};
