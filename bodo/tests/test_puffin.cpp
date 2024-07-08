#include <arrow/util/key_value_metadata.h>
#include <fstream>
#include "../libs/_bodo_to_arrow.h"
#include "../libs/_puffin.h"
#include "./test.hpp"

/**
 * Reads a binary file from the bodo/tests/data/ directory and returns
 * it as a string.
 */
std::string read_data_file(std::string name) {
    std::string full_path = "bodo/tests/data/" + name;
    // Open a stream at the end of the file to fetch the file size.
    std::ifstream endstream(full_path,
                            std::ifstream::ate | std::ifstream::binary);
    size_t file_size = endstream.tellg();
    endstream.close();
    // Open a new stream at the start of the file and read a number of bytes
    // into a temporary buffer based on the observed file size
    std::ifstream fstream(full_path, std::ios::in | std::ios::binary);
    std::vector<char> buffer(file_size + 1);
    fstream.read(buffer.data(), file_size);
    fstream.close();
    // Convert into a std::string and return the string.
    std::string result(buffer.data(), file_size);
    return result;
}

// Helper utility to create nullable arrays used for testing.
// Creates a nullable column from vectors of values and nulls
template <Bodo_CTypes::CTypeEnum dtype, typename T>
std::shared_ptr<array_info> nullable_array_from_vector(
    std::vector<T> numbers, std::vector<bool> nulls) {
    size_t length = numbers.size();
    auto result = alloc_nullable_array_no_nulls(length, dtype);
    T *buffer = result->data1<bodo_array_type::NULLABLE_INT_BOOL, T>();
    for (size_t i = 0; i < length; i++) {
        if (nulls[i]) {
            buffer[i] = (T)numbers[i];
        } else {
            result->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i, false);
        }
    }
    return result;
}

// Variant of nullable_array_from_vector to build a string array from vectors
std::shared_ptr<array_info> string_array_from_vectors(
    bodo::vector<std::string> strings, bodo::vector<bool> nulls) {
    size_t length = strings.size();

    bodo::vector<uint8_t> null_bitmask((length + 7) >> 3, 0);
    for (size_t i = 0; i < length; i++) {
        SetBitTo(null_bitmask.data(), i, nulls[i]);
    }
    return create_string_array(Bodo_CTypes::STRING, null_bitmask, strings, -1);
}

/**
 * @brief Generate a dummy arrow schema for testing purposes
 * with the given number of fields. The types are assumed irrelevant
 * and each field is given an id `index + 1`.
 *
 * @param num_fields
 * @return std::shared_ptr<arrow::Schema>
 */
std::shared_ptr<arrow::Schema> generate_dummy_arrow_schema(size_t num_fields) {
    arrow::FieldVector fields;
    for (size_t i = 0; i < num_fields; i++) {
        auto field = arrow::field("field_" + std::to_string(i), arrow::int32());
        // TODO: Replace PARQUET:field_id with ICEBERG_FIELD_ID_MD_KEY, but
        // that requires code refactoring due to unused functions.
        auto metadata = arrow::KeyValueMetadata::Make({"PARQUET:field_id"},
                                                      {std::to_string(i + 1)});
        fields.emplace_back(field->WithMetadata(metadata));
    }
    return arrow::schema(fields);
}

/**
 * @brief Verifies that a PuffinFile matches the expected setup and
 *        properties of the nyc example puffin file.
 * @param[in] puff: the puffin file being verified.
 * @param[in] original_str: the original string that puff came from.
 * @param[in] trino_version: true if the puffin file came directly from
 *            the string produced by Trino, false if it was generated
 *            with our Bodo logic.
 */
void verify_nyc_puffin_file_metadata(const std::unique_ptr<PuffinFile> &puff,
                                     std::string original_str,
                                     bool trino_version) {
    // Verify the overarching statistics: 3 blobs, 1 property
    bodo::tests::check(puff != nullptr);
    bodo::tests::check(puff->num_blobs() == 3);
    bodo::tests::check(puff->has_properties());
    bodo::tests::check(puff->get_properties().size() == 1);
    bodo::tests::check(puff->get_properties().begin()->first == "created-by");
    if (trino_version) {
        bodo::tests::check(puff->get_properties().begin()->second ==
                           "Trino version 438-galaxy-1-u123-gcad7af3f6af");
    } else {
        bodo::tests::check(puff->get_properties().begin()->second ==
                           "Bodo version " + get_bodo_version());
    }

    // Verify the properties of the first blob's metadata
    BlobMetadata &meta_1 = puff->get_blob_metadata(0);
    bodo::tests::check(meta_1.get_type() == "apache-datasketches-theta-v1");
    bodo::tests::check(meta_1.get_fields().size() == 1);
    bodo::tests::check(meta_1.get_fields()[0] == 1);
    if (trino_version) {
        bodo::tests::check(meta_1.get_snapshot_id() == 3063996302594631353);
        bodo::tests::check(meta_1.get_sequence_number() == 1);
        bodo::tests::check(meta_1.get_offset() == 4);
        bodo::tests::check(meta_1.get_length() == 2150);
        bodo::tests::check(meta_1.has_compression_codec());
        bodo::tests::check(meta_1.get_compression_codec() == "zstd");
    } else {
        bodo::tests::check(meta_1.get_snapshot_id() == 123456789);
        bodo::tests::check(meta_1.get_sequence_number() == 5);
        bodo::tests::check((size_t)meta_1.get_offset() == 4);
        bodo::tests::check((size_t)meta_1.get_length() ==
                           puff->get_blob(0).size());
        bodo::tests::check(!meta_1.has_compression_codec());
    }
    bodo::tests::check(meta_1.has_properties());
    bodo::tests::check(meta_1.get_properties().size() == 1);
    bodo::tests::check(meta_1.get_properties().begin()->first == "ndv");
    bodo::tests::check(meta_1.get_properties().begin()->second == "265");

    // Verify the properties of the second blob's metadata
    BlobMetadata &meta_2 = puff->get_blob_metadata(1);
    bodo::tests::check(meta_2.get_type() == "apache-datasketches-theta-v1");
    bodo::tests::check(meta_2.get_fields().size() == 1);
    bodo::tests::check(meta_2.get_fields()[0] == 2);
    if (trino_version) {
        bodo::tests::check(meta_2.get_snapshot_id() == 3063996302594631353);
        bodo::tests::check(meta_2.get_sequence_number() == 1);
        bodo::tests::check(meta_2.get_offset() == 2154);
        bodo::tests::check(meta_2.get_length() == 85);
        bodo::tests::check(meta_2.has_compression_codec());
        bodo::tests::check(meta_2.get_compression_codec() == "zstd");
    } else {
        bodo::tests::check(meta_2.get_snapshot_id() == 123456789);
        bodo::tests::check(meta_2.get_sequence_number() == 5);
        bodo::tests::check((size_t)meta_2.get_offset() ==
                           4 + puff->get_blob(0).size());
        bodo::tests::check((size_t)meta_2.get_length() ==
                           puff->get_blob(1).size());
        bodo::tests::check(!meta_2.has_compression_codec());
    }
    bodo::tests::check(meta_2.has_properties());
    bodo::tests::check(meta_2.get_properties().size() == 1);
    bodo::tests::check(meta_2.get_properties().begin()->first == "ndv");
    bodo::tests::check(meta_2.get_properties().begin()->second == "7");

    // Verify the properties of the third blob's metadata
    BlobMetadata &meta_3 = puff->get_blob_metadata(2);
    bodo::tests::check(meta_3.get_type() == "apache-datasketches-theta-v1");
    bodo::tests::check(meta_3.get_fields().size() == 1);
    bodo::tests::check(meta_3.get_fields()[0] == 3);
    if (trino_version) {
        bodo::tests::check(meta_3.get_snapshot_id() == 3063996302594631353);
        bodo::tests::check(meta_3.get_sequence_number() == 1);
        bodo::tests::check(meta_3.get_offset() == 2239);
        bodo::tests::check(meta_3.get_length() == 2126);
        bodo::tests::check(meta_3.has_compression_codec());
        bodo::tests::check(meta_3.get_compression_codec() == "zstd");
    } else {
        bodo::tests::check(meta_3.get_snapshot_id() == 123456789);
        bodo::tests::check(meta_3.get_sequence_number() == 5);
        bodo::tests::check((size_t)meta_3.get_offset() ==
                           4 + puff->get_blob(0).size() +
                               puff->get_blob(1).size());
        bodo::tests::check((size_t)meta_3.get_length() ==
                           puff->get_blob(2).size());
        bodo::tests::check(!meta_3.has_compression_codec());
    }
    bodo::tests::check(meta_3.has_properties());
    bodo::tests::check(meta_3.get_properties().size() == 1);
    bodo::tests::check(meta_3.get_properties().begin()->first == "ndv");
    bodo::tests::check(meta_3.get_properties().begin()->second == "262");

    if (trino_version) {
        // Verify that the blobs match the expected sections of the raw string:
        bodo::tests::check(puff->get_blob(0) ==
                           decode_zstd(original_str.substr(4, 2150)));
        bodo::tests::check(puff->get_blob(1) ==
                           decode_zstd(original_str.substr(2154, 85)));
        bodo::tests::check(puff->get_blob(2) ==
                           decode_zstd(original_str.substr(2239, 2126)));
    }
}

static bodo::tests::suite tests([] {
    bodo::tests::test("test_puffin_read_nyc_metadata", [] {
        // Read in the nyc example file and parse it as a puffin file
        std::string nyc_example =
            read_data_file("puffin_data/nyc_example_puffin.stats");
        std::unique_ptr<PuffinFile> puff = PuffinFile::deserialize(nyc_example);

        // Verify that it has the expected properties
        verify_nyc_puffin_file_metadata(puff, nyc_example, true);
    });
    bodo::tests::test("test_puffin_read_nyc_serialize", [] {
        // Read in the nyc example file and parse it as a puffin file
        std::string nyc_example =
            read_data_file("puffin_data/nyc_example_puffin.stats");
        std::unique_ptr<PuffinFile> puff = PuffinFile::deserialize(nyc_example);

        // Re-serialize it and verify that it matches the original string
        auto serialize_result = puff->serialize();
        bodo::tests::check(serialize_result.first == nyc_example);
    });
    bodo::tests::test("test_puffin_read_nyc_theta_conversion", [] {
        std::shared_ptr<arrow::Schema> schema = generate_dummy_arrow_schema(4);
        // Read in the nyc example file and parse it as a puffin file
        std::string nyc_example =
            read_data_file("puffin_data/nyc_example_puffin.stats");
        std::unique_ptr<PuffinFile> puff = PuffinFile::deserialize(nyc_example);

        // Convert to a theta sketch with 4 columns (with the 4th as a
        // non-existant dummy column to make sure that the nullopt is
        // generated).
        auto collection = puff->to_theta_sketches(schema);

        // Verify that the theta sketch collection has the expected setup.
        bodo::tests::check(collection->column_has_sketch(0));
        bodo::tests::check(collection->column_has_sketch(1));
        bodo::tests::check(collection->column_has_sketch(2));
        bodo::tests::check(!collection->column_has_sketch(3));

        // Verify that the theta sketch collection has the expected
        // estimates.
        std::unique_ptr<array_info> ndvs = collection->compute_ndv();
        double *data =
            ndvs->data1<bodo_array_type::NULLABLE_INT_BOOL, double>();
        bodo::tests::check(data[0] == 265.0);
        bodo::tests::check(data[1] == 7.0);
        bodo::tests::check(data[2] == 262.0);
    });
    bodo::tests::test("test_puffin_read_nyc_theta_to_puffin", [] {
        // Read in the nyc example file and parse it as a puffin file
        std::string nyc_example =
            read_data_file("puffin_data/nyc_example_puffin.stats");
        std::unique_ptr<PuffinFile> puff = PuffinFile::deserialize(nyc_example);

        std::shared_ptr<arrow::Schema> schema = generate_dummy_arrow_schema(3);

        // Convert to a theta sketch with 3 columns
        auto collection_1 = puff->to_theta_sketches(schema);

        // Convert the theta sketches back to a puffin file and verify it
        // matches the same properties as the original puffin file.
        // Pass in a dummy snapshot_id & sequence_number: 123456789, 5
        std::unique_ptr<PuffinFile> new_puff =
            PuffinFile::from_theta_sketches(collection_1, schema, 123456789, 5);
        verify_nyc_puffin_file_metadata(new_puff, nyc_example, false);

        // Re-deserialize to make sure the serialized blobs were valid
        auto collection_2 = puff->to_theta_sketches(schema);
        bodo::tests::check(collection_2->column_has_sketch(0));
        bodo::tests::check(collection_2->column_has_sketch(1));
        bodo::tests::check(collection_2->column_has_sketch(2));
        std::unique_ptr<array_info> ndvs = collection_2->compute_ndv();
        double *data =
            ndvs->data1<bodo_array_type::NULLABLE_INT_BOOL, double>();
        bodo::tests::check(data[0] == 265.0);
        bodo::tests::check(data[1] == 7.0);
        bodo::tests::check(data[2] == 262.0);
    });
    bodo::tests::test("test_puffin_read_nyc_insert_data", [] {
        std::shared_ptr<arrow::Schema> schema = generate_dummy_arrow_schema(3);

        // Read in the nyc example file and parse it as a puffin file
        std::string nyc_example =
            read_data_file("puffin_data/nyc_example_puffin.stats");
        std::unique_ptr<PuffinFile> puff = PuffinFile::deserialize(nyc_example);

        // Convert to a theta sketch with 3 columns
        auto collection_1 = puff->to_theta_sketches(schema);

        // Merge with a batch of data with 5 rows and the following properties:
        // Column 0: 5 unique numbers, 2 of which overlap with existing data
        // Column 1: 5 unique strings, all of which overlap with existing data
        // Column 2: 4 unique strings, none of which overlap with existing data
        auto A0 = nullable_array_from_vector<Bodo_CTypes::INT32, int32_t>(
            {261, 9991, 2, 9993, 9994}, {true, true, true, true, true});
        auto A1 = string_array_from_vectors(
            {"Queens", "Bronx", "EWR", "Brooklyn", "Manhattan"},
            {true, true, true, true, true});
        auto A2 = string_array_from_vectors(
            {"Alpha", "Beta", "Gamma", "Delta", "Alpha"},
            {true, true, true, true, true});
        std::shared_ptr<table_info> T = std::make_shared<table_info>();
        T->columns.push_back(A0);
        T->columns.push_back(A1);
        T->columns.push_back(A2);
        std::vector<bool> alloc_sketches = {true, true, true};
        auto collection_2 =
            std::make_unique<UpdateSketchCollection>(alloc_sketches);
        auto arrow_table = bodo_table_to_arrow(T);
        for (int i = 0; i < arrow_table->num_columns(); i++) {
            auto column = arrow_table->column(i);
            collection_2->update_sketch(arrow_table->column(i), i);
        }
        auto collection_3 = CompactSketchCollection::merge_sketches(
            {collection_1, collection_2->compact_sketches()});

        // Convert the theta sketches back to a puffin file.
        // Pass in a dummy snapshot_id & sequence_number: 123456789, 5
        std::unique_ptr<PuffinFile> new_puff =
            PuffinFile::from_theta_sketches(collection_3, schema, 123456789, 5);

        // Re-deserialize to make sure the serialized blobs were valid
        // and have the correct new estimates
        std::shared_ptr<CompactSketchCollection> collection_4 =
            new_puff->to_theta_sketches(schema);
        bodo::tests::check(collection_4->column_has_sketch(0));
        bodo::tests::check(collection_4->column_has_sketch(1));
        bodo::tests::check(collection_4->column_has_sketch(2));
        std::unique_ptr<array_info> ndvs = collection_4->compute_ndv();
        double *data =
            ndvs->data1<bodo_array_type::NULLABLE_INT_BOOL, double>();
        bodo::tests::check(data[0] == 268.0);
        bodo::tests::check(data[1] == 7.0);
        bodo::tests::check(data[2] == 266.0);
    });
});
