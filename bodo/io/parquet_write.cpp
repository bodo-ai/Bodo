// Functions to write Bodo arrays to parquet

#include <arrow/filesystem/azurefs.h>
#include <arrow/filesystem/filesystem.h>
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <regex>
#if _MSC_VER >= 1900
#undef timezone
#endif

#include "parquet_write.h"

#include <arrow/compute/cast.h>
#include <arrow/ipc/writer.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <arrow/util/base64.h>
#include <arrow/util/key_value_metadata.h>
#include <parquet/arrow/schema.h>
#include <parquet/arrow/writer.h>
#include <parquet/file_writer.h>

#include "arrow_compat.h"

#include "../libs/_array_hash.h"
#include "../libs/_bodo_common.h"
#include "../libs/_bodo_to_arrow.h"
#include "../libs/_dict_builder.h"
#include "_fs_io.h"

// In general, when reading a parquet dataset we want it to have at least
// the same number of row groups as the number of processes we are reading with,
// but ideally much more to avoid possibility of multiple processes reading
// from the same row group (which can happen depending on how many process we
// read with and how the rows are distributed). At the same time, we don't want
// row groups to be too small because there will be a lot of overhead when
// reading (and could limit the benefit of dictionary encoding?)
// We also have to account for column projection. So, for example, we would like
// each column to be at least 1 MB on disk.
constexpr int64_t DEFAULT_ROW_GROUP_SIZE = 1000000;  // in number of rows

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

Bodo_Fs::FsEnum filesystem_type(const char *fname) {
    Bodo_Fs::FsEnum fs_type;
    if (strncmp(fname, "s3://", 5) == 0) {
        fs_type = Bodo_Fs::s3;
    } else if ((strncmp(fname, "abfs://", 7) == 0 ||
                strncmp(fname, "abfss://", 8) == 0) ||
               strncmp(fname, "hdfs://", 7) == 0) {
        fs_type = Bodo_Fs::hdfs;
    } else if ((strncmp(fname, "gcs://", 6) == 0) ||
               (strncmp(fname, "gs://", 5) == 0)) {
        fs_type = Bodo_Fs::gcs;
    } else {  // local
        fs_type = Bodo_Fs::posix;
    }
    return fs_type;
}

// ----------------------------------------------------------------------------
// The following three functions are copied from Arrow.
// We modify them slightly in order to pass a different schema to generate
// the Arrow schema string that is embedded as custom metadata in the parquet
// file. We do this because we don't want Bodo dictionary-encoded string arrays
// to be typed as Arrow dictionary arrays, but as strings. Arrow will by default
// type them as dictionary arrays because we use Arrow DictionaryArray to write
// them to parquet.
// NOTE: as far as parquet is concerned Arrow StringArray and DictionaryArray
// have the same parquet type: "BYTE_ARRAY / String"

// XXX Unfortunately we need to copy this function from Arrow because Arrow
// doesn't expose it in a header. And simply declaring it doesn't seem to be
// enough (there are dynamic linking issues, but maybe they can be solved and
// worth exploring)
// Only changes to this function are addition of namespace prefixes and clang
// formatting
static arrow::Status GetSchemaMetadata(
    const ::arrow::Schema &schema, ::arrow::MemoryPool *pool,
    const parquet::ArrowWriterProperties &properties,
    std::shared_ptr<const arrow::KeyValueMetadata> *out) {
    if (!properties.store_schema()) {
        *out = nullptr;
        return arrow::Status::OK();
    }

    static const std::string kArrowSchemaKey = "ARROW:schema";
    std::shared_ptr<arrow::KeyValueMetadata> result;
    if (schema.metadata()) {
        result = schema.metadata()->Copy();
    } else {
        result = ::arrow::key_value_metadata({}, {});
    }

    ARROW_ASSIGN_OR_RAISE(std::shared_ptr<arrow::Buffer> serialized,
                          ::arrow::ipc::SerializeSchema(schema, pool));

    // The serialized schema is not UTF-8, which is required for Thrift
    std::string schema_as_string = serialized->ToString();
    std::string schema_base64 = ::arrow::util::base64_encode(schema_as_string);
    result->Append(kArrowSchemaKey, schema_base64);
    *out = result;
    return arrow::Status::OK();
}

// This function is copied from Arrow.
// (https://github.com/apache/arrow/blob/apache-arrow-11.0.0/cpp/src/parquet/arrow/writer.cc#L521)
// Bodo change: pass a different schema for metadata (added schema_for_metadata
// parameter)
static arrow::Result<std::unique_ptr<parquet::arrow::FileWriter>>
FileWrite_Open(
    const ::arrow::Schema &schema, const ::arrow::Schema &schema_for_metadata,
    ::arrow::MemoryPool *pool, std::shared_ptr<::arrow::io::OutputStream> sink,
    std::shared_ptr<parquet::WriterProperties> properties,
    std::shared_ptr<parquet::ArrowWriterProperties> arrow_properties) {
    std::shared_ptr<parquet::SchemaDescriptor> parquet_schema;
    RETURN_NOT_OK(parquet::arrow::ToParquetSchema(
        &schema, *properties, *arrow_properties, &parquet_schema));

    auto schema_node = std::static_pointer_cast<parquet::schema::GroupNode>(
        parquet_schema->schema_root());

    std::shared_ptr<const arrow::KeyValueMetadata> metadata;
    // Bodo change: use schema_for_metadata to generate the Arrow schema to
    // be embedded in the parquet custom metadata
    RETURN_NOT_OK(GetSchemaMetadata(schema_for_metadata, pool,
                                    *arrow_properties, &metadata));

    std::unique_ptr<parquet::ParquetFileWriter> base_writer;
    PARQUET_CATCH_NOT_OK(base_writer = parquet::ParquetFileWriter::Open(
                             std::move(sink), schema_node,
                             std::move(properties), std::move(metadata)));

    std::unique_ptr<parquet::arrow::FileWriter> writer;
    auto schema_ptr = std::make_shared<::arrow::Schema>(schema);
    RETURN_NOT_OK(parquet::arrow::FileWriter::Make(
        pool, std::move(base_writer), std::move(schema_ptr),
        std::move(arrow_properties), &writer));
    return writer;
}

// This function is copied from Arrow.
// (https://github.com/apache/arrow/blob/apache-arrow-11.0.0/cpp/src/parquet/arrow/writer.cc#L560)
// Bodo change: passing through schema_for_metadata to FileWrite_Open
static arrow::Status WriteTable(
    const ::arrow::Table &table, ::arrow::MemoryPool *pool,
    std::shared_ptr<::arrow::io::OutputStream> sink, int64_t chunk_size,
    std::shared_ptr<parquet::WriterProperties> properties,
    std::shared_ptr<parquet::ArrowWriterProperties> arrow_properties,
    std::shared_ptr<arrow::Schema> schema_for_metadata) {
    std::unique_ptr<parquet::arrow::FileWriter> writer;
    ARROW_ASSIGN_OR_RAISE(
        writer, FileWrite_Open(*table.schema(), *schema_for_metadata, pool,
                               std::move(sink), std::move(properties),
                               std::move(arrow_properties)));
    RETURN_NOT_OK(writer->WriteTable(table, chunk_size));
    return writer->Close();
}
// ----------------------------------------------------------------------------

void pq_write_create_dir(const char *_path_name) {
    int myrank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    std::string orig_path(_path_name);
    std::string path_name;
    std::string dirname;
    std::string fname;
    std::shared_ptr<::arrow::io::OutputStream> out_stream;
    Bodo_Fs::FsEnum fs_option;
    bool is_parallel = true;
    const char *prefix = "";

    extract_fs_dir_path(_path_name, is_parallel, prefix, ".parquet", myrank,
                        num_ranks, &fs_option, &dirname, &fname, &orig_path,
                        &path_name);

    create_dir_parallel(fs_option, myrank, dirname, path_name, orig_path,
                        "parquet", true);
}

void pq_write_create_dir_py_entry(const char *_path_name) {
    try {
        pq_write_create_dir(_path_name);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

int64_t pq_write(const char *_path_name,
                 const std::shared_ptr<arrow::Table> &table,
                 const char *compression, bool is_parallel,
                 const char *bucket_region, int64_t row_group_size,
                 const char *prefix,
                 std::vector<bodo_array_type::arr_type_enum> bodo_array_types,
                 bool create_dir, std::string filename,
                 arrow::fs::FileSystem *arrow_fs) {
    tracing::Event ev("pq_write", is_parallel);
    ev.add_attribute("g_path", _path_name);
    ev.add_attribute("g_compression", compression);
    ev.add_attribute("nrows", table->num_rows());
    ev.add_attribute("prefix", prefix);
    ev.add_attribute("filename", filename);

    if (row_group_size == -1) {
        row_group_size = DEFAULT_ROW_GROUP_SIZE;
    } else if (row_group_size <= 0) {
        throw std::runtime_error(
            "to_parquet(): row_group_size must be greater than 0");
    }
    ev.add_attribute("g_row_group_size", row_group_size);

    int myrank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    std::string orig_path(_path_name);  // original path passed to this function
    std::string path_name;              // original path passed to this function
                                        // (excluding prefix)
    std::string dirname;  // path and directory name to store the parquet
                          // files (only if is_parallel=true)
    std::string fname;    // name of parquet file to write (excludes path)
    std::shared_ptr<::arrow::io::OutputStream> out_stream;
    Bodo_Fs::FsEnum fs_option;

    // Filesystem object to use if arrow_fs not provided.
    // Needs to be declared here so that it is not destroyed before
    // write is done.
    std::shared_ptr<arrow::fs::FileSystem> fs;

    extract_fs_dir_path(_path_name, is_parallel, prefix, ".parquet", myrank,
                        num_ranks, &fs_option, &dirname, &fname, &orig_path,
                        &path_name);
    std::filesystem::path out_path(dirname);
    out_path /= fname;  // append file name to output path

    // Create the fs
    if (arrow_fs == nullptr) {
        std::smatch match;
        // Ensure the path contains the "sv" and "sig" query parameters
        // since we only need to do this special path
        // if there's a SAS token.
        // This is ugly and should be moved to python, abfs_get_fs
        // when we upgrade to Arrow 20.
        std::regex r(".*[?&]sv=.*&sig=.*");
        regex_search(orig_path, match, r);
        if (fs_option == Bodo_Fs::abfs && match.size() != 0) {
#ifndef _WIN32
            arrow::fs::AzureOptions options;
            std::string updated_out_path;
            auto opt_res =
                arrow::fs::AzureOptions::FromUri(orig_path, &updated_out_path);
            out_path = std::filesystem::path(updated_out_path);
            CHECK_ARROW_AND_ASSIGN(opt_res, "AzureOptions::FromUri", options);
            auto fs_res = arrow::fs::AzureFileSystem::Make(options);
            CHECK_ARROW_AND_ASSIGN(fs_res, "AzureFileSystem::Make", fs);
            arrow_fs = fs.get();
#else
            throw std::runtime_error(
                "to_parquet: AzureFileSystem not supported on Windows.");
#endif
        } else {
            fs = get_fs_for_path(_path_name, is_parallel);
            arrow_fs = fs.get();
        }
    }

    // Avoid "\" generated on Windows for remote object storage
    // Get filesystem object if not provided
    std::string out_path_str = arrow_fs->type_name() == "local"
                                   ? out_path.string()
                                   : out_path.generic_string();

    // If filename is provided, use that instead of the generic one.
    // Currently this is used for Iceberg.
    // This will be refactored and moved to iceberg_pq_write in the
    // next PR.
    if ((filename.length() > 0) && is_parallel) {
        fname = filename;
    }

    // We need to create a directory when writing a distributed
    // table to a posix or hadoop filesystem.
    if (is_parallel && create_dir) {
        create_dir_parallel(fs_option, myrank, dirname, path_name, orig_path,
                            "parquet", true);
    }

    // Do not write a file if there are no rows to write.
    if (table->num_rows() == 0) {
        return 0;
    }

    arrow::Result<std::shared_ptr<arrow::io::OutputStream>> result =
        arrow_fs->OpenOutputStream(out_path_str);
    CHECK_ARROW_AND_ASSIGN(result, "FileOutputStream::Open", out_stream);

    auto pool = bodo::BufferPool::DefaultPtr();

    std::vector<std::shared_ptr<arrow::Field>> schema_for_metadata_vector;
    bool has_dictionary_columns = false;
    std::shared_ptr<arrow::Field> dict_str_field;
    for (int i = 0; i < table->num_columns(); i++) {
        auto field = table->field(i);
        if (bodo_array_types[i] == bodo_array_type::DICT) {
            if (!arrow::is_dictionary(field->type()->id())) {
                throw std::runtime_error(
                    "to_parquet: Arrow type of dictionary-encoded column is "
                    "not dictionary");
            }
            // For the custom metadata that Arrow puts in the parquet file, we
            // will indicate that this is a string column, so Arrow doesn't
            // identify it as dictionary when reading the written parquet file
            schema_for_metadata_vector.push_back(
                arrow::field(field->name(), arrow::utf8(), /*nullable=*/true,
                             field->metadata()));
            has_dictionary_columns = true;
            dict_str_field = field;
        } else {
            schema_for_metadata_vector.push_back(field);
        }
    }

    // Make metadata Schema object
    std::shared_ptr<arrow::Schema> schema_for_metadata =
        std::make_shared<arrow::Schema>(schema_for_metadata_vector,
                                        table->schema()->metadata());

    // set compression option
    ::arrow::Compression::type codec_type;
    if (strcmp(compression, "snappy") == 0) {
        codec_type = ::arrow::Compression::SNAPPY;
    } else if (strcmp(compression, "brotli") == 0) {
        codec_type = ::arrow::Compression::BROTLI;
    } else if (strcmp(compression, "gzip") == 0) {
        codec_type = ::arrow::Compression::GZIP;
    } else {
        codec_type = ::arrow::Compression::UNCOMPRESSED;
    }
    parquet::WriterProperties::Builder prop_builder;
    prop_builder.compression(codec_type);
    std::shared_ptr<parquet::WriterProperties> writer_properties =
        prop_builder.build();
    std::shared_ptr<parquet::ArrowWriterProperties> arrow_writer_properties =
        ::parquet::ArrowWriterProperties::Builder()
            .coerce_timestamps(::arrow::TimeUnit::MICRO)
            ->allow_truncated_timestamps()
            ->store_schema()
            // `enable_compliant_nested_types` ensures that nested types have
            // 'element' as their `name`. This is important for reading Iceberg
            // datasets written by Bodo, but also for standardization.
            ->enable_compliant_nested_types()
// Required for copying TIME data to Snowflake
#if ARROW_VERSION_MAJOR >= 22
            ->set_time_adjusted_to_utc(true)
#endif
            ->build();

    if (has_dictionary_columns) {
        // Make sure that Arrow stores dictionary-encoded column as Parquet
        // type BYTE_ARRAY / String, because in the Arrow schema that is
        // inserted in the parquet metadata we are saying that it's a string
        // column
        // XXX This only checks it for the last dict column, is that
        // sufficient??
        std::shared_ptr<parquet::schema::Node> node;
        arrow::Status arrowOpStatus =
            parquet::arrow::FieldToNode(dict_str_field, *writer_properties,
                                        *arrow_writer_properties, &node);
        if (!arrowOpStatus.ok()) {
            throw std::runtime_error(
                "Arrow unable to convert dictionary array to node.");
        }
        if (node->is_primitive()) {
            auto primitive_node =
                std::dynamic_pointer_cast<parquet::schema::PrimitiveNode>(node);
            if ((primitive_node->physical_type() !=
                 parquet::Type::BYTE_ARRAY) ||
                (!primitive_node->logical_type()->Equals(
                    *parquet::LogicalType::String()))) {
                throw std::runtime_error(
                    "Arrow is not storing dictionary array as parquet string");
            }
        } else {
            throw std::runtime_error(
                "Arrow is not storing dictionary array as parquet string");
        }
    }

    // open file and write table
    arrow::Status status = /*parquet::arrow::*/ WriteTable(
        *table, pool, out_stream, row_group_size, writer_properties,
        // store_schema() = true is needed to write the schema metadata to
        // file
        // .coerce_timestamps(::arrow::TimeUnit::MICRO)->allow_truncated_timestamps()
        // not needed when moving to parquet 2.0
        arrow_writer_properties, schema_for_metadata);
    CHECK_ARROW(status, "parquet::arrow::WriteTable");
    arrow::Result<int64_t> tell_result = out_stream->Tell();
    int64_t file_size;
    CHECK_ARROW_AND_ASSIGN(tell_result, "arrow::io::OutputStream::Tell",
                           file_size);
    ev.add_attribute("file_size", file_size);
    return file_size;
}

/**
 * @brief Convert the metadata string to a KeyValueMetadata object
 * to include in the Parquet schema footer.
 *
 * @param metadata The string metadata provided from Python.
 *
 * @return std::shared_ptr<arrow::KeyValueMetadata>
 */
std::shared_ptr<arrow::KeyValueMetadata> convert_parquet_schema_metadata(
    const char *metadata) {
    std::string_view new_metadata(metadata);
    std::shared_ptr<arrow::KeyValueMetadata> schema_metadata;
    if (new_metadata.size() > 0 && new_metadata[0] != 0) {
        std::unordered_map<std::string, std::string> new_metadata_map = {
            {"pandas", new_metadata.data()}};
        schema_metadata = ::arrow::key_value_metadata(new_metadata_map);
    }
    return schema_metadata;
}

int64_t pq_write_py_entry(const char *_path_name, table_info *table,
                          array_info *col_names_arr, const char *metadata,
                          const char *compression, bool is_parallel,
                          const char *bucket_region, int64_t row_group_size,
                          const char *prefix, bool convert_timedelta_to_int64,
                          const char *tz, bool downcast_time_ns_to_us,
                          bool create_dir) {
    try {
        tracing::Event ev("pq_write_py_entry", is_parallel);
        ev.add_attribute("g_metadata", metadata);

        std::shared_ptr<table_info> table_ptr =
            std::shared_ptr<table_info>(table);
        std::shared_ptr<array_info> col_names_arr_ptr =
            std::shared_ptr<array_info>(col_names_arr);
        std::vector<std::string> col_names =
            array_to_string_vector(col_names_arr_ptr);

        // Generate the metadata for the arrow table, including any index
        // metadata.
        std::shared_ptr<arrow::KeyValueMetadata> schema_metadata =
            convert_parquet_schema_metadata(metadata);
        std::shared_ptr<arrow::Table> arrow_table = bodo_table_to_arrow(
            table_ptr, col_names, schema_metadata, convert_timedelta_to_int64,
            tz, arrow::TimeUnit::NANO, downcast_time_ns_to_us);
        std::vector<bodo_array_type::arr_type_enum> bodo_array_types;
        for (auto col : table_ptr->columns) {
            bodo_array_types.emplace_back(col->arr_type);
        }
        int64_t file_size = pq_write(
            _path_name, arrow_table, compression, is_parallel, bucket_region,
            row_group_size, prefix, bodo_array_types, create_dir, "", nullptr);
        return file_size;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return -1;
    }
}

void pq_write_partitioned_py_entry(
    const char *_path_name, table_info *in_table, array_info *in_col_names_arr,
    array_info *in_col_names_arr_no_partitions, table_info *in_categories_table,
    int *partition_cols_idx, const int num_partition_cols,
    const char *compression, bool is_parallel, const char *bucket_region,
    int64_t row_group_size, const char *prefix, const char *tz) {
    // TODOs
    // - Do is parallel here?
    // - sequential (only rank 0 writes, or all write with same name -which?-)
    // - create directories
    //     - what if directories already have files?
    // - write index
    // - write metadata?
    // - convert values to strings for other dtypes like datetime, decimal, etc
    // (see array_val_to_str)

    try {
        if (!is_parallel) {
            throw std::runtime_error(
                "to_parquet partitioned not implemented in sequential mode");
        }

        // convert raw pointers to smart pointers to enable automatic
        // refcounting
        std::shared_ptr<table_info> table =
            std::shared_ptr<table_info>(in_table);
        std::shared_ptr<array_info> col_names_arr =
            std::shared_ptr<array_info>(in_col_names_arr);
        std::shared_ptr<array_info> col_names_arr_no_partitions =
            std::shared_ptr<array_info>(in_col_names_arr_no_partitions);
        std::shared_ptr<table_info> categories_table =
            std::shared_ptr<table_info>(in_categories_table);

        // new_table will have partition columns at the beginning and the rest
        // after (to use multi_col_key for hashing which assumes that keys are
        // at the beginning), and we will then drop the partition columns from
        // it for writing
        std::shared_ptr<table_info> new_table = std::make_shared<table_info>();
        std::vector<bool> is_part_col(table->ncols(), false);
        std::vector<std::shared_ptr<array_info>> partition_cols;
        std::vector<std::string> part_col_names;
        offset_t *offsets = (offset_t *)col_names_arr->data2();
        for (int i = 0; i < num_partition_cols; i++) {
            int j = partition_cols_idx[i];
            is_part_col[j] = true;
            partition_cols.push_back(table->columns[j]);
            new_table->columns.push_back(table->columns[j]);
            char *cur_str = col_names_arr->data1() + offsets[j];
            size_t len = offsets[j + 1] - offsets[j];
            part_col_names.emplace_back(cur_str, len);
        }
        for (uint64_t i = 0; i < table->ncols(); i++) {
            if (!is_part_col[i]) {
                new_table->columns.push_back(table->columns[i]);
            }
        }
        // Convert all local dictionaries to global for dict columns.
        // to enable hashing. Here we need a global dictionary with
        // unique values.
        // TODO: Does parquet actually require global values
        for (auto a : partition_cols) {
            if (a->arr_type == bodo_array_type::DICT) {
                make_dictionary_global_and_unique(a, is_parallel);
            }
        }

        const uint32_t seed = SEED_HASH_PARTITION;
        std::shared_ptr<uint32_t[]> hashes =
            hash_keys(partition_cols, seed, is_parallel);
        bodo::unord_map_container<multi_col_key, partition_write_info,
                                  multi_col_key_hash>
            key_to_partition;

        // TODO nullable partition cols?

        std::string fname = gen_pieces_file_name(
            dist_get_rank(), dist_get_size(), prefix, ".parquet");

        for (uint64_t i = 0; i < new_table->nrows(); i++) {
            multi_col_key key(hashes[i], new_table, i, num_partition_cols);
            partition_write_info &p = key_to_partition[key];
            if (p.rows.size() == 0) {
                // generate output file name
                p.fpath = std::string(_path_name);
                if (p.fpath.back() != '/') {
                    p.fpath += "/";
                }
                int64_t cat_col_idx = 0;
                for (int j = 0; j < num_partition_cols; j++) {
                    auto part_col = partition_cols[j];
                    // convert partition col value to string
                    std::string value_str;
                    if (part_col->arr_type == bodo_array_type::CATEGORICAL) {
                        int64_t code = part_col->get_code_as_int64(i);
                        // TODO can code be -1 (NA) for partition columns?
                        value_str = array_val_to_str(
                            categories_table->columns[cat_col_idx++], code);
                    } else if (part_col->arr_type == bodo_array_type::DICT) {
                        // check nullable bitmask and set string to empty
                        // if nan. Since we called
                        // `make_dictionary_global_and_unique` on all dict
                        // encoded arrays, we can be sure that there's no nulls
                        // in the dictionary of the dict-encoded arrays (since
                        // `is_local_unique` means no nulls in the
                        // dict)
                        // TODO(njriasan): Simplify/remove this assumption by
                        // having a null count in individual arrays we can just
                        // check.
                        bool isna = !GetBit(
                            (uint8_t *)part_col->child_arrays[1]
                                ->null_bitmask<
                                    bodo_array_type::NULLABLE_INT_BOOL>(),
                            i);
                        if (isna) {
                            value_str = "null";
                        } else {
                            int32_t dict_ind =
                                ((int32_t *)part_col->child_arrays[1]
                                     ->data1<bodo_array_type::
                                                 NULLABLE_INT_BOOL>())[i];
                            // get start_offset and end_offset of the string
                            // value referred to by dict_index i
                            offset_t start_offset =
                                ((offset_t *)part_col->child_arrays[0]
                                     ->data2<
                                         bodo_array_type::STRING>())[dict_ind];
                            offset_t end_offset =
                                ((offset_t *)part_col->child_arrays[0]
                                     ->data2<bodo_array_type::STRING>())
                                    [dict_ind + 1];
                            // get length of the string value
                            offset_t len = end_offset - start_offset;
                            // extract string value from string buffer
                            std::string val(
                                &((char *)part_col->child_arrays[0]
                                      ->data1<bodo_array_type::STRING>())
                                    [start_offset],
                                len);
                            value_str = val;
                        }
                    } else {
                        value_str = array_val_to_str(part_col, i);
                    }
                    p.fpath += part_col_names[j] + "=" + value_str + "/";
                }
                p.fpath += fname;
            }
            p.rows.push_back(i);
        }
        hashes.reset();

        // drop partition columns from new_table (they are not written to
        // parquet)
        new_table->columns.erase(
            new_table->columns.begin(),
            new_table->columns.begin() + num_partition_cols);

        std::vector<std::string> arrow_column_names =
            array_to_string_vector(col_names_arr_no_partitions);
        for (auto &it : key_to_partition) {
            const partition_write_info &p = it.second;
            std::shared_ptr<table_info> part_table =
                RetrieveTable(new_table, p.rows, new_table->ncols());
            // NOTE: we pass is_parallel=False because we already took care
            // of is_parallel here
            Bodo_Fs::FsEnum fs_type = filesystem_type(p.fpath.c_str());
            if (fs_type == Bodo_Fs::FsEnum::posix) {
                // s3 and hdfs create parent directories automatically when
                // writing partitioned columns
                std::filesystem::path path = p.fpath;
                std::filesystem::create_directories(path.parent_path());
            }
            std::shared_ptr<arrow::Table> arrow_table = bodo_table_to_arrow(
                part_table, arrow_column_names, {},
                false /* TODO: convert_timedelta_to_int64*/, tz,
                arrow::TimeUnit::NANO, false /* TODO: downcast_time_ns_to_us*/
            );
            std::vector<bodo_array_type::arr_type_enum> bodo_array_types;
            for (auto col : part_table->columns) {
                bodo_array_types.emplace_back(col->arr_type);
            }
            pq_write(p.fpath.c_str(), arrow_table, compression, false,
                     bucket_region, row_group_size, prefix, bodo_array_types);
        }

    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}
