// Copyright (C) 2019 Bodo Inc. All rights reserved.

// Functions to write Bodo arrays to parquet

#if _MSC_VER >= 1900
#undef timezone
#endif

#include "../libs/_array_hash.h"
#include "../libs/_bodo_common.h"
#include "../libs/_datetime_ext.h"
#include "_fs_io.h"
#include "arrow/ipc/writer.h"
#include "arrow/util/base64.h"
#include "parquet/arrow/schema.h"
#include "parquet/arrow/writer.h"
#include "parquet/file_writer.h"

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
#define CHECK_ARROW_AND_ASSIGN(res, msg, lhs) \
    CHECK_ARROW(res.status(), msg)            \
    lhs = std::move(res).ValueOrDie();

/// Convert Bodo date (year, month, day) from int64 to Arrow date32
static inline int32_t bodo_date64_to_arrow_date32(int64_t date) {
    int64_t year = date >> 32;
    int64_t month = (date >> 16) & 0xFFFF;
    int64_t day = date & 0xFFFF;
    // NOTE that get_days_from_date returns int64 and we are downcasting to
    // int32
    return get_days_from_date(year, month, day);
}

/// Convert Bodo date array (year, month, day elements) to Arrow date32 array
static void CastBodoDateToArrowDate32(const int64_t *input, int64_t length,
                                      int32_t *output) {
    for (int64_t i = 0; i < length; ++i) {
        *output++ = bodo_date64_to_arrow_date32(*input++);
    }
}

void bodo_array_to_arrow(
    arrow::MemoryPool *pool, const array_info *array,
    const std::string &col_name,
    std::vector<std::shared_ptr<arrow::Field>> &schema_vector,
    std::shared_ptr<arrow::ChunkedArray> *out) {
    // allocate null bitmap
    std::shared_ptr<arrow::ResizableBuffer> null_bitmap;
    int64_t null_bytes = arrow::bit_util::BytesForBits(array->length);
    arrow::Result<std::unique_ptr<arrow::ResizableBuffer>> res =
        AllocateResizableBuffer(null_bytes, pool);
    CHECK_ARROW_AND_ASSIGN(res, "AllocateResizableBuffer", null_bitmap);
    // Padding zeroed by AllocateResizableBuffer
    memset(null_bitmap->mutable_data(), 0, static_cast<size_t>(null_bytes));

    int64_t null_count_ = 0;
    if (array->arr_type == bodo_array_type::ARROW) {
        schema_vector.push_back(arrow::field(col_name, array->array->type()));
        std::shared_ptr<arrow::ArrayData> typ_data = array->array->data();
        *out = std::make_shared<arrow::ChunkedArray>(array->array);
    }

    if (array->arr_type == bodo_array_type::NULLABLE_INT_BOOL ||
        array->arr_type == bodo_array_type::STRING) {
        // set arrow bit mask based on bodo bitmask
        for (int64_t i = 0; i < array->length; i++) {
            if (!GetBit((uint8_t *)array->null_bitmask, i)) {
                null_count_++;
                SetBitTo(null_bitmap->mutable_data(), i, false);
            } else {
                SetBitTo(null_bitmap->mutable_data(), i, true);
            }
        }
        if (array->dtype == Bodo_CTypes::_BOOL) {
            // special case: nullable bool column are bit vectors in Arrow
            schema_vector.push_back(arrow::field(col_name, arrow::boolean()));
            int64_t nbytes = ::arrow::bit_util::BytesForBits(array->length);
            std::shared_ptr<::arrow::Buffer> buffer;
            arrow::Result<std::unique_ptr<arrow::Buffer>> res =
                AllocateBuffer(nbytes, pool);
            CHECK_ARROW_AND_ASSIGN(res, "AllocateBuffer", buffer);

            int64_t i = 0;
            uint8_t *in_data = (uint8_t *)array->data1;
            const auto generate = [&in_data, &i]() -> bool {
                return in_data[i++] != 0;
            };
            ::arrow::internal::GenerateBitsUnrolled(buffer->mutable_data(), 0,
                                                    array->length, generate);

            auto arr_data =
                arrow::ArrayData::Make(arrow::boolean(), array->length,
                                       {null_bitmap, buffer}, null_count_, 0);
            *out = std::make_shared<arrow::ChunkedArray>(
                arrow::MakeArray(arr_data));
        }
    }
    // TODO: Reuse some of this code to enable to_parquet with Categorical
    // Arrays?
    if (array->arr_type == bodo_array_type::DICT) {
        // C++ dictionary arrays in Bodo are dictionary arrays in Arrow.
        // We construct the array using arrow::DictionaryArray::FromArrays
        // https://arrow.apache.org/docs/cpp/api/array.html#_CPPv4N5arrow15DictionaryArray10FromArraysERKNSt10shared_ptrI8DataTypeEERKNSt10shared_ptrI5ArrayEERKNSt10shared_ptrI5ArrayEE

        // Dummy vector to enable recursive calls.
        std::vector<std::shared_ptr<arrow::Field>> dummy_schema_vector;
        std::vector<std::shared_ptr<arrow::ChunkedArray>> dict_parts(2);

        // Recurse on the dictionary
        bodo_array_to_arrow(pool, array->info1, col_name, dummy_schema_vector,
                            &dict_parts[0]);
        // Recurse on the index array
        bodo_array_to_arrow(pool, array->info2, col_name, dummy_schema_vector,
                            &dict_parts[1]);

        // Extract the types from the dictionary call.
        std::shared_ptr<arrow::DataType> type = arrow::dictionary(
            dummy_schema_vector[1]->type(), dummy_schema_vector[0]->type()
            // TODO: Can we provide ordered?
        );
        schema_vector.push_back(arrow::field(col_name, type));

        // bodo_array_to_arrow for primitive and string arrays returns
        // single-chunk chunked arrays, so we know there is only one chunk
        std::shared_ptr<arrow::Array> dictionary = dict_parts[0]->chunk(0);
        std::shared_ptr<arrow::Array> index_array = dict_parts[1]->chunk(0);
        arrow::Result<std::shared_ptr<arrow::Array>> result =
            arrow::DictionaryArray::FromArrays(type, index_array, dictionary);
        std::shared_ptr<arrow::Array> dict_array;
        CHECK_ARROW_AND_ASSIGN(result, "arrow::DictionaryArray::FromArrays",
                               dict_array)

        *out = std::make_shared<arrow::ChunkedArray>(dict_array);
    }

    if (array->arr_type == bodo_array_type::NUMPY ||
        (array->arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
         array->dtype != Bodo_CTypes::_BOOL)) {
        int64_t in_num_bytes;
        std::shared_ptr<arrow::DataType> type;
        arrow::Result<std::shared_ptr<arrow::DataType>> type_res;
        switch (array->dtype) {
            case Bodo_CTypes::INT8:
                in_num_bytes = sizeof(int8_t) * array->length;
                type = arrow::int8();
                break;
            case Bodo_CTypes::UINT8:
                in_num_bytes = sizeof(uint8_t) * array->length;
                type = arrow::uint8();
                break;
            case Bodo_CTypes::INT16:
                in_num_bytes = sizeof(int16_t) * array->length;
                type = arrow::int16();
                break;
            case Bodo_CTypes::UINT16:
                in_num_bytes = sizeof(uint16_t) * array->length;
                type = arrow::uint16();
                break;
            case Bodo_CTypes::INT32:
                in_num_bytes = sizeof(int32_t) * array->length;
                type = arrow::int32();
                break;
            case Bodo_CTypes::UINT32:
                in_num_bytes = sizeof(uint32_t) * array->length;
                type = arrow::uint32();
                break;
            case Bodo_CTypes::INT64:
                in_num_bytes = sizeof(int64_t) * array->length;
                type = arrow::int64();
                break;
            case Bodo_CTypes::UINT64:
                in_num_bytes = sizeof(uint64_t) * array->length;
                type = arrow::uint64();
                break;
            case Bodo_CTypes::FLOAT32:
                in_num_bytes = sizeof(float) * array->length;
                type = arrow::float32();
                break;
            case Bodo_CTypes::FLOAT64:
                in_num_bytes = sizeof(double) * array->length;
                type = arrow::float64();
                break;
            case Bodo_CTypes::DECIMAL:
                in_num_bytes = BYTES_PER_DECIMAL * array->length;
                type_res =
                    arrow::Decimal128Type::Make(array->precision, array->scale);
                CHECK_ARROW_AND_ASSIGN(type_res, "arrow::Decimal128Type::Make",
                                       type);
                break;
            case Bodo_CTypes::DATE:
                // input from Bodo uses int64 for dates
                in_num_bytes = sizeof(int64_t) * array->length;
                type = arrow::date32();
                break;
            case Bodo_CTypes::DATETIME:
                // input from Bodo uses int64 for datetimes (datetime64[ns])
                in_num_bytes = sizeof(int64_t) * array->length;
                type = arrow::timestamp(arrow::TimeUnit::NANO);
                // convert Bodo NaT to Arrow null bitmap
                for (int64_t i = 0; i < array->length; i++) {
                    if (array->at<int64_t>(i) ==
                        std::numeric_limits<int64_t>::min()) {
                        // if value is NaT (equals
                        // std::numeric_limits<int64_t>::min()) we set it as a
                        // null element in output Arrow array
                        null_count_++;
                        SetBitTo(null_bitmap->mutable_data(), i, false);
                    } else {
                        SetBitTo(null_bitmap->mutable_data(), i, true);
                    }
                }
                break;
            default:
                std::cerr << "Fatal error: invalid dtype found in conversion"
                             " of numeric Bodo array to Arrow"
                          << std::endl;
                exit(1);
        }
        schema_vector.push_back(arrow::field(col_name, type));
        std::shared_ptr<arrow::Buffer> out_buffer;
        if (array->dtype == Bodo_CTypes::DATE) {
            // allocate buffer to store date32 values in Arrow format
            arrow::Result<std::unique_ptr<arrow::Buffer>> res =
                AllocateBuffer(sizeof(int32_t) * array->length, pool);
            CHECK_ARROW_AND_ASSIGN(res, "AllocateBuffer", out_buffer);
            CastBodoDateToArrowDate32((int64_t *)array->data1, array->length,
                                      (int32_t *)out_buffer->mutable_data());
        } else {
            // we can use the same input buffer (no need to cast or convert)
            out_buffer = std::make_shared<arrow::Buffer>(
                (uint8_t *)array->data1, in_num_bytes);
        }

        auto arr_data = arrow::ArrayData::Make(
            type, array->length, {null_bitmap, out_buffer}, null_count_, 0);
        *out =
            std::make_shared<arrow::ChunkedArray>(arrow::MakeArray(arr_data));
    } else if (array->arr_type == bodo_array_type::STRING) {
        std::shared_ptr<arrow::DataType> arrow_type;
        if (array->dtype == Bodo_CTypes::BINARY) {
#if OFFSET_BITWIDTH == 64
            arrow_type = arrow::large_binary();
#else
            arrow_type = arrow::binary();
#endif
        } else {
#if OFFSET_BITWIDTH == 64
            arrow_type = arrow::large_utf8();
#else
            arrow_type = arrow::utf8();
#endif
        }
        schema_vector.push_back(arrow::field(col_name, arrow_type));

        // we use the same input Bodo buffers (no need to copy to new buffers)
        const int64_t n_strings = array->length;
        const int64_t n_chars = ((offset_t *)array->data2)[n_strings];

        std::shared_ptr<arrow::Buffer> chars_buffer =
            std::make_shared<arrow::Buffer>((uint8_t *)array->data1, n_chars);

        std::shared_ptr<arrow::Buffer> offsets_buffer =
            std::make_shared<arrow::Buffer>((uint8_t *)array->data2,
                                            sizeof(offset_t) * (n_strings + 1));

        auto arr_data = arrow::ArrayData::Make(
            arrow_type, n_strings, {null_bitmap, offsets_buffer, chars_buffer},
            null_count_, /*offset=*/0);
        *out =
            std::make_shared<arrow::ChunkedArray>(arrow::MakeArray(arr_data));

    } else if (array->arr_type == bodo_array_type::LIST_STRING) {
        // TODO try to have Arrow reuse Bodo buffers instead of copying to
        // new buffers
        schema_vector.push_back(
            arrow::field(col_name, arrow::list(arrow::utf8())));
        int64_t num_lists = array->length;
        char *chars = (char *)array->data1;
        int64_t *char_offsets = (int64_t *)array->data2;
        int64_t *string_offsets = (int64_t *)array->data3;
        arrow::ListBuilder list_builder(
            pool, std::make_shared<arrow::StringBuilder>(pool));
        arrow::StringBuilder &string_builder = *(
            static_cast<arrow::StringBuilder *>(list_builder.value_builder()));
        for (int64_t i = 0; i < num_lists; i++) {
            bool is_null = !GetBit((uint8_t *)array->null_bitmask, i);
            if (is_null) {
                list_builder.AppendNull();
            } else {
                list_builder.Append();
                int64_t l_string = string_offsets[i];
                int64_t r_string = string_offsets[i + 1];
                for (int64_t j = l_string; j < r_string; j++) {
                    bool is_null =
                        !GetBit((uint8_t *)array->sub_null_bitmask, j);
                    if (is_null) {
                        string_builder.AppendNull();
                    } else {
                        int64_t l_char = char_offsets[j];
                        int64_t r_char = char_offsets[j + 1];
                        int64_t length = r_char - l_char;
                        string_builder.Append((uint8_t *)(chars + l_char),
                                              length);
                    }
                }
            }
        }
        std::shared_ptr<arrow::Array> result;
        list_builder.Finish(&result);
        *out = std::make_shared<arrow::ChunkedArray>(result);
    }
}

/**
 * Struct used during pq_write_partitioned to store the information for a
 * partition that this process is going to write: the file path of the parquet
 * file for this partition (e.g. sales_date=2020-01-01/part-00.parquet), and
 * the rows in the table that correspond to this partition.
 */
struct partition_write_info {
    std::string fpath;          // path and filename
    std::vector<int64_t> rows;  // rows in this partition
};

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
// Bodo change: pass a different schema for metadata (added schema_for_metadata
// parameter)
static arrow::Status OpenFileWriter(
    const ::arrow::Schema &schema, const ::arrow::Schema &schema_for_metadata,
    ::arrow::MemoryPool *pool, std::shared_ptr<::arrow::io::OutputStream> sink,
    std::shared_ptr<parquet::WriterProperties> properties,
    std::shared_ptr<parquet::ArrowWriterProperties> arrow_properties,
    std::unique_ptr<parquet::arrow::FileWriter> *writer) {
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

    auto schema_ptr = std::make_shared<::arrow::Schema>(schema);
    return parquet::arrow::FileWriter::Make(
        pool, std::move(base_writer), std::move(schema_ptr),
        std::move(arrow_properties), writer);
}

// This function is copied from Arrow.
// Bodo change: passing through schema_for_metadata to OpenFileWriter
static arrow::Status WriteTable(
    const ::arrow::Table &table, ::arrow::MemoryPool *pool,
    std::shared_ptr<::arrow::io::OutputStream> sink, int64_t chunk_size,
    std::shared_ptr<parquet::WriterProperties> properties,
    std::shared_ptr<parquet::ArrowWriterProperties> arrow_properties,
    std::shared_ptr<arrow::Schema> schema_for_metadata) {
    std::unique_ptr<parquet::arrow::FileWriter> writer;
    RETURN_NOT_OK(OpenFileWriter(*table.schema(), *schema_for_metadata, pool,
                                 std::move(sink), std::move(properties),
                                 std::move(arrow_properties), &writer));
    RETURN_NOT_OK(writer->WriteTable(table, chunk_size));
    return writer->Close();
}
// ----------------------------------------------------------------------------

/*
 * Write the Bodo table (the chunk in this process) to a parquet file.
 * @param _path_name path of output file or directory
 * @param table table to write to parquet file
 * @param col_names_arr array containing the table's column names (index not
 * included)
 * @param index array containing the table index
 * @param write_index true if we need to write index passed in 'index', false
 * otherwise
 * @param metadata string containing table metadata
 * @param is_parallel true if the table is part of a distributed table (in this
 *        case, this process writes a file named "part-000X.parquet" where X is
 *        my rank into the directory specified by 'path_name'
 * @param write_rangeindex_to_metadata : true if writing a RangeIndex to
 * metadata
 * @param ri_start,ri_stop,ri_step start,stop,step parameters of given
 * RangeIndex
 * @param idx_name name of the given index
 * @param row_group_size Row group size in number of rows
 */
void pq_write(const char *_path_name, const table_info *table,
              const array_info *col_names_arr, const array_info *index,
              bool write_index, const char *metadata, const char *compression,
              bool is_parallel, bool write_rangeindex_to_metadata,
              const int ri_start, const int ri_stop, const int ri_step,
              const char *idx_name, const char *bucket_region,
              int64_t row_group_size) {
    tracing::Event ev("pq_write", is_parallel);
    ev.add_attribute("g_path", _path_name);
    ev.add_attribute("g_write_index", write_index);
    ev.add_attribute("g_metadata", metadata);
    ev.add_attribute("g_compression", compression);
    ev.add_attribute("g_write_rangeindex_to_metadata",
                     write_rangeindex_to_metadata);
    // Write actual values of start, stop, step to the metadata which is a
    // string that contains %d
    int check;
    std::vector<char> new_metadata;
    if (write_rangeindex_to_metadata) {
        new_metadata.resize((strlen(metadata) + strlen(idx_name) + 50));
        check = sprintf(new_metadata.data(), metadata, idx_name, ri_start,
                        ri_stop, ri_step);
    } else {
        new_metadata.resize((strlen(metadata) + 1 + (strlen(idx_name) * 4)));
        check = sprintf(new_metadata.data(), metadata, idx_name, idx_name,
                        idx_name, idx_name);
    }
    if (size_t(check + 1) > new_metadata.size())
        throw std::runtime_error(
            "Fatal error: number of written char for metadata is greater "
            "than new_metadata size");

    if (row_group_size == -1) row_group_size = DEFAULT_ROW_GROUP_SIZE;
    if (row_group_size <= 0)
        throw std::runtime_error(
            "to_parquet(): row_group_size must be greater than 0");
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

    extract_fs_dir_path(_path_name, is_parallel, ".parquet", myrank, num_ranks,
                        &fs_option, &dirname, &fname, &orig_path, &path_name);

    // We need to create a directory when writing a distributed
    // table to a posix filesystem.
    if (fs_option == Bodo_Fs::posix && is_parallel) {
        create_dir_posix(myrank, dirname, path_name);
    }
    // Do not write a file if there are no rows to write.
    if (table->nrows() == 0) {
        return;
    }
    open_outstream(fs_option, is_parallel, myrank, "parquet", dirname, fname,
                   orig_path, &out_stream, bucket_region);

    // copy column names to a std::vector<string>
    std::vector<std::string> col_names;
    char *cur_str = col_names_arr->data1;
    offset_t *offsets = (offset_t *)col_names_arr->data2;
    for (int64_t i = 0; i < col_names_arr->length; i++) {
        size_t len = offsets[i + 1] - offsets[i];
        col_names.emplace_back(cur_str, len);
        cur_str += len;
    }

    auto pool = ::arrow::default_memory_pool();

    // convert Bodo table to Arrow: construct Arrow Schema and ChunkedArray
    // columns
    std::vector<std::shared_ptr<arrow::Field>> schema_vector;
    std::vector<std::shared_ptr<arrow::Field>> schema_for_metadata_vector;
    std::vector<std::shared_ptr<arrow::ChunkedArray>> columns(
        table->columns.size());
    for (size_t i = 0; i < table->columns.size(); i++) {
        auto col = table->columns[i];
        bodo_array_to_arrow(pool, col, col_names[i], schema_vector,
                            &columns[i]);
    }

    // dictionary-encoded column handling
    // See comments on top of GetSchemaMetadata for why we generate a different
    // schema for the Arrow metadata string
    if (schema_vector.size() != table->ncols())
        throw std::runtime_error(
            "to_parquet: number of fields in schema doesn't match number of "
            "columns in Bodo table");
    bool has_dictionary_columns = false;
    std::shared_ptr<arrow::Field> dict_str_field;
    for (auto i = 0; i < schema_vector.size(); i++) {
        auto field = schema_vector[i];
        if (table->columns[i]->arr_type == bodo_array_type::DICT) {
            if (!arrow::is_dictionary(field->type()->id()))
                throw std::runtime_error(
                    "to_parquet: Arrow type of dictionary-encoded column is "
                    "not dictionary");
            // For the custom metadata that Arrow puts in the parquet file, we
            // will indicate that this is a string column, so Arrow doesn't
            // identify it as dictionary when reading the written parquet file
            schema_for_metadata_vector.push_back(
                arrow::field(col_names[i], arrow::utf8()));
            has_dictionary_columns = true;
            dict_str_field = field;
        } else {
            schema_for_metadata_vector.push_back(field);
        }
    }

    if (write_index) {
        // if there is an index, construct ChunkedArray index column and add
        // metadata to the schema
        std::shared_ptr<arrow::ChunkedArray> chunked_arr;
        if (strcmp(idx_name, "null") != 0)
            bodo_array_to_arrow(pool, index, idx_name, schema_vector,
                                &chunked_arr);
        else
            bodo_array_to_arrow(pool, index, "__index_level_0__", schema_vector,
                                &chunked_arr);
        columns.push_back(chunked_arr);
    }

    std::shared_ptr<arrow::KeyValueMetadata> schema_metadata;
    if (new_metadata.size() > 0 && new_metadata[0] != 0)
        schema_metadata =
            ::arrow::key_value_metadata({{"pandas", new_metadata.data()}});

    // make Arrow Schema object
    std::shared_ptr<arrow::Schema> schema =
        std::make_shared<arrow::Schema>(schema_vector, schema_metadata);
    std::shared_ptr<arrow::Schema> schema_for_metadata =
        std::make_shared<arrow::Schema>(schema_for_metadata_vector,
                                        schema_metadata);

    // make Arrow table from Schema and ChunkedArray columns
    // Since we reuse Bodo buffers, the row group size of the Arrow table
    // has to be the same as the local number of rows of the Bodo table
    std::shared_ptr<arrow::Table> arrow_table =
        arrow::Table::Make(schema, columns, /*row_group_size=*/table->nrows());

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
            ->build();

    if (has_dictionary_columns) {
        // Make sure that Arrow stores dictionary-encoded column as Parquet
        // type BYTE_ARRAY / String, because in the Arrow schema that is
        // inserted in the parquet metadata we are saying that it's a string
        // column
        std::shared_ptr<parquet::schema::Node> node;
        parquet::arrow::FieldToNode(dict_str_field, *writer_properties,
                                    *arrow_writer_properties, &node);
        if (node->is_primitive()) {
            auto primitive_node =
                std::dynamic_pointer_cast<parquet::schema::PrimitiveNode>(node);
            if ((primitive_node->physical_type() !=
                 parquet::Type::BYTE_ARRAY) ||
                (!primitive_node->logical_type()->Equals(
                    *parquet::LogicalType::String())))
                throw std::runtime_error(
                    "Arrow is not storing dictionary array as parquet string");
        } else {
            throw std::runtime_error(
                "Arrow is not storing dictionary array as parquet string");
        }
    }

    // open file and write table
    arrow::Status status = /*parquet::arrow::*/ WriteTable(
        *arrow_table, pool, out_stream, row_group_size, writer_properties,
        // store_schema() = true is needed to write the schema metadata to
        // file
        // .coerce_timestamps(::arrow::TimeUnit::MICRO)->allow_truncated_timestamps()
        // not needed when moving to parquet 2.0
        arrow_writer_properties, schema_for_metadata);
    CHECK_ARROW(status, "parquet::arrow::WriteTable");
}

void pq_write_py_entry(const char *_path_name, const table_info *table,
                       const array_info *col_names_arr, const array_info *index,
                       bool write_index, const char *metadata,
                       const char *compression, bool is_parallel,
                       bool write_rangeindex_to_metadata, const int ri_start,
                       const int ri_stop, const int ri_step,
                       const char *idx_name, const char *bucket_region,
                       int64_t row_group_size) {
    try {
        pq_write(_path_name, table, col_names_arr, index, write_index, metadata,
                 compression, is_parallel, write_rangeindex_to_metadata,
                 ri_start, ri_stop, ri_step, idx_name, bucket_region,
                 row_group_size);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

/*
 * Write the Bodo table (this process' chunk) to a partitioned directory of
 * parquet files. This process will write N files if it has N partitions in its
 * local data.
 * @param _path_name path of base output directory for partitioned dataset
 * @param table table to write to parquet files
 * @param col_names_arr array containing the table's column names (index not
 * included)
 * @param col_names_arr_no_partitions array containing the table's column names
 * (index and partition columns not included)
 * @param categories_table table containing categories arrays for each partition
 * column that is a categorical array. Categories could be (for example) strings
 * like "2020-01-01", "2020-01-02", etc.
 * @param partition_cols_idx indices of partition columns in table
 * @param num_partition_cols number of partition columns
 * @param is_parallel true if the table is part of a distributed table
 * @param row_group_size Row group size in number of rows
 */
void pq_write_partitioned(const char *_path_name, table_info *table,
                          const array_info *col_names_arr,
                          const array_info *col_names_arr_no_partitions,
                          table_info *categories_table, int *partition_cols_idx,
                          int num_partition_cols, const char *compression,
                          bool is_parallel, const char *bucket_region,
                          int64_t row_group_size) {
    // TODOs
    // - Do is parallel here?
    // - sequential (only rank 0 writes, or all write with same name -which?-)
    // - create directories
    //     - what if directories already have files?
    // - write index
    // - write metadata?
    // - convert values to strings for other dtypes like datetime, decimal, etc
    // (see array_info::val_to_str)

    try {
        if (!is_parallel)
            throw std::runtime_error(
                "to_parquet partitioned not implemented in sequential mode");

        // new_table will have partition columns at the beginning and the rest
        // after (to use multi_col_key for hashing which assumes that keys are
        // at the beginning), and we will then drop the partition columns from
        // it for writing
        table_info *new_table = new table_info();
        std::vector<bool> is_part_col(table->ncols(), false);
        std::vector<array_info *> partition_cols;
        std::vector<std::string> part_col_names;
        offset_t *offsets = (offset_t *)col_names_arr->data2;
        for (int i = 0; i < num_partition_cols; i++) {
            int j = partition_cols_idx[i];
            is_part_col[j] = true;
            partition_cols.push_back(table->columns[j]);
            new_table->columns.push_back(table->columns[j]);
            char *cur_str = col_names_arr->data1 + offsets[j];
            size_t len = offsets[j + 1] - offsets[j];
            part_col_names.emplace_back(cur_str, len);
        }
        for (int64_t i = 0; i < table->ncols(); i++) {
            if (!is_part_col[i])
                new_table->columns.push_back(table->columns[i]);
        }

        const uint32_t seed = SEED_HASH_PARTITION;
        uint32_t *hashes = hash_keys(partition_cols, seed, is_parallel);
        UNORD_MAP_CONTAINER<multi_col_key, partition_write_info,
                            multi_col_key_hash>
            key_to_partition;

        // TODO nullable partition cols?

        std::string fname =
            gen_pieces_file_name(dist_get_rank(), dist_get_size(), ".parquet");

        new_table->num_keys = num_partition_cols;
        for (int64_t i = 0; i < new_table->nrows(); i++) {
            multi_col_key key(hashes[i], new_table, i);
            partition_write_info &p = key_to_partition[key];
            if (p.rows.size() == 0) {
                // generate output file name
                p.fpath = std::string(_path_name);
                if (p.fpath.back() != '/') p.fpath += "/";
                int64_t cat_col_idx = 0;
                for (int j = 0; j < num_partition_cols; j++) {
                    auto part_col = partition_cols[j];
                    // convert partition col value to string
                    std::string value_str;
                    if (part_col->arr_type == bodo_array_type::CATEGORICAL) {
                        int64_t code = part_col->get_code_as_int64(i);
                        // TODO can code be -1 (NA) for partition columns?
                        value_str = categories_table->columns[cat_col_idx++]
                                        ->val_to_str(code);
                    } else {
                        value_str = part_col->val_to_str(i);
                    }
                    p.fpath += part_col_names[j] + "=" + value_str + "/";
                }
                p.fpath += fname;
            }
            p.rows.push_back(i);
        }
        delete[] hashes;

        // drop partition columns from new_table (they are not written to
        // parquet)
        new_table->columns.erase(
            new_table->columns.begin(),
            new_table->columns.begin() + num_partition_cols);

        for (auto it = key_to_partition.begin(); it != key_to_partition.end();
             it++) {
            const partition_write_info &p = it->second;
            // RetrieveTable steals the reference but we still need them
            for (auto a : new_table->columns) incref_array(a);
            table_info *part_table =
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
            pq_write(p.fpath.c_str(), part_table, col_names_arr_no_partitions,
                     nullptr, /*TODO*/ false, /*TODO*/ "", compression, false,
                     false, -1, -1, -1, /*TODO*/ "", bucket_region,
                     row_group_size);
            delete_table_decref_arrays(part_table);
        }
        delete new_table;

    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}
