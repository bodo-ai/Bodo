#include <arrow/array.h>
#include <arrow/table.h>
#include "/tmp/datasketches-prefix/include/DataSketches/theta_constants.hpp"
#include "/tmp/datasketches-prefix/include/DataSketches/theta_sketch.hpp"
#include "/tmp/datasketches-prefix/include/DataSketches/theta_union.hpp"
#include "_array_utils.h"
#include "_bodo_common.h"

/**
 * @brief Indicate which arrow types support theta sketches.
 *
 * @param type The Pyarrow type
 * @return Does the type support theta sketches?
 */
inline bool type_supports_theta_sketch(std::shared_ptr<arrow::DataType> type) {
    switch (type->id()) {
        case arrow::Type::INT32:
        case arrow::Type::INT64:
        case arrow::Type::DATE32:
        case arrow::Type::TIME64:
        case arrow::Type::TIMESTAMP:
        case arrow::Type::LARGE_STRING:
        case arrow::Type::LARGE_BINARY:
        case arrow::Type::DICTIONARY:
        case arrow::Type::FLOAT:
        case arrow::Type::DOUBLE:
            return true;
        default:
            return false;
    }
}

// The type representing a collection of theta sketches as an array where each
// entry could be a theta sketch or absent, indicating that a column does not
// use a sketch. This type allows updates to be done.
typedef std::optional<datasketches::update_theta_sketch>
    *theta_sketch_collection_t;

// Variant of theta_sketch_collection_t that is in its immutable form
typedef std::vector<std::optional<datasketches::compact_theta_sketch>>
    immutable_theta_sketch_collection_t;

/**
 * @brief initializes a collection of theta sketches for a group of columns.
 * @param[in] ndv_cols a vector of booleans, one per column in the table, where
 *            true indicates that we want to generate a theta sketch for that
 *            column and false indicates that we do not want to, and have an
 * empty option type.
 * @return a buffer of pointers to each generated theta sketch, with non-ndv
 *         column indices instead mapping to null.
 */
theta_sketch_collection_t init_theta_sketches(
    const std::vector<bool> &ndv_cols);

/**
 * @brief modifies a collection of theta sketches in-place as they receive a
 *        new batch of data for a single column.
 * @param[in] sketches: the collection of theta sketches, with an empty option
 *            instead for any columns that we do not want NDV info for.
 * @param[in] in_col: the most recently received batch of data that we wish
 *            to insert into the NDV information for one of the columns.
 * @param[in] col_idx: which column is being inserted.
 * @param[in] dict_hits: vector indicating which dictionary indices
 *            are used (nullopt if not a dictionary column)
 */
void update_theta_sketches(
    theta_sketch_collection_t sketches,
    const std::shared_ptr<arrow::ChunkedArray> &in_col, size_t col_idx,
    std::optional<std::shared_ptr<arrow::Buffer>> dict_hits = std::nullopt);

/**
 * @brief takes in a collection of theta sketches and returns the immutable
 * version.
 * @param[in] sketches the collection of theta sketches, with an empty option
 *            instead for any columns that we do not want NDV info for.
 * @param[in] n_sketches: how many theta sketches are in the collection.
 *
 * @return the theta sketch collection in its immutable form.
 */
immutable_theta_sketch_collection_t compact_theta_sketches(
    theta_sketch_collection_t sketches, size_t n_sketches);

/**
 * @brief gathers a collection of theta sketches onto rank 0 and combines them
 *        into one collection that has the combined NDV info from all ranks.
 * @param[in] sketches the collection of theta sketches, with an empty option
 *            instead for any columns that we do not want NDV info for. It is
 *            assumed that the nullptr columns are the same across all ranks.
 *
 * @return the combined theta sketch collections in their immutable form
 * on rank zero (on other ranks returns nullptr).
 */
immutable_theta_sketch_collection_t merge_parallel_theta_sketches(
    immutable_theta_sketch_collection_t sketches);

/**
 * @brief takes in multiple collections of theta sketches and combines them
 *        into one collection by merging all of the sketches across the multiple
 *        collections from a single column into one sketch.
 * @param[in] sketch_collections: a vector of collections of theta sketches
 *            that are to be combined into a single collection. It is assumed
 * that all the collection have the same length.
 *
 * @return the combined theta sketch collection in its immutable form.
 */
immutable_theta_sketch_collection_t merge_theta_sketches(
    std::vector<immutable_theta_sketch_collection_t> sketch_collections);

/**
 * @brief serializes a collection of theta sketches
 * @param[in] sketches: the collection of sketches that are to be
 *            serialized.
 *
 * @return the serialized sketches as a vector of optional strings.
 */
std::vector<std::optional<std::string>> serialize_theta_sketches(
    immutable_theta_sketch_collection_t sketches);

/**
 * @brief converts a collection of optional strings to a
 *        collection of theta sketches.
 * @param[in] strings: a vector of optional strings representing
 *            serialized theta sketches, or an absent theta sketch.
 *
 * @return the combined theta sketch collection in its immutable form.
 */
immutable_theta_sketch_collection_t deserialize_theta_sketches(
    std::vector<std::optional<std::string>> strings);

/**
 * @brief Computes the number of distinct values estimates for each theta
 * sketch, returning -1 for any columns that do not have a sketch. This is
 * primarily used for testing.
 *
 * @param merged_collection The final collection of theta sketches to compute
 * NDV for.
 * @param n_fields The number of fields in the table.
 * @return std::unique_ptr<array_info> An array of NDV estimates for each
 * column.
 */
std::unique_ptr<array_info> compute_ndv(
    immutable_theta_sketch_collection_t merged_collection, size_t n_fields);
