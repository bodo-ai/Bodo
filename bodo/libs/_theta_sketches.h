#include <arrow/array.h>
#include <arrow/table.h>
#include <theta_sketch.hpp>
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
        case arrow::Type::DECIMAL:
        case arrow::Type::FLOAT:
        case arrow::Type::DOUBLE:
            return true;
        default:
            return false;
    }
}

/**
 * @brief Indicate which arrow types enable theta sketches by default.
 *
 * @param type The Pyarrow type
 * @return Do we want theta sketches enabled by default?
 */
inline bool is_default_theta_sketch_type(
    std::shared_ptr<arrow::DataType> type) {
    if (!type_supports_theta_sketch(type)) {
        return false;
    }
    switch (type->id()) {
        case arrow::Type::INT32:
        case arrow::Type::INT64:
        case arrow::Type::DATE32:
        case arrow::Type::TIME64:
        case arrow::Type::TIMESTAMP:
        case arrow::Type::LARGE_STRING:
        case arrow::Type::LARGE_BINARY:
        case arrow::Type::DICTIONARY:
        case arrow::Type::DECIMAL:
            return true;
        case arrow::Type::FLOAT:
        case arrow::Type::DOUBLE:
        default:
            return false;
    }
}

/**
 * @brief Class that represents a collection of theta sketches
 * that are compact and can be merged but not updated. This is
 * basically a wrapper around a vector of
 * std::optional<datasketches::compact_theta_sketch> with the
 * associated functions.
 *
 * TODO: Update several APIs to return std::unique_ptr instead of
 * std::shared_ptr.
 *
 */
class CompactSketchCollection {
   public:
    /**
     * @brief Construct a new Compact Sketch Collection object. This moves the
     * compact sketches into the internal vector.
     *
     * @param sketches The sketches to be stored in the collection.
     */
    CompactSketchCollection(
        std::vector<std::optional<datasketches::compact_theta_sketch>>&&
            sketches)
        : sketches(std::move(sketches)) {};
    // Destructor.
    ~CompactSketchCollection() {};

    /**
     * @brief Gather the CompactSketchCollection onto rank 0 and combine them
     *        into one collection that has the combined NDV info from all ranks.
     *
     * @return the combined theta sketch collections in their immutable form
     * on rank zero (on other ranks returns nullptr). This returns a new object.
     */
    std::shared_ptr<CompactSketchCollection> merge_parallel_sketches();

    /**
     * @brief Take multiple compact theta sketches and combine them
     *        into one collection by merging all of the sketches across the
     * multiple collections from a single column into one sketch.
     * @param[in] sketch_collections: a vector of sketch representations
     *            that are to be combined into a single sketch representation.
     *            It is assumed that all representations have the same
     * present/absent sketches.
     *
     * @return The combined result.
     */
    static std::shared_ptr<CompactSketchCollection> merge_sketches(
        std::vector<std::shared_ptr<CompactSketchCollection>>
            sketch_collections);

    /**
     * @brief Serializes this collection of theta sketches into a vector
     * of strings, maintaining nulls for any missing sketches.
     *
     * @return the serialized sketches as a vector of optional strings.
     */
    std::vector<std::optional<std::string>> serialize_sketches();

    /**
     * @brief converts a collection of optional strings into a
     * CompactSketchCollection.
     * @param[in] strings: a vector of optional strings representing
     *            serialized theta sketches, or an absent theta sketch.
     *
     * @return The CompactSketchCollection.
     */
    static std::shared_ptr<CompactSketchCollection> deserialize_sketches(
        std::vector<std::optional<std::string>> strings);

    /**
     * @brief Computes the number of distinct values estimates for each theta
     * sketch, returning -1 for any columns that do not have a sketch. This is
     * primarily used for testing.
     *
     * @return std::unique_ptr<array_info> A nullable float array of NDV
     * estimates for each column.
     */
    std::unique_ptr<array_info> compute_ndv();

    /**
     * @brief Determine if a given column index has a sketch. This is used
     * for both testing and writing the puffin file.
     *
     * @param idx The index of the sketch to retrieve.
     * @return Does the sketch at the given index exist?
     */
    bool column_has_sketch(size_t idx) const {
        return sketches[idx].has_value();
    }

    /**
     * @brief Get the value of the sketch at the given index. This is used
     * for both testing and writing the puffin file.
     *
     * @param idx The index of the sketch to retrieve.
     * @return The sketch at the given index.
     */
    const datasketches::compact_theta_sketch& get_value(size_t idx) const {
        if (!sketches[idx].has_value()) {
            throw std::runtime_error("Sketch does not exist");
        }
        return sketches[idx].value();
    }

    /**
     * @brief Get the max number of sketches possible in the collection.
     * This is used for both testing and writing the puffin file.
     *
     * @return The max number of sketches possible in the collection.
     */
    size_t max_num_sketches() const { return sketches.size(); }

   private:
    std::vector<std::optional<datasketches::compact_theta_sketch>> sketches;
};

/**
 * @brief Class that represents a collection of theta sketches
 * with update capabilities. This is basically a wrapper around
 * a vector of std::optional<datasketches::update_theta_sketch>
 * with the associated functions.
 *
 * TODO: Update several APIs to return std::unique_ptr instead of
 * std::shared_ptr.
 *
 */
class UpdateSketchCollection {
   public:
    /**
     * @brief Construct a new Update Sketch Collection object.
     * This initializes the internal vector of sketches to have
     * ndv_cols.size() elements, with a new sketch being created
     * where ndv_cols[i] is true and a nullptr where ndv_cols[i]
     * is false.
     *
     * @param ndv_cols A vector of booleans, one per column in the
     * table, where true indicates that we want to generate a theta
     * sketch for that column.
     *
     */
    UpdateSketchCollection(const std::vector<bool>& ndv_cols)
        : sketches(ndv_cols.size()) {
        for (size_t col_idx = 0; col_idx < ndv_cols.size(); col_idx++) {
            if (ndv_cols[col_idx]) {
                this->sketches[col_idx] =
                    datasketches::update_theta_sketch::builder().build();
            } else {
                this->sketches[col_idx] = std::nullopt;
            }
        }
    }
    // Destructor.
    ~UpdateSketchCollection() {};

    /**
     * @brief Modifies a column's sketch in place when we have received a new
     *        batch of data.
     * @param[in] col: the most recently received batch of data that we wish
     *            to insert into the NDV information for one of the columns.
     * @param[in] col_idx: which column is being inserted.
     * @param[in] dict_hits: vector indicating which dictionary indices
     *            are used (nullopt if not a dictionary column)
     */
    void update_sketch(
        const std::shared_ptr<arrow::ChunkedArray>& col, size_t col_idx,
        std::optional<std::shared_ptr<arrow::Buffer>> dict_hits = std::nullopt);

    /**
     * @brief Finalize the update sketches into their compact form by calling
     * compact on each entry.
     *
     * @return A new CompactSketchCollection object that compacts the sketches
     * and maintains nulls for any nullptr.
     */
    std::shared_ptr<CompactSketchCollection> compact_sketches();

    /**
     * @brief Determine if a given column index has a sketch. This is used
     * primarily for testing.
     *
     * @param idx The index of the sketch to retrieve.
     * @return Does the sketch at the given index exist?
     */
    bool column_has_sketch(size_t idx) const {
        return sketches[idx].has_value();
    }

    /**
     * @brief Get the value of the sketch at the given index. This is used
     * primarily for testing.
     *
     * @param idx The index of the sketch to retrieve.
     * @return The sketch at the given index.
     */
    const datasketches::update_theta_sketch& get_value(size_t idx) const {
        if (!sketches[idx].has_value()) {
            throw std::runtime_error("Sketch does not exist");
        }
        return sketches[idx].value();
    }

   private:
    std::vector<std::optional<datasketches::update_theta_sketch>> sketches;
};
