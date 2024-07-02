#include "_bodo_common.h"
#include "_table_builder.h"

/**
 * @brief A wrapper around a sorted table_info that stores the min and max rows.
 * It is assumed that range is always pinned, and that table is usually
 * unpinned. This allows for fast sorting of chunks without unpinning.
 *
 */
struct TableAndRange {
    // The data table - assumed to be sorted. Can be pinned or unpinned.
    std::shared_ptr<table_info> table;
    // table with only 2 rows for the min and max of table - always pinned
    std::shared_ptr<table_info> range;
    // Offset of the first row in the range table. This is to represent tables
    // where we only want the suffix of the table, and the first row of range
    // (the min) is actully the `offset`th row of the table.
    int64_t offset;

    TableAndRange(std::shared_ptr<table_info> table, int64_t n_key_t,
                  int64_t offset = 0);

    /**
     * Update the offset into the table and adjust the range accordingly
     */
    void UpdateOffset(int64_t n_key_t, int64_t offset);
};

struct HeapComprator {
    int64_t n_key_t;
    // TODO(aneesh) can these be vectors instead?
    const std::vector<int64_t>& vect_ascending;
    const std::vector<int64_t>& na_position;
    const std::vector<int64_t>& dead_keys;

    // TODO(aneesh) we should either templetize KeyComparisonAsPython or move to
    // something like converting rows to bitstrings for faster comparision.
    bool operator()(const TableAndRange& a, const TableAndRange& b) {
        return !KeyComparisonAsPython(n_key_t, vect_ascending.data(),
                                      a.range->columns, 0, 0, b.range->columns,
                                      0, 0, na_position.data());
    }
};

struct SortedChunkedTableBuilder {
    const int64_t n_key_t;
    const std::vector<int64_t>& vect_ascending;
    const std::vector<int64_t>& na_position;
    const std::vector<int64_t>& dead_keys;

    const size_t chunk_size;

    HeapComprator comp;

    std::vector<TableAndRange> table_heap;

    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;
    std::unique_ptr<ChunkedTableBuilder> sorted_table_builder;

    SortedChunkedTableBuilder(int64_t n_key_t,
                              const std::vector<int64_t>& vect_ascending,
                              const std::vector<int64_t>& na_position,
                              const std::vector<int64_t>& dead_keys,
                              size_t chunk_size = 4096)
        : n_key_t(n_key_t),
          vect_ascending(vect_ascending),
          na_position(na_position),
          dead_keys(dead_keys),
          chunk_size(chunk_size),
          comp({n_key_t, vect_ascending, na_position, dead_keys}) {}

    void AppendChunk(std::shared_ptr<table_info> chunk);

    std::vector<TableAndRange> Finalize();
};
