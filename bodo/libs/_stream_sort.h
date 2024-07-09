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

enum class StreamSortPhase {
    INIT,
    PRE_BUILD,
    BUILD,
    GLOBAL_SORT,
    PRODUCE_OUTPUT,
    INVALID
};

struct StreamSortState {
    int64_t n_key_t;
    std::vector<int64_t> vect_ascending;
    std::vector<int64_t> na_position;
    std::vector<int64_t> dead_keys;
    bool parallel = true;

    // builder will create a sorted list from the chunks passed to consume batch
    SortedChunkedTableBuilder builder;
    // output from the sorted builder
    std::vector<TableAndRange> local_chunks;

    // Index of chunk to yield
    size_t output_idx = 0;
    // List of chunks ready to yield
    std::vector<std::shared_ptr<table_info>> output_chunks;

    // We need to track the phase of the algorithm to know which objects we need
    // to initialize
    StreamSortPhase phase = StreamSortPhase::INIT;

    // This will be initialized the first time consume_batch is called
    std::shared_ptr<table_info> dummy_output_chunk;

    StreamSortState(int64_t n_key_t_, std::vector<int64_t>&& vect_ascending_,
                    std::vector<int64_t>&& na_position_,
                    size_t chunk_size = 4096)
        : n_key_t(n_key_t_),
          vect_ascending(vect_ascending_),
          na_position(na_position_),
          // Note that builder only stores references to the vectors owned by
          // this object, so we must refer to the instances on this class, not
          // the arguments.
          builder(n_key_t, vect_ascending, na_position, dead_keys, chunk_size) {
    }

    /**
     * @brief Consume an unsorted table and use it for global sorting
     *
     * @param table unsorted input table
     * @param parallel whether or not multiple ranks are participating in the
     * sort
     * @param is_last if this table is guaranteed to be the last input this
     * state will see.
     * @return boolean indicating if the consume phase is complete
     */
    bool consume_batch(std::shared_ptr<table_info> table, bool parallel,
                       bool is_last);

    /**
     * Get bounds for parallel sorting based on the chunks consumed so far.
     * These bounds determine which elements are assigned to which ranks.
     */
    std::shared_ptr<table_info> get_parallel_sort_bounds();

    void global_sort();

    std::pair<std::shared_ptr<table_info>, bool> get_output();

    // Helper methods
    std::vector<std::vector<std::shared_ptr<table_info>>>
    partition_chunks_by_rank(int n_pes, std::shared_ptr<table_info> bounds);
};
