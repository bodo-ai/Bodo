#pragma once
#include <cuda/memory_resource>
#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/hashing.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

struct CudfBloomFilter {
    rmm::device_buffer bitset;  // holds (m_bits + 63)/64 words
    std::size_t m_bits{0};
    int k_hashes{0};
    std::size_t n_items{0};
};

/*
 * @brief Build bloom filter from keys (table_view of key columns)
 *
 * @param keys - table of the key columns
 * @param false_positive_rate - the desired false positive rate
 * @param stream - the stream to place operations on
 */
std::shared_ptr<CudfBloomFilter> build_bloom_filter_from_table(
    cudf::table_view const& keys, uint64_t total_size,
    double false_positive_rate, rmm::cuda_stream_view stream);

/*
 * @brief Build bloom filter from size
 *
 * @param false_positive_rate - the desired false positive rate
 * @param stream - the stream to place operations on
 */
std::shared_ptr<CudfBloomFilter> build_empty_bloom_filter(
    uint64_t total_size, double false_positive_rate,
    rmm::cuda_stream_view stream);

/*
 * @brief Updates prev_mask to indicate which rows are in the bloom filter.
 *
 * @param probe_table - the table to see which rows are in the bloom filter
 * @param probe_key_indices - the column indices in probe_table that are checked
 * in the bloom filter
 * @param bf - the previously built bloom filter to use
 * @param prev_mask - mask that says which rows are in the bloom filter.
 *           If you pass an existing mask then the new mask and previous mask
 * are OR'ed together.
 * @param stream - the stream to place operations on
 */
void filter_table_with_bloom(
    cudf::table_view const& probe_table,
    std::vector<cudf::size_type> const& probe_key_indices,
    CudfBloomFilter const& bf, std::unique_ptr<cudf::column>& prev_mask,
    rmm::cuda_stream_view stream);

/*
 * @brief Merges the local bloom filter with the ones from other ranks to
 *        make a global bloom filter.
 *
 * @param dst - the bloom filter to merge into
 * @param src - the bloom filter to merge from
 * @param stream - the stream to place operations on
 */
void mergeBloomBitset(rmm::device_buffer& dst, rmm::device_buffer const& src,
                      cudaStream_t stream);
