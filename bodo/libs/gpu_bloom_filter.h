#pragma once
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/column/column.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/hashing.hpp> // for cudf::hashing::hash
#include <cudf/transform.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>

struct BloomFilter {
  rmm::device_buffer bitset; // holds (m_bits + 63)/64 words
  std::size_t m_bits;
  int k_hashes;
  std::size_t n_items;
};

BloomFilter build_bloom_filter_from_table(
    cudf::table_view const& keys,
    std::size_t expected_items,
    double false_positive_rate,
    rmm::cuda_stream_view stream);

std::unique_ptr<cudf::table> filter_table_with_bloom(
    cudf::table_view const& probe_table,
    std::vector<cudf::size_type> const& probe_key_indices,
    BloomFilter const& bf,
    rmm::cuda_stream_view stream);
