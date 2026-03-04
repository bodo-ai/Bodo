#pragma once
#include <cuda/memory_resource>
#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/hashing.hpp>  // for cudf::hashing::hash
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

struct CudfBloomFilter {
    rmm::device_buffer bitset;  // holds (m_bits + 63)/64 words
    rmm::device_buffer hash_buffer;
    std::size_t m_bits{0};
    int k_hashes{0};
    std::size_t n_items{0};
};

std::shared_ptr<CudfBloomFilter> build_bloom_filter_from_table(
    cudf::table_view const& keys, double false_positive_rate,
    rmm::cuda_stream_view stream);

void filter_table_with_bloom(
    cudf::table_view const& probe_table,
    std::vector<cudf::size_type> const& probe_key_indices,
    CudfBloomFilter const& bf, std::unique_ptr<cudf::column>& prev_mask,
    rmm::cuda_stream_view stream);
