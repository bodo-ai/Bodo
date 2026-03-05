#include "gpu_bloom_filter.h"
#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/hashing.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/utilities.hpp>
#include <cudf/unary.hpp>
#include <rmm/device_uvector.hpp>
#include <cuda_runtime.h>
#include <bitset>

#define CUDA_TRY(call)                                                     \
  do {                                                                     \
    cudaError_t const status = (call);                                     \
    if (status != cudaSuccess) {                                           \
      throw std::runtime_error(                                            \
        std::string{"CUDA error: "} + cudaGetErrorString(status));         \
    }                                                                      \
  } while (0)

/*
 * @brief Compute number of words and hashes for given table size and false position percentage.
 *
 * @param n - number of entries in the table
 * @param p - the desired false positive percentage
 * @param m_out - the number of bids needed
 * @param k_out - the number of hashes to be used
 */
void compute_bloom_params(std::size_t n, double p, std::size_t &m_out, int &k_out) {
  if (n == 0) { m_out = 1; k_out = 1; return; }
  double m = -static_cast<double>(n) * std::log(p) / (std::log(2.0) * std::log(2.0));
  std::size_t m_bits = static_cast<std::size_t>(std::ceil(m));
  int k = std::max(1, static_cast<int>(std::round((m / static_cast<double>(n)) * std::log(2.0))));
  m_out = m_bits;
  k_out = k;
}

/*
 * @brief Device kernel to set bits in the bloom filter from the input keys in low_high_buf.
 *
 * @param low_high_buf - input key hashes. length = 2*n, low[0..n-1], high[0..n-1]
 * @param n - number of input keys
 * @param bitset - bloom filter to set bits in
 * @param m_bits - number of bits for the bloom filter
 * @param k_hashes - number of hashes to use
 */
__global__ void set_bits_kernel_doublehash(
    const uint64_t* low_high_buf,
    std::size_t n,
    uint64_t* bitset,
    std::size_t m_bits,
    std::size_t k_hashes) {

    // Which input key to set bits for.
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Get first and second hash value for that input key.
    uint64_t h1 = low_high_buf[idx];
    uint64_t h2 = low_high_buf[idx + n];

    // Produce k_hashes using double hashing: h_i = h1 + i * h2 (mod 2^64)
    for (std::size_t j = 0; j < k_hashes; ++j) {
        uint64_t combined = h1 + j * h2;
        // map to bit index in [0, m_bits)
        uint64_t bit = combined % m_bits;
        uint64_t word_idx = bit >> 6;            // /64
        uint64_t bit_in_word = bit & 63;         // %64
        uint64_t mask = uint64_t(1) << bit_in_word;
        // atomic OR into bitset word
        atomicOr(reinterpret_cast<unsigned long long*>(&bitset[word_idx]),
                 static_cast<unsigned long long>(mask));
    }
}

std::shared_ptr<CudfBloomFilter> build_bloom_filter_from_table(
    cudf::table_view const& keys,
    uint64_t total_size,
    double false_positive_rate,
    rmm::cuda_stream_view stream) {

    std::shared_ptr<CudfBloomFilter> bf = std::make_shared<CudfBloomFilter>();
    bf->n_items = total_size;
    // Compute number of bits and hashes the bloom filter will use.
    compute_bloom_params(bf->n_items, false_positive_rate, bf->m_bits, bf->k_hashes);

    // allocate bitset words
    std::size_t words = (bf->m_bits + 63) / 64;
    bf->bitset = rmm::device_buffer(words * sizeof(uint64_t), stream);
    // zero initialize
    CUDA_TRY(cudaMemsetAsync(bf->bitset.data(), 0, words * sizeof(uint64_t), stream.value()));

    // compute two 64-bit hashes for each row of keys
    auto hash_table = cudf::hashing::murmurhash3_x64_128(keys, 0, stream);
    auto hash_table_view = hash_table->view();
    if (hash_table_view.num_columns() != 2) {
        throw std::runtime_error("murmurhash3_x64_128 did not return 2 columns");
    }

    auto low_col = hash_table_view.column(0);
    auto high_col = hash_table_view.column(1);
    if (low_col.type().id() != cudf::type_id::UINT64 || high_col.type().id() != cudf::type_id::UINT64) {
        throw std::runtime_error("murmurhash3_x64_128 returned unexpected column types");
    }

    std::size_t n = low_col.size();
    if (n == 0) {
        return bf;
    }

    // Allocate a single device buffer to own both low and high arrays:
    // layout: [low0..low_{n-1}, high0..high_{n-1}] (length = 2*n uint64_t)
    bf->hash_buffer = rmm::device_buffer(2 * n * sizeof(uint64_t), stream);
    // Copy device->device: low -> buf[0..n-1], high -> buf[n..2n-1]
    const uint64_t* low_dev_ptr = low_col.data<uint64_t>();
    const uint64_t* high_dev_ptr = high_col.data<uint64_t>();
    uint64_t* dst_ptr = static_cast<uint64_t*>(bf->hash_buffer.data());

    CUDA_TRY(cudaMemcpyAsync(
              dst_ptr,
              low_dev_ptr,
              n * sizeof(uint64_t),
              cudaMemcpyDeviceToDevice,
              stream.value()));
    CUDA_TRY(cudaMemcpyAsync(
              dst_ptr + n,
              high_dev_ptr,
              n * sizeof(uint64_t),
              cudaMemcpyDeviceToDevice,
              stream.value()));

    constexpr int block = 256;
    int grid = static_cast<int>((n + block - 1) / block);
    set_bits_kernel_doublehash<<<grid, block, 0, stream.value()>>>(
        dst_ptr, n,
        reinterpret_cast<uint64_t*>(bf->bitset.data()),
        bf->m_bits, bf->k_hashes);
    CUDA_TRY(cudaGetLastError());

    return bf;
}

// Device kernel: double hashing using 128-bit (low, high)
/*
 * @brief Device kernel to test if entry is in bloom filter.
 *
 * @param low_high_buf - input key hashes. length = 2*n, low[0..n-1], high[0..n-1]
 * @param n - number of input keys
 * @param bitset - bloom filter to test bits in
 * @param m_bits - number of bits for the bloom filter
 * @param k_hashes - number of hashes to use
 * @param mask_out - the bitmask to indicate if a given index is in the bloom filter
 */
__global__ void test_bits_kernel_doublehash(
    const uint64_t* low_high_buf, // length = 2*n: low[0..n-1], high[0..n-1]
    std::size_t n,
    const uint64_t * bitset,
    std::size_t m_bits,
    std::size_t k_hashes,
    uint8_t* mask_out) {

    // Which input key to test bits for.
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
  
    uint64_t h1 = low_high_buf[idx];
    uint64_t h2 = low_high_buf[idx + n];
  
    bool maybe_in = true;
    for (std::size_t j = 0; j < k_hashes; ++j) {
        uint64_t combined = h1 + j * h2; // double hashing
        uint64_t bit = combined % m_bits;
        uint64_t word_idx = bit >> 6;
        uint64_t bit_in_word = bit & 63;
        uint64_t mask = uint64_t(1) << bit_in_word;
    
        uint64_t word = bitset[word_idx];
        // If any hash is not in bloom filter then we know the key isn't.
        if ((word & mask) == 0) {
            maybe_in = false;
            break;
        }
    }
  
    // Set the mask to say if the key is in the bloom filter.
    mask_out[idx] = maybe_in ? uint8_t{1} : uint8_t{0};
}

void filter_table_with_bloom(
    cudf::table_view const& probe_table,
    std::vector<cudf::size_type> const& probe_key_indices,
    CudfBloomFilter const& bf,
    std::unique_ptr<cudf::column> &prev_mask,
    rmm::cuda_stream_view stream) {

    // select key columns from probe_table
    cudf::table_view probe_keys = probe_table.select(probe_key_indices);
  
    // compute 128-bit murmur3 hashes -> returns unique_ptr<cudf::table> with 2 uint64 columns
    auto probe_hash_table = cudf::hashing::murmurhash3_x64_128(probe_keys, 0, stream);
    auto hash_table_view = probe_hash_table->view();
  
    if (hash_table_view.num_columns() != 2) {
        throw std::runtime_error("murmurhash3_x64_128 did not return 2 columns");
    }
    auto low_col  = hash_table_view.column(0);
    auto high_col = hash_table_view.column(1);
    if (low_col.type().id() != cudf::type_id::UINT64 ||
        high_col.type().id() != cudf::type_id::UINT64) {
        throw std::runtime_error("murmurhash3_x64_128 returned unexpected column types");
    }
  
    std::size_t n = static_cast<std::size_t>(low_col.size());
    if (n == 0) {
        return;
    }
  
    // allocate device buffer to own both low and high arrays:
    // layout: [low0..low_{n-1}, high0..high_{n-1}] (length = 2*n uint64_t)
    rmm::device_buffer hashes_buf(2 * n * sizeof(uint64_t), stream);
    uint64_t* dst_ptr = static_cast<uint64_t*>(hashes_buf.data());
  
    // copy device->device: low -> dst[0..n-1], high -> dst[n..2n-1]
    const uint64_t* low_dev_ptr  = low_col.data<uint64_t>();
    const uint64_t* high_dev_ptr = high_col.data<uint64_t>();
  
    CUDA_TRY(cudaMemcpyAsync(
        dst_ptr,
        low_dev_ptr,
        n * sizeof(uint64_t),
        cudaMemcpyDeviceToDevice,
        stream.value()));
  
    CUDA_TRY(cudaMemcpyAsync(
        dst_ptr + n,
        high_dev_ptr,
        n * sizeof(uint64_t),
        cudaMemcpyDeviceToDevice,
        stream.value()));
  
    rmm::device_buffer mask_buf(n * sizeof(uint8_t), stream);
    uint8_t* mask_ptr = static_cast<uint8_t*>(mask_buf.data());
    // initialize mask to zero (optional)
    CUDA_TRY(cudaMemsetAsync(mask_ptr, 0, n * sizeof(uint8_t), stream.value()));
  
    constexpr int block = 256;
    int grid = static_cast<int>((n + block - 1) / block);
    test_bits_kernel_doublehash<<<grid, block, 0, stream.value()>>>(
        dst_ptr, n,
        reinterpret_cast<const uint64_t*>(bf.bitset.data()),
        bf.m_bits, bf.k_hashes,
        mask_ptr);
    CUDA_TRY(cudaGetLastError());
  
    // convert mask to cudf::column (boolean)
    auto mask_column = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::UINT8}, n,
        cudf::mask_state::UNALLOCATED, stream);
    auto mask_mut_view = mask_column->mutable_view();
    // copy mask_buf into mask_column's data (device->device on same stream)
    CUDA_TRY(cudaMemcpyAsync(
        mask_mut_view.data<uint8_t>(),
        mask_ptr,
        n * sizeof(uint8_t),
        cudaMemcpyDeviceToDevice,
        stream.value()));
  
    // convert uint8 mask to bool column (0/1 -> false/true)
    std::unique_ptr<cudf::column> bool_mask = cudf::cast(mask_mut_view, cudf::data_type{cudf::type_id::BOOL8}, stream);
    if (prev_mask) {
        // OR together the old and new masks.
        prev_mask = cudf::binary_operation(
                prev_mask->view(),
                bool_mask->view(),
                cudf::binary_operator::LOGICAL_OR,
                cudf::data_type{cudf::type_id::BOOL8},
                stream);

    } else {
        prev_mask = std::move(bool_mask);
    }
}

__global__ void atomic_or_u64_kernel(uint64_t* dst, uint64_t const* src, size_t words) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < words) {
        atomicOr(reinterpret_cast<unsigned long long*>(&dst[idx]), static_cast<unsigned long long>(src[idx]));
    }
}

void mergeBloomBitset(rmm::device_buffer& dst, rmm::device_buffer const&src, cudaStream_t stream) {
    if (dst.size() != src.size()) {
        throw std::runtime_error("mergeBloomBitset buffers must be the same size.");
    }

    size_t bytes = dst.size();
    if (bytes == 0) return;

    void *dst_ptr = dst.data();
    const void *src_ptr = src.data();

    constexpr int block = 256;
    size_t words = bytes / 8;
    int grid = static_cast<int>((words + block - 1) / block);
    atomic_or_u64_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<uint64_t*>(dst_ptr), reinterpret_cast<uint64_t const*>(src_ptr), words);
    CUDA_TRY(cudaGetLastError());
}
