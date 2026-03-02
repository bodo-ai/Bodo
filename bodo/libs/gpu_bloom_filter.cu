#include "gpu_bloom_filter.h"
//#include <thrust/transform.h>
//#include <thrust/execution_policy.h>
//#include <thrust/device_ptr.h>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/hashing.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/utilities.hpp>
#include <cudf/unary.hpp>
//#include <cudf/strings/detail/convert/convert_datetime.hpp>
//#include <cudf/strings/detail/utilities.hpp>
#include <rmm/device_uvector.hpp>
#include <cuda_runtime.h>

#define CUDA_TRY(call)                                                     \
  do {                                                                     \
    cudaError_t const status = (call);                                     \
    if (status != cudaSuccess) {                                           \
      throw std::runtime_error(                                            \
        std::string{"CUDA error: "} + cudaGetErrorString(status));         \
    }                                                                      \
  } while (0)

// utility: compute m and k
void compute_bloom_params(std::size_t n, double p, std::size_t &m_out, int &k_out) {
  if (n == 0) { m_out = 1; k_out = 1; return; }
  double m = -static_cast<double>(n) * std::log(p) / (std::log(2.0) * std::log(2.0));
  std::size_t m_bits = static_cast<std::size_t>(std::ceil(m));
  int k = std::max(1, static_cast<int>(std::round((m / static_cast<double>(n)) * std::log(2.0))));
  m_out = m_bits;
  k_out = k;
}

// splitmix64 for generating h2 from h1
__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  x = x ^ (x >> 31);
  return x;
}

#if 0
// Kernel to set bits for build hashes
__global__ void set_bits_kernel(const uint64_t* hashes,
                                std::size_t n,
                                uint64_t* bitset_words,
                                std::size_t words_count,
                                std::size_t m_bits,
                                int k) {
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  uint64_t h1 = hashes[idx];
  uint64_t h2 = splitmix64(h1);
  for (int i = 0; i < k; ++i) {
    uint64_t combined = h1 + static_cast<uint64_t>(i) * h2;
    std::size_t bitpos = static_cast<std::size_t>(combined % m_bits);
    std::size_t word = bitpos >> 6; // /64
    uint64_t mask = 1ULL << (bitpos & 63ULL);
    // atomic OR on 64-bit word
    atomicOr(reinterpret_cast<unsigned long long*>(&bitset_words[word]), static_cast<unsigned long long>(mask));
  }
}
#endif

// Kernel to test bits for probe hashes and produce boolean mask
__global__ void test_bits_kernel(const uint64_t* hashes,
                                 std::size_t n,
                                 const uint64_t* bitset_words,
                                 std::size_t words_count,
                                 std::size_t m_bits,
                                 int k,
                                 uint8_t* out_mask) {
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  uint64_t h1 = hashes[idx];
  uint64_t h2 = splitmix64(h1);
  bool present = true;
  for (int i = 0; i < k; ++i) {
    uint64_t combined = h1 + static_cast<uint64_t>(i) * h2;
    std::size_t bitpos = static_cast<std::size_t>(combined % m_bits);
    std::size_t word = bitpos >> 6;
    uint64_t mask = 1ULL << (bitpos & 63ULL);
    if ((bitset_words[word] & mask) == 0ULL) { present = false; break; }
  }
  out_mask[idx] = present ? 1 : 0;
}

// Device kernel: double hashing using 128-bit (low, high)
__global__ void set_bits_kernel_doublehash(
    const uint64_t* low_high_buf, // length = 2*n, low[0..n-1], high[0..n-1]
    std::size_t n,
    uint64_t* bitset,
    std::size_t words,
    std::size_t m_bits,
    std::size_t k_hashes) {

    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint64_t h1 = low_high_buf[idx];
    uint64_t h2 = low_high_buf[idx + n];

    // produce k_hashes using double hashing: h_i = h1 + i * h2 (mod 2^64)
    for (std::size_t j = 0; j < k_hashes; ++j) {
        uint64_t combined = h1 + j * h2;
        // map to bit index in [0, m_bits)
        uint64_t bit = combined % m_bits;
        uint64_t word_idx = bit >> 6;            // /64
        uint64_t bit_in_word = bit & 63;         // %64
        uint64_t mask = uint64_t(1) << bit_in_word;
        // atomic OR into bitset word
        atomicOr(&bitset[word_idx], mask);
    }
}

// Build bloom filter from keys (table_view of key columns)
BloomFilter build_bloom_filter_from_table(
    cudf::table_view const& keys,
    std::size_t expected_items,
    double false_positive_rate,
    rmm::cuda_stream_view stream) {

    BloomFilter bf;
    bf.n_items = expected_items;
    compute_bloom_params(expected_items, false_positive_rate, bf.m_bits, bf.k_hashes);

    // allocate bitset words
    std::size_t words = (bf.m_bits + 63) / 64;
    bf.bitset = rmm::device_buffer(words * sizeof(uint64_t), stream);
    // zero initialize
    CUDA_TRY(cudaMemsetAsync(bf.bitset.data(), 0, words * sizeof(uint64_t), stream.value()));

    // compute 64-bit hash for each row of keys
    // cudf::hashing::hash returns a column of uint64_t (one per row)
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
    bf.hash_buffer = rmm::device_buffer(2 * n * sizeof(uint64_t), stream);
    // Copy device->device: low -> buf[0..n-1], high -> buf[n..2n-1]
    const uint64_t* low_dev_ptr = low_col.data<uint64_t>();
    const uint64_t* high_dev_ptr = high_col.data<uint64_t>();
    uint64_t* dst_ptr = static_cast<uint64_t*>(bf.hash_buffer.data());

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

    int block = 256;
    int grid = static_cast<int>((n + block - 1) / block);
    set_bits_kernel_doublehash<<<grid, block, 0, stream.value()>>>(
        dst_ptr, n,
        reinterpret_cast<uint64_t*>(bf.bitset.data()),
        words, bf.m_bits, bf.k_hashes);
    CUDA_TRY(cudaGetLastError());

    return bf;
}

#if 0
// Filter probe table using bloom filter on specified key indices
std::unique_ptr<cudf::table> filter_table_with_bloom(
    cudf::table_view const& probe_table,
    std::vector<cudf::size_type> const& probe_key_indices,
    BloomFilter const& bf,
    rmm::cuda_stream_view stream) {

  // 1) select key columns from probe_table
  cudf::table_view probe_keys = probe_table.select(probe_key_indices);

  // 2) compute hashes for probe keys
  auto probe_hash_col = cudf::hashing::hash(probe_keys, cudf::hash_id::HASH_MURMUR3, 0, stream);
  auto probe_hash_view = probe_hash_col->view();
  const uint64_t* probe_hashes = reinterpret_cast<const uint64_t*>(probe_hash_view.data<uint64_t>());
  std::size_t n = probe_hash_view.size();

  // 3) allocate mask (uint8_t per row)
  rmm::device_buffer mask_buf(n * sizeof(uint8_t), stream);
  uint8_t* mask_ptr = static_cast<uint8_t*>(mask_buf.data());

  // 4) run test kernel
  std::size_t words = (bf.m_bits + 63) / 64;
  if (n > 0) {
    int block = 256;
    int grid = static_cast<int>((n + block - 1) / block);
    test_bits_kernel<<<grid, block, 0, stream.value()>>>(
        probe_hashes, n,
        reinterpret_cast<const uint64_t*>(bf.bitset.data()),
        words, bf.m_bits, bf.k_hashes,
        mask_ptr);
    CUDA_TRY(cudaGetLastError());
  }

  // 5) convert mask to cudf::column (boolean)
  auto mask_column = cudf::make_numeric_column(cudf::data_type{cudf::type_id::UINT8}, n, cudf::mask_state::UNALLOCATED, stream);
  // copy mask_buf into mask_column's data
  auto mask_view = mask_column->mutable_view();
  CUDA_TRY(cudaMemcpyAsync(mask_view.data<uint8_t>(), mask_ptr, n * sizeof(uint8_t), cudaMemcpyDeviceToDevice, stream.value()));

  // convert uint8 mask to bool column (0/1 -> false/true)
  auto bool_mask = cudf::cast(mask_view, cudf::data_type{cudf::type_id::BOOL8}, stream);

  // 6) apply boolean mask to probe_table
  auto result = cudf::apply_boolean_mask(probe_table, *bool_mask, stream);

  return result;
}

#endif
