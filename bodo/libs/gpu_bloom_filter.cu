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
        atomicOr(reinterpret_cast<unsigned long long*>(&bitset[word_idx]),
                 static_cast<unsigned long long>(mask));
    }
}

void print_uint64_buffer(rmm::device_buffer const& buf, std::size_t N) {
    // host buffer
    std::vector<uint64_t> host(N);

    // copy device → host
    cudaMemcpy(host.data(), buf.data(), N * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // print
    for (std::size_t i = 0; i < N; ++i) {
        std::cout << "word[" << i << "] = " << std::bitset<64>(host[i]) << "\n";
    }
}

// Build bloom filter from keys (table_view of key columns)
std::shared_ptr<CudfBloomFilter> build_bloom_filter_from_table(
    cudf::table_view const& keys,
    double false_positive_rate,
    rmm::cuda_stream_view stream) {

    std::shared_ptr<CudfBloomFilter> bf = std::make_shared<CudfBloomFilter>();
    bf->n_items = keys.num_rows();
    compute_bloom_params(bf->n_items, false_positive_rate, bf->m_bits, bf->k_hashes);

    // allocate bitset words
    std::size_t words = (bf->m_bits + 63) / 64;
    bf->bitset = rmm::device_buffer(words * sizeof(uint64_t), stream);
    // zero initialize
    CUDA_TRY(cudaMemsetAsync(bf->bitset.data(), 0, words * sizeof(uint64_t), stream.value()));

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
    std::cout << "build_bloom 0 " << bf->n_items << " " << bf->m_bits << " " << bf->k_hashes << " " << words << " " << n << std::endl;
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

    int block = 256;
    int grid = static_cast<int>((n + block - 1) / block);
    set_bits_kernel_doublehash<<<grid, block, 0, stream.value()>>>(
        dst_ptr, n,
        reinterpret_cast<uint64_t*>(bf->bitset.data()),
        words, bf->m_bits, bf->k_hashes);
    CUDA_TRY(cudaGetLastError());
    print_uint64_buffer(bf->bitset, words);

    return bf;
}

// Device kernel: double hashing using 128-bit (low, high)
__global__ void test_bits_kernel_doublehash(
    const uint64_t* low_high_buf, // length = 2*n: low[0..n-1], high[0..n-1]
    std::size_t n,
    const uint64_t * bitset,
    std::size_t words,
    std::size_t m_bits,
    std::size_t k_hashes,
    uint8_t* mask_out) {

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
        if ((word & mask) == 0) {
            maybe_in = false;
            break;
        }
    }
  
    mask_out[idx] = maybe_in ? uint8_t{1} : uint8_t{0};
}

#include <cudf/reduction.hpp>
#include <cudf/aggregation.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>

std::size_t count_true(cudf::column_view const& col, rmm::cuda_stream_view stream) {
    auto const agg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
    auto result = cudf::reduce(
        col,
        *agg,
        cudf::data_type(cudf::type_id::INT64, 0),
        //cudf::null_policy::EXCLUDE,
        stream
    );

    auto scalar = static_cast<cudf::numeric_scalar<int64_t> const*>(result.get());
    return scalar->value(stream);
}

void filter_table_with_bloom(
    cudf::table_view const& probe_table,
    std::vector<cudf::size_type> const& probe_key_indices,
    CudfBloomFilter const& bf,
    std::unique_ptr<cudf::column> &prev_mask,
    rmm::cuda_stream_view stream) {

    // 1) select key columns from probe_table
    cudf::table_view probe_keys = probe_table.select(probe_key_indices);
  
    // 2) compute 128-bit murmur3 hashes -> returns unique_ptr<cudf::table> with 2 uint64 columns
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
        // nothing to filter; return a copy of the input table (or an empty table as appropriate)
        return;
    }
  
    // 3) allocate device buffer to own both low and high arrays:
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
  
    // 4) allocate mask (uint8_t per row)
    rmm::device_buffer mask_buf(n * sizeof(uint8_t), stream);
    uint8_t* mask_ptr = static_cast<uint8_t*>(mask_buf.data());
    // initialize mask to zero (optional)
    CUDA_TRY(cudaMemsetAsync(mask_ptr, 0, n * sizeof(uint8_t), stream.value()));
  
    // 5) run test kernel (reads hashes_buf, writes mask_buf, may also set bits in bf.bitset)
    std::size_t words = (bf.m_bits + 63) / 64;
    int block = 256;
    int grid = static_cast<int>((n + block - 1) / block);
    test_bits_kernel_doublehash<<<grid, block, 0, stream.value()>>>(
        dst_ptr, n,
        reinterpret_cast<const uint64_t*>(bf.bitset.data()),
        words, bf.m_bits, bf.k_hashes,
        mask_ptr);
    CUDA_TRY(cudaGetLastError());
  
    // 6) convert mask to cudf::column (boolean)
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
    std::cout << "trues in bool_mask " << count_true(bool_mask->view(), stream) << " " << probe_table.num_rows() << std::endl;
    if (prev_mask) {
        // OR together the old and new masks.
        prev_mask = cudf::binary_operation(
                prev_mask->view(),
                bool_mask->view(),
                cudf::binary_operator::LOGICAL_OR,
                cudf::data_type{cudf::type_id::BOOL8},
                stream);

    std::cout << "trues in prev_mask " << count_true(prev_mask->view(), stream) << std::endl;
    } else {
        prev_mask = std::move(bool_mask);
    std::cout << "moving bool_mask to prev_mask" << std::endl;
    }
}
