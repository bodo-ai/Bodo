#include "gpu_utils.h"

#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <cassert>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/hashing.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/bit.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/device_uvector.hpp>

template <bool HasNulls>
__global__ void set_bools_false_kernel(
    bool* __restrict__ target_bools,
    int32_t const* __restrict__ indices,
    cudf::bitmask_type const* __restrict__ indices_mask,
    cudf::size_type num_indices) {

    cudf::size_type tid = blockIdx.x * blockDim.x + threadIdx.x;
    cudf::size_type stride = gridDim.x * blockDim.x;

    for (cudf::size_type i = tid; i < num_indices; i += stride) {
        bool is_valid = true;
        if constexpr (HasNulls) {
            is_valid = cudf::bit_is_set(indices_mask, i);
        }

        if (is_valid) {
            // Fetch index using __ldg to hit the read-only data cache
            int32_t target_idx = __ldg(&indices[i]);
            target_bools[target_idx] = false;
        }
    }
}

void cudf_set_bools_false_from_indices(
    cudf::mutable_column_view target_bools, 
    cudf::column_view const indices,
    rmm::cuda_stream_view stream) {
    if (indices.is_empty()) {
        return;
    }

    bool* d_target = target_bools.head<bool>();
    int32_t const* d_indices = indices.head<int32_t>();
    
    cudf::bitmask_type const* d_mask = indices.null_mask();

    int constexpr block_size = 256;
    int grid_size = std::min(
        (indices.size() + block_size - 1) / block_size,
        cudf::size_type{65536} 
    );

    if (indices.has_nulls()) {
        set_bools_false_kernel<true><<<grid_size, block_size, 0, stream.value()>>>(
            d_target,
            d_indices,
            d_mask,
            indices.size()
        );
    } else {
        set_bools_false_kernel<false><<<grid_size, block_size, 0, stream.value()>>>(
            d_target,
            d_indices,
            nullptr,
            indices.size()
        );
    }
}

