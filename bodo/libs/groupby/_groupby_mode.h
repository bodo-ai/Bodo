// Copyright (C) 2023 Bodo Inc. All rights reserved.
#pragma once

#include "../_bodo_common.h"
#include "_groupby_common.h"

/**
 * Mode computation for array of any supported array type or dtype.
 * Switches on the array type and forwards to the correct templated
 * helper function.
 *
 * @param[in] arr: array that needs to be mode computed.
 * @param[in,out] out_arr: array to store mode computed values in.
 * @param[in] grp_info: grouping info for the array that is passed.
 * @param pool Memory pool to use for allocations during the execution of this
 * function.
 * @param mm Memory manager associated with the pool.
 */
void mode_computation(
    std::shared_ptr<array_info> arr, std::shared_ptr<array_info> out_arr,
    const grouping_info& grp_info,
    bodo::IBufferPool* const pool = bodo::BufferPool::DefaultPtr(),
    std::shared_ptr<::arrow::MemoryManager> mm =
        bodo::default_buffer_memory_manager());
