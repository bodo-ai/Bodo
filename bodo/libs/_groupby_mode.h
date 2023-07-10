// Copyright (C) 2023 Bodo Inc. All rights reserved.
#include "_array_hash.h"
#include "_array_utils.h"
#include "_bodo_common.h"
#include "_groupby_common.h"
#include "_groupby_hashing.h"
#include "_shuffle.h"

/**
 * Mode computation for array of any supported array type or dtype.
 * Switches on the array type and forwards to the correct templated
 * helper function.
 *
 * @param[in] arr: array that needs to be mode computed.
 * @param[in,out] out_arr: array to store mode computed values in.
 * @param[in] grp_info: grouping info for the array that is passed.
 */
void mode_computation(std::shared_ptr<array_info> arr,
                      std::shared_ptr<array_info> out_arr,
                      const grouping_info& grp_info);
