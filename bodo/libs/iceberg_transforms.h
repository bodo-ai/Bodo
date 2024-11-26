// Copyright (C) 2022 Bodo Inc. All rights reserved.

#pragma once

#include <Python.h>

#include "_bodo_common.h"

// Iceberg Transforms
// See more details here: https://iceberg.apache.org/spec/#partition-transforms
// Used for writing datasets with partition-specs and sort-orders.

/**
 * @brief Compute the Iceberg bucket transform (i.e. hash mod N) on the input
 * array and return a transformed array. We return a nullable uint32 array.
 *
 * Supported datatypes: INT32, INT64, DATE, DATETIME, STRING (incl DICT).
 * Some others like UINT32, etc. might work, but correctness is not guaranteed.
 *
 * See bucket transform details:
 * https://iceberg.apache.org/spec/#bucket-transform-details
 *
 * See hash requirements here:
 * https://iceberg.apache.org/spec/#appendix-b-32-bit-hash-requirements
 *
 * @param in_arr Array to transform
 * @param N number of buckets, i.e. value to mod the hash value to compute
 * bucket value
 * @param is_parallel Whether the operation is being performed in parallel. Used
 * for tracing.
 * @return std::shared_ptr<array_info> Transformed array.
 */
std::shared_ptr<array_info> array_transform_bucket_N(
    std::shared_ptr<array_info> in_arr, int64_t N, bool is_parallel);

/**
 * @brief Compute the Iceberg truncate transform on the input array and return a
 * transformed array (of the same type).
 *
 * Supported datatypes: INT32, INT64, STRING (incl DICT).
 * DECIMAL is not supported at this time.
 *
 * See more details here:
 * https://iceberg.apache.org/spec/#truncate-transform-details
 *
 * @param in_arr Array to transform
 * @param width Truncate width
 * @param is_parallel Whether the operation is being performed in parallel. Used
 * for tracing.
 * @return std::shared_ptr<array_info> Transformed array.
 */
std::shared_ptr<array_info> array_transform_truncate_W(
    std::shared_ptr<array_info> in_arr, int64_t width, bool is_parallel);

/**
 * @brief Compute the Iceberg void transform on the input array and return a
 * transformed array. In this case it simply return a nullable int32 array
 * with all null bits set to false (i.e. all nulls). All datatypes are
 * supported.
 *
 * @param in_arr Array to transform
 * @param is_parallel Whether the operation is being performed in parallel. Used
 * for tracing.
 * @return std::shared_ptr<array_info> Transformed array.
 */
std::shared_ptr<array_info> array_transform_void(
    std::shared_ptr<array_info> in_arr, bool is_parallel);

/**
 * @brief Compute the Iceberg year transform on the input array and return a
 * transformed array. Returns a nullable int32 array representing number of
 * years since 1970. Input can be DATETIME or DATE arrays.
 *
 * @param in_arr Array to transform
 * @param is_parallel Whether the operation is being performed in parallel. Used
 * for tracing.
 * @return std::shared_ptr<array_info> Transformed array.
 */
std::shared_ptr<array_info> array_transform_year(
    std::shared_ptr<array_info> in_arr, bool is_parallel);

/**
 * @brief Compute the Iceberg month transform on the input array and return a
 * transformed array. Returns a nullable int32 array representing the number of
 * months from 1970-01-01. Input can be DATETIME or DATE arrays.
 *
 * @param in_arr Array to transform
 * @param is_parallel Whether the operation is being performed in parallel. Used
 * for tracing.
 * @return std::shared_ptr<array_info> Transformed array.
 */
std::shared_ptr<array_info> array_transform_month(
    std::shared_ptr<array_info> in_arr, bool is_parallel);

/**
 * @brief Compute the Iceberg day transform on the input array and return a
 * transformed array. The result type is supposed to be "date", but it's not
 * really defined what "date" is. We return a nullable int64 array representing
 * the number of days from 1970-01-01. Input can be DATETIME or DATE arrays.
 *
 * @param in_arr Array to transform
 * @param is_parallel Whether the operation is being performed in parallel. Used
 * for tracing.
 * @return std::shared_ptr<array_info> Transformed array.
 */
std::shared_ptr<array_info> array_transform_day(
    std::shared_ptr<array_info> in_arr, bool is_parallel);

/**
 * @brief Compute the Iceberg hour transform on the input array and return a
 * transformed array. We return a nullable int32 array representing hours from
 * 1970-01-01 00:00:00. Input must be a DATETIME array.
 *
 * @param in_arr Array to transform
 * @param is_parallel Whether the operation is being performed in parallel. Used
 * for tracing.
 * @return std::shared_ptr<array_info> Transformed array.
 */
std::shared_ptr<array_info> array_transform_hour(
    std::shared_ptr<array_info> in_arr, bool is_parallel);

/**
 * @brief Compute the identity transform. In most cases, it returns the array as
 * is without any copies. In certain cases, we still need to perform some sort
 * of transformation (e.g. DATETIME needs to be converted to microseconds from
 * nanoseconds, and DATE needs to be converted to days since epoch from the bit
 * style representation).
 *
 * @param in_arr Array to transform
 * @param is_parallel Whether the operation is being performed in parallel. Used
 * for tracing.
 * @return std::shared_ptr<array_info> Transformed array.
 */
std::shared_ptr<array_info> iceberg_identity_transform(
    std::shared_ptr<array_info> in_arr, bool is_parallel);

/**
 * @brief Main entrypoint function to apply a transform on an input array.
 *
 * @param in_arr Array to transform.
 * @param transform_name Transform to perform. Should be one of "bucket",
 * "truncate", "void", "year", "month", "day", "hour".
 * @param arg Argument for the transform. In case of bucket, it's the number of
 * buckets (N), incase of truncate, it's the width (W). In all other cases, this
 * argument is irrelevant and the value is ignored.
 * @param is_parallel Whether the operation is being performed in parallel. Used
 * for tracing.
 * @return std::shared_ptr<array_info> Transformed array.
 */
std::shared_ptr<array_info> iceberg_transform(
    std::shared_ptr<array_info> in_arr, std::string transform_name, int64_t arg,
    bool is_parallel);

std::string transform_val_to_str(std::string transform_name,
                                 std::shared_ptr<array_info> in_arr,
                                 std::shared_ptr<array_info> transformed_arr,
                                 size_t idx);

/**
 * @brief Return python object representation of value in position `idx` of
 * `arr` array.
 *
 * @param arr Transformed array to get the value of
 * @param idx The index of the element we need to get the python object
 * representation of.
 * @return PyObject* Python Object representation of value in position `idx` of
 * `arr` array
 */
PyObject* iceberg_transformed_val_to_py(std::shared_ptr<array_info> arr,
                                        size_t idx);
