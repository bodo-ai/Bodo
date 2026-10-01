// Differential tests for the int64-backed decimal fast path
// (decimal_int64_binary_op in bodo/libs/_decimal_ext.cpp) against the
// existing 128-bit implementations (Arrow compute kernels and the gandiva
// decimal utilities).

#include <arrow/array.h>
#include <arrow/array/builder_decimal.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/compute/api.h>
#include <arrow/scalar.h>
#include <arrow/type.h>
#include <arrow/util/decimal.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "../libs/_bodo_to_arrow.h"
#include "../libs/_decimal_ext.h"
#include "./test.hpp"

namespace {

// Build a Bodo decimal array_info from unscaled values (nullopt for null).
// The array is allocated with the final precision so the storage width
// (8-byte for p <= 18 when BODO_DECIMAL_INT64_STORAGE is enabled) matches,
// and values are written with the width-aware setter.
std::shared_ptr<array_info> make_bodo_decimal_array(
    int precision, int scale,
    const std::vector<std::optional<__int128>>& unscaled_values) {
    auto arr = alloc_nullable_array_no_nulls(
        (int64_t)unscaled_values.size(), Bodo_CTypes::DECIMAL, 0,
        bodo::BufferPool::DefaultPtr(), bodo::default_buffer_memory_manager(),
        "", precision, scale);
    arr->precision = precision;
    arr->scale = scale;
    uint8_t* data = (uint8_t*)arr->data1<bodo_array_type::NULLABLE_INT_BOOL>();
    const size_t w = bodo_array_item_size(*arr);
    for (size_t i = 0; i < unscaled_values.size(); i++) {
        const auto& v = unscaled_values[i];
        if (!v.has_value()) {
            arr->set_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(i, false);
        } else {
            decimal_set_value(data + w * i, precision, (__int128_t)*v);
        }
    }
    return arr;
}

// Read a 16 byte little-endian two's complement decimal value as __int128.
__int128 read_le_decimal128(const uint8_t* p) {
    int64_t lo;
    int64_t hi;
    memcpy(&lo, p, 8);
    memcpy(&hi, p + 8, 8);
    return ((__int128)hi << 64) | (uint64_t)lo;
}

__int128 int128_pow10(int k) {
    __int128 v = 1;
    for (int i = 0; i < k; i++) {
        v *= 10;
    }
    return v;
}

// Build a decimal128 array from unscaled values (nullopt for null).
std::shared_ptr<arrow::Array> make_decimal_array(
    int precision, int scale,
    const std::vector<std::optional<__int128>>& unscaled_values) {
    auto type = arrow::decimal128(precision, scale);
    arrow::Decimal128Builder builder(type);
    for (const auto& v : unscaled_values) {
        if (!v.has_value()) {
            bodo::tests::check(builder.AppendNull().ok(), "AppendNull failed");
        } else {
            arrow::Decimal128 dec((int64_t)(*v >> 64),
                                  (uint64_t)(__uint128_t)*v);
            bodo::tests::check(builder.Append(dec).ok(), "Append failed");
        }
    }
    auto res = builder.Finish();
    bodo::tests::check(res.ok(), "builder.Finish failed");
    return res.ValueOrDie();
}

std::shared_ptr<arrow::Scalar> make_decimal_scalar(
    int precision, int scale, std::optional<__int128> value) {
    auto type = arrow::decimal128(precision, scale);
    if (!value.has_value()) {
        return arrow::MakeNullScalar(type);
    }
    arrow::Decimal128 dec((int64_t)(*value >> 64),
                          (uint64_t)(__uint128_t)*value);
    return std::make_shared<arrow::Decimal128Scalar>(dec, type);
}

std::shared_ptr<arrow::Array> make_int64_array(
    const std::vector<std::optional<int64_t>>& values) {
    arrow::Int64Builder builder;
    for (const auto& v : values) {
        if (!v.has_value()) {
            bodo::tests::check(builder.AppendNull().ok(), "AppendNull failed");
        } else {
            bodo::tests::check(builder.Append(*v).ok(), "Append failed");
        }
    }
    auto res = builder.Finish();
    bodo::tests::check(res.ok(), "builder.Finish failed");
    return res.ValueOrDie();
}

// Compare two decimal128 arrays element-wise: same length, same null pattern
// and equal unscaled values.
void check_decimal_arrays_equal(const std::shared_ptr<arrow::Array>& expected,
                                const std::shared_ptr<arrow::Array>& actual,
                                const char* msg) {
    bodo::tests::check(expected->length() == actual->length(),
                       "array length mismatch");
    auto exp_dec = std::static_pointer_cast<arrow::Decimal128Array>(expected);
    auto act_dec = std::static_pointer_cast<arrow::Decimal128Array>(actual);
    for (int64_t i = 0; i < expected->length(); i++) {
        if (exp_dec->IsNull(i) != act_dec->IsNull(i)) {
            bodo::tests::check(false, msg);
            return;
        }
        if (!exp_dec->IsNull(i)) {
            __int128 e = read_le_decimal128(exp_dec->GetValue(i));
            __int128 a = read_le_decimal128(act_dec->GetValue(i));
            if (e != a) {
                std::string err = std::string(msg) + ": row " +
                                  std::to_string(i) + " expected " +
                                  exp_dec->FormatValue(i) + " got " +
                                  act_dec->FormatValue(i);
                bodo::tests::check(false, err.c_str());
                return;
            }
        }
    }
}

// Compare two boolean arrays element-wise including nulls.
void check_bool_arrays_equal(const std::shared_ptr<arrow::Array>& expected,
                             const std::shared_ptr<arrow::Array>& actual,
                             const char* msg) {
    bodo::tests::check(expected->length() == actual->length(),
                       "array length mismatch");
    for (int64_t i = 0; i < expected->length(); i++) {
        bodo::tests::check(expected->IsNull(i) == actual->IsNull(i), msg);
        if (!expected->IsNull(i)) {
            bodo::tests::check(
                std::static_pointer_cast<arrow::BooleanArray>(expected)->Value(
                    i) == std::static_pointer_cast<arrow::BooleanArray>(actual)
                              ->Value(i),
                msg);
        }
    }
}

// Generate unscaled decimal values within precision/scale bounds. Values for
// precision <= 18 always fit in int64; larger precisions produce values that
// do not fit so the 128-bit slow path is exercised as well.
std::vector<std::optional<__int128>> gen_values(std::mt19937_64& gen,
                                                int precision, int scale,
                                                size_t n,
                                                double null_probability) {
    std::uniform_real_distribution<double> null_dist(0.0, 1.0);
    int64_t leading_digits = precision - scale;
    __int128 bound = int128_pow10(leading_digits);
    // Generate the value in up to two 9-digit chunks to stay within int64
    // random number generation, then combine into __int128.
    std::uniform_int_distribution<int64_t> chunk_dist(0, 999999999);
    std::vector<std::optional<__int128>> out;
    out.reserve(n);
    for (size_t i = 0; i < n; i++) {
        if (null_dist(gen) < null_probability) {
            out.push_back(std::nullopt);
            continue;
        }
        __int128 v = 0;
        int64_t remaining = leading_digits;
        while (remaining > 0) {
            int64_t chunk_digits = std::min<int64_t>(remaining, 9);
            __int128 chunk = chunk_dist(gen) % int128_pow10(chunk_digits);
            v = v * int128_pow10(chunk_digits) + chunk;
            remaining -= chunk_digits;
        }
        // Random sign.
        if (null_dist(gen) < 0.5) {
            v = -v;
        }
        // Clamp to the precision bound (v < 10^leading_digits).
        if (v >= bound) {
            v = bound - 1;
        } else if (v <= -bound) {
            v = -(bound - 1);
        }
        out.push_back(v);
    }
    return out;
}

// Snowflake-style result precision/scale rules (bodo/pandas/_util.cpp
// getOpPrecisionScale).
std::pair<int, int> op_result_precision_scale(const std::string& op, int p1,
                                              int s1, int p2, int s2) {
    int l1 = p1 - s1;
    int l2 = p2 - s2;
    if (op == "add" || op == "subtract") {
        int scale = std::max(s1, s2);
        return {std::max(l1, l2) + 1 + scale, scale};
    } else if (op == "multiply") {
        int scale = std::min(s1 + s2, std::max(std::max(s1, s2), 12));
        return {l1 + l2 + scale, scale};
    }
    throw std::runtime_error("unsupported op");
}

}  // namespace

static bodo::tests::suite tests([] {
    // Differential test of add/subtract/multiply on decimal arrays (both
    // precision <= 18) against Arrow compute, which is the pre-existing
    // evaluation path for result precision <= 38.
    bodo::tests::test("test_decimal_int64_arith_vs_arrow_compute", [] {
        std::mt19937_64 gen(42);
        struct Case {
            int p1, s1, p2, s2;
        };
        std::vector<Case> cases = {
            {15, 2, 16, 2}, {9, 0, 9, 4},  {18, 0, 18, 0},
            {12, 5, 10, 7}, {10, 0, 5, 3}, {6, 6, 4, 2},
        };
        for (const auto& c : cases) {
            for (const std::string op : {"add", "subtract", "multiply"}) {
                auto [out_p, out_s] =
                    op_result_precision_scale(op, c.p1, c.s1, c.p2, c.s2);
                // Multiply with scale reduction is not covered by the exact
                // (non-rounding) fast path, skip those cases here.
                if (op == "multiply" && c.s1 + c.s2 != out_s) {
                    continue;
                }
                auto left = make_decimal_array(
                    c.p1, c.s1, gen_values(gen, c.p1, c.s1, 97, 0.2));
                auto right = make_decimal_array(
                    c.p2, c.s2, gen_values(gen, c.p2, c.s2, 97, 0.2));

                auto ref_res = arrow::compute::CallFunction(
                    op, {arrow::Datum(left), arrow::Datum(right)});
                bodo::tests::check(ref_res.ok(), "reference compute failed");

                auto fast_res =
                    decimal_int64_binary_op(left, c.p1, c.s1, right, c.p2, c.s2,
                                            out_p, out_s, op, false);
                bodo::tests::check(fast_res.ok(), "fast path failed");
                check_decimal_arrays_equal(ref_res.ValueOrDie().make_array(),
                                           fast_res.ValueOrDie().make_array(),
                                           "fast path mismatch vs Arrow");
            }
        }
    });

    // Array-scalar and scalar-array arithmetic against Arrow compute.
    bodo::tests::test("test_decimal_int64_arith_scalar_mixed", [] {
        std::mt19937_64 gen(1234);
        int p1 = 15, s1 = 2, p2 = 12, s2 = 0;
        auto [out_p, out_s] = op_result_precision_scale("add", p1, s1, p2, s2);
        auto left =
            make_decimal_array(p1, s1, gen_values(gen, p1, s1, 53, 0.2));
        auto right_arr =
            make_decimal_array(p2, s2, gen_values(gen, p2, s2, 1, 0.0));
        auto right_scalar = make_decimal_scalar(p2, s2, 123456789);

        for (bool scalar_on_left : {false, true}) {
            arrow::Datum d1 = scalar_on_left ? arrow::Datum(right_scalar)
                                             : arrow::Datum(left);
            arrow::Datum d2 = scalar_on_left ? arrow::Datum(left)
                                             : arrow::Datum(right_scalar);
            auto ref_res = arrow::compute::CallFunction("add", {d1, d2});
            bodo::tests::check(ref_res.ok(), "reference compute failed");
            auto fast_res = decimal_int64_binary_op(
                d1, scalar_on_left ? p2 : p1, scalar_on_left ? s2 : s1, d2,
                scalar_on_left ? p1 : p2, scalar_on_left ? s1 : s2, out_p,
                out_s, "add", false);
            bodo::tests::check(fast_res.ok(), "fast path failed");
            check_decimal_arrays_equal(ref_res.ValueOrDie().make_array(),
                                       fast_res.ValueOrDie().make_array(),
                                       "scalar fast path mismatch");
        }
    });

    // Comparisons on decimal arrays (same and mixed scales) against Arrow
    // compute.
    bodo::tests::test("test_decimal_int64_comparisons_vs_arrow_compute", [] {
        std::mt19937_64 gen(5678);
        struct Case {
            int p1, s1, p2, s2;
        };
        std::vector<Case> cases = {
            {15, 2, 16, 2},
            {9, 0, 9, 4},
            {18, 0, 18, 0},
            {10, 3, 12, 3},
        };
        for (const auto& c : cases) {
            auto left = make_decimal_array(
                c.p1, c.s1, gen_values(gen, c.p1, c.s1, 89, 0.2));
            auto right = make_decimal_array(
                c.p2, c.s2, gen_values(gen, c.p2, c.s2, 89, 0.2));
            for (const std::string op :
                 {"equal", "not_equal", "less", "greater", "less_equal",
                  "greater_equal"}) {
                auto ref_res = arrow::compute::CallFunction(
                    op, {arrow::Datum(left), arrow::Datum(right)});
                bodo::tests::check(ref_res.ok(), "reference comparison failed");
                auto fast_res = decimal_int64_binary_op(
                    left, c.p1, c.s1, right, c.p2, c.s2, 1, 0, op, false);
                bodo::tests::check(fast_res.ok(), "fast comparison failed");
                check_bool_arrays_equal(ref_res.ValueOrDie().make_array(),
                                        fast_res.ValueOrDie().make_array(),
                                        "comparison fast path mismatch");
            }
        }
    });

    // Decimal compared with integer arrays and scalars.
    bodo::tests::test("test_decimal_int64_cmp_int_operands", [] {
        std::mt19937_64 gen(999);
        int p1 = 15, s1 = 2;
        auto dec_arr =
            make_decimal_array(p1, s1, gen_values(gen, p1, s1, 71, 0.2));
        std::vector<std::optional<__int128>> int_vals_raw =
            gen_values(gen, 18, 0, 71, 0.2);
        std::vector<std::optional<int64_t>> int_vals;
        for (const auto& v : int_vals_raw) {
            // Values were generated within a 19-digit bound; clamp to int64.
            int_vals.push_back(v.has_value()
                                   ? (std::optional<int64_t>)(int64_t)*v
                                   : std::nullopt);
        }
        auto int_arr = make_int64_array(int_vals);
        arrow::Datum int_scalar = arrow::Datum((int64_t)12345);

        // Expected result computed exactly by scaling the integers to the
        // decimal scale.
        auto dec_type =
            std::static_pointer_cast<arrow::Decimal128Type>(dec_arr->type());
        int s = dec_type->scale();
        auto dec_vals =
            std::static_pointer_cast<arrow::Decimal128Array>(dec_arr);
        auto int64_vals = std::static_pointer_cast<arrow::Int64Array>(int_arr);
        for (bool scalar_int : {false, true}) {
            arrow::Datum right =
                scalar_int ? int_scalar : arrow::Datum(int_arr);
            std::vector<bool> expected_lt(dec_arr->length());
            std::vector<bool> expected_valid(dec_arr->length());
            for (int64_t i = 0; i < dec_arr->length(); i++) {
                bool right_null =
                    scalar_int ? !right.scalar()->is_valid : int_arr->IsNull(i);
                bool is_null = dec_arr->IsNull(i) || right_null;
                expected_valid[i] = !is_null;
                if (is_null) {
                    continue;
                }
                __int128 d = read_le_decimal128(dec_vals->GetValue(i));
                __int128 iv = scalar_int ? (__int128)12345 * int128_pow10(s)
                                         : (__int128)int64_vals->Value(i) *
                                               int128_pow10(s);
                expected_lt[i] = d < iv;
            }

            auto fast_res = decimal_int64_binary_op(dec_arr, p1, s1, right, 19,
                                                    0, 1, 0, "less", false);
            bodo::tests::check(fast_res.ok(), "fast comparison failed");
            auto res_arr = fast_res.ValueOrDie().make_array();
            for (int64_t i = 0; i < dec_arr->length(); i++) {
                bodo::tests::check(res_arr->IsNull(i) == !expected_valid[i],
                                   "null mismatch in decimal/int compare");
                if (expected_valid[i]) {
                    bodo::tests::check(
                        std::static_pointer_cast<arrow::BooleanArray>(res_arr)
                                ->Value(i) == expected_lt[i],
                        "decimal/int comparison mismatch");
                }
            }
        }
    });

    // Max precision (38) path with gandiva semantics: differential test
    // against arrow_array_decimal_arithmetic_util, including values that do
    // not fit in int64 (128-bit slow path) and scale reduction with
    // rounding.
    bodo::tests::test("test_decimal_int64_max_precision_vs_gandiva", [] {
        std::mt19937_64 gen(31415);
        struct Case {
            int p1, s1, p2, s2;
        };
        std::vector<Case> cases = {
            {20, 2, 19, 3},
            {38, 0, 18, 0},
            {25, 10, 24, 9},
            {19, 0, 19, 0},
        };
        for (const auto& c : cases) {
            for (const std::string op : {"add", "subtract", "multiply"}) {
                auto [out_p, out_s] =
                    op_result_precision_scale(op, c.p1, c.s1, c.p2, c.s2);
                // The fast path caps the result at precision 38 like
                // decimal_arithmetic does.
                out_p = std::min(out_p, 38);
                // Cap generated magnitudes so the exact operation result
                // always fits in 38 digits (the reference gandiva utility
                // silently appends undefined values on overflow, so
                // overflow cases are tested separately below).
                auto left = make_decimal_array(
                    c.p1, c.s1,
                    gen_values(gen, std::min(c.p1, 20), c.s1, 41, 0.2));
                auto right = make_decimal_array(
                    c.p2, c.s2,
                    gen_values(gen, std::min(c.p2, 20), c.s2, 41, 0.2));

                auto ref = arrow_array_decimal_arithmetic_util(
                    std::static_pointer_cast<arrow::Decimal128Array>(left),
                    c.p1, c.s1,
                    std::static_pointer_cast<arrow::Decimal128Array>(right),
                    c.p2, c.s2, left->length(), out_p, out_s, op);
                auto fast =
                    decimal_int64_binary_op(left, c.p1, c.s1, right, c.p2, c.s2,
                                            out_p, out_s, op, true);
                std::string case_msg = "case p1=" + std::to_string(c.p1) +
                                       " s1=" + std::to_string(c.s1) +
                                       " p2=" + std::to_string(c.p2) +
                                       " s2=" + std::to_string(c.s2) +
                                       " op=" + op +
                                       " out_p=" + std::to_string(out_p) +
                                       " out_s=" + std::to_string(out_s);
                if (ref == nullptr) {
                    bodo::tests::check(!fast.ok(),
                                       (case_msg + ": expected overflow in "
                                                   "fast path")
                                           .c_str());
                } else {
                    bodo::tests::check(
                        fast.ok(), (case_msg + ": fast path failed: " +
                                    (fast.ok() ? "" : fast.status().ToString()))
                                       .c_str());
                    check_decimal_arrays_equal(
                        ref, fast.ValueOrDie().make_array(),
                        (case_msg + ": max precision fast path mismatch")
                            .c_str());
                }
            }
        }
    });

    // Overflow in the max precision path: overflowing rows only occur on
    // some ranks, and raising an error on just those ranks deadlocks the
    // other ranks in mismatched MPI collectives, so the array kernels match
    // the legacy array utility and store its zero-initialized result.
    // Scalar-scalar overflow is rank-uniform and still raises.
    bodo::tests::test("test_decimal_int64_max_precision_overflow", [] {
        auto left = make_decimal_array(
            38, 0, {int128_pow10(38) - 1});  // 10^38 - 1, the max value
        auto right = make_decimal_array(
            18, 0, {(__int128)999999999999999999LL});  // ~1e18
        auto fast = decimal_int64_binary_op(left, 38, 0, right, 18, 0, 38, 0,
                                            "multiply", true);
        bodo::tests::check(fast.ok(), "array fast path failed");
        auto fast_arr = fast.ValueOrDie().make_array();
        bodo::tests::check(!fast_arr->IsNull(0), "expected non-null result");
        auto fast_dec =
            std::static_pointer_cast<arrow::Decimal128Array>(fast_arr);
        bodo::tests::check(read_le_decimal128(fast_dec->GetValue(0)) == 0,
                           "expected zero result for overflow");

        // Multirow decimal(38,0) * decimal(38,0) through the array-level
        // kernel: 10^19 does not fit in int64, so rows take the 128-bit slow
        // path, and the product 10^38 overflows decimal(38,0) on a subset of
        // rows (including a null operand row).
        auto big_left = make_bodo_decimal_array(
            38, 0,
            {std::optional<__int128>(int128_pow10(19)),
             std::optional<__int128>(1), std::nullopt,
             std::optional<__int128>(-int128_pow10(19))});
        auto big_right = make_bodo_decimal_array(
            38, 0,
            {std::optional<__int128>(int128_pow10(19)),
             std::optional<__int128>(int128_pow10(19)),
             std::optional<__int128>(int128_pow10(19)),
             std::optional<__int128>(int128_pow10(19))});
        auto fast_big = decimal_int64_binary_op_arrays(
            big_left, false, big_right, false, 38, 0, "multiply", true);
        bodo::tests::check(fast_big != nullptr,
                           "arrays kernel failed on 2x38-bit multiply");
        bodo::tests::check(
            !fast_big->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(2),
            "expected null in row 2");
        const uint8_t* big_vals =
            (const uint8_t*)
                fast_big->data1<bodo_array_type::NULLABLE_INT_BOOL>();
        const size_t big_w = bodo_array_item_size(*fast_big);
        // Rows 0 and 3 overflow decimal(38,0) (positive and negative), row 1
        // stays within precision.
        bodo::tests::check(decimal_get_value(big_vals + big_w * 0, 38) == 0,
                           "expected zero result for overflow");
        bodo::tests::check(decimal_get_value(big_vals + big_w * 3, 38) == 0,
                           "expected zero result for negative overflow");
        bodo::tests::check(
            decimal_get_value(big_vals + big_w * 1, 38) == int128_pow10(19),
            "expected non-overflowing row to keep its value");

        // Same input through the Datum-level kernel.
        auto datum_left = decimal_int64_binary_op(
            arrow::Datum(make_decimal_array(
                38, 0, {std::optional<__int128>(int128_pow10(19))})),
            38, 0,
            arrow::Datum(make_decimal_array(
                38, 0, {std::optional<__int128>(int128_pow10(19))})),
            38, 0, 38, 0, "multiply", true);
        bodo::tests::check(datum_left.ok(), "datum kernel failed");
        auto datum_arr = datum_left.ValueOrDie().make_array();
        auto datum_dec =
            std::static_pointer_cast<arrow::Decimal128Array>(datum_arr);
        bodo::tests::check(read_le_decimal128(datum_dec->GetValue(0)) == 0,
                           "expected zero result for overflow");
    });

    // Null scalar operand produces an all-null / null result.
    bodo::tests::test("test_decimal_int64_null_scalar", [] {
        std::mt19937_64 gen(2718);
        int p1 = 15, s1 = 2;
        auto arr = make_decimal_array(p1, s1, gen_values(gen, p1, s1, 10, 0.0));
        auto null_scalar = make_decimal_scalar(p1, s1, std::nullopt);

        auto res =
            decimal_int64_binary_op(arr, p1, s1, arrow::Datum(null_scalar), p1,
                                    s1, 16, 2, "add", false);
        bodo::tests::check(res.ok(), "fast path failed");
        auto res_arr = res.ValueOrDie().make_array();
        for (int64_t i = 0; i < res_arr->length(); i++) {
            bodo::tests::check(res_arr->IsNull(i), "expected all null");
        }

        auto res_scalar = decimal_int64_binary_op(arrow::Datum(null_scalar), p1,
                                                  s1, arrow::Datum(null_scalar),
                                                  p1, s1, 16, 2, "add", false);
        bodo::tests::check(res_scalar.ok(), "fast path failed");
        bodo::tests::check(res_scalar.ValueOrDie().scalar()->is_valid == false,
                           "expected null scalar");
    });

    // In-place array-level kernel: differential test against the Datum-level
    // kernel for decimal-decimal arithmetic and comparisons (arrays and
    // scalars, with nulls).
    bodo::tests::test("test_decimal_int64_arrays_kernel_vs_datum_kernel", [] {
        std::mt19937_64 gen(271828);
        struct Case {
            int p1, s1, p2, s2;
        };
        std::vector<Case> cases = {
            {15, 2, 16, 2},
            {9, 0, 9, 4},
            {18, 0, 18, 0},
            {12, 5, 10, 7},
        };
        for (const auto& c : cases) {
            auto vals1 = gen_values(gen, c.p1, c.s1, 61, 0.2);
            auto vals2 = gen_values(gen, c.p2, c.s2, 61, 0.2);
            for (const std::string op :
                 {"add", "subtract", "multiply", "equal", "not_equal", "less",
                  "greater", "less_equal", "greater_equal"}) {
                bool is_comparison = op == "equal" || op == "not_equal" ||
                                     op == "less" || op == "greater" ||
                                     op == "less_equal" ||
                                     op == "greater_equal";
                auto [out_p, out_s] =
                    is_comparison
                        ? std::pair<int, int>(1, 0)
                        : op_result_precision_scale(op, c.p1, c.s1, c.p2, c.s2);
                if (op == "multiply" && c.s1 + c.s2 != out_s) {
                    continue;
                }
                for (bool scalar_side : {false, true}) {
                    auto bodo_left = make_bodo_decimal_array(c.p1, c.s1, vals1);
                    auto bodo_right =
                        make_bodo_decimal_array(c.p2, c.s2, vals2);
                    auto left_arr = make_decimal_array(c.p1, c.s1, vals1);
                    auto right_arr = make_decimal_array(c.p2, c.s2, vals2);
                    auto left_scalar =
                        make_decimal_scalar(c.p1, c.s1, vals1[0]);
                    auto right_scalar =
                        make_decimal_scalar(c.p2, c.s2, vals2[0]);
                    bool left_is_scalar = scalar_side;
                    bool right_is_scalar = !scalar_side;
                    auto datum_left = scalar_side ? arrow::Datum(left_scalar)
                                                  : arrow::Datum(left_arr);
                    auto datum_right = scalar_side ? arrow::Datum(right_arr)
                                                   : arrow::Datum(right_scalar);

                    auto ref = decimal_int64_binary_op(datum_left, c.p1, c.s1,
                                                       datum_right, c.p2, c.s2,
                                                       out_p, out_s, op, false);
                    bodo::tests::check(ref.ok(), "datum kernel failed");
                    auto fast = decimal_int64_binary_op_arrays(
                        bodo_left, left_is_scalar, bodo_right, right_is_scalar,
                        out_p, out_s, op, false);
                    bodo::tests::check(fast != nullptr,
                                       "arrays kernel not supported");
                    auto ref_arr = ref.ValueOrDie().make_array();
                    const uint8_t* fast_vals =
                        (const uint8_t*)
                            fast->data1<bodo_array_type::NULLABLE_INT_BOOL>();
                    const size_t fast_w = bodo_array_item_size(*fast);
                    for (int64_t i = 0; i < ref_arr->length(); i++) {
                        bool ref_null = ref_arr->IsNull(i);
                        bool fast_null = !fast->get_null_bit<
                            bodo_array_type::NULLABLE_INT_BOOL>(i);
                        bodo::tests::check(ref_null == fast_null,
                                           "null mismatch");
                        if (!ref_null) {
                            if (is_comparison) {
                                bodo::tests::check(
                                    std::static_pointer_cast<
                                        arrow::BooleanArray>(ref_arr)
                                            ->Value(i) == GetBit(fast_vals, i),
                                    "bool mismatch");
                            } else {
                                auto ref_dec = std::static_pointer_cast<
                                    arrow::Decimal128Array>(ref_arr);
                                bodo::tests::check(
                                    read_le_decimal128(ref_dec->GetValue(i)) ==
                                        decimal_get_value(
                                            fast_vals + fast_w * i, out_p),
                                    "arrays kernel mismatch vs datum kernel");
                            }
                        }
                    }
                }
            }
        }
    });

    // 8-byte int64-backed decimal storage: read path decodes decimal(p <= 18)
    // Arrow Decimal128Arrays into 8-byte little-endian buffers, the sink
    // widens back to 16 bytes, and decimal(p > 18) keeps the 16-byte form.
    bodo::tests::test("test_decimal_int64_storage_round_trip", [] {
        struct Case {
            int precision, scale;
            bool expect_8_byte;
        };
        std::vector<Case> cases = {
            {15, 2, true}, {18, 0, true},  {18, 18, true},
            {1, 0, true},  {19, 4, false}, {38, 6, false},
        };
        for (const auto& c : cases) {
            // Values include negatives, zero, the precision boundary and
            // nulls; dictionary-free plain arrays suffice here.
            std::vector<std::optional<__int128>> unscaled = {
                std::optional<__int128>(0),
                std::optional<__int128>(-1),
                std::optional<__int128>(12345),
                std::nullopt,
            };
            __int128 max_unscaled = 1;
            for (int i = 0; i < c.precision; i++) {
                max_unscaled *= 10;
            }
            max_unscaled -= 1;
            unscaled.push_back(max_unscaled);
            unscaled.push_back(-max_unscaled);

            auto arr = make_decimal_array(c.precision, c.scale, unscaled);

            // The default conversion is the JIT boundary and always keeps
            // 16-byte decimal buffers, regardless of the env flag. An
            // explicit decimal_int64_storage=true request uses 8-byte
            // buffers for decimal(p <= 18).
            for (bool explicit_int64 : {false, true}) {
                auto bodo_arr = arrow_array_to_bodo(
                    arr, bodo::BufferPool::DefaultPtr(),
                    /*array_id=*/-1, /*dicts_ref_arr=*/nullptr,
                    /*decimal_int64_storage=*/explicit_int64);
                bodo::tests::check(bodo_arr->dtype == Bodo_CTypes::DECIMAL,
                                   "round trip: wrong dtype");
                bodo::tests::check((int)bodo_arr->precision == c.precision,
                                   "round trip: precision mismatch");
                bodo::tests::check((int)bodo_arr->scale == c.scale,
                                   "round trip: scale mismatch");
                size_t want_w = bodo_array_item_size(*bodo_arr);
                bodo::tests::check(
                    want_w == ((c.expect_8_byte && explicit_int64)
                                   ? (size_t)8
                                   : (size_t)16),
                    "round trip: unexpected storage width");
                const uint8_t* vals =
                    (const uint8_t*)
                        bodo_arr->data1<bodo_array_type::NULLABLE_INT_BOOL>();
                for (int64_t i = 0; i < (int64_t)bodo_arr->length; i++) {
                    bool is_null =
                        !bodo_arr
                             ->get_null_bit<bodo_array_type::NULLABLE_INT_BOOL>(
                                 i);
                    bodo::tests::check(is_null == arr->IsNull(i),
                                       "round trip: null mismatch");
                    if (!is_null) {
                        __int128 got =
                            decimal_load_wide(vals + want_w * i, want_w);
                        bodo::tests::check(
                            got == (__int128)*unscaled[i],
                            "round trip: value mismatch on read path");
                    }
                }

                // Sink conversion must widen back to the 16-byte Arrow form.
                arrow::TimeUnit::type time_unit = arrow::TimeUnit::NANO;
                auto out_arrow = bodo_array_to_arrow(
                    arrow::default_memory_pool(), bodo_arr, false, "",
                    time_unit, false, bodo::default_buffer_memory_manager());
                auto out_dec =
                    std::static_pointer_cast<arrow::Decimal128Array>(out_arrow);
                bodo::tests::check(
                    out_dec->length() == (int64_t)bodo_arr->length,
                    "round trip: sink length mismatch");
                for (int64_t i = 0; i < out_dec->length(); i++) {
                    if (arr->IsNull(i)) {
                        bodo::tests::check(out_dec->IsNull(i),
                                           "round trip: sink null mismatch");
                        continue;
                    }
                    bodo::tests::check(read_le_decimal128(out_dec->GetValue(
                                           i)) == (__int128)*unscaled[i],
                                       "round trip: sink value mismatch");
                }
            }
        }
    });
});
