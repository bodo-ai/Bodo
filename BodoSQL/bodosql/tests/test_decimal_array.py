import operator
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import bodo
import bodosql
from bodo.tests.utils import check_func, pytest_mark_one_rank
from bodo.utils.typing import BodoError


@pytest.fixture(
    params=[
        pytest.param(
            np.array(
                [
                    Decimal("1.6"),
                    None,
                    Decimal("-0.222"),
                    Decimal("1111.316"),
                    Decimal("1234.00046"),
                    Decimal("5.1"),
                    Decimal("-11131.0056"),
                    Decimal("0.0"),
                ]
                * 10
            ),
            marks=pytest.mark.slow,
        ),
        np.array(
            [
                Decimal("1.6"),
                None,
                Decimal("-0.222"),
                Decimal("1111.316"),
                Decimal("1234.00046"),
                Decimal("5.1"),
                Decimal("-11131.0056"),
                Decimal("0.0"),
            ]
        ),
    ]
)
def decimal_arr_value(request):
    return request.param


def test_np_sort(memory_leak_check):
    def impl(arr):
        return np.sort(arr)

    A = np.array(
        [
            Decimal("1.6"),
            Decimal("-0.222"),
            Decimal("1111.316"),
            Decimal("1234.00046"),
            Decimal("5.1"),
            Decimal("-11131.0056"),
            Decimal("0.0"),
        ]
        * 20
    )

    check_func(impl, (A,))


def test_np_repeat(decimal_arr_value, memory_leak_check):
    def impl(arr):
        return np.repeat(arr, 2)

    check_func(impl, (decimal_arr_value,))


def test_np_unique(memory_leak_check):
    def impl(arr):
        return np.unique(arr)

    # Create an array here because np.unique fails on NA in pandas
    arr = np.array(
        [
            Decimal("1.6"),
            Decimal("-0.222"),
            Decimal("5.1"),
            Decimal("1111.316"),
            Decimal("-0.2220001"),
            Decimal("-0.2220"),
            Decimal("1234.00046"),
            Decimal("5.1"),
            Decimal("-11131.0056"),
            Decimal("0.0"),
            Decimal("5.11"),
            Decimal("0.00"),
            Decimal("0.01"),
            Decimal("0.03"),
            Decimal("0.113"),
            Decimal("1.113"),
        ]
    )
    arr = pd.array(arr, dtype=pd.ArrowDtype(pa.decimal128(38, 12)))
    check_func(impl, (arr,), sort_output=True, is_out_distributed=False)


@pytest.mark.slow
def test_unbox(decimal_arr_value, memory_leak_check):
    # just unbox
    def impl(arr_arg):
        return True

    check_func(impl, (decimal_arr_value,))

    # unbox and box
    def impl2(arr_arg):
        return arr_arg

    check_func(impl2, (decimal_arr_value,))


@pytest.mark.slow
def test_len(decimal_arr_value, memory_leak_check):
    def test_impl(A):
        return len(A)

    check_func(test_impl, (decimal_arr_value,))


@pytest.mark.slow
def test_shape(decimal_arr_value, memory_leak_check):
    def test_impl(A):
        return A.shape

    check_func(test_impl, (decimal_arr_value,))


@pytest.mark.slow
def test_dtype(decimal_arr_value, memory_leak_check):
    def test_impl(A):
        return A.dtype

    check_func(test_impl, (decimal_arr_value,))


@pytest.mark.slow
def test_ndim(decimal_arr_value, memory_leak_check):
    def test_impl(A):
        return A.ndim

    check_func(test_impl, (decimal_arr_value,))


@pytest.mark.slow
def test_decimal_coerce(memory_leak_check):
    ts = Decimal("4.5")

    def f(df, ts):
        df["ts"] = ts
        return df

    df1 = pd.DataFrame({"a": 1 + np.arange(6)})
    check_func(f, (df1, ts))


def test_series_astype_str(decimal_arr_value, memory_leak_check):
    """test decimal conversion to string.
    Using a checksum for checking output since Bodo's output can have extra 0 digits
    """

    def test_impl(A):
        S2 = A.astype(str).values
        s = 0.0
        for i in bodo.prange(len(S2)):
            val = 0
            if not (
                bodo.libs.array_kernels.isna(S2, i)
                or S2[i] == "None"
                or S2[i] == "<NA>"
            ):
                val = float(S2[i])
            s += val
        return s

    S = pd.Series(decimal_arr_value)
    check_func(test_impl, (S,), py_output=float(S.sum()), atol=1e-2, rtol=1e-2)


@pytest.mark.slow
@pytest.mark.parametrize(
    "decimal_value",
    [
        # long value to exercise both 64-bit slots
        Decimal("422222222.511133333444411"),
        # short value to test an empty 64-bit slot
        Decimal("4.5"),
    ],
)
def test_decimal_constant_lowering(decimal_value, memory_leak_check):
    def f():
        return decimal_value

    bodo_f = bodo.jit(f)
    val_ret = bodo_f()
    assert val_ret == pa.scalar(decimal_value, pa.decimal128(38, 18))


def test_join(decimal_arr_value, memory_leak_check):
    """test joining DataFrames with decimal data columns
    TODO: add decimal array to regular df tests and remove this
    """

    def test_impl(df1, df2):
        return df1.merge(df2, on="A")

    # double the size of the input array to avoid issues on 3 processes
    decimal_arr_value = np.concatenate((decimal_arr_value, decimal_arr_value))
    n = len(decimal_arr_value)
    df1 = pd.DataFrame(
        {
            "A": np.arange(n),
            "B": pd.array(
                decimal_arr_value, dtype=pd.ArrowDtype(pa.decimal128(38, 18))
            ),
        }
    )
    df2 = pd.DataFrame(
        {
            "A": np.arange(n) + 3,
            "C": pd.array(
                decimal_arr_value, dtype=pd.ArrowDtype(pa.decimal128(38, 18))
            ),
        }
    )
    check_func(test_impl, (df1, df2), sort_output=True, reset_index=True)


def test_constructor(memory_leak_check):
    def test_impl1():
        return Decimal("1.1")

    def test_impl2():
        return Decimal()

    def test_impl3():
        return Decimal(1)

    def test_impl4():
        return Decimal(" -1.135E  ")

    check_func(
        test_impl1, (), py_output=pa.scalar(Decimal("1.1"), pa.decimal128(38, 18))
    )
    check_func(test_impl2, (), py_output=pa.scalar(Decimal(), pa.decimal128(38, 18)))
    check_func(test_impl3, (), py_output=pa.scalar(Decimal(1), pa.decimal128(38, 18)))
    check_func(
        test_impl4, (), py_output=pa.scalar(Decimal("-1.135"), pa.decimal128(38, 18))
    )


# TODO: fix memory leak and add memory_leak_check
@pytest.mark.slow
def test_constant_lowering(decimal_arr_value):
    def impl():
        return decimal_arr_value

    pd.testing.assert_series_equal(
        pd.Series(bodo.jit(impl)()), pd.Series(decimal_arr_value), check_dtype=False
    )


@pytest.mark.slow
def test_constructor_error(memory_leak_check):
    """Test that an invalid constructor throws a BodoError"""

    def impl():
        return Decimal([1.1, 2.2, 3.2])

    with pytest.raises(BodoError, match=r"decimal.Decimal\(\) value type must be"):
        bodo.jit(impl)()


def test_decimal_ops(memory_leak_check):
    def test_impl_eq(d1, d2):
        return d1 == d2

    def test_impl_ne(d1, d2):
        return d1 != d2

    def test_impl_gt(d1, d2):
        return d1 > d2

    def test_impl_ge(d1, d2):
        return d1 >= d2

    def test_impl_lt(d1, d2):
        return d1 < d2

    def test_impl_le(d1, d2):
        return d1 <= d2

    test_funcs = [
        test_impl_eq,
        test_impl_ne,
        test_impl_gt,
        test_impl_ge,
        test_impl_lt,
        test_impl_le,
    ]

    d1 = Decimal("-1.1")
    d2 = Decimal("100.2")

    for func in test_funcs:
        check_func(func, (d1, d1))
        check_func(func, (d1, d2))
        check_func(func, (d2, d1))


@pytest.mark.smoke
def test_setitem_int(decimal_arr_value, memory_leak_check):
    def test_impl(A, val):
        A[2] = val
        return A

    val = decimal_arr_value[0]
    check_func(test_impl, (decimal_arr_value, val))


@pytest.mark.smoke
def test_setitem_arr(decimal_arr_value, memory_leak_check):
    def test_impl(A, idx, val):
        A[idx] = val
        return A

    np.random.seed(0)
    idx = np.random.randint(0, len(decimal_arr_value), 11)
    val = np.array([round(Decimal(val), 10) for val in np.random.rand(11)])
    check_func(
        test_impl, (decimal_arr_value, idx, val), dist_test=False, copy_input=True
    )

    # Single Decimal as a value, reuses the same idx
    val = Decimal("1.31131")
    check_func(
        test_impl, (decimal_arr_value, idx, val), dist_test=False, copy_input=True
    )

    idx = np.random.ranf(len(decimal_arr_value)) < 0.2
    val = np.array([round(Decimal(val), 10) for val in np.random.rand(idx.sum())])
    check_func(
        test_impl, (decimal_arr_value, idx, val), dist_test=False, copy_input=True
    )

    # Single Decimal as a value, reuses the same idx
    val = Decimal("1.31131")
    check_func(
        test_impl, (decimal_arr_value, idx, val), dist_test=False, copy_input=True
    )

    idx = slice(1, 4)
    val = np.array([round(Decimal(val), 10) for val in np.random.rand(3)])
    check_func(
        test_impl, (decimal_arr_value, idx, val), dist_test=False, copy_input=True
    )

    # Single Decimal as a value, reuses the same idx
    val = Decimal("1.31131")
    check_func(
        test_impl, (decimal_arr_value, idx, val), dist_test=False, copy_input=True
    )


@pytest.mark.slow
def test_decimal_arr_nbytes(memory_leak_check):
    """Test DecimalArrayType nbytes"""

    def impl(A):
        return A.nbytes

    arr = np.array(
        [
            Decimal("1.6"),
            None,
            Decimal("-0.222"),
            Decimal("1111.316"),
            Decimal("1234.00046"),
            Decimal("5.1"),
            Decimal("-11131.0056"),
            Decimal("0.0"),
        ]
    )
    py_out = 128 + bodo.get_size()  # 1 extra byte for null_bit_map per rank
    check_func(impl, (arr,), py_output=py_out, only_1D=True, only_1DVar=True)
    check_func(impl, (arr,), py_output=129, only_seq=True)


@pytest.mark.parametrize(
    "value",
    [
        # int32 array
        pd.array([1, -1, 3, 4, -2, 0, None, 4], "Int32"),
        # int64 array
        pd.array([5, -1, 0, None, -2, 10, None, 12], "Int64"),
        # numpy array
        np.array([5, -21131, 0, 7, -2, 10, 12340, 12]),
        # float32 array
        pd.array([5.1, -1.1, 0.54, None, -2.1, 101.1, None, 1.234], "Float32"),
        # float64 array
        pd.array(
            [1.111, -1.12, 1000000.54, None, -2000.1, -101.1, None, 1.234], "Float64"
        ),
        # decimal array
        np.array(
            [
                Decimal("1.62"),
                Decimal("-1.222"),
                Decimal("1.316"),
                Decimal("-4.00046"),
                Decimal("5.14"),
                None,
                Decimal("-131.0056"),
                Decimal("0.0"),
            ]
        ),
        # int32 scalar
        np.int32(1),
        # int64 scalar
        -1,
        # float32 scalar
        np.float32(1.4),
        # float64 scalar
        -0.4,
        # decimal scalar
        Decimal("1.45"),
    ],
)
def test_decimal_comparison(value, memory_leak_check):
    arr = np.array(
        [
            Decimal("1.6"),
            None,
            Decimal("-0.222"),
            Decimal("1111.316"),
            Decimal("1234.00046"),
            Decimal("5.1"),
            Decimal("-11131.0056"),
            Decimal("0.0"),
        ]
    )
    pa_val = (
        pa.scalar(value)
        if bodo.utils.typing.is_scalar_type(bodo.typeof(value))
        else pa.array(value)
    )

    for op, pa_op in (
        (operator.gt, pa.compute.greater),
        (operator.ge, pa.compute.greater_equal),
        (operator.eq, pa.compute.equal),
        (operator.ne, pa.compute.not_equal),
        (operator.lt, pa.compute.less),
        (operator.le, pa.compute.less_equal),
    ):

        def impl(a, b):
            return op(a, b)

        py_output = pd.array(pa_op(pa.array(arr), pa_val), pd.BooleanDtype())
        check_func(impl, (arr, value), py_output=py_output)
        py_output = pd.array(pa_op(pa_val, pa.array(arr)), pd.BooleanDtype())
        check_func(impl, (value, arr), py_output=py_output)


def test_decimal_comparison_error_checking():
    """Make sure decimal comparison with invalid type raises an error"""
    arr = np.array(
        [
            Decimal("1.6"),
            None,
            Decimal("-0.222"),
        ]
    )
    other = np.array(["1.2", "3.1", None], object)

    def impl(a, b):
        return a < b

    with pytest.raises(BodoError, match=r"Invalid decimal comparison with"):
        bodo.jit(impl)(arr, other)


@pytest_mark_one_rank
def test_decimal_scalar_int_cast(memory_leak_check):
    """Test casting decimal scalars to integers"""

    def impl(a, b, flag):
        if flag:
            c = a
        else:
            c = b
        return c

    # cast to decimal type since wider
    a = pa.scalar(-123, pa.decimal128(20, 0))
    b = np.int32(-3)
    check_func(impl, (a, b, True), py_output=a, only_seq=True)
    check_func(
        impl,
        (a, b, False),
        py_output=pa.scalar(int(b), pa.decimal128(20, 0)),
        only_seq=True,
    )

    # cast to int type since wider
    a = pa.scalar(-123, pa.decimal128(3, 0))
    b = -3
    check_func(impl, (a, b, True), py_output=a.cast(pa.int64()).as_py(), only_seq=True)
    check_func(impl, (a, b, False), py_output=b, only_seq=True)


@pytest_mark_one_rank
def test_setitem_cast_decimal(memory_leak_check):
    """Test automatic cast of decimal to int/float in array setitem"""

    def test_impl(A, val):
        A[2] = val
        return A

    # Int case
    arr = pd.array([1, 2, 3, 4, 5], "Int64")
    val = Decimal("123")
    # Pandas doesn't support Decimal setitem yet so create output manually
    out = arr.copy()
    out[2] = int(val)
    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_extension_array_equal(
        bodo_func(arr.copy(), val), out, check_dtype=False
    )

    # Float case
    arr = pd.array([1.2, 2.3, 3.1, 4.1, 5.4], "Float32")
    val = Decimal("12.3")
    # Pandas doesn't support Decimal setitem yet so create output manually
    out = arr.copy()
    out[2] = float(val)
    bodo_func = bodo.jit(test_impl)
    pd.testing.assert_extension_array_equal(
        bodo_func(arr.copy(), val), out, check_dtype=False
    )


def test_box_arrow_array_precision_scale(precision_scale_decimal_array):
    """
    Test that we can box/unbox an arrow decimal array without 38, 18 precision and scale.
    """

    def impl(arr):
        return arr

    check_func(
        impl, (precision_scale_decimal_array,), py_output=precision_scale_decimal_array
    )


def test_cast_decimal_to_decimal_scalar(
    precision_scale_decimal_array, memory_leak_check
):
    # This takes the unsafe cast route because the leading digits increase.
    def impl1(arr):
        out_arr = bodo.libs.decimal_arr_ext.alloc_decimal_array(len(arr), 28, 3)
        for i in bodo.prange(len(arr)):
            if bodo.libs.array_kernels.isna(arr, i):
                out_arr[i] = None
            else:
                out_arr[i] = (
                    bodosql.kernels.snowflake_conversion_array_kernels.numeric_to_decimal(
                        arr[i], 28, 3, False
                    )
                )

        return out_arr

    # This takes the safe cast route because the leading digits decrease, so we
    # need to ensure the data fits.
    def impl2(arr):
        out_arr = bodo.libs.decimal_arr_ext.alloc_decimal_array(len(arr), 15, 1)
        for i in bodo.prange(len(arr)):
            if bodo.libs.array_kernels.isna(arr, i):
                out_arr[i] = None
            else:
                out_arr[i] = (
                    bodosql.kernels.snowflake_conversion_array_kernels.numeric_to_decimal(
                        arr[i], 15, 1, False
                    )
                )
        return out_arr

    py_output1 = pd.array(
        ["1", "1.55", "1.56", "10.56", "1000.5", None, None, "10004.1", "-11.41"],
        dtype=pd.ArrowDtype(pa.decimal128(28, 3)),
    )
    check_func(impl1, (precision_scale_decimal_array,), py_output=py_output1)
    py_output2 = pd.array(
        ["1", "1.6", "1.6", "10.6", "1000.5", None, None, "10004.1", "-11.4"],
        dtype=pd.ArrowDtype(pa.decimal128(15, 1)),
    )
    check_func(impl2, (precision_scale_decimal_array,), py_output=py_output2)


def test_cast_decimal_to_decimal_scalar_loss_null_on_error(
    precision_scale_decimal_array, memory_leak_check
):
    """
    Test that when decimal scalars are truncated we correctly output NULL
    values if error on null is set.
    """

    def impl(arr):
        out_arr = bodo.libs.decimal_arr_ext.alloc_decimal_array(len(arr), 28, 3)
        for i in bodo.prange(len(arr)):
            if bodo.libs.array_kernels.isna(arr, i):
                out_arr[i] = None
            else:
                out_arr[i] = (
                    bodosql.kernels.snowflake_conversion_array_kernels.numeric_to_decimal(
                        arr[i], 4, 3, True
                    )
                )

        return out_arr

    py_output = pd.array(
        ["1", "1.55", "1.56", None, None, None, None, None, None],
        dtype=pd.ArrowDtype(pa.decimal128(4, 3)),
    )
    check_func(impl, (precision_scale_decimal_array,), py_output=py_output)


@pytest_mark_one_rank
def test_cast_decimal_to_decimal_scalar_error(precision_scale_decimal_array):
    """
    Test that decimals that don't fit raise an error when numeric_to_decimal
    doesn't have null on error set.
    """

    @bodo.jit
    def impl(arr):
        out_arr = bodo.libs.decimal_arr_ext.alloc_decimal_array(len(arr), 28, 3)
        for i in bodo.prange(len(arr)):
            if bodo.libs.array_kernels.isna(arr, i):
                out_arr[i] = None
            else:
                out_arr[i] = (
                    bodosql.kernels.snowflake_conversion_array_kernels.numeric_to_decimal(
                        arr[i], 4, 3, False
                    )
                )

        return out_arr

    with pytest.raises(Exception, match=r"Number out of representable range"):
        impl(precision_scale_decimal_array)


def test_cast_decimal_to_decimal_array(
    precision_scale_decimal_array, memory_leak_check
):
    # This takes the unsafe cast route because the leading digits increase.
    def impl1(arr):
        return bodosql.kernels.snowflake_conversion_array_kernels.numeric_to_decimal(
            arr, 28, 3, False
        )

    # This takes the safe cast route because the leading digits decrease, so we
    # need to ensure the data fits.
    def impl2(arr):
        return bodosql.kernels.snowflake_conversion_array_kernels.numeric_to_decimal(
            arr, 15, 1, False
        )

    py_output1 = pd.array(
        ["1", "1.55", "1.56", "10.56", "1000.5", None, None, "10004.1", "-11.41"],
        dtype=pd.ArrowDtype(pa.decimal128(28, 3)),
    )
    check_func(impl1, (precision_scale_decimal_array,), py_output=py_output1)
    py_output2 = pd.array(
        ["1", "1.6", "1.6", "10.6", "1000.5", None, None, "10004.1", "-11.4"],
        dtype=pd.ArrowDtype(pa.decimal128(15, 1)),
    )
    check_func(impl2, (precision_scale_decimal_array,), py_output=py_output2)


def test_cast_decimal_to_decimal_array_loss_null_on_error(
    precision_scale_decimal_array, memory_leak_check
):
    """
    Test that when decimal scalars are truncated we correctly output NULL
    values if error on null is set.
    """

    def impl(arr):
        return bodosql.kernels.snowflake_conversion_array_kernels.numeric_to_decimal(
            arr, 4, 3, True
        )

    py_output = pd.array(
        ["1", "1.55", "1.56", None, None, None, None, None, None],
        dtype=pd.ArrowDtype(pa.decimal128(4, 3)),
    )
    check_func(impl, (precision_scale_decimal_array,), py_output=py_output)


@pytest_mark_one_rank
def test_cast_decimal_to_decimal_array_error(
    precision_scale_decimal_array,
):
    """
    Test that decimals that don't fit raise an error when numeric_to_decimal
    doesn't have null on error set.
    """

    @bodo.jit
    def impl(arr):
        return bodosql.kernels.snowflake_conversion_array_kernels.numeric_to_decimal(
            arr, 4, 3, False
        )

    with pytest.raises(Exception, match=r"Number out of representable range"):
        impl(precision_scale_decimal_array)


@pytest.mark.parametrize(
    "arr, answer",
    [
        pytest.param(
            pd.array(
                ["1", "50000", "20", "400", "-1"] * 3,
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            pd.Series(["151260"], dtype=pd.ArrowDtype(pa.large_string())),
            id="scale_0-no_null",
        ),
        pytest.param(
            pd.array(
                ["1", "50000", None, "400", None] * 3,
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            pd.Series(["151203"], dtype=pd.ArrowDtype(pa.large_string())),
            id="scale_0-some_null",
        ),
        pytest.param(
            pd.array([None] * 500, dtype=pd.ArrowDtype(pa.decimal128(38, 0))),
            pd.Series([None], dtype=pd.ArrowDtype(pa.large_string())),
            id="scale_0-all_null",
        ),
        pytest.param(
            pd.array(
                [
                    None if i % 2 == 0 or i // 100 < 2 else f"{i**3}.{i}"
                    for i in range(300)
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 3)),
            ),
            pd.Series(["812487512.500"], dtype=pd.ArrowDtype(pa.large_string())),
            id="scale_3-some_null",
        ),
    ],
)
def test_decimal_sum(arr, answer, memory_leak_check):
    """Test pd.Series.sum() on decimals"""

    def impl(arr):
        # Wrap in a series to get the sum
        total = pd.Series(arr).sum()
        # Convert to a string to confirm the scale
        asStr = bodosql.kernels.to_char(total, is_scalar=True)
        # Convert back to a Series to allow comparing against None
        result = pd.Series([asStr])
        return result

    check_func(
        impl,
        (arr,),
        py_output=answer,
        is_out_distributed=False,
        check_dtype=False,
        reset_index=True,
    )


def test_decimal_sum_overflow():
    """Test error handling when pd.Series.sum() on decimals
    fails due to overflow"""

    def impl(arr):
        return pd.Series(arr).sum()

    arr = pd.array([99.9] * 100, dtype=pd.ArrowDtype(pa.decimal128(38, 36)))

    with pytest.raises(
        RuntimeError, match="Overflow detected in groupby sum of Decimal data"
    ):
        check_func(
            impl,
            (arr,),
            py_output=-1,
            is_out_distributed=False,
            check_dtype=False,
            reset_index=True,
        )


@pytest.mark.parametrize(
    "arg0, arg1, answers",
    [
        pytest.param(
            pa.scalar(Decimal("132.00"), pa.decimal128(5, 2)),
            pa.scalar(Decimal("2.45"), pa.decimal128(4, 2)),
            pa.scalar(Decimal("2.15"), pa.decimal128(5, 2)),
            id="scalar-scalar-same_scale-fast",
        ),
        pytest.param(
            pa.scalar(Decimal("-1234"), pa.decimal128(10, 0)),
            pa.scalar(Decimal("2.9"), pa.decimal128(10, 2)),
            pa.scalar(Decimal("-1.5"), pa.decimal128(12, 2)),
            id="scalar-scalar-smaller_scale-fast",
        ),
        pytest.param(
            pa.scalar(Decimal("98765432.10"), pa.decimal128(10, 2)),
            pa.scalar(Decimal("-9876543210"), pa.decimal128(10, 0)),
            pa.scalar(Decimal("98765432.10"), pa.decimal128(12, 2)),
            id="scalar-scalar-larger_scale-fast",
        ),
        pytest.param(
            pa.scalar(Decimal("9876543210987654321098765432"), pa.decimal128(38, 0)),
            pa.scalar(Decimal("1234567.89"), pa.decimal128(38, 2)),
            pa.scalar(Decimal("7535.60"), pa.decimal128(38, 2)),
            id="scalar-scalar-smaller_scale-unsafe",
        ),
        pytest.param(
            pa.scalar(Decimal("-987654321.987654321"), pa.decimal128(38, 18)),
            pa.scalar(Decimal("10000000000000000000.89"), pa.decimal128(38, 2)),
            pa.scalar(Decimal("-987654321.987654321"), pa.decimal128(38, 18)),
            id="scalar-scalar-larger_scale-unsafe",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("254.016"),
                    None,
                    Decimal("1234.5678"),
                    None,
                    Decimal("-56.9"),
                    Decimal("-0.01"),
                    Decimal("-0.09999"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            pd.array(
                [
                    Decimal("123.456"),
                    None,
                    Decimal("-98.0"),
                    Decimal("1234"),
                    Decimal("0.0101"),
                    None,
                    Decimal("-0.123456789"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            pd.array(
                [
                    Decimal("7.104"),
                    None,
                    Decimal("58.5678"),
                    None,
                    Decimal("-0.0067"),
                    None,
                    Decimal("-0.09999"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            id="array-array-same_scale-fast",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("625.00"),
                    None,
                    Decimal("65536.33"),
                    None,
                    Decimal("-99999999.99"),
                    Decimal("0.0"),
                    Decimal("-20.48"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
            ),
            pd.array(
                [
                    Decimal("12.25"),
                    None,
                    Decimal("-12345.67"),
                    Decimal("0.0"),
                    Decimal("12345.6789"),
                    None,
                    Decimal("-1.2345"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(9, 4)),
            ),
            pd.array(
                [
                    Decimal("0.25"),
                    None,
                    Decimal("3807.98"),
                    None,
                    Decimal("-0.9"),
                    None,
                    Decimal("-0.728"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(12, 4)),
            ),
            id="array-array-smaller_scale-fast",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("0"),
                    None,
                    Decimal("12345670.89"),
                    None,
                    Decimal("-123456.78"),
                    Decimal("13.5"),
                    Decimal("0"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
            ),
            pd.array(
                [
                    Decimal("12345"),
                    None,
                    Decimal("666"),
                    Decimal("-1"),
                    Decimal("99999"),
                    None,
                    Decimal("-98765"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(5, 0)),
            ),
            pd.array(
                [
                    Decimal("0"),
                    None,
                    Decimal("28.89"),
                    None,
                    Decimal("-23457.78"),
                    None,
                    Decimal("0"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
            ),
            id="array-array-larger_scale-fast",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("9876543210"),
                    None,
                    Decimal("13"),
                    None,
                    Decimal("-99999999999999999"),
                    Decimal("0"),
                    Decimal("-1"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(24, 0)),
            ),
            pd.array(
                [
                    Decimal("6173.1234567789"),
                    None,
                    Decimal("-0.51206"),
                    Decimal("0"),
                    Decimal("1234.5678"),
                    None,
                    Decimal("-0.00000864200001300579"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(24, 20)),
            ),
            pd.array(
                [
                    Decimal("2490.28956163860000000000"),
                    None,
                    Decimal("0.19850000000000000000"),
                    None,
                    Decimal("-671.40000000000000000000"),
                    None,
                    Decimal("-0.00000825249506102173"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 20)),
            ),
            id="array-array-smaller_scale-unsafe",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("12345678901234.5678901234567890"),
                    None,
                    Decimal("12345678901234.5678901234567890"),
                    None,
                    Decimal("-12345678901234.5678901234567890"),
                    Decimal("-12345678901234.5678901234567890"),
                    Decimal("-12345678901234.5678901234567890"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(32, 16)),
            ),
            pd.array(
                [
                    Decimal("987654321"),
                    None,
                    Decimal("-987654321987654321"),
                    Decimal("-987654321987654321"),
                    Decimal("202020202020"),
                    None,
                    Decimal("-202020202020"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(28, 0)),
            ),
            pd.array(
                [
                    Decimal("987543055.5678901234567890"),
                    None,
                    Decimal("12345678901234.5678901234567890"),
                    None,
                    Decimal("-22446578014.5678901234567890"),
                    None,
                    Decimal("-22446578014.5678901234567890"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 16)),
            ),
            id="array-array-larger_scale-unsafe",
        ),
        pytest.param(
            pd.array(
                [Decimal("10987654320.423456789"), None] * 5,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            None,
            pd.array(
                [None] * 10,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            id="array-null",
        ),
        pytest.param(
            None,
            pd.array(
                [Decimal("10987654320.423456789"), None] * 5,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            pd.array(
                [None] * 10,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            id="null-array",
        ),
    ],
)
def test_decimal_modulo(arg0, arg1, answers, memory_leak_check):
    """Test modulo on decimals"""

    def impl(arg0, arg1):
        return bodosql.kernels.modulo_numeric(arg0, arg1)

    check_func(impl, (arg0, arg1), py_output=answers, only_1DVar=True)

    # If the arguments are also arrays, test arr-scalar and scalar-arr by
    # selecting the first rows.
    if isinstance(arg0, pd.core.arrays.base.ExtensionArray) and isinstance(
        arg1, pd.core.arrays.base.ExtensionArray
    ):
        arg0_replicated = (
            pd.concat([pd.Series(arg0[:1]), pd.Series([None])]).repeat(3).values
        )
        arg1_replicated = (
            pd.concat([pd.Series(arg1[:1]), pd.Series([None])]).repeat(3).values
        )
        answer_replicated = (
            pd.concat([pd.Series(answers[:1]), pd.Series([None])]).repeat(3).values
        )
        check_func(
            impl,
            (arg0[0], arg1_replicated),
            py_output=answer_replicated,
            only_1DVar=True,
        )
        check_func(
            impl,
            (arg0_replicated, arg1[0]),
            py_output=answer_replicated,
            only_1DVar=True,
        )


@pytest.mark.parametrize(
    "array_0, array_1",
    [
        pytest.param(True, True, id="array-array"),
        pytest.param(True, False, id="array-scalar"),
        pytest.param(False, True, id="scalar-array"),
        pytest.param(False, False, id="scalar-scalar"),
    ],
)
@pytest.mark.parametrize(
    "arg0, arg1, exception_type, msg",
    [
        pytest.param(
            pa.scalar(Decimal("98765432109876543210"), pa.decimal128(38, 0)),
            pa.scalar(Decimal("187654.87654321"), pa.decimal128(38, 30)),
            RuntimeError,
            "Invalid rescale during decimal modulo",
            id="smaller_scale-overflow_on_rescale",
        ),
        pytest.param(
            pa.scalar(Decimal("-187654.87654321"), pa.decimal128(38, 30)),
            pa.scalar(Decimal("98765432109876543210"), pa.decimal128(38, 0)),
            RuntimeError,
            "Invalid rescale during decimal modulo",
            id="larger_scale-overflow_on_rescale",
        ),
        pytest.param(
            pa.scalar(Decimal("800000000000000001"), pa.decimal128(38, 18)),
            pa.scalar(Decimal("0"), pa.decimal128(38, 18)),
            RuntimeError,
            "Invalid modulo by zero",
            id="zero_divisor",
        ),
    ],
)
def test_decimal_modulo_error(arg0, arg1, exception_type, msg, array_0, array_1):
    """Test modulo of decimals in ways that will cause error"""

    def impl(arg0, arg1):
        return bodosql.kernels.modulo_numeric(arg0, arg1)

    if array_0:
        arg0 = pd.array([arg0.as_py(), None] * 3, dtype=pd.ArrowDtype(arg0.type))

    if array_1:
        arg1 = pd.array([arg1.as_py(), None] * 3, dtype=pd.ArrowDtype(arg1.type))

    with pytest.raises(exception_type, match=msg):
        check_func(impl, (arg0, arg1), py_output=arg0)


@pytest.mark.parametrize(
    "arg, answer",
    [
        pytest.param(
            pa.scalar(Decimal("85.23"), pa.decimal128(4, 2)),
            1,
            id="scalar-positive",
        ),
        pytest.param(
            pa.scalar(Decimal("0"), pa.decimal128(4, 2)),
            0,
            id="scalar-zero",
        ),
        pytest.param(
            pa.scalar(Decimal("-12.345"), pa.decimal128(10, 5)),
            -1,
            id="scalar-negative",
        ),
        pytest.param(
            pa.scalar(
                Decimal("123456789012345678901234567890.12345678"), pa.decimal128(38, 8)
            ),
            1,
            id="scalar-large",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("512.32"),
                    Decimal("0"),
                    Decimal("-153245.152"),
                    None,
                    Decimal("123456789012345678901234567890.12345678"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 8)),
            ),
            pd.array(
                [1, 0, -1, None, 1],
            ),
            id="array",
        ),
    ],
)
def test_decimal_sign(arg, answer, memory_leak_check):
    """Test adding decimals"""

    def impl(arg0):
        return bodosql.kernels.sign(arg0)

    check_func(impl, (arg,), py_output=answer)


@pytest.mark.parametrize(
    "arg0, arg1, answer",
    [
        pytest.param(
            pa.scalar(Decimal("85.23"), pa.decimal128(4, 2)),
            pa.scalar(Decimal("19.45"), pa.decimal128(4, 2)),
            pa.scalar(Decimal("104.68"), pa.decimal128(5, 2)),
            id="scalar-scalar-same_scale-fast",
        ),
        pytest.param(
            pa.scalar(Decimal("543"), pa.decimal128(3, 0)),
            pa.scalar(Decimal("16.45"), pa.decimal128(4, 2)),
            pa.scalar(Decimal("559.45"), pa.decimal128(6, 2)),
            id="scalar-scalar-smaller_scale-fast",
        ),
        pytest.param(
            pa.scalar(Decimal("1.23456"), pa.decimal128(6, 5)),
            pa.scalar(Decimal("9876543210"), pa.decimal128(10, 0)),
            pa.scalar(Decimal("9876543211.23456"), pa.decimal128(16, 5)),
            id="scalar-scalar-larger_scale-fast",
        ),
        pytest.param(
            pa.scalar(Decimal("-85.23"), pa.decimal128(38, 2)),
            pa.scalar(Decimal("19.45"), pa.decimal128(38, 2)),
            pa.scalar(Decimal("-65.78"), pa.decimal128(38, 2)),
            id="scalar-scalar-same_scale-unsafe",
        ),
        pytest.param(
            pa.scalar(Decimal("123456789012345678901234567890"), pa.decimal128(38, 0)),
            pa.scalar(Decimal("00.12"), pa.decimal128(4, 2)),
            pa.scalar(
                Decimal("123456789012345678901234567890.12"), pa.decimal128(38, 2)
            ),
            id="scalar-scalar-smaller_scale-unsafe",
        ),
        pytest.param(
            pa.scalar(Decimal("112112321123.4321123454321"), pa.decimal128(30, 18)),
            pa.scalar(Decimal("-12345.6789"), pa.decimal128(35, 4)),
            pa.scalar(Decimal("112112308777.7532123454321"), pa.decimal128(38, 18)),
            id="scalar-scalar-larger_scale-unsafe",
        ),
        pytest.param(
            pd.array(
                [Decimal("-5.00"), Decimal("4.56"), Decimal("9.99"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(3, 2)),
            ),
            pd.array(
                [Decimal("-6.45"), None, Decimal("0.01"), Decimal("1.23")] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(3, 2)),
            ),
            pd.array(
                [Decimal("-11.45"), None, Decimal("10.00"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(4, 2)),
            ),
            id="array-array-same_scale-safe",
        ),
        pytest.param(
            pd.array(
                [Decimal("994"), Decimal("456"), Decimal("-12"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(3, 0)),
            ),
            pd.array(
                [Decimal("9.45"), None, Decimal("0.01"), Decimal("1.23")] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(3, 2)),
            ),
            pd.array(
                [Decimal("1003.45"), None, Decimal("-11.99"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(6, 2)),
            ),
            id="array-array-smaller_scale-safe",
        ),
        pytest.param(
            pd.array(
                [Decimal("1.234"), Decimal("5.678"), Decimal("-9.101"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(4, 3)),
            ),
            pd.array(
                [Decimal("9999.9"), None, Decimal("-9999.9"), Decimal("0")] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(5, 1)),
            ),
            pd.array(
                [Decimal("10001.134"), None, Decimal("-10009.001"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(8, 3)),
            ),
            id="array-array-larger_scale-safe",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("12345678901234567890"),
                    Decimal("-1024"),
                    Decimal("1000000000000000000000000000000000001"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            pd.array(
                [
                    Decimal("97531975319753197531"),
                    None,
                    Decimal("-3000000000000000000000000000000000076"),
                    Decimal("0"),
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            pd.array(
                [
                    Decimal("109877654220987765421"),
                    None,
                    Decimal("-2000000000000000000000000000000000075"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            id="array-array-same_scale-unsafe",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("12345678901234567"),
                    Decimal("-1024"),
                    Decimal("123456789123456789"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            pd.array(
                [
                    Decimal("97531975319753197"),
                    None,
                    Decimal("-7654321987654321.987654321123456789"),
                    Decimal("0"),
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            pd.array(
                [
                    Decimal("109877654220987764"),
                    None,
                    Decimal("115802467135802467.012345678876543211"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            id="array-array-smaller_scale-unsafe",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("987654.32109876543210"),
                    Decimal("3.1415926"),
                    Decimal("-123456.7890123456"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(24, 18)),
            ),
            pd.array(
                [
                    Decimal("98765432109876543210"),
                    None,
                    Decimal("-12345678901234567890"),
                    Decimal("0"),
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(24, 0)),
            ),
            pd.array(
                [
                    Decimal("98765432109877530864.32109876543210"),
                    None,
                    Decimal("-12345678901234691346.7890123456"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            id="array-array-larger_scale-unsafe",
        ),
        pytest.param(
            pa.scalar(Decimal("0.99"), pa.decimal128(4, 2)),
            pd.array(
                [Decimal("1.99"), Decimal("99.99"), Decimal("-99.99"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(4, 2)),
            ),
            pd.array(
                [Decimal("2.98"), Decimal("100.98"), Decimal("-99"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(5, 2)),
            ),
            id="scalar-array-same_scale-safe",
        ),
        pytest.param(
            pd.array(
                [Decimal("15"), Decimal("-99"), Decimal("56"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(2, 0)),
            ),
            pa.scalar(Decimal("98.12"), pa.decimal128(4, 2)),
            pd.array(
                [Decimal("113.12"), Decimal("-0.88"), Decimal("154.12"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(5, 2)),
            ),
            id="array-scalar-smaller_scale-safe",
        ),
        pytest.param(
            pd.array(
                [Decimal("1.2345"), Decimal("-9.9999"), Decimal("5.4321"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(5, 4)),
            ),
            pa.scalar(Decimal("999.5"), pa.decimal128(4, 1)),
            pd.array(
                [Decimal("1000.7345"), Decimal("989.5001"), Decimal("1004.9321"), None]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(8, 4)),
            ),
            id="array-scalar-larger_scale-safe",
        ),
        pytest.param(
            pd.array(
                [Decimal("9" * 36), Decimal("-" + "9" * 36), Decimal("5" * 36), None]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            pa.scalar(Decimal("9" * 36), pa.decimal128(38, 0)),
            pd.array(
                [
                    Decimal("1" + "9" * 35 + "8"),
                    Decimal("0"),
                    Decimal("1" + "5" * 35 + "4"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            id="array-scalar-same_scale-unsafe",
        ),
        pytest.param(
            pa.scalar(Decimal("123456789"), pa.decimal128(38, 0)),
            pd.array(
                [
                    Decimal("9" * 20 + ".25"),
                    Decimal("9" * 25 + ".5"),
                    Decimal("9" * 30 + ".75"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 2)),
            ),
            pd.array(
                [
                    Decimal("100000000000123456788.25"),
                    Decimal("10000000000000000123456788.5"),
                    Decimal("1000000000000000000000123456788.75"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 2)),
            ),
            id="scalar-array-smaller_scale-unsafe",
        ),
        pytest.param(
            pa.scalar(Decimal("987654321.123456789"), pa.decimal128(38, 18)),
            pd.array(
                [
                    Decimal("9" * 10 + ".3"),
                    Decimal("9" * 14 + ".4"),
                    Decimal("9" * 18 + ".5"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 6)),
            ),
            pd.array(
                [
                    Decimal("10987654320.423456789"),
                    Decimal("100000987654320.523456789"),
                    Decimal("1000000000987654320.623456789"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            id="scalar-array-larger_scale-unsafe",
        ),
        pytest.param(
            pa.scalar(Decimal("1" + "5" * 36), pa.decimal128(38, 0)),
            pa.scalar(Decimal("-1" + "3" * 36), pa.decimal128(38, 1)),
            pa.scalar(Decimal("2" * 36), pa.decimal128(38, 1)),
            id="scalar-scalar-unsafe_rescale_edgecase",
        ),
        pytest.param(
            pd.array(
                [Decimal("10987654320.423456789"), None] * 5,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            None,
            pd.array(
                [None] * 10,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            id="array-null",
        ),
        pytest.param(
            None,
            pd.array(
                [Decimal("10987654320.423456789"), None] * 5,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            pd.array(
                [None] * 10,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            id="null-array",
        ),
    ],
)
def test_decimal_addition(arg0, arg1, answer, memory_leak_check):
    """Test adding decimals"""

    def impl(arg0, arg1):
        return bodosql.kernels.add_numeric(arg0, arg1)

    check_func(impl, (arg0, arg1), py_output=answer)


@pytest.mark.parametrize(
    "arg0, arg1",
    [
        pytest.param(
            pa.scalar(Decimal("87654321.87654321"), pa.decimal128(38, 30)),
            pa.scalar(Decimal("87654321.87654321"), pa.decimal128(38, 30)),
            id="scalar-scalar-same_scale-overflow_on_add",
        ),
        pytest.param(
            pa.scalar(Decimal("87654321.87654321"), pa.decimal128(38, 30)),
            pa.scalar(Decimal("800000000000000001"), pa.decimal128(38, 20)),
            id="scalar-scalar-larger_scale-overflow_on_rescale",
        ),
        pytest.param(
            pa.scalar(Decimal("800000000000000001"), pa.decimal128(38, 20)),
            pa.scalar(Decimal("87654321.87654321"), pa.decimal128(38, 30)),
            id="scalar-scalar-smaller_scale-overflow_on_rescale",
        ),
        pytest.param(
            pd.array(
                [Decimal("87654321.87654321"), None] * 3,
                dtype=pd.ArrowDtype(pa.decimal128(38, 30)),
            ),
            pd.array(
                [Decimal("87654321.87654321"), None] * 3,
                dtype=pd.ArrowDtype(pa.decimal128(38, 30)),
            ),
            id="array-array-same_scale-overflow_on_add",
        ),
        pytest.param(
            pd.array(
                [Decimal("800000000000000001"), None] * 3,
                dtype=pd.ArrowDtype(pa.decimal128(38, 20)),
            ),
            pd.array(
                [Decimal("87654321.87654321"), None] * 3,
                dtype=pd.ArrowDtype(pa.decimal128(38, 30)),
            ),
            id="scalar-scalar-smaller_scale-overflow_on_rescale",
        ),
    ],
)
def test_decimal_addition_error(arg0, arg1):
    """Test adding decimals in ways that will cause error"""

    def impl(arg0, arg1):
        return bodosql.kernels.add_numeric(arg0, arg1)

    with pytest.raises(ValueError, match="Number out of representable range"):
        check_func(impl, (arg0, arg1), py_output=-1)


@pytest.mark.parametrize(
    "arg0, arg1, answer",
    [
        pytest.param(
            pa.scalar(Decimal("104.68"), pa.decimal128(5, 2)),
            pa.scalar(Decimal("85.23"), pa.decimal128(4, 2)),
            pa.scalar(Decimal("19.45"), pa.decimal128(4, 2)),
            id="scalar-scalar-smaller_scale-fast",
        ),
        pytest.param(
            pa.scalar(Decimal("559.45"), pa.decimal128(6, 2)),
            pa.scalar(Decimal("543"), pa.decimal128(6, 2)),
            pa.scalar(Decimal("16.45"), pa.decimal128(6, 2)),
            id="scalar-scalar-same_scale-fast",
        ),
        pytest.param(
            pa.scalar(Decimal("1.23456"), pa.decimal128(6, 5)),
            pa.scalar(Decimal("9876543211.23456"), pa.decimal128(16, 5)),
            pa.scalar(Decimal("-9876543210"), pa.decimal128(10, 0)),
            id="scalar-scalar-larger_scale-fast",
        ),
        pytest.param(
            pa.scalar(Decimal("-65.78"), pa.decimal128(38, 2)),
            pa.scalar(Decimal("-85.23"), pa.decimal128(38, 2)),
            pa.scalar(Decimal("19.45"), pa.decimal128(38, 2)),
            id="scalar-scalar-same_scale-unsafe",
        ),
        pytest.param(
            pa.scalar(
                Decimal("123456789012345678901234567890.12"), pa.decimal128(38, 2)
            ),
            pa.scalar(Decimal("123456789012345678901234567890"), pa.decimal128(38, 0)),
            pa.scalar(Decimal("00.12"), pa.decimal128(4, 2)),
            id="scalar-scalar-smaller_scale-unsafe",
        ),
        pytest.param(
            pa.scalar(Decimal("112112308777.7532123454321"), pa.decimal128(38, 18)),
            pa.scalar(Decimal("112112321123.4321123454321"), pa.decimal128(30, 18)),
            pa.scalar(Decimal("-12345.6789"), pa.decimal128(38, 18)),
            id="scalar-scalar-larger_scale-unsafe",
        ),
        pytest.param(
            pd.array(
                [Decimal("-11.45"), None, Decimal("10.00"), Decimal("1.23")] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(4, 2)),
            ),
            pd.array(
                [Decimal("-5.00"), Decimal("4.56"), Decimal("9.99"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(4, 2)),
            ),
            pd.array(
                [Decimal("-6.45"), None, Decimal("0.01"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(4, 2)),
            ),
            id="array-array-same_scale-safe",
        ),
        pytest.param(
            pd.array(
                [Decimal("1003.45"), None, Decimal("-11.99"), Decimal("1.23")] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(6, 2)),
            ),
            pd.array(
                [Decimal("994"), Decimal("456"), Decimal("-12"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(3, 0)),
            ),
            pd.array(
                [Decimal("9.45"), None, Decimal("0.01"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(6, 2)),
            ),
            id="array-array-smaller_scale-safe",
        ),
        pytest.param(
            pd.array(
                [Decimal("10001.134"), None, Decimal("-10009.001"), Decimal("0")] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(8, 3)),
            ),
            pd.array(
                [Decimal("1.234"), Decimal("5.678"), Decimal("-9.101"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(4, 3)),
            ),
            pd.array(
                [Decimal("9999.9"), None, Decimal("-9999.9"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(8, 3)),
            ),
            id="array-array-larger_scale-safe",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("109877654220987765421"),
                    None,
                    Decimal("-2000000000000000000000000000000000075"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            pd.array(
                [
                    Decimal("12345678901234567890"),
                    Decimal("-1024"),
                    Decimal("1000000000000000000000000000000000001"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            pd.array(
                [
                    Decimal("97531975319753197531"),
                    None,
                    Decimal("-3000000000000000000000000000000000076"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            id="array-array-same_scale-unsafe",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("109877654220987764"),
                    None,
                    Decimal("115802467135802467.012345678876543211"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            pd.array(
                [
                    Decimal("12345678901234567"),
                    None,
                    Decimal("123456789123456789"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            pd.array(
                [
                    Decimal("97531975319753197"),
                    None,
                    Decimal("-7654321987654321.987654321123456789"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            id="array-array-smaller_scale-unsafe",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("98765432109877530864.32109876543210"),
                    None,
                    Decimal("-12345678901234691346.7890123456"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            pd.array(
                [
                    Decimal("987654.32109876543210"),
                    Decimal("3.1415926"),
                    Decimal("-123456.7890123456"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(24, 18)),
            ),
            pd.array(
                [
                    Decimal("98765432109876543210"),
                    None,
                    Decimal("-12345678901234567890"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            id="array-array-larger_scale-unsafe",
        ),
        pytest.param(
            pd.array(
                [Decimal("2.98"), Decimal("100.98"), Decimal("-99"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(5, 2)),
            ),
            pa.scalar(Decimal("0.99"), pa.decimal128(5, 2)),
            pd.array(
                [Decimal("1.99"), Decimal("99.99"), Decimal("-99.99"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(5, 2)),
            ),
            id="array-scalar-same_scale-safe",
        ),
        pytest.param(
            pd.array(
                [Decimal("113.12"), Decimal("-0.88"), Decimal("154.12"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(5, 2)),
            ),
            pa.scalar(Decimal("99"), pa.decimal128(2, 0)),
            pd.array(
                [Decimal("14.12"), Decimal("-99.88"), Decimal("55.12"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(5, 2)),
            ),
            id="array-scalar-smaller_scale-safe",
        ),
        pytest.param(
            pd.array(
                [Decimal("1000.7345"), Decimal("989.5001"), Decimal("1004.9321"), None]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(8, 4)),
            ),
            pa.scalar(Decimal("1.2345"), pa.decimal128(5, 4)),
            pd.array(
                [Decimal("999.5"), Decimal("988.2656"), Decimal("1003.6976"), None] * 2,
                dtype=pd.ArrowDtype(pa.decimal128(8, 4)),
            ),
            id="array-scalar-larger_scale-safe",
        ),
        pytest.param(
            pd.array(
                [Decimal("9" * 36), Decimal("-" + "9" * 36), Decimal("5" * 36), None]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            pa.scalar(Decimal("-" + "9" * 36), pa.decimal128(38, 0)),
            pd.array(
                [
                    Decimal("1" + "9" * 35 + "8"),
                    Decimal("0"),
                    Decimal("1" + "5" * 35 + "4"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            id="array-scalar-same_scale-unsafe",
        ),
        pytest.param(
            pa.scalar(Decimal("123456789"), pa.decimal128(38, 0)),
            pd.array(
                [
                    Decimal("-" + "9" * 20 + ".25"),
                    Decimal("-" + "9" * 25 + ".5"),
                    Decimal("-" + "9" * 30 + ".75"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 2)),
            ),
            pd.array(
                [
                    Decimal("100000000000123456788.25"),
                    Decimal("10000000000000000123456788.5"),
                    Decimal("1000000000000000000000123456788.75"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 2)),
            ),
            id="scalar-array-smaller_scale-unsafe",
        ),
        pytest.param(
            pa.scalar(Decimal("987654321.123456789"), pa.decimal128(38, 18)),
            pd.array(
                [
                    Decimal("-" + "9" * 10 + ".3"),
                    Decimal("-" + "9" * 14 + ".4"),
                    Decimal("-" + "9" * 18 + ".5"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 6)),
            ),
            pd.array(
                [
                    Decimal("10987654320.423456789"),
                    Decimal("100000987654320.523456789"),
                    Decimal("1000000000987654320.623456789"),
                    None,
                ]
                * 2,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            id="scalar-array-larger_scale-unsafe",
        ),
        pytest.param(
            pa.scalar(Decimal("1" + "5" * 36), pa.decimal128(38, 0)),
            pa.scalar(Decimal("1" + "3" * 36), pa.decimal128(38, 1)),
            pa.scalar(Decimal("2" * 36), pa.decimal128(38, 1)),
            id="scalar-scalar-unsafe_rescale_edgecase",
        ),
        pytest.param(
            pd.array(
                [Decimal("10987654320.423456789"), None] * 5,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            None,
            pd.array(
                [None] * 10,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            id="array-null",
        ),
        pytest.param(
            None,
            pd.array(
                [Decimal("10987654320.423456789"), None] * 5,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            pd.array(
                [None] * 10,
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            id="null-array",
        ),
    ],
)
def test_decimal_subtraction(arg0, arg1, answer, memory_leak_check):
    """Test subtracting decimals"""

    def impl(arg0, arg1):
        return bodosql.kernels.subtract_numeric(arg0, arg1)

    check_func(impl, (arg0, arg1), py_output=answer)


@pytest.mark.parametrize(
    "arg0, arg1",
    [
        pytest.param(
            pa.scalar(Decimal("87654321.87654321"), pa.decimal128(38, 30)),
            pa.scalar(Decimal("-87654321.87654321"), pa.decimal128(38, 30)),
            id="scalar-scalar-same_scale-overflow_on_add",
        ),
        pytest.param(
            pa.scalar(Decimal("87654321.87654321"), pa.decimal128(38, 30)),
            pa.scalar(Decimal("-800000000000000001"), pa.decimal128(38, 20)),
            id="scalar-scalar-larger_scale-overflow_on_rescale",
        ),
        pytest.param(
            pa.scalar(Decimal("800000000000000001"), pa.decimal128(38, 20)),
            pa.scalar(Decimal("-87654321.87654321"), pa.decimal128(38, 30)),
            id="scalar-scalar-smaller_scale-overflow_on_rescale",
        ),
        pytest.param(
            pd.array(
                [Decimal("87654321.87654321"), None] * 3,
                dtype=pd.ArrowDtype(pa.decimal128(38, 30)),
            ),
            pd.array(
                [Decimal("-87654321.87654321"), None] * 3,
                dtype=pd.ArrowDtype(pa.decimal128(38, 30)),
            ),
            id="array-array-same_scale-overflow_on_add",
        ),
        pytest.param(
            pd.array(
                [Decimal("800000000000000001"), None] * 3,
                dtype=pd.ArrowDtype(pa.decimal128(38, 20)),
            ),
            pd.array(
                [Decimal("-87654321.87654321"), None] * 3,
                dtype=pd.ArrowDtype(pa.decimal128(38, 30)),
            ),
            id="scalar-scalar-smaller_scale-overflow_on_rescale",
        ),
    ],
)
def test_decimal_subtraction_error(arg0, arg1):
    """Test subtracting decimals in ways that will cause error"""

    def impl(arg0, arg1):
        return bodosql.kernels.subtract_numeric(arg0, arg1)

    with pytest.raises(ValueError, match="Number out of representable range"):
        check_func(impl, (arg0, arg1), py_output=-1)


@pytest.mark.parametrize(
    "arg1, arg2, expected",
    [
        pytest.param(
            pd.array(
                [
                    "1",
                    "1.55",
                    "1.56",
                    "10.56",
                    "1000.5",
                    None,
                    None,
                    "10004.1",
                    "-11.41",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            pd.array(
                [
                    "1",
                    "1.55",
                    "1.56",
                    "10.56",
                    "1000.5",
                    None,
                    None,
                    "10004.1",
                    "-11.41",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            pd.array(
                [
                    "1",
                    "2.4025",
                    "2.4336",
                    "111.5136",
                    "1001000.25",
                    None,
                    None,
                    "100082016.81",
                    "130.1881",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 6)),
            ),
            id="array_array",
        ),
        pytest.param(
            pd.array(
                [
                    "1",
                    "1.55",
                    "1.56",
                    "10.56",
                    "1000.5",
                    None,
                    None,
                    "10004.1",
                    "-11.41",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            pa.scalar(Decimal("2"), pa.decimal128(4, 2)),
            pd.array(
                [
                    "2.0",
                    "3.1",
                    "3.12",
                    "21.12",
                    "2001.0",
                    None,
                    None,
                    "20008.2",
                    "-22.82",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 8)),
            ),
            id="array_scalar",
        ),
        pytest.param(
            pa.scalar(Decimal("2"), pa.decimal128(4, 2)),
            pd.array(
                [
                    "1",
                    "1.55",
                    "1.56",
                    "10.56",
                    "1000.5",
                    None,
                    None,
                    "10004.1",
                    "-11.41",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            pd.array(
                [
                    "2.0",
                    "3.1",
                    "3.12",
                    "21.12",
                    "2001.0",
                    None,
                    None,
                    "20008.2",
                    "-22.82",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 8)),
            ),
            id="scalar_array",
        ),
        pytest.param(
            pa.scalar(Decimal("2.3"), pa.decimal128(4, 2)),
            pa.scalar(Decimal("4.5"), pa.decimal128(4, 2)),
            pa.scalar(Decimal("10.3500"), pa.decimal128(8, 4)),
            id="scalar_scalar",
        ),
    ],
)
def test_decimal_array_multiplication(arg1, arg2, expected, memory_leak_check):
    def impl(arr1, arr2):
        return bodosql.kernels.numeric_array_kernels.multiply_decimals(arr1, arr2)

    check_func(impl, (arg1, arg2), py_output=expected)


def test_decimal_array_multiplication_overflow_handling():
    """
    Test that an appropriate error is raised when one or more
    decimal multiplications overflow.
    """

    @bodo.jit(distributed=["arr1", "arr2"])
    def impl(arr1, arr2):
        return bodosql.kernels.numeric_array_kernels.multiply_decimals(arr1, arr2)

    arr1 = pd.array(
        [
            "1",
            "99.9999999",  # This will overflow
            None,
            None,
            "99.9999999",
            "1",  # This won't overflow
        ],
        dtype=pd.ArrowDtype(pa.decimal128(38, 36)),
    )
    arr2 = pd.array(
        [
            "1",
            "99.9999999",
            None,
            None,
            "99.9999999",
            "1",
        ],
        dtype=pd.ArrowDtype(pa.decimal128(38, 36)),
    )

    with pytest.raises(ValueError, match="Number out of representable range"):
        out = impl(arr1, arr2)
        print(out)

    arr1 = Decimal("99999999999999.9999999")
    arr2 = Decimal("99999999999999.9999999")
    with pytest.raises(ValueError, match="Number out of representable range"):
        out = impl(arr1, arr2)
        print(out)


@pytest.mark.parametrize(
    "arg1, arg2, expected",
    [
        pytest.param(
            pd.array(
                [
                    "1",
                    "1.55",
                    "1.56",
                    "10.56",
                    "1000.5",
                    None,
                    None,
                    "10004.1",
                    "-11.41",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            pd.array(
                [
                    "2.1",
                    "0.005",
                    "2.4336",
                    "0.000001",
                    "1001000.25",
                    "1.4",
                    None,
                    "100082016.81",
                    "130.1881",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 6)),
            ),
            pd.array(
                [
                    "0.47619048",
                    "310.00000000",
                    "0.64102564",
                    "10560000.00000000",
                    "0.00099950",
                    None,
                    None,
                    "0.00009996",
                    "-0.08764242",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 8)),
            ),
            id="array_array",
        ),
        pytest.param(
            pd.array(
                [
                    "1",
                    "1.55",
                    "1.56",
                    "10.56",
                    "1000.5",
                    None,
                    None,
                    "10004.1",
                    "-11.41",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            pa.scalar(Decimal("2"), pa.decimal128(4, 2)),
            pd.array(
                [
                    "0.50000000",
                    "0.77500000",
                    "0.78000000",
                    "5.28000000",
                    "500.25000000",
                    None,
                    None,
                    "5002.05000000",
                    "-5.70500000",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 8)),
            ),
            id="array_scalar",
        ),
        pytest.param(
            pa.scalar(Decimal("2"), pa.decimal128(4, 2)),
            pd.array(
                [
                    "1",
                    "1.55",
                    "1.56",
                    "10.56",
                    "1000.5",
                    None,
                    None,
                    "10004.1",
                    "-11.41",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            pd.array(
                [
                    "2.00000000",
                    "1.29032258",
                    "1.28205128",
                    "0.18939394",
                    "0.00199900",
                    None,
                    None,
                    "0.00019992",
                    "-0.17528484",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 8)),
            ),
            id="scalar_array",
        ),
        pytest.param(
            pa.scalar(Decimal("-2.12"), pa.decimal128(4, 2)),
            pa.scalar(Decimal("-1.345"), pa.decimal128(5, 3)),
            pa.scalar(Decimal("1.57620818"), pa.decimal128(13, 8)),
            id="scalar_scalar",
        ),
    ],
)
def test_decimal_array_division(arg1, arg2, expected, memory_leak_check):
    """Test decimal division"""

    def impl(arr1, arr2):
        return bodosql.kernels.numeric_array_kernels.divide_decimals(arr1, arr2)

    check_func(impl, (arg1, arg2), py_output=expected)


@pytest.mark.parametrize(
    "arg1, arg2, expected",
    [
        pytest.param(
            pd.array(
                [
                    "1",
                    "1.55",
                    "1.56",
                    "10.56",
                    "1000.5",
                    None,
                    None,
                    "10004.1",
                    "-11.41",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            pd.array(
                [
                    "2.1",
                    "0",
                    "2.4336",
                    "0",
                    "1001000.25",
                    "0",
                    None,
                    "100082016.81",
                    "130.1881",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 6)),
            ),
            pd.array(
                [
                    "0.47619048",
                    "0",
                    "0.64102564",
                    "0",
                    "0.00099950",
                    None,
                    None,
                    "0.00009996",
                    "-0.08764242",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 8)),
            ),
            id="array_array",
        ),
        pytest.param(
            pd.array(
                [
                    "1",
                    "1.55",
                    "1.56",
                    "10.56",
                    "1000.5",
                    None,
                    None,
                    "10004.1",
                    "-11.41",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            pa.scalar(Decimal("0"), pa.decimal128(4, 2)),
            pd.array(
                [
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    None,
                    None,
                    "0",
                    "0",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 8)),
            ),
            id="array_scalar",
        ),
        pytest.param(
            pa.scalar(Decimal("2"), pa.decimal128(4, 2)),
            pd.array(
                [
                    "1",
                    "0",
                    "1.56",
                    "10.56",
                    "0",
                    None,
                    None,
                    "10004.1",
                    "-11.41",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            pd.array(
                [
                    "2.00000000",
                    "0",
                    "1.28205128",
                    "0.18939394",
                    "0",
                    None,
                    None,
                    "0.00019992",
                    "-0.17528484",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 8)),
            ),
            id="scalar_array",
        ),
        pytest.param(
            pa.scalar(Decimal("-2.12"), pa.decimal128(4, 2)),
            pa.scalar(Decimal("0"), pa.decimal128(5, 3)),
            pa.scalar(Decimal("0"), pa.decimal128(13, 8)),
            id="scalar_scalar",
        ),
        pytest.param(
            pa.scalar(Decimal("-2.12"), pa.decimal128(4, 2)),
            0,
            pa.scalar(Decimal("0"), pa.decimal128(13, 8)),
            id="decimal-int",
        ),
        pytest.param(
            pd.array(
                [
                    "1",
                    "1.55",
                    None,
                    "10004.1",
                    "-11.41",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            pd.array(
                [
                    2,
                    0,
                    6,
                    None,
                    3,
                ],
            ),
            pd.array(
                [
                    "0.5",
                    "0",
                    None,
                    None,
                    "-3.80333333",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 8)),
            ),
            id="decimal-int-array",
        ),
        pytest.param(
            pa.scalar(Decimal("-2.12"), pa.decimal128(4, 2)),
            0.00,
            0,
            id="decimal_float",
        ),
        pytest.param(
            pd.array(
                [
                    "1",
                    "1.55",
                    None,
                    "10004.1",
                    "-11.41",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            pd.array(
                [
                    2.1,
                    0,
                    1.23,
                    None,
                    130.1881,
                ],
            ),
            pd.array(
                [
                    0.47619048,
                    0,
                    None,
                    None,
                    -0.08764242,
                ],
            ),
            id="decimal-float-array",
        ),
    ],
)
def test_decimal_div0(arg1, arg2, expected, memory_leak_check):
    """Test decimal div0"""

    def impl(arr1, arr2):
        return bodosql.kernels.div0(arr1, arr2)

    check_func(impl, (arg1, arg2), py_output=expected)


@pytest_mark_one_rank
def test_decimal_array_division_error_handling():
    """
    Test that an appropriate error is raised when there is division by zero or overflow
    """

    @bodo.jit(distributed=["arr1", "arr2"])
    def impl(arr1, arr2):
        return bodosql.kernels.numeric_array_kernels.divide_decimals(arr1, arr2)

    arr1 = pd.array(
        [
            "1",
            "9999999999999999999.99",  # This will overflow
            None,
            None,
            "9999999.99",
            "1",
        ],
        dtype=pd.ArrowDtype(pa.decimal128(38, 2)),
    )
    arr2 = pd.array(
        [
            "1",
            "0.0000000000000001",
            None,
            None,
            "99.9999999",
            "1",
        ],
        dtype=pd.ArrowDtype(pa.decimal128(38, 36)),
    )

    with pytest.raises(ValueError, match="Number out of representable range"):
        out = impl(arr1, arr2)
        print(out)

    arr1 = pd.array(
        [
            "1",
            "9999999999999999999.99",
            None,
            None,
            "9999999.99",
            "1",
        ],
        dtype=pd.ArrowDtype(pa.decimal128(38, 2)),
    )
    arr2 = pd.array(
        [
            "1",
            "0.0",  # Division by zero
            None,
            None,
            "99.9999999",
            "1",
        ],
        dtype=pd.ArrowDtype(pa.decimal128(38, 36)),
    )

    with pytest.raises(RuntimeError, match="Decimal division by zero error"):
        out = impl(arr1, arr2)
        print(out)


@pytest.mark.parametrize(
    "arg, round_scale, answer",
    [
        # Scalar tests
        pytest.param(
            pa.scalar(
                Decimal("12345.1224567"),
                pa.decimal128(25, 7),
            ),
            2,
            pa.scalar(
                Decimal("12345.12"),
                pa.decimal128(26, 2),
            ),
            id="scalar-round_down",
        ),
        pytest.param(
            pa.scalar(
                Decimal("12345.1254567"),
                pa.decimal128(25, 7),
            ),
            2,
            pa.scalar(
                Decimal("12345.13"),
                pa.decimal128(26, 2),
            ),
            id="scalar-round_up",
        ),
        pytest.param(
            pa.scalar(
                Decimal("12345.12545"),
                pa.decimal128(25, 5),
            ),
            5,
            pa.scalar(
                Decimal("12345.12545"),
                pa.decimal128(25, 5),
            ),
            id="scalar-same_scale",
        ),
        pytest.param(
            pa.scalar(
                Decimal("12345.12545"),
                pa.decimal128(25, 5),
            ),
            8,
            pa.scalar(
                Decimal("12345.12545"),
                pa.decimal128(25, 5),
            ),
            id="scalar-larger_scale",
        ),
        pytest.param(
            pa.scalar(
                Decimal("12345.12545"),
                pa.decimal128(25, 5),
            ),
            0,
            pa.scalar(
                Decimal("12345"),
                pa.decimal128(25, 5),
            ),
            id="scalar-scale_to_zero",
        ),
        pytest.param(
            pa.scalar(
                Decimal("987654321.123456789"),
                pa.decimal128(25, 9),
            ),
            -1,
            pa.scalar(
                Decimal("987654320"),
                pa.decimal128(25, 0),
            ),
            id="scalar-negative_scale-round_down",
        ),
        pytest.param(
            pa.scalar(
                Decimal("987654321.123456789"),
                pa.decimal128(25, 9),
            ),
            -5,
            pa.scalar(
                Decimal("987700000"),
                pa.decimal128(25, 0),
            ),
            id="scalar-negative_scale-round_up",
        ),
        pytest.param(
            pa.scalar(
                Decimal("0"),
                pa.decimal128(25, 5),
            ),
            3,
            pa.scalar(
                Decimal("0"),
                pa.decimal128(25, 3),
            ),
            id="scalar-zero",
        ),
        pytest.param(
            pa.scalar(
                Decimal("0"),
                pa.decimal128(25, 5),
            ),
            8,
            pa.scalar(
                Decimal("0"),
                pa.decimal128(25, 5),
            ),
            id="scalar-zero-larger_scale",
        ),
        pytest.param(
            pa.scalar(
                Decimal("0"),
                pa.decimal128(25, 5),
            ),
            -5,
            pa.scalar(
                Decimal("0"),
                pa.decimal128(25, 0),
            ),
            id="scalar-zero-negative_scale",
        ),
        pytest.param(
            pa.scalar(
                Decimal("1"),
                pa.decimal128(1, 0),
            ),
            -5,
            pa.scalar(
                Decimal("0"),
                pa.decimal128(1, 0),
            ),
            id="scalar-scale_larger_than_precision-negative",
        ),
        pytest.param(
            pa.scalar(
                Decimal("9999999999"),
                pa.decimal128(10, 0),
            ),
            -1,
            pa.scalar(
                Decimal("10000000000"),
                pa.decimal128(11, 0),
            ),
            id="scalar-negative_round-propagate",
        ),
        pytest.param(
            pa.scalar(
                Decimal("9999999999.99999"),
                pa.decimal128(15, 5),
            ),
            4,
            pa.scalar(
                Decimal("10000000000.0000"),
                pa.decimal128(16, 4),
            ),
            id="scalar-round-propagate",
        ),
        pytest.param(
            pa.scalar(
                Decimal("0.00999"),
                pa.decimal128(5, 5),
            ),
            4,
            pa.scalar(
                Decimal("0.01"),
                pa.decimal128(6, 4),
            ),
            id="scalar-round-propagate-small",
        ),
        pytest.param(
            pa.scalar(
                Decimal(
                    "99999 99999 99999 99999. 99999 99999 99999 999".replace(" ", "")
                ),
                pa.decimal128(38, 18),
            ),
            17,
            pa.scalar(
                Decimal(
                    "1 00000 00000 00000 00000. 00000 00000 00000 00".replace(" ", "")
                ),
                pa.decimal128(38, 17),
            ),
            id="scalar-barely_fits",
        ),
        pytest.param(
            pa.scalar(
                Decimal("-12345.1224567"),
                pa.decimal128(25, 7),
            ),
            2,
            pa.scalar(
                Decimal("-12345.12"),
                pa.decimal128(26, 2),
            ),
            id="scalar-round_down-negative",
        ),
        pytest.param(
            pa.scalar(
                Decimal("-12345.1254567"),
                pa.decimal128(25, 7),
            ),
            2,
            pa.scalar(
                Decimal("-12345.13"),
                pa.decimal128(26, 2),
            ),
            id="scalar-round_up-negative",
        ),
        pytest.param(
            pa.scalar(
                Decimal("-987654321.123456789"),
                pa.decimal128(25, 9),
            ),
            -1,
            pa.scalar(
                Decimal("-987654320"),
                pa.decimal128(25, 0),
            ),
            id="scalar-negative_scale-round_down-negative",
        ),
        pytest.param(
            pa.scalar(
                Decimal("-987654321.123456789"),
                pa.decimal128(25, 9),
            ),
            -5,
            pa.scalar(
                Decimal("-987700000"),
                pa.decimal128(25, 0),
            ),
            id="scalar-negative_scale-round_up-negative",
        ),
        # Array tests
        pytest.param(
            pd.array(
                [
                    Decimal("12345.1225"),
                    Decimal("12345.1224567"),
                    Decimal("12345.1225"),
                    Decimal("12345.1224567"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(25, 7)),
            ),
            2,
            pd.array(
                [
                    Decimal("12345.12"),
                    Decimal("12345.12"),
                    Decimal("12345.12"),
                    Decimal("12345.12"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(26, 2)),
            ),
            id="array-round_down",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("12345.1254"),
                    None,
                    Decimal("12345.1264567"),
                    Decimal("12345.1254"),
                    None,
                    Decimal("12345.1264567"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(25, 7)),
            ),
            2,
            pd.array(
                [
                    Decimal("12345.13"),
                    None,
                    Decimal("12345.13"),
                    Decimal("12345.13"),
                    None,
                    Decimal("12345.13"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(26, 2)),
            ),
            id="array-round_up",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("12345.12245"),
                    Decimal("67891.122"),
                    Decimal("12345.12245"),
                    Decimal("67891.122"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(25, 5)),
            ),
            5,
            pd.array(
                [
                    Decimal("12345.12245"),
                    Decimal("67891.122"),
                    Decimal("12345.12245"),
                    Decimal("67891.122"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(26, 5)),
            ),
            id="array-round_same_scale",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("12345.12245"),
                    Decimal("67891.125"),
                    Decimal("12345.12245"),
                    Decimal("67891.125"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(25, 5)),
            ),
            10,
            pd.array(
                [
                    Decimal("12345.12245"),
                    Decimal("67891.125"),
                    Decimal("12345.12245"),
                    Decimal("67891.125"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(26, 5)),
            ),
            id="array-larger_scale",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("12345.12245123"),
                    Decimal("67891.122"),
                    Decimal("12345.12245123"),
                    Decimal("67891.122"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(25, 8)),
            ),
            0,
            pd.array(
                [
                    Decimal("12345"),
                    Decimal("67891"),
                    Decimal("12345"),
                    Decimal("67891"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(26, 0)),
            ),
            id="array-scale_to_zero",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("12344.12245123"),
                    Decimal("987654321.123456789"),
                    Decimal("12344.12245123"),
                    Decimal("987654321.123456789"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(25, 9)),
            ),
            -1,
            pd.array(
                [
                    Decimal("12340"),
                    Decimal("987654320"),
                    Decimal("12340"),
                    Decimal("987654320"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(26, 0)),
            ),
            id="array-negative_scale-round_down",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("12346.12245123"),
                    Decimal("987654329.123456789"),
                    Decimal("12346.12245123"),
                    Decimal("987654329.123456789"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(25, 9)),
            ),
            -1,
            pd.array(
                [
                    Decimal("12350"),
                    Decimal("987654330"),
                    Decimal("12350"),
                    Decimal("987654330"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(26, 0)),
            ),
            id="array-negative_scale-round_up",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("0"),
                    Decimal("0"),
                    Decimal("0"),
                    Decimal("0"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(25, 5)),
            ),
            3,
            pd.array(
                [
                    Decimal("0"),
                    Decimal("0"),
                    Decimal("0"),
                    Decimal("0"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(26, 3)),
            ),
            id="array-zero",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("0"),
                    Decimal("0"),
                    Decimal("0"),
                    Decimal("0"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(25, 5)),
            ),
            -3,
            pd.array(
                [
                    Decimal("0"),
                    Decimal("0"),
                    Decimal("0"),
                    Decimal("0"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(26, 0)),
            ),
            id="array-zero-negative_scale",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("9999999999"),
                    Decimal("99999"),
                    Decimal("99"),
                    Decimal("9"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(10, 0)),
            ),
            -1,
            pd.array(
                [
                    Decimal("10000000000"),
                    Decimal("100000"),
                    Decimal("100"),
                    Decimal("10"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(11, 0)),
            ),
            id="array-negative_round-propagate",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("9999999999.99999999"),
                    Decimal("99999.99999"),
                    Decimal("9999999999.99999999"),
                    Decimal("99999.99999"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(20, 8)),
            ),
            4,
            pd.array(
                [
                    Decimal("10000000000.0000"),
                    Decimal("100000.0000"),
                    Decimal("10000000000.0000"),
                    Decimal("100000.0000"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(21, 4)),
            ),
            id="array-positive_round-propagate",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("0.00999"),
                    Decimal("0.00999"),
                    Decimal("0.00999"),
                    Decimal("0.00999"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(21, 5)),
            ),
            4,
            pd.array(
                [
                    Decimal("0.01"),
                    Decimal("0.01"),
                    Decimal("0.01"),
                    Decimal("0.01"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(6, 4)),
            ),
            id="array-round-propagate-small",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("-12345.1225"),
                    Decimal("-12345.1224567"),
                    Decimal("-12345.1225"),
                    Decimal("-12345.1224567"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(25, 7)),
            ),
            2,
            pd.array(
                [
                    Decimal("-12345.12"),
                    Decimal("-12345.12"),
                    Decimal("-12345.12"),
                    Decimal("-12345.12"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(26, 2)),
            ),
            id="array-round_down-negative",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("-12345.1254"),
                    None,
                    Decimal("-12345.1264567"),
                    Decimal("-12345.1254"),
                    None,
                    Decimal("-12345.1264567"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(25, 7)),
            ),
            2,
            pd.array(
                [
                    Decimal("-12345.13"),
                    None,
                    Decimal("-12345.13"),
                    Decimal("-12345.13"),
                    None,
                    Decimal("-12345.13"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(26, 2)),
            ),
            id="array-round_up-negative",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("-12344.12245123"),
                    Decimal("-987654321.123456789"),
                    Decimal("-12344.12245123"),
                    Decimal("-987654321.123456789"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(25, 9)),
            ),
            -1,
            pd.array(
                [
                    Decimal("-12340"),
                    Decimal("-987654320"),
                    Decimal("-12340"),
                    Decimal("-987654320"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(26, 0)),
            ),
            id="array-negative_scale-round_down-negative",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("-12346.12245123"),
                    Decimal("-987654329.123456789"),
                    Decimal("-12346.12245123"),
                    Decimal("-987654329.123456789"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(25, 9)),
            ),
            -1,
            pd.array(
                [
                    Decimal("-12350"),
                    Decimal("-987654330"),
                    Decimal("-12350"),
                    Decimal("-987654330"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(26, 0)),
            ),
            id="array-negative_scale-round_up-negative",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("-2.5"),
                    Decimal("-3.5"),
                    Decimal("2.5"),
                    Decimal("3.5"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(25, 9)),
            ),
            0,
            pd.array(
                [
                    Decimal("-3"),
                    Decimal("-4"),
                    Decimal("3"),
                    Decimal("4"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(26, 0)),
            ),
            id="array-mixed_sign",
        ),
    ],
)
def test_round_decimal(arg, round_scale, answer, memory_leak_check):
    """
    Test rounding decimals.
    """

    def impl(arr):
        return bodosql.kernels.numeric_array_kernels.round_decimal(
            arr,
            round_scale,
        )

    check_func(impl, (arg,), py_output=answer)


@pytest.mark.parametrize(
    "arg, round_scale",
    [
        pytest.param(
            pd.array(
                [
                    Decimal(
                        "99999 99999 99999 99999 99999 99999 99999 999".replace(" ", "")
                    ),
                    Decimal(
                        "99999 99999 99999 99999 99999 99999 99999 999".replace(" ", "")
                    ),
                    Decimal(
                        "99999 99999 99999 99999 99999 99999 99999 999".replace(" ", "")
                    ),
                    Decimal(
                        "99999 99999 99999 99999 99999 99999 99999 999".replace(" ", "")
                    ),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            -1,
            id="array-all_overflow",
        ),
        pytest.param(
            pd.array(
                [
                    None,
                    Decimal(
                        "99999 99999 99999 99999 99999 99999 99999 999".replace(" ", "")
                    ),
                    Decimal("12345"),
                    Decimal("12345"),
                    Decimal("12345"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            -1,
            id="array-one_overflow",
        ),
        pytest.param(
            pd.array(
                [
                    None,
                    Decimal(
                        "-99999 99999 99999 99999 99999 99999 99999 999".replace(
                            " ", ""
                        )
                    ),
                    Decimal("12345"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            -1,
            id="array-one_overflow-negative",
        ),
        # Scalar tests
        pytest.param(
            pa.scalar(
                Decimal(
                    "99999 99999 99999 99999 99999 99999 99999 999".replace(" ", "")
                ),
                pa.decimal128(38, 0),
            ),
            -1,
            id="scalar-overflow-negative_scale",
        ),
        pytest.param(
            pa.scalar(
                Decimal(
                    "-99999 99999 99999 99999 99999 99999 99999 999".replace(" ", "")
                ),
                pa.decimal128(38, 0),
            ),
            -1,
            id="scalar-overflow-negative_scale-negative",
        ),
    ],
)
def test_round_decimal_overflow(arg, round_scale):
    """
    Test overflow in rounding decimals.
    """

    def impl(arr):
        return bodosql.kernels.numeric_array_kernels.round_decimal(
            arr,
            round_scale,
        )

    with pytest.raises(ValueError, match="Number out of representable range"):
        check_func(impl, (arg,))


@pytest.mark.parametrize(
    "arg, round_scale, answer",
    [
        # Scalar tests
        pytest.param(
            pa.scalar(
                Decimal("648.2935"),
                pa.decimal128(20, 5),
            ),
            2,
            pa.scalar(
                Decimal("648.30"),
                pa.decimal128(21, 2),
            ),
            id="scalar-positive_scale-positive",
        ),
        pytest.param(
            pa.scalar(
                Decimal("-152.5826"),
                pa.decimal128(20, 5),
            ),
            2,
            pa.scalar(
                Decimal("-152.58"),
                pa.decimal128(21, 2),
            ),
            id="scalar-positive_scale-negative",
        ),
        pytest.param(
            pa.scalar(
                Decimal("-0.15122"),
                pa.decimal128(20, 5),
            ),
            2,
            pa.scalar(
                Decimal("-0.15"),
                pa.decimal128(21, 2),
            ),
            id="scalar-positive_scale-close_to_zero",
        ),
        pytest.param(
            pa.scalar(
                Decimal("648.2935"),
                pa.decimal128(20, 5),
            ),
            -2,
            pa.scalar(
                Decimal("700"),
                pa.decimal128(21, 0),
            ),
            id="scalar-negative_scale-positive",
        ),
        pytest.param(
            pa.scalar(
                Decimal("-152.5826"),
                pa.decimal128(20, 5),
            ),
            -2,
            pa.scalar(
                Decimal("-100"),
                pa.decimal128(21, 0),
            ),
            id="scalar-negative_scale-negative",
        ),
        pytest.param(
            pa.scalar(
                Decimal("-0.15122"),
                pa.decimal128(20, 5),
            ),
            -2,
            pa.scalar(
                Decimal("0"),
                pa.decimal128(21, 0),
            ),
            id="scalar-negative_scale-close_to_zero",
        ),
        pytest.param(
            pa.scalar(
                Decimal("648.2935"),
                pa.decimal128(20, 5),
            ),
            0,
            pa.scalar(
                Decimal("649"),
                pa.decimal128(21, 0),
            ),
            id="scalar-zero_scale-positive",
        ),
        pytest.param(
            pa.scalar(
                Decimal("-152.5826"),
                pa.decimal128(20, 5),
            ),
            0,
            pa.scalar(
                Decimal("-152"),
                pa.decimal128(21, 0),
            ),
            id="scalar-zero_scale-negative",
        ),
        pytest.param(
            pa.scalar(
                Decimal("-0.15122"),
                pa.decimal128(20, 5),
            ),
            0,
            pa.scalar(
                Decimal("0"),
                pa.decimal128(21, 0),
            ),
            id="scalar-zero_scale-close_to_zero",
        ),
        # Array tests
        pytest.param(
            pd.array(
                [
                    Decimal("648.2935"),
                    Decimal("-152.5826"),
                    Decimal("-0.15122"),
                    Decimal("0.5233"),
                    Decimal("0"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(20, 5)),
            ),
            2,
            pd.array(
                [
                    Decimal("648.30"),
                    Decimal("-152.58"),
                    Decimal("-0.15"),
                    Decimal("0.53"),
                    Decimal("0"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(21, 2)),
            ),
            id="array-positive_scale",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("1231249024183219648.2935"),
                    Decimal("-1234921349252.5826"),
                    Decimal("-88888888888.15122340213482"),
                    Decimal("99999999999.523294234123433"),
                    Decimal("0"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            -5,
            pd.array(
                [
                    Decimal("1231249024183300000"),
                    Decimal("-1234921300000"),
                    Decimal("-88888800000"),
                    Decimal("100000000000"),
                    Decimal("0"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            id="array-stress_test",
        ),
    ],
)
def test_ceil_decimal(arg, round_scale, answer, memory_leak_check):
    """
    Test ceil decimals.
    """

    def impl(arr):
        return bodosql.kernels.ceil(
            arr,
            round_scale,
        )

    check_func(impl, (arg,), py_output=answer)


@pytest.mark.parametrize(
    "arg, round_scale",
    [
        pytest.param(
            pd.array(
                [
                    None,
                    Decimal(
                        "99999 99999 99999 99999 99999 99999 99999 999".replace(" ", "")
                    ),
                    Decimal("12345"),
                    Decimal("12345"),
                    Decimal("12345"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            -1,
            id="array-overflow",
        ),
        pytest.param(
            pa.scalar(
                Decimal(
                    "99999 99999 99999 99999 99999 99999 99999 999".replace(" ", "")
                ),
                pa.decimal128(38, 0),
            ),
            -1,
            id="scalar-overflow",
        ),
    ],
)
def test_ceil_decimal_overflow(arg, round_scale):
    """
    Test overflow in ceiling of decimals.
    """

    def impl(arr):
        return bodosql.kernels.ceil(
            arr,
            round_scale,
        )

    with pytest.raises(Exception, match="Number out of representable range"):
        check_func(impl, (arg,))


@pytest.mark.parametrize(
    "arg, round_scale, answer",
    [
        # Scalar tests
        pytest.param(
            pa.scalar(
                Decimal("648.2935"),
                pa.decimal128(20, 5),
            ),
            2,
            pa.scalar(
                Decimal("648.29"),
                pa.decimal128(21, 2),
            ),
            id="scalar-positive_scale-positive",
        ),
        pytest.param(
            pa.scalar(
                Decimal("-152.5826"),
                pa.decimal128(20, 5),
            ),
            2,
            pa.scalar(
                Decimal("-152.59"),
                pa.decimal128(21, 2),
            ),
            id="scalar-positive_scale-negative",
        ),
        pytest.param(
            pa.scalar(
                Decimal("-0.15122"),
                pa.decimal128(20, 5),
            ),
            2,
            pa.scalar(
                Decimal("-0.16"),
                pa.decimal128(21, 2),
            ),
            id="scalar-positive_scale-close_to_zero",
        ),
        pytest.param(
            pa.scalar(
                Decimal("648.2935"),
                pa.decimal128(20, 5),
            ),
            -2,
            pa.scalar(
                Decimal("600"),
                pa.decimal128(21, 0),
            ),
            id="scalar-negative_scale-positive",
        ),
        pytest.param(
            pa.scalar(
                Decimal("-152.5826"),
                pa.decimal128(20, 5),
            ),
            -2,
            pa.scalar(
                Decimal("-200"),
                pa.decimal128(21, 0),
            ),
            id="scalar-negative_scale-negative",
        ),
        pytest.param(
            pa.scalar(
                Decimal("-0.15122"),
                pa.decimal128(20, 5),
            ),
            -2,
            pa.scalar(
                Decimal("-100"),
                pa.decimal128(21, 0),
            ),
            id="scalar-negative_scale-close_to_zero",
        ),
        pytest.param(
            pa.scalar(
                Decimal("648.2935"),
                pa.decimal128(20, 5),
            ),
            0,
            pa.scalar(
                Decimal("648"),
                pa.decimal128(21, 0),
            ),
            id="scalar-zero_scale-positive",
        ),
        pytest.param(
            pa.scalar(
                Decimal("-152.5826"),
                pa.decimal128(20, 5),
            ),
            0,
            pa.scalar(
                Decimal("-153"),
                pa.decimal128(21, 0),
            ),
            id="scalar-zero_scale-negative",
        ),
        pytest.param(
            pa.scalar(
                Decimal("-0.15122"),
                pa.decimal128(20, 5),
            ),
            0,
            pa.scalar(
                Decimal("-1"),
                pa.decimal128(21, 0),
            ),
            id="scalar-zero_scale-close_to_zero",
        ),
        # Array tests
        pytest.param(
            pd.array(
                [
                    Decimal("648.2935"),
                    Decimal("-152.5826"),
                    Decimal("-0.15122"),
                    Decimal("0.5233"),
                    Decimal("0"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(20, 5)),
            ),
            2,
            pd.array(
                [
                    Decimal("648.29"),
                    Decimal("-152.59"),
                    Decimal("-0.16"),
                    Decimal("0.52"),
                    Decimal("0"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(21, 2)),
            ),
            id="array-positive_scale",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("1231249024183219648.2935"),
                    Decimal("-1234921349252.5826"),
                    Decimal("-88888888888.15122340213482"),
                    Decimal("-99999999999.523294234123433"),
                    Decimal("0"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            -2,
            pd.array(
                [
                    Decimal("1231249024183219600"),
                    Decimal("-1234921349300"),
                    Decimal("-88888888900"),
                    Decimal("-100000000000"),
                    Decimal("0"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            id="array-stress_test",
        ),
    ],
)
def test_floor_decimal(arg, round_scale, answer, memory_leak_check):
    """
    Test floor decimals.
    """

    def impl(arr):
        return bodosql.kernels.floor(
            arr,
            round_scale,
        )

    check_func(impl, (arg,), py_output=answer)


@pytest.mark.parametrize(
    "arg, round_scale",
    [
        pytest.param(
            pd.array(
                [
                    None,
                    Decimal(
                        "-99999 99999 99999 99999 99999 99999 99999 999".replace(
                            " ", ""
                        )
                    ),
                    Decimal("12345"),
                    Decimal("12345"),
                    Decimal("12345"),
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            -1,
            id="array-overflow",
        ),
        pytest.param(
            pa.scalar(
                Decimal(
                    "-99999 99999 99999 99999 99999 99999 99999 999".replace(" ", "")
                ),
                pa.decimal128(38, 0),
            ),
            -1,
            id="scalar-overflow",
        ),
    ],
)
def test_floor_decimal_overflow(arg, round_scale):
    """
    Test overflow in floor of decimals.
    """

    def impl(arr):
        return bodosql.kernels.floor(
            arr,
            round_scale,
        )

    with pytest.raises(Exception, match="Number out of representable range"):
        check_func(impl, (arg,))


@pytest.mark.parametrize(
    "arg, round_scale, answer",
    [
        # Scalar tests
        pytest.param(
            pa.scalar(
                Decimal("648.2935"),
                pa.decimal128(20, 5),
            ),
            2,
            pa.scalar(
                Decimal("648.29"),
                pa.decimal128(21, 2),
            ),
            id="scalar-positive_scale-positive",
        ),
        pytest.param(
            pa.scalar(
                Decimal("-152.5826"),
                pa.decimal128(20, 5),
            ),
            2,
            pa.scalar(
                Decimal("-152.58"),
                pa.decimal128(21, 2),
            ),
            id="scalar-positive_scale-negative",
        ),
        pytest.param(
            pa.scalar(
                Decimal("-0.15122"),
                pa.decimal128(20, 5),
            ),
            2,
            pa.scalar(
                Decimal("-0.15"),
                pa.decimal128(21, 2),
            ),
            id="scalar-positive_scale-close_to_zero",
        ),
        pytest.param(
            pa.scalar(
                Decimal("648.2935"),
                pa.decimal128(20, 5),
            ),
            -2,
            pa.scalar(
                Decimal("600"),
                pa.decimal128(21, 0),
            ),
            id="scalar-negative_scale-positive",
        ),
        pytest.param(
            pa.scalar(
                Decimal("-152.5826"),
                pa.decimal128(20, 5),
            ),
            -2,
            pa.scalar(
                Decimal("-100"),
                pa.decimal128(21, 0),
            ),
            id="scalar-negative_scale-negative",
        ),
        pytest.param(
            pa.scalar(
                Decimal("-0.15122"),
                pa.decimal128(20, 5),
            ),
            -2,
            pa.scalar(
                Decimal("0"),
                pa.decimal128(21, 0),
            ),
            id="scalar-negative_scale-close_to_zero",
        ),
        pytest.param(
            pa.scalar(
                Decimal("648.2935"),
                pa.decimal128(20, 5),
            ),
            0,
            pa.scalar(
                Decimal("648"),
                pa.decimal128(21, 0),
            ),
            id="scalar-zero_scale-positive",
        ),
        pytest.param(
            pa.scalar(
                Decimal("-152.5826"),
                pa.decimal128(20, 5),
            ),
            0,
            pa.scalar(
                Decimal("-152"),
                pa.decimal128(21, 0),
            ),
            id="scalar-zero_scale-negative",
        ),
        pytest.param(
            pa.scalar(
                Decimal("-0.15122"),
                pa.decimal128(20, 5),
            ),
            0,
            pa.scalar(
                Decimal("0"),
                pa.decimal128(21, 0),
            ),
            id="scalar-zero_scale-close_to_zero",
        ),
        pytest.param(
            pa.scalar(
                Decimal("-152.5826"),
                pa.decimal128(20, 5),
            ),
            5,
            pa.scalar(
                Decimal("-152.5826"),
                pa.decimal128(20, 5),
            ),
            id="scalar-no-change",
        ),
        # Array tests
        pytest.param(
            pd.array(
                [
                    Decimal("648.2935"),
                    Decimal("-152.5826"),
                    Decimal("-0.15122"),
                    Decimal("0.5233"),
                    Decimal("0"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(20, 5)),
            ),
            2,
            pd.array(
                [
                    Decimal("648.29"),
                    Decimal("-152.58"),
                    Decimal("-0.15"),
                    Decimal("0.52"),
                    Decimal("0"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(21, 2)),
            ),
            id="array-positive_scale",
        ),
        pytest.param(
            pd.array(
                [
                    Decimal("1231249024183219648.2935"),
                    Decimal("-1234921349252.5826"),
                    Decimal("-88888888888.15122340213482"),
                    Decimal("-99999999999.523294234123433"),
                    Decimal("0"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            -2,
            pd.array(
                [
                    Decimal("1231249024183219600"),
                    Decimal("-1234921349200"),
                    Decimal("-88888888800"),
                    Decimal("-99999999900"),
                    Decimal("0"),
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            id="array-stress_test",
        ),
    ],
)
def test_trunc_decimal(arg, round_scale, answer, memory_leak_check):
    """
    Test floor decimals.
    """

    def impl(arr):
        return bodosql.kernels.trunc(
            arr,
            round_scale,
        )

    check_func(impl, (arg,), py_output=answer)


@pytest.mark.parametrize(
    "arg1, arg2, expected",
    [
        pytest.param(
            pd.array(
                [
                    "1",
                    "1.55",
                    "1.56",
                    "10.56",
                    "1000.5",
                    None,
                    None,
                    "10004.1",
                    "-11.41",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            pd.array(
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    None,
                    None,
                    1.0,
                    1.0,
                ],
                dtype=pd.Float64Dtype(),
            ),
            pd.array(
                [
                    2,
                    2.55,
                    2.56,
                    11.56,
                    1001.5,
                    None,
                    None,
                    10005.1,
                    -10.41,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="array-decimal_first",
        ),
        pytest.param(
            pd.array(
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    None,
                    None,
                    1.0,
                    1.0,
                ],
                dtype=pd.Float64Dtype(),
            ),
            pd.array(
                [
                    "1",
                    "1.55",
                    "1.56",
                    "10.56",
                    "1000.5",
                    None,
                    None,
                    "10004.1",
                    "-11.41",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            pd.array(
                [
                    2,
                    2.55,
                    2.56,
                    11.56,
                    1001.5,
                    None,
                    None,
                    10005.1,
                    -10.41,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="array-float_first",
        ),
        pytest.param(
            pa.scalar(Decimal("15.50"), pa.decimal128(4, 2)),
            1.0,
            16.50,
            id="scalar-decimal_first",
        ),
        pytest.param(
            1.0,
            pa.scalar(Decimal("15.50"), pa.decimal128(4, 2)),
            16.50,
            id="scalar-float_first",
        ),
        pytest.param(
            1.0,
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            pd.array(
                [
                    3.5,
                    101.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="scalar-array-float_first",
        ),
        pytest.param(
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            1.0,
            pd.array(
                [
                    3.5,
                    101.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="array-scalar-decimal_first",
        ),
        pytest.param(
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            pa.scalar(Decimal("1.0"), pa.decimal128(2, 1)),
            pd.array(
                [
                    3.5,
                    101.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="array-scalar-float_first",
        ),
        pytest.param(
            pa.scalar(Decimal("1.0"), pa.decimal128(2, 1)),
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            pd.array(
                [
                    3.5,
                    101.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="scalar-array-decimal_first",
        ),
    ],
)
def test_decimal_array_float_addition(arg1, arg2, expected, memory_leak_check):
    """
    Tests decimal/float addition works correctly.
    """

    def impl(arr1, arr2):
        return bodosql.kernels.add_numeric(arr1, arr2)

    check_func(impl, (arg1, arg2), py_output=expected)


@pytest.mark.parametrize(
    "arg1, arg2, expected",
    [
        pytest.param(
            pd.array(
                [
                    "1",
                    "1.55",
                    "1.56",
                    "10.56",
                    "1000.5",
                    None,
                    None,
                    "10004.1",
                    "-11.41",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            pd.array(
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    None,
                    None,
                    1.0,
                    1.0,
                ],
                dtype=pd.Float64Dtype(),
            ),
            pd.array(
                [
                    0,
                    0.55,
                    0.56,
                    9.56,
                    999.5,
                    None,
                    None,
                    10003.1,
                    -12.41,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="array-decimal_first",
        ),
        pytest.param(
            pd.array(
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    None,
                    None,
                    1.0,
                    1.0,
                ],
                dtype=pd.Float64Dtype(),
            ),
            pd.array(
                [
                    "1",
                    "1.55",
                    "1.56",
                    "10.56",
                    "1000.5",
                    None,
                    None,
                    "10004.1",
                    "-11.41",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            pd.array(
                [
                    0,
                    -0.55,
                    -0.56,
                    -9.56,
                    -999.5,
                    None,
                    None,
                    -10003.1,
                    12.41,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="array-float_first",
        ),
        pytest.param(
            pa.scalar(Decimal("15.50"), pa.decimal128(4, 2)),
            1.0,
            14.50,
            id="scalar-decimal_first",
        ),
        pytest.param(
            1.0,
            pa.scalar(Decimal("15.50"), pa.decimal128(4, 2)),
            -14.50,
            id="scalar-float_first",
        ),
        pytest.param(
            1.0,
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            pd.array(
                [
                    -1.5,
                    -99.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="scalar-array-float_first",
        ),
        pytest.param(
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            1.0,
            pd.array(
                [
                    1.5,
                    99.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="array-scalar-decimal_first",
        ),
        pytest.param(
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            pa.scalar(Decimal("1.0"), pa.decimal128(2, 1)),
            pd.array(
                [
                    1.5,
                    99.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="array-scalar-float_first",
        ),
        pytest.param(
            pa.scalar(Decimal("1.0"), pa.decimal128(2, 1)),
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            pd.array(
                [
                    -1.5,
                    -99.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="scalar-array-decimal_first",
        ),
    ],
)
def test_decimal_array_float_subtraction(arg1, arg2, expected, memory_leak_check):
    """
    Tests decimal/float subtraction works correctly.
    """

    def impl(arr1, arr2):
        return bodosql.kernels.subtract_numeric(arr1, arr2)

    check_func(impl, (arg1, arg2), py_output=expected)


@pytest.mark.parametrize(
    "arg1, arg2, expected",
    [
        pytest.param(
            pd.array(
                [
                    "1",
                    "1.55",
                    "1.56",
                    "10.56",
                    "1000.5",
                    None,
                    None,
                    "10004.1",
                    "-11.41",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            pd.array(
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    None,
                    None,
                    1.0,
                    1.0,
                ],
                dtype=pd.Float64Dtype(),
            ),
            pd.array(
                [
                    1,
                    1.55,
                    1.56,
                    10.56,
                    1000.5,
                    None,
                    None,
                    10004.1,
                    -11.41,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="array-decimal_first",
        ),
        pytest.param(
            pd.array(
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    None,
                    None,
                    1.0,
                    1.0,
                ],
                dtype=pd.Float64Dtype(),
            ),
            pd.array(
                [
                    "1",
                    "1.55",
                    "1.56",
                    "10.56",
                    "1000.5",
                    None,
                    None,
                    "10004.1",
                    "-11.41",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            pd.array(
                [
                    1,
                    1.55,
                    1.56,
                    10.56,
                    1000.5,
                    None,
                    None,
                    10004.1,
                    -11.41,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="array-float_first",
        ),
        pytest.param(
            pa.scalar(Decimal("15.50"), pa.decimal128(4, 2)),
            1.0,
            15.50,
            id="scalar-decimal_first",
        ),
        pytest.param(
            1.0,
            pa.scalar(Decimal("15.50"), pa.decimal128(4, 2)),
            15.50,
            id="scalar-float_first",
        ),
        pytest.param(
            1.0,
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="scalar-array-float_first",
        ),
        pytest.param(
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            1.0,
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="array-scalar-decimal_first",
        ),
        pytest.param(
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            pa.scalar(Decimal("1.0"), pa.decimal128(2, 1)),
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="array-scalar-float_first",
        ),
        pytest.param(
            pa.scalar(Decimal("1.0"), pa.decimal128(2, 1)),
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="scalar-array-decimal_first",
        ),
    ],
)
def test_decimal_array_float_multiplication(arg1, arg2, expected, memory_leak_check):
    """
    Tests decimal/float multiplication works correctly.
    """

    def impl(arr1, arr2):
        return bodosql.kernels.multiply_numeric(arr1, arr2)

    check_func(impl, (arg1, arg2), py_output=expected)


@pytest.mark.parametrize(
    "arg1, arg2, expected",
    [
        pytest.param(
            pd.array(
                [
                    "1",
                    "1.55",
                    "1.56",
                    "10.56",
                    "1000.5",
                    None,
                    None,
                    "10004.1",
                    "-11.41",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            pd.array(
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    None,
                    None,
                    1.0,
                    1.0,
                ],
                dtype=pd.Float64Dtype(),
            ),
            pd.array(
                [
                    1,
                    1.55,
                    1.56,
                    10.56,
                    1000.5,
                    None,
                    None,
                    10004.1,
                    -11.41,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="array-decimal_first",
        ),
        pytest.param(
            pd.array(
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    None,
                    None,
                    1.0,
                    1.0,
                ],
                dtype=pd.Float64Dtype(),
            ),
            pd.array(
                [
                    "1",
                    "1.55",
                    "1.56",
                    "10.56",
                    "1000.5",
                    None,
                    None,
                    "10004.1",
                    "-11.41",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            pd.array(
                [
                    1 / 1,
                    1 / 1.55,
                    1 / 1.56,
                    1 / 10.56,
                    1 / 1000.5,
                    None,
                    None,
                    1 / 10004.1,
                    1 / -11.41,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="array-float_first",
        ),
        pytest.param(
            pa.scalar(Decimal("15.50"), pa.decimal128(4, 2)),
            1.0,
            15.50,
            id="scalar-decimal_first",
        ),
        pytest.param(
            1.0,
            pa.scalar(Decimal("15.50"), pa.decimal128(4, 2)),
            float(1 / 15.50),
            id="scalar-float_first",
        ),
        pytest.param(
            1.0,
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            pd.array(
                [
                    1 / 2.5,
                    1 / 100.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="scalar-array-float_first",
        ),
        pytest.param(
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            1.0,
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="array-scalar-decimal_first",
        ),
        pytest.param(
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            pa.scalar(Decimal("1.0"), pa.decimal128(2, 1)),
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="array-scalar-float_first",
        ),
        pytest.param(
            pa.scalar(Decimal("1.0"), pa.decimal128(2, 1)),
            pd.array(
                [
                    2.5,
                    100.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            pd.array(
                [
                    1 / 2.5,
                    1 / 100.25,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="scalar-array-decimal_first",
        ),
    ],
)
def test_decimal_array_float_division(arg1, arg2, expected, memory_leak_check):
    """
    Tests decimal/float division works correctly.
    """

    def impl(arr1, arr2):
        return bodosql.kernels.divide_numeric(arr1, arr2)

    check_func(impl, (arg1, arg2), py_output=expected)


@pytest.mark.parametrize(
    "arg1, arg2, expected",
    [
        pytest.param(
            pd.array(
                [
                    2,
                    2,
                    2,
                    2,
                    2,
                    None,
                    None,
                    2,
                    2,
                ],
                dtype=pd.Int32Dtype(),
            ),
            pd.array(
                [
                    "1",
                    "1.55",
                    "1.56",
                    "10.56",
                    "1000.5",
                    None,
                    None,
                    "10004.1",
                    "-11.41",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            pd.array(
                [
                    "2.00",
                    "3.10",
                    "3.12",
                    "21.12",
                    "2001.00",
                    None,
                    None,
                    "20008.20",
                    "-22.82",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            id="array-decimal_first",
        ),
        pytest.param(
            pd.array(
                [
                    "1",
                    "1.55",
                    "1.56",
                    "10.56",
                    "1000.5",
                    None,
                    None,
                    "10004.1",
                    "-11.41",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            pd.array(
                [
                    2,
                    2,
                    2,
                    2,
                    2,
                    None,
                    None,
                    2,
                    2,
                ],
                dtype=pd.Int32Dtype(),
            ),
            pd.array(
                [
                    "2.00",
                    "3.10",
                    "3.12",
                    "21.12",
                    "2001.00",
                    None,
                    None,
                    "20008.20",
                    "-22.82",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            id="array-decimal_first",
        ),
        pytest.param(
            pa.scalar(Decimal("1.5"), pa.decimal128(38, 2)),
            np.int32(2),
            pa.scalar(Decimal("3.0"), pa.decimal128(38, 2)),
            id="scalar-decimal_first",
        ),
        pytest.param(
            np.int32(2),
            pa.scalar(Decimal("1.5"), pa.decimal128(38, 2)),
            pa.scalar(Decimal("3.0"), pa.decimal128(38, 2)),
            id="scalar-int_first",
        ),
        pytest.param(
            np.int32(2),
            pd.array(
                ["1.0", "2.0", "3.0"],
                dtype=pd.ArrowDtype(pa.decimal128(38, 2)),
            ),
            pd.array(
                ["2.0", "4.0", "6.0"],
                dtype=pd.ArrowDtype(pa.decimal128(38, 2)),
            ),
            id="scalar-decimal-array",
        ),
        pytest.param(
            pa.scalar(Decimal("2.0"), pa.decimal128(38, 2)),
            pd.array(
                [1, 2, 3],
                dtype=pd.Int32Dtype(),
            ),
            pd.array(
                ["2.0", "4.0", "6.0"],
                dtype=pd.ArrowDtype(pa.decimal128(38, 2)),
            ),
            id="scalar-int-array",
        ),
    ],
)
def test_decimal_int_multiplication(arg1, arg2, expected):
    """
    Tests multiplication of integer and decimal arrays, through casting of
    integer into decimal.
    """

    def impl(a, b):
        return bodosql.kernels.multiply_numeric(a, b)

    check_func(impl, (arg1, arg2), py_output=expected)


@pytest.mark.parametrize(
    "arg1, arg2, expected",
    [
        pytest.param(
            pd.array(
                [
                    2,
                    2,
                    2,
                    None,
                    2,
                ],
                dtype=pd.Int32Dtype(),
            ),
            pd.array(
                ["1", "10", "100", None, "1000"],
                dtype=pd.ArrowDtype(pa.decimal128(22, 3)),
            ),
            pd.array(
                ["2", "0.2", "0.02", None, "0.002"],
                dtype=pd.ArrowDtype(pa.decimal128(22, 3)),
            ),
            id="array-decimal_first",
        ),
        pytest.param(
            pd.array(
                ["1", "10", "100", None, "1000"],
                dtype=pd.ArrowDtype(pa.decimal128(22, 3)),
            ),
            pd.array(
                [
                    2,
                    2,
                    2,
                    None,
                    2,
                ],
                dtype=pd.Int32Dtype(),
            ),
            pd.array(
                ["0.5", "5", "50", None, "500"],
                dtype=pd.ArrowDtype(pa.decimal128(22, 3)),
            ),
            id="array-decimal_first",
        ),
        pytest.param(
            pa.scalar(Decimal("1.5"), pa.decimal128(38, 2)),
            np.int32(2),
            pa.scalar(Decimal("0.75"), pa.decimal128(38, 8)),
            id="scalar-decimal_first",
        ),
        pytest.param(
            np.int32(2),
            pa.scalar(Decimal("2.0"), pa.decimal128(38, 2)),
            pa.scalar(Decimal("1.0"), pa.decimal128(18, 6)),
            id="scalar-int_first",
        ),
        pytest.param(
            np.int32(2),
            pd.array(
                ["2.0", "2.0", "2.0"],
                dtype=pd.ArrowDtype(pa.decimal128(38, 2)),
            ),
            pd.array(
                ["1.0", "1.0", "1.0"],
                dtype=pd.ArrowDtype(pa.decimal128(38, 2)),
            ),
            id="scalar-decimal-array",
        ),
        pytest.param(
            pa.scalar(Decimal("2.0"), pa.decimal128(38, 2)),
            pd.array(
                [2, 2, 2],
                dtype=pd.Int32Dtype(),
            ),
            pd.array(
                ["1.0", "1.0", "1.0"],
                dtype=pd.ArrowDtype(pa.decimal128(38, 2)),
            ),
            id="scalar-int-array",
        ),
    ],
)
def test_decimal_int_division(arg1, arg2, expected):
    """
    Tests multiplication of integer and decimal arrays, through casting of
    integer into decimal.
    """

    def impl(a, b):
        return bodosql.kernels.divide_numeric(a, b)

    check_func(impl, (arg1, arg2), py_output=expected)


@pytest.mark.parametrize(
    "arg1, arg2, expected",
    [
        pytest.param(
            pd.array(
                [
                    2,
                    2,
                    2,
                    None,
                    2,
                ],
                dtype=pd.Int32Dtype(),
            ),
            pd.array(
                ["-1", "-10", "-100", None, "-1000"],
                dtype=pd.ArrowDtype(pa.decimal128(22, 3)),
            ),
            pd.array(
                ["1", "-8", "-98", None, "-998"],
                dtype=pd.ArrowDtype(pa.decimal128(22, 3)),
            ),
            id="array-decimal_first",
        ),
        pytest.param(
            pd.array(
                ["1", "10", "100", None, "1000"],
                dtype=pd.ArrowDtype(pa.decimal128(22, 3)),
            ),
            pd.array(
                [
                    -2,
                    -2,
                    -2,
                    None,
                    -2,
                ],
                dtype=pd.Int32Dtype(),
            ),
            pd.array(
                ["-1", "8", "98", None, "998"],
                dtype=pd.ArrowDtype(pa.decimal128(22, 3)),
            ),
            id="array-decimal_first",
        ),
        pytest.param(
            pa.scalar(Decimal("3.25"), pa.decimal128(38, 2)),
            np.int32(-2),
            pa.scalar(Decimal("1.25"), pa.decimal128(38, 2)),
            id="scalar-decimal_first",
        ),
        pytest.param(
            np.int32(5),
            pa.scalar(Decimal("-1.5"), pa.decimal128(38, 2)),
            pa.scalar(Decimal("3.5"), pa.decimal128(38, 2)),
            id="scalar-int_first",
        ),
        pytest.param(
            np.int32(5),
            pd.array(
                ["-2.0", "-2.0", "-2.0"],
                dtype=pd.ArrowDtype(pa.decimal128(38, 2)),
            ),
            pd.array(
                ["3.0", "3.0", "3.0"],
                dtype=pd.ArrowDtype(pa.decimal128(38, 2)),
            ),
            id="scalar-decimal-array",
        ),
        pytest.param(
            pa.scalar(Decimal("5.5"), pa.decimal128(38, 2)),
            pd.array(
                [-2, -2, -2],
                dtype=pd.Int32Dtype(),
            ),
            pd.array(
                ["3.5", "3.5", "3.5"],
                dtype=pd.ArrowDtype(pa.decimal128(38, 2)),
            ),
            id="scalar-int-array",
        ),
    ],
)
def test_decimal_int_addition(arg1, arg2, expected):
    """
    Tests multiplication of integer and decimal arrays, through casting of
    integer into decimal.
    TODO: Not yet supported. Currently only casts the integer to decimal,
          then recalls add_numeric. Thus, will raise bodo.utils.typing.BodoError
    """

    def impl(a, b):
        return bodosql.kernels.add_numeric(a, b)

    check_func(impl, (arg1, arg2), py_output=expected)


@pytest.mark.parametrize(
    "arg1, arg2, expected",
    [
        pytest.param(
            pd.array(
                [
                    2,
                    2,
                    2,
                    None,
                    2,
                ],
                dtype=pd.Int32Dtype(),
            ),
            pd.array(
                ["1", "10", "100", None, "1000"],
                dtype=pd.ArrowDtype(pa.decimal128(22, 3)),
            ),
            pd.array(
                ["1", "-8", "-98", None, "-998"],
                dtype=pd.ArrowDtype(pa.decimal128(22, 3)),
            ),
            id="array-decimal_first",
        ),
        pytest.param(
            pd.array(
                ["1", "10", "100", None, "1000"],
                dtype=pd.ArrowDtype(pa.decimal128(22, 3)),
            ),
            pd.array(
                [
                    2,
                    2,
                    2,
                    None,
                    2,
                ],
                dtype=pd.Int32Dtype(),
            ),
            pd.array(
                ["-1", "8", "98", None, "998"],
                dtype=pd.ArrowDtype(pa.decimal128(22, 3)),
            ),
            id="array-decimal_first",
        ),
        pytest.param(
            pa.scalar(Decimal("3.25"), pa.decimal128(38, 2)),
            np.int32(2),
            pa.scalar(Decimal("1.25"), pa.decimal128(38, 2)),
            id="scalar-decimal_first",
        ),
        pytest.param(
            np.int32(5),
            pa.scalar(Decimal("1.5"), pa.decimal128(38, 2)),
            pa.scalar(Decimal("3.5"), pa.decimal128(38, 2)),
            id="scalar-int_first",
        ),
        pytest.param(
            np.int32(5),
            pd.array(
                ["2.0", "2.0", "2.0"],
                dtype=pd.ArrowDtype(pa.decimal128(38, 2)),
            ),
            pd.array(
                ["3.0", "3.0", "3.0"],
                dtype=pd.ArrowDtype(pa.decimal128(38, 2)),
            ),
            id="scalar-decimal-array",
        ),
        pytest.param(
            pa.scalar(Decimal("5.5"), pa.decimal128(38, 2)),
            pd.array(
                [2, 2, 2],
                dtype=pd.Int32Dtype(),
            ),
            pd.array(
                ["3.5", "3.5", "3.5"],
                dtype=pd.ArrowDtype(pa.decimal128(38, 2)),
            ),
            id="scalar-int-array",
        ),
    ],
)
def test_decimal_int_subtraction(arg1, arg2, expected):
    """
    Tests multiplication of integer and decimal arrays, through casting of
    integer into decimal.
    TODO: Not yet supported. Currently only casts the integer to decimal,
          then recalls add_numeric. Thus, will raise bodo.utils.typing.BodoError
    """

    def impl(a, b):
        return bodosql.kernels.subtract_numeric(a, b)

    check_func(impl, (arg1, arg2), py_output=expected)


@pytest.mark.parametrize(
    "arr, expected",
    [
        pytest.param(
            pd.array(
                [
                    "1",
                    "1.55",
                    "1.56",
                    "10.56",
                    "1000.5",
                    None,
                    None,
                    "10004.1",
                    "-11.41",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            pd.array(
                [
                    1,
                    1.55,
                    1.56,
                    10.56,
                    1000.5,
                    None,
                    None,
                    10004.1,
                    -11.41,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="basic",
        ),
        pytest.param(
            pd.array(
                [
                    "0.01",
                    "0.02",
                    "0.03",
                    "0.04",
                    "0.05",
                    None,
                    "0.07",
                    "0.08",
                    "0.09",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            pd.array(
                [
                    0.01,
                    0.02,
                    0.03,
                    0.04,
                    0.05,
                    None,
                    0.07,
                    0.08,
                    0.09,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="small_values",
        ),
        pytest.param(
            pd.array(
                [
                    "12345678",
                    "-12345678",
                    "0",
                    "999999999999",
                    "-999999999999",
                    None,
                    None,
                    "1123456789",
                    "-1987654321",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 0)),
            ),
            pd.array(
                [
                    12345678,
                    -12345678,
                    0,
                    999999999999,
                    -999999999999,
                    None,
                    None,
                    1123456789,
                    -1987654321,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="integer_values",
        ),
        pytest.param(
            pd.array(
                ["0", "0", "0", "0", None],
                dtype=pd.ArrowDtype(pa.decimal128(22, 0)),
            ),
            pd.array(
                [0, 0, 0, 0, None],
                dtype=pd.Float64Dtype(),
            ),
            id="zeroes",
        ),
        pytest.param(
            pd.array(
                [
                    "99999999999999999999999999999999999999",
                    "-99999999999999999999999999999999999999",
                    "99999999999999999999999999999999999999",
                    "-99999999999999999999999999999999999999",
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 0)),
            ),
            pd.array(
                [
                    99999999999999999999999999999999999999,
                    -99999999999999999999999999999999999999,
                    99999999999999999999999999999999999999,
                    -99999999999999999999999999999999999999,
                    None,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="max_values",
        ),
        pytest.param(
            pd.array(
                [
                    "9.9999999999999999999999999999999999999",
                    "9.9999999999999999999999999999999999999",
                    "9.9999999999999999999999999999999999999",
                    "9.9999999999999999999999999999999999999",
                    None,
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 37)),
            ),
            pd.array(
                [
                    9.9999999999999999999999999999999999999,
                    9.9999999999999999999999999999999999999,
                    9.9999999999999999999999999999999999999,
                    9.9999999999999999999999999999999999999,
                    None,
                ],
                dtype=pd.Float64Dtype(),
            ),
            id="max_values",
        ),
    ],
)
def test_decimal_to_float_array(arr, expected, memory_leak_check):
    """
    Test converting a decimal array to float.
    """

    def impl(arr):
        return bodosql.kernels.to_double(arr, None)

    check_func(impl, (arr,), py_output=expected)


def test_str_to_decimal_scalar(memory_leak_check):
    """
    Test converting a string scalar to decimal.
    """

    def impl(s):
        return bodosql.kernels.snowflake_conversion_array_kernels.string_to_decimal(
            s, 4, 2, True
        )

    check_func(
        impl, ("1.12",), py_output=pa.scalar(Decimal("1.12"), pa.decimal128(4, 2))
    )
    check_func(impl, ("E",), py_output=pa.scalar(Decimal("0"), pa.decimal128(4, 2)))
    check_func(impl, ("-E",), py_output=pa.scalar(Decimal("-0"), pa.decimal128(4, 2)))
    check_func(impl, ("+E",), py_output=pa.scalar(Decimal("+0"), pa.decimal128(4, 2)))
    check_func(
        impl, ("1.1E-1",), py_output=pa.scalar(Decimal("0.11"), pa.decimal128(4, 2))
    )
    check_func(
        impl, ("-1.1E",), py_output=pa.scalar(Decimal("-1.1"), pa.decimal128(4, 2))
    )
    # Check rounding scale
    check_func(
        impl, ("1.125",), py_output=pa.scalar(Decimal("1.13"), pa.decimal128(4, 2))
    )
    # Check for too large a leading digit
    check_func(impl, ("-100",), py_output=None)
    # Check for None -> None
    check_func(impl, (None,), py_output=None)


def test_str_to_decimal_array(memory_leak_check):
    """
    Test converting a string array to decimal.
    """

    def impl(arr):
        return bodosql.kernels.snowflake_conversion_array_kernels.string_to_decimal(
            arr, 4, 2, True
        )

    arr = pd.array(
        [
            "1.12",
            "E",
            "-E",
            "+E",
            "1.1E-1",
            "-1.1E",
            "1.125",
            "-100",
            None,
        ]
        * 2
    )
    py_output = pd.array(
        [
            "1.12",
            "0",
            "0",
            "0",
            "0.11",
            "-1.1",
            "1.13",
            None,
            None,
        ]
        * 2,
        dtype=pd.ArrowDtype(pa.decimal128(4, 2)),
    )
    check_func(impl, (arr,), py_output=py_output)


def test_str_to_decimal_scalar_edge_case(memory_leak_check):
    """
    Tests an edge case where using the full precision + scale doesn't
    fit in a decimal 128, but the scale can be rounded.
    """

    def impl(s):
        return bodosql.kernels.snowflake_conversion_array_kernels.string_to_decimal(
            s, 38, 4, True
        )

    leading = "9" + "0" * 33
    value = f"{leading}.12345"
    check_func(
        impl,
        (value,),
        py_output=pa.scalar(Decimal(f"{leading}.1235"), pa.decimal128(38, 4)),
    )


def test_str_to_decimal_error():
    """
    Test that with null_on_error=False strings that don't fit or don't parse raise
    an error.
    """

    @bodo.jit
    def impl(s):
        return bodosql.kernels.snowflake_conversion_array_kernels.string_to_decimal(
            s, 4, 2, False
        )

    with pytest.raises(
        RuntimeError,
        match="String value is out of range for decimal or doesn't parse properly",
    ):
        impl("-100")
    with pytest.raises(
        RuntimeError,
        match="String value is out of range for decimal or doesn't parse properly",
    ):
        impl("1 0")
    with pytest.raises(
        RuntimeError,
        match="String value is out of range for decimal or doesn't parse properly",
    ):
        impl(pd.array(["-100"] * 5))
    with pytest.raises(
        RuntimeError,
        match="String value is out of range for decimal or doesn't parse properly",
    ):
        impl(pd.array(["1 0"] * 5))


@pytest.mark.parametrize(
    "arr, expected",
    [
        pytest.param(
            pd.array(
                [
                    "1",
                    "1.55",
                    "1.56",
                    "10.56",
                    None,
                    "1000.5",
                    "10004.1",
                    "1100004.12345",
                    "11.41",
                    "0",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 5)),
            ),
            pd.array(
                [
                    "1.00000",
                    "1.55000",
                    "1.56000",
                    "10.56000",
                    None,
                    "1000.50000",
                    "10004.10000",
                    "1100004.12345",
                    "11.41000",
                    "0.00000",
                ]
            ),
            id="scale-5",
        ),
        pytest.param(
            pd.array(
                ["0", None, "0.00", "-0.0", "0.00000"],
                dtype=pd.ArrowDtype(pa.decimal128(22, 5)),
            ),
            pd.array(["0.00000", None, "0.00000", "0.00000", "0.00000"]),
            id="all-zeroes-scale-5",
        ),
        pytest.param(
            pd.array(
                ["32.4", "-123.987", None, "0.123", "-0.001"],
                dtype=pd.ArrowDtype(pa.decimal128(22, 3)),
            ),
            pd.array(["32.400", "-123.987", None, "0.123", "-0.001"]),
            id="mixed-signs-scale-3",
        ),
        pytest.param(
            pd.array(
                [
                    "1234567890.12",
                    "-9876543210.99",
                    None,
                    "1234567890.12",
                    "-9876543210.99",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 2)),
            ),
            pd.array(
                [
                    "1234567890.12",
                    "-9876543210.99",
                    None,
                    "1234567890.12",
                    "-9876543210.99",
                ]
            ),
            id="large-values-scale-2",
        ),
        pytest.param(
            pd.array(
                [
                    "1234567890.123456789012345678",
                    "-9876543210.99",
                    None,
                    "1234567890.12",
                    "-9876543210.99",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 18)),
            ),
            pd.array(
                [
                    "1234567890.123456789012345678",
                    "-9876543210.990000000000000000",
                    None,
                    "1234567890.120000000000000000",
                    "-9876543210.990000000000000000",
                ]
            ),
            id="large-scale-1",
        ),
        pytest.param(
            pd.array(
                [
                    "0.12345678901234567890123456789012345678",
                    "0.99",
                    None,
                    "0.12",
                    "0.99",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(38, 38)),
            ),
            pd.array(
                [
                    "0.12345678901234567890123456789012345678",
                    "0.99000000000000000000000000000000000000",
                    None,
                    "0.12000000000000000000000000000000000000",
                    "0.99000000000000000000000000000000000000",
                ]
            ),
            id="large-scale-2",
        ),
        pytest.param(
            pd.array(
                ["1234567890", "-9876543210", "1234567890", "-9876543210", None],
                dtype=pd.ArrowDtype(pa.decimal128(22, 0)),
            ),
            pd.array(["1234567890", "-9876543210", "1234567890", "-9876543210", None]),
            id="scale-0-integers",
        ),
    ],
)
def test_decimal_array_to_str_array(arr, expected, memory_leak_check):
    def impl(arr):
        return bodosql.kernels.to_char(arr)

    check_func(impl, (arr,), py_output=expected)


@pytest.mark.parametrize(
    "scalar, expected",
    [
        pytest.param(
            pa.scalar(Decimal("1.0"), pa.decimal128(15, 5)),
            "1.00000",
            id="basic-positive",
        ),
        # Zeroes
        pytest.param(
            pa.scalar(Decimal("0"), pa.decimal128(10, 3)),
            "0.000",
            id="zero-with-scale",
        ),
        pytest.param(
            pa.scalar(Decimal("0"), pa.decimal128(38, 37)),
            "0.0000000000000000000000000000000000000",
            id="zero-with-large-scale",
        ),
        pytest.param(
            pa.scalar(Decimal("-0.00"), pa.decimal128(10, 3)),
            "0.000",
            id="negative-zero-with-scale",
        ),
        # Negative Numbers
        pytest.param(
            pa.scalar(Decimal("-12.345"), pa.decimal128(12, 3)),
            "-12.345",
            id="negative-number",
        ),
        # Large Values (Precision and Scale Variation)
        pytest.param(
            pa.scalar(Decimal("123456789012.34567"), pa.decimal128(20, 7)),
            "123456789012.3456700",
            id="large-value-high-precision",
        ),
        pytest.param(
            pa.scalar(Decimal("1234567890.12"), pa.decimal128(15, 2)),
            "1234567890.12",
            id="large-value-low-precision",
        ),
        # Large Scale
        pytest.param(
            pa.scalar(
                Decimal("0.12345678901234567890123456789012345678"),
                pa.decimal128(38, 38),
            ),
            "0.12345678901234567890123456789012345678",
            id="large-scale-1",
        ),
        pytest.param(
            pa.scalar(Decimal("1234567890.123456789012345678"), pa.decimal128(38, 18)),
            "1234567890.123456789012345678",
            id="large-scale-2",
        ),
        # Scientific Notation (Precision and Scale Variation)
        pytest.param(
            pa.scalar(Decimal("1.23E5"), pa.decimal128(10, 2)),
            "123000.00",
            id="scientific-positive-scale",
        ),
        pytest.param(
            pa.scalar(Decimal("-4.56E-3"), pa.decimal128(10, 5)),
            "-0.00456",
            id="scientific-negative-scale",
        ),
        pytest.param(
            pa.scalar(
                Decimal("0.00000000000000000000000000000000000001"),
                pa.decimal128(38, 38),
            ),
            "0.00000000000000000000000000000000000001",
            id="suppress-scientific-notation",
        ),
        # Scale 0 (Integers)
        pytest.param(
            pa.scalar(Decimal("9876543210"), pa.decimal128(15, 0)),
            "9876543210",
            id="scale-zero-integer",
        ),
    ],
)
def test_decimal_to_str_scalar(scalar, expected, memory_leak_check):
    def impl(scalar):
        return bodosql.kernels.to_char(scalar, is_scalar=True)

    check_func(impl, (scalar,), py_output=expected)


@pytest.mark.skip(reason="TODO: fix comparison issue in testing function")
@pytest.mark.parametrize(
    "df, expected",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1, 1, 2, 3, 3],
                    "B": pd.array(
                        [
                            "1",
                            "2",
                            "3",
                            "4",
                            "5",
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
                    ),
                }
            ),
            None,
            id="basic",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1, 2, 3, 4, 5],
                    "B": pd.array(
                        [
                            "1",
                            "2",
                            "3",
                            "4",
                            "5",
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
                    ),
                }
            ),
            None,
            id="all-separate",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1, 1, 1, 1, 1],
                    "B": pd.array(
                        [
                            "1",
                            "2",
                            "3",
                            "4",
                            "5",
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
                    ),
                }
            ),
            None,
            id="all-same-group",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1, 1, 1, 1, 1, 1],
                    "B": pd.array(
                        [
                            "1",
                            "2",
                            "3",
                            "4",
                            "5",
                            "6",
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(10, 2)),
                    ),
                }
            ),
            None,
            id="even",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1, 1, 1, 1, 1],
                    "B": pd.array(
                        [
                            "5.12309891",
                            "6.123125236",
                            "1.6325",
                            "2.5123",
                            "4.10906127",
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 10)),
                    ),
                }
            ),
            None,
            id="decimals",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1, 1, 1, 1, 1],
                    "B": pd.array(
                        [
                            "0",
                            "6.123125236",
                            "-1.6325",
                            "2.5123",
                            "4.10906127",
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 10)),
                    ),
                }
            ),
            None,
            id="negatives-and-zeroes",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1, 1, 1, 1, 1],
                    "B": pd.array(
                        [
                            "0",
                            None,
                            "-1.6325",
                            None,
                            "4.10906127",
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 10)),
                    ),
                }
            ),
            None,
            id="nulls",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1, 1, 2, 2, 2],
                    "B": pd.array(
                        [
                            None,
                            None,
                            "-1.6325",
                            "0.45",
                            "4.10906127",
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(20, 10)),
                    ),
                }
            ),
            None,
            id="group-of-nulls",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1, 1, 2, 2, 3, 3],
                    "B": pd.array(
                        [
                            "1.1",
                            "1.1",
                            "1.1234567890123456789012345678901234",
                            "1.1234567890123456789012345678901234",
                            "1.0000000000000000000000000000000001",
                            "1.0000000000000000000000000000000003",
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(37, 34)),
                    ),
                }
            ),
            pd.Series(
                {
                    1: Decimal("1.1"),
                    2: Decimal("1.1234567890123456789012345678901234"),
                    3: Decimal("1.0000000000000000000000000000000002"),
                },
                name="B",
            ).rename_axis("A"),
            id="large_scale_precision",
        ),
    ],
)
def test_decimal_median(df, expected, memory_leak_check):
    def impl(df):
        A = df.groupby("A")["B"].median()
        return A

    if expected is not None:
        check_func(impl, (df,), sort_output=True, py_output=expected, check_dtype=False)
    else:
        check_func(impl, (df,), sort_output=True, check_dtype=False)


@pytest.mark.parametrize(
    "df, error_msg",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1, 1, 2, 2, 3, 3],
                    "B": pd.array(
                        [
                            "1",
                            "2",
                            "3",
                            "4",
                            "5",
                            "6",
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(38, 35)),
                    ),
                }
            ),
            "too large for MEDIAN operation",
            id="overflow_1",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1, 1, 2, 2, 3, 3],
                    "B": pd.array(
                        [
                            "1",
                            "2",
                            "3",
                            "4",
                            "5",
                            "6",
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(37, 35)),
                    ),
                }
            ),
            "too large for MEDIAN operation",
            id="overflow_2",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1, 1, 2, 2, 3, 3],
                    "B": pd.array(
                        [
                            "1",
                            "2",
                            "3",
                            "4",
                            "5",
                            "6",
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(38, 36)),
                    ),
                }
            ),
            "too large for MEDIAN operation",
            id="scale_too_large",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "A": [1, 1, 2, 2, 3, 3],
                    "B": pd.array(
                        [
                            "12345678901234567890123456789012345.123",
                            "12345678901234567890123456789012345.123",
                            "12345678901234567890123456789012345.123",
                            "12345678901234567890123456789012345.123",
                            "12345678901234567890123456789012345.123",
                            "12345678901234567890123456789012345.123",
                        ],
                        dtype=pd.ArrowDtype(pa.decimal128(38, 3)),
                    ),
                }
            ),
            "Intermediate values for MEDIAN do not fit within Decimal",
            id="intermediate_overflow",
        ),
    ],
)
def test_decimal_median_overflow(df, error_msg):
    def impl(df):
        A = df.groupby("A")["B"].median()
        return A

    with pytest.raises(Exception, match=error_msg):
        check_func(impl, (df,), sort_output=True, check_dtype=False)


@pytest.mark.parametrize(
    "val, expected",
    [
        pytest.param(
            pa.scalar(Decimal("1.0"), pa.decimal128(15, 5)),
            pa.scalar(Decimal("1.0"), pa.decimal128(15, 5)),
            id="basic-positive",
        ),
        # Zeroes
        pytest.param(
            pa.scalar(Decimal("0"), pa.decimal128(10, 3)),
            pa.scalar(Decimal("0"), pa.decimal128(10, 3)),
            id="zero-with-scale",
        ),
        pytest.param(
            pa.scalar(Decimal("0"), pa.decimal128(38, 37)),
            pa.scalar(Decimal("0"), pa.decimal128(38, 37)),
            id="zero-with-large-scale",
        ),
        pytest.param(
            pa.scalar(Decimal("-0.00"), pa.decimal128(10, 3)),
            pa.scalar(Decimal("0.00"), pa.decimal128(10, 3)),
            id="negative-zero-with-scale",
        ),
        # Negative Numbers
        pytest.param(
            pa.scalar(Decimal("-12.345"), pa.decimal128(12, 3)),
            pa.scalar(Decimal("12.345"), pa.decimal128(12, 3)),
            id="negative-number",
        ),
        # Large Values (Precision and Scale Variation)
        pytest.param(
            pa.scalar(Decimal("123456789012.34567"), pa.decimal128(20, 7)),
            pa.scalar(Decimal("123456789012.34567"), pa.decimal128(20, 7)),
            id="large-value",
        ),
        # Large Scale
        pytest.param(
            pa.scalar(
                Decimal("0.12345678901234567890123456789012345678"),
                pa.decimal128(38, 38),
            ),
            pa.scalar(
                Decimal("0.12345678901234567890123456789012345678"),
                pa.decimal128(38, 38),
            ),
            id="large-scale-1",
        ),
        pytest.param(
            pa.scalar(
                Decimal("-0.12345678901234567890123456789012345678"),
                pa.decimal128(38, 38),
            ),
            pa.scalar(
                Decimal("0.12345678901234567890123456789012345678"),
                pa.decimal128(38, 38),
            ),
            id="large-scale-negative",
        ),
        pytest.param(
            pa.scalar(
                Decimal("-0.00000000000000000000000000000000000001"),
                pa.decimal128(38, 38),
            ),
            pa.scalar(
                Decimal("0.00000000000000000000000000000000000001"),
                pa.decimal128(38, 38),
            ),
            id="small_number",
        ),
        pytest.param(
            pd.array(
                [
                    "1234567890.12",
                    "-9876543210.99",
                    None,
                    "0",
                    "-0.00000001",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 10)),
            ),
            pd.array(
                [
                    "1234567890.12",
                    "9876543210.99",
                    None,
                    "0",
                    "0.00000001",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 10)),
            ),
            id="array",
        ),
    ],
)
def test_decimal_abs(val, expected, memory_leak_check):
    def impl(val):
        return bodosql.kernels.numeric_array_kernels.abs_decimal(val)

    check_func(impl, (val,), py_output=expected)


@pytest.mark.parametrize(
    "val, expected",
    [
        pytest.param(
            pa.scalar(Decimal("1.0"), pa.decimal128(10, 3)),
            pa.scalar(Decimal("1"), pa.decimal128(37, 0)),
            id="one",
        ),
        pytest.param(
            pa.scalar(Decimal("6.4"), pa.decimal128(38, 37)),
            pa.scalar(Decimal("720"), pa.decimal128(37, 0)),
            id="rounded",
        ),
        pytest.param(
            pa.scalar(Decimal("12.3"), pa.decimal128(15, 5)),
            pa.scalar(Decimal("479001600"), pa.decimal128(37, 0)),
            id="large-rounded",
        ),
        pytest.param(
            pa.scalar(Decimal("-0.34"), pa.decimal128(10, 3)),
            pa.scalar(Decimal("1.0"), pa.decimal128(37, 0)),
            id="zero-rounded",
        ),
        pytest.param(
            pd.array(
                [
                    "1",
                    "2",
                    None,
                    "3",
                    "5",
                    "11",
                    "18",
                    "25",
                    "33",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 10)),
            ),
            pd.array(
                [
                    "1",
                    "2",
                    None,
                    "6",
                    "120",
                    "39916800",
                    "6402373705728000",
                    "15511210043330985984000000",
                    "8683317618811886495518194401280000000",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(37, 0)),
            ),
            id="array",
        ),
    ],
)
def test_decimal_factorial(val, expected, memory_leak_check):
    """
    Test computing the factorial of a decimal scalar or array.
    """

    def impl(val):
        return bodosql.kernels.factorial(val)

    check_func(impl, (val,), py_output=expected)


@pytest.mark.parametrize(
    "val, error_msg",
    [
        pytest.param(
            pa.scalar(Decimal("-1"), pa.decimal128(10, 3)),
            "is negative",
            id="negative",
        ),
        pytest.param(
            pa.scalar(Decimal("35"), pa.decimal128(10, 3)),
            "is too large",
            id="too_large",
        ),
        pytest.param(
            pa.scalar(
                Decimal("99999999999999999999999999999999999999"), pa.decimal128(38, 0)
            ),
            "is too large",
            id="max_precision",
        ),
        pytest.param(
            pd.array(
                [
                    "-3",
                    "2",
                    None,
                    "3",
                    "5",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 10)),
            ),
            "is negative",
            id="array-negative",
        ),
        pytest.param(
            pd.array(
                [
                    "353",
                    "2",
                    None,
                    "3",
                    "5",
                ],
                dtype=pd.ArrowDtype(pa.decimal128(22, 10)),
            ),
            "is too large",
            id="array-positive",
        ),
    ],
)
def test_decimal_factorial_error(val, error_msg):
    """
    Test errors during computing the factorial of a decimal scalar or array.
    """

    def impl(val):
        return bodosql.kernels.factorial(val)

    with pytest.raises(Exception, match=error_msg):
        check_func(impl, (val,))
