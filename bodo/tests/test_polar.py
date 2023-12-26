# Copyright (C) 2023 Bodo Inc. All rights reserved.

"""E2E and unit tests for the polar_format function.
"""
import numpy as np
import pytest
from numpy import arccosh, cos, dot, pi, sqrt
from numpy.linalg import norm
from scipy.fftpack import fft2, fftshift

import bodo
from bodo.tests.utils import check_func
from bodo.utils.typing import BodoError


@pytest.mark.skipif(
    bodo.get_size() > 1,
    reason="[BSE-991] TODO: investigate parallel support for taylor function",
)
@pytest.mark.skip(
    reason="TODO: Retrieve data from https://s3.console.aws.amazon.com/s3/buckets/data"
)
def test_polar_format_e2e(datapath, memory_leak_check):
    """
    Full E2E test of the polar_format function from the workload. The function
    has been modified to receive its arguments via a tuple instead via dictionaries.
    """

    @bodo.jit
    def ft2(f, delta=1):
        F = fftshift(fft2(fftshift(f))) * delta**2
        return F

    @bodo.jit
    def sig_taylor(nsamples, S_L=43):
        xi = np.linspace(-0.5, 0.5, nsamples)
        A = 1.0 / pi * arccosh(10 ** (S_L * 1.0 / 20))
        n_bar = int(2 * A**2 + 0.5) + 1
        sigma_p = n_bar / sqrt(A**2 + (n_bar - 0.5) ** 2)

        # Compute F_m
        m = np.arange(1, n_bar)
        n = np.arange(1, n_bar)
        F_m = np.zeros(n_bar - 1)
        for i in m:
            num = 1
            den = 1
            for j in n:
                num = (
                    num
                    * (-1) ** (i + 1)
                    * (1 - i**2 * 1.0 / sigma_p**2 / (A**2 + (j - 0.5) ** 2))
                )
                if i != j:
                    den = den * (1 - i**2 * 1.0 / j**2)

            F_m[i - 1] = num / den

        w = np.ones(nsamples)
        for i in m:
            w += F_m[i - 1] * cos(2 * pi * i * xi)

        w = w / w.max()
        return w

    # The main function being tested
    def polar_format(phs, env_data, taylor=20):
        ##############################################################################
        #                                                                            #
        #  This is the Polar Format algorithm.  The phase history data as well as    #
        #  platform and image plane dictionaries are taken as inputs.                #
        #                                                                            #
        #  The phase history data is collected on a two-dimensional surface in       #
        #  k-space.  For each pulse, a strip of this surface is collected.  The      #
        #  first step in this program is to project each strip onto the (ku,kv)      #
        #  plane defined by the normal vector contained in the image plane           #
        #  dictionary.  This will result in data that is unevenly spaced in (ku,kv). #
        #  This unevenly spaced data is interpolated onto an evenly spaced (ku,kv)   #
        #  grid defined in the image plane dictionary.  The interpolation is done    #
        #  along the radial direction first, then along the along-track direction.   #
        #  Further details of this method are given in both the Jakowitz and Carrera #
        #  texts.                                                                    #
        #                                                                            #
        ##############################################################################

        # BODO CHANGE: Retrieve relevent parameters (hardcoded into a single tuple
        # instead of passed in via dictionary inputs)
        c, npulses, f_0, pos, R_c, n_hat, k, k_ui, k_vi = env_data

        # Compute k_xi offset
        psi = pi / 2 - np.arccos(np.dot(R_c, n_hat) / norm(R_c))
        k_ui = k_ui + 4 * pi * f_0 / c * np.cos(psi)

        # Compute number of samples in scene
        nu = k_ui.size
        nv = k_vi.size

        # Compute x and y unit vectors. x defined to lie along R_c.
        # z = cross(vec[0], vec[-1]); z =z/norm(z)
        u_hat = (R_c - dot(R_c, n_hat) * n_hat) / norm((R_c - dot(R_c, n_hat) * n_hat))
        v_hat = np.cross(u_hat, n_hat)

        # Compute r_hat, the diretion of k_r, for each pulse
        r_norm = norm(pos, axis=1)
        # BODO CHANGE: replace r_norm = np.array([r_norm]).T with r_norm.reshape((len(r_norm), 1))
        r_norm = r_norm.reshape((len(r_norm), 1))
        r_norm = np.tile(r_norm, (1, 3))

        r_hat = pos / r_norm

        # Convert to matrices to make projections easier
        r_hat = np.asmatrix(r_hat)
        u_hat = np.asmatrix([u_hat])
        v_hat = np.asmatrix([v_hat])

        # NOTE: commented out dead code in original function
        # k_matrix = np.tile(k, (npulses, 1))
        k_matrix = np.asmatrix(k)

        # Compute kx and ky meshgrid
        ku = r_hat * u_hat.T * k_matrix
        ku = np.asarray(ku)
        kv = r_hat * v_hat.T * k_matrix
        kv = np.asarray(kv)

        # Create taylor windows
        win1 = sig_taylor(int(phs.shape[1]), S_L=taylor)
        win2 = sig_taylor(int(phs.shape[0]), S_L=taylor)

        # Radially interpolate kx and ky data from polar raster
        # onto evenly spaced kx_i and ky_i grid for each pulse
        # BODO CHANGE: replace np.zeroes([a, b]) with np.zeros((a, b))
        real_rad_interp = np.zeros((npulses, nu))
        imag_rad_interp = np.zeros((npulses, nu))
        ky_new = np.zeros((npulses, nu))
        for i in range(npulses):
            real_rad_interp[i, :] = np.interp(
                k_ui, ku[i, :], phs.real[i, :] * win1, left=0, right=0
            )
            imag_rad_interp[i, :] = np.interp(
                k_ui, ku[i, :], phs.imag[i, :] * win1, left=0, right=0
            )
            ky_new[i, :] = np.interp(k_ui, ku[i, :], kv[i, :])

        # Interpolate in along track direction to obtain polar formatted data
        # BODO CHANGE: replace np.zeroes([a, b]) with np.zeros((a, b))
        real_polar = np.zeros((nv, nu))
        imag_polar = np.zeros((nv, nu))
        isSort = ky_new[npulses // 2, nu // 2] < ky_new[npulses // 2 + 1, nu // 2]
        if isSort:
            for i in range(nu):
                real_polar[:, i] = np.interp(
                    k_vi, ky_new[:, i], real_rad_interp[:, i] * win2, left=0, right=0
                )
                imag_polar[:, i] = np.interp(
                    k_vi, ky_new[:, i], imag_rad_interp[:, i] * win2, left=0, right=0
                )
        else:
            for i in range(nu):
                real_polar[:, i] = np.interp(
                    k_vi,
                    ky_new[::-1, i],
                    real_rad_interp[::-1, i] * win2,
                    left=0,
                    right=0,
                )
                imag_polar[:, i] = np.interp(
                    k_vi,
                    ky_new[::-1, i],
                    imag_rad_interp[::-1, i] * win2,
                    left=0,
                    right=0,
                )

        real_polar = np.nan_to_num(real_polar)
        imag_polar = np.nan_to_num(imag_polar)
        phs_polar = np.nan_to_num(real_polar + 1j * imag_polar)

        img = np.abs(ft2(phs_polar))

        return img

    # Load the data from S3 (TODO)
    phs = np.load("data/polar_format_e2e_in.npy")
    pos = np.load("data/polar_format_e2e_pos.npy")
    k_r, k_u, k_v = np.load("data/polar_format_e2e_k.npy")
    expected_output = np.load("data/polar_format_e2e_out.npy")
    # Hardcode the inputs that would normally be placed in the dictionaries
    c = 299792458.0
    npulses = 1950
    f_0 = 10000000000.0
    R_c = np.array(
        [8.660254037844386403e03, 5.682954107300020041e-15, 5.000000000000000000e03],
        dtype=np.float64,
    )
    n_hat = np.array([0, 0, 1], dtype=np.int64)
    env_args = (c, npulses, f_0, pos, R_c, n_hat, k_r, k_u, k_v)
    # Load the expected output from the compressed file
    res = polar_format(phs, env_args, 17)
    # Verify that the function run with regular Python matches the previously calculated output
    np.testing.assert_allclose(res, expected_output, rtol=1e-5, atol=1e-8)
    # Now test with Bodo
    check_func(
        polar_format,
        (phs, env_args, 17),
        py_output=expected_output,
        convert_to_nullable_float=False,
        only_seq=True,
    )


def test_division_bug(datapath, memory_leak_check):
    """
    Tests a specific bug that occurs when a numpy array is
    divided by its maximum.
    """

    def impl(A):
        return A / A.max()

    check_func(impl, (np.linspace(0, 10, 101),), convert_to_nullable_float=False)


@pytest.mark.skipif(
    bodo.get_size() > 1,
    reason="[BSE-991] TODO: investigate parallel support for taylor function",
)
def test_taylor(memory_leak_check):
    def taylor(nsamples, S_L=43):
        xi = np.linspace(-0.5, 0.5, nsamples)
        A = 1.0 / pi * arccosh(10 ** (S_L * 1.0 / 20))
        n_bar = int(2 * A**2 + 0.5) + 1
        sigma_p = n_bar / sqrt(A**2 + (n_bar - 0.5) ** 2)
        m = np.arange(1, n_bar)
        n = np.arange(1, n_bar)
        F_m = np.zeros(n_bar - 1)
        for i in m:
            num = 1
            den = 1
            for j in n:
                num = (
                    num
                    * (-1) ** (i + 1)
                    * (1 - i**2 * 1.0 / sigma_p**2 / (A**2 + (j - 0.5) ** 2))
                )
                if i != j:
                    den = den * (1 - i**2 * 1.0 / j**2)

            F_m[i - 1] = num / den
        w = np.ones(nsamples)
        for i in m:
            w += F_m[i - 1] * cos(2 * pi * i * xi)
        w = w / w.max()
        return w

    check_func(taylor, (2048, 17), only_seq=True)
    check_func(taylor, (1950, 17), only_seq=True)


@pytest.fixture(
    params=[
        pytest.param((np.complex128, "C"), id="complex128-C"),
        pytest.param((np.complex128, "F"), id="complex128-F", marks=pytest.mark.slow),
        pytest.param((np.complex128, "A"), id="complex128-A", marks=pytest.mark.slow),
        pytest.param((np.complex64, "C"), id="complex64-C", marks=pytest.mark.slow),
        pytest.param((np.complex64, "F"), id="complex64-F", marks=pytest.mark.slow),
        pytest.param((np.complex64, "A"), id="complex64-A", marks=pytest.mark.slow),
    ]
)
def grid_layouts(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param((5, 5), id="odd_dims"),
        pytest.param((10, 10), id="even_dims"),
        pytest.param((1024, 1001), id="big_mismatched_dims"),
        pytest.param((1, 513), id="one_by_large_odd", marks=pytest.mark.slow),
        pytest.param((1, 840), id="one_by_large_even", marks=pytest.mark.slow),
        pytest.param((513, 1), id="large_odd_by_one", marks=pytest.mark.slow),
        pytest.param((840, 1), id="large_even_by_one", marks=pytest.mark.slow),
        pytest.param((1, 1), id="one_by_one", marks=pytest.mark.slow),
        pytest.param((513, 840), id="large_odd_by_even", marks=pytest.mark.slow),
        pytest.param((840, 513), id="large_even_by_odd", marks=pytest.mark.slow),
        pytest.param((540, 740), id="large_even_by_even", marks=pytest.mark.slow),
        pytest.param((513, 867), id="large_odd_by_odd", marks=pytest.mark.slow),
    ]
)
def fft_arr(request, grid_layouts):
    """Returns a grid of numbers used for testing various fft functions
    with various dimensions, layouts and dtypes."""
    rows, cols = request.param
    dtype, layout = grid_layouts
    real_component = 3 * np.sin(np.arange(rows * cols))
    imag_component = 2.5j * np.cos(np.arange(rows * cols) ** 2)
    combined = (real_component + imag_component).reshape((rows, cols))
    if dtype != np.complex128:
        combined = combined.astype(dtype)
    if layout == "F":
        combined = combined.T
    if layout == "A":
        combined = np.hstack(
            [np.vstack([combined, combined]), np.vstack([combined, combined])]
        )[rows:, cols:]
    return combined


def test_fftshift(fft_arr, memory_leak_check):
    """
    Tests scipy.fftpack.fftshift. Currently only supported
    on sequential data.
    """

    def impl(data):
        res = fftshift(data)
        return res

    check_func(impl, (fft_arr,), convert_to_nullable_float=False)


def test_fft2(fft_arr, memory_leak_check):
    """
    Tests scipy.fftpack.fft2. Currently only supported
    on sequential data. The output is further transformed to check that
    the array layout is being handled correctly. Assumes that the input
    array has more than 1 row and more than 2 columns.
    """

    def impl(data):
        res = fft2(data)
        return res

    def impl_layout(data):
        res = fft2(data)
        return res, res[0], res[:, 0], res[1, 2]

    # Single precision results have higher error
    # than the default closeness
    rtol = 1e-05
    if fft_arr.dtype == np.complex64:
        rtol = 1e-03

    check_func(impl, (fft_arr,), convert_to_nullable_float=False, rtol=rtol)
    check_func(
        impl, (fft_arr,), convert_to_nullable_float=False, only_seq=True, rtol=rtol
    )


def test_ft2(fft_arr, memory_leak_check):
    """
    Tests the combination of fftshift and fft2 used in the polar_format workload.
    """

    def ft2(f, delta=1):
        F = fftshift(fft2(fftshift(f))) * delta**2
        return F

    # Single precision results have higher error
    # than the default closeness
    rtol = 1e-05
    if fft_arr.dtype == np.complex64:
        rtol = 1e-03

    check_func(ft2, (fft_arr,), convert_to_nullable_float=False, rtol=rtol)


def test_fft_error(memory_leak_check):
    """
    Verifies that fft2 raises an error on unsupported types.
    """

    def impl(A):
        return fft2(A)

    with pytest.raises(
        BodoError,
        match="fft2 currently unsupported on input of type .*",
    ):
        bodo.jit(impl)(np.array([1, 2, 3, 4, 5], dtype=np.int64))


def test_fftshift_error(memory_leak_check):
    """
    Verifies that fftshift raises an error on unsupported types.
    """

    def impl(A):
        return fftshift(A)

    with pytest.raises(
        BodoError,
        match="fftshift currently unsupported on input of type .*",
    ):
        bodo.jit(impl)(np.array([1, 2, 3, 4, 5], dtype=np.int64))


@pytest.mark.parametrize(
    "A, reps",
    [
        pytest.param(
            np.array([[i] for i in range(1950)]), (1, 3), id="column_repeat-single"
        ),
        pytest.param(
            np.array(np.array(range(1950)).reshape((390, 5))),
            (1, 3),
            id="column_repeat-multiple",
        ),
        pytest.param(
            np.array([i for i in range(2048)]),
            (1, 1),
            id="dimension_upcast_with_transpose-a",
        ),
        pytest.param(
            np.array([i for i in range(2048)]),
            (2, 1),
            id="dimension_upcast_with_transpose-b",
        ),
        pytest.param(
            np.array([i for i in range(2048)]),
            (3, 1),
            id="dimension_upcast_with_transpose-c",
        ),
        pytest.param(
            np.array([i for i in range(2048)]),
            (13, 1),
            id="dimension_upcast_with_transpose-d",
        ),
        pytest.param(
            np.array([i for i in range(2048)]),
            (47, 1),
            id="dimension_upcast_with_transpose-e",
        ),
        pytest.param(
            np.array([i for i in range(2048)]),
            (1950, 1),
            id="dimension_upcast_with_transpose-f",
        ),
    ],
)
def test_tile(A, reps, memory_leak_check):
    """
    Tests the correctness of np.tile on the dimension/tuple cases that we
    care about for the polar_format workload. These cases are:

    - Turning an (n, 1) array into an (n, 3) array
    - Turning an array of n elements into an array with dimension (m, n)
    """
    func_text = "def impl(A):\n"
    func_text += f"  return np.tile(A, {reps})\n"
    loc_vars = {}
    exec(func_text, {"np": np}, loc_vars)
    impl = loc_vars["impl"]
    # Input to row repeat is not distributed
    is_row_repeat = A.ndim == 1
    if is_row_repeat:
        check_func(
            impl,
            (A,),
            convert_to_nullable_float=False,
            is_out_distributed=True,
            distributed=[],
        )
    else:
        check_func(impl, (A,), convert_to_nullable_float=False)


def test_tile_non_constant(memory_leak_check):
    """
    Tests the correctness of np.tile on the dimension/tuple cases tested
    by test_tile, but where one of the elements is non-constant.
    """

    def impl1(A, n):
        return np.tile(A, (1, n))

    def impl2(A, n):
        return np.tile(A, (n, 1))

    A1 = np.linspace(0, 1, 11).reshape((11, 1))
    check_func(impl1, (A1, 7), convert_to_nullable_float=False)
    A2 = np.linspace(0, 1, 11)
    # Argument isn't distributed in this case
    check_func(
        impl2,
        (A2, 7),
        convert_to_nullable_float=False,
        is_out_distributed=True,
        distributed=[],
    )


@pytest.mark.parametrize(
    "r0, c0, r1, c1",
    [
        pytest.param(2, 2, 2, 2, id="A"),
        pytest.param(2, 5, 5, 3, id="B"),
        pytest.param(8, 1, 1, 12, id="C"),
        pytest.param(10, 10, 10, 5, id="D"),
        pytest.param(1, 20, 20, 1, id="E"),
    ],
)
@pytest.mark.parametrize(
    "already_matrix",
    [
        pytest.param(True, id="matrix_input"),
        pytest.param(False, id="use_asmatrix"),
    ],
)
def test_matrix_multiply(r0, c0, r1, c1, already_matrix, memory_leak_check):
    """
    Tests using the * operator where the inputs are matrices. Whereas nummpy arrays will do
    element-wise multiplication, numpy matrices will do matrix multiplication as if we
    were using the @ operator.

    If already_matrix is True, converts the input to a matrix before calling the function.
    If False, leaves the inputs as 2D numpy arrays and has the function do the conversion.
    """

    # Construct 2 matrices using the desired dimensions by taking 1d arrays, extracting
    # the prefix of desired length, and reshaping into 2d arrays.
    data_a = np.linspace(0, 30, 101) * ((-1) ** np.arange(101))
    data_b = np.tan(data_a)
    A = data_a[: r0 * c0].reshape((r0, c0))
    B = data_b[: r1 * c1].reshape((r1, c1))

    if already_matrix:

        def impl(A, B):
            return A * B

        A = np.asmatrix(A)
        B = np.asmatrix(B)
    else:

        def impl(A, B):
            a_matrix = np.asmatrix(A)
            b_matrix = np.asmatrix(B)
            return a_matrix * b_matrix

    check_func(impl, (A, B), convert_to_nullable_float=False, only_seq=True)


def test_asmatrix_2d(memory_leak_check):
    """
    Tests calling np.asmatrix on a 2d numpy array.
    """

    def impl(A):
        return np.asmatrix(A)

    A = np.arange(20).reshape(5, 4)
    check_func(impl, (A,), convert_to_nullable_float=False, only_seq=True)


def test_asmatrix_1d_no_list(memory_leak_check):
    """
    Tests calling np.asmatrix on a 1d numpy array.
    """

    def impl(A):
        return np.asmatrix(A)

    A = np.arange(20)
    check_func(impl, (A,), convert_to_nullable_float=False, only_seq=True)


def test_asmatrix_1d_with_list(memory_leak_check):
    """
    Tests calling np.asmatrix on a 1d numpy array stored in a singleton list.
    """

    def impl(A):
        return np.asmatrix([A])

    A = np.arange(8)
    check_func(impl, (A,), convert_to_nullable_float=False, only_seq=True)


def test_polar_format_matrix_subset(memory_leak_check):
    """
    Tests the full subset of polar_format that goes through numpy matrices. Indexes into
    the outputs at the end to ensure that the necessary operations are still supported
    after converting back to arrays.
    """

    def impl(r_hat, u_hat, v_hat, k, i):
        r_hat = np.asmatrix(r_hat)
        u_hat = np.asmatrix([u_hat])
        v_hat = np.asmatrix([v_hat])
        k_matrix = np.asmatrix(k)
        ku = r_hat * u_hat.T * k_matrix
        ku = np.asarray(ku)
        kv = r_hat * v_hat.T * k_matrix
        kv = np.asarray(kv)
        return ku, kv, ku[i, :], kv[i, :]

    r_hat = np.linspace(0, 1, 5850).reshape((1950, 3))
    u_hat = np.array([1.00000000e00, 6.56211017e-19, 0.00000000e00])
    v_hat = np.array([6.56211017e-19, -1.00000000e00, 0.00000000e00])
    k = np.linspace(0, 1, 2048)
    check_func(
        impl,
        (r_hat, u_hat, v_hat, k, 13),
        convert_to_nullable_float=False,
        only_seq=True,
    )


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((), id="scalar"),
        pytest.param((60,), id="1d"),
        pytest.param((12, 5), id="2d"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(np.int32, id="int32"),
        pytest.param(
            np.float16,
            id="float16",
            marks=pytest.mark.skip("[BSE-984] TODO: support np.isnan on np.float16"),
        ),
        pytest.param(np.float32, id="float32"),
        pytest.param(np.float64, id="float64"),
        pytest.param(
            np.complex128,
            id="complex128",
            marks=pytest.mark.skip(
                "[BSE-985] TODO: support np.nan_to_num on complex data"
            ),
        ),
    ],
)
def test_nan_to_num(shape, dtype, memory_leak_check):
    """
    Tests np.nan_to_num on a float array of various dimensions
    without any additional arguments.
    """

    def impl1(A):
        return np.nan_to_num(A)

    def impl2(A):
        return np.nan_to_num(A, nan=-3.14, posinf=-2, neginf=-1)

    a = np.arange(60)
    data = np.tan(a) ** 5
    data[a % 7 == 3] = np.inf
    data[a % 13 == 12] = -np.inf
    data[a % 7 == 5] = np.nan
    if dtype == np.complex128:
        data = data + 1j * data[::-1]
    else:
        data = data.astype(dtype)
    if len(shape) == 0:
        for val in data:
            check_func(impl1, (val,))
            check_func(impl2, (val,))
    else:
        check_func(impl1, (data.reshape(shape),), convert_to_nullable_float=False)
        check_func(impl2, (data.reshape(shape),), convert_to_nullable_float=False)


@pytest.mark.parametrize(
    "bound_args",
    [
        pytest.param("neither", id="no_bounds"),
        pytest.param("both", id="double_bounds"),
        pytest.param("left", id="left_bound"),
        pytest.param("right", id="right_bound"),
        pytest.param("custom", id="custom_bounds"),
    ],
)
def test_np_interp(bound_args, memory_leak_check):
    """
    Tests np.interp both without and with the clipping bounds.
    """

    def impl_neither(x, xp, fp):
        return np.interp(x, xp, fp)

    def impl_left(x, xp, fp):
        return np.interp(x, xp, fp, left=0)

    def impl_right(x, xp, fp):
        return np.interp(x, xp, fp, right=0)

    def impl_both(x, xp, fp):
        return np.interp(x, xp, fp, left=0, right=0)

    def impl_custom(x, xp, fp, left, right):
        return np.interp(x, xp, fp, left=left, right=right)

    impls = {
        "neither": impl_both,
        "left": impl_left,
        "right": impl_right,
        "both": impl_neither,
        "custom": impl_custom,
    }

    impl = impls[bound_args]

    x = np.linspace(0, 10, 101)
    xp = np.linspace(2.5, 7.5, 5)
    fp = np.tan(xp)
    args = (x, xp, fp)
    if bound_args == "custom":
        args += (-1, 10.5)
    check_func(
        impl,
        args,
        convert_to_nullable_float=False,
        # TODO: support in parallel (requires a parallel binary search for interp_bin_search)
        only_seq=True,
    )


@pytest.mark.parametrize(
    "arg0, arg1",
    [
        pytest.param("I", "F", id="int-float"),
        pytest.param("F", "I", id="float-int"),
    ],
)
def test_np_dot_heterogeneous(arg0, arg1, memory_leak_check):
    """
    Tests np.dot where both arguments are arrays but they have
    different dtypes.
    """

    def impl(A, B):
        return np.dot(A, B)

    arrs = {
        "I": np.array([1, 2, 3, 4, 5], dtype=np.int64),
        "F": np.array([0.5, 0, -1.0, 3.5, 2.25], dtype=np.float64),
    }
    A, B = arrs[arg0], arrs[arg1]
    check_func(impl, (A, B), convert_to_nullable_float=False)


def test_norm_axis(memory_leak_check):
    """
    Tests the use of np.linalg.norm with the keyword argument
    axis=1.
    """

    def impl1(A):
        return norm(A, axis=1)

    def impl2(A):
        return norm(A)

    A = np.linspace(-3.0, 8.8, 60).reshape((5, 12)).T
    check_func(impl1, (A,), convert_to_nullable_float=False)
    check_func(impl2, (A,), convert_to_nullable_float=False, only_seq=True)
