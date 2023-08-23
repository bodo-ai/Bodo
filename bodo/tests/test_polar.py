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


@pytest.mark.skip(reason="[BSE-912] TODO: complete coverage for polar_format function")
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
        r_norm.reshape((len(r_norm), 1))
        r_norm = np.tile(r_norm, (1, 3))

        r_hat = pos / r_norm

        # Convert to matrices to make projections easier
        r_hat = np.asmatrix(r_hat)
        u_hat = np.asmatrix([u_hat])
        v_hat = np.asmatrix([v_hat])

        k_matrix = np.tile(k, (npulses, 1))
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
        ky_new = np.zeros([npulses, nu])
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

    # Load the simulated phase history from the compressed file
    phs = np.loadtxt(
        datapath("data/polar_format_e2e_in.gz"), dtype=np.dtype("complex128")
    )
    # Hardcode the inputs that would normally be placed in the dictionaries
    c = 299792458.0
    npulses = 1950
    f_0 = 10000000000.0
    pos = np.loadtxt(datapath("data/polar_format_e2e_pos.txt"))
    R_c = np.array(
        [8.660254037844386403e03, 5.682954107300020041e-15, 5.000000000000000000e03],
        dtype=np.float64,
    )
    n_hat = np.array([0, 0, 1], dtype=np.int64)
    k_r, k_u, k_v = np.loadtxt(datapath("data/polar_format_e2e_k.txt"))
    env_args = (c, npulses, f_0, pos, R_c, n_hat, k_r, k_u, k_v)
    # Load the expected output from the compressed file
    expected_output = np.loadtxt(datapath("data/polar_format_e2e_out.gz"))
    res = polar_format(phs, env_args, 17)
    # Verify that the function run with regular Python matches the previously calculated output
    np.testing.assert_array_equal(res, expected_output)
    # Now test with Bodo
    check_func(
        polar_format,
        (phs, env_args, 17),
        py_output=expected_output,
        convert_to_nullable_float=False,
    )


def test_division_bug(datapath, memory_leak_check):
    """
    Tests a specific bug that occurs when a numpy array is
    divided by its maximum.
    """

    def impl(A):
        return A / A.max()

    check_func(impl, (np.linspace(0, 10, 101),), convert_to_nullable_float=False)


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

    # [BSE-991] TODO: investigate parallel support
    check_func(taylor, (2048, 17), only_seq=True)
    check_func(taylor, (1950, 17), only_seq=True)


@pytest.fixture
def complex_arr():
    """Returns a 5x5 grid of complex numbers used for testing various fft functions"""
    real_component = np.sin(np.arange(25))
    imag_component = 1j * np.cos(np.arange(25) ** 2)
    return (real_component + imag_component).reshape((5, 5))


@pytest.mark.skip(
    reason="[BSE-943] TODO: Support scipy.fftpack.fftshift for workload"
)
def test_fftshift(complex_arr, memory_leak_check):
    def impl(data):
        return fftshift(data)

    check_func(impl, (complex_arr,), convert_to_nullable_float=False)


@pytest.mark.skip(reason="[BSE-944] TODO: Support scipy.fftpack.fft2 for workload")
def test_fft2(complex_arr, memory_leak_check):
    def impl(data):
        return fft2(data)

    check_func(impl, (complex_arr,), convert_to_nullable_float=False)


@pytest.mark.skip(
    reason="[BSE-943] / [BSE-944] TODO: Support scipy.fftpack.fftshift / scipy.fftpack.fft2 for workload"
)
def test_ft2(complex_arr, memory_leak_check):
    def ft2(f, delta=1):
        F = fftshift(fft2(fftshift(f))) * delta**2
        return F

    real_component = np.sin(np.arange(25))
    imag_component = 1j * np.cos(np.arange(25) ** 2)
    complex_arr = (real_component + imag_component).reshape((5, 5))
    check_func(ft2, (complex_arr,), convert_to_nullable_float=False)


@pytest.mark.skip(reason="[BSE-938] TODO: Support np.tile for workload")
@pytest.mark.parametrize(
    "A, reps",
    [
        pytest.param(
            np.array([[i] for i in range(1950)]), (1, 3), id="column_replication"
        ),
        pytest.param(
            np.array([i for i in range(2048)]),
            (1950, 1),
            id="dimension_upcast_with_transpose",
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

    def impl(A, reps):
        return np.tile(A, reps)

    check_func(impl, (A, reps), convert_to_nullable_float=False)


@pytest.mark.skip(
    reason="[BSE-949] TODO: Support np.asmatrix and matrix multiplication with * operator for workload"
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

    If already_matrix is True, converts the input to a matrix before calling hte function.
    If False, leaves the inputs as 2D numpy arrays and has the function do the conversion.
    """

    # Construct 2 matrices using the desired dimensions by taking 1d arrays, extracting
    # the prefix of desired length, and reshaping into 2d arrays.
    data_a = np.arange(100)
    data_b = (np.arange(0, 300, 3) % 19) - 9
    A = data_a[: r0 * c0].reshape((r0, c0))
    B = data_b[: r1 * c1].reshape((r1, c1))

    if already_matrix:

        def impl(A, B):
            return A * B

        A = np.asmatrix(A)
        B = np.asmatrix(A)
    else:

        def impl(A, B):
            a_matrix = np.asmatrix(A)
            b_matrix = np.asmatrix(B)
            return a_matrix * b_matrix

    check_func(impl, (A, B), convert_to_nullable_float=False)


@pytest.mark.skip(reason="[BSE-949] TODO: Support np.asmatrix for workload")
def test_asmatrix_2d(memory_leak_check):
    """
    Tests calling np.asmatrix on a 2d numpy array.
    """

    def impl(A):
        return np.asmatrix(A)

    A = np.arange(20).reshape(5, 4)
    check_func(impl, (A,), convert_to_nullable_float=False)


@pytest.mark.skip(reason="[BSE-949] TODO: Support np.asmatrix for workload")
def test_asmatrix_1d_no_list(memory_leak_check):
    """
    Tests calling np.asmatrix on a 1d numpy array.
    """

    def impl(A):
        return np.asmatrix(A)

    A = np.arange(20)
    check_func(impl, (A,), convert_to_nullable_float=False)


@pytest.mark.skip(reason="[BSE-949] TODO: Support np.asmatrix for workload")
def test_asmatrix_1d_with_list(memory_leak_check):
    """
    Tests calling np.asmatrix on a 1d numpy array stored in a singleton list.
    """

    def impl(A):
        return np.asmatrix([A])

    A = np.arange(8)
    check_func(impl, (A,), convert_to_nullable_float=False)


@pytest.mark.skip(
    reason="[BSE-949] TODO: Support np.asmatrix and matrix multiplication with * operator for workload"
)
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
    check_func(impl, (r_hat, u_hat, v_hat, k, 13))


@pytest.mark.skip(reason="[BSE-945] TODO: Support np.nan_to_num for workload")
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((60,), id="1d"),
        pytest.param((12, 5), id="2d"),
    ],
)
def test_nan_to_num(shape, memory_leak_check):
    """
    Tests np.nan_to_num on a float array of various dimensions
    without any additional arguments.
    """

    def impl(A):
        return np.nan_to_num(A)

    a = np.arange(60)
    data = np.tan(a) ** 5
    data[a % 7 == 3] = np.inf
    data[a % 13 == 12] = -np.inf
    data[a % 7 == 5] = np.nan
    check_func(impl, (data.reshape(shape),))


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
