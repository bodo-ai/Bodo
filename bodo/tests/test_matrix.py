import numpy as np
import pytest

from bodo.tests.utils import check_func


@pytest.fixture(
    params=[
        pytest.param((2, 2), id="2x2"),
        pytest.param((1, 4), id="1x4"),
        pytest.param((20, 13), id="20x13"),
    ]
)
def matrix_shape(request):
    """
    The various shapes used to test matrix functions with. The parameterized
    choices have the following properities:
    - A shape where the rows and cols are the same
    - A shape where one of the lengths is 1 and the other isn't
    - A shape where the rows and cols are different, and neither is one (+ its a bit larger)
    """
    return request.param


@pytest.fixture(
    params=[
        pytest.param("C", id="C"),
        pytest.param("F", id="F", marks=pytest.mark.slow),
        pytest.param("A", id="A", marks=pytest.mark.slow),
    ]
)
def matrix_layout(request):
    """
    The various data layouts to test matrix functions with:
    - C: the data is contiguous in row-major order
    - F: the data is contiguous in column-major order
    - A: neither of the above
    """
    return request.param


def make_matrix(dtype, shape, layout="C"):
    """
    Helper for testing_matrices fixture. Takes in a dtype, shape, and layout
    then creates a pair of matrices with these properties in the manner
    described by testing_matrices. The shape of the second matrix returned
    will be the transpose of the shape of the first.
    """
    rows, cols = shape
    if isinstance(dtype, np.integer):
        # If the dtype is an integer, create a 1D array with the desired number
        # of total elements containing integers in a range.
        a1 = np.arange(rows * cols).astype(dtype)
        a2 = np.arange(-rows * cols, rows * cols * 2, 3).astype(dtype)
    else:
        # If the dtype is a float, do the same but with np.linspace to get
        # a range of floating point values
        a1 = np.linspace(-1, 1, rows * cols).astype(dtype)
        a2 = np.linspace(-1024, 1024, rows * cols).astype(dtype)
    if dtype == np.complex128:
        # If the dtype is complex, add an imaginary component to each number
        a1 = (a1 + 1j * (np.arange(rows * cols) % 9 + 4)) ** 2
        a2 = (a1 + 1j * (np.arange(rows * cols) % 9 - 4)) ** 2
    # Reshape the arrays from 1D arrays to 2D arrays with the desires
    # shapes, convert to the desired layout, then convert to matrices
    if layout == "C":
        a1 = a1.reshape((rows, cols))
        a2 = a2.reshape((cols, rows))
    elif layout == "F":
        a1 = a1.reshape((cols, rows)).T
        a2 = a2.reshape((rows, cols)).T
    else:
        a1 = a1.reshape((rows, cols))
        a1 = np.hstack([a1[:, 0:1], a1])
        a1 = np.vstack([a1[0:1, :], a1])
        a1 = a1[1:, 1:]
        a2 = a2.reshape((cols, rows))
        a2 = np.hstack([a2[:, 0:1], a2])
        a2 = np.vstack([a2[0:1, :], a2])
        a2 = a2[1:, 1:]

    m1 = np.asmatrix(a1)
    m2 = np.asmatrix(a2)
    return m1, m2


@pytest.fixture(
    params=[
        pytest.param(np.uint8, id="uint8", marks=pytest.mark.slow),
        pytest.param(np.int32, id="int32", marks=pytest.mark.slow),
        pytest.param(np.float64, id="float64"),
        pytest.param(np.complex128, id="complex128", marks=pytest.mark.slow),
    ]
)
def testing_matrices(request, matrix_shape, matrix_layout):
    """
    For each combination of dtype and matrix dimensions,
    returns 2 matrices of that dtype where all of the values
    are distinct. One of the values has the dimensions specified,
    the other has the dimensions but flipped.

    E.g. if matrix_dtype=uint8 and matrix_shape=(1, 3) the funciton will
    return a 1x3 and a 3x1 matrix of dtype uint8.

    The matrices will have the data layout as described by matrix_layout.
    """
    return make_matrix(request.param, matrix_shape, matrix_layout)


@pytest.fixture(
    params=[
        pytest.param(
            np.uint8,
            id="uint8",
            marks=pytest.mark.skip(
                reason="[BSE-924] TODO: support matrix multiplication on integers"
            ),
        ),
        pytest.param(
            np.int32,
            id="int32",
            marks=pytest.mark.skip(
                reason="[BSE-924] TODO: support matrix multiplication on integers"
            ),
        ),
        pytest.param(np.float64, id="float64"),
        pytest.param(np.complex128, id="complex128"),
    ]
)
def testing_matrices_for_multiplication(request, matrix_shape, matrix_layout):
    """
    A variant of testing_matrices that skips integers since
    matrix multiplication is unsupported on integer types.
    """
    return make_matrix(request.param, matrix_shape, matrix_layout)


def test_matrix_unboxing(testing_matrices, memory_leak_check):
    """
    Tests that the type np.matrix can be safely unboxed
    """

    def impl(m):
        return True

    m1, m2 = testing_matrices

    check_func(impl, (m1,))
    check_func(impl, (m2,))


def test_matrix_boxing(testing_matrices, memory_leak_check):
    """
    Tests that the type np.matrix can be safely boxed after it has
    been unboxed.
    """

    def impl(m):
        return m

    m1, m2 = testing_matrices

    check_func(impl, (m1,))
    check_func(impl, (m2,))


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param("transpose"),
        pytest.param("length"),
        pytest.param("shape"),
        pytest.param("ndim"),
        pytest.param("addition"),
        pytest.param("subtraction"),
    ],
)
def test_matrix_ops(operation, testing_matrices, memory_leak_check):
    def impl1(m1, m2):
        return m1.T, m2.T

    def impl2(m1, m2):
        return m1.shape, m2.shape

    def impl3(m1, m2):
        return len(m1), len(m2)

    def impl4(m1, m2):
        return m1.ndim, m2.ndim

    def impl5(m1, m2):
        return m1 + m1, m1 + m2.T, m1.T + m2

    def impl6(m1, m2):
        return m1 - m1, m1 - m2.T, m1.T - m2

    impls = {
        "transpose": impl1,
        "shape": impl2,
        "length": impl3,
        "ndim": impl4,
        "addition": impl5,
        "subtraction": impl6,
    }

    impl = impls[operation]

    check_func(impl, testing_matrices, only_seq=True)


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param("star"),
        pytest.param("at", marks=pytest.mark.slow),
        pytest.param("dot", marks=pytest.mark.slow),
        pytest.param(
            "matmul",
            marks=pytest.mark.skip(
                reason="[BSE-925] TODO: support np.matmul on np.matrix type"
            ),
        ),
        pytest.param("star_transpose_a"),
        pytest.param("star_transpose_b"),
    ],
)
def test_matrix_multiplication(
    operation, testing_matrices_for_multiplication, memory_leak_check
):
    def impl1(m1, m2):
        return m1 * m2

    def impl2(m1, m2):
        return m1 @ m2

    def impl3(m1, m2):
        return np.dot(m1, m2)

    def impl4(m1, m2):
        return np.matmul(m1, m2)

    def impl5(m1, m2):
        return m1 * m1.T

    def impl6(m1, m2):
        return m1.T * m1

    impls = {
        "star": impl1,
        "at": impl2,
        "dot": impl3,
        "matmul": impl4,
        "star_transpose_a": impl5,
        "star_transpose_b": impl6,
    }

    impl = impls[operation]
    # Only star operator has parallel support currently
    if operation == "star":
        check_func(
            impl,
            testing_matrices_for_multiplication,
            distributed=[("m1", 0)],
            is_out_distributed=True,
        )
    else:
        check_func(impl, testing_matrices_for_multiplication, only_seq=True)


def test_matrix_markov(memory_leak_check):
    """
    Tests a specific pattern of matrix multiplication where the
    same square matrix is multiplied by itself multiple times before
    being multiplied by a column matrix.
    """

    def impl(A, pi):
        return pi * A * A * A * A * A * A * A * A

    pi = np.matrix([[0.25, 0.3, 0.2, 0.25]], dtype=np.float64)
    A = np.matrix(
        [
            [0.2, 0.5, 0.2, 0.1],
            [0.4, 0.4, 0.1, 0.1],
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.25, 0.75],
        ],
        dtype=np.float64,
    )
    check_func(impl, (A, pi), convert_to_nullable_float=False, only_seq=True)


@pytest.mark.slow
@pytest.mark.parametrize(
    "dtype_a, dtype_b",
    [
        pytest.param(np.uint8, np.int32, id="uint8-int32"),
        pytest.param(np.int32, np.float64, id="int32-float64"),
        pytest.param(np.float64, np.int16, id="float64-int16"),
        pytest.param(np.float32, np.uint64, id="float32-uint64"),
        pytest.param(np.uint16, np.complex128, id="uint16-complex128"),
        pytest.param(np.complex128, np.float64, id="complex128-float64"),
    ],
)
def test_mixed_type_arithmetic(dtype_a, dtype_b):
    """
    Tests that matrix arithmetic works on matrices of different
    (but compatible) dtypes.
    """

    def impl(A, B):
        return A + B, A - B

    A, _ = make_matrix(dtype_a, (100, 8))
    _, B = make_matrix(dtype_b, (100, 8))
    check_func(impl, (A, B.T), only_seq=True)


def test_conv_matrix(matrix_layout, memory_leak_check):
    """
    Tests converting an array to a matrix and back to an array, with
    multiplications interleaved along the way to ensure that the
    correct type of multiplication is used at each step.
    """

    def impl(A1):
        # This multiplication should be element-wise
        A2 = A1 * A1
        M1 = np.asmatrix(A2)
        # The second multiplication should be true matrix multiplication
        M2 = M1 * M1 * M1
        A3 = np.asarray(M2)
        # And the final multiplication should be element-wise
        A4 = A3 * A3
        return A4

    A, _ = make_matrix(np.float64, (3, 3), matrix_layout)
    # [BSE-986] TODO: investigate parallel support
    check_func(impl, (A,), only_seq=True)


@pytest.mark.parametrize(
    "A",
    [
        pytest.param(np.asmatrix(np.arange(10).reshape(2, 5)), id="matrix"),
        pytest.param(np.linspace(0, 1, 5), id="1d_arr"),
        pytest.param(np.arange(12).reshape(4, 3), id="2d_arr-C"),
        pytest.param(np.linspace(0, 15.5, 32).reshape(8, 4).T, id="2d_arr-F"),
        pytest.param(np.linspace(0, 15.5, 32).reshape(8, 4)[1:, 1:], id="2d_arr-A"),
        pytest.param(np.int64(-1), id="scalar_int"),
        pytest.param(np.float32(3.14), id="scalar_float"),
        pytest.param(np.complex128(1 - 2j), id="scalar_complex"),
    ],
)
def test_asmatrix(A, memory_leak_check):
    """
    Tests calling np.asmatrix directly on a scalar, 1D array, 2D array
    or another matrix.
    """

    def impl(A):
        return np.asmatrix(A)

    # Matrix and 2D arrays input can be parallelized
    test_parallel = isinstance(A, np.matrix) or (
        isinstance(A, np.ndarray) and A.ndim == 2
    )
    check_func(impl, (A,), convert_to_nullable_float=False, only_seq=not test_parallel)


@pytest.mark.parametrize(
    "A",
    [
        pytest.param(np.linspace(0, 1, 5), id="1d_arr"),
        pytest.param(np.int64(-1), id="scalar_int"),
        pytest.param(np.float32(3.14), id="scalar_float"),
        pytest.param(np.complex128(1 - 2j), id="scalar_complex"),
    ],
)
def test_asmatrix_list(A, memory_leak_check):
    """
    Tests calling np.asmatrix on a scalar or 1D array wrapped in a list.
    """

    def impl1(A):
        return np.asmatrix([A])

    def impl2(A):
        return np.asmatrix([A, A, A])

    # [BSE-986] TODO: investigate parallel support
    check_func(impl1, (A,), convert_to_nullable_float=False, only_seq=True)
    check_func(impl2, (A,), convert_to_nullable_float=False, only_seq=True)


def test_asarray(testing_matrices, memory_leak_check):
    """
    Tests calling np.asmatrix on a scalar or 1D array wrapped in a list.
    """

    def impl(M):
        return np.asarray(M)

    # [BSE-986] TODO: investigate parallel support
    M, _ = testing_matrices
    check_func(impl, (M,), only_seq=True)
