import numpy as np
import pytest

from bodo.tests.utils import check_func


@pytest.fixture(
    params=[
        pytest.param(np.uint8, id="uint8"),
        pytest.param(np.int32, id="int32"),
        pytest.param(np.float64, id="float64"),
        pytest.param(np.complex128, id="complex128"),
    ]
)
def matrix_dtype(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param((2, 2), id="2x2"),
        pytest.param((1, 4), id="1x4"),
        pytest.param((20, 13), id="20x13"),
    ]
)
def matrix_shape(request):
    return request.param


@pytest.fixture
def testing_matrices(matrix_dtype, matrix_shape):
    """
    For each combination of dtype and matrix dimensions,
    returns 2 matrices of that dtype where all of the values
    are distinct. One of the values has the dimensions specified,
    the other has the dimensions but flipped.

    E.g. if matrix_dtype=uint8 and matrix_shape=(1, 3) the funciton will
    return a 1x3 and a 3x1 matrix of dtype uint8.
    """
    rows, cols = matrix_shape
    if isinstance(matrix_dtype, np.integer):
        # If the dtype is an integer, create a 1D array with the desired number
        # of total elements containing integers in a range.
        a1 = np.arange(rows * cols).astype(matrix_dtype)
        a2 = np.arange(-rows * cols, rows * cols * 2, 3).astype(matrix_dtype)
    else:
        # If the dtype is a float, do the same but with np.linspace to get
        # a range of floating point values
        a1 = np.linspace(-1, 1, rows * cols).astype(matrix_dtype)
        a2 = np.linspace(-1024, 1024, rows * cols).astype(matrix_dtype)
    if matrix_dtype == np.complex128:
        # If the dtype is complex, add an imaginary component to each number
        a1 = (a1 + 1j * (np.arange(rows * cols) % 9 + 4)) ** 2
        a2 = (a1 + 1j * (np.arange(rows * cols) % 9 - 4)) ** 2
    # Reshape the arrays from 1D arrays to 2D arrays with the desires
    # shapes, then convert to matrices
    m1 = np.asmatrix(a1.reshape((rows, cols)))
    m2 = np.asmatrix(a2.reshape((cols, rows)))
    return m1, m2


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
        pytest.param(
            "multiplication_star",
            marks=pytest.mark.skip(
                "[BSE-695] TODO: support addition, subtraction and multiplication of np.matrix"
            ),
        ),
        pytest.param(
            "multiplication_at",
            marks=pytest.mark.skip(
                "[BSE-695] TODO: support addition, subtraction and multiplication of np.matrix"
            ),
        ),
        pytest.param(
            "multiplication_dot",
            marks=pytest.mark.skip(
                "[BSE-695] TODO: support addition, subtraction and multiplication of np.matrix"
            ),
        ),
        pytest.param(
            "multiplication_matmul",
            marks=pytest.mark.skip(
                "[BSE-695] TODO: support addition, subtraction and multiplication of np.matrix"
            ),
        ),
        pytest.param(
            "addition",
            marks=pytest.mark.skip(
                "[BSE-695] TODO: support addition, subtraction and multiplication of np.matrix"
            ),
        ),
        pytest.param(
            "subtraction",
            marks=pytest.mark.skip(
                "[BSE-695] TODO: support addition, subtraction and multiplication of np.matrix"
            ),
        ),
        pytest.param("ndim"),
    ],
)
def test_matrix_ops(operation, testing_matrices, memory_leak_check):
    def impl1(m1, m2):
        return m1.T, m2.T

    def impl2(m1, m2):
        return m1.shape, m2.shape

    def impl3(m1, m2):
        return len(m1), len(m2)

    def impl4A(m1, m2):
        return m1 * m2

    def impl4B(m1, m2):
        return m1 @ m2

    def impl4C(m1, m2):
        return np.dot(m1, m2)

    def impl4D(m1, m2):
        return np.matmul(m1, m2)

    def impl5(m1, m2):
        return m1 + m2.T

    def impl6(m1, m2):
        return m1 - m2.T

    def impl7(m1, m2):
        return m1.ndim, m2.ndim

    impls = {
        "transpose": impl1,
        "shape": impl2,
        "length": impl3,
        "multiplication_star": impl4A,
        "multiplication_at": impl4B,
        "multiplication_dot": impl4C,
        "multiplication_matmul": impl4D,
        "addition": impl5,
        "subtraction": impl6,
        "ndim": impl7,
    }

    impl = impls[operation]

    check_func(impl, testing_matrices)


@pytest.mark.skip(
    reason="[BSE-966] Ensure that np.asmatrix and np.asarray correctly convert between the two types"
)
def test_conversion(memory_leak_check):
    def impl(A1):
        # This multiplication should be element-wise
        A2 = A1 * A1
        M1 = np.asmatrix(A2)
        # The second multiplication should be true matrix multiplication
        M2 = M1 * M1
        A3 = np.asarray(M2)
        # And the final multiplication should be element-wise
        A4 = A3 * A3
        return A4

    check_func(impl, (np.arange(9).reshape(3, 3),))
