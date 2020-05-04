.. _numpy:


Supported Numpy Operations
--------------------------

Below is the list of the data-parallel Numpy operators that Bodo can optimize
and parallelize.

#. Numpy `element-wise` array operations:

    * Unary operators: ``+`` ``-`` ``~``
    * Binary operators: ``+`` ``-`` ``*`` ``/`` ``/?`` ``%`` ``|`` ``>>`` ``^``
      ``<<`` ``&`` ``**`` ``//``
    * Comparison operators: ``==`` ``!=`` ``<`` ``<=`` ``>`` ``>=``
    * data-parallel math operations: ``add``, ``subtract``, ``multiply``,
      ``divide``, ``logaddexp``, ``logaddexp2``, ``true_divide``,
      ``floor_divide``, ``negative``, ``power``, ``remainder``,
      ``mod``, ``fmod``, ``abs``, ``absolute``, ``fabs``, ``rint``, ``sign``,
      ``conj``, ``exp``, ``exp2``, ``log``, ``log2``, ``log10``, ``expm1``,
      ``log1p``, ``sqrt``, ``square``, ``reciprocal``, ``conjugate``
    * Trigonometric functions: ``sin``, ``cos``, ``tan``, ``arcsin``,
      ``arccos``, ``arctan``, ``arctan2``, ``hypot``, ``sinh``, ``cosh``,
      ``tanh``, ``arcsinh``, ``arccosh``, ``arctanh``, ``deg2rad``,
      ``rad2deg``, ``degrees``, ``radians``
    * Bit manipulation functions: ``bitwise_and``, ``bitwise_or``,
      ``bitwise_xor``, ``bitwise_not``, ``invert``, ``left_shift``,
      ``right_shift``

#. Numpy reduction functions ``sum``, ``prod``, ``min``, ``max``, ``argmin``
   and ``argmax``. Currently, `int64` data type is not supported for
   ``argmin`` and ``argmax``.

#. Numpy array creation functions ``empty``, ``zeros``, ``ones``,
   ``empty_like``, ``zeros_like``, ``ones_like``, ``full_like``, ``copy``,
   ``arange`` and ``linspace``.

#. Numpy array ``reshape`` (shape values cannot be -1).

#. Random number generator functions: ``rand``, ``randn``,
   ``ranf``, ``random_sample``, ``sample``, ``random``,
   ``standard_normal``, ``chisquare``, ``weibull``, ``power``, ``geometric``,
   ``exponential``, ``poisson``, ``rayleigh``, ``normal``, ``uniform``,
   ``beta``, ``binomial``, ``f``, ``gamma``, ``lognormal``, ``laplace``,
   ``randint``, ``triangular``.

#. Numpy ``dot`` function between a matrix and a vector, or two vectors.

#. Numpy array comprehensions, such as::

    A = np.array([i**2 for i in range(N)])

Optional arguments are not supported unless if explicitly mentioned here.
For operations on multi-dimensional arrays, automatic broadcast of
dimensions of size 1 is not supported.

Numpy dot() Parallelization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `np.dot` function has different distribution rules based on the number of
dimensions and the distributions of its input arrays. The example below
demonstrates two cases::

    @bodo.jit
    def example_dot(N, D):
        X = np.random.ranf((N, D))
        Y = np.random.ranf(N)
        w = np.dot(Y, X)
        z = np.dot(X, w)
        return z.sum()

    example_dot(1024, 10)
    example_dot.distributed_diagnostics()

Here is the output of `distributed_diagnostics()`::

    Data distributions:
      $X.130               1D_Block
      $Y.131               1D_Block
      $b.2.158             REP

    Parfor distributions:
      0                    1D_Block
      1                    1D_Block
      3                    1D_Block

    Distributed listing for function example_dot, ../tmp/dist_rep.py (4)
    ----------------------------------| parfor_id/variable: distribution
    @bodo.jit                         |
    def example_dot(N, D):            |
        X = np.random.ranf((N, D))----| #0: 1D_Block, $X.130: 1D_Block
        Y = np.random.ranf(N)---------| #1: 1D_Block, $Y.131: 1D_Block
        w = np.dot(Y, X)--------------| $b.2.158: REP
        z = np.dot(X, w)--------------| #3: 1D_Block
        return z.sum()                |

The first `dot` has a 1D array with `1D_Block` distribution as first input
(`Y`), while the second input is a 2D array with `1D_Block` distribution (`X`).
Hence, `dot` is a sum reduction across distributed datasets and therefore,
the output (`w`) is on the `reduce` side and is assigned `REP` distribution.

The second `dot` has a 2D array with `1D_Block` distribution (`X`) as first
input, while the second input is a REP array (`w`). Hence, the computation is
data-parallel across rows of `X`, which implies a `1D_Block` distribution for
output (`z`).

Variable `z` does not exist in the distribution report since
the compiler optimizations were able to eliminate it. Its values are generated
and consumed on-the-fly, without memory load/store overheads.
