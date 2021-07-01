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
    * data-parallel math operations:

      * :func:`numpy.add` 
      * :func:`numpy.subtract`
      * :func:`numpy.multiply`
      * :func:`numpy.divide`
      * :func:`numpy.logaddexp`
      * :func:`numpy.logaddexp2`
      * :func:`numpy.true_divide`
      * :func:`numpy.floor_divide`
      * :func:`numpy.negative`
      * :func:`numpy.positive`
      * :func:`numpy.power`
      * :func:`numpy.remainder`
      * :func:`numpy.mod`
      * :func:`numpy.fmod`
      * :func:`numpy.abs`
      * :func:`numpy.absolute`
      * :func:`numpy.fabs`
      * :func:`numpy.rint`
      * :func:`numpy.sign`
      * :func:`numpy.conj`
      * :func:`numpy.exp`
      * :func:`numpy.exp2`
      * :func:`numpy.log`
      * :func:`numpy.log2`
      * :func:`numpy.log10`
      * :func:`numpy.expm1`
      * :func:`numpy.log1p`
      * :func:`numpy.sqrt`
      * :func:`numpy.square`
      * :func:`numpy.reciprocal`
      * :func:`numpy.gcd`
      * :func:`numpy.lcm`
      * :func:`numpy.conjugate`
      

    * Trigonometric functions: 
    
      * :func:`numpy.sin`
      * :func:`numpy.cos`
      * :func:`numpy.tan`
      * :func:`numpy.arcsin`
      * :func:`numpy.arccos`
      * :func:`numpy.arctan`
      * :func:`numpy.arctan2`
      * :func:`numpy.hypot`
      * :func:`numpy.sinh`
      * :func:`numpy.cosh`
      * :func:`numpy.tanh`
      * :func:`numpy.arcsinh`
      * :func:`numpy.arccosh`
      * :func:`numpy.arctanh`
      * :func:`numpy.deg2rad`
      * :func:`numpy.rad2deg`
      * :func:`numpy.degrees`
      * :func:`numpy.radians`     

    * Bit manipulation functions: 
    
      * :func:`numpy.bitwise_and`
      * :func:`numpy.bitwise_or`
      * :func:`numpy.bitwise_xor`
      * :func:`numpy.bitwise_not`
      * :func:`numpy.invert`
      * :func:`numpy.left_shift`
      * :func:`numpy.right_shift`

    * Comparison functions:

      * :func:`numpy.logical_and`
      * :func:`numpy.logical_or`
      * :func:`numpy.logical_xor`
      * :func:`numpy.logical_not`

    * Floating functions:

      * :func:`numpy.isfinite`
      * :func:`numpy.isinf`
      * :func:`numpy.signbit`
      * :func:`numpy.ldexp`
      * :func:`numpy.floor`
      * :func:`numpy.ceil`
      * :func:`numpy.trunc`



#. Numpy reduction functions:

      * :func:`numpy.sum`
      * :func:`numpy.prod`
      * :func:`numpy.min`
      * :func:`numpy.max`
      * :func:`numpy.argmin`
      * :func:`numpy.argmax`
      * :func:`numpy.all`
      * :func:`numpy.any`

#. Numpy array creation functions:

    * :func:`numpy.empty`
    * :func:`numpy.identity`
    * :func:`numpy.zeros`
    * :func:`numpy.ones`
    * :func:`numpy.empty_like`
    * :func:`numpy.zeros_like`
    * :func:`numpy.ones_like`
    * :func:`numpy.full_like`
    * :func:`numpy.array`
    * :func:`numpy.asarray`
    * :func:`numpy.copy`
    * :func:`numpy.arange`
    * :func:`numpy.linspace`
    * :func:`numpy.repeat` (only scalar `num_repeats`)


#. Numpy array manipulation functions:

    * :func:`numpy.shape`
    * :func:`numpy.reshape` (shape values cannot be -1).
    * :func:`numpy.sort`
    * :func:`numpy.concatenate`
    * :func:`numpy.append`
    * :func:`numpy.unique` the output is assumed to be "small" relative to input and is replicated. Use Series.drop_duplicates() if the output should remain distributed.
    * :func:`numpy.where` (1 and 3 arguments)
    * :func:`numpy.union1d`
    * :func:`numpy.intersect1d` (no distributed support yet)
    * :func:`numpy.setdiff1d` (no distributed support yet)
    * :func:`numpy.hstack` (concatenates elements on each rank without maintaining order) 


#. Numpy mathematical and statistics functions:

      * :func:`numpy.cumsum`
      * :func:`numpy.diff`
      * :func:`numpy.percentile`
      * :func:`numpy.quantile`
      * :func:`numpy.median`
      * :func:`numpy.mean`
      * :func:`numpy.std`


#. Random number generator functions:

    * :func:`numpy.random.rand`
    * :func:`numpy.random.randn`
    * :func:`numpy.random.ranf`
    * :func:`numpy.random.random_sample`
    * :func:`numpy.random.sample`
    * :func:`numpy.random.random`
    * :func:`numpy.random.standard_normal`
    * :func:`numpy.random.multivariate_normal` (must provide size)
    * :func:`numpy.random.chisquare`
    * :func:`numpy.random.weibull`
    * :func:`numpy.random.power`
    * :func:`numpy.random.geometric`
    * :func:`numpy.random.exponential`
    * :func:`numpy.random.poisson`
    * :func:`numpy.random.rayleigh`
    * :func:`numpy.random.normal`
    * :func:`numpy.random.uniform`
    * :func:`numpy.random.beta`
    * :func:`numpy.random.binomial`
    * :func:`numpy.random.f`
    * :func:`numpy.random.gamma`
    * :func:`numpy.random.lognormal`
    * :func:`numpy.random.laplace`
    * :func:`numpy.random.randint`
    * :func:`numpy.random.triangular`

#. :func:`numpy.dot` function between a matrix and a vector, or two vectors.

#. Numpy array comprehensions, such as::

    A = np.array([i**2 for i in range(N)])

#. Numpy I/O: :func:`numpy.ndarray.tofile` and :func:`numpy.fromfile`. The File I/O section contains :ref:`example usage and more system specific instructions <numpy-binary-section>`.


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
