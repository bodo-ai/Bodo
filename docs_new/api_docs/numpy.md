# Numpy Operations {#numpy}

Below is the list of the data-parallel Numpy operators that Bodo can
optimize and parallelize.


## Numpy *element-wise* array operations

### Unary operators 

- `+` 
- `-`
- `~`

### Binary operators
- `+` 
- `-` 
- `*` 
- `/` 
- `/?`
- `%` 
- `|` 
- `>>` 
- `^` 
- `<<`
- `&` 
- `**` 
- `//`

### Comparison operators 

- `==`
- `!=`
- `<` 
- `<=` 
- `>` 
- `>=`

### Data-parallel math operations

- `numpy.add`
- `numpy.subtract`
- `numpy.multiply`
- `numpy.divide`
- `numpy.logaddexp`
- `numpy.logaddexp2`
- `numpy.true_divide`
- `numpy.floor_divide`
- `numpy.negative`
- `numpy.positive`
- `numpy.power`
- `numpy.remainder`
- `numpy.mod`
- `numpy.fmod`
- `numpy.abs`
- `numpy.absolute`
- `numpy.fabs`
- `numpy.rint`
- `numpy.sign`
- `numpy.conj`
- `numpy.exp`
- `numpy.exp2`
- `numpy.log`
- `numpy.log2`
- `numpy.log10`
- `numpy.expm1`
- `numpy.log1p`
- `numpy.sqrt`
- `numpy.square`
- `numpy.reciprocal`
- `numpy.gcd`
- `numpy.lcm`
- `numpy.conjugate`

### Trigonometric functions

 - `numpy.sin`
 - `numpy.cos`
 - `numpy.tan`
 - `numpy.arcsin`
 - `numpy.arccos`
 - `numpy.arctan`
 - `numpy.arctan2`
 - `numpy.hypot`
 - `numpy.sinh`
 - `numpy.cosh`
 - `numpy.tanh`
 - `numpy.arcsinh`
 - `numpy.arccosh`
 - `numpy.arctanh`
 - `numpy.deg2rad`
 - `numpy.rad2deg`
 - `numpy.degrees`
 - `numpy.radians`
 
 
### Bit manipulation functions

- `numpy.bitwise_and`
- `numpy.bitwise_or`
- `numpy.bitwise_xor`
- `numpy.bitwise_not`
- `numpy.invert`
- `numpy.left_shift`
- `numpy.right_shift`
 
 
### Comparison functions

- `numpy.logical_and`
- `numpy.logical_or`
- `numpy.logical_xor`
- `numpy.logical_not`

### Floating functions

- `numpy.isfinite`
- `numpy.isinf`
- `numpy.signbit`
- `numpy.ldexp`
- `numpy.floor`
- `numpy.ceil`
- `numpy.trunc`

## Numpy reduction functions

- `numpy.sum`
- `numpy.prod`
- `numpy.min`
- `numpy.max`
- `numpy.argmin`
- `numpy.argmax`
- `numpy.all`
- `numpy.any`

## Numpy array creation functions

- `numpy.empty`
- `numpy.identity`
- `numpy.zeros`
- `numpy.ones`
- `numpy.empty_like`
- `numpy.zeros_like`
- `numpy.ones_like`
- `numpy.full_like`
- `numpy.array`
- `numpy.asarray`
- `numpy.copy`
- `numpy.arange`
- `numpy.linspace`
- `numpy.repeat`  only scalar `num_repeats`

### Numpy array manipulation functions

- `numpy.shape`
- `numpy.reshape`   
  
    `shape` values cannot be -1.
   
- `numpy.sort`
- `numpy.concatenate`
- `numpy.append`
- `numpy.unique`  The output is assumed to be "small" relative to input and is replicated.
                  Use `Series.drop_duplicates()` if the output should remain distributed.

- `numpy.where` (1 and 3 arguments)
- `numpy.select`  The default value for numeric/boolean types is `0/False`. For all other
                  types, the default is `pd.NA`. If any of the values in
                  `choicelist` are nullable, or the default is `pd.NA` or `None`, the
                  output will be a nullable pandas array instead of a numpy
                  array.  
- `numpy.union1d`
- `numpy.intersect1d`  no distributed support yet
- `numpy.setdiff1d`  no distributed support yet
- `numpy.hstack`  concatenates elements on each rank without maintaining order

## Numpy mathematical and statistics functions

- `numpy.cumsum`
- `numpy.diff`
- `numpy.percentile`
- `numpy.quantile`
- `numpy.median`
- `numpy.mean`
- `numpy.std`

## Random number generator functions

- `numpy.random.rand`
- `numpy.random.randn`
- `numpy.random.ranf`
- `numpy.random.random_sample`
- `numpy.random.sample`
- `numpy.random.random`
- `numpy.random.standard_normal`
- `numpy.random.multivariate_normal` (must provide size)
- `numpy.random.chisquare`
- `numpy.random.weibull`
- `numpy.random.power`
- `numpy.random.geometric`
- `numpy.random.exponential`
- `numpy.random.poisson`
- `numpy.random.rayleigh`
- `numpy.random.normal`
- `numpy.random.uniform`
- `numpy.random.beta`
- `numpy.random.binomial`
- `numpy.random.f`
- `numpy.random.gamma`
- `numpy.random.lognormal`
- `numpy.random.laplace`
- `numpy.random.randint`
- `numpy.random.triangular`

## `numpy.dot` function 

- `numpy.dot` between a matrix and a vector
- `numpy.dot`two vectors.


##  Numpy I/O

- `numpy.ndarray.tofile` 
- `numpy.fromfile`

Our scalable I/O section contains [example usage and more system specific instructions][numpy-binary-section].


##  Miscellaneous

- Numpy array comprehension : e.g. : A = np.array([i**2 for i in range(N)])
  
  
!!! note

    Optional arguments are not supported unless if explicitly mentioned
    here. For operations on multi-dimensional arrays, automatic broadcast of
    dimensions of size 1 is not supported.

## Numpy dot() Parallelization

The `np.dot` function has different distribution rules based
on the number of dimensions and the distributions of its input arrays.
The example below demonstrates two cases:

```py
@bodo.jit
def example_dot(N, D):
    X = np.random.ranf((N, D))
    Y = np.random.ranf(N)
    w = np.dot(Y, X)
    z = np.dot(X, w)
    return z.sum()

example_dot(1024, 10)
example_dot.distributed_diagnostics()
```

Here is the output of `distributed_diagnostics()`:

```console

Data distributions:
  $X.130               1D_Block
  $Y.131               1D_Block
  $b.2.158             REP

Parfor distributions:
  0                    1D_Block
  1                    1D_Block
  3                    1D_Block

Distributed listing for function example_dot, ../tmp/dist_rep.py (4)
++++++++++++++++++++++++++++++++++| parfor_id/variable: distribution
@bodo.jit                         |
def example_dot(N, D):            |
    X = np.random.ranf((N, D))++++| #0: 1D_Block, $X.130: 1D_Block
    Y = np.random.ranf(N)++++++++-| #1: 1D_Block, $Y.131: 1D_Block
    w = np.dot(Y, X)++++++++++++++| $b.2.158: REP
    z = np.dot(X, w)++++++++++++++| #3: 1D_Block
    return z.sum()                |
```

The first `dot` has a 1D array with `1D_Block`
distribution as first input `Y`), while the second input is
a 2D array with `1D_Block` distribution (`X`).
Hence, `dot` is a sum reduction across distributed datasets
and therefore, the output (`w`) is on the `reduce` side and is 
assigned `REP` distribution.


The second `dot` has a 2D array with `1D_Block`
distribution (`X`) as first input, while the second input is
a REP array (`w`). Hence, the computation is data-parallel
across rows of `X`, which implies a `1D_Block`
distribution for output (`z`).


Variable `z` does not exist in the distribution report since
the compiler optimizations were able to eliminate it. Its values are
generated and consumed on-the-fly, without memory load/store overheads.


[comment]: <> (autoref for [numpy-binary-section] will be populated as the section is added) 
[todo]: <> (remove above comment when [numpy-binary-section] section is added) 
