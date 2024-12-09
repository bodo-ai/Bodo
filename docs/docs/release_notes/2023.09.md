Bodo 2023.9 Release (Date: 09/01/2023) {#September_2023}
========================================

## New Features and Improvements

Compilation / Performance improvements:

- BodoSQL generated plans have been further optimized to reduce runtime, compile time, and memory usage.
- Performance and compilation time improvements to several window functions:
    - `LEAD`
    - `LAG`
    - `AVG`
    - `VARIANCE_POP` and equivalent functions
    - `VARIANCE_SAMP` and equivalent functions
    - `STDDEV_POP` and equivalent functions
    - `STDDEV_SAMP` and equivalent functions
    - `FIRST_VALUE`
    - `LAST_VALUE`
    - `RATIO_TO_REPORT`


Python coverage:

- Partial support for the `np.matrix` type with the following functionality (non-distributed):
    - `np.asmatrix` to convert a scalar, 1D array, 2D array, list of scalars, or list of 1D arrays to a matrix
    - `np.asarray` to convert a matrix to a 2D array
    - Addition and subtraction with `+` and `-`
    - Matrix multiplication with `*`, `@` or `np.dot`
    - Calling `len()` on a Matrix
    - Using `.ndim`, `.shape` and `.T` (non-distributed) on a Matrix

- Support for the following Numpy functions:
    - `np.interp` non-distributed (added support for keyword arguments `left` and `right`)
    - `np.tile` (added support for specific patterns, see [Numpy][numpy] docs)
    - `np.linalg.norm` (added support for keyword argument `axis=1` when the input is a 2D array)
    - `np.nan_to_num`
    - `np.dot` (added support for heterogeneous typing between integer & float array inputs)
    - `scipy.fftpack.fftshift` (non-distributed)
    - `scipy.fftpack.fft2` (non-distributed)


BodoSQL:

- Added support for `HASH(*)`
- Added support for `PERCENTILE_CONT` and `PERCENTILE_DISC` (non-window support)

### 2023.9.5 New Features and Improvements


Compilation / Performance improvements:

- BodoSQL generated plans have been further optimized to reduce runtime and memory usage.
- Support for executing `UNION` in vectorized mode
- Support for executing `ARRAY_AGG` on numeric types in a `GROUP BY`

### 2023.9.6 New Features and Improvements

Fix critical bugs in vectorized execution mode.

- BodoSQL generated plans have been further optimized to reduce runtime and memory usage.
- `GET_PATH` and JSON field accesses via `:` are supported in some usages.
