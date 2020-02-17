.. _dev_sentinel_functions:

Sentinel Functions
------------------

Bodo transforms Pandas APIs (and others if needed) into *sentinel*
functions that can be analyzed and optimized throughout the pipeline.
Different stages of the compiler handle these functions if necessary,
with all the analysis for them available if needed.

For example, ``get_series_data`` function is used for getting the underlying
data array of a Series object. BodoSeriesPass removes this function
if the data array is available at that point in the program
(Series object was created using ``init_series`` and not altered).


For the pipline to handle a sentinel function properly
the following has to be specified:

- side effects for dead code elimination
- aliasing
- inlining (if necessary)
- array analysis
- distributed analysis (including array access analysis)
- distributed transformation

For example, ``get_series_data`` does not have side effects and can be removed
if output is not live. In addition, the output is aliased with the input,
and both have the same parallel distribution.
