# CORR

`#!sql CORR(Y, X)`

Compute the correlation over the window of both inputs, or `NULL` if
the window is empty. Equivalent to `#!sql COVAR(Y, X) / (STDDEV_POP(Y) * STDDEV_POP(X))`
