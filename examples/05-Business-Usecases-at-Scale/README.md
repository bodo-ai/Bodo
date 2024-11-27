# Using Bodo to answer business questions in a fraction of the time

This module demonstrates how bodo can be leveraged to answer a diverse range of business questions using familar pandas APIs. Using bodo's compiler and automatic parallelization capabilities, the code can be kept relatively simple, even when the datasets scale to enormous sizes. 

## Running bodo in the terminal

In [7-Terminal-examples](7-Terminal-examples), a subset of the notebooks in this module have been transformed into python scripts if you would prefer running bodo from your terminal.

You can run these scripts like a normal python program and bodo will automatically execute functions decorated with `@bodo.jit` in parallel. By default these functions will be executed on all availible cores. To run on fewer cores, you can use the environment variable `BODO_NUM_WORKERS`. For example, to run the script `beer_review.py` on 4 cores, use:

``` shell
export BODO_NUM_WORKERS=4; python beer_review.py
```

Note that the code outside bodo jit functions will still be executed on a single core as in regular python programs.