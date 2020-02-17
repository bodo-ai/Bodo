.. _dev_debugging:

Debugging
---------

Debugging the Python code
~~~~~~~~~~~~~~~~~~~~~~~~~~
- `pdb <https://docs.python.org/3/library/pdb.html>`_: :code:`import pdb; pdb.set_trace()` for breakpoints

- `NUMBA_DEBUG_PRINT_AFTER <https://numba.pydata.org/numba-doc/dev/reference/envvars.html?highlight=numba_debug_print#envvar-NUMBA_DEBUG_PRINT_AFTER>`_
  enviroment variable::

    # example of printing after parfor pass
    export NUMBA_DEBUG_PRINT_AFTER='parfor_pass'
    # other common ones: 'bodo_distributed_pass', 'bodo_series_pass'

- mpiexec redirect stdout from differet processes to different files::

    export PYTHONUNBUFFERED=1 # set the enviroment variable
    mpiexec -outfile-pattern="out_%r.log" -n 8 python small_test01.py

  or::

    # use the flag instead of setting the enviroment variable
    mpiexec -outfile-pattern="out_%r.log" -n 8 python -u small_test01.py


Debugging the C++ code
~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to debug C++ code, the method is to use sanitizers of the C++ and C
compiler of GCC.
The compilation option need to be added to the `setup.py` initialization program::

    eca = ["-std=c++11", "-fsanitize=address"]
    ela = ["-std=c++11", "-fsanitize=address"]

In the docker, the next step is to do add the library::

    export LD_PRELOAD=/root/miniconda3/envs/BODODEV/lib/libasan.so.5

Then we can see running times error using sanitizers.
