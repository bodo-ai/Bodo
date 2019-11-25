.. _install:


Debugging the C++ code
----------------------

In order to debug C++ code, the method is to use sanitizers of the C++ and C
compiler of GCC.
The compilation option need to be added to the `setup.py` initialization program::

    eca = ["-std=c++11", "-fsanitize=address"]
    ela = ["-std=c++11", "-fsanitize=address"]

In the docker, the next step is to do add the library::

    export LD_PRELOAD=/root/miniconda3/envs/BODODEV/lib/libasan.so.5

Then we can see running times error using sanitizers.
