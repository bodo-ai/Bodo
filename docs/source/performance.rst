.. _performance:

Performance Measurement
===========================

In this section, we show some tips on how Bodo's performance can be measured.

.. note::  Make sure the computation is long enough to have realistic measurement.

Since Bodo decorated functions are `JIT-compiled <https://numba.pydata.org/numba-doc/dev/reference/jit-compilation.html>`_, the compilation time is non-negligible but it only happens once. To avoid measuring compilation time, place timers inside the functions. For example::

	"""
	calc_pi.py: computes the value of Pi using Monte-Carlo Integration
	"""

	import numpy as np
	import bodo
	import time

	n = 2 * 10**8

	def calc_pi(n):
	    t1 = time.time()
	    x = 2 * np.random.ranf(n) - 1
	    y = 2 * np.random.ranf(n) - 1
	    pi = 4 * np.sum(x**2 + y**2 < 1) / n
	    print("Execution time:", time.time()-t1, "\nresult:", pi)
	    return pi

	bodo_calc_pi = bodo.jit(calc_pi)
	print("python:")
	calc_pi(n)
	print("bodo:")
	bodo_calc_pi(n)

The output of this code is as follows::

	python:
	Execution time: 5.060443162918091
	result: 3.14165914
	bodo:
	Execution time: 2.165610068012029
	result: 3.14154512

------------------

Bodo's parallel speedup can be measured similarly::

	"""	
	calc_pi.py: computes the value of Pi using Monte-Carlo Integration	
	"""	

	import numpy as np	
	import bodo	
	import time	

	@bodo.jit	
	def calc_pi(n):	
	    t1 = time.time()	
	    x = 2 * np.random.ranf(n) - 1	
	    y = 2 * np.random.ranf(n) - 1	
	    pi = 4 * np.sum(x**2 + y**2 < 1) / n	
	    print("Execution time:", time.time()-t1, "\nresult:", pi)	
	    return pi	

	calc_pi(2 * 10**8)

Launched on four parallel cores::

    $ mpiexec -n 4 python calc_pi.py
	Execution time: 0.5736249439651147
	result: 3.14161474

And the time it takes can be compared with python performance. Here, we have a :code:`5.06/0.57 ~= 9x` parallel speedup.

------------------

You can also have multiple timers inside a function to see how much time each section takes::

	"""
	calc_pi.py: computes the value of Pi using Monte-Carlo Integration
	"""

	import numpy as np
	import bodo
	import time

	n = 2 * 10**8

	def calc_pi(n):
	    t1 = time.time()
	    x = 2 * np.random.ranf(n) - 1
	    y = 2 * np.random.ranf(n) - 1
	    t2 = time.time()
	    print("Initializing x,y takes: ", t2-t1)

	    pi = 4 * np.sum(x**2 + y**2 < 1) / n
	    print("calculation takes:", time.time()-t2, "\nresult:", pi)
	    return pi

	bodo_calc_pi = bodo.jit(calc_pi)
	print("python: ------------------")
	calc_pi(n)
	print("bodo: ------------------")
	bodo_calc_pi(n)

The output is as follows::

	python: ------------------
	Initializing x,y takes:  3.9832258224487305
	calculation takes: 1.1460411548614502
	result: 3.14156454
	bodo: ------------------
	Initializing x,y takes:  3.0611653940286487
	calculation takes: 0.35728363902308047
	result: 3.14155538


.. note::
    Note that Bodo execution took longer in the last example than previous ones,
    since the presence of timers in the middle of computation can inhibit some code
    optimizations (e.g. code reordering and fusion). Therefore, one should be
    cautious about adding timers in the middle of computation.


.. _disable-jit:

Disabling JIT Compilation
~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes it is convenient to disable JIT compilation
without removing the `jit` decorators in the code,
to enable easy performance comparison with regular Python or
perform debugging.
This can be done by setting the environment variable
`NUMBA_DISABLE_JIT` to `1`,
which makes the `jit` decorators act as if they
perform no operation. In this case, the invocation of decorated
functions calls the original Python functions instead of compiled versions.
