.. _performance:

Performance Measurement
=======================

This section provides tips on performance measurement of Bodo programs. It is important to keep the following in mind when measuring program run time:

    #. Every program has some overhead, so large data sets may be necessary for useful measurements.
    #. Performance can vary from one run to another. Several measurements are always needed.
    #. It is important to use a sequence of tests with increasing input size, which helps understand the impact of problem size on program performance.
    #. It is useful to use simple programs to study performance factors. Complex program are impacted by multiple factors and their performance is harder to understand.
    #. Longer computations typically provide more reliable run time information.


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

And the time it takes can be compared with python performance. Here, we have a :code:`5.06/0.57 ~= 9x` speedup
(from parallelism and sequential optimizations).



We can add multiple timers inside a function to see how much time each section takes::

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


Expected Speedup
~~~~~~~~~~~~~~~~

The speed up achieved using Bodo depends on various factors such as problem size,
parallel overheads of the operations, and the hardware platform's attributes.
For example, the program above can scale almost linearly
(e.g. 100 speed up on 100 cores)
for large enough problem sizes, since the only communication overhead is
parallel summation of the partial sums obtained by `np.sum` on each processor.
On the other hand, some operations such as join and groupby operations
require significantly larger communication of data, requiring fast cluster
interconnection networks to scale to large number of cores.
