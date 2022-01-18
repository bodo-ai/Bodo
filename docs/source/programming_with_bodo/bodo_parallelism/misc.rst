.. _parallelism_misc:

Miscellaneous Resources
=======================

.. _run_on_single_rank:

Run code on a single rank
-------------------------

In cases where some code needs to be run on a single MPI rank, you can do so in a python script as follows::

    if bodo.get_rank() == 0:
        # Remove directory
        import os, shutil
        if os.path.exists("data/data.pq"):
            shutil.rmtree("data/data.pq")
    
    # To synchronize all ranks before proceeding
    bodo.barrier()

When running code on an IPyParallel cluster using the ``%%px`` magic, you can do this instead:

.. code:: ipython3

    %%px --targets 0
    # Install package
    !conda install pandas-datareader


An alias can be defined for convenience:

.. code:: ipython3

    %alias_magic p0 px -p "--targets 0"

This can be used as any other magic:

.. code:: ipython3

    %%p0
    # Install package
    !conda install pandas-datareader


.. _run_on_each_node:

Run code once on each node
--------------------------

In cases where some code needs to be run once on each node in a multi-node cluster,
such as a file system operation, installing packages, etc., it can be done as follows::

    if bodo.get_rank() in bodo.get_nodes_first_ranks():
        # Remove directory on all nodes
        import os, shutil
        if os.path.exists("data/data.pq"):
            shutil.rmtree("data/data.pq")
    
    # To synchronize all ranks before proceeding
    bodo.barrier()

The same can be done when running on an IPyParallel cluster using the ``%%px`` magic:

.. code:: ipython3

    %%px
    if bodo.get_rank() in bodo.get_nodes_first_ranks():
        # Install package on all nodes
        !conda install pandas-datareader


.. warning::

    Running code on a single rank or a subset of ranks can lead to deadlocks.
    Ensure that your code doesn't include any MPI or Bodo functions.
