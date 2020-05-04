.. _jupyter:

Jupyter Notebook Setup
######################

To use Bodo with Jupyter Notebook, install ``jupyter`` and ``ipyparallel``
in your Bodo Conda environment::

    conda install jupyter ipyparallel -c conda-forge

Create an MPI profile for IPython::

    ipython profile create --parallel --profile=mpi

Edit the ``~/.ipython/profile_mpi/ipython_config.py`` file
and add the following line::

    c.IPClusterEngines.engine_launcher_class = 'MPIEngineSetLauncher'

Start the Jupyter notebook in your Bodo Conda environment::

    jupyter notebook

Go to `IPython Clusters` tab and select the
number of engines (i.e., cores) you'd like to use and click `Start` next to the
`mpi` profile. Alternatively, you can use ``ipcluster start -n 4 --profile=mpi``
in a terminal to start the engines. Initialization of the engines can take several seconds.

Now start a new notebook and run this code in a cell to setup the environment::

    import ipyparallel as ipp
    c = ipp.Client(profile='mpi')
    view = c[:]
    view.activate()

This should complete without any errors. An error may appear if the cluster
is not initialized yet (usually `NoEnginesRegistered`).
In this case, just wait a few seconds and try again.

You can now run Bodo functions on the execution engines
using `ipyparallel` hooks such as ``%%px`` magic
and the work will be distributed
across the engines. For example, run this code in a cell::

    %%px --block
    import bodo
    import numpy as np
    import time

    @bodo.jit
    def calc_pi(n):
        t1 = time.time()
        x = 2 * np.random.ranf(n) - 1
        y = 2 * np.random.ranf(n) - 1
        pi = 4 * np.sum(x**2 + y**2 < 1) / n
        print("Execution time:", time.time()-t1, "\nresult:", pi)

    calc_pi(2 * 10**8)


If you wish to run across multiple nodes, you can add the following to
`ipcluster_config.py`::

    c.MPILauncher.mpi_args = ["-machinefile", "path_to_file/machinefile"]

where `machinefile` (or `hostfile`) is a file with the hostnames of available nodes that MPI can use.
More information about `machinefiles` can be found
`here <https://www.open-mpi.org/faq/?category=running#mpirun-hostfile>`_.

It is important to note that other MPI systems and launchers (such as QSUB/PBS)
may use a different user interface for the allocation of computational nodes.
