.. _ipyparallelsetup:

Interactive Bodo Cluster Setup using Ipyparallel
=================================================

Bodo can be used with ``ipyparallel`` to allow interactive code execution on a
local or remote cluster.


- :ref:`quickstart_local`
- :ref:`quickstart_multiple_hosts`
- :ref:`run_bodo_ipyparallel`
- :ref:`run_from_python_script`


.. _quickstart_local:

Getting started on your machine
-------------------------------

Install IPyParallel, JupyterLab and Bodo in your conda environment::

    conda install bodo ipyparallel=7 jupyterlab=3 -c bodo.ai -c conda-forge

Start a JupyterLab server::

    jupyter lab

Start a new notebook and run the following code in a cell to start an IPyParallel cluster::

    import ipyparallel as ipp
    c = ipp.Cluster(profile="mpi", engine_launcher_class='MPI', n=4)
    c.start_cluster_sync()
    rc = c.connect_client_sync()
    rc.wait_for_engines(n=c.n)
    view = rc[:]
    view.activate()
    view.block = True
    

This starts a local 4-core MPI cluster on your machine. You can now start using 
the ``%%px`` cell magic to parallelize your code execution, or use ``%autopx`` to
run all cells on the IPyParallel cluster by default.
Read more `here <https://ipyparallel.readthedocs.io/en/latest/tutorial/magics.html#parallel-magic-commands>`_.

.. _setupverify_local:

Verifying your setup
~~~~~~~~~~~~~~~~~~~~

Run the following code to verify that your IPyParallel cluster is set up correctly::

    %%px
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    print(f"Hello World from rank {comm.Get_rank()}. total ranks={comm.Get_size()}")

The correct output is::

    Hello World from rank 0. total ranks=4
    Hello World from rank 1. total ranks=4
    Hello World from rank 2. total ranks=4
    Hello World from rank 3. total ranks=4

.. _quickstart_multiple_hosts:

Running on multiple hosts
-------------------------

To start an IPyParallel cluster across multiple hosts, you need to do the following:

- Install IPyParallel and Bodo on all hosts::

    conda install bodo ipyparallel=7 -c bodo.ai -c conda-forge

- Install JupyterLab on one of the hosts. Let's call it the controller node::

    conda install jupyterlab=3 -c bodo.ai -c conda-forge

- Set up passwordless SSH between each of these hosts (this is needed for ``mpiexec``).

- The controller node must be able to connect to all engines via TCP on any port.
  If you have a restricted network, please refer to the ``ipyparallel``
  `documentation <https://ipyparallel.readthedocs.io/en/latest/tutorial/process.html>`_
  for other options such as SSH tunneling.

- Create a machinefile that contains list of IP addresses or host names where you want to launch engines.
  
  .. note::
    Make sure your machinefile is in the following format::

        ip_1
        ip_2
        ...

  You can find more information about `machinefiles` `here <https://www.open-mpi.org/faq/?category=running#mpirun-hostfile>`_.
  It is important to note that other MPI systems and launchers (such as QSUB/PBS)
  may use a different user interface for the allocation of computational nodes.

Start a JupyterLab server on the controller node::

    jupyter lab

Starting an IPyParallel cluster across multiple hosts requires a couple of additional steps. Start a new notebook and run the following code in a cell::

    import ipyparallel as ipp
    c = ipp.Cluster(profile="mpi",
                    engine_launcher_class='MPI',
                    n=4,  # Number of engines, you can change this
                    controller_ip='*',
                    controller_args=["--nodb"])
    c.engine_launcher_class.mpi_args = ["-machinefile", <PATH_TO_MACHINEFILE>]
    c.start_controller_sync()

This will start the IPyParallel controller on the controller node.
Next, the connection info for this cluster needs to be copied to all the hosts. You can
run the following code to do this::

    # Get connection info
    connection_info = await c.controller.get_connection_info()
    engine_info = connection_info['engine']

    import os
    import sys
    import json
    from subprocess import run, STDOUT

    def send_connection_info(connection_info, connection_file):
        env = os.environ.copy()
        env["CONNECTION_INFO"] = json.dumps(connection_info)
        cmd =     [
            'mpiexec',
            '-ppn',
            '1',
            '-machinefile', 
            <PATH_TO_MACHINEFILE>,
            'sh',
            '-c',
            f'echo $CONNECTION_INFO > "{connection_file}"'
            
        ]
        p = run(cmd, capture_output=True, text=True, input=None, env=env)
        if p.returncode:
            print(p.stderr, file=sys.stderr)
            p.check_returncode()
        return p

    send_connection_info(
        engine_info,
        os.path.join(
            c.profile_dir,
            'security',
            f'ipcontroller-{c.cluster_id}-engine.json',
        ),
    )

.. note::

    You can skip the step above if your IPython profile directory is on a shared file-system.

You can now start your engines by running the following code::

    c.start_engines_sync()
    rc = c.connect_client_sync()
    rc.wait_for_engines(n=c.n)
    view = rc[:]
    view.activate()
    view.block = True

You have now successfully started an IPyParallel cluster across multiple hosts.

.. _setupverify_multiple_hosts:

Verifying your setup
~~~~~~~~~~~~~~~~~~~~

Run the following code to verify that your IPyParallel cluster is set up correctly::

    %%px
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    print(f"Hello World from rank {comm.Get_rank()} on host {MPI.Get_processor_name()}. total ranks={comm.Get_size()}")

On a cluster with two hosts running 4 engines, the correct output is::

    Hello World from rank 0 on host A. total ranks=4
    Hello World from rank 1 on host A. total ranks=4
    Hello World from rank 2 on host B. total ranks=4
    Hello World from rank 3 on host B. total ranks=4

.. _run_bodo_ipyparallel:

Running Bodo on your IPyParallel Cluster
----------------------------------------

You are now ready to run your Bodo code. Here's an example of Monte Carlo Pi calculation with Bodo::

    %%px
    import numpy as np
    import time

    @bodo.jit
    def calc_pi(n):
        t1 = time.time()
        x = 2 * np.random.ranf(n) - 1
        y = 2 * np.random.ranf(n) - 1
        pi = 4 * np.sum(x ** 2 + y ** 2 < 1) / n
        print("Execution time:", time.time() - t1, "\nresult:", pi)
        return pi
    
    calc_pi(10000000)


.. _run_from_python_script:

Running from a python script
----------------------------

You can run code on an IPyParallel cluster from a python script instead of IPython or JupyterLab as follows:

- Setup the cluster using the same steps as above.

- Define the function you want to run on the cluster:

    .. code-block:: python

        import inspect
        import bodo

        @bodo.jit
        def calc_pi(n):
            t1 = time.time()
            x = 2 * np.random.ranf(n) - 1
            y = 2 * np.random.ranf(n) - 1
            pi = 4 * np.sum(x ** 2 + y ** 2 < 1) / n
            print("Execution time:", time.time() - t1, "\nresult:", pi)
            return pi


- We define a Python wrapper for ``calc_pi`` called ``bodo_exec`` which will be sent to the engines to compute. This wrapper will call the Bodo function on the engines, collect the result and send it back to the client.

    .. code-block:: python


        def bodo_exec(points):
            return calc_pi(points)

- We can send the source code to be executed at the engines, using the ``execute`` method of ipyparallel's ``DirectView`` object.
  After the imports and code definitions are sent to the engines, the computation is started by actually calling the ``calc_pi`` function (now defined on the engines) and returning the result to the client.


     .. code-block:: python

        def main():

            # remote code execution: import required modules on engines
            view.execute("import numpy as np")
            view.execute("import bodo")
            view.execute("import time")

            # send code of Bodo functions to engines
            bodo_funcs = [calc_pi]
            for f in bodo_funcs:
                # get source code of Bodo function
                f_src = inspect.getsource(f)
                # execute the source code thus defining the function on engines
                view.execute(f_src).get()

            points = 200000000
            ar = view.apply(bodo_exec, points)
            result = ar.get()
            print("Result is", result)

            client.close()

        main()
