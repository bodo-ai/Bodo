# Interactive Bodo Cluster Setup using IPyParallel {#ipyparallelsetup}

Bodo can be used with IPyParallel to allow interactive code execution
on a local or remote cluster.

## Getting started on your machine {#quickstart_local}

Install IPyParallel, JupyterLab, and Bodo in your conda environment:

```shell
conda install bodo ipyparallel=8.1 jupyterlab=3 -c bodo.ai -c conda-forge
```
Start a JupyterLab server:

```shell
jupyter lab
```
Start a new notebook and run the following code in a cell to start an
IPyParallel cluster:


```py
import ipyparallel as ipp
import psutil; 

n = min(psutil.cpu_count(logical=False), 8)
rc = ipp.Cluster(engines='mpi', n=n).start_and_connect_sync(activate=True)
```

This starts a local N-core MPI cluster on your machine, where N is the
minimum of the number of cores on your machine and 8. You can now start
using the `%%px` cell magic to parallelize your code execution, or use
`%autopx` to run all cells on the IPyParallel cluster by default. Read
more
[here](https://ipyparallel.readthedocs.io/en/latest/tutorial/magics.html#parallel-magic-commands).

### Verifying your setup {#setupverify_local}

Run the following code to verify that your IPyParallel cluster is set up
correctly:

```py
%%px
import bodo
print(f"Hello World from rank {bodo.get_rank()}. Total ranks={bodo.get_size()}")
```

The correct output is:

```console
Hello World from rank 0. Total ranks=N
Hello World from rank 1. Total ranks=N
...
Hello World from rank N-1. Total ranks=N
```

Where N is the minimum of the number of cores on your machine and 8.

## Running on multiple hosts {#quickstart_multiple_hosts}

To start an IPyParallel cluster across multiple hosts:

-   Install IPyParallel and Bodo on all hosts:
    
    ```console
    conda install bodo ipyparallel=8.1 -c bodo.ai -c conda-forge
    ```
    
-   Install JupyterLab on one of the hosts. Let's call it the
    controller node:

    ```console
    conda install jupyterlab=3 -c bodo.ai -c conda-forge
    ```
    
-   Set up passwordless SSH between each of these hosts (this is needed
    for `mpiexec`). See the section on [passwordless ssh][passwordless_ssh] for instructions.

-   The controller node must be able to connect to all engines via TCP
    on any port. If you have a restricted network, please refer to the
    IPyParallel
    [documentation](https://ipyparallel.readthedocs.io/en/latest/tutorial/process.html)
    for other options such as SSH tunneling.

-   Create a hostfile that contains list of IP addresses or host names
    where you want to launch engines.

    !!! note
    
        Make sure your hostfile is in the following format:   
        ```
        ip_1 ip_2 ...
        ```

    You can find more information about `hostfiles`
    [here](https://www.open-mpi.org/faq/?category=running#mpirun-hostfile){target="blank"}.
    It is important to note that other MPI systems and launchers (such
    as QSUB/PBS) may use a different user interface for the allocation
    of computational nodes.

-   Create the default IPython profile on all nodes by executing the
    following from the controller node:
    ```console
    mpiexec -ppn 1 -f <PATH_TO_HOSTFILE> ipython profile create
    ```
    
Now you can start a JupyterLab server on the controller node:

```console
jupyter lab
```

Starting an IPyParallel cluster across multiple hosts requires setting a
couple of additional configuration options. Start a new notebook and run
the following code in a cell:

```py

import ipyparallel as ipp
c = ipp.Cluster(engines='mpi',
                n=8,  # Number of engines: Set this to the total number of physical cores in your cluster
                controller_ip='*',
                controller_args=["--nodb"])
c.engine_launcher_class.mpi_args = ["-f", <PATH_TO_HOSTFILE>]
rc = c.start_and_connect_sync()
view = rc.broadcast_view(block=True)
view.activate()
```

You have now successfully started an IPyParallel cluster across multiple
hosts.

### Verifying your setup {#setupverify_multiple_hosts}

Run the following code to verify that your IPyParallel cluster is set up
correctly:

```py
%%px
import bodo
import socket
print(f"Hello World from rank {bodo.get_rank()} on host {socket.gethostname()}. Total ranks={bodo.get_size()}")
```

On a cluster with two hosts running 4 engines, the correct output is:

```pydocstring
Hello World from rank 0 on host A. Total ranks=4
Hello World from rank 1 on host A. Total ranks=4
Hello World from rank 2 on host B. Total ranks=4
Hello World from rank 3 on host B. Total ranks=4
```

## Running Bodo on your IPyParallel Cluster {#run_bodo_ipyparallel}

You are now ready to run your Bodo code. Here is an example function
with Bodo:

```py
%%px
import bodo

@bodo.jit
def process_data(n):
    df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n)**2})
    df["C"] = df.apply(lambda r: 2* r.A + r.B if r.A > 10 else 0, axis=1)
    return df["C"].sum()

process_data(100000000)
```

## Running from a Python Script {#run_from_python_script}

You can run code on an IPyParallel cluster from a python script instead
of IPython or JupyterLab as follows:

-   Setup the cluster using the same steps as above.

-   Define the function you want to run on the cluster:

    ``` python
    import inspect
    import bodo
    
    @bodo.jit
    def process_data(n):
        df = pd.DataFrame({"A": np.arange(n), "B": np.arange(n)**2})
        df["C"] = df.apply(lambda r: 2* r.A + r.B if r.A > 10 else 0, axis=1)
        return df["C"].sum()
    
    process_data(100000000)
    ```

-   We define a Python wrapper for `process_data` called `bodo_exec`
    which will be sent to the engines to compute. This wrapper will call
    the Bodo function on the engines, collect the result and send it
    back to the client.

    ``` python
    def bodo_exec(points):
        return process_data(points)
    ```

-   We can send the source code to be executed at the engines, using the
    `execute` method of ipyparallel's `DirectView` object. After the
    imports and code definitions are sent to the engines, the
    computation is started by actually calling the `process_data`
    function (now defined on the engines) and returning the result to
    the client.

    ``` python
    def main():
    
        # remote code execution: import required modules on engines
        view.execute("import pandas as pd")
        view.execute("import numpy as np")
        view.execute("import bodo")
        view.execute("import time")
    
        # send code of Bodo functions to engines
        bodo_funcs = [process_data]
        for f in bodo_funcs:
            # get source code of Bodo function
            f_src = inspect.getsource(f)
            # execute the source code thus defining the function on engines
            view.execute(f_src).get()
    
        points = 200000000
        ar = view.apply(bodo_exec, points)
        result = ar.get()
        print("Result is", result)
    
        rc.close()
    
    main()
    ```

## Useful IPyParallel References

-   [IPyParallel Documentation](https://ipyparallel.readthedocs.io/en/latest/)
-   [Using MPI with IPython](https://ipyparallel.readthedocs.io/en/latest/reference/mpi.html)
-   [IPython Parallel in 2021](https://blog.jupyter.org/ipython-parallel-in-2021-2945985c032a)


[comment]: <> (Autoref to [passwordless_ssh] will be populated it is added)
[todo]: <> (Modify/remove comment above as [passwordless_ssh] is added)