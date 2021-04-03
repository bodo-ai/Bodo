.. _install:


Installation
============

Bodo is a Python package and can be installed in a Conda environment easily. See the section on :ref:`conda` if you don't already have Conda installed. Create a Conda environment, install Bodo and its
dependencies as shown below::

    conda create -n Bodo python
    source activate Bodo
    conda install bodo -c bodo.ai -c conda-forge

Bodo uses `MPI <https://en.wikipedia.org/wiki/Message_Passing_Interface>`_ for parallelization,
which is automatically installed as part of
the conda install command above. Note that this command installs Bodo Community Edition by default, which is free and works for up to 4 cores.
See the section below on using the :ref:`enterprise`. For more information on Bodo Enterprise Edition and pricing, please `contact us <https://bodo.ai/contact/>`_ .


Testing your Installation
--------------------------

Once you have activated your Conda environment and installed Bodo in it, you can test it using the example program below.
This program has two functions:

- The function ``gen_data`` creates a sample dataset with 20,000 rows, and writes to a parquet file called ``example1.pq``.
- The function ``test`` reads ``example1.pq`` and performs multiple computations on it.

.. code-block:: python3

    import bodo
    import pandas as pd
    import numpy as np
    import time

    @bodo.jit
    def gen_data():
        NUM_GROUPS = 30
        NUM_ROWS = 20_000_000
        df = pd.DataFrame({
            "A": np.arange(NUM_ROWS) % NUM_GROUPS,
            "B": np.arange(NUM_ROWS)
        })
        df.to_parquet("example1.pq")

    @bodo.jit
    def test():
        df = pd.read_parquet("example1.pq")
        t0 = time.time()
        df2 = df.groupby("A")["B"].agg(
            (lambda a: (a==1).sum(), lambda a: (a==2).sum(), lambda a: (a==3).sum())
        )
        m = df2.mean()
        print("Result:", m, "\nCompute time:", time.time() - t0, "secs")

    gen_data()
    test()

Save this code in a file called ``example.py``, and run it on a single core as follows::

    python example.py

To run the code on 4 cores, you can use ``mpiexec``::

    $ mpiexec -n 4 python example.py

You can benchmark this code against native Pandas by simply removing the ``@bodo.jit`` decorators before the functions.
To learn more about benchmarking with Bodo, see the section on :ref:`performance`.

.. _conda:

Installing Conda
----------------
Install Conda using the instructions below.

On Linux::

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b
    export PATH=$HOME/miniconda3/bin:$PATH

On macOS::

    curl https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -L -o miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b
    export PATH=$HOME/miniconda3/bin:$PATH


Optional Dependencies
---------------------

Some Bodo functionality may require other dependencies as the table
below summarizes. All optional dependencies except Hadoop can be
installed using the commands ``conda install gcsfs sqlalchemy
hdf5='*=*mpich*' openjdk -c conda-forge`` and ``pip install
deltalake``.

.. list-table::
   :header-rows: 1

   * - Functionality
     - Dependency
   * - ``pd.read_sql / df.to_sql``
     - ``sqlalchemy``
   * - ``HDF5``
     - ``hdf5 (MPI version)``
   * - ``GCS I/O``
     - ``gcsfs``
   * - ``Delta Lake``
     - ``deltalake``
   * - ``HDFS or ADLS Gen2``
     - `hadoop <http://hadoop.apache.org/docs/stable/>`_ (only the Hadoop client is needed)


.. _enterprise:

Bodo Enterprise Edition
-----------------------

Bodo Enterprise Edition allows unrestricted use of Bodo on any number of cores.

License key
~~~~~~~~~~~

Bodo Enterprise Edition requires a license key to run. The key can be provided in two ways:

- Through the environment variable ``BODO_LICENSE``

- A file called ``bodo.lic`` in the current working directory

In both cases, the file or environment variable must contain the key exactly
as provided.

If Bodo cannot find the license (environment variable does not exist or is empty,
and no license file is found), you will only be able to run Bodo on up to 4 cores.
If you try to run Bodo on more than 4 cores and if Bodo cannot find the license (environment variable does not exist or is empty, and no license file is found), it will exit with “Bodo license not found” error.

If the key content is invalid Bodo will exit with "Invalid license"
error. This typically means that the key is missing data or contains extraneous
characters. Please make sure the license file has not been modified, or that
the environment variable contains the key verbatim. Note that some shells might
append extra characters when displaying the file contents. A valid way to export
the key is this::

    export BODO_LICENSE=`cat bodo.lic


Automated ``BODO_LICENSE`` environment variable Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can automate setting of the ``BODO_LICENSE`` environment variable in your ``~/.bashrc`` script (or the ``~/.zshrc`` script for macOS) using::

    echo 'export BODO_LICENSE="<COPY_PASTE_THE_LICENSE_HERE>"' >> ~/.bashrc


For more fine grained control and usage with the ``Bodo`` conda environment as created above, we recommend the following steps to automate setting the ``BODO_LICENSE`` environment variable (closely follows `these <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#macos-and-linux>`_ steps):

1. Ensure that you're in the correct conda environment.

2. Navigate to the ``$CONDA_PREFIX`` directory and create some additional conda environment activation and deactivation steps::

        cd $CONDA_PREFIX
        mkdir -p ./etc/conda/activate.d
        mkdir -p ./etc/conda/deactivate.d
        touch ./etc/conda/activate.d/env_vars.sh
        touch ./etc/conda/deactivate.d/env_vars.sh

3. Edit ``./etc/conda/activate.d/env_vars.sh`` as follows::

        #!/bin/sh

        export BODO_LICENSE="<COPY_PASTE_THE_LICENSE_HERE>"

4. Similarly, edit ``./etc/conda/deactivate.d/env_vars.sh`` as follows::

        #!/bin/sh

        unset BODO_LICENSE

5. Deactivate (``conda deactivate``) and reactivate the ``Bodo`` conda environment (``conda activate Bodo``) to ensure that the environment variable ``BODO_LICENSE`` is automatically added when the environment is activated.


Using MPI in clusters with Bodo Enterprise Edition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MPI can be configured on clusters easily.
The cluster nodes need to have passwordless SSH enabled between them,
and there should be a host file listing their addresses
(example tutorial `here <https://mpitutorial.com/tutorials/running-an-mpi-cluster-within-a-lan/>`_).For best performance, MPI usually needs to be configured to launch one process per physical core.
This avoids potential resource contention between processes (due to high efficiency of MPI).
For example, a cluster of four nodes, each with 16 physical cores, would use 64 MPI processes::

    $ mpiexec -n 64 python example.py

For cloud instances, one physical core usually corresponds to two vCPUs.
For example, an instance with 32 vCPUs has 16 physical cores.
