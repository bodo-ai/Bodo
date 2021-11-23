.. _enterprise:

Configuring Bodo Enterprise Edition
=====================================

Bodo Enterprise Edition allows unrestricted use of Bodo on any number of cores. Ensure you have finished :ref:`install` before
configuring Bodo Enterprise Edition.

- :ref:`licensekey`
- :ref:`mpienterpriseclusters`

.. _licensekey:

License Key
------------

Bodo Enterprise Edition requires a license key to run. The key can be provided in two ways:

- Through the environment variable ``BODO_LICENSE``

- A file called ``bodo.lic`` in the current working directory

In both cases, the file or environment variable must contain the key exactly
as provided.

If Bodo cannot find the license (environment variable does not exist or is empty,
and no license file is found), you will only be able to run Bodo on up to 8 cores.
If you try to run Bodo on more than 8 cores and if Bodo cannot find the license (the environment variable does not exist or is empty, and no license file is found), it will exit with the ``Bodo license not found`` error.

If the key content is invalid Bodo will exit with the ``Invalid license``
error. This typically means that the key is missing data or contains extraneous
characters. Please make sure the license file has not been modified, or that
the environment variable contains the key verbatim. Note that some shells might
append extra characters when displaying the file contents. A valid way to export
the key is this::

    export BODO_LICENSE=`cat bodo.lic


Automated ``BODO_LICENSE`` environment variable Setup
------------------------------------------------------

You can automate setting of the ``BODO_LICENSE`` environment variable in your ``~/.bashrc`` script (or the ``~/.zshrc`` script for macOS) using::

    echo 'export BODO_LICENSE="<COPY_PASTE_THE_LICENSE_HERE>"' >> ~/.bashrc


For more fine grained control and usage with the Bodo ``conda`` environment as created during :ref:`install`, we recommend the following steps to automate setting the ``BODO_LICENSE`` environment variable (closely follows `these <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#macos-and-linux>`_ steps):

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


.. _mpienterpriseclusters:

Using MPI in Clusters with Bodo Enterprise Edition
---------------------------------------------------

MPI can be configured on clusters easily.
The cluster nodes need to have passwordless SSH enabled between them,
and there should be a host file listing their addresses
(example tutorial `here <https://mpitutorial.com/tutorials/running-an-mpi-cluster-within-a-lan/>`_).For best performance, MPI usually needs to be configured to launch one process per physical core.
This avoids potential resource contention between processes (due to high efficiency of MPI).
For example, a cluster of four nodes, each with 16 physical cores, would use 64 MPI processes::

    $ mpiexec -n 64 python example.py

For cloud instances, one physical core usually corresponds to two vCPUs.
For example, an instance with 32 vCPUs has 16 physical cores.

.. seealso:: :ref:`ipyparallelsetup`
