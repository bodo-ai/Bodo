.. _install:


Installation
============

Bodo is a Python package and can be installed in a Conda environment easily.
Install Conda if not installed already. For example:

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

Bodo is distributed using a private Conda channel. Install Bodo and its
dependencies as shown below (replace "<username>" and "<token>" with the username
and token for Bodo's Conda channel provided by a Bodo associate)::

    conda create -n Bodo python
    source activate Bodo
    export BODO_CONDA_USERNAME=<username>
    export BODO_CONDA_TOKEN=<token>
    conda install bodo -c https://"$BODO_CONDA_USERNAME":"$BODO_CONDA_TOKEN"@bodo.jfrog.io/artifactory/api/conda/bodo.ai -c conda-forge

Bodo uses `MPI <https://en.wikipedia.org/wiki/Message_Passing_Interface>`_ for parallelization,
which is automatically installed as part of
the conda command above. MPI can be configured on clusters easily.
The cluster nodes need to have passwordless SSH enabled between them,
and there should be a host file listing their addresses
(example tutorial `here <https://mpitutorial.com/tutorials/running-an-mpi-cluster-within-a-lan/>`_).


License key
-----------

Bodo requires a license key to run. The key can be provided in two ways:

- Through the environment variable ``BODO_LICENSE``

- A file called ``bodo.lic`` in the current working directory

In both cases, the file or environment variable must contain the key exactly
as provided.

If Bodo cannot find the license (environment variable does not exist or is empty,
and no license file is found), it will exit with "Bodo license not found" error.

If the key content is invalid Bodo will exit with "Invalid license"
error. This typically means that the key is missing data or contains extraneous
characters. Please make sure the license file has not been modified, or that
the environment variable contains the key verbatim. Note that some shells might
append extra characters when displaying the file contents. A valid way to export
the key is this: ``export BODO_LICENSE=`cat bodo.lic```


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
