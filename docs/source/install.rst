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

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b
    export PATH=$HOME/miniconda3/bin:$PATH

Unpack the Bodo installation package and install Bodo and its
dependencies (replace "{full-path-to-bodo-package-directory}")::

    conda create -n Bodo python
    source activate Bodo
    conda install bodo h5py scipy "hdf5=*=*mpich*" -c file://{full-path-to-bodo-package-directory} -c conda-forge

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
