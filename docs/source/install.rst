.. _install:


Installation
============

Bodo is a Python package and can be installed in a Conda environment easily.
Install Conda if not installed already. For example::

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b
    export PATH=$HOME/miniconda3/bin:$PATH

Unpack the Bodo installation package and install Bodo and its
dependencies (replace "{full-path-to-bodo-package-directory}")::

    conda create -n Bodo python
    source activate Bodo
    conda install bodo h5py scipy hdf5=*=*mpich* -c file://{full-path-to-bodo-package-directory} -c conda-forge

Bodo uses `MPI <https://en.wikipedia.org/wiki/Message_Passing_Interface>`_ for parallelization,
which is automatically installed as part of
the conda command above. MPI can be configured on clusters easily.
The cluster nodes need to have passwordless SSH enabled between them,
and there should be a host file listing their addresses
(example tutorial `here <https://mpitutorial.com/tutorials/running-an-mpi-cluster-within-a-lan/>`_).
