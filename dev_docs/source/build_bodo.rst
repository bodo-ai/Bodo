.. _build_bodo:


Building Bodo from Source
-------------------------

On Mac/Linux
~~~~~~~~~~~~~~
We use `Anaconda <https://www.anaconda.com/download/>`_ distribution of
Python for setting up Bodo. These commands install Bodo and its dependencies
such as Numba on Ubuntu Linux::

    # Linux: wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    # Mac: wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b
    export PATH=$HOME/miniconda3/bin:$PATH
    conda create -n DEV python=3.7 numpy scipy pandas>=1.0.0 boost-cpp cmake h5py mpich mpi -c conda-forge
    source activate DEV
    # Linux: conda install gcc_linux-64 gxx_linux-64 gfortran_linux-64 -c conda-forge
    # Mac: conda install clang_osx-64 clangxx_osx-64 gfortran_osx-64 -c conda-forge
    conda install numba=0.48.0 -c conda-forge
    conda install -c bodo.ai -c conda-forge hdf5=*=*mpich*
    conda install -c conda-forge pyarrow=0.16.0
    # The following 2 commands are required for s3 related development and tests
    # conda install -c conda-forge botocore s3fs
    # conda install -c conda-forge boto3
    git clone https://github.com/Bodo-inc/Bodo.git
    cd Bodo
    # build Bodo
    HDF5_DIR=$CONDA_PREFIX python setup.py develop


On Windows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Install Visual Studio Community 2017 (15.9.18)
* From the Visual Studio installer, install following individual components::

    Windows 10 SDK (10.0.17763.0)
    Windows Universal CRT SDK
    VC++ 2015.3 v14.00 (v140) toolset for desktop

* Install `Miniconda for Windows <https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe>`_.
* Start 'Anaconda (Miniconda3) prompt'
* Setup the Conda environment in Anaconda Prompt::

    conda create -n DEV python=3.7 numpy scipy pandas>=1.0.0 boost-cpp cmake h5py -c conda-forge
    source activate DEV
    conda install numba=0.48.0 -c conda-forge
    conda install vc vs2015_runtime vs2015_win-64
    conda install -c defaults -c intel impi_rt impi-devel
    conda install -c conda-forge pyarrow=0.16.0
    git clone https://github.com/Bodo-inc/Bodo.git
    cd Bodo
    # build Bodo
    # For later HDF5 support: set HDF5_DIR=%CONDA_PREFIX%\Library
    python setup.py develop


Troubleshooting Windows Build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* HDF5 is currently not supported for windows version of Bodo.
* Testing for windows version is currently not available due to package conflicts.
* It might be necessary to remove all the different visual studio versions installed and fresh start above instruction.


Running Example/Test
~~~~~~~~~~~~~~~~~~~~~~~~~
A command line for running the Pi example on 4 cores::

    mpiexec -n 4 python examples/pi.py

If you run into gethostbyname failed error, try
`this fix <https://stackoverflow.com/questions/23112515/mpich2-gethostbyname-failed>`_.

Running unit tests::

    conda install pytest
    pytest -x -s -v -m "not slow" -W ignore

To run s3 related unit tests, in addition::
    
    export AWS_ACCESS_KEY_ID=bodotest1
    export AWS_SECRET_ACCESS_KEY=bodosecret1

The two environment variables will be read in `conftest.py <https://github.com/Bodo-inc/Bodo/blob/master/bodo/tests/conftest.py>`_
and set for `minio <https://min.io/?gclid=Cj0KCQiAsvTxBRDkARIsAH4W_j9rNeSft9zVArxg1Zo4RAfXS31dC9Aq-amIigRAT_yAPQbKdU0RvD4aAv0UEALw_wcB>`_.

In case of issues, reinstalling in a new conda environment is recommended.


Other useful packages for development::

    conda install pytest sphinx pylint jupyter
    .. conda install daal4py -c defaults -c intel







