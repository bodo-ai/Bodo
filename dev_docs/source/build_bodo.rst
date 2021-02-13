.. _build_bodo_source:


Building Bodo from Source
-------------------------

On Mac/Linux
~~~~~~~~~~~~
We use `Anaconda <https://www.anaconda.com/download/>`_ distribution of
Python for setting up Bodo. These commands install Bodo and its dependencies
such as Numba on Ubuntu Linux::

    # Linux: wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    # Mac: wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b
    export PATH=$HOME/miniconda3/bin:$PATH
    conda create -n DEV python=3.8 numpy"<1.20" scipy pandas='1.2.*' boost-cpp=1.74.0 cmake h5py mpich mpi -c conda-forge
    source activate DEV
    # Linux: conda install gcc_linux-64 gxx_linux-64 -c conda-forge
    # Mac: conda install clang_osx-64 clangxx_osx-64 -c conda-forge
    # If you don't want to install Numba from source: conda install numba=0.52.0 -c conda-forge
    # For development/debugging purposes, it's best to install Numba from source instead
    # START OF BUILD NUMBA FROM SOURCE
    git clone https://github.com/numba/numba.git
    # make sure you checkout version 0.52.0. numba's master may not work with Bodo master
    # if you need to remove any numba in your miniconda env before rebuilding from source try:
    # conda remove numba=<version>
    cd numba; git checkout 0.52.0
    python setup.py develop
    cd ..
    # END OF BUILD NUMBA FROM SOURCE
    conda install mpi4py pytest cython -c conda-forge
    conda install -c conda-forge hdf5='*=*mpich*' pyarrow=2.0.0 pymysql sqlalchemy
    # The following is required for s3 related development and tests
    # conda install -c conda-forge boto3 botocore "s3fs=0.4.2"
    git clone https://github.com/Bodo-inc/Bodo.git
    cd Bodo
    # build Bodo
    python setup.py develop

For HDFS related development, use the :ref:`docker image <docker-images>`.

Troubleshooting MacOS Build
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* If your Mac0S sdk hasn't been configured properly, you may get a clang related error like the following::

     ld: unsupported tapi file type '!tapi-tbd' in YAML file

* Add the following line to your .zshrc file::

    export CONDA_BUILD_SYSROOT=/opt/MacOSX10.9.sdk

* Execute buildscripts/setup_osx_sdk.sh to resolve this issue.


On Windows
~~~~~~~~~~

* Install Visual Studio Community 2017 (15.9.18)
* From the Visual Studio installer, install following individual components::

    Windows 10 SDK (10.0.17763.0)
    Windows Universal CRT SDK
    VC++ 2015.3 v14.00 (v140) toolset for desktop

* Install `Miniconda for Windows <https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe>`_.
* Start 'Anaconda (Miniconda3) prompt'
* Setup the Conda environment in Anaconda Prompt::

    conda create -n DEV python=3.8 numpy scipy pandas="1.2.*" boost-cpp=1.74.0 cmake h5py -c conda-forge
    source activate DEV
    conda install numba=0.52.0 -c conda-forge
    conda install vc vs2015_runtime vs2015_win-64
    conda install -c defaults -c intel impi_rt impi-devel
    conda install -c conda-forge pyarrow=2.0.0
    git clone https://github.com/Bodo-inc/Bodo.git
    cd Bodo
    # build Bodo
    python setup.py develop


Troubleshooting Windows Build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* HDF5 is currently not supported for windows version of Bodo.
* Testing for windows version is currently not available due to package conflicts.
* It might be necessary to remove all the different visual studio versions installed and fresh start above instruction.


Running Example/Test
~~~~~~~~~~~~~~~~~~~~
A command line for running the Pi example on 4 cores::

    mpiexec -n 4 python examples/pi.py

If you run into gethostbyname failed error, try
`this fix <https://stackoverflow.com/questions/23112515/mpich2-gethostbyname-failed>`_.

Running unit tests::

    conda install pytest
    pytest -x -s -v -m "not slow"

To run s3 related unit tests, in addition::

    export AWS_ACCESS_KEY_ID=bodotest1
    export AWS_SECRET_ACCESS_KEY=bodosecret1

The two environment variables will be read in `conftest.py <https://github.com/Bodo-inc/Bodo/blob/master/bodo/tests/conftest.py>`_
and set for `minio <https://min.io/?gclid=Cj0KCQiAsvTxBRDkARIsAH4W_j9rNeSft9zVArxg1Zo4RAfXS31dC9Aq-amIigRAT_yAPQbKdU0RvD4aAv0UEALw_wcB>`_.

In case of issues, reinstalling in a new conda environment is recommended.

To run HDFS related unit tests, use the :ref:`docker image <docker-images>`.

Other useful packages for development::

    conda install pytest sphinx pylint jupyter xlrd xlsxwriter openpyxl mpi4py ipyparallel matplotlib jupyterlab aws-sdk-cpp


Cleaning Bodo
~~~~~~~~~~~~~

The script `clean.sh` is provided to clean up all the leftover files after compilation.
It removes all C++ compiled code and the `__pycache__` directories.
