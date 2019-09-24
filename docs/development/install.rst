.. _install:


Building Bodo from Source
-------------------------

We use `Anaconda <https://www.anaconda.com/download/>`_ distribution of
Python for setting up Bodo. These commands install Bodo and its dependencies
such as Numba on Ubuntu Linux::

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b
    export PATH=$HOME/miniconda3/bin:$PATH
    conda create -n DEV numpy scipy pandas boost cmake h5py pyarrow mpich mpi
    source activate DEV
    # Linux: conda install gcc_linux-64 gxx_linux-64 gfortran_linux-64
    # Mac: conda install clang_osx-64 clangxx_osx-64 gfortran_osx-64
    conda install -c numba/label/dev llvmlite
    git clone https://github.com/numba/numba.git
    cd numba
    python setup.py develop
    cd ..
    git clone https://github.com/Bodo-inc/Bodo.git
    cd Bodo
    # build Bodo
    HDF5_DIR=$CONDA_PREFIX python setup.py develop


A command line for running the Pi example on 4 cores::

    mpiexec -n 4 python examples/pi.py
If you run into gethostbyname failed error, try `this <https://stackoverflow.com/questions/23112515/mpich2-gethostbyname-failed>`_
    
Running unit tests::

    conda install pytest
    pytest -x -s -v -W ignore

In case of issues, reinstalling in a new conda environment is recommended.


Other useful packages for development::
    conda install pytest sphinx pylint jupyter
    conda install daal4py -c defaults -c intel

Test Suite
----------


We use `pytest` for testing and run the test suite on different
number of processors::

    pytest -s -v -W ignore
    mpiexec -n 2 pytest -s -v -W ignore
    mpiexec -n 3 pytest -s -v -W ignore


Building Documentation
----------------------

The documentation is under the `docs` directory of the repository and uses
the reStructuredText format.
It is built with `Sphinx <http://www.sphinx-doc.org>`_ and the bootstrap theme::

    conda install sphinx
    pip install sphinx_bootstrap_theme

After updating documentation, run :code:`make html` in the `docs` folder to build.


Building from Source on Windows
-------------------------------

Building Bodo on Windows requires Build Tools for Visual Studio 2017 (14.0):

* Install `Build Tools for Visual Studio 2017 (14.0) <https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017>`_.
* Install `Miniconda for Windows <https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe>`_.
* Start 'Anaconda prompt'
* Setup the Conda environment in Anaconda Prompt::

    conda create -n Bodo -c ehsantn -c numba/label/dev -c anaconda -c conda-forge python=3.7 pandas pyarrow h5py numba scipy boost libboost tbb-devel mkl-devel
    activate Bodo
    conda install vc vs2015_runtime vs2015_win-64
    conda install -c intel impi_rt impi-devel
    git clone https://github.com/IntelLabs/bodo.git
    cd bodo
    set HDF5_DIR=%CONDA_PREFIX%\Library
    python setup.py develop

.. "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64

Troubleshooting Windows Build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* If the ``cl`` compiler throws the error fatal ``error LNK1158: cannot run ‘rc.exe’``,
  add Windows Kits to your PATH (e.g. ``C:\Program Files (x86)\Windows Kits\8.0\bin\x86``).
* Some errors can be mitigated by ``set DISTUTILS_USE_SDK=1``.
* For setting up Visual Studio, one might need go to registry at
  ``HKEY_LOCAL_MACHINE\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\SxS\VS7``,
  and add a string value named ``14.0`` whose data is ``C:\Program Files (x86)\Microsoft Visual Studio 14.0\``.
