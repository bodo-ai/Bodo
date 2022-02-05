.. _build_bodo_source:


Building Bodo from Source
-------------------------

On Mac/Linux
~~~~~~~~~~~~
We use `Anaconda <https://www.anaconda.com/download/>`_ distribution of
Python for setting up Bodo. These commands install Bodo and its dependencies
such as Numba on Ubuntu Linux::

    # Linux: wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    # Intel Mac: wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
    # M1 Mac: wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-arm64.sh -O miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b
    export PATH=$HOME/miniconda3/bin:$PATH
    conda create -n DEV 'python<=3.9' 'numpy=1.20' scipy pandas='1.3.*' boost-cpp cmake h5py mpich mpi -c conda-forge
    source activate DEV
    conda install mpi4py pytest cython -c conda-forge
    # Linux: conda install 'gcc_linux-64>=9' 'gxx_linux-64>=9' -c conda-forge
    # Intel Mac: conda install clang_osx-64 clangxx_osx-64 -c conda-forge
    # M1 Mac: conda install clang_osx-arm64 clangxx_osx-arm64 -c conda-forge
    # If you don't want to install Numba from source: conda install numba=0.55.0 -c conda-forge
    # For development/debugging purposes, it's best to install Numba from source instead
    # START OF BUILD NUMBA FROM SOURCE
    conda install llvmlite -c conda-forge
    git clone https://github.com/numba/numba.git
    # make sure you checkout version 0.55.0. numba's master may not work with Bodo master
    # if you need to remove any numba in your miniconda env before rebuilding from source try:
    # conda remove numba=<version>
    cd numba; git checkout 0.55.0
    python setup.py develop
    cd ..
    # END OF BUILD NUMBA FROM SOURCE
    conda install -c conda-forge hdf5==1.10.*='*mpich*' pyarrow=5.0.0 pymysql sqlalchemy
    # Highly recommended to speed up compilation times
    conda install ccache -c conda-forge
    # Highly recommended for enforcing style requirements (see dev_process.rst)
    conda install pre-commit -c conda-forge
    # Required for IO with S3
    conda install -c conda-forge fsspec
    # The following is required for s3 related development and tests
    # conda install -c conda-forge boto3 botocore
    # The following is required for IO with gcs
    # conda install -c conda-forge gcsfs
    # The following is required for matplotlib Testing
    # conda install -c conda-forge 'matplotlib<=3.4.3'
    # for PySpark testing:
    # conda install -c conda-forge pyspark openjdk
    # Required for ML tests
    # conda install -c conda-forge scikit-learn='1.0.*'
    # For building docs locally
    # conda install sphinx -c conda-forge
    # pip install sphinx_rtd_theme
    # pip install sphinx_rtd_dark_mode
    git clone https://github.com/Bodo-inc/Bodo.git
    cd Bodo
    # build Bodo
    python setup.py develop

For local development you will also want to enable more detailed error messages.
This can be done with ``export NUMBA_DEVELOPER_MODE=1``. To ensure this is activated
on every new shell you should also add this to your preferred shell's rc file.

For HDFS related development, use the :ref:`docker image <docker-images>`.

Troubleshooting MacOS Build
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* If your Mac0S sdk hasn't been configured properly, you may get a clang related error like the following::

     ld: unsupported tapi file type '!tapi-tbd' in YAML file

* Add the following line to your .zshrc file::

    export CONDA_BUILD_SYSROOT=/opt/MacOSX10.15.sdk

* Execute buildscripts/setup_osx_sdk.sh to resolve this issue.


On Windows
~~~~~~~~~~

* Install `Microsoft Build Tools for Visual Studio 2019 <https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2019>`_::

  In Build tools, install C++ build tools and ensure the latest versions of MSVCv142 - VS 2019 C++ x64/x86 build
  tools and Windows 10 SDK are checked.

* Install `Miniconda for Windows <https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe>`_.

* Start "Anaconda Prompt (miniconda3)"

* Set up the Visual C++ environment::

  The Visual Studio and VS C++ variables need to be in your environment.
  If they are not already set (for example look for ``VCINSTALLDIR``,
  ``VCToolsVersion`` and ``VisualStudioVersion`` in your environment variables)
  you can do so by running ``vcvars64.bat`` inside the miniconda
  prompt. For Microsoft Visual Studio this batch file is located in::

  C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat

* Set up the Conda environment in Anaconda Prompt::

    conda create -n DEV python numpy scipy pandas="1.3.*" boost-cpp -c conda-forge
    conda activate DEV
    conda install numba=0.55.0 -c conda-forge
    conda install mpi4py msmpi cython -c conda-forge
    conda install -c conda-forge pyarrow=5.0.0
    # Required for IO with S3
    conda install -c conda-forge fsspec
    # The following is required for s3 related development and tests
    # conda install -c conda-forge boto3 botocore
    # The following is required for IO with gcs
    # conda install -c conda-forge gcsfs
    # The following is required for matplotlib Testing
    # conda install -c conda-forge 'matplotlib<=3.4.3'
    # Required for ML tests
    # conda install -c conda-forge scikit-learn='1.0.*'
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

    conda install pytest sphinx pylint jupyter xlrd xlsxwriter openpyxl ipyparallel pyspark jupyterlab aws-sdk-cpp


Cleaning Bodo
~~~~~~~~~~~~~

The script `clean.sh` is provided to clean up all the leftover files after compilation.
It removes all C++ compiled code and the `__pycache__` directories.
