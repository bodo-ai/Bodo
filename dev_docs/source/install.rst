.. _install:


Building Bodo from Source
-------------------------

We use `Anaconda <https://www.anaconda.com/download/>`_ distribution of
Python for setting up Bodo. These commands install Bodo and its dependencies
such as Numba on Ubuntu Linux::

    # Linux: wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    # Mac: wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
    chmod +x miniconda.sh
    ./miniconda.sh -b
    export PATH=$HOME/miniconda3/bin:$PATH
    conda create -n DEV python=3.7 numpy scipy pandas boost-cpp cmake h5py mpich mpi -c conda-forge
    source activate DEV
    # Linux: conda install gcc_linux-64 gxx_linux-64 gfortran_linux-64 -c conda-forge
    # Mac: conda install clang_osx-64 clangxx_osx-64 gfortran_osx-64 -c conda-forge
    conda install numba=0.46.0 -c conda-forge
    conda install -c bodo.ai -c conda-forge hdf5=*=*mpich*
    conda install -c bodo.ai -c conda-forge pyarrow=0.15.1 arrow-cpp=0.15.1=*transfer_s3*
    git clone https://github.com/Bodo-inc/Bodo.git
    cd Bodo
    # build Bodo
    HDF5_DIR=$CONDA_PREFIX python setup.py develop


A command line for running the Pi example on 4 cores::

    mpiexec -n 4 python examples/pi.py

If you run into gethostbyname failed error, try
`this fix <https://stackoverflow.com/questions/23112515/mpich2-gethostbyname-failed>`_.

Running unit tests::

    conda install pytest
    pytest -x -s -v -m "not slow" -W ignore

In case of issues, reinstalling in a new conda environment is recommended.


Other useful packages for development::
    conda install pytest sphinx pylint jupyter
    .. conda install daal4py -c defaults -c intel


Building Documentation
----------------------

The `documentation <https://docs.bodo.ai>`_ is under the `docs` directory of the repository and uses
the reStructuredText format.
It is built with `Sphinx <http://www.sphinx-doc.org>`_ and the read the doc theme::

    conda install sphinx
    conda install alabaster

After updating documentation,
run :code:`make html` in the `docs` folder to build.
Open `index.html` to view the documentation.
To upload the documentation, use the :code:`gh-pages.py`
script under :code:`docs`::

    python gh-pages.py

Then verify the repository under the :code:`gh-pages` directory and
:code:`git push` to `Bodo-doc <https://github.com/Bodo-inc/Bodo-doc>`_ repo :code:`gh-pages` branch.

The developer documentation is under the `dev_docs` directory
and can be built with :code:`make html` as well.


Develop using Docker
--------------------
Two Docker images can be used for Bodo development:

(**optional**: the reason we have this is mainly for Mac users since Mac doesn't show certain dependency errors and valgrind does not support the newest macOS)

1. Bodo development(:code:`Bodo/docker/bodo_dev`)
    - has conda enviroment setup
2. Bodo development with valgrind(:code:`Bodo/docker/bodo_dev_valgrind`)
    - has conda enviroment setup
    - has python and valgrind configured
        - :code:`PYTHONMALLOC=malloc`, whereas default is :code:`PYTHONMALLOC=pymalloc`
        - a different version of numpy
        - more configurations can be found in the dockerfile

Setup:

1. `Build <https://docs.docker.com/engine/reference/commandline/build/>`_ Docker image:
    1. Bodo development
    ::
        cd docker/bodo_dev
        docker build -t bodo_dev . 
    2. Bodo development with valgrind  
    ::
        cd docker/bodo_dev_valgrind
        docker build -t bodo_dev_valgrind .  
    `List images <https://docs.python.org/3/library/pdb.html>`_  with :code:`docker images`
    `Remove image <https://docs.docker.com/engine/reference/commandline/rmi/>`_ with :code:`docker rmi your_image_id`

2. `Run <https://docs.docker.com/engine/reference/commandline/run/>`_ a command in the new containers:
    1. Bodo development
    ::
        # -it, connect the container to terminal
        # replace ~/Bodo with your path to Bodo
        docker run -it -v ~/Bodo:/Bodo bodo_dev
    2. Bodo development with valgrind 
    ::
        # -it, connect the container to terminal
        # replace ~/Bodo with your path to Bodo
        docker run -it -v ~/Bodo:/Bodo bodo_dev_valgrind
        
    Other useful flags & `bind mounts <https://docs.docker.com/storage/bind-mounts/>`_:
    ::
        # bodo_dev is the image we are using here 
        # -v your_path:path_in_docker, mounts directory
        # -m, memory limit
        # --oom-kill-disable, whether to disable OOM Killer for the container or no
        docker run -it -m 16000m --oom-kill-disable -v ~/Bodo:/Bodo -v ~/claims_poc:/claims_poc build bodo_dev
        
3. Build Bodo in container:
   ::
       cd ../Bodo
       HDF5_DIR=$CONDA_PREFIX python setup.py develop

4. Use valgrind in Bodo development with valgrind 
   :: 
       cd ../src
       
       # run valgrind with python, replace your_python_script.py with your own
       valgrind --suppressions=valgrind-python.supp --error-limit=no --track-origins=yes python -u your_python_script.py
       
       # redirect valgrind log and python stdout to out.txt
       valgrind --suppressions=valgrind-python.supp --error-limit=no --track-origins=yes python -u your_python_script.py &>out.txt
       
       # valgrind with mpiexec
       valgrind --suppressions=valgrind-python.supp --error-limit=no --track-origins=yes mpiexec -n 2 python -u your_python_script.py

To run a command in a running container: Use :code:`docker container ls` to find the running container ID
::
    # replace d030f4d9c8ac with your container ID
    docker exec -it d030f4d9c8ac bash    

`List <https://docs.docker.com/engine/reference/commandline/ps/>`_ all running and stopped containers: :code:`docker ps`

To `stop <https://docs.docker.com/engine/reference/commandline/stop/>`_ and `remove <https://docs.docker.com/engine/reference/commandline/rm/>`_ a container:
:: 
    # first stop the container
    docker stop your_container_ID
    # then remove the container 
    docker rm your_container_ID

To remove all stopped containers:
:: 
    docker rm -v $(docker ps -qa)




Building from Source on Windows
-------------------------------

* Install Visual Studio Community 2017 (15.9.18)
* From the Visual Studio installer, install following individual components::

    Windows 10 SDK (10.0.17763.0)
    Windows Universal CRT SDK
    VC++ 2015.3 v14.00 (v140) toolset for desktop

* Install `Miniconda for Windows <https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe>`_.
* Start 'Anaconda (Miniconda3) prompt'
* Setup the Conda environment in Anaconda Prompt::

    conda create -n DEV numpy scipy pandas boost-cpp cmake h5py pyarrow
    source activate DEV
    conda install -c defaults numba
    conda install vc vs2015_runtime vs2015_win-64
    conda install -c defaults -c intel impi_rt impi-devel
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
