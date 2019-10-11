.. _development:

Bodo Development
================

Technology Overview and Architecture
------------------------------------

This `slide deck <https://drive.google.com/file/d/1mHrbjAEfP6p-o-bWJOVdtmKNEA7lreDt/view?usp=sharing>`_
provides an overview of Bodo technology and software architecture.

Numba Development
-----------------

Bodo sits on top of Numba and is heavily tied to many of its features.
Therefore, understanding Numba's internal details and being able to
develop Numba extensions is necessary.


- Start with `basic overview of Numba use <http://numba.pydata.org/numba-doc/latest/user/5minguide.html>`_ and try the examples.
- `User documentation <http://numba.pydata.org/numba-doc/latest/user/index.html>`_ is generally helpful for overview of features.
- | `ParallelAccelerator documentation <http://numba.pydata.org/numba-doc/latest/user/parallel.html>`_
    provides overview of parallel analysis and transformations in Numba
    (also used in Bodo).
- `Setting up Numba for development <http://numba.pydata.org/numba-doc/latest/developer/contributing.html>`_
- | `Numba architecture page <http://numba.pydata.org/numba-doc/latest/developer/architecture.html>`_
    is a good starting point for understanding the internals.
- | The `overload guide page of Numba <http://numba.pydata.org/numba-doc/latest/extending/overloading-guide.html>`_
    is useful for understanding the process of implementing
    new functionality and specializing to data types.
- | Learning Numba IR is crucial for understanding transformations.
    See the `IR classes <https://github.com/numba/numba/blob/master/numba/ir.py>`_.
    Setting `NUMBA_DEBUG_ARRAY_OPT=1` shows the IR at different stages
    of ParallelAccelerator and Bodo transformations. Run `a simple parallel
    example <http://numba.pydata.org/numba-doc/latest/user/parallel.html#explicit-parallel-loops>`_
    and make sure you understad the IR at different stages.
- | `Exending Numba page <http://numba.pydata.org/numba-doc/latest/extending/index.html>`_
    provides details on how to provide native implementations for data types and functions.
    The low-level API should be avoided as much as possible for ease of development and
    code readability. The `unicode support <https://github.com/numba/numba/blob/master/numba/unicode.py>`_
    in Numba is an example of a modern extension for Numba (documentation planned).
- | A more complex extension is `the new dictionary implementation in
    Numba <https://github.com/numba/numba/blob/master/numba/dictobject.py>`_ (documentation planned).
    It has examples of calling into C code which is implemented as
    `a C extension library <https://github.com/numba/numba/blob/master/numba/_dictobject.c>`_.
    For a simpler example of calling into C library, see Bodo's I/O features like
    `get_file_size <https://github.com/IntelLabs/bodo/blob/master/bodo/io.py#L12>`_.
- | `Developer reference manual <http://numba.pydata.org/numba-doc/latest/developer/index.html>`_
    provides more details if necessary.

Bodo Development
----------------

Bodo implements Pandas and Numpy API as a DSL.
Data structures are implemented as Numba extensions, and
compiler stages are responsible for different levels of abstraction.
For example, `Series data type support <https://github.com/IntelLabs/bodo/blob/master/bodo/hiframes/pd_series_ext.py>`_
and `Series transformations <https://github.com/IntelLabs/bodo/blob/master/bodo/transforms/series_pass.py>`_
implement the `Pandas Series API <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html>`_.
Follow the pipeline for a simple function like `Series.sum()`
for initial understanding of the transformations.

Code Structures
---------------

Below is the high level structure of the code.

- `decorators.py` is the starting point, which defines decorators of Bodo.
  Currently just `@jit` is provided but more is expected.
- `compiler.py` defines the compiler pipeline for this decorator.
- `transforms` directory defines Bodo specific analysis and transformation passes.

  - `untyped_pass.py`: transforms the IR to remove features that Numba's type inference cannot support
    such as non-uniform dictionary input of `pd.DataFrame({})`.
  - `dataframe_pass.py`: converts data frame operations to Series and Array operations
    as much as possible to provide implementation and enable optimization.
    Creates specialized IR nodes for complex operations like Join.
  - `series_pass.py`: converts Series operations to array operations as much as possible
    to provide implementation and enable optimization.
  - `distributed_analysis.py`: analyzes the IR to decide parallelism of arrays and parfors
    for distributed transformation.
  - `distributed_pass.py`: parallelizes the IR for distributed execution and inserts MPI calls.

- `hiframes` directory provides Pandas functionality such as DataFrame, Series and Index.
- `ir` directory defines and implements Bodo specific IR nodes such as Sort and Join.
- `libs` directory provides supporting data structures and libraries such as strings,
  dictionary, quantiles, timsort. It also includes helper C extensions.
- `io` directory provides I/O support such as CSV, HDF5, Parquet and Numpy.
- `tests` provides unittests.

For each function implemented (either overloading Pandas or internal),
the following has to be specified:

- side effects for dead code elimination
- aliasing (inlining if necessary)
- array analysis
- distributed analysis (including array access analysis)
- distributed transformation

Develop using Docker
---------------
Two Docker images can be used for Bodo development:

1. Bodo development(:code:`Bodo/docker/bodo_dev`)
    - has conda enviroment setup
2. Bodo development with valgrind(:code:`Bodo/docker/bodo_dev_valgrind`)
    - has conda enviroment setup
    - has python and valgrind configured

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
        
    Other useful falgs & `bind mounts <https://docs.docker.com/storage/bind-mounts/>`_:
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
    
    
Debugging
---------------
- `pdb <https://docs.python.org/3/library/pdb.html>`_: :code:`import pdb; pdb.set_trace()` for breakpoints

- `NUMBA_DEBUG_PRINT_AFTER <https://numba.pydata.org/numba-doc/dev/reference/envvars.html?highlight=numba_debug_print#envvar-NUMBA_DEBUG_PRINT_AFTER>`_ enviroment variable: 
  ::
      # example of printing after parfor pass
      export NUMBA_DEBUG_PRINT_AFTER='parfor_pass'
      # other common ones: 'bodo_distributed_pass', 'bodo_series_pass'

- mpiexec redirect stdout from differet processes to different files:
  ::
      export PYTHONUNBUFFERED=1 # set the enviroment variable 
      mpiexec -outfile-pattern="out_%r.log" -n 8 python small_test01.py

  or :
  ::
      # use the flag instead of setting the enviroment variable
      mpiexec -outfile-pattern="out_%r.log" -n 8 python -u small_test01.py

Papers
------

These papers provide deeper dive in technical ideas
(may not be necessary for many developers):

- `Bodo paper on automatic parallelization for distributed memory <http://dl.acm.org/citation.cfm?id=3079099>`_
- `Bodo paper on system architecture versus Spark <http://dl.acm.org/citation.cfm?id=3103004>`_
- `Bodo Dataframe DSL approach <https://arxiv.org/abs/1704.02341>`_
- `ParallelAccelerator DSL approach <https://users.soe.ucsc.edu/~lkuper/papers/parallelaccelerator-ecoop17.pdf>`_
