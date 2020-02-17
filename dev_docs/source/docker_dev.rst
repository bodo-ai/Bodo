.. _docker_dev:

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
    1. Bodo development::

        cd docker/bodo_dev
        docker build -t bodo_dev . 

    2. Bodo development with valgrind::

        cd docker/bodo_dev_valgrind
        docker build -t bodo_dev_valgrind .  
        
    `List images <https://docs.python.org/3/library/pdb.html>`_  with :code:`docker images`
    `Remove image <https://docs.docker.com/engine/reference/commandline/rmi/>`_ with :code:`docker rmi your_image_id`

2. `Run <https://docs.docker.com/engine/reference/commandline/run/>`_ a command in the new containers:
    1. Bodo development::

        # -it, connect the container to terminal
        # replace ~/Bodo with your path to Bodo
        docker run -it -v ~/Bodo:/Bodo bodo_dev

    2. Bodo development with valgrind::

        # -it, connect the container to terminal
        # replace ~/Bodo with your path to Bodo
        docker run -it -v ~/Bodo:/Bodo bodo_dev_valgrind
        
    Other useful flags & `bind mounts <https://docs.docker.com/storage/bind-mounts/>`_::

        # bodo_dev is the image we are using here 
        # -v your_path:path_in_docker, mounts directory
        # -m, memory limit
        # --oom-kill-disable, whether to disable OOM Killer for the container or no
        docker run -it -m 16000m --oom-kill-disable -v ~/Bodo:/Bodo -v ~/claims_poc:/claims_poc build bodo_dev
        
3. Build Bodo in container::

       cd ../Bodo
       HDF5_DIR=$CONDA_PREFIX python setup.py develop

4. Use valgrind in Bodo development with valgrind::

       cd ../src
       
       # run valgrind with python, replace your_python_script.py with your own
       valgrind --suppressions=valgrind-python.supp --error-limit=no --track-origins=yes python -u your_python_script.py
       
       # redirect valgrind log and python stdout to out.txt
       valgrind --suppressions=valgrind-python.supp --error-limit=no --track-origins=yes python -u your_python_script.py &>out.txt
       
       # valgrind with mpiexec
       valgrind --suppressions=valgrind-python.supp --error-limit=no --track-origins=yes mpiexec -n 2 python -u your_python_script.py

To run a command in a running container: Use :code:`docker container ls` to find the running container ID::

    # replace d030f4d9c8ac with your container ID
    docker exec -it d030f4d9c8ac bash    

`List <https://docs.docker.com/engine/reference/commandline/ps/>`_ all running and stopped containers: :code:`docker ps`

To `stop <https://docs.docker.com/engine/reference/commandline/stop/>`_ and `remove <https://docs.docker.com/engine/reference/commandline/rm/>`_ a container::

    # first stop the container
    docker stop your_container_ID
    # then remove the container 
    docker rm your_container_ID

To remove all stopped containers:: 

    docker rm -v $(docker ps -qa)
