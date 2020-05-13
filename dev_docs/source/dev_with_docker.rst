.. _docker_dev:

Develop using Docker
--------------------

Docker images
~~~~~~~~~~~~~~
.. _docker-images:

Three Docker images can be used for Bodo development:

(**optional**: the reason we have this is mainly for Mac users since Mac doesn't show certain dependency errors, valgrind does not support the newest macOS, and hdfs3(Hadoop) setup is difficult for Mac.)

1. Bodo development(:code:`Bodo/docker/bodo_dev`)
    - has conda enviroment setup
2. Bodo development with valgrind(:code:`Bodo/docker/bodo_dev_valgrind`)
    - has conda enviroment setup
    - has python and valgrind configured
        - :code:`PYTHONMALLOC=malloc`, whereas default is :code:`PYTHONMALLOC=pymalloc`
        - a different version of numpy
        - more configurations can be found in the dockerfile
3. Bodo development for hdfs(:code:`Bodo/docker/bodo_dev_hdfs`)
    - has conda enviroment setup
    - has hadoop and libhdfs installed
    - has hadoop configured for `pseudo distributed <https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/SingleCluster.html#Pseudo-Distributed_Operation>`_
    - has environment variables configured for hadoop and `arrow <https://arrow.apache.org/docs/python/filesystems_deprecated.html>`_
    - has ssh configured
    - builds Bodo(requires mounting Bodo directory)

Setup:

1. `Build <https://docs.docker.com/engine/reference/commandline/build/>`_ Docker image:
    1. Bodo development::

        cd docker/bodo_dev
        docker build -t bodo_dev . 

    2. Bodo development with valgrind::

        cd docker/bodo_dev_valgrind
        docker build -t bodo_dev_valgrind .  
    
    3. Bodo development for hdfs::

        cd docker/bodo_dev_hdfs
        docker build -t bodo_dev_hdfs . 

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

    3. Bodo development for hdfs::

        # -it, connect the container to terminal
        # -p publish container port so that we can browse web interface
        # MUST replace ~/Bodo with your path to Bodo
        docker run -it -p 9870:9870 -v ~/Bodo:/Bodo bodo_dev_hdfs
        
    Other useful flags & `bind mounts <https://docs.docker.com/storage/bind-mounts/>`_::

        # bodo_dev is the image we are using here 
        # -v your_path:path_in_docker, mounts directory
        # -m, memory limit
        # --oom-kill-disable, whether to disable OOM Killer for the container or no
        docker run -it -m 16000m --oom-kill-disable -v ~/Bodo:/Bodo -v ~/claims_poc:/claims_poc build bodo_dev
        
3. Build Bodo in container, not necessary for ``bodo_dev_hdfs``::

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

5. Run Hadoop(NameNode daemon and DataNode daemon) in Bodo development for hdfs::

       /etc/init.d/ssh start
       # check that you can ssh to the localhost without a passphrase
       ssh localhost
       cd /opt/hadoop-3.2.1
       # Format the filesystem
       bin/hdfs namenode -format
       # Start NameNode daemon and DataNode daemon
       sbin/start-dfs.sh
       jps

  The output should look something like this::

    66692 SecondaryNameNode
    66535 DataNode
    67350 Jps
    66422 NameNode

  Web interface is available locally at: http://localhost:9870/.
  Test hdfs test suite with ``pytest -s -v -m "hdfs"`` .

  To stop Hadoop, run ``sbin/stop-dfs.sh``. 

  If you need to restart Hadoop after, run ``rm -rf /tmp/hadoop-root`` (without removing this directory, DataNode will not restart), and then repeat above steps from formatting the filesystem. If ``jps`` output is still not as expected, run a new docker container.

Other useful docker commands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
