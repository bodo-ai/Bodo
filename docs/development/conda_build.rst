.. _development:

Conda Build Bodo
----------------------
::

    cd docker/build_instance/
    
    # run docker build container
    docker build -t build .  # if necessary
    docker run -it build bash
    
    # clone and build bodo in the container
    git clone https://github.com/Bodo-inc/Bodo.git
    pip install astunparse
    python buildscripts/remove_docstring.py
    cd Bodo/buildscripts/bodo-conda-recipe/
    conda-build . -c defaults -c numba/label/dev --no-test

Open a new terminal window (replace yourContainerID :code:`docker container ls` and the name of `.tar.bz2` File)::
    
    # copy built file to host and pack into conda channel
    docker cp yourContainerID:/root/miniconda3/conda-bld/linux-64/bodo-2019.09.2-py37hc547734_0.tar.bz2 .
    mkdir bodo-inc
    mkdir bodo-inc/linux-64
    cp bodo-2019.09.1-py37hc547734_0.tar.bz2 bodo-inc/linux-64/
    conda index bodo-inc/
