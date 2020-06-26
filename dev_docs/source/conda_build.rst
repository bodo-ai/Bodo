.. _condabuild:

Conda Build
----------------------
Conda Build Bodo
~~~~~~~~~~~~~~~~~~
::

    # tags Bodo version
    git tag 2019.10
    git push --tags

    cd docker/build_instance/

    # run docker build container
    docker build -t build .  # if necessary
    docker run -it build bash

    # clone and build bodo in the container
    git clone https://github.com/Bodo-inc/Bodo.git
    pip install astunparse
    cd Bodo
    python buildscripts/remove_docstring.py
    cd buildscripts/bodo-conda-recipe/
    # if binary should check license expiration
    # export CHECK_LICENSE_EXPIRED=1
    # if binary should check license max cores
    # export CHECK_LICENSE_CORE_COUNT=1
    conda-build . -c bodo.ai -c conda-forge --no-test

Open a new terminal window (replace yourContainerID :code:`docker container ls`
and the name of `.tar.bz2` File)::

    # copy built file to host and pack into conda channel
    docker cp yourContainerID:/root/miniconda3/conda-bld/linux-64/bodo-2019.09.2-py37hc547734_0.tar.bz2 .
    mkdir bodo-inc
    mkdir bodo-inc/linux-64
    cp bodo-2019.09.1-py37hc547734_0.tar.bz2 bodo-inc/linux-64/
    conda index bodo-inc/

To build :code:`bodo` from the file bodo-inc::

    conda install bodo -c file:///bodo-inc/ -c bodo.ai -c conda-forge
    
Conda Build :code:`arrow-cpp`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can ship our own :code:`arrow-cpp` package on `bodo.ai Anaconda Cloud <https://anaconda.org/bodo.ai/arrow-cpp/files>`_ if the conda package misses certain features.
Deployment is done through `CI <https://dev.azure.com/bodo-inc/Bodo/_build?definitionId=4&_a=summary>`_. :code:`CONDA_UPLOAD_TOKEN` is a enviroment
variable generated from bodo.ai Anaconda Cloud account and set through Azure Pipeline's UI. To update the package,
change :code:`build/number` properly and manually run the `pipeline <https://dev.azure.com/bodo-inc/Bodo/_build?definitionId=4&_a=summary>`_.
