# A initial version
# TODO: hava Python compiled from source to work with conda
FROM buildpack-deps:stretch
ARG PYTHON_VERSION=3.8.0

ENV PYTHONMALLOC malloc
ENV MKL_DISABLE_FAST_MM 1

WORKDIR /src

RUN set -e \
    && apt-get update && apt-get install -y  valgrind --no-install-recommends  \
    && curl "https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz" > python.tgz  \
    && mkdir -p /usr/src/python \
    && tar -xzC /usr/src/python --strip-components=1 -f python.tgz \
    && rm python.tgz \
    && cd /usr/src/python \
    && ./configure \
        --with-pydebug \
        --with-valgrind \
        --without-pymalloc \
    && make -j "$(nproc)" \
    && make install \
    && find /usr/lib -type d -a -name '__pycache__' -exec rm -rf '{}' + \
    && cp Misc/valgrind-python.supp /src \
    && rm -rf /usr/src/python \
    && apt-get install -y python3-pip --no-install-recommends && rm -rf /var/lib/apt/lists/* \
    && ln -s $(which python3) /usr/local/bin/python \
    && ln -s $(which pip3) /usr/local/bin/pip

WORKDIR /root
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && chmod +x miniconda.sh\
    && ./miniconda.sh -b
ENV PATH /root/miniconda3/bin/:${PATH}
#
# Creation of the BODODEV environment
#
RUN conda create -n BODODEV numpy scipy pandas>=1.0.0 boost-cpp cmake h5py mpich mpi gcc_linux-64 gxx_linux-64 gfortran_linux-64 -c conda-forge
RUN conda install -n BODODEV python hdf5=*=*mpich* -c conda-forge
RUN conda install -n BODODEV python pyarrow=0.17.0 -c conda-forge
RUN conda install -n BODODEV python numba=0.49.0 -c conda-forge
RUN conda install -v BODODEV python nomkl numpy -c conda-forge
RUN conda install -v BODODEV python pytest sqlalchemy mypysql mpi4py -c conda-forge
#
# Creation of the .bashrc file
#
RUN echo "source activate BODODEV" > ~/.bashrc

ENTRYPOINT ["bash"]
