# Make sure to build the docker image from the top folder:
# docker build -t bodo_dev_valgrind -f docker/bodo_dev_valgrind/Dockerfile .

# A initial version
# TODO: hava Python compiled from source to work with conda
FROM --platform=linux/amd64 buildpack-deps:stable
ARG PYTHON_VERSION=3.12

ENV PYTHONMALLOC malloc
ENV MKL_DISABLE_FAST_MM 1

# Ubuntu Install. Best to do in 1 line and remove cache at end
# https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#run
RUN apt-get update && apt-get install -y wget bzip2 git vim make curl && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Builds python with the proper flags, and then copies the suppression file generated during
# the build process to /src.
RUN set -e \
    && apt-get update && apt-get install -y valgrind --no-install-recommends  \
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
# Setup Dev Environment
RUN curl -fsSL https://pixi.sh/install.sh | bash
RUN pixi install
RUN pixi run build
RUN pixi add nomkl

# Creation of the .bashrc file
RUN echo "pixi shell" > ~/.bashrc

ENTRYPOINT ["bash"]
