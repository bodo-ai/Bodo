# Make sure to build the docker image from the top folder:
# docker build -t bodo_dev -f docker/bodo_dev/Dockerfile .
FROM --platform=linux/amd64 ubuntu:latest AS env_setup

WORKDIR /root

# Ubuntu Install. Best to do in 1 line and remove cache at end
# https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#run
RUN apt-get update && apt-get install -y wget bzip2 git vim && rm -rf /var/lib/apt/lists/*

# Setup Mambaforge
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh -O mambaforge.sh \
	&& chmod +x mambaforge.sh \
	&& ./mambaforge.sh -b
ENV PATH /root/mambaforge/bin:${PATH}

# Creation of the BODODEV environment
ADD buildscripts/envs/conda-lock.yml conda-lock.yml
RUN mamba install -y -c conda-forge conda-lock
RUN conda-lock install --dev --mamba -n BODODEV conda-lock.yml
RUN mamba install -y -n BODODEV -c conda-forge flaky ipython
RUN conda run pip install astor

# Creation of the initial .bashrc to be right up
RUN echo "source activate BODODEV" > ~/.bashrc
