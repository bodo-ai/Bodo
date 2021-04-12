FROM ubuntu:20.04

RUN rm /bin/sh && ln -s /bin/bash /bin/sh
RUN apt-get update && apt-get install -y wget git

ADD buildscripts/setup_conda.sh .
ADD buildscripts/aws/setup_minio.sh .
ADD buildscripts/aws/test_installs.sh .

ENV CONDA_ENV BodoCodeBuild
ENV RUN_NIGHTLY no

RUN ./setup_conda.sh
RUN ./setup_minio.sh
RUN ./test_installs.sh

# This docker image is LARGE, so we remove unnecessary files
# after the environment is created. This will increase the docker
# image size because it creates layers, so we use docker-squash
# https://pypi.org/project/docker-squash/ to generate a new
# smaller image.
# TODO: Determine if there are more files that can be removed
# i.e. source files that are compiled.
ADD buildscripts/aws/clean_conda.sh .
RUN ./clean_conda.sh


CMD ["/bin/bash"]
