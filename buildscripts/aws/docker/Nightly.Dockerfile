FROM ubuntu:20.04

RUN rm /bin/sh && ln -s /bin/bash /bin/sh
RUN apt-get update && apt-get install -y wget git curl

ADD buildscripts/setup_conda.sh .
ADD buildscripts/aws/test_installs.sh .

ENV CONDA_ENV BodoCodeBuild
ENV RUN_NIGHTLY yes

RUN ./setup_conda.sh
RUN ./test_installs.sh

# This docker image is VERY LARGE, so we remove unnecessary files
# after the environment is created. This will increase the docker
# image size because it creates layers, so we use docker-squash
# https://pypi.org/project/docker-squash/ to generate a new
# smaller image.
# TODO: Determine if there are more files that can be removed
# i.e. source files that are compiled.

# Cleanup leftover compressed files
RUN bash -c "rm -r /root/miniconda3/pkgs/*.bz2"
RUN bash -c "rm -r /root/miniconda3/pkgs/*.conda"

# Cleanup leftover cache files
RUN bash -c "rm -r /root/miniconda3/pkgs/cache"
RUN bash -c "rm -r /root/.cache"

CMD ["/bin/bash"]
