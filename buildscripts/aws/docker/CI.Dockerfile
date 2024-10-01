FROM ubuntu:20.04

RUN rm /bin/sh && ln -s /bin/bash /bin/sh
RUN apt-get update && apt-get install -y wget git jq unzip && rm -rf /var/lib/apt/lists/*
RUN wget "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" \
    && unzip awscli-exe-linux-x86_64.zip \
    && ./aws/install \
    && rm -rf awscli-exe-linux-x86_64.zip

ADD buildscripts/setup_conda.sh .
ADD buildscripts/aws/setup_minio.sh .
ADD buildscripts/envs/conda-lock.yml ./buildscripts/envs/conda-lock.yml

ENV CONDA_ENV=BodoCodeBuild
ENV PYTHON_VERSION=3.12

RUN ./setup_conda.sh
RUN ./setup_minio.sh

# TODO: Determine if there are more files that can be removed
# i.e. source files that are compiled.

CMD ["/bin/bash"]
