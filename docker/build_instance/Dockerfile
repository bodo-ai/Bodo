FROM ubuntu:16.04

WORKDIR /root
RUN rm /bin/sh && ln -s /bin/bash /bin/sh


# Ubuntu Install. Best to do in 1 line and remove cache at end
# https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#run
RUN apt-get update && apt-get install -y wget bzip2 git vim make && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O miniforge.sh \
	&& chmod +x miniforge.sh \
	&& ./miniforge.sh -b
ENV PATH /root/miniforge3/bin:${PATH}
RUN echo "export PATH=/root/miniforge3/bin:${PATH}" >> /root/.bashrc

RUN mamba install -y conda-build anaconda-client 
