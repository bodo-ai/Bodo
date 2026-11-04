# To test: docker build -t test_dockerhub .
FROM mambaorg/micromamba:latest AS builder

# Install PyIceberg build dependencies
# TODO: Remove when PyIceberg has a 3.15 wheel/conda-forge package available.
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libc6-dev && \
    rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER

ENV DEBIAN_FRONTEND=noninteractive

ENV JUPYTER_ENABLE_LAB=yes

COPY --chown=$MAMBA_USER:$MAMBA_USER ./buildscripts/docker/docker_release/env.yaml /tmp/env.yaml

RUN micromamba install  -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

FROM ubuntu:latest

USER root
# Copy the conda environment from the builder stage
COPY --chown=root --from=builder /opt/conda /opt/conda

# Install necessary packages
RUN apt-get update && \
    apt-get install -y vim && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH=/opt/conda/bin:$PATH

# Create a working directory for the Jupyter Notebook
WORKDIR /app

# Copy Getting-Started notebook files to the container
COPY ./buildscripts/docker/docker_release/pi.ipynb ./buildscripts/docker/docker_release/quickstart.ipynb /app/

# Expose port 8888 for the Jupyter Notebook
EXPOSE 8888

# Start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
