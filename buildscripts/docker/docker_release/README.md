# Bodo DockerHub Image

Bodo publishes a new DockerHub image with each release, enabling users to run Bodo inside Docker containers.

The image includes two example notebooks to help you get started with Bodo:
    - quickstart.ipynb: Demonstrates how to use Bodo and BodoSQL for Pandas and data transformation applications.
    - pi.ipynb: Implements a Monte Carlo simulation to estimate the value of pi.

NOTE: 
- This image is compatible with Linux x86 and macOS x86 platforms only.
- The image is intended for testing and learning purposes and should not be used in production.

## Prerequisites
Before you begin, ensure the following:

1. The latest version of Docker is installed on your machine.
2. The Docker daemon is running (e.g., via Docker Desktop).

## Download and Run the Container

The Bodo image is available on DockerHub:
https://hub.docker.com/r/bodoai/bodoai_image

### Steps

1. Download the Bodo image:
```shell
docker pull bodoai/bodo-jupyterlab:latest
```

2. Start the Docker container with JupyterLab:
Replace `8` in `--cpus=8` with the desired number of CPUs:

```shell
docker run -p 8888:8888 --cpus=8 bodoai/bodo-jupyterlab
```

3. Access the JupyterLab environment:
At the end of the terminal output, you’ll see a link similar to this:

```shell
 http://127.0.0.1:8888/?token=7b48621b8f12...
```
Open the link in your browser to access the two included notebooks.

Enjoy!

You can also run container using the shell instead of JupyterLab by doing the following steps:

```
docker run -dt bodoai/bodo-jupyterlab
docker exec -it <container_id> bash
```
The first command starts the container in detached mode and will print the `container_id`. The second one, opens a shell session inside the container.


## Stop the container
To stop the running container:
1. Find the container ID by running `docker ps`
2. Stop the container using the following command, replacing <container-id> with the actual ID: `docker stop <container-id>`

## Build the Image Locally
If you prefer to build the image locally instead of downloading it:
1. Download [docker_release folder](https://github.com/bodo-ai/Bodo/tree/main/buildscripts/docker) containing the Dockerfile from the repository (e.g., by using tools like GitHub’s “Download ZIP” for a single folder).
2. Update those file paths (`env.yml`, `pi.ipynb`, and `quickstart.ipynb`) in the `COPY` step in the  Dockerfile to ensure they match your local directory structure.
3. Navigate to the folder and build the Docker image:

```shell
docker build -t bodo-img .
```

4. Start the container:

```shell
docker run -p 8888:8888 --cpus=8 bodo-img
```
