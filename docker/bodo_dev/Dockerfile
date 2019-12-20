FROM ubuntu:latest AS env_setup

WORKDIR /root
RUN apt-get update && apt-get install -y wget bzip2 git vim make curl\
	&& wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
	&& chmod +x miniconda.sh\
	&& ./miniconda.sh -b
ENV PATH /root/miniconda3/bin/:${PATH}
RUN conda create -n BODODEV numpy scipy pandas boost-cpp cmake h5py mpich mpi gcc_linux-64 gxx_linux-64 gfortran_linux-64 -c conda-forge
RUN echo "source activate BODODEV" > ~/.bashrc
RUN conda install numba=0.46.0 -c conda-forge
RUN conda install -c conda-forge hdf5=*=*mpich*
RUN conda install -c bodo.ai -c conda-forge pyarrow arrow-cpp=*=*transfer_s3*
