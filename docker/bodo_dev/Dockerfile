FROM ubuntu:latest AS env_setup

WORKDIR /root
RUN apt-get update && apt-get install -y wget bzip2 git vim make curl\
	&& wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
	&& chmod +x miniconda.sh\
	&& ./miniconda.sh -b
ENV PATH /root/miniconda3/bin/:${PATH}
RUN conda create -n BODODEV numpy scipy pandas boost cmake h5py pyarrow mpich mpi gcc_linux-64 gxx_linux-64 gfortran_linux-64
RUN echo "source activate BODODEV" > ~/.bashrc
RUN conda install numba=0.47.0
RUN conda install -c defaults -c conda-forge hdf5=*=*mpich*
