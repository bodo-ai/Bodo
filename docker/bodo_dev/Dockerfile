FROM ubuntu:latest AS env_setup
#
# The docker is compiled for example with the command
# docker build -t bodo_dev .
#

WORKDIR /root
#
# Ubuntu installs and Miniconda
#
RUN apt-get update && apt-get install -y wget bzip2 git vim make curl\
	&& wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
	&& chmod +x miniconda.sh\
	&& ./miniconda.sh -b
ENV PATH /root/miniconda3/bin/:${PATH}
#
# Creation of the BODODEV environment
#
RUN conda create -n BODODEV numpy scipy pandas='1.2.*' boost-cpp=1.74.0 cmake h5py mpich mpi gcc_linux-64 gxx_linux-64 python=3.8 -c conda-forge
RUN conda install -n BODODEV hdf5=*=*mpich* -c conda-forge
RUN conda install -n BODODEV python pyarrow=2.0.0 -c conda-forge
RUN conda install -n BODODEV pytest sqlalchemy pymysql xlrd openpyxl mpi4py cython
RUN conda install -n BODODEV numba=0.52.0 scikit-learn -c conda-forge
RUN conda run -n BODODEV pip install astor
#
# Creation of the initial .bashrc to be right up
#
RUN echo "source activate BODODEV" > ~/.bashrc
