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
RUN conda install python=3.10 mamba -c conda-forge
RUN mamba install numpy=1.21 scipy pandas=${BODO_PD_VERSION:-'1.4.*'} boost-cpp=1.74.0 cmake h5py mpich mpi 'gcc_linux-64>=9' 'gxx_linux-64>=9' mpi4py -c conda-forge
RUN mamba install hdf5='1.12.*=*mpich*' -c conda-forge
RUN mamba install pyarrow=8.0.0 fsspec>=2021.09 s3fs boto3 botocore -c conda-forge
RUN mamba install pytest sqlalchemy pymysql xlrd openpyxl cython -c conda-forge
RUN mamba install numba=0.55.2 scikit-learn='1.0.*' -c conda-forge
RUN mamba install ccache -c conda-forge
RUN conda run pip install astor
#
# Creation of the initial .bashrc to be right up
#
RUN echo "source activate base" > ~/.bashrc
