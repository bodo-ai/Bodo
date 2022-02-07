FROM ubuntu:latest AS env_setup
#
# The docker is compiled for example with the command
# docker build -t bodo_dev_hdfs .
#

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /root
#
# Ubuntu installs and Miniconda
#
RUN apt-get update && apt-get install -y wget bzip2 git vim make curl \
	&& wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
	&& chmod +x miniconda.sh\
	&& ./miniconda.sh -b
ENV PATH /root/miniconda3/bin/:${PATH}
#
# Creation of the BODODEV environment
#

RUN conda create -n BODODEV numpy scipy pandas='1.3.*' boost-cpp=1.74.0 cmake h5py mpich='3.4.*=h*' mpi 'gcc_linux-64>=9' 'gxx_linux-64>=9' -c conda-forge
RUN conda install -n BODODEV python numba=0.55.0 -c conda-forge
RUN conda install -n BODODEV hdf5='1.10=*mpich*' -c conda-forge
RUN conda install -n BODODEV python pyarrow=5.0.0 fsspec -c conda-forge
RUN conda install -n BODODEV pytest ipython
RUN conda install -n BODODEV sqlalchemy pymysql xlrd openpyxl mpi4py cython -c conda-forge
RUN conda install -n BODODEV scikit-learn='1.0.*' -c conda-forge
RUN conda install -n BODODEV openjdk=8
RUN conda install -n BODODEV hdfs3 -c conda-forge

# installing hadoop and libhdfs (JNI)
ARG hdfs=3.2.1
ENV HADOOP_HOME=/opt/hadoop-${hdfs}
ENV HADOOP_YARN_HOME=$HADOOP_HOME
ENV HADOOP_MAPRED_HOME=$HADOOP_HOME
RUN wget -q -O - "https://www.apache.org/dyn/mirrors/mirrors.cgi?action=download&filename=hadoop/common/hadoop-${hdfs}/hadoop-${hdfs}.tar.gz" | tar -xzf - -C /opt

# copy hadoop configuration for pseudo-distributed
COPY core-site.xml $HADOOP_HOME/etc/hadoop/
COPY hdfs-site.xml $HADOOP_HOME/etc/hadoop/

# install ssh server
RUN apt-get update
RUN apt-get install -y openssh-server

# configure ssh
RUN ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa
RUN cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
RUN chmod 0600 ~/.ssh/authorized_keys

#
# set up initial .bashrc for ssh
#

RUN echo "export PATH=$HOME/miniconda3/bin:$PATH" >> ~/.bashrc
RUN echo "source activate BODODEV" >> ~/.bashrc

# hadoop related enviroment variables
RUN echo "export HDFS_NAMENODE_USER=root" >> ~/.bashrc
RUN echo "export HDFS_DATANODE_USER=root" >> ~/.bashrc
RUN echo "export HDFS_SECONDARYNAMENODE_USER=root" >> ~/.bashrc
RUN echo "export JAVA_HOME=/root/miniconda3/envs/BODODEV" >> ${HADOOP_HOME}/etc/hadoop/hadoop-env.sh

# build bodo
RUN echo "cd /Bodo" >> ~/.bashrc
RUN echo "python setup.py develop" >> ~/.bashrc

# environment variables for arrow
RUN echo "export hdfs=3.2.1" >> ~/.bashrc
RUN echo "export HADOOP_HOME=/opt/hadoop-${hdfs}" >> ~/.bashrc
RUN echo "export ARROW_LIBHDFS_DIR=$HADOOP_HOME/lib/native" >> ~/.bashrc
RUN echo "export CLASSPATH=`$HADOOP_HOME/bin/hdfs classpath --glob`" >> ~/.bashrc
RUN echo "export HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_HOME/lib/native" >> ~/.bashrc
RUN echo "export HADOOP_OPTS='-Djava.library.path=$HADOOP_HOME/lib'" >> ~/.bashrc
