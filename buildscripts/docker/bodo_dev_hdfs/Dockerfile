# Make sure to build the docker image from the top folder:
# docker build -t bodo_dev_hdfs -f docker/bodo_dev_hdfs/Dockerfile .
FROM --platform=linux/amd64 ghcr.io/prefix-dev/pixi:latest AS env_setup
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /root

# Ubuntu Install. Best to do in 1 line and remove cache at end
# https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#run
RUN apt-get update && apt-get install -y wget bzip2 git vim && rm -rf /var/lib/apt/lists/*

# Setup Dev Environment
RUN pixi install
RUN pixi run build

# Installing hadoop and libhdfs (JNI)
ARG hdfs=3.3.2
ENV HADOOP_HOME=/opt/hadoop-${hdfs}
ENV HADOOP_YARN_HOME=$HADOOP_HOME
ENV HADOOP_MAPRED_HOME=$HADOOP_HOME
RUN wget -q -O - "https://www.apache.org/dyn/mirrors/mirrors.cgi?action=download&filename=hadoop/common/hadoop-${hdfs}/hadoop-${hdfs}.tar.gz" | tar -xzf - -C /opt

# copy hadoop configuration for pseudo-distributed
COPY docker/bodo_dev_hdfs/core-site.xml $HADOOP_HOME/etc/hadoop/
COPY docker/bodo_dev_hdfs/hdfs-site.xml $HADOOP_HOME/etc/hadoop/
COPY docker/bodo_dev_hdfs/reset_hdfs.sh /root

# configure ssh
RUN ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa
RUN cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
RUN chmod 0600 ~/.ssh/authorized_keys

# Set up initial .bashrc for ssh
RUN apt-get update && apt-get install -y openssh-server
RUN echo "pixi shell" >> ~/.bashrc

# hadoop related enviroment variables
RUN echo "export HDFS_NAMENODE_USER=root" >> ~/.bashrc
RUN echo "export HDFS_DATANODE_USER=root" >> ~/.bashrc
RUN echo "export HDFS_SECONDARYNAMENODE_USER=root" >> ~/.bashrc
RUN echo "export JAVA_HOME=$JAVA_HOME" >> ${HADOOP_HOME}/etc/hadoop/hadoop-env.sh

# build bodo
RUN echo "cd /Bodo" >> ~/.bashrc
RUN echo "pixi run build-bodo" >> ~/.bashrc

# environment variables for arrow
RUN echo "export hdfs=3.3.2" >> ~/.bashrc
RUN echo "export HADOOP_HOME=/opt/hadoop-${hdfs}" >> ~/.bashrc
RUN echo "export ARROW_LIBHDFS_DIR=$HADOOP_HOME/lib/native" >> ~/.bashrc
RUN echo "export CLASSPATH=`$HADOOP_HOME/bin/hdfs classpath --glob`" >> ~/.bashrc
RUN echo "export HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_HOME/lib/native" >> ~/.bashrc
RUN echo "export HADOOP_OPTS='-Djava.library.path=$HADOOP_HOME/lib'" >> ~/.bashrc
