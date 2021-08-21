FROM centos:centos7

RUN yum update -y
RUN yum install -y centos-release-scl
RUN yum install -y ncurses git vim wget
RUN yum -y groupinstall "Development Tools"
RUN yum install -y rh-python38
RUN yum install -y rh-python38-python-devel

RUN yum install -y mpich-devel
RUN yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
RUN yum install -y hdf5-mpich-devel

RUN yum install -y openssl-devel
RUN yum install -y boost-devel

SHELL ["usr/bin/scl", "enable", "rh-python38" ]

RUN pip install Cython numpy numba==0.53.0 wheel pandas==1.3.* pyarrow==5.0.0
RUN MPICC=/usr/lib64/mpich/bin/mpicc pip install mpi4py
RUN ln -s /opt/rh/rh-python38/root/usr/local/lib64/python3.8/site-packages/pyarrow/libarrow.so.300 /opt/rh/rh-python38/root/usr/local/lib64/python3.8/site-packages/pyarrow/libarrow.so
RUN ln -s /opt/rh/rh-python38/root/usr/local/lib64/python3.8/site-packages/pyarrow/libparquet.so.300 /opt/rh/rh-python38/root/usr/local/lib64/python3.8/site-packages/pyarrow/libparquet.so

# Once inside the image call
# scl enable rh-python38 bash
# Then git clone Bodo
# Then Bodo can be built with the command
# CONDA_PREFIX='' setup_centos7=true CFLAGS="-I/usr/include/mpich-x86_64/ -I/opt/rh/rh-python38/root/usr/local/lib64/python3.8/site-packages/pyarrow/include/ -L/usr/lib64/mpich/lib/ -L/opt/rh/rh-python38/root/usr/local/lib64/python3.8/site-packages/pyarrow/ -Wl,-rpath,/usr/lib64/mpich/lib/ -Wl,-rpath,/opt/rh/rh-python38/root/usr/local/lib64/python3.8/site-packages/pyarrow/" python3 setup.py bdist_wheel
