# NOTE: It's safer to use a specific version of the manylinux2014_x86_64
# image, to maintain the same hash for libmpi.so
FROM quay.io/pypa/manylinux2014_x86_64:2021-12-12-e5100b5

# We don't need to install dev tools or Python.
# manylinux2014 image already has dev tools installed and
# multiple versions of Python

# Set up the Python environment to build Bodo pip packages
ADD setup-py-env.sh /
RUN chmod +x /setup-py-env.sh ; /setup-py-env.sh

# Build mpich ourselves to have less dependencies.
# yum install -y mpich-devel installs compilers that we don't need.
# We also don't need Fortran or C++ support in MPI which just
# complicates things and increases the size of the pip packages
RUN curl https://www.mpich.org/static/downloads/3.3.2/mpich-3.3.2.tar.gz --output mpich-3.3.2.tar.gz
RUN tar xf mpich-3.3.2.tar.gz
RUN mkdir /mpich ; cd mpich-3.3.2 ; ./configure --prefix=/mpich --enable-fast --enable-shared --disable-fortran --disable-opencl --disable-cxx --with-device=ch3
RUN cd mpich-3.3.2 ; make ; make install

# Install remaining build dependencies for Bodo
RUN yum install -y openssl-devel
RUN yum install -y boost-devel
