# NOTE: It's safer to use a specific version of the manylinux2014_x86_64
# image, to maintain the same hash for libmpi.so
FROM quay.io/pypa/manylinux2014_x86_64:2021-12-12-e5100b5

# We don't need to install dev tools or Python.
# manylinux2014 image already has dev tools installed and
# multiple versions of Python

# Set up the Python environment to build Bodo pip packages
ADD setup_py_env.sh /
RUN chmod +x /setup_py_env.sh ; /setup_py_env.sh
ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-11.0.15.0.9-2.el7_9.x86_64
