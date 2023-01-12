conda create -y -n DEV mamba python=3.10 numpy=1.21 scipy pandas='1.4.*' boost-cpp=1.74 cmake h5py mpich mpi -c conda-forge
source activate DEV
mamba install -y "gcc_linux-64>=9" "gxx_linux-64>=9" -c conda-forge
mamba install -y numba=0.55.2 -c conda-forge
mamba install -y mpi4py cython -c conda-forge
mamba install -y -c conda-forge h5py 'hdf5=1.12.*=*mpich*' pyarrow=8.0.0 pymysql sqlalchemy snowflake-connector-python
mamba install -y -c conda-forge fsspec
mamba install -y -c conda-forge boto3 botocore
mamba install -y -c conda-forge s3fs
mamba install -y -c conda-forge ccache
mamba install -y -c conda-forge maven py4j openjdk=11
# Add for easier e2e testing debugging
mamba install -y -c conda-forge credstash pytest
# pip install deltalake
