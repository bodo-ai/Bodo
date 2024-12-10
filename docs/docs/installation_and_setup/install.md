______________________________________________________________________

hide:

- tags

tags:

- install

______________________________________________________________________

# Installing Bodo Engine {#install}

Bodo compute engine can be installed using either `pip` or `conda` (see how to install [conda][conda] below).
To install Bodo and its dependencies with `pip`, use the following command:

```console
pip install bodo
```

For production environments, we recommend creating a `conda` environment and installing
Bodo and its dependencies in it as shown below:

```console
conda create -n Bodo python=3.12 -c conda-forge
conda activate Bodo
conda install bodo -c bodo.ai -c conda-forge
```

Bodo uses [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface){target="blank"}
for parallelization, which is automatically installed as part of the
`pip` and `conda` install commands above.

## How to Install Conda {#conda}

Install conda using the instructions below.

### On Linux

```shell
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b
export PATH=$HOME/miniconda3/bin:$PATH
```

### On MacOS

```shell
curl https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -L -o miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b
export PATH=$HOME/miniconda3/bin:$PATH
```

## Optional Dependencies {#optionaldep}

Some Bodo functionality may require other dependencies, as summarized in
the table below.

All optional dependencies except Hadoop, HDF5, and OpenJDK can be installed through pip using the command:

```console
pip install gcsfs sqlalchemy snowflake-connector-python deltalake
```

All optional dependencies except Hadoop can be
installed through conda using the command:

```console
conda install gcsfs sqlalchemy snowflake-connector-python hdf5='1.14.*=*mpich*' openjdk=11 deltalake -c conda-forge
```

<br/>
<center>

| Functionality | Dependency
|--------------------------|------------------------------------------------------------------------------------
|`pd.read_sql / df.to_sql` |`sqlalchemy`\
|`Snowflake I/O` |`snowflake-connector-python`\
|`GCS I/O` |`gcsfs`\
|`Delta Lake` |`deltalake`\
|`HDFS or ADLS Gen2` |[hadoop](http://hadoop.apache.org/docs/stable/){target="html"} (only the Hadoop client is needed)
|`HDF5` |`hdf5 (MPI version)`

</center>

## Testing your Installation {#testinstall }

Once you have installed Bodo with pip or activated your `conda` environment and installed Bodo in
it, you can test it using the example program below. This program has
two functions:

- The function `gen_data` creates a sample dataset with 20,000 rows
  and writes to a parquet file called `example1.pq`.
- The function `test` reads `example1.pq` and performs multiple
  computations on it.

```python3
import bodo
import pandas as pd
import numpy as np
import time

@bodo.jit
def gen_data():
    NUM_GROUPS = 30
    NUM_ROWS = 20_000_000
    df = pd.DataFrame({
        "A": np.arange(NUM_ROWS) % NUM_GROUPS,
        "B": np.arange(NUM_ROWS)
    })
    df.to_parquet("example1.pq")

@bodo.jit
def test():
    df = pd.read_parquet("example1.pq")
    t0 = time.time()
    df2 = df.groupby("A")["B"].agg(
        (lambda a: (a==1).sum(), lambda a: (a==2).sum(), lambda a: (a==3).sum())
    )
    m = df2.mean()
    print("Result:", m, "\nCompute time:", time.time() - t0, "secs")

gen_data()
test()
```

Save this code in a file called `example.py`, and run it on all cores
core as follows:

```console
python example.py
```

Alternatively, to run it on a single core:

```console
BODO_NUM_WORKERS=1 python example.py
```

!!! note
You may need to delete `example1.pq` between consecutive runs.

## Enabling parallelism in Clusters

Bodo relies on MPI for parallel compute. MPI can be configured on clusters
easily. The cluster nodes need to have passwordless SSH enabled between them,
and there should be a host file listing their addresses (see an example tutorial
[here](https://mpitutorial.com/tutorials/running-an-mpi-cluster-within-a-lan/){target="blank"}).
MPI usually needs to be configured to launch one process per physical core for
best performance. This avoids potential resource contention between processes
due to the high efficiency of MPI. For example, a cluster of four nodes, each
with 16 physical cores, can use up to 64 MPI processes:

```shell
BODO_NUM_WORKERS=64 python example.py
```

For cloud instances, one physical core typically corresponds to two vCPUs.
For example, an instance with 32 vCPUs has 16 physical cores.

## Setting up passwordless SSH on your multi-node cluster {#passwordless_ssh}

Using MPI on a multi-node cluster requires setting up passwordless SSH
between the hosts. There are multiple ways to do this. Here is one way:

1. Generate an SSH key pair using a tool like `ssh-keygen`, for
   instance:

   ```shell
   ssh-keygen -b 2048 -f cluster_ssh_key -N ""
   ```

1. Copy over the generated private key (`cluster_ssh_key`) and public key (`cluster_ssh_key.pub`) to all the hosts and
   store them in `~/.ssh/id_rsa` and `~/.ssh/id_rsa.pub` respectively.

1. Add the public key to `~/.ssh/authorized_keys` on all hosts.

1. To disable host key checking, add the following to `~/.ssh/config`
   on each host:

   ```shell
   Host *
       StrictHostKeyChecking no
   ```
