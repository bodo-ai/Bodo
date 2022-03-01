# Installing Bodo Community Edition {#install}


Bodo is available as a Python package on [pip](https://pypi.org/project/bodo/){target=blank},
and can be installed as follows:

```
pip install bodo
```


Bodo can also be installed as a using the `conda` command (see how to install [conda][conda] below). 
If you are installing bodo through conda, we recommend creating a `conda` environment and installing 
Bodo and its dependencies in it as shown below:

```console
conda create -n Bodo python=3.9 -c conda-forge
conda activate Bodo
conda install bodo -c bodo.ai -c conda-forge
```

Bodo uses [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface){target="blank"}
for parallelization, which is automatically installed as part of the
`conda` install command above. This command installs Bodo Community
Edition by default, which is free and works on up to 8 cores. For
information on Bodo Enterprise Edition and pricing, please [contact
us](https://bodo.ai/contact/).

!!! seealso "See Also"
    [Configuring Bodo Enterprise Edition][enterprise]

## How to Install Conda {#conda}

Install Conda using the instructions below.

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

### On Windows

```console
start /wait "" Miniconda3-latest-Windows-x86_64.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%UserProfile%\Miniconda3
```

Open the Anaconda Prompt to use Bodo (click Start, select Anaconda
Prompt). You may use other terminals if you have already added Anaconda
to your PATH.

## Optional Dependencies {#optionaldep}

Some Bodo functionality may require other dependencies, as summarized in
the table below. All optional dependencies except Hadoop can be
installed using the commands

```console
conda install gcsfs sqlalchemy snowflake-connector-python hdf5='1.10.*=*mpich*' openjdk -c conda-forge
```

and 

```console
pip install deltalake
```
<br/>
<center>

| Functionality            |   Dependency
|--------------------------|------------------------------------------------------------------------------------
|`pd.read_sql / df.to_sql` |`sqlalchemy`                                                                       
|`Snowflake I/O`           |`snowflake-connector-python`                                                       
|`GCS I/O`                 |`gcsfs`                                                                            
|`Delta Lake`              |`deltalake`                                                                        
|`HDFS or ADLS Gen2`       |[hadoop](http://hadoop.apache.org/docs/stable/){target="html"} (only the Hadoop client is needed) 
|`HDF5`                    |`hdf5 (MPI version)`                                                               

</center>

## Testing your Installation {#testinstall }

Once you have activated your `conda` environment and installed Bodo in
it, you can test it using the example program below. This program has
two functions:

-   The function `gen_data` creates a sample dataset with 20,000 rows
    and writes to a parquet file called `example1.pq`.
-   The function `test` reads `example1.pq` and performs multiple
    computations on it.

``` python3
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

Save this code in a file called `example.py`, and run it on a single
core as follows:

```console
python example.py
```

Alternatively, to run the code on four cores, you can use `mpiexec`:

```console
mpiexec -n 8 python example.py
```

!!! note
    You may need to delete `example1.pq` between consecutive runs.

!!! seealso "See Also"
    [Interactive Bodo Cluster Setup using IPyParallel][ipyparallelsetup]

[comment]: <> (Autorefs to [enterprise] and [ipyparallelsetup] will be populated once those sections are added)
[todo]: <> (Modify/remove comment above as [enterprise] and [ipyparallelsetup] sections are added)