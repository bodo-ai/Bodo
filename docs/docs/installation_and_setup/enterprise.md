# Configuring Bodo Enterprise Edition {#enterprise}

Bodo Enterprise Edition allows unrestricted use of Bodo on any number of
cores. Ensure you have [installed Bodo][install]
before configuring Bodo Enterprise Edition.

## License Key {#licensekey}

Bodo Enterprise Edition requires a license key to run. The key can be
provided in two ways:

- Through the environment variable `BODO_LICENSE`
- A file called `bodo.lic` in the current working directory

In both cases, the file or environment variable must contain the key
exactly as provided.

If Bodo cannot find the license, you will only be able to run
Bodo on up to 8 cores. If you try to run Bodo on more than 8 cores and
if Bodo cannot find the license (the environment variable does not exist
or is empty, and no license file is found), it will exit with the
`Bodo license not found` error.

[todo]: <> (add an example to show Bodo license not found error)

If the contents of the license key are invalid, Bodo will exit with the `Invalid license`
error. This typically means that the key is missing data or contains
extraneous characters.
 
Please make sure the license file has not been
modified, or that the environment variable contains the key verbatim.
Note that some shells might append extra characters when displaying the
file contents. A good way to export the key is this:

```shell
export BODO_LICENSE=`cat bodo.lic`
```

## Automated `BODO_LICENSE` environment variable Setup

You can automate setting of the `BODO_LICENSE` environment variable in
your `~/.bashrc` script (or the `~/.zshrc` script for macOS) using:

```shell
echo 'export BODO_LICENSE="<COPY_PASTE_THE_LICENSE_HERE>"' >> ~/.bashrc
```

For more fine-grained control and usage with the Bodo `conda`
environment as created when [installing bodo][install],
we recommend the following steps to automate setting the `BODO_LICENSE`
environment variable (very similar to
[these](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#macos-and-linux){target="html"}
steps):

1.  Ensure that you are in the correct conda environment.

2.  Navigate to the `$CONDA_PREFIX` directory and create some additional
    conda environment activation and deactivation steps:
    
    ```shell
    cd $CONDA_PREFIX
    mkdir -p ./etc/conda/activate.d
    mkdir -p ./etc/conda/deactivate.d
    touch ./etc/conda/activate.d/env_vars.sh
    touch ./etc/conda/deactivate.d/env_vars.sh
    ```
    
3.  Edit `./etc/conda/activate.d/env_vars.sh` as follows:

    ```shell
    #!/bin/sh 
      
    export BODO_LICENSE="<COPY_PASTE_THE_LICENSE_HERE>"
    ```

4.  Similarly, edit `./etc/conda/deactivate.d/env_vars.sh` as follows:
 
    ```shell
    #!/bin/sh

    unset BODO_LICENSE
    ```
    
5.  Deactivate (`conda deactivate`) and reactivate the `Bodo` conda
    environment (`conda activate Bodo`) to ensure that the environment
    variable `BODO_LICENSE` is automatically added when the environment
    is activated.

## Using MPI in Clusters with Bodo Enterprise Edition {#mpienterpriseclusters}

MPI can be configured on clusters easily. The cluster nodes need to have
passwordless SSH enabled between them, and there should be a host file
listing their addresses (see an example tutorial
[here](https://mpitutorial.com/tutorials/running-an-mpi-cluster-within-a-lan/){target="blank"}).
MPI usually needs to be configured to launch one
process per physical core for best performance. This avoids potential resource contention
between processes due to the high efficiency of MPI. For example, a
cluster of four nodes, each with 16 physical cores, would use 64 MPI
processes:

```shell
mpiexec -n 64 python example.py
```

For cloud instances, one physical core usually corresponds to two vCPUs.
For example, an instance with 32 vCPUs has 16 physical cores.

!!! seealso "See Also"
    [Interactive Bodo Cluster Setup using IPyParallel][ipyparallelsetup]

## Setting up passwordless SSH on your multi-node cluster {#passwordless_ssh}

Using MPI on a multi-node cluster requires setting up passwordless SSH
between the hosts. There are multiple ways to do this. Here is one way:

1.  Generate an SSH key pair using a tool like `ssh-keygen`, for
    instance:
    
    ```shell
    ssh-keygen -b 2048 -f cluster_ssh_key -N ""
    ```
    
2.  Copy over the generated private key (`cluster_ssh_key`) and public key (`cluster_ssh_key.pub`) to all the hosts and 
    store them in `~/.ssh/id_rsa` and `~/.ssh/id_rsa.pub` respectively.

3.  Add the public key to `~/.ssh/authorized_keys` on all hosts.

4.  To disable host key checking, add the following to `~/.ssh/config`
    on each host:

    ```shell
    Host *
        StrictHostKeyChecking no
    ```
    
