# Recommended MPI Settings {#recommended_mpi_settings}

These are our recommendations to tune your application environment and
achieve the best possible performance with Bodo.

!!! info "Important"
    These recommendations are only applicable when you are running your workload
    on a cluster. You do not need to do any of this on your laptop.

[Intel-MPI
library](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/mpi-library.html#gs.cfkkrf){target="blank"}
is the preferred distribution for message passing interface (MPI)
specification.

Note that Bodo automatically installs `mpich`. Hence, after installing
Intel-MPI, remove [mpich] using this command:

```shell
conda remove -y --force mpich mpi
```

Intel-MPI provides different tuning collective algorithms.

Based on our internal benchmarking, we recommend setting these
environment variables as follows:

```shell
export I_MPI_ADJUST_ALLREDUCE=4
export I_MPI_ADJUST_REDUCE=3
```

## MPI Process Placement

Bodo assigns chunks of data and computation to MPI processes, also
called *ranks*. For example, for a dataframe with a billion rows on a
1000-core cluster, the first one million rows are assigned to rank 0,
the second one million rows to rank 1, and so on. MPI placement
indicates how these ranks are assigned to physical cores across the
cluster, and can significantly impact performance depending on hardware
configuration and application behavior. We recommend trying *block
mapping* and *round-robin mapping* options below for your application to
achieve the best performance.

!!! info "Important"
    These options are only supported in [SPMD launch mode][spmd].

### Block Mapping

In block mapping, cores of each node in the `hostfile` are filled with
ranks before moving on to the next node. For example, for a cluster with
50-core nodes, the first 50 ranks will be on node 0, the second 50 ranks
on node 1 and so on. This mapping has the advantage of fast
communication between neighboring ranks on the same node.

We provide instructions on setting block placement for
MPICH and Intel MPI below. The following assumes the hostfile only
contains a list of hosts (e.g. it does not specify number of processes
per host) and the number of cores on each host is the same.

**Block Mapping with MPICH and Intel MPI**:

```shell
mpiexec -n <N> -f <hostfile> -ppn <P> python bodo_file.py
```
where `N` is the number of MPI processes, `hostfile` contains the list
of hosts, and `P` the number of processes (cores) per node.


### Round-Robin Mapping

In round-robin mapping, MPI assigns one rank per node in hostfile and
starts over when it reaches end of the host list. For example, for a
cluster with 50-core nodes, rank 0 is assigned to node 0, rank 1 is
assigned to node 1 and so on. Rank 50 is assigned to node 0, 51 to node
1, and so on. This mapping has the advantage of avoiding communication
hotspots in the network and tends to make large shuffles faster.

We provide instructions on setting round-robin placement for
MPICH and Intel MPI below. The following assumes the hostfile only
contains a list of hosts (e.g. it does not specify number of processes
per host) and the number of cores on each host is the same.


**Round-Robin with MPICH**:

```shell
mpiexec -n <N> -f <hostfile> python bodo_file.py
```
**Round-Robin with Intel MPI**:

```shell
mpiexec -n <N> -f <hostfile> -rr python bodo_file.py
```

### Useful References

- More information on controlling process placement with Intel MPI can be found
[here](https://www.intel.com/content/www/us/en/developer/articles/technical/controlling-process-placement-with-the-intel-mpi-library.html){target="blank"}.

- See how to use the Hydra Process Manager for MPICH [here](https://wiki.mpich.org/mpich/index.php/Using_the_Hydra_Process_Manager){target="blank"}.

## Recommended AWS Network Interface {#recommended_aws_nic}

[Elastic Fabric Adapter (EFA)](https://aws.amazon.com/hpc/efa/) is a
network interface for Amazon EC2 instances that has shown better
inter-node communications at scale on AWS.

To enable EFA with Intel-MPI on your cluster, follow instructions
[here](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html).

Some points to note in addition to the referenced instructions:

1.  All instances must be in the same subnet. For more information, see
    the "EFA Limitations" section
    [here](https://www.hpcworkshops.com/07-efa/00-efa-basics.html).

2.  All instances must be part of a security group that allows all
    inbound and outbound traffic to and from the security group itself.
    Follow these
    [instructions](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html#efa-start-security)
    to set up the security group correctly.

3.  For use with Intel-MPI, a minimal installation of the EFA drivers is
    sufficient and recommended:

    ```shell
    sudo ./efa_installer.sh -y --minimal
    ```

    Depending on where the drivers were downloaded from, you might need
    to include a `--no-verify` flag:

    ```shell
    sudo ./efa_installer.sh -y --minimal --no-verify
    ```

We recommend the following versions for the EFA installer and Intel-MPI:

```console
EFA_INSTALLER_VERSION: 1.13.0
Intel-MPI: v3.1 (2021.3.1.315)
```

Other version combinations are not guaranteed to work as they have not
been tested.

For EFA installer versions *>= 1.12.0*, enabling fork is required by
setting environment variable `FI_EFA_FORK_SAFE=1`.

To confirm correct settings are enabled, run following
`mpiexec` with `I_MPI_DEBUG=5` :

```shell
I_MPI_DEBUG=5 mpiexec -f hostfile -rr -n <CORES> python -u -c "from mpi4py import MPI"
```

Check that `libfabric provider` is `efa` and
environment variables are set as shown below:

```console
[0] MPI startup(): Intel(R) MPI Library, Version 2021.3.1  Build 20210719 (id: 48425b416)
[0] MPI startup(): Copyright (C) 2003-2021 Intel Corporation.  All rights reserved.
[0] MPI startup(): library kind: release
[0] MPI startup(): libfabric version: 1.13.0rc1-impi
[0] MPI startup(): libfabric provider: efa
...
[0] MPI startup(): I_MPI_ADJUST_ALLREDUCE=4
[0] MPI startup(): I_MPI_ADJUST_REDUCE=3
[0] MPI startup(): I_MPI_DEBUG=5
```

## Automatic Worker Number Detection

Bodo can automatically detect the number of workers to spawn based on MPI_UNIVERSE_SIZE.
The following sections demonstrate how to configure Intel MPI and MPICH to set
MPI_UNIVERSE_SIZE.

### Intel MPI

For Intel MPI to correctly set MPI_UNIVERSE_SIZE, you need to create
`~/.mpiexec.conf` if it doesn't exist and add `--usize SYSTEM`.
A valid hostfile is also required.

### MPICH

For MPICH to correctly set MPI_UNIVERSE_SIZE, you need to pass the
`-f <path_to_hostfile>` and `-usize SYSTEM` flags to mpiexec. Mpich doesn't
have a way to set either of these flags in a configuration file
so the flags must be passed on the command line every time.
We recommend creating an alias for `mpiexec` with these flags.
The hostfile should specify how many processes to run on each host. For example:

```shell
host1:4
host2:4
```

This specifies that 4 processes should run on each of `host1` and `host2`.
Generally, the number of processes should be equal to the number of
physical cores on each host.
