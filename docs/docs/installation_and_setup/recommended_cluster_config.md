# Recommended Cluster Configuration {#recommended_cluster_config}

This page describes best practices for configuring compute clusters
for Bodo applications.

## Minimizing Communication Overheads

Communication across cores is usually the largest overhead in parallel
applications including Bodo. To minimize it:

-   ***For a given number of physical cores, use fewer large nodes with high core count rather than many
    small nodes with a low core count.***
    
    This ensures that more cross core communication happens inside nodes. For
    example, a cluster with two `c5n.18xlarge` AWS instances will
    generally perform better than a cluster with four `c5n.9xlarge`
    instances, even though the two options have equivalent cost and
    compute power.
    
-   ***Use node types that support high bandwidth networking.***
 
    AWS instance types with `n` in their name, such as `c5n.18xlarge`, `m5n.24xlarge`
    and `r5n.24xlarge` provide high bandwidth. On Azure, use virtual
    machines that support 
    [Accelerated Networking](https://docs.microsoft.com/en-us/azure/virtual-network/create-vm-accelerated-networking-cli){target="blank"}.

-   ***Use instance types that support
    [RDMA](https://en.wikipedia.org/wiki/Remote_direct_memory_access){target="blank"}
    networking.*** 
    
    Examples of such instance types are [Elastic Fabric Adapter (EFA)](https://aws.amazon.com/hpc/efa/){target="blank"} (AWS) and
    [Infiniband](https://docs.microsoft.com/en-us/azure/virtual-machines/workloads/hpc/enable-infiniband){target="blank"}
    (Azure). In our empirical testing, we found that EFA can
    significantly accelerate inter-node communication during expensive
    operations such as shuffle (which is used in join, groupby, sorting
    and others).
    
    -   [List of AWS EC2 instance types that support EFA](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html#efa-instance-types){target="blank"}.
        For more information about EFA refer to the section on
        [Recommended AWS Network Interface ][recommended_aws_nic].
        
    -   [RDMA capable Azure VM Sizes](https://docs.microsoft.com/en-us/azure/virtual-machines/sizes-hpc#rdma-capable-instances){target="blank"}.

-   ***Ensure that the server nodes are located physically close to each
    other.***
     
    On AWS this can be done by adding all instances to a [placement group](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/placement-groups.html#placement-groups-cluster){target="blank"}
    with the `cluster` strategy. Similarly on Azure, you can use [Proximity Placement Groups](https://docs.microsoft.com/en-us/azure/virtual-machines/co-location){target="blank"}.

For most applications, we recommend using `c5n.18xlarge` instances on
AWS for best performance. For memory intensive use cases `r5n.24xlarge`
instances are a good alternative. Both instance types support 100 Gbps
networking as well as EFA.

## Other Best Practices

-   ***Ensure that the file descriptor limit (`ulimit -n`) is set to a
    large number like `65000`.***

- Launch scripts with mpiexec on clusters e.g.

```shell
mpiexec -f <hostfile> -usize SYSTEM python file.py
```

-   ***Avoid unnecessary threading inside the application since it can
    conflict with MPI parallelism.*** 
    
    You can set the following environment variables in your shell (e.g. in `bashrc`)
    to avoid threading:

    ```shell
    export OPENBLAS_NUM_THREADS=1
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    ```
    
-   ***Use [Intel MPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html)
    for best performance.***
    See our [recommended MPI settings][recommended_mpi_settings] for more
    details.
