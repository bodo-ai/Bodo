.. _recommended_cluster_config:

Recommended Cluster Configuration
=================================

This section describes best practices for configuring compute clusters for Bodo applications.

Communication across cores is usually the largest overhead in parallel applications including Bodo. To minimize it:

- It is better to use fewer large nodes with high core count than many small nodes
  for a given total number of physical cores. This ensures that more cross core
  communication happens inside nodes. For example, a cluster with two ``c5n.18xlarge``
  AWS instances will generally perform better than a cluster with four ``c5n.9xlarge``
  instances, even though the two options have equivalent cost and compute power.

- Use node types that support high bandwidth networking. AWS instance types with ``n`` in their name,
  such as ``c5n.18xlarge``, ``m5n.24xlarge`` and ``r5n.24xlarge`` provide high bandwidth. On Azure, use virtual machines that support 
  `Accelerated Networking <https://docs.microsoft.com/en-us/azure/virtual-network/create-vm-accelerated-networking-cli>`_.

- Use instance types that support
  `RDMA <https://en.wikipedia.org/wiki/Remote_direct_memory_access>`_ networking such as 
  `Elastic Fabric Adapter (EFA) <https://aws.amazon.com/hpc/efa/>`_ (AWS) and 
  `Infiniband <https://docs.microsoft.com/en-us/azure/virtual-machines/workloads/hpc/enable-infiniband>`_ (Azure). 
  In our empirical testing, we found that EFA can significantly accelerate inter-node communication during expensive operations
  such as shuffle (which is used in join, groupby, sorting and others).
  
  - `List of AWS EC2 instance types that support EFA <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html#efa-instance-types>`_.
    For more information about EFA refer to :ref:`recommended_aws_nic`.
  
  - `RDMA capable Azure VM Sizes <https://docs.microsoft.com/en-us/azure/virtual-machines/sizes-hpc#rdma-capable-instances>`_.

- Ensure that the server nodes are located physically close to each other. On AWS this can be done by
  adding all instances to a `placement group <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/placement-groups.html#placement-groups-cluster>`_ 
  with the ``cluster`` strategy. Similarly on Azure, you can use
  `Proximity Placement Groups <https://docs.microsoft.com/en-us/azure/virtual-machines/co-location>`_.

For most appplications, we recommend using ``c5n.18xlarge`` instances on AWS for best performance.
For memory intensive use cases ``r5n.24xlarge`` instances are a good alternative.
Both instance types support 100 Gbps networking as well as EFA.


Other Best Practices
--------------------

- Ensure that the file descriptor limit (``ulimit -n``) is set to a large number like ``65000``.
  This is especially useful when using IPyParallel which opens direct connections between
  ``ipengine`` and ``ipcontroller`` processes.

- Avoid unnecessary threading inside the application since it can conflict with MPI parallelism. Set the following environment variables in your shell (e.g. in ``bashrc``) to avoid threading::

    export OPENBLAS_NUM_THREADS=1
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1

- Use `Intel MPI <https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html>`_ for best performance. See :ref:`recommended_mpi_settings` for more details.

