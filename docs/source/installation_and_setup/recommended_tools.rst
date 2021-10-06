.. _recommended_tools:

These are our recommendations to tune your application environment and achieve the best possible performance with Bodo. 

Recommended MPI Settings
========================

`Intel-MPI library <https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/mpi-library.html#gs.cfkkrf>`_ is the preferred distribution for message passing interface (MPI) specification.

Intel-MPI provides different tuning collective algorithms.

Based on our internal benchmarking, we recommend setting these environment variables as follows::

    export I_MPI_ALLREDUCE=4
    export I_MPI_REDUCE=3

Use `-rr` option to place ranks in round-robin scheduling as follows::

    mpiexec -n <CORES> -f hostfile -rr python bodo_file.py

Recommended AWS Network Interface
=================================

`Elastic Fabric Adapter (EFA) <https://aws.amazon.com/hpc/efa/>`_ is a network interface for Amazon EC2 instances that has shown better inter-node communications at scale on AWS. 

To enable EFA with Intel-MPI on your cluster, follow instructions `here <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html>`_.

We recommend the following versions for the EFA installer and Intel-MPI::

    EFA_INSTALLER_VERSION: 1.13.0
    Intel-MPI: v3.1 (2021.3.1.315)

Other version combinations are not guaranteed to work as they have not been tested. 

For EFA installer versions >= 1.12.0, enabling fork is required by setting environment variable `FI_EFA_FORK_SAFE=1`

To confirm correct settings are enabled, run following `mpiexec` with `I_MPI_DEBUG=5` ::

    I_MPI_DEBUG=5 mpiexec -f hostfile -rr -n <CORES> python -u -c "from mpi4py import MPI"

Check that `libfabric provider` is `efa` and environment variables are set as shown below::

    [0] MPI startup(): Intel(R) MPI Library, Version 2021.3.1  Build 20210719 (id: 48425b416)
    [0] MPI startup(): Copyright (C) 2003-2021 Intel Corporation.  All rights reserved.
    [0] MPI startup(): library kind: release
    [0] MPI startup(): libfabric version: 1.13.0rc1-impi
    [0] MPI startup(): libfabric provider: efa
    ...
    [0] MPI startup(): I_MPI_ADJUST_ALLREDUCE=4
    [0] MPI startup(): I_MPI_ADJUST_REDUCE=3
    [0] MPI startup(): I_MPI_DEBUG=5

