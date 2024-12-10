# Connecting to a Cluster {#connecting_to_a_cluster}

We recommend interacting with clusters primarily through Jupyter
Notebooks and Jobs. However, it may be necessary to connect directly to
a cluster in some cases. In that case, you can connect through a notebook
terminal.

### Connecting with a Notebook Terminal

![Notebook-Terminal](../../platform2-gifs/notebook-terminal.gif#center)

If your cluster has more than one node , you can connect to any of the cluster nodes by running the following command in the terminal:

```shell
ssh <NODE-IP>
```

![Connect-Cluster](../../platform2-gifs/connect-to-cluster.gif#center)

Through this terminal, you can access the `/bodofs` folder, which
is shared by all the instances in the cluster and the Notebook instance.
[Verify your connection][verify_your_connection] to interact directly with your cluster.

### Verify your Connection {#verify_your_connection}

Once you have connected to a node in your cluster, you should verify
that you can run operations across all the instances in the cluster.

1. Verify the path to the hostfile for your cluster. You can find it by
   running:

   ```shell
   ls -la /home/bodo/hostfile
   ```

1. Check that you can run a command across you cluster. To do this,
   run:

   ```shell
   mpiexec -n <TOTAL_CORE_COUNT> -f /home/bodo/hostfile hostname
   ```

   This will print one line per each core in the cluster, with one
   unique hostname per cluster node.

   Your cluster's `TOTAL_CORE_COUNT` is usually half the
   number of vCPUs on each instance times the number of instances in
   your cluster. For example, if you have a 4 instance cluster of
   c5.4xlarge, then your `TOTAL_CORE_COUNT` is 32.

1. Verify that you can run a python command across your cluster. For
   example, run:

   ```
    mpiexec -n <TOTAL_CORE_COUNT> -f /home/bodo/hostfile python --version
   ```

If all commands succeed, you should be able to execute workloads across
your cluster. You can place scripts and data that are shared
across cluster nodes in `/bodofs`.
