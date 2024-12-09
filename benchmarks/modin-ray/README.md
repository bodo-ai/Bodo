# Installation

Ensure that you have a python environment with modin/ray installed. For conveinience, you can use `env.yml` which includes all of the packages needed for running this benchmark.  

``` shell
conda env create -f env.yml
```

You will also need to set your aws credentials which can be done by running `aws configure`

Refer to the [Modin documentation](https://modin.readthedocs.io/en/0.20.1/getting_started/quickstart.html) for more information about installation.

# Running the benchmark

This script will run the code to setup the ray cluster, submit a small job to make sure all the nodes are up and running, and submit the full benchmark. This script can take up to an hour to run.

```shell
./run.sh
```

# (Optional) Connect to dashboard or SSH into your Ray cluster

While running the script you can view the status of the jobs in a separate terminal by running

``` shell
ray dashboard modin-cluster.yaml
```

to set up the dashboard in your browser, or ssh into the head node by running:

``` shell
ray attach modin-cluster.yaml
```

# Cleanup

Note that this script will automatically shut down the ray cluster after running the benchmarks. If there was an issue running the script or you need to shut down the cluster manually, you can run

``` shell
ray down modin-cluster.yaml -y
``` 

In some cases, if the head node is not responding, you may need to terminate the nodes in your AWS EC2 console.