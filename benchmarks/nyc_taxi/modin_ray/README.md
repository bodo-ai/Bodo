# Installation

Ensure that you have a python environment with Modin/Ray installed. For convenience, you can use `env.yml` which includes all of the packages needed to run this benchmark.

``` shell
conda env create -f env.yml
conda activate benchmark_modin
```

You will also need to set your AWS credentials which can be done by running `aws configure`.

You can refer to the [Modin documentation](https://modin.readthedocs.io/en/0.20.1/getting_started/quickstart.html) for more information about installation.

# Running the benchmark

This script will set up a Ray cluster, submit the benchmark jobs, and shutdown the cluster, which can take up to an 3 hours to run. 

```shell
./run.sh
```

# (Optional) Connect to dashboard or SSH into your Ray cluster

While running the script, you can view the status of the jobs in a separate terminal by running

``` shell
ray dashboard modin-cluster.yaml
```

to set up the dashboard in your browser or ssh into the head node by running

``` shell
ray attach modin-cluster.yaml
```

# Cleanup

Note that this script will automatically shut down the ray cluster after running the benchmarks. If there was an issue running the script or you need to shut down the cluster manually, you can run

``` shell
ray down modin-cluster.yaml -y
``` 

In some cases, if the head node is not responding, you may need to terminate the nodes using the EC2 console.