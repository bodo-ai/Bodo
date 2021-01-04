.. _benchmarking_getting_started:

Benchmarking Tips
=================
Frequently during your time at Bodo you will be asked to benchmark a Bodo product's performance
and then compare to an alternative product. The most common benchmarking comparison will
compare Bodo to Spark.

Each benchmarking situation will have slightly different requirements, but in general whenever possible
you should try and benchmark Bodo performance using the Bodo Platform (either on ``development`` or ``staging``)
and benchmark Spark on `AWS EMR <https://aws.amazon.com/emr/>`_.

Benchmarking Bodo Products on the Platform
------------------------------------------
To ensure stability in the Bodo Platform, you should do all benchmarking of Bodo releases
on the `staging` environment and any benchmarking requiring development AMI on the `development` environment. 
We prefer all testing to occur in the platform because the autoshutdown feature can help reduce AWS charges 
if you forget to remove your machine instances, and because it provides the simplest Bodo configuration.

If you are using a released version of Bodo skip to Executing Your Benchmark.
Otherwise you will first need to create an AMI for your Bodo Products.

Registering a Development AMI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TODO

Executing your Benchmarking
~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can also create a Jupyter Notebook on the platform to run your benchmark. Note that using notebooks for benchmarking
is not ideal when the code is not stable. The reason being is that the notebook does not return logs until the end of the execution,
and often when a failure/hang happens, you don't see anything reported to you.  To create a notebook,
you first create a cluster with the nodes you want to run your code on and then
attach a Jupyter notebook. Refer to the user documentation on `clusters <https://docs.bodo.ai/latest/source/bodo_platform.html#creating-clusters>`_
and `notebooks <https://docs.bodo.ai/latest/source/bodo_platform.html#attaching-a-notebook-to-a-cluster>`_
to create your notebook instance. Finally, you can benchmark now run on your cluster using IPyParallel.
Refer to the `getting started tutorial <https://github.com/Bodo-inc/Bodo-tutorial/blob/master/bodo_getting_started.ipynb>`_
if you need guidance on how to run your notebook over the whole cluster.

If you cannot easily convert benchmark to an IPython Notebook you can execute your benchmark on your cluster 
by sshing onto one of the instances. To do this, provide an ssh public key in the advanced settings of your cluster creation.
Then, you can ssh onto one of the instances using the information in your Amazon account. **Warning**:
Make sure you turn off autoshutdown or set a value much larger than your expected benchmark execution time,
as the Bodo Platform has no way of detecting your ssh activity and may remove your cluster while your
benchmark is in progress.

**Important**: Unless you are purposely benchmarking compilation time, make sure to place
all timers inside Bodo functions so you only account for execution time.

Benchmarking Spark on EMR
-------------------------
When benchmarking Spark, we opt to use AWS EMR because it tends to provide the best Spark
performance. In general, configuring a Spark cluster can prove to be difficult, as there are
many configuration settings you must manually specify (for example memory allocation). EMR
seems to avoid many of these issues and allows you to run either Pyspark or Scala code 
via Jupyter Notebooks.

**Important**: When creating a Spark benchmark save any notebook you create to Github for
further use, ideally to the repository you are benchmarking.

Creating a Cluster on EMR
~~~~~~~~~~~~~~~~~~~~~~~~~
To create a cluster in EMR, follow the steps outlined in the `AWS guide <https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-gs-launch-sample-cluster.html>`_.
However, in the ``Quick Options`` page, select ``Go to advanced options``. You will mostly select the default options, but
there are a few execeptions.

#. In software configuration make sure at least, ``Hadoop``, ``Hive``, ``JupyterEnterpriseGateway``,
   and ``Spark`` are selected. If any other software is already selected, 
   **keep** that software selected.

   .. figure:: ../figs/EMRSoftware.png
    :alt: EMR Cluster Software

#. When you configure your cluster, make sure the cluster (master + core) has
   the same instance types are your comparison on the Bodo platform. For example, if you select
   3 ``m5.xlarge`` instances as your Bodo cluster, you should have 2 core and 1 master ``m5.xlarge``
   instance. Choose the ``On-demand`` (not ``Spot``) option show a disappearing instance won't interfere with your
   benchmark.

   .. figure:: ../figs/EMRClusterInstances.png
    :alt: EMR Cluster Instance Options

#. Optionally select an EC2 key pair in case you want to ssh onto your cluster. You do not
   have the option to create an ssh key from this screen, so restart creating your cluster
   if you want a key.

   **Note**: Do not try and run your benchmark by sshing onto your machine as you may encounter
   software issues. 

Attaching a Notebook on EMR
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Once you have created your cluster, you need to attach a notebook to run your benchmark.
This can be done by directly following the examples on the
`AWS documentation <https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-managed-notebooks-create.html>`_.
Please also consider the following details:


   - Our past attempts to link Git repositories have failed. We highly recommend
     you avoid this feature and just copy your notebook into the UI.

   - EMR **does not** have an autoshutdown feature. Please remember to delete
     your notebooks and clusters when you are finished using them.

Common Spark/Pyspark Pitfalls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Spark attempts to do a variety of optimizations under the hood, most notably
lazy execution, which can make it difficult when attempting to benchmark
particular operations. While this discussion is not complete (and if you 
encounter more information, please add it), these tips are meant to help
you more accurately benchmark Spark workloads.

#. When possible, benchmark an operation in its entirety to avoid the impact of lazy execution.
   If you try and time individual operations timings may be inaccurate because they are either lazily executed
   or contain the execution time of previous operations that are being executed lazily. 
   
#. Partition operations with cache and persist. If you need to split timing into individual operations,
   cache and persist can be used to force an operation to execute. However, this will prevent possible
   optimizations, so you should check that the overall time to execute the entire workload is the same or
   at least very close. For example, if you are using this to time IO read, this can be useful if the execution requires
   reading the whole file, but if you need only a few columns, Spark will no longer be able to avoid reading
   unnecessary columns, so the measurement may be inaccurate. (TODO: Include an example of cache and persist).

#. Only run the benchmark once. Spark's programming model is centered 
   around an expectation of failure. As a result, Spark will cache data
   in memory in case it needs to repeat the exact same operation.
   You should only time running the first trial of a notebook so you don't encounter
   these caching optimizations.
