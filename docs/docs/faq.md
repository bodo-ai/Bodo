# Frequently Asked Questions (FAQ)


### How well does Bodo scale?

Bodo scales linearly across cores and nodes by using HPC technologies such as MPI in the backend.
Bodo can parallelize applications from local laptops all the way to large clusters (5000+ cores) effectively.

### What data formats does Bodo support?

Bodo supports various data storage formats such as Iceberg, Snowflake, Parquet, CSV, and JSON natively.
See the [File I/O][file_io] docs for more details.


### What are the hardware requirements to run Bodo?

Bodo is fully portable across various CPU architectures and environments.
We currently distribute packages for Mac and Linux with x86 and ARM CPU architectures (Windows support is in progress).
Bodo can run on any on-premises or cloud environment.


### How is Bodo different from Dask, Ray, or Spark?

Unlike traditional distributed computing frameworks, Bodo:

- Seamlessly supports native Python APIs like Pandas and NumPy using a compiler approach.
- Eliminates runtime overheads common in driver-executor models by leveraging Message Passing Interface (MPI) technology for true parallel execution.


### Will Bodo “just work” on my existing code?

Pretty close. You only need to annotate your Python compute functions with `@bodo.jit`, make sure they are jittable,
and Bodo will handle the parallelization and optimization automatically.
The vast majority of your application logic will remain unchanged, as long as it uses common libraries like Pandas, NumPy and Scikit-Learn.


### What types of workloads are ideal for Bodo?

Bodo excels at accelerating large-scale data processing, AI/ML workflows, and many other workloads requiring significant computation.


### Can I use Bodo in Jupyter Notebooks or other IDEs?

Bodo works in Jupyter, VS Code and other IDEs that support Python.


### Does Bodo support cloud environments?

Bodo can run on any cloud environment using virtual machines or Kubernetes clusters.
Creating multi-node clusters just requires network configuration for cross-node communication.
Bodo cloud service simplifies managing compute clusters and jobs on AWS and Azure.


### How can I get help if I encounter an issue?

For support, you can join the [Bodo Community Slack](https://bodocommunity.slack.com/join/shared_invite/zt-qwdc8fad-6rZ8a1RmkkJ6eOX1X__knA#/shared-invite/email),
where users and contributors actively discuss and resolve issues.
You can also refer to the documentation or raise an issue on GitHub if you’ve encountered a bug or need technical guidance.

### Are there any usage limits or restrictions?

No, there are no usage limits. Bodo is open source and distributed under the Apache-2.0 License. You can use Bodo in personal, academic, and commercial projects.

### How does Bodo handle security and privacy?

Since Bodo runs in your environment, you have complete control over the data and compute resources, ensuring that sensitive information remains secure.

### How does Bodo handle fault tolerance and failures? 

The Bodo compiler creates binaries that run efficiently on bare metal, eliminating most software failures
of other framework (e.g., scheduler errors, JVM errors, ...).
To handle rare case of hardware failures, we recommend simply configuring job restarts
in your environment.
Bodo's high performance ensures fast restart and job completion even with failures.

### Does Bodo have a job scheduler?

Bodo provides parallel compute that can work with any job scheduler.
Bodo does not bundle its own job scheduler.

### Am I supposed to have MPI nodes set up already?

MPI setup is required for distributed execution, which involves passwordless SSH access across nodes. See [cluster installation docs][cluster_setup].

### How do I know if my function is actually being parallelized?

You can see distributed diagnostics of the compiled function to understand parallelism decisions by the compiler. See [parallism docs][basics].
In addition, the Bodo compiler throws a warning if it didn't find any parallelism opportunities in a function.
You can also see your local machine's CPU usage using tools like `top` and `htop`.


### How do I handle unsupported operations or libraries?

You can use unsupported operations and libraries outside JIT functions, or use
the `@bodo.wrap_python` decorator to use any regular Python function inside JIT functions.
See [wrap_python docs][objmode].


### Can Bodo read from databases like Postgres, Snowflake, or BigQuery?

Bodo can read from all databases that are readable in Python.
The Snowflake connector is especially optimized for very high performance.

### Does Bodo work inside Databricks environment?

Bodo currently does not work inside Databricks due to networking restrictions.
We plan to investigate and provide a solution.


### Can I create custom user-defined functions (UDFs) with Bodo?

Yes, Bodo is particularly good at accelerating UDFs in Pandas APIs such as `DataFrame.apply` and `Series.map`.

### What is the difference between the open source compute engine and the Bodo Cloud Platform?

The Bodo Cloud Platform simplifies managing compute clusters, notebooks and jobs that use the Bodo engine.
Bodo Cloud Platform currently supports AWS and Azure.

### Does Bodo have a SQL interface?

Yes, BodoSQL is a high performance SQL engine that provides vectorized streaming execution and
support distributed clusters using MPI.
BodoSQL compiles SQL queries into optimized binaries, which is particularly
good for large batch jobs.


### When is Bodo *Not* appropriate? What is it *not* designed for?
Bodo is designed for large-scale data processing and may not be appropriate for other use cases
such as accelerating Python web frameworks (e.g. Django) and other non-compute Python applications.
