# Frequently Asked Questions (FAQ)


### How well does Bodo DataFrames scale?

Bodo DataFrames scales linearly across cores and nodes by using HPC technologies such as MPI in the backend.
It can parallelize applications from local laptops all the way to large clusters (5000+ cores) effectively.

### What data formats does Bodo DataFrames support?

Bodo DataFrames supports various data storage formats such as Iceberg, Snowflake, Parquet, CSV, and JSON natively.
See the [File I/O][file_io] docs for more details.


### What are the hardware requirements to run Bodo DataFrames?

Bodo DataFrames is fully portable across various CPU architectures and environments.
We currently distribute packages for Mac with x86 and ARM CPU architectures and x86 Linux and Windows.
Bodo DataFrames can run on any on-premises or cloud environment.


### How is Bodo DataFrames different from Dask, Ray, or Spark?

Unlike traditional distributed computing frameworks, Bodo DataFrames:

- Automatically scales and accelerates Pandas workloads with a single line of code change.
- Eliminates runtime overheads common in driver-executor models by leveraging Message Passing Interface (MPI) technology for true parallel execution.

### Will Bodo DataFrames “just work” on my existing code?

Yes, Bodo DataFrames is designed as a drop-in replacement for Pandas,
simply replace the import `import pandas as pd` with `import bodo.pandas as pd` to transparently accelerate and scale Pandas workloads.
For cases where Bodo DataFrames does not support a specific Pandas operation,
it will warn the user before collecting the data and continuing execution in vanilla Pandas.

### What types of workloads are ideal for Bodo DataFrames?

Bodo DataFrames excels at accelerating large-scale data processing, AI/ML workflows, and many other workloads requiring significant computation.


### Can I use Bodo DataFrames in Jupyter Notebooks or other IDEs?

Bodo DataFrames works in Jupyter, VS Code and other IDEs that support Python.


### Does Bodo DataFrames support cloud environments?

Bodo DataFrames can run on any cloud environment using virtual machines or Kubernetes clusters.
Creating multi-node clusters just requires network configuration for cross-node communication.
Bodo cloud service simplifies managing compute clusters and jobs on AWS and Azure.


### What is the difference between Bodo DataFrames' Pandas API vs JIT decorator and when should I use one over the other?

Bodo DataFrames provides two methods of accelerating and scaling Pandas code. The first method involves using the
Pandas API through the `bodo.pandas` module, which will lazily accumulate operations,
optimize and then run them in a parallel, C++ runtime using MPI.
The second method involves annotating functions with the Just-In-Time (JIT) compilation decorator (`bodo.jit`) in order to
compile Python code to optimized, parallel binaries that run on an MPI backend.
The Pandas API is easier to use; just replace the `import pandas as pd` with `import bodo.pandas as pd`,
while JIT provides better performance in some cases at the cost of extra set up such as
moving performance critical pieces to separate functions and ensuring those functions are jittable.
In addition to Pandas, JIT also natively supports Numpy and Scikit-learn.
We recommend using Bodo DataFrames' Pandas APIs when getting started and experimenting with JIT
for even better performance. See our [Python local quick start guide][quickstart-local-python] for examples.

### How can I get help if I encounter an issue?

For support, you can join the [Bodo Community Slack](https://bodocommunity.slack.com/join/shared_invite/zt-qwdc8fad-6rZ8a1RmkkJ6eOX1X__knA#/shared-invite/email),
where users and contributors actively discuss and resolve issues.
You can also refer to the documentation or raise an issue on GitHub if you’ve encountered a bug or need technical guidance.

### Are there any usage limits or restrictions?

No, there are no usage limits. Bodo DataFrames is open source and distributed under the Apache-2.0 License. You can use Bodo DataFrames in personal, academic, and commercial projects.

### How does Bodo DataFrames handle security and privacy?

Since Bodo DataFrames runs in your environment, you have complete control over the data and compute resources, ensuring that sensitive information remains secure.

### How does Bodo DataFrames handle fault tolerance and failures?

Bodo DataFrames runs parallel processes in an efficient, C++ runtime, eliminating most software failures
of other framework (e.g., scheduler errors, JVM errors, ...).
To handle rare case of hardware failures, we recommend simply configuring job restarts
in your environment.
Bodo DataFrames' high performance ensures fast restart and job completion even with failures.

### Does Bodo DataFrames have a job scheduler?

Bodo DataFrames provides parallel compute that can work with any job scheduler.
Bodo DataFrames does not bundle its own job scheduler.

### Am I supposed to have MPI nodes set up already?

MPI setup is required for distributed execution, which involves passwordless SSH access across nodes. See [cluster installation docs][cluster_setup].

### How do I know if my workload is actually being parallelized?

If you are using Bodo DataFrames' Pandas API (`bodo.pandas`), all operations will automatically be executed in parallel
unless otherwise indicated in the form of a user warning. For functions annotated with `bodo.jit`,
you can see distributed diagnostics of the compiled function to understand parallelism decisions by the compiler. See [parallelism docs][basics].
In addition, the Bodo compiler throws a warning if it didn't find any parallelism opportunities in a function.
You can also see your local machine's CPU usage using tools like `top` and `htop` for both Pandas API and JIT.


### How do I handle unsupported operations or libraries inside of JIT?

You can use unsupported operations and libraries outside JIT functions, or use
the `@bodo.wrap_python` decorator to use any regular Python function inside JIT functions.
See [wrap_python docs][objmode].


### Can Bodo DataFrames read from databases like Postgres, Snowflake, or BigQuery?

Bodo DataFrames can read from all databases that are readable in Python.
The Snowflake connector is especially optimized for very high performance.

### Does Bodo DataFrames work inside Databricks environment?

Bodo DataFrames currently does not work inside Databricks due to networking restrictions.
We plan to investigate and provide a solution.

### Can I create custom user-defined functions (UDFs) with Bodo DataFrames?

Yes, Bodo DataFrames is particularly good at accelerating UDFs in Pandas APIs such as `DataFrame.apply` and `Series.map`.

### What is the difference between the open source Bodo DataFrames and the Bodo Cloud Platform?

The Bodo Cloud Platform simplifies managing compute clusters, notebooks and jobs that use Bodo DataFrames.
Bodo Cloud Platform currently supports AWS and Azure.

### Does Bodo DataFrames have a SQL interface?

Yes, BodoSQL is a high performance SQL engine that provides vectorized streaming execution and
support distributed clusters using MPI.
BodoSQL compiles SQL queries into optimized binaries, which is particularly
good for large batch jobs.


### When is Bodo DataFrames *Not* appropriate? What is it *not* designed for?
Bodo DataFrames is designed for large-scale data processing and may not be appropriate for other use cases
such as accelerating Python web frameworks (e.g. Django) and other non-compute Python applications.
