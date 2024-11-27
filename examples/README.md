![](bodo-gray-green.svg)

# Let's Learn Bodo through Examples!

This is where you can find examples to help you get started using Bodo.

Bodo is the next generation big-data processing engine that brings supercomputing-style performance and scalability to native Python and SQL codes automatically. Bodo has several advantages over other big data transformation systems that makes it one of the most performant and cost-effective solutions for large scale data analytics, particularly ETL and ELT.

This repository teaches you to use Bodo effectively through examples. If you know SQL and Python, you already know how to use Bodo, and you don't need any new language or API. You will just `import bodo` and learn some programming tricks to improve your existing applications to save $$$ on compute resources while delivering value in a much shorter time-frame. Benchmarks have shown that Bodo can be [orders of magnitude faster](https://www.bodo.ai/blog/performance-and-cost-of-bodo-vs-spark-dask-ray) than its competitors like Spark.


## How to run these examples?

We recommend that you run these examples on the [Bodo Platform](https://platform.bodo.ai/account/sign-up). You can [sign up to our platform]((https://platform.bodo.ai/account/sign-up)) to try it out. Some examples like modules 1 to 3 can run on small clusters, e.g., 2 nodes of c5.2xlarge with total of 8 physical cores (16 vCPU) and 32GB RAM, and some examples need larger clusters. The description provided with each example indicates the size of cluster that is required to run it. 

You can also run these examples locally by [installing bodo](https://docs.bodo.ai/latest/installation_and_setup/install/#install)  on your laptop. However, we recommend using the [Bodo Platform](https://platform.bodo.ai) for the best experience as it provides a notebook environment with all the code available and required packages already installed for you. 

## What if I wanted to test my code with my data?

If you wanted to run your application codes with your own data, please refer to the instructions [here](https://docs.bodo.ai/latest/installation_and_setup/bodo_platform_aws/#setup-iam-role) on how to set up the identity access management, policies, and credentials to integrate your cloud provider with bodo platform. This allows bodo to spin up EC2 instances, create a cluster, and enable you to access your data within your VPC. Everything, including your data stays in your VPC.


## Modules outline

[Module 1](01-Getting-Started-Tutorials) Gives a qucik introduction for using bodo to accelerate your python programs, [module 2](02-ETL-Snowflake) focuses on compute heavy data transformations through ETL applications. You will find examples a data warehouse like Snowflake in Module 02, and a data Lakehouse example with Iceberg in [module 03](03-ETL-Iceberg).

 Modules [04](04-ML-at-Scale) and [05](05-Business-Usecases-at-Scale) contain larger scale examples with [Machine Learning](04-ML-at-Scale), [Business use cases](05-Business-Usecases-at-Scale) (financial, transportation, etc.). Module [06](06-Compare-Bodo-with-Spark) contains a performance comparison of Bodo vs Spark on a set of queries derived from the TPC-H benchmark suite. 

