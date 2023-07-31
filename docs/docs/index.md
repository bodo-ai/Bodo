---
hide:
  - navigation
  - toc
---

# About Bodo

Bodo is the first HPC-based compute engine for SQL and Python data processing. 
With its new just-in-time (JIT) inferential compiler, Bodo brings
supercomputing-style performance and scalability to native SQL and Python
code automatically. Bodo significantly improves the performance of long-running
data engineering workloads on popular data warehouses by typically saving more than 60% of the infrastructure cost. 

Bodo’s compiler optimization and parallel runtime system technologies bring HPC levels of performance and efficiency to 
large-scale data processing for the first time. Data warehouses focus on decades-old database techniques such as 
indexing—ensuring that a minimal amount of rows is scanned to match query filters that target small portions of the data.
However, modern queries that require heavy computation on large data also need MPI parallelization and low-level code 
optimization techniques to run efficiently. The Bodo Compute Engine brings these optimization techniques to data
engineering without requiring any code change or tuning.


<center>
<div class="video-wrapper">
    <iframe width="900" 
            height="500" 
            src="https://www.youtube.com/embed/PO5ke4MD_cI"
            title="YouTube video player" 
            frameborder="0" 
            allow=" accelerometer; 
                    autoplay; 
                    clipboard-write; 
                    encrypted-media; 
                    gyroscope; 
                    picture-in-picture" 
            allowfullscreen>
        </iframe>
</div>
</center>


Bodo operates on the data in your data warehouse or data lake without copy.
Data always stays in your own VPC, so you can comply with the security standards that your organization may require.
The Bodo Platform provides a simple, interactive workflow for development, deployment, and monitoring.
Quickly get up and running with the most challenging workloads in your existing cloud account.

This documentation covers the basics of using Bodo and provides a
reference of supported SQL and Python features and APIs.
To get started with Bodo using the Bodo Platform, refer to our [Quick Start Guide][bodoplatformquickstart]. 

If you want to try the free community edition of Bodo on your local setup, you can follow the steps outlined [here][install].
