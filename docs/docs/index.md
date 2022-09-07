---
hide:
  - navigation
  - toc
---

# About Bodo

Bodo is a new just-in-time (JIT) inferential compiler that brings
supercomputing-style performance and scalability to native Python
analytics code automatically. Bodo has several advantages over other big
data analytics systems (which are usually distributed scheduler
libraries):

-   Simple programming with native Python APIs 
    such as Pandas and Numpy (no "Pandas-like" API layers)
    
-   Extreme performance and scalability using true parallelism
    and advanced compiler technology
    
-   Very high reliability due to binary code generation,
    which avoids distributed library failures
    
-   Simple deployment using standard Python workflows

-   Flexible integration with other systems such as
    cloud storage, data warehouses, and visualization tools

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

This documentation covers the basics of using Bodo and provides a
reference of supported Python features and APIs. In a nutshell, Bodo
provides a JIT compilation workflow using the [`@bodo.jit` decorator][jit]. 
It replaces the decorated Python functions with an optimized and 
parallelized binary version automatically. For example, the program 
below can perform data transformation on large datasets:

```py

@bodo.jit
def data_transform(file_name):
    df = pd.read_parquet(file_name)
    df = df[df.C.dt.month == 1]
    df2 = df.groupby("A")["B", "D"].agg(
        lambda S: (S == "ABC").sum()
    )
    df2.to_parquet("output.pq")

```

To run Bodo programs such as this example, programmers can simply use
the command line such as `mpiexec -n 1024 python data_transform.py` (to
run on 1024 cores), or use [Jupyter Notebook][ipyparallelsetup].

