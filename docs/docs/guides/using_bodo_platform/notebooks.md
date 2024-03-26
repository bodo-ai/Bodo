# Using Notebooks {#notebooks}

Jupyter servers act as your interface to your shared file system and compute clusters.
Users can execute code from their notebooks on the compute cluster from the Jupyter interface.
A Jupyter server is automatically provisioned when you first enter the workspace.

![Notebook-View](../../platform2-screenshots/notebook_view.png#center)

You can update/restart Jupyter servers in the "Workspace Settings."

![Notebook-Manager](../../platform2-gifs/notebook_manager.gif#center)

## Attaching a Notebook to a Cluster {#attaching_notebook_to_cluster}

To attach a notebook to a cluster, select the cluster from the drop-down in the top-left.

![Attach-Cluster](../../platform2-gifs/attach-cluster.gif#center)

To execute your code across the attached cluster, select the _Parallel Python_ cell type from the cell type selector dropdown.

![Run-Code-Notebook](../../platform2-gifs/parallel-python.gif#center)

To run a SQL query, first select the catalog you want to use, then select the SQL cell type from the cell type selector dropdown. 
For more information on SQL catalogs, refer to the [SQL Catalogs usage guide][sql_catalog].

![Run-Code-Notebook](../../platform2-gifs/sql.gif#center)


!!! note 
    Execution is only allowed when the notebook is attached to a cluster.
    If you execute a cell without a cluster attached, the following warning will be shown:

![Detached-Notebook-Warning](../../platform2-gifs/not-attached-to-cluster-warning.gif#center)
