.. _Databricks_integration:

Bodo Databricks Integration (Beta)
=====================================

You can develop your workflows with Bodo on interactive clusters on Databricks and seamlessly deploy them on Databricks using the existing infrastructure. Bodo also supports reading Delta datasets directly.

Please follow the steps below to set up Bodo to run on Databricks. Make sure to test your installation on the Databricks with our publicly available sample starter notebook.


Install and Run Bodo on Databricks
------------------------------------

- Create a cluster with Docker Container Services enabled.
- Use the relevant Docker image url on Bodo’s Docker hub  `Docker image url <https://hub.docker.com/repository/docker/bodoaidocker/ret-dbr-poc>`_ on Bodo’s Docker hub. (``bodo==2021.9`` available at bodoaidocker/ret-dbr-poc:v2021.9).
- Clone `DB_integration_setup repository <https://github.com/Bodo-inc/DB_integration_setup>`_ and upload the *setup_bodo*\  folder to the relevant folder in the Databricks workspace. 
- Enter the Bodo license in the last cell of *setup_nb_px*\  notebook (or enter the path to the Bodo license).
- Open the sample notebook ``run_bodo_workflow`` and enter the full path to *setup_nb_px*\  notebook in the ``%run`` cell.
- Enter the number of engines (physical cores) that you’d like to use in the first cell.
- Check the output of ``bodo.get_size()`` to ensure that all the engines that you requested are running.
- The Zeppelin notebook can now be used with ``%%px`` in the beginning of the cell to run the code on all the cores available.


Notes
-------

- The Databricks Bodo setup has been tested on DBR 9.0.
- Bodo supports reading Delta datasets directly from the data lake (with ``pd.read_parquet``). Please use the mounted data lake path to read a dataset.
- More details on using Databricks clusters with Docker images can be found `here <https://docs.databricks.com/clusters/custom-containers.html>`_.
- To install a new Python library, please use the *Libraries*\  tab.
- Please reach out to a Bodo Solutions Engineer for any further details through `discourse <https://discourse.bodo.ai>`_ or `community Slack channel <https://bodocommunity.slack.com/ssb/redirect>`_.