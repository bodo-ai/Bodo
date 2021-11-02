.. _Databricks_integration:

Bodo Databricks Integration (Beta)
=====================================

Users can develop their workflows with Bodo on interactive clusters on Databricks and seamlessly deploy them using the existing deployment infrastructure. Bodo also supports reading Delta datasets directly.

Please follow the steps in the relevant section below that applies to your Databricks cluster to set up Bodo. Make sure to test your installation on the Databricks with our publicly available sample starter notebook.


Install and Run Bodo on Databricks
------------------------------------

Section 1. If you are using a Databricks Runtime version that has conda (DBR-ML)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Create a cluster with a DBR ML version.
- On the *Libraries*\ tab of the cluster, use ``PyPI`` as the source to install the following libraries:
 | ``numpy==1.20.3`` 
 | ``bodo==2021.10.1``
 | ``ipyparallel==7.1.0``
 | ``deltalake==0.5.3``
- Clone the *DB_integration_setup*\ repository and upload the folder *setup_bodo_dbr_ml*\  (available `here <https://github.com/Bodo-inc/DB_integration_setup>`_) to the relevant folder in the Databricks workspace.
- Enter the Bodo license in the last cell of *setup_nb_px*\ notebook (or enter the path to the Bodo license).
- Open the sample notebook *run_bodo_workflow*\ and enter the full path to *setup_nb_px*\ in the ``%run`` cell.
- Enter the number of engines (physical cores) you’d like to use in the first cell.
- Check the output of ``bodo.get_size()`` to ensure that all the requested engines are running.
- The Zeppelin notebook can now be used with ``%%px`` in the beginning of the cell to run the code on all available cores.


Section 2. If you are using a Standard Databricks Runtime version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Create a cluster with a Standard DBR version.
- On the *Libraries*\ tab of the cluster, use ``PyPI`` as the source to install the following libraries:
 | ``numpy==1.20.3`` 
 | ``bodo==2021.10.1``
 | ``ipyparallel==7.1.0``
 | ``deltalake==0.5.3``
- Clone the *DB_integration_setup*\ repository and upload the folder *setup_bodo_std_dbr_std*\ (available `here <https://github.com/Bodo-inc/DB_integration_setup>`_) to the relevant folder in the Databricks workspace.
- Enter the Bodo license in the last cell of *setup_nb_px*\ notebook (or enter the path to the Bodo license).
- Open the sample notebook *run_bodo_workflow*\ and enter the full path to *setup_nb_px*\ in the ``%run`` cell.
- Enter the number of engines (physical cores) you’d like to use in the first cell.
- Check the output of ``bodo.get_size()`` to ensure that all the requested engines are running.
- The Zeppelin notebook can now be used with ``%%px`` in the beginning of the cell to run the code on all available cores.

Section 3. If you have the option of using a Docker image to spin up a cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Create a cluster with Docker Container Services enabled.
- Use the relevant Docker image url on Bodo’s Docker hub  `Docker image url <https://hub.docker.com/repository/docker/bodoaidocker/ret-dbr-poc>`_ on Bodo’s Docker hub. (``bodo==2021.9`` available at bodoaidocker/ret-dbr-poc:v2021.9).
- Clone `DB_integration_setup repository <https://github.com/Bodo-inc/DB_integration_setup>`_ and upload the *setup_bodo*\  folder to the relevant folder in the Databricks workspace. 
- Enter the Bodo license in the last cell of *setup_nb_px*\  notebook (or enter the path to the Bodo license).
- Open the sample notebook *run_bodo_workflow*\ and enter the full path to *setup_nb_px*\  notebook in the ``%run`` cell.
- Enter the number of engines (physical cores) that you’d like to use in the first cell.
- Check the output of ``bodo.get_size()`` to ensure that all the engines that you requested are running.
- The Zeppelin notebook can now be used with ``%%px`` in the beginning of the cell to run the code on all the cores available.


Notes
~~~~~~~

- The Databricks Bodo setup has been tested on DBR 9.0, and on DBR 8.3 ML.
- To install a new Python library, please use the *Libraries*\ tab - Notebook scoped installation with Bodo does not currently work.
- Please specify specific versions of libraries that are installed on the cluster (for e.g., ``pip install bodo==2021.10.1``, and not simply ``pip install bodo``)
- Bodo supports reading Delta datasets directly from the data lake (with ``pd.read_parquet``). Please use the mounted data lake path to read a dataset.
- More details on using Databricks clusters with Docker images can be found `here <https://docs.databricks.com/clusters/custom-containers.html>`_.
- Please reach out to a Bodo Solutions Engineer for any further details through `discourse <https://discourse.bodo.ai>`_ or `community Slack channel <https://bodocommunity.slack.com/ssb/redirect>`_.


Potential Issues (Preemptive Troubleshooting)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Python executable used in each of the 3 sections above is available at a different location, depending on the type of Databricks cluster you are using. The location of the executable has been hard-coded in the Standard DBR version of the script to install Bodo.
