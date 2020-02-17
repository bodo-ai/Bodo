.. _pocsql:

Customer Code Rewrite 
-----------------------

Run SQL code
~~~~~~~~~~~~~

We use OmniSci to run SQL code on docker.
Follow the preparation and activatetion steps in this 
`link <https://www.omnisci.com/platform/downloads/OS-installation-recipes?compute_mode=cpuonly&environment=docker>`_
and then verify `omnisql <https://docs.omnisci.com/latest/3_omnisql.html>`_
is working on docker.

Start a new container with OmniSci image and mount(with :code:`-v`) necessary directories(here I mounted :code:`~/claims_poc` and :code:`/~/Downloads/data`)::
  
  docker run -it -p 6274:6274  -v ~/claims_poc:/claims_poc -v ~/Downloads/data:/data omnisci/core-os-cpu

Keep this terminal open, and then open another terminal. Inside the new terminal:

1. use :code:`docker container ls` to find the CONTAINER ID of the container with IMAGE omnisci/core-os-cpu

2. use :code:`docker exec -it your_container_ID bash` to interact with the container we just started

Inside this container(which we are interacting with in the new terminal), you can run your SQL code with the following command::
  
  cat ../claims_poc/tests/omnisci/get_csv.sql | bin/omnisql -p HyperInteractive
  
SQL syntax may need to be rewritten, more information can be found in **SQL section** of 
`OmiSci documentation <https://docs.omnisci.com/latest/>`_.

OmniSci could crash on long queries if the data set is too big relative to the memory size on host machine, so just take a sample of large data for testing purposes.

After you are done with the container, stop and remove the container::
  
  # first stop the container
  docker stop your_container_ID
  # then remove the container
  docker rm your_container_ID

Test Python code correctness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compare python output and SQL output with :code:`pd.testing.assert_frame_equal()`

.. code-block:: Python

  dataframe_sql = pd.read_csv('tests/SQL_output/COLUMN_MAPPINGS.csv', sep='|')
  pd.testing.assert_frame_equal(
	dataframe_python.sort_values(dataframe_python.columns.to_list()).reset_index(drop=True),
	dataframe_sql.sort_values(dataframe_sql.columns.to_list()).reset_index(drop=True))

Test intermediate tables since some columns of tables are never used.

Example with `claims_poc <https://github.com/Bodo-inc/claims_poc>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you try to run this example on docker, there is a sample of data  inside claims_poc repo, so you could just mount :code:`claims_poc` directory when starting the docker. And in the edited SQL code(:code:`tests/omnisci/get_csv.sql`), you need to change the file paths to proper ones. 

`original SQL code <https://github.com/Bodo-inc/claims_poc/blob/master/iPhone_Claims.sql>`_

`edited SQL code for OmniSci <https://github.com/Bodo-inc/claims_poc/blob/master/tests/omnisci/get_csv.sql>`_

- Don't try to run this query with OmniSci as it requires data that was not in the repo
- But do look at it for inspirations

`test Python code correctness <https://github.com/Bodo-inc/claims_poc/blob/master/tests/test_python.py>`_

test1.py through test9.py are used to `test Bodo code correctness <https://github.com/Bodo-inc/claims_poc/tree/master/tests>`_
