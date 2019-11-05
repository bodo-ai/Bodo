.. _development:

Run SQL code
------------

We use OmniSci to run SQL code on docker.
Follow the preparation and activatetion steps in this 
`link <https://www.omnisci.com/platform/downloads/OS-installation-recipes?compute_mode=cpuonly&environment=docker>`_
and then verify `omnisql <https://docs.omnisci.com/latest/3_omnisql.html>`_
is working on docker.

Start a Bash session with OmniSci image and mount necessary directories::
  
  docker run -it --name omnisci -p 6274:6274  -v ~/claims_poc:/claims_poc -v ~/Downloads/data:/data omnisci/core-os-cpu

Then run your SQL code with the following command::
  
  cat ../claims_poc/tests/omnisci/get_csv.sql | bin/omnisql -p HyperInteractive
  
SQL syntax may need to be rewritten, more information can be found in **SQL section** of 
`OmiSci documentation <https://docs.omnisci.com/latest/>`_.

OmniSci could crash on long queries if the data set is too big relative to the memory size on host machine, so just take a sample of large data for testing purposes.

Test Python code correctness
----------------------------

Compare python output and SQL output with :code:`pd.testing.assert_frame_equal()`

.. code-block:: Python

  dataframe_sql = pd.read_csv('tests/SQL_output/COLUMN_MAPPINGS.csv', sep='|')
  pd.testing.assert_frame_equal(
	dataframe_python.sort_values(dataframe_python.columns.to_list()).reset_index(drop=True),
	dataframe_sql.sort_values(dataframe_sql.columns.to_list()).reset_index(drop=True))

Test intermediate tables since some columns of tables are never used.

Example with `claims_poc <https://github.com/Bodo-inc/claims_poc>`_
-------------------------------------------------------------------

`original SQL code <https://github.com/Bodo-inc/claims_poc/blob/master/iPhone_Claims.sql>`_

`edited SQL code for OmniSci <https://github.com/Bodo-inc/claims_poc/blob/master/tests/omnisci/get_csv.sql>`_

`test Python code correctness <https://github.com/Bodo-inc/claims_poc/blob/master/tests/test_python.py>`_

test1.py through test9.py are used to `test Bodo code correctness <https://github.com/Bodo-inc/claims_poc/tree/master/tests>`_
