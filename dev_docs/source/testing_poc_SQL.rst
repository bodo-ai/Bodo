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

AWS instance for the SQL tests in the CI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
There are 3 steps needed to setup the SQL testing for the CI:
- Create an RDS instance and make it publicly accessible
- Create a database for testing with some non trivial data. A good example of an SQL database is `https://github.com/datacharmer/test_db`
  The data is inserted in the database via ``mysql -u admin -p < employees.sql``. The name will be ``employees`` in it.
- Change the address of the database and the credentials used in the tests (currently setup in ``bodo/tests/test_sql.py``).
  
[DEPRECATED OLD SETUP] AWS instance for the SQL tests in the CI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to effectively test the SQL code in BODO, we need to have SQL test code in ``test_sql.py`` for
the ``df.to.sql`` and ``pd.read_sql`` commands.

In order to make this test meaningful we need to set up our own SQL server. This was done in AWS as
we can take eligible for free tier instances.

The set up however, was kind of non-trivial:

- The instance that turns out to be best was Ubuntu 18.04 which uses SQL 5.7 (the version 8.0.xx were
  apparently more problematic). Instance is standard EC2.
- The server is installed via ``sudo apt-get install mysql-server``
- One needs to set up an account accessible to outside. The method is to do::

.. code-block:: mysql

  # sudo mysql -u root
  mysql> CREATE USER 'admin'@'localhost' IDENTIFIED BY 'some_pass';
  mysql> GRANT ALL PRIVILEGES ON *.* TO 'admin'@'localhost' WITH GRANT OPTION;
  mysql> CREATE USER 'admin'@'%' IDENTIFIED BY 'some_pass';
  mysql> GRANT ALL PRIVILEGES ON *.* TO 'admin'@'%' WITH GRANT OPTION;
  mysql> FLUSH PRIVILEGES;
  mysql> EXIT

- The port 3306 has to be explicitly allowed in AWS. Follow the documentation on `https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/authorizing-access-to-an-instance.html`
- By default the database can be accessed only locally. This can be changed by having ``bind-address = 0.0.0.0``
  in the file ``/etc/mysql/mysql.conf.d/mysqld.cnf``
- A good nontrivial example of an SQL database is `https://github.com/datacharmer/test_db`
- The data is inserted in the database via ``mysql -u admin -p < employees.sql``. The name will be ``employees``
  in it.
- After change to the config use ``sudo systemctl restart mysql`` to restart the config.

Other advice:

- In order to test if the AWS firewall allows connection to the MYSQL port (it is 3306 usually), one can do
  ``telnet ec2-34-230-65-187.compute-1.amazonaws.com 3306``. If it blocks at ``Trying ...`` then the AWS firewall
  is present, otherwise, it is another problem.
- In order to check that MYSQL is correctly opened on the side of the server, one needs to use ``netstat -tlnp``
  and one should see ``0.0.0.0:3306``
- In order to check that the username/password is correctly set up for the database is to do (note the lack of sudo)::

  # mysql -u username -p
  Password:
  mysql>

- Docker runs are done as root and this can create some confusion for setup.
- Error messages are in ``/var/log/mysql/error.log`` but did not turned out to be particularly useful.
- The ``skip-grant-tables`` is a false track, do not use it.
- We could probably do things more simply with an RDS instance. See `https://aws.amazon.com/rds/mysql/`

Other useful links:
- `link <https://stackoverflow.com/questions/1559955/host-xxx-xx-xxx-xxx-is-not-allowed-to-connect-to-this-mysql-server>`_
- `link <https://stackoverflow.com/questions/37879448/mysql-fails-on-mysql-error-1524-hy000-plugin-auth-socket-is-not-loaded>`_
- `link <https://copir.net/how-to-fix-error-1698-28000-access-denied-for-user-root-localhost-in-ubuntu-18-04/>`_
- `link <https://support.rackspace.com/how-to/install-mysql-server-on-the-ubuntu-operating-system/>`_
