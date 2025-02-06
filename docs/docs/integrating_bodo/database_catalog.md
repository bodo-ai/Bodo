# Native SQL with Database Catalogs {#sql_catalog}


Database Catalogs are configuration objects that grant BodoSQL access to load tables from a remote database. 
Bodo platform now supports adding Database catalogs through the UI and provides users the option to write native
SQL code to run on the tables in the connected remote database.  


## Adding a Database Catalog

:fontawesome-brands-aws: Supported On AWS ·
:material-microsoft-azure:{.azure} Supported on Azure 


In your workspaces view, navigate to the _Catalogs_ section in the sidebar.
Click on _CREATE CATALOG_ and fill up the form with the required values.  

![Catalogs](../platform2-screenshots/catalogspage.png#center) 


Currently, we support Snowflake Database Catalogs, and AWS Glue Catalogs on the Bodo Platform.  
See [`SnowflakeCatalog`][snowflake-catalog-api] and [`GlueCatalog`][glue-catalog-api] for details on
the required parameters.

Upon submitting the form, you will see that your Catalog has been created and is now
available to use in your interactive notebook. 

![Catalog List](../platform2-screenshots/added_catalog.png)


## Running a Job With a Database Catalog

:fontawesome-brands-aws: Supported On AWS ·
:material-microsoft-azure:{.azure} Supported on Azure 

To run a SQL job with the database catalog you need to create a job template in the jobs tab.

![Job List](../platform2-screenshots/empty_jobs_tab.png)

Configure the job template normally and under Advanced Options, you can select the Catalog you want to use.

![Job Template](../platform2-screenshots/job_template_catalog.png)

Catalogs can also be given to bodosdk jobs.
When the job is run, the SQL code will be executed on the tables in the connected remote database.
 
## Using Database Catalogs in Interactive Notebooks

:fontawesome-brands-aws: Supported On AWS ·
:material-microsoft-azure:{.azure} Supported on Azure

!!! Important
    Using Database Catalogs in Interactive Notebooks is only supported for Snowflake Database Catalogs.

When you create a code cell in your interactive notebook, you will notice a blue selector on the
top right hand corner of the code cell. By default, this will be set to _Parallel-Python_.
This means that any code written in this cell will execute on all cores of the attached cluster. 

![Code cell](../platform2-screenshots/code_block_basic.png)

To enable running native SQL code, you can set the cell type in the blue selector to SQL, and you 
will need to select your Catalog from the Catalog selector to the left of the cell type selector as shown in the 
figure below. 

![Native SQL cell](../platform2-screenshots/selectcatalog.png)

The output of the SQL query is automatically saved in a distributed dataframe named _LAST\_SQL\_OUTPUT_. This dataframe will be
overwritten every time a SQL query is run. 

    
## Viewing Database Catalogs Data

To view the connection data stored in a catalog first connect to a cluster and then run the following in a code cell:

```python
import bodo_platform_utils
bodo_platform_utils.catalog.get_data("catalog_name")
```


!!! seealso "See Also"

    [Database Catalogs][database-catalogs], [BodoSDK Catalog API](https://pypi.org/project/bodosdk/#catalog)
