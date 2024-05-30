# Getting started with Tabular Database Catalogs


[Tabular](https://tabular.io){target=blank} is a universal data platform built around Iceberg. 
Bodo supports Tabular Database Catalogs, which allow you to read and write tables 
using the REST Iceberg Catalog provided by Tabular.

In this guide, we will walk you through the steps to create and use Tabular Database Catalogs on the Bodo Platform.

## Prerequisites

- A Bodo Platform account with an active subscription is required.
- A Tabular account is required. 
- Service credentials for the Tabular account are required. See the [Tabular documentation](https://docs.tabular.io/en/creating-and-modifying-credentials.html#creating-a-service-credential){target=blank} for more information.

## Creating a Tabular Database Catalog 

Once you have a Tabular account, 
log into the Bodo Platform and navigate to the _Catalogs_ section in the sidebar. Click on _CREATE CATALOG_
and fill up the form with the required values. 

A few things to note : 

- The `Catalog Type` should be set to `Tabular`, this will automatically fetch the required fields for the catalog details. 
- The `Iceberg REST URL` field of the catalog form should be filled with the Iceberg REST URI for your Tabular account. Typically this will be `https://api.tabular.io/ws/`. If you are using a test account on tabular, the URI may be `https://api.test.tabular.io/ws/`. 
- The `Credential` field should be filled with the service credential you created in the Tabular account, which has the format `clientid:clientsecret`.

Upon submitting the form, you will see that your Catalog has been created and is now available to use.


## Using Interactive Notebooks with Tabular Catalogs
First, [create a cluster][creating_clusters] on the Bodo Platform to run the interactive notebook and [attach][attaching_notebook_to_cluster] the notebook to the cluster. 

We created a tabular catalog with the following details: 
- `Catalog Name`: `test-tabular`
- `Iceberg REST URL`: `https://api.test.tabular.io/ws/`
- `Warehouse`: `sandbox`
- 
First set the cell type to SQL and select the Tabular catalog from the catalog selector dropdown. 
The sql query you run will be executed on the tables using the Tabular catalog you selected. 

We will run a simple query to read the table `nyc_taxi_yellow` in the `examples` database in our Tabular account. 

```sql
select * from \"examples\".\"nyc_taxi_yellow\" limit 10
```

Note that we had to use double quotes around the database and table names 
because Bodo SQL currently requires them to be quoted to be case-sensitive. 

After running the query, the results are stored in a distributed dataframe named `LAST_SQL_OUTPUT`.
You can now use the `LAST_SQL_OUTPUT` dataframe in your code to perform further operations. 

!!! seealso "See Also"
    - [Database Catalogs](../integrating_bodo/database_catalog.md)
    - [Using Notebooks](../guides/using_bodo_platform/notebooks.md)


