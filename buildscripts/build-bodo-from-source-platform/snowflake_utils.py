import json

import pandas as pd
import snowflake.connector


def get_snowflake_connection(
    creds_filename, database=None, schema=None, warehouse=None
):
    """
    Read Snowflake credentials from a JSON credentials file and create a
    Snowflake connection object.

    Args:
        creds_filename (str): Path to file with Snowflake credentials. The file is expected
            to be a JSON file with the following keys: 'SF_USERNAME', 'SF_PASSWORD' and 'SF_ACCOUNT'.
        database (str, optional): Snowflake Database to use. When not provided, it uses the default
            database for the user.
        schema (str, optional): Database schema to use. When not provided, it uses the default
            schema for the user.
        warehouse (str, optional): Snowflake warehouse to use. When not provided, it uses the
            default warehouse for the user.

    Returns:
        Snowflake connector connection object
    """
    # Read Snowflake Credentials from a file
    with open(creds_filename) as f:
        sf_creds = json.load(f)

    username = sf_creds["SF_USERNAME"]
    password = sf_creds["SF_PASSWORD"]
    account = sf_creds["SF_ACCOUNT"]

    return snowflake.connector.connect(
        user=username,
        password=password,
        account=account,
        database=database,
        schema=schema,
        warehouse=warehouse,
    )


def get_current_warehouse(con):
    """
    Check the warehouse for your connection.

    Args:
        con : Snowflake connector connection object

    Returns:
        str: Name of the current warehouse
    """
    res = con.cursor().execute("select CURRENT_WAREHOUSE();")
    res = res.fetchall()  # Return a list of tuples
    return res[0][0]


def list_databases(con):
    """
    List all databases. This might be useful in case you don't know
    which database is being used by the query.

    Args:
        con : Snowflake connector connection object

    Returns:
        List of databases.
    """
    result = con.cursor().execute("show databases;")
    result = result.fetchall()
    return result


def list_schemas_in_database(con, db):
    """
    List the schemas in the database.

    Args:
        con : Snowflake connector connection object.
        db (str): Database to list the schemas in.

    Returns:
        List of schemas in the database.
    """
    result = con.cursor().execute(f"show schemas in {db};")
    result = result.fetchall()
    return result


def get_table_info(con, sf_db, sf_schema, table_name):
    """
    Get some basic information about a table. e.g. total number
    of rows, total size in bytes, whether it's transient or not, etc.

    Args:
        con : Snowflake connector connection object.
        sf_db (str): Database the table is in.
        sf_schema (str): Database schema the table is in
        table_name (str): Name of the table.

    Returns:
        Pandas DataFrame.
    """

    result = con.cursor().execute(
        f"select * from {sf_db}.information_schema.tables WHERE table_schema = '{sf_schema.upper()}' and table_name = '{table_name.upper()}';"
    )
    result = result.fetch_pandas_all()
    return result


def get_table_column_types(con, sf_db, sf_schema, table_name):
    """
    Get information about the columns in a table, in particular their
    types, whether they are nullable or not, etc.

    Args:
        con : Snowflake connector connection object.
        sf_db (str): Database the table is in.
        sf_schema (str): Database schema the table is in
        table_name (str): Name of the table.

    Returns:
        Pandas DataFrame.
    """

    query = f"show columns in table {sf_db.upper()}.{sf_schema.upper()}.{table_name.upper()};"

    # Get the column names of the query output
    result_desc = con.cursor().describe(query)
    columns = [a.name for a in result_desc]
    # Get the actual query result
    result = con.cursor().execute(query)
    result = result.fetchall()
    # Put the data in a Pandas Dataframe to make it easier to display
    out = pd.DataFrame({c: [r[i] for r in result] for i, c in enumerate(columns)})
    return out


def execute_query_in_snowflake(con, query, parquet_output_location=None):
    """
    Simple utility to execute a user query in Snowflake.

    Args:
        con : Snowflake connector connection object.
        query (str): User query to execute.
        parquet_output_location : If provided, the output is saved as a parquet file at the specified location.

    Returns:
        Pandas DataFrame
    """

    res = con.cursor().execute(query)
    out_df = res.fetch_pandas_all()
    if parquet_output_location:
        out_df.to_parquet(parquet_output_location)
    return out_df


def execute_query_and_save_to_snowflake_table(con, query, snowflake_output_table_name):
    """
    Simple utility to execute a user query in Snowflake and save the output to a
    Snowflake table.

    Args:
        con : Snowflake connector connection object.
        query : User query to execute.
        snowflake_output_table_name (str): The Snowflake table to save the output to.
            A full path, i.e. `{database}.{schema}.{table_name}` is recommended.

    """
    query = f"create or replace table {snowflake_output_table_name} as ({query})"
    res = con.cursor().execute(query)
    res = res.fetchall()
    return res


def compare_bodo_and_snowflake_outputs(
    con, bodo_out_table_loc, snowflake_out_table_loc
):
    """
    Compare Bodo and Snowflake outputs that are both stored in Snowflake tables.
    Handles issues in ordering.

    Args:
        con : Snowflake connector connection object.
        bodo_out_table_loc (str): Table where Bodo output is stored
        snowflake_out_table_loc (str): Table where Snowflake output is stored.

    Returns:
        Pandas DataFrame with difference between the two tables. If the table
        is empty, that means that there are no differences and the outputs match
        exactly.
        If the columns in the two tables don't match, it will throw an exception.
    """

    query = f"(select * from {bodo_out_table_loc}) except (select * from {snowflake_out_table_loc})"
    res = con.cursor().execute(query)
    res = res.fetch_pandas_all()
    return res
