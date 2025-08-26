"""
This script can be used as part of the procedure to upload a DataFrame of data to our Snowflake
account (specifically in TEST_DB.PUBLIC). To do so, follow the steps below in order:

1. Convert your DataFrame to a parquet file located in the same folder as this script. For example,
   if you have a DataFrame `df`, you can use `df.to_parquet("new_table.pq", index=False)` to write it
   to a parquet file named "new_table.pq".

2. Export the Snowflake credentials into your local environment as SF_USERNAME and SF_PASSWORD. To find
   which values to use, look in 1Password for the "Snowflake (bodopartner.us-east-1)" credentials. For
   ease of debugging errors, it is preferred to also run `export BODO_SF_DEBUG_LEVEL=1`.

3. Log into the Snowflake UI for our account (https://bodopartner.us-east-1.snowflakecomputing.com/) using
   those same credentials, and use the UI to create a new table with the desired name, column names, and
   column types. For example, if I wanted to create the table with the name OMEGA_SIX and columns
   alpha, beta, and gamma with types integer, string, and date (the types of which should match the
   types from `df` in step #1), I would use the following command in Snowflake:

    `CREATE OR REPLACE TABLE OMEGA_SIX(alpha INTEGER, beta VARCHAR, gamma DATE);`

     *** Make sure the current default database is TEST_DB, and the default schema is PUBLIC ***

    If you are unsure what types to use, copy the function get_schema lower in the file
    into the same python process where you ran `df.to_parquet(...)` and call
    get_schema(df) to learn what Snowflake types should be used.

4. Immediately after creating the table, run a command to ensure that SYSADMIN has access to
   the newly created table. If you ever re-run the `CREATE OR REPLACE TABLE` command, you will need to
   re-run this command. For the table above, this will look as follows:

   `GRANT OWNERSHIP ON OMEGA_SIX TO SYSADMIN;`

5. At the bottom of this script, change the strings in lines to use the names of the parquet file and
   table name you created in Snowflake:

    parquet_file_path = "new_table.pq"
    snowflake_table_name = "OMEGA_SIX"

6. Run this Python file: python upload_table_to_snowflake.py. When you are done and sure everything worked,
   undo the changes from step #5.

7. To confirm everything worked, go back to the Snowflake UI and try examining the data from the table,
   e.g. `SELECT * FROM OMEGA_SIX;`. If you see an error message, it could mean one of the following:
   - The table was not created in Snowflake.
   - The column types in Snowflake were provided incorrectly.
   - You did not run the GRANT OWNERSHIP command on the most recent version of the table.
   - The snowflake credentials were exported incorrectly.
"""

import os

import numba
import pandas as pd

import bodo
from bodo.io.snowflake import execute_copy_into, snowflake_connect


def upload_pq_to_sf(conn_str, df, pq_str, table_name):
    cursor = snowflake_connect(conn_str).cursor()
    try:

        def test_impl_execute_copy_into(cursor, stage_name, location, sf_schema, df_in):
            with numba.objmode():
                execute_copy_into(
                    cursor,
                    stage_name,
                    location,
                    sf_schema,
                    dict(zip(df_in.columns, bodo.typeof(df_in).data)),
                )

        bodo_impl = bodo.jit()(test_impl_execute_copy_into)
        stage_name = "tmp_sf_upload_stage_42"
        create_stage_sql = f'CREATE STAGE "{stage_name}"'
        cursor.execute(create_stage_sql, _is_internal=True).fetchall()
        upload_put_sql = (
            f"PUT 'file://{pq_str}' @\"{stage_name}\" AUTO_COMPRESS=FALSE "
            f"/* tests.test_sql:test_snowflake_write_create_table_handle_exists() */"
        )
        cursor.execute(upload_put_sql, _is_internal=True)
        sf_schema = bodo.io.snowflake.gen_snowflake_schema(
            df.columns, bodo.typeof(df).data
        )
        bodo_impl(
            cursor,
            stage_name,
            table_name,
            sf_schema,
            df,
        )
    finally:
        cleanup_stage_sql = f'DROP STAGE IF EXISTS "{stage_name}" '
        cursor.execute(cleanup_stage_sql, _is_internal=True).fetchall()


def get_schema(df):
    return bodo.io.snowflake.gen_snowflake_schema(df.columns, bodo.typeof(df).data)


def get_conn_str():
    username = os.environ.get("SF_USERNAME")
    password = os.environ.get("SF_PASSWORD")
    account = "bodopartner.us-east-1"
    database = "TEST_DB"
    schema = "public"
    warehouse = "DEMO_WH"
    return f"snowflake://{username}:{password}@{account}/{database}/{schema}?warehouse={warehouse}"


parquet_file_path = "new_table.pq"
snowflake_table_name = "OMEGA_SIX"

if __name__ == "__main__":
    conn_str = get_conn_str()
    df = pd.read_parquet(parquet_file_path)
    upload_pq_to_sf(conn_str, df, parquet_file_path, snowflake_table_name)
