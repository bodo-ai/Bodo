# Example usage: python -u BodoSQLWrapper.py -c creds.json -f query.sql
# To see all options, run: python -u BodoSQLWrapper.py --help

import argparse
import json
import time
from urllib.parse import urlencode

import bodosql

import bodo
import bodo.utils.tracing as tracing

# Turn verbose mode on
bodo.set_verbose_level(2)


@bodo.jit(cache=True)
def run_sql_query(
    query_str, bc, pq_out_filename, sf_out_table_name, sf_write_conn, print_output
):
    """Boilerplate function to execute a query string.

    Args:
        query_str (str): Query text to execute
        bc (bodosql.BodoSQLContext): BodoSQLContext to use for query execution.
        pq_out_filename (str): When provided (i.e. not ''), the query output is written to this location as a parquet file.
        sf_out_table_name (str): When provided (i.e. not ''), the query output is written to this table in Snowflake.
        sf_write_conn (str): Snowflake connection string. Required for Snowflake write.
        print_output: Flag to print query result.
    """

    print(f"Started executing query:\n{query_str}")
    t0 = time.time()
    output = bc.sql(query_str)
    print(f"Finished executing the query. It took {time.time() - t0} seconds.")
    print("Output Shape: ", output.shape)
    if print_output:
        print("Output:")
        print(output)
    if pq_out_filename != "":
        print("Saving output as parquet dataset to: ", pq_out_filename)
        t0 = time.time()
        output.to_parquet(pq_out_filename)
        print(f"Finished parquet write. It took {time.time() - t0} seconds.")
    if sf_out_table_name != "":
        print("Saving output to Snowflake table: ", sf_out_table_name)
        t0 = time.time()
        output.to_sql(sf_out_table_name, sf_write_conn, if_exists="replace")
        print(f"Finished snowflake write. It took {time.time() - t0} seconds.")


def main(args):

    # Read in the query text from the file
    with open(args.filename, "r") as f:
        sql_text = f.read()

    # Fetch and create catalog
    with open(args.catalog_creds, "r") as f:
        catalog = json.load(f)

    warehouse = args.warehouse if args.warehouse else catalog.get("SF_WAREHOUSE")
    if warehouse is None:
        raise ValueError(
            "No warehouse specified in either the credentials file or through the arguments."
        )

    database = args.database if args.database else catalog.get("SF_DATABASE")
    if database is None:
        raise ValueError(
            "No database specified in either the credentials file or through the arguments."
        )

    # Create catalog from credentials and args
    bsql_catalog = bodosql.SnowflakeCatalog(
        username=catalog["SF_USERNAME"],
        password=catalog["SF_PASSWORD"],
        account=catalog["SF_ACCOUNT"],
        warehouse=warehouse,
        database=database,
    )

    # Create context
    bc = bodosql.BodoSQLContext(catalog=bsql_catalog)

    # Generate the plan and write it to a file
    if args.generate_plan_filename:
        plan_text = bc.generate_plan(sql_text)
        if bodo.get_rank() == 0:
            with open(args.generate_plan_filename, "w") as f:
                f.write(plan_text)
            print("Saved Plan to: ", args.generate_plan_filename)

    # Convert to pandas and write to file
    if args.pandas_out_filename:
        pandas_text = bc.convert_to_pandas(sql_text)
        if bodo.get_rank() == 0:
            with open(args.pandas_out_filename, "w") as f:
                f.write(pandas_text)
            print("Saved Pandas Version to: ", args.pandas_out_filename)

    sf_write_conn = ""
    sf_out_table_name = ""
    if args.sf_out_table_loc:
        params = {"warehouse": bsql_catalog.warehouse}
        db, schema, sf_out_table_name = (
            args.sf_out_table_loc.split(".") if args.sf_out_table_loc else ("", "", "")
        )
        sf_write_conn = f"snowflake://{bsql_catalog.username}:{bsql_catalog.password}@{bsql_catalog.account}/{db}/{schema}?{urlencode(params)}"
    print_output = False
    if args.print_output:
        print_output = True

    # Run the query
    if args.trace:
        tracing.start()
    t0 = time.time()
    run_sql_query(
        sql_text,
        bc,
        args.pq_out_filename if args.pq_out_filename else "",
        sf_out_table_name,
        sf_write_conn,
        print_output,
    )
    bodo.barrier()
    if args.trace:
        tracing.dump(fname=args.trace)
    if bodo.get_rank() == 0:
        print("Total (compilation + execution) time:", time.time() - t0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="BodoSQLWrapper",
        description="Runs SQL queries from files",
    )

    parser.add_argument(
        "-c",
        "--catalog_creds",
        required=True,
        help="Path to Snowflake credentials file. The following keys must be present: SF_USERNAME, SF_PASSWORD and SF_ACCOUNT. The following keys are optional: SF_WAREHOUSE, SF_DATABASE",
    )
    parser.add_argument(
        "-f", "--filename", required=True, help="Path to .sql file with the query."
    )
    parser.add_argument(
        "-w",
        "--warehouse",
        required=False,
        help="Optional: Snowflake warehouse to use for getting metadata, as well as I/O. When provided, this will override the default value in the credentials file.",
    )
    parser.add_argument(
        "-d",
        "--database",
        required=False,
        help="Optional: Snowflake Database which has the required tables. When provided, this will override the default value in the credentials file.",
    )
    parser.add_argument(
        "-o",
        "--pq_out_filename",
        required=False,
        help="Optional: Write the query output as a parquet dataset to this location.",
    )
    parser.add_argument(
        "-s",
        "--sf_out_table_loc",
        required=False,
        help="Optional: Write the query output as a Snowflake table. Please provide a full table path of the form <database>.<schema>.<table_name>",
    )
    parser.add_argument(
        "-p",
        "--pandas_out_filename",
        required=False,
        help="Optional: Write the pandas code generated from the SQL query to this location.",
    )
    parser.add_argument(
        "-t",
        "--trace",
        required=False,
        help="Optional: If provided, the tracing will be used and the trace file will be written to this location",
    )
    parser.add_argument(
        "-g",
        "--generate_plan_filename",
        required=False,
        help="Optional: Write the SQL plan to this location.",
    )
    parser.add_argument(
        "-u",
        "--print_output",
        required=False,
        action='store_true',
        help="Optional: If provided, the result will printed to std. Useful when testing and don't necessarily want to save results.",
    )

    args = parser.parse_args()

    main(args)
