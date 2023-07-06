"""Helper script to copy over TPCH data."""


def cquery(tbl, schema):
    return f"""CREATE OR REPLACE TABLE TEST_DB.{schema}.{tbl} LIKE SNOWFLAKE_SAMPLE_DATA.{schema}.{tbl};"""


def iquery(tbl, schema):
    return f"""INSERT INTO TEST_DB.{schema}.{tbl} SELECT * FROM SNOWFLAKE_SAMPLE_DATA.{schema}.{tbl};"""


schemata = ("TPCH_SF100", "TPCH_SF1000")
tables = (
    "CUSTOMER",
    "LINEITEM",
    "NATION",
    "ORDERS",
    "PART",
    "PARTSUPP",
    "REGION",
    "SUPPLIER",
)

print(
    "-- Run the following commands in a snowflake console. Run each line separately, or"
)
print("-- copy over each section (separated by -- ----) and run all commands")
print()

for i, s in enumerate(schemata):
    if i > 0:
        print("-- ---")
    print(f"DROP SCHEMA IF EXISTS TEST_DB.{s};")
    print(f"CREATE SCHEMA TEST_DB.{s};")
    print("-- ---")
    for t in tables:
        print(cquery(t, s))
    print("-- ---")
    for t in tables:
        print(iquery(t, s))
