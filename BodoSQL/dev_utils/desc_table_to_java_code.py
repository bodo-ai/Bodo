# Helper function when copying the schema from Snowflake to BodoSQL Java CI
# basically, run describe table in the web ui, download the output, and then run this script
# on that CSV. It will print out the Java code to copy and paste into the BodoSQL Java CI.

import os

import pandas as pd


def getColumnNameString(df, i):
    return df["name"].iloc[i]


def getBodoSQLColumnDataTypeString(df, i):
    snowflakeTypeName = df["type"].iloc[i]

    if snowflakeTypeName.startswith("VARCHAR"):
        return "BodoSQLColumnDataType.STRING"
    elif snowflakeTypeName.startswith("CHAR"):
        return "BodoSQLColumnDataType.STRING"
    elif snowflakeTypeName.startswith("TIMESTAMP_NTZ"):
        return "BodoSQLColumnDataType.DATETIME"
    elif snowflakeTypeName.startswith("TIMESTAMP_LTZ"):
        return "BodoSQLColumnDataType.TZ_AWARE_TIMESTAMP"
    elif snowflakeTypeName == "FLOAT":
        return "BodoSQLColumnDataType.FLOAT64"
    elif snowflakeTypeName.startswith("NUMBER"):
        if snowflakeTypeName.endswith(",0)"):
            return "BodoSQLColumnDataType.INT64"
        else:
            return "BodoSQLColumnDataType.FLOAT64"
    elif snowflakeTypeName.startswith("BOOL"):
        return "BodoSQLColumnDataType.BOOL8"
    elif snowflakeTypeName == "DATE":
        return "BodoSQLColumnDataType.DATE"
    elif snowflakeTypeName == "VARIANT":
        return "BodoSQLColumnDataType.VARIANT"
    elif snowflakeTypeName == "ARRAY":
        return "BodoSQLColumnDataType.ARRAY"
    elif snowflakeTypeName == "OBJECT":
        return "BodoSQLColumnDataType.JSON_OBJECT"
    else:
        raise Exception("Unsupported Snowflake type: " + snowflakeTypeName)


def getIsNullString(df, i):
    if df["null?"].iloc[i] == "Y":
        return "true"
    return "false"


def genJavaCode(csv_path):
    df = pd.read_csv(csv_path)

    for i in range(len(df)):
        colName = getColumnNameString(df, i)
        colDataType = getBodoSQLColumnDataTypeString(df, i)
        isNull = getIsNullString(df, i)
        print(f'column("{colName}", {colDataType}, {isNull})')


if __name__ == "__main__":
    import argparse

    # Just stick the path to the csv file in here
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--file",
        dest="file",
        action="store",
        type=str,
        default="./keaton.csv",
        help="csv file to read from",
    )

    args = parser.parse_args()
    genJavaCode(os.path.normpath(args.file))
