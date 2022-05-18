import numpy as np
import pandas as pd
import pyspark.sql.types as spark_types
from utils import create_iceberg_table, get_spark

spark = get_spark()

table_name = "file_subset_deleted_rows_table"

# Write a simple dataset
print("Writing a simple dataset...")
df = pd.DataFrame(
    {
        "A": np.array(["A", "B", "C", "D"] * 5),
        "B": np.array(["lorem", "ipsum"] * 10),
        "C": np.array((["A"] * 2) + (["b"] * 18)),
        "D": np.array([1, 2] * 10, np.int32),
        "E": np.array([1, 2] * 10, np.float32),
        "K": np.array([54] * 20, np.int64),
    }
)
schema = spark_types.StructType(
    [
        spark_types.StructField("A", spark_types.StringType(), True),
        spark_types.StructField("B", spark_types.StringType(), True),
        spark_types.StructField("C", spark_types.StringType(), True),
        spark_types.StructField("D", spark_types.IntegerType(), True),
        spark_types.StructField("E", spark_types.FloatType(), True),
        spark_types.StructField("K", spark_types.LongType(), True),
    ]
)
create_iceberg_table(df, schema, table_name, spark)

# Add more data
print("Adding some data...")
spark.sql(
    f"""
    INSERT INTO hadoop_prod.iceberg_db.{table_name}
    VALUES
    ('QWERTY', 'dolor', 'C', 5, 5.34, 32),
    ('ASDFGH', 'sit', 'D', 56, 9.87, 12);
"""
)

# Delete all rows except those from last insert
print("Deleting rows...")
spark.sql(
    f"""
    DELETE FROM hadoop_prod.iceberg_db.{table_name}
    WHERE K = 54;
"""
)
