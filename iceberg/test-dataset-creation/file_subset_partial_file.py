import numpy as np
import pandas as pd
import pyspark.sql.types as spark_types
from utils import create_iceberg_table, get_spark

spark = get_spark()

table_name = "file_subset_partial_file_table"

# Write a simple dataset
print("Writing a simple dataset...")
df = pd.DataFrame(
    {
        "A": np.array(["A", "B", "C", "D"] * 500000),
        "B": np.arange(0, 2000000, dtype=np.int64),
    }
)
schema = spark_types.StructType(
    [
        spark_types.StructField("A", spark_types.StringType(), True),
        spark_types.StructField("B", spark_types.LongType(), True),
    ]
)
create_iceberg_table(df, schema, table_name, spark)


# Delete some rows
print("Deleting some rows...")
spark.sql(
    f""" 
    DELETE FROM hadoop_prod.iceberg_db.{table_name}
    WHERE B = 16;
"""
)


## Seems like the way at least Spark does it is that it creates new files for the
## rows don't get filtered out, rather than storing any information about
## rows to skip.
