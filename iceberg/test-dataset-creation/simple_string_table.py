import numpy as np
import pandas as pd
import pyspark.sql.types as spark_types
from utils import create_iceberg_table

df = pd.DataFrame(
    {
        "A": np.array(["A", "B", "C", "D"] * 25),
        "B": np.array(["lorem", "ipsum"] * 50),
        "C": np.array((["A"] * 10) + (["b"] * 90)),
    }
)
schema = spark_types.StructType(
    [
        spark_types.StructField("A", spark_types.StringType(), True),
        spark_types.StructField("B", spark_types.StringType(), True),
        spark_types.StructField("C", spark_types.StringType(), True),
    ]
)

create_iceberg_table(df, schema, "simple_string_table")
