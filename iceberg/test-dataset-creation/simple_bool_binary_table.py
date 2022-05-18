import numpy as np
import pandas as pd
import pyspark.sql.types as spark_types
from utils import create_iceberg_table

df = pd.DataFrame(
    {
        "A": np.array([True, False, True, True] * 25, dtype=np.bool_),
        "B": np.array([False, None] * 50, dtype=np.bool_),
        "C": np.array([1, 1, 0, 1, 0] * 20).tobytes(),
    }
)
schema = spark_types.StructType(
    [
        spark_types.StructField("A", spark_types.BooleanType(), False),
        spark_types.StructField("B", spark_types.BooleanType(), True),
        spark_types.StructField("C", spark_types.BinaryType(), True),
    ]
)

create_iceberg_table(df, schema, "simple_bool_binary_table")
