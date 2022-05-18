import sys

from utils import get_spark

table_name = sys.argv[1]

spark = get_spark()

df = spark.table(f"hadoop_prod.iceberg_db.{table_name}")
print(df.count())
print(df.schema)

df = df.toPandas()
print(df)
