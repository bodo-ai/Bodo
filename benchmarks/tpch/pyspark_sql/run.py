import glob
import time

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

base = "/home/taanders/tpch/from_s3/SF1"

lineitem = spark.read.parquet(f"{base}/lineitem.pq")
orders = spark.read.parquet(f"{base}/orders.pq")
customer = spark.read.parquet(f"{base}/customer.pq")
part = spark.read.parquet(f"{base}/part.pq")
partsupp = spark.read.parquet(f"{base}/partsupp.pq")
supplier = spark.read.parquet(f"{base}/supplier.pq")
nation = spark.read.parquet(f"{base}/nation.pq")
region = spark.read.parquet(f"{base}/region.pq")

tables = {
    "lineitem": lineitem,
    "orders": orders,
    "customer": customer,
    "part": part,
    "partsupp": partsupp,
    "supplier": supplier,
    "nation": nation,
    "region": region,
}

for name, df in tables.items():
    df.createOrReplaceTempView(name)

for path in sorted(glob.glob("../sql/*.sql")):
    name = path.split("/")[-1]
    print(f"Running {name}")

    with open(path) as f:
        sql = f.read()

    start = time.time()
    df = spark.sql(sql)
    df.collect()  # force execution
    end = time.time()

    print(f"{name} took {end - start:.2f} seconds")
