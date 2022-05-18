# pip install avro

import copy
import json
import sys

import pandas as pd
from avro.datafile import DataFileReader
from avro.io import DatumReader

fname = sys.argv[1]

# Read data from an avro file
with open(fname, "rb") as f:
    reader = DataFileReader(f, DatumReader())
    metadata = copy.deepcopy(reader.meta)
    schema_from_file = json.loads(metadata["avro.schema"])
    fields = [field for field in reader]
    reader.close()

print(f"Schema from file:\n {schema_from_file}")
print(f"Fields:\n {fields}")

df = pd.DataFrame.from_records(fields)
print(f"df:\n", df)
df.to_csv("to_inspect.csv")
