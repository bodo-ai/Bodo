"""
Helper script to dump the content of avro files for manual inspection.
You will need to install the avro package to use this script.

pip install avro
"""
import sys

import avro
from avro.datafile import DataFileReader
from avro.io import DatumReader

reader = DataFileReader(open(sys.argv[1], "rb"), DatumReader())
for value in reader:
    print(value)
