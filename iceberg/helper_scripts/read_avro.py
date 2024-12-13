"""
Helper script to dump the content of avro files for manual inspection.
Avro is installed by default on the dev Pixi environment.
"""

import sys

from avro.datafile import DataFileReader
from avro.io import DatumReader

reader = DataFileReader(open(sys.argv[1], "rb"), DatumReader())
for value in reader:
    print(value)
