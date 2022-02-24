"""Test connector for Google Cloud Storage."""

import pandas as pd

from bodo.tests.utils import check_func


def test_read_parquet_gcs():
    def impl(pq_file):
        df = pd.read_parquet(pq_file)
        return len(df)

    # We use len and expected_length here because this speeds up
    # computation by reading the number of rows instead of reading in
    # all the data
    expected_length = 837469
    for pq_file in [
        "gcs://anaconda-public-data/nyc-taxi/nyc.parquet/part.0.parquet",
        "gs://anaconda-public-data/nyc-taxi/nyc.parquet/part.0.parquet",
    ]:
        check_func(impl, (pq_file,), py_output=expected_length)


def test_read_parquet_gcs_filters():
    """
    Verify that filters work correctly with gcs.
    """

    def impl(pq_file):
        df = pd.read_parquet(pq_file)
        df = df[df.VendorID == 1]
        return len(df)

    # We use len and expected_length here because this speeds up
    # computation by reading the number of rows instead of reading in
    # all the data
    expected_length = 390146
    for pq_file in [
        "gcs://anaconda-public-data/nyc-taxi/nyc.parquet/part.0.parquet",
        "gs://anaconda-public-data/nyc-taxi/nyc.parquet/part.0.parquet",
    ]:
        check_func(impl, (pq_file,), py_output=expected_length)
