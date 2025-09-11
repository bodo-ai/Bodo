import numba

import bodo
from bodo.io.iceberg.theta import read_puffin_file_ndvs
from bodo.tests.iceberg_database_helpers.metadata_utils import (
    get_metadata_field,
    get_metadata_path,
)
from bodo.utils.utils import run_rank0


@numba.njit
def get_statistics_ndvs(puffin_file_name, iceberg_schema):
    return read_puffin_file_ndvs(puffin_file_name, iceberg_schema)


# Note: We mark df as distributed but for testing we are only
# using 1 rank.
@bodo.jit(distributed=["df"])
def create_table_jit(df, table_name, conn, db_schema):
    df.to_sql(table_name, conn, db_schema, if_exists="replace")


@run_rank0
def check_ndv_metadata(
    warehouse_loc, db_schema, table_name, expected_ndvs, num_statistics=1
):
    """
    Check the NDV information found in the metadata file and return the path to the
    puffin file for further testing.
    """
    metadata_path = get_metadata_path(warehouse_loc, db_schema, table_name)
    statistics_lst = get_metadata_field(metadata_path, "statistics")
    assert len(statistics_lst) == num_statistics, (
        f"Expected {num_statistics} statistics file(s)"
    )
    if num_statistics > 1:
        # Need to fetch the latest snapshot and iterate through them to select the match statistics file
        latest_snapshot_id = get_metadata_field(metadata_path, "current-snapshot-id")
        for entry in statistics_lst:
            if entry["snapshot-id"] == latest_snapshot_id:
                statistics = entry
                break
    elif num_statistics == 0:
        assert len(statistics_lst) == 0, (
            "Found a statistics file when none should exist"
        )
        return None
    else:
        statistics = statistics_lst[0]
    # Check the NDVs match expectations
    blob_metadata = statistics["blob-metadata"]
    seen_fields = set()
    for blob in blob_metadata:
        fields = blob["fields"]
        assert len(fields) == 1, "Expected only one field in the puffin file"
        field = fields[0]
        properties = blob["properties"]
        ndv = properties["ndv"]
        assert field in expected_ndvs, "Unexpected field ID blob"
        assert ndv == expected_ndvs[field], f"Incorrect NDV for blob {field}"
        seen_fields.add(field)
    assert len(seen_fields) == len(expected_ndvs), (
        "An expected column didn't have a theta sketch"
    )
    # Check the puffin file exists, can be read, and the theta sketch is correct.
    return statistics["statistics-path"]


@run_rank0
def check_no_statistics_file(warehouse_loc, db_schema, table_name):
    import json

    metadata_path = get_metadata_path(warehouse_loc, db_schema, table_name)
    with open(metadata_path) as f:
        metadata = json.load(f)
    assert "statistics" not in metadata or len(metadata["statistics"]) == 0, (
        "Found a statistics file when none should exist"
    )
