"""
Helpers for testing functionality involving Iceberg metadata.
"""

import json
import os
import re


def get_metadata_path(warehouse_loc: str, db_schema: str, table_name: str):
    """
    Convert the iceberg path information to the expected location of the metadata file.
    """
    directory = os.path.join(warehouse_loc, db_schema, table_name, "metadata")
    # We need to find all files that match v{i}.metadata.json and select the highest i
    files = os.listdir(directory)
    pat = re.compile("v(\d+).metadata.json")
    candidate_files = [f for f in files if pat.match(f)]
    if len(candidate_files) == 0:
        raise ValueError(f"No metadata files found in {directory}")
    max_version = max([int(pat.match(f).group(1)) for f in candidate_files])
    return os.path.join(directory, f"v{max_version}.metadata.json")


def get_metadata_field(metadata_path: str, field: str):
    """
    Read the metadata file as a json and read the given
    top level field.
    """
    with open(metadata_path) as f:
        metadata = json.load(f)
    return metadata[field]
