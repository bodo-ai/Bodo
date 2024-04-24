# Copyright (C) 2024 Bodo Inc. All rights reserved.
"""
Python classes/functionality exposed for writing puffin files via
the Bodo Iceberg connector.
"""

from dataclasses import dataclass


@dataclass
class BlobMetadata:
    """
    Python equivalent of BlobMetadata in Java. This is used for passing
    information to the Java connector via JSON.
    """

    type: str
    source_snapshot_id: int
    source_snapshot_sequence_number: int
    fields: list[int]
    properties: dict[str, str]


@dataclass
class StatisticsFile:
    """
    Python equivalent of the StatisticsFile interface in Java.
    This is used for passing information to the Java connector via JSON.
    """

    snapshot_id: int
    path: str
    file_size_in_bytes: int
    file_footer_size_in_bytes: int
    blob_metadata: list[BlobMetadata]
