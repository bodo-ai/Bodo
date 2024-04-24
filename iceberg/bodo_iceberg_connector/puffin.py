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
    sourceSnapshotId: int
    sourceSnapshotSequenceNumber: int
    fields: list[int]
    properties: dict[str, str]


@dataclass
class StatisticsFile:
    """
    Python equivalent of the StatisticsFile interface in Java.
    This is used for passing information to the Java connector via JSON.
    """

    snapshotId: int
    path: str
    fileSizeInBytes: int
    fileFooterSizeInBytes: int
    blobMetadata: list[BlobMetadata]

    @staticmethod
    def empty():
        return StatisticsFile(-1, "", -1, -1, [])
