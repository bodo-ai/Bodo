# Copyright (C) 2022 Bodo Inc. All rights reserved.
import pyarrow._hdfs
import pyarrow.fs

from bodo.ext import (  # noqa
    _hdf5,
    arrow_cpp,
    csv_cpp,
    fsspec_reader,
    hdfs_reader,
    json_cpp,
    s3_reader,
)

from ._hdfs import HadoopFileSystem

# HadoopFileSystem is a class defined by Arrow in a Cython file (_hdfs.pyx).
# We need to monkey-patch it for now because it doesn't recognize "abfs://" and
# "abfss://" prefixes and incorrectly modifies URIs with those prefixes. The
# implementation of HadoopFileSystem where this change is required needs to be
# in Cython, so we have a modified copy of Arrow's _hdfs.pyx in bodo/io
pyarrow._hdfs.HadoopFileSystem = HadoopFileSystem
pyarrow.fs.HadoopFileSystem = HadoopFileSystem
