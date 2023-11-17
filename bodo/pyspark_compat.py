# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""
PySpark monkey patches to fix issues related to our uses. Should be imported before any
PySpark use (BodoSQL and Iceberg tests currently).
"""

# Wrap in try/except to avoid making PySpark package a required dependency
try:
    import pyspark
    from pyspark.sql.types import (
        BooleanType,
        ByteType,
        DayTimeIntervalType,
        DoubleType,
        FloatType,
        IntegerType,
        LongType,
        ShortType,
        StringType,
        StructField,
        StructType,
        TimestampNTZType,
        TimestampType,
    )

    def _to_corrected_pandas_type(dt):
        """
        When converting Spark SQL records to Pandas `pandas.DataFrame`, the inferred data type
        may be wrong. This method gets the corrected data type for Pandas if that type may be
        inferred incorrectly.
        """
        import numpy as np

        if type(dt) == ByteType:
            return np.int8
        elif type(dt) == ShortType:
            return np.int16
        elif type(dt) == IntegerType:
            return np.int32
        elif type(dt) == LongType:
            return np.int64
        elif type(dt) == FloatType:
            return np.float32
        elif type(dt) == DoubleType:
            return np.float64
        elif type(dt) == BooleanType:
            return bool
        # Bodo change: use ns precision for datetime/timedelta to avoid Pandas error
        elif type(dt) == TimestampType:
            return np.dtype("datetime64[ns]")
        elif type(dt) == TimestampNTZType:
            return np.dtype("datetime64[ns]")
        elif type(dt) == DayTimeIntervalType:
            return np.dtype("timedelta64[ns]")
        else:
            return None

    if pyspark.__version__ != "3.4.1":
        import hashlib
        import inspect
        import warnings

        lines = inspect.getsource(
            pyspark.sql.pandas.conversion.PandasConversionMixin._to_corrected_pandas_type
        )
        if (
            hashlib.sha256(lines.encode()).hexdigest()
            != "e4721e01fdf9ec32cd9f7e8d5297faaf0ac6272705f234e7e6398505b797a4d4"
        ):  # pragma: no cover
            warnings.warn(
                "pyspark.sql.pandas.conversion.PandasConversionMixin._to_corrected_pandas_type has changed"
            )

    pyspark.sql.pandas.conversion.PandasConversionMixin._to_corrected_pandas_type = (
        _to_corrected_pandas_type
    )
except ImportError:
    pass
