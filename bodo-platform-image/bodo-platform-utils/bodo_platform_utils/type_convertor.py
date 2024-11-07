import pandas as pd
import numpy as np
import base64
import datetime
import pyarrow as pa
import decimal
from enum import StrEnum
import typing as pt
from dataclasses import dataclass, field, asdict
import bodo


class JDBCType(StrEnum):
    """
    Enum for JDBC types supported by Bodo
    Uesd for converting JDBC params into Bodo params
    and Bodo array types to JDBC ones

    Note: This will be converted to the string in json.dumps
    """

    INT8 = "INT8"
    INT32 = "INT32"
    INT16 = "INT16"
    INT64 = "INT64"

    FLOAT32 = "FLOAT32"
    FLOAT64 = "FLOAT64"
    # Double is the same as FLOAT64. Do we need both?
    DOUBLE = "DOUBLE"

    BINARY = "BINARY"
    BOOL = "BOOL"
    STRING = "STRING"
    DECIMAL = "DECIMAL"
    NULL = "NULL"

    TIMESTAMP_NTZ = "TIMESTAMP_NTZ"
    TIMESTAMP_LTZ = "TIMESTAMP_LTZ"
    TIME = "TIME"
    DATE = "DATE"

    ARRAY = "ARRAY"
    MAP = "MAP"
    STRUCT = "STRUCT"


class Mapping(pt.TypedDict):
    type: str
    value: pt.Any


def get_value_for_type(mapping: Mapping) -> pt.Any:
    try:
        type = JDBCType(mapping["type"])
    except ValueError:
        raise Exception(f"Unsupported type: {mapping['type']}")

    value = mapping["value"]

    if value is None:
        return None
    if type == JDBCType.INT64:
        return np.int64(value)
    elif type == JDBCType.BINARY:
        if value == "Z":
            return None
        return base64.b16decode(value)
    elif type == JDBCType.BOOL:
        return np.bool_(value == "true")
    elif type == JDBCType.STRING:
        return str(value)
    elif type == JDBCType.DATE:
        return datetime.datetime.strptime(value, "%Y-%m-%d")
    elif type == JDBCType.DOUBLE:
        return np.float64(value)
    elif type == JDBCType.DECIMAL:
        return pa.scalar(decimal.Decimal(value.strip()))
    elif type == JDBCType.FLOAT32:
        return np.float32(value)
    elif type == JDBCType.FLOAT64:
        return np.float64(value)
    elif type == JDBCType.INT32:
        return np.int32(value)
    elif type == JDBCType.INT16:
        return np.int16(value)
    elif type == JDBCType.INT8:
        return np.int8(value)
    elif type == JDBCType.TIMESTAMP_NTZ:
        return pd.Timestamp(value)
    elif type == JDBCType.TIMESTAMP_LTZ:
        return pd.Timestamp(value)
    elif type == JDBCType.TIME:
        time_obj = datetime.datetime.strptime(value, "%H:%M:%S")
        time_scalar = pa.scalar(time_obj.time(), type=pa.time64("us"))
        return pa.scalar(time_scalar)
    else:
        raise Exception(f"Unhandled known value type: {type}")


@dataclass
class ResultMetadata:
    name: str
    type: JDBCType
    nullable: bool
    precision: pt.Optional[int] = None
    scale: pt.Optional[int] = None
    additional_info: pt.Dict = field(default_factory=dict)


def unit_to_scale(unit: str) -> int:
    """Convert time / timestamp unit string to scale value for JDBC metadata"""
    match unit.lower():
        case "s":
            return 0
        case "ms":
            return 3
        case "us":
            return 6
        case "ns":
            return 9
        case _:
            raise ValueError(f"Unsupported unit {unit}")


def scale_to_unit(scale: int) -> str:
    """Extract the time unit str from a scale value"""
    match scale:
        case 0:
            return "s"
        case 3:
            return "ms"
        case 6:
            return "us"
        case 9:
            return "ns"
        case v:
            raise ValueError(f"Unsupported scale value {v}")


def parse_output_types(output: "pd.DataFrame") -> pt.List[pt.Dict]:
    """
    Parse the output types of the query result
    Args:
        output: Query result dataframe
    Return:
        List of dictionaries containing the column name, type and nullable status
    """

    def _get_col_type(col_name: str, col_dtype) -> ResultMetadata:
        """
        Inner function to get the column type for specific columns

        Returns:
            Tuple of
                - Column type
                - Nullable status
                - Dictionary of additional type information. For example, timezone
                  for timestamp with timezone columns. Otherwise None
        """

        # Special handling for object type
        # TODO: Remove this after removing all Bodo object boxing
        if col_dtype == np.dtype("O"):
            not_na_vals = output[col_name][output[col_name].notna()]
            if len(not_na_vals):
                first_val = not_na_vals.iloc[0]
            else:
                # if value is unknown cast it to string
                return ResultMetadata(col_name, JDBCType.STRING, True)
            if isinstance(first_val, str):
                return ResultMetadata(col_name, JDBCType.STRING, True)
            if isinstance(first_val, bytes):
                return ResultMetadata(col_name, JDBCType.BINARY, True)
            if isinstance(first_val, datetime.date):
                return ResultMetadata(col_name, JDBCType.DATE, True)
            if isinstance(first_val, (datetime.time, bodo.hiframes.time_ext.Time)):
                if isinstance(first_val, datetime.time):
                    unit = "s" if first_val.microsecond == 0 else "us"
                    scale = unit_to_scale(unit)
                else:
                    scale = first_val.precision
                    unit = scale_to_unit(scale)

                return ResultMetadata(
                    col_name,
                    JDBCType.TIME,
                    True,
                    precision=0,
                    scale=scale,
                    additional_info={"unit": unit},
                )

            if isinstance(first_val, (datetime.datetime, np.datetime64)):
                return ResultMetadata(
                    col_name,
                    JDBCType.TIMESTAMP_NTZ,
                    True,
                    precision=0,
                    scale=9,
                    additional_info={"unit": "ns"},
                )
            raise ValueError(
                f"Unsupported object type for column {col_name}: {type(first_val)}"
            )

        # Special PyArrow-based Arrays
        if isinstance(col_dtype, pd.ArrowDtype):
            pa_type = col_dtype.pyarrow_dtype
            if pa.types.is_boolean(pa_type):
                return ResultMetadata(col_name, JDBCType.BOOL, True)
            if pa.types.is_int8(pa_type):
                return ResultMetadata(col_name, JDBCType.INT8, True)
            if pa.types.is_int16(pa_type):
                return ResultMetadata(col_name, JDBCType.INT16, True)
            if pa.types.is_int32(pa_type):
                return ResultMetadata(col_name, JDBCType.INT32, True)
            if pa.types.is_int64(pa_type):
                return ResultMetadata(col_name, JDBCType.INT64, True)
            if pa.types.is_float32(pa_type):
                return ResultMetadata(col_name, JDBCType.FLOAT32, True)
            if pa.types.is_float64(pa_type):
                return ResultMetadata(col_name, JDBCType.FLOAT64, True)
            if pa.types.is_date32(pa_type):
                return ResultMetadata(col_name, JDBCType.DATE, True)
            if pa.types.is_decimal128(pa_type):
                return ResultMetadata(
                    col_name,
                    JDBCType.DECIMAL,
                    True,
                    precision=pa_type.precision,
                    scale=pa_type.scale,
                )
            if pa.types.is_large_binary(pa_type) or pa.types.is_binary(pa_type):
                return ResultMetadata(col_name, JDBCType.BINARY, True)
            if pa.types.is_large_string(pa_type) or pa.types.is_string(pa_type):
                return ResultMetadata(col_name, JDBCType.STRING, True)
            if pa.types.is_time32(pa_type) or pa.types.is_time64(pa_type):
                return ResultMetadata(
                    col_name,
                    JDBCType.TIME,
                    True,
                    precision=0,
                    scale=unit_to_scale(pa_type.unit),
                    additional_info={"unit": pa_type.unit},
                )
            if pa.types.is_null(pa_type):
                return ResultMetadata(col_name, JDBCType.NULL, True)
            if pa.types.is_large_list(pa_type) or pa.types.is_list(pa_type):
                return ResultMetadata(col_name, JDBCType.ARRAY, True)
            if pa.types.is_map(pa_type):
                return ResultMetadata(col_name, JDBCType.MAP, True)
            if pa.types.is_struct(pa_type):
                return ResultMetadata(col_name, JDBCType.STRUCT, True)
            if pa.types.is_timestamp(pa_type):
                if pa_type.tz is None:
                    return ResultMetadata(
                        col_name,
                        JDBCType.TIMESTAMP_NTZ,
                        True,
                        precision=0,
                        scale=unit_to_scale(pa_type.unit.upper()),
                        additional_info={"unit": pa_type.unit},
                    )
                else:
                    return ResultMetadata(
                        col_name,
                        JDBCType.TIMESTAMP_LTZ,
                        True,
                        scale=unit_to_scale(pa_type.unit.upper()),
                        additional_info={
                            "unit": pa_type.unit,
                            "timezone": str(pa_type.tz),
                        },
                    )

            raise ValueError(
                f"Unsupported PyArrow dtype {pa_type} for column {col_name}"
            )

        # Numpy and Nullable dtypes
        # Integers
        if col_dtype == np.int8:
            return ResultMetadata(col_name, JDBCType.INT8, False)
        if col_dtype == np.int16:
            return ResultMetadata(col_name, JDBCType.INT16, False)
        if col_dtype == np.int32:
            return ResultMetadata(col_name, JDBCType.INT32, False)
        if col_dtype == np.int64:
            return ResultMetadata(col_name, JDBCType.INT64, False)
        if isinstance(col_dtype, pd.Int8Dtype):
            return ResultMetadata(col_name, JDBCType.INT8, True)
        if isinstance(col_dtype, pd.Int16Dtype):
            return ResultMetadata(col_name, JDBCType.INT16, True)
        if isinstance(col_dtype, pd.Int32Dtype):
            return ResultMetadata(col_name, JDBCType.INT32, True)
        if isinstance(col_dtype, pd.Int64Dtype):
            return ResultMetadata(col_name, JDBCType.INT64, True)

        # Floats
        if col_dtype == np.float32:
            return ResultMetadata(col_name, JDBCType.FLOAT32, False)
        if col_dtype == np.float64:
            return ResultMetadata(col_name, JDBCType.FLOAT64, False)
        if isinstance(col_dtype, pd.Float32Dtype):
            return ResultMetadata(col_name, JDBCType.FLOAT32, True)
        if isinstance(col_dtype, pd.Float64Dtype):
            return ResultMetadata(col_name, JDBCType.FLOAT64, True)

        # Booleans
        if col_dtype == np.bool_:
            return ResultMetadata(col_name, JDBCType.BOOL, False)
        if isinstance(col_dtype, pd.BooleanDtype):
            return ResultMetadata(col_name, JDBCType.BOOL, True)

        # Datetime
        if isinstance(col_dtype, np.dtype) and np.issubdtype(col_dtype, np.datetime64):
            return ResultMetadata(
                col_name,
                JDBCType.TIMESTAMP_NTZ,
                True,
                precision=0,
                scale=9,
                additional_info={"unit": "ns"},
            )
        if isinstance(col_dtype, pd.DatetimeTZDtype):
            return ResultMetadata(
                col_name,
                JDBCType.TIMESTAMP_LTZ,
                True,
                scale=unit_to_scale(col_dtype.unit),
                additional_info={"unit": col_dtype.unit, "timezone": str(col_dtype.tz)},
            )

        # Strings
        if isinstance(col_dtype, pd.StringDtype):
            return ResultMetadata(col_name, JDBCType.STRING, True)

        raise ValueError(f"Unsupported dtype {col_dtype} for column {col_name}")

    return [
        asdict(_get_col_type(col_name, col_type))
        for col_name, col_type in output.dtypes.items()
    ]
