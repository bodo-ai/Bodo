import pandas as pd
import numpy as np
import base64
from datetime import datetime
import pyarrow as pa
import decimal


def get_value_for_type(mapping):
    type = mapping["type"]
    value = mapping["value"]
    if value is None:
        return None
    if type == "INT64":
        return np.int64(value)
    elif type == "BINARY":
        if value == "Z":
            return None
        return base64.b16decode(value)
    elif type == "BOOL8":
        return np.bool_(value == "true")
    elif type == "STRING":
        return str(value)
    elif type == "DATE":
        return datetime.strptime(value, "%Y-%m-%d")
    elif type == "DOUBLE":
        return np.float64(value)
    elif type == "DECIMAL":
        return pa.scalar(decimal.Decimal(value.strip()))
    elif type == "FLOAT32":
        return np.float32(value)
    elif type == "FLOAT64":
        return np.float64(value)
    elif type == "INT32":
        return np.int32(value)
    elif type == "INT16":
        return np.int16(value)
    elif type == "INT8":
        return np.int8(value)
    elif type == "TIMESTAMP_NTZ":
        return pd.Timestamp(value)
    elif type == "TIMESTAMP_LTZ":
        return pd.Timestamp(value)
    elif type == "TIME":
        time_obj = datetime.strptime(value, "%H:%M:%S")
        time_scalar = pa.scalar(time_obj.time(), type=pa.time64('us'))
        return pa.scalar(time_scalar)
    else:
        raise Exception(f"Unsupported type: {type}")

