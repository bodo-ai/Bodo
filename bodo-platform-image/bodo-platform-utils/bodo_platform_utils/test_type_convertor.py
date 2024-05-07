import datetime as datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import decimal
from bodo_platform_utils.type_convertor import get_value_for_type


class TestTypeConverter:
    def test_bool8_conversion(self):
        assert get_value_for_type({"type": "BOOL8", "value": "true"}) == np.bool_(True)
        assert type(get_value_for_type({"type": "BOOL8", "value": "true"})) == np.bool_

    def test_bool8_negative_conversion(self):
        assert get_value_for_type({"type": "BOOL8", "value": "false"}) == np.bool_(False)
        assert type(get_value_for_type({"type": "BOOL8", "value": "true"})) == np.bool_

    def test_bool8_none_conversion(self):
        assert get_value_for_type({"type": "BOOL8", "value": None}) == None

    def test_int64_conversion(self):
        assert get_value_for_type({"type": "INT64", "value": "123"}) == np.int64(123)
        assert type(get_value_for_type({"type": "INT64", "value": "123"})) == np.int64

    def test_empty_binary_conversion(self):
        assert get_value_for_type({"type": "BINARY", "value": "Z"}) is None

    def test_string_conversion(self):
        assert get_value_for_type({"type": "STRING", "value": "test"}) == 'test'
        assert type(get_value_for_type({"type": "STRING", "value": "test"})) == str

    def test_empty_string_conversion(self):
        assert get_value_for_type({"type": "STRING", "value": ""}) == ''
        assert type(get_value_for_type({"type": "STRING", "value": ""})) == str

    def test_none_string_conversion(self):
        assert get_value_for_type({"type": "STRING", "value": None}) is None


    def test_date_conversion(self):
        assert get_value_for_type({"type": "DATE", "value": "2021-01-01"}) == datetime.datetime(2021, 1, 1, 0, 0)

    def test_double_conversion(self):
        assert get_value_for_type({"type": "DOUBLE", "value": "123.45"}) == np.float64(123.45)
        assert type(get_value_for_type({"type": "DOUBLE", "value": "123.45"})) == np.float64

    def test_float32_conversion(self):
        assert get_value_for_type({"type": "FLOAT32", "value": "12.45"}) == np.float32(12.45)
        assert type(get_value_for_type({"type": "FLOAT32", "value": "12.45"})) == np.float32

    def test_float64_conversion(self):
        assert get_value_for_type({"type": "FLOAT64", "value": "12.45"}) == np.float64(12.45)
        assert type(get_value_for_type({"type": "FLOAT64", "value": "12.45"})) == np.float64

    def test_timestamp_ntz_conversion(self):
        assert get_value_for_type({"type": "TIMESTAMP_NTZ", "value":  "2024-04-29 10:41:28.584"}) == \
               pd.Timestamp('2024-04-29 10:41:28.584000')
        assert type(get_value_for_type({"type": "TIMESTAMP_NTZ", "value":  "2024-04-29 10:41:28.584"})) == pd.Timestamp

    def test_timestamp_ltz_conversion(self):
        assert get_value_for_type({"type": "TIMESTAMP_LTZ", "value": "2021-01-01"}) == pd.Timestamp('2021-01-01')
        assert type(get_value_for_type({"type": "TIMESTAMP_LTZ", "value": "2021-01-01"})) == pd.Timestamp

    def test_int32_conversion(self):
        assert get_value_for_type({"type": "INT32", "value": "123"}) == np.int64(123)
        assert type(get_value_for_type({"type": "INT32", "value": "123"})) == np.int32

    def test_int16_conversion(self):
        assert get_value_for_type({"type": "INT16", "value": "1"}) == np.int16(1)
        assert type(get_value_for_type({"type": "INT16", "value": "1"})) == np.int16

    def test_int8_conversion(self):
        assert get_value_for_type({"type": "INT8", "value": "1"}) == np.int16(1)
        assert type(get_value_for_type({"type": "INT8", "value": "1"})) == np.int8


    def test_time_conversion(self):
        assert get_value_for_type({"type": "TIME", "value": "12:34:56"}) == pa.scalar(datetime.time(12, 34, 56))

    def test_decimal_conversion(self):
        assert get_value_for_type({"type": "DECIMAL", "value": "123.45 "}) == pa.scalar(decimal.Decimal('123.45'))
        assert type(get_value_for_type({"type": "DECIMAL", "value": "123.45 "})) == pa.lib.Decimal128Scalar
