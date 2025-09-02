import pyarrow as pa
import pytest

import bodo.io.parquet_pio


def test_pa_schema_reduction():
    # Same schema
    schema_a = pa.schema([("a", pa.int32())])
    schema_b = pa.schema([("a", pa.int32())])
    res, _ = bodo.io.parquet_pio.pa_schema_unify_reduction(
        (schema_a, 1), (schema_b, 1), None
    )
    assert res == schema_a

    # Different, but compatible schema
    schema_a = pa.schema([("a", pa.int32())])
    schema_b = pa.schema([("b", pa.int32())])
    res, _ = bodo.io.parquet_pio.pa_schema_unify_reduction(
        (schema_a, 1), (schema_b, 1), None
    )
    assert res == pa.schema([("a", pa.int32()), ("b", pa.int32())])

    # Different, incompatible schema
    schema_a = pa.schema([("a", pa.int32())])
    schema_b = pa.schema([("a", pa.int64())])
    with pytest.raises(pa.lib.ArrowTypeError):
        res, _ = bodo.io.parquet_pio.pa_schema_unify_reduction(
            (schema_a, 1), (schema_b, 1), None
        )

    # Different, incompatible schema, but a row count of 0 for one of the inputs
    schema_a = pa.schema([("a", pa.int32())])
    schema_b = pa.schema([("a", pa.int64())])
    res, _ = bodo.io.parquet_pio.pa_schema_unify_reduction(
        (schema_a, 1), (schema_b, 0), None
    )
    assert res == schema_a
    res, _ = bodo.io.parquet_pio.pa_schema_unify_reduction(
        (schema_a, 0), (schema_b, 1), None
    )
    assert res == schema_b
