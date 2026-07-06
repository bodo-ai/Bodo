import pyarrow as pa
import pyiceberg.expressions as pie
import pytest
from pyiceberg.schema import DoubleType, LongType, NestedField, Schema
from pyiceberg.types import DoubleType, LongType, NestedField

from bodo.tests.utils import pytest_mark_one_rank


@pytest_mark_one_rank
@pytest.mark.parametrize(
    "filter_expr, expected_output",
    [
        (
            pie.And(
                left=pie.LessThan(
                    term=pie.Reference(name="L_QUANTITY"),
                    literal=24,
                ),
                right=pie.And(
                    left=pie.GreaterThanOrEqual(
                        term=pie.Reference(name="L_DISCOUNT"),
                        literal=0.05,
                    ),
                    right=pie.LessThanOrEqual(
                        term=pie.Reference(name="L_DISCOUNT"),
                        literal=0.07,
                    ),
                ),
            ),
            (
                "(pc.field('{L_QUANTITY}') < f0) & ((pc.field('{L_DISCOUNT}') >= f1) & (pc.field('{L_DISCOUNT}') <= f2))",
                [
                    ("f0", pa.scalar(24, type=pa.int64())),
                    ("f1", pa.scalar(0.05, type=pa.float64())),
                    ("f2", pa.scalar(0.07, type=pa.float64())),
                ],
            ),
        ),
    ],
)
def test_pyiceberg_filter_to_pyarrow_format_str(filter_expr, expected_output):
    """Test the conversion of PyIceberg filter expressions to PyArrow format strings and scalars."""
    from bodo.io.iceberg.common import (
        pyiceberg_filter_to_pyarrow_format_str_and_scalars,
    )

    schema = Schema(
        NestedField(
            field_id=1,
            name="L_QUANTITY",
            field_type=LongType(),
            required=False,
        ),
        NestedField(
            field_id=2,
            name="L_DISCOUNT",
            field_type=DoubleType(),
            required=False,
        ),
    )

    expr, scalars = pyiceberg_filter_to_pyarrow_format_str_and_scalars(
        filter_expr, schema, True
    )

    assert expr == expected_output[0]
    assert scalars == expected_output[1]
