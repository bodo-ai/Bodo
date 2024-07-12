import pyarrow as pa
import pytest

from bodo.io.iceberg import (
    ICEBERG_FIELD_ID_MD_KEY,
    add_iceberg_field_id_md_to_pa_schema,
    with_iceberg_field_id_md,
)
from bodo.tests.utils import pytest_mark_one_rank

pytestmark = pytest.mark.iceberg


@pytest_mark_one_rank
@pytest.mark.parametrize(
    "field,next_field_id,expected_out",
    [
        pytest.param(
            pa.field("A", pa.int64(), False),
            1,
            pa.field("A", pa.int64(), False, metadata={ICEBERG_FIELD_ID_MD_KEY: "1"}),
            id="simple_int64",
        ),
        pytest.param(
            pa.field("LIST_OF_MAPS", pa.list_(pa.map_(pa.int64(), pa.float64()))),
            5,
            pa.field(
                "LIST_OF_MAPS",
                pa.list_(
                    pa.field(
                        "item",
                        pa.map_(
                            pa.field(
                                "key",
                                pa.int64(),
                                False,
                                metadata={ICEBERG_FIELD_ID_MD_KEY: "7"},
                            ),
                            pa.field(
                                "value",
                                pa.float64(),
                                metadata={ICEBERG_FIELD_ID_MD_KEY: "8"},
                            ),
                        ),
                        metadata={ICEBERG_FIELD_ID_MD_KEY: "6"},
                    ),
                ),
                metadata={ICEBERG_FIELD_ID_MD_KEY: "5"},
            ),
            id="list_of_maps",
        ),
        pytest.param(
            pa.field(
                "STRUCT_OF_LIST_AND_DECIMAL",
                pa.struct(
                    [
                        pa.field("p1", pa.list_(pa.string()), False),
                        pa.field("p2", pa.decimal128(38, 0)),
                    ]
                ),
            ),
            2,
            pa.field(
                "STRUCT_OF_LIST_AND_DECIMAL",
                pa.struct(
                    [
                        pa.field(
                            "p1",
                            pa.list_(
                                pa.field(
                                    "item",
                                    pa.string(),
                                    metadata={ICEBERG_FIELD_ID_MD_KEY: "4"},
                                )
                            ),
                            False,
                            metadata={ICEBERG_FIELD_ID_MD_KEY: "3"},
                        ),
                        pa.field(
                            "p2",
                            pa.decimal128(38, 0),
                            metadata={ICEBERG_FIELD_ID_MD_KEY: "5"},
                        ),
                    ]
                ),
                metadata={ICEBERG_FIELD_ID_MD_KEY: "2"},
            ),
            id="struct_of_list_and_decimal",
        ),
        pytest.param(
            pa.field("PRICE", pa.large_list(pa.float64())),
            4,
            pa.field(
                "PRICE",
                pa.large_list(
                    pa.field(
                        "item", pa.float64(), metadata={ICEBERG_FIELD_ID_MD_KEY: "5"}
                    )
                ),
                metadata={ICEBERG_FIELD_ID_MD_KEY: "4"},
            ),
            id="large_list",
        ),
        pytest.param(
            pa.field("ACTORS", pa.list_(pa.string(), list_size=5)),
            1,
            pa.field(
                "ACTORS",
                pa.list_(
                    pa.field(
                        "item", pa.string(), metadata={ICEBERG_FIELD_ID_MD_KEY: "2"}
                    ),
                    list_size=5,
                ),
                metadata={ICEBERG_FIELD_ID_MD_KEY: "1"},
            ),
            id="fixed_size_list",
        ),
    ],
)
def test_with_iceberg_field_id_md(
    field: pa.Field, next_field_id: int, expected_out: pa.Field
):
    """
    Test the with_iceberg_field_id_md utility function.
    """
    out, next_field_id = with_iceberg_field_id_md(field, next_field_id)
    assert expected_out.equals(out, check_metadata=True)


@pytest_mark_one_rank
@pytest.mark.parametrize(
    "schema,ref_schema,expected_out",
    [
        (
            pa.schema(
                [
                    # Add metadata which should be overwritten and
                    # some that should be retained.
                    pa.field(
                        "A",
                        pa.int32(),
                        metadata={
                            ICEBERG_FIELD_ID_MD_KEY: "54",
                            "SOMETHING_ELSE": "YES",
                        },
                    ),
                    pa.field("LIST_C", pa.list_(pa.map_(pa.int64(), pa.float64()))),
                    pa.field(
                        "STRUCT_D",
                        pa.struct(
                            [
                                pa.field("p1", pa.list_(pa.string())),
                                pa.field("p2", pa.decimal128(38, 0)),
                            ]
                        ),
                    ),
                ]
            ),
            pa.schema(
                [
                    pa.field(
                        "A",
                        pa.int32(),
                        metadata={ICEBERG_FIELD_ID_MD_KEY: "1"},
                    ),
                    pa.field(
                        "LIST_C",
                        pa.large_list(
                            pa.field(
                                "item",
                                pa.map_(
                                    pa.field(
                                        "key",
                                        pa.int64(),
                                        metadata={ICEBERG_FIELD_ID_MD_KEY: "4"},
                                        nullable=False,
                                    ),
                                    pa.field(
                                        "value",
                                        pa.float64(),
                                        metadata={ICEBERG_FIELD_ID_MD_KEY: "7"},
                                    ),
                                ),
                                metadata={ICEBERG_FIELD_ID_MD_KEY: "3"},
                            )
                        ),
                        metadata={ICEBERG_FIELD_ID_MD_KEY: "5"},
                    ),
                    pa.field(
                        "B", pa.large_string(), metadata={ICEBERG_FIELD_ID_MD_KEY: "2"}
                    ),
                    pa.field(
                        "STRUCT_D",
                        pa.struct(
                            [
                                pa.field(
                                    "p1",
                                    pa.large_list(
                                        pa.field(
                                            "item",
                                            pa.large_string(),
                                            metadata={ICEBERG_FIELD_ID_MD_KEY: "10"},
                                        )
                                    ),
                                    metadata={ICEBERG_FIELD_ID_MD_KEY: "8"},
                                ),
                                pa.field(
                                    "p2",
                                    pa.decimal128(38, 0),
                                    metadata={ICEBERG_FIELD_ID_MD_KEY: "9"},
                                ),
                            ]
                        ),
                        metadata={ICEBERG_FIELD_ID_MD_KEY: "6"},
                    ),
                ]
            ),
            pa.schema(
                [
                    pa.field(
                        "A",
                        pa.int32(),
                        metadata={
                            ICEBERG_FIELD_ID_MD_KEY: "1",
                            "SOMETHING_ELSE": "YES",
                        },
                    ),
                    pa.field(
                        "LIST_C",
                        pa.large_list(
                            pa.field(
                                "item",
                                pa.map_(
                                    pa.field(
                                        "key",
                                        pa.int64(),
                                        metadata={ICEBERG_FIELD_ID_MD_KEY: "4"},
                                        nullable=False,
                                    ),
                                    pa.field(
                                        "value",
                                        pa.float64(),
                                        metadata={ICEBERG_FIELD_ID_MD_KEY: "7"},
                                    ),
                                ),
                                metadata={ICEBERG_FIELD_ID_MD_KEY: "3"},
                            )
                        ),
                        metadata={ICEBERG_FIELD_ID_MD_KEY: "5"},
                    ),
                    pa.field(
                        "STRUCT_D",
                        pa.struct(
                            [
                                pa.field(
                                    "p1",
                                    pa.large_list(
                                        pa.field(
                                            "item",
                                            pa.large_string(),
                                            metadata={ICEBERG_FIELD_ID_MD_KEY: "10"},
                                        )
                                    ),
                                    metadata={ICEBERG_FIELD_ID_MD_KEY: "8"},
                                ),
                                pa.field(
                                    "p2",
                                    pa.decimal128(38, 0),
                                    metadata={ICEBERG_FIELD_ID_MD_KEY: "9"},
                                ),
                            ]
                        ),
                        metadata={ICEBERG_FIELD_ID_MD_KEY: "6"},
                    ),
                ]
            ),
        )
    ],
)
def test_add_iceberg_field_id_md_to_pa_schema_with_ref(
    schema, ref_schema, expected_out
):
    """
    Test that add_iceberg_field_id_md_to_pa_schema works as expected
    when a reference schema is provided.
    """
    out = add_iceberg_field_id_md_to_pa_schema(schema, ref_schema)
    assert expected_out.equals(out, check_metadata=True)


@pytest_mark_one_rank
@pytest.mark.parametrize(
    "schema,expected_out",
    [
        (
            pa.schema(
                [
                    pa.field("A", pa.int32()),
                    pa.field("LIST_C", pa.list_(pa.map_(pa.int64(), pa.float64()))),
                    pa.field("B", pa.string()),
                    pa.field(
                        "STRUCT_D",
                        pa.struct(
                            [
                                pa.field("p1", pa.list_(pa.string())),
                                pa.field("p2", pa.decimal128(38, 0)),
                            ]
                        ),
                    ),
                ]
            ),
            pa.schema(
                [
                    pa.field("A", pa.int32(), metadata={ICEBERG_FIELD_ID_MD_KEY: "1"}),
                    pa.field(
                        "LIST_C",
                        pa.large_list(
                            pa.field(
                                "element",
                                pa.map_(
                                    pa.field(
                                        "key",
                                        pa.int64(),
                                        metadata={ICEBERG_FIELD_ID_MD_KEY: "6"},
                                        nullable=False,
                                    ),
                                    pa.field(
                                        "value",
                                        pa.float64(),
                                        metadata={ICEBERG_FIELD_ID_MD_KEY: "7"},
                                    ),
                                ),
                                metadata={ICEBERG_FIELD_ID_MD_KEY: "5"},
                            )
                        ),
                        metadata={ICEBERG_FIELD_ID_MD_KEY: "2"},
                    ),
                    pa.field(
                        "B", pa.large_string(), metadata={ICEBERG_FIELD_ID_MD_KEY: "3"}
                    ),
                    pa.field(
                        "STRUCT_D",
                        pa.struct(
                            [
                                pa.field(
                                    "p1",
                                    pa.large_list(
                                        pa.field(
                                            "element",
                                            pa.large_string(),
                                            metadata={ICEBERG_FIELD_ID_MD_KEY: "10"},
                                        )
                                    ),
                                    metadata={ICEBERG_FIELD_ID_MD_KEY: "8"},
                                ),
                                pa.field(
                                    "p2",
                                    pa.decimal128(38, 0),
                                    metadata={ICEBERG_FIELD_ID_MD_KEY: "9"},
                                ),
                            ]
                        ),
                        metadata={ICEBERG_FIELD_ID_MD_KEY: "4"},
                    ),
                ]
            ),
        )
    ],
)
def test_add_iceberg_field_id_md_to_pa_schema_without_ref(schema, expected_out):
    """
    Test that add_iceberg_field_id_md_to_pa_schema works as expected
    when a reference schema is not provided, i.e. we call the Iceberg
    Java Library to get the init field ID assignments.
    """
    out = add_iceberg_field_id_md_to_pa_schema(schema)
    assert expected_out.equals(out, check_metadata=True)


@pytest_mark_one_rank
@pytest.mark.parametrize(
    "schema,expected_out",
    [
        pytest.param(
            pa.schema(
                [
                    pa.field(
                        "A",
                        pa.uint8(),
                        metadata={ICEBERG_FIELD_ID_MD_KEY: "1"},
                    ),
                    pa.field(
                        "B",
                        pa.uint16(),
                        metadata={ICEBERG_FIELD_ID_MD_KEY: "2"},
                    ),
                    pa.field(
                        "C",
                        pa.int8(),
                        metadata={ICEBERG_FIELD_ID_MD_KEY: "3"},
                    ),
                    pa.field(
                        "D",
                        pa.int16(),
                        metadata={ICEBERG_FIELD_ID_MD_KEY: "4"},
                    ),
                    pa.field(
                        "E",
                        pa.uint32(),
                        metadata={ICEBERG_FIELD_ID_MD_KEY: "5"},
                    ),
                ]
            ),
            '{"type":"struct","schema-id":0,"fields":[{"id":1,"name":"A","required":false,"type":"int"},{"id":2,"name":"B","required":false,"type":"int"},{"id":3,"name":"C","required":false,"type":"int"},{"id":4,"name":"D","required":false,"type":"int"},{"id":5,"name":"E","required":false,"type":"long"}]}',
            id="int_upcast",
        )
    ],
)
def test_pyarrow_to_iceberg_schema(schema, expected_out):
    from bodo_iceberg_connector.schema_helper import pyarrow_to_iceberg_schema_str

    out_schema = pyarrow_to_iceberg_schema_str(schema)
    assert out_schema == expected_out


@pytest_mark_one_rank
@pytest.mark.parametrize(
    "schema, expected_err",
    [
        pytest.param(
            pa.schema(
                [
                    pa.field(
                        "A",
                        pa.null(),
                        metadata={ICEBERG_FIELD_ID_MD_KEY: "1"},
                    ),
                ]
            ),
            "Currently Cant Handle Purely Null Fields",
            id="null",
        ),
        pytest.param(
            pa.schema(
                [
                    pa.field(
                        "A",
                        pa.uint64(),
                        metadata={ICEBERG_FIELD_ID_MD_KEY: "1"},
                    ),
                ]
            ),
            "Unsupported PyArrow DataType: uint64",
            id="uint64",
        ),
    ],
)
def test_pyarrow_to_iceberg_schema_unsupported(schema, expected_err):
    from bodo_iceberg_connector.errors import IcebergError
    from bodo_iceberg_connector.schema_helper import pyarrow_to_iceberg_schema_str

    with pytest.raises(IcebergError, match=expected_err):
        out_schema = pyarrow_to_iceberg_schema_str(schema)
