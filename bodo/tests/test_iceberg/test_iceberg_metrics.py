"""
File that contains tests for optional data file metrics.
For any primitive type this should produce a value count, null count,
minimum value, and maximum value. For nested data types it should compute
these metrics for its child arrays, except that it omits the min/max. The
exception to this is that struct arrays must match all fields.
"""

import datetime
import typing as pt
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from avro.datafile import DataFileReader
from avro.io import DatumReader

import bodo
from bodo.tests.iceberg_database_helpers.metadata_utils import (
    get_metadata_field,
    get_metadata_path,
)
from bodo.tests.utils import pytest_mark_one_rank

pytestmark = [pytest.mark.iceberg, pytest.mark.skip]


# Note: We mark df as distributed but for testing we are only
# using 1 rank.
@bodo.jit(distributed=["df"])
def create_table_jit(df, table_name, conn, db_schema):
    df.to_sql(table_name, conn, db_schema, if_exists="replace")


def update_field_mapping(
    field_id_map: dict[str, int], field: dict[str, pt.Any], prefix: str = ""
) -> None:
    """
    Update the field_id_map to include any arrays found within this field.
    This will map every field name to its field id.
    For primitive types this will just map the column name to field id.
    For structs we append `.<field_name>`. For lists will append `.element`
    and for maps we will append `.key` and `.value` since these aren't named.
    """
    field_name = f"{prefix}{field['name']}"
    field_id_map[field_name] = field["id"]
    type = field["type"]
    if isinstance(type, dict):
        if type["type"] == "struct":
            new_prefix = f"{field_name}."
            for child_field in type["fields"]:
                update_field_mapping(field_id_map, child_field, new_prefix)
        elif type["type"] == "list":
            # TODO: Support types nested within lists. This is supported in
            # runtime but not this testing infrastructure.
            new_name = f"{field_name}.element"
            field_id_map[new_name] = type["element-id"]
        elif type["type"] == "map":
            # TODO: Support types nested within maps. This is supported in
            # runtime but not this testing infrastructure.
            key_name = f"{field_name}.key"
            value_name = f"{field_name}.value"
            field_id_map[key_name] = type["key-id"]
            field_id_map[value_name] = type["value-id"]
        else:
            raise TypeError(f"Unsupported nested type: {type}")


def extract_schema_information(metadata_json_file: str) -> tuple[str, dict[str, int]]:
    """
    Extract the information about a schema that cannot be determined
    statically from the metadata file. This returns:
    - The path to the manifest list file.
    - A mapping from field name to field id.
    """
    schemas = get_metadata_field(metadata_json_file, "schemas")
    assert len(schemas) == 1, "There should only be 1 schema"
    schema = schemas[0]
    fields = schema["fields"]
    field_id_map = {}
    for field in fields:
        update_field_mapping(field_id_map, field)
    snapshots = get_metadata_field(metadata_json_file, "snapshots")
    assert len(snapshots) == 1, "There should only be 1 snapshot"
    snapshot = snapshots[0]
    # Extract the manifest list file
    manifest_list_path = snapshot["manifest-list"]
    return manifest_list_path, field_id_map


def get_manifest_file_path(manifest_list_path: str) -> str:
    """
    Get the path to the manifest file from the manifest list file.
    """
    manifest_file_paths = []
    with open(manifest_list_path, "rb") as f:
        reader = DataFileReader(f, DatumReader())
        for value in reader:
            manifest_file_paths.append(value["manifest_path"])
    assert len(manifest_file_paths) == 1, "There should only be 1 manifest file"
    return manifest_file_paths[0]


def get_manifest_file_metrics(
    manifest_file_path: str,
) -> tuple[dict[int, int], dict[int, int], dict[int, bytes], dict[int, bytes]]:
    """
    Extract the metrics we write about from the manifest file.
    """
    value_counts_list = []
    null_counts_list = []
    lower_bounds_list = []
    upper_bounds_list = []
    with open(manifest_file_path, "rb") as f:
        reader = DataFileReader(f, DatumReader())
        for value in reader:
            data_file_value = value["data_file"]
            # Unwrap the key values.
            value_counts = {
                x["key"]: x["value"] for x in data_file_value["value_counts"]
            }
            null_counts = {
                x["key"]: x["value"] for x in data_file_value["null_value_counts"]
            }
            lower_bounds = {
                x["key"]: x["value"] for x in data_file_value["lower_bounds"]
            }
            upper_bounds = {
                x["key"]: x["value"] for x in data_file_value["upper_bounds"]
            }
            value_counts_list.append(value_counts)
            null_counts_list.append(null_counts)
            lower_bounds_list.append(lower_bounds)
            upper_bounds_list.append(upper_bounds)
    assert len(value_counts_list) == 1, "There should only be 1 data file"
    assert len(null_counts_list) == 1, "There should only be 1 data file"
    assert len(lower_bounds_list) == 1, "There should only be 1 data file"
    assert len(upper_bounds_list) == 1, "There should only be 1 data file"
    value_counts = value_counts_list[0]
    null_counts = null_counts_list[0]
    lower_bounds = lower_bounds_list[0]
    upper_bounds = upper_bounds_list[0]
    return value_counts, null_counts, lower_bounds, upper_bounds


def validate_metrics(
    warehouse_loc: str,
    db_schema: str,
    table_name: str,
    expected_value_counts: dict[str, int],
    expected_null_counts: dict[str, int],
    expected_lower_bounds: dict[str, bytes],
    expected_upper_bounds: dict[str, bytes],
):
    metadata_json_file = get_metadata_path(warehouse_loc, db_schema, table_name)
    manifest_list_path, name_to_id_map = extract_schema_information(metadata_json_file)
    manifest_file_path = get_manifest_file_path(manifest_list_path)
    value_counts, null_counts, lower_bounds, upper_bounds = get_manifest_file_metrics(
        manifest_file_path
    )
    for name, id in name_to_id_map.items():
        if name in expected_value_counts:
            assert value_counts[id] == expected_value_counts[name], (
                "Value counts do not match"
            )
        else:
            assert id not in value_counts, "Unexpected value counts"
        if name in expected_null_counts:
            assert null_counts[id] == expected_null_counts[name], (
                "Null value counts do not match"
            )
        else:
            assert id not in null_counts, "Unexpected null value counts"
        if name in expected_lower_bounds:
            assert lower_bounds[id] == expected_lower_bounds[name], (
                "Lower bounds do not match"
            )
        else:
            assert id not in lower_bounds, "Unexpected lower bound"
        if name in expected_upper_bounds:
            assert upper_bounds[id] == expected_upper_bounds[name], (
                "Upper bounds do not match"
            )
        else:
            assert id not in upper_bounds, "Unexpected upper bound"


@pytest_mark_one_rank
def test_numeric_metrics(
    iceberg_database,
    iceberg_table_conn,
    memory_leak_check,
):
    table_name = "numeric_metrics_table"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)

    df = pd.DataFrame(
        {
            "A": pd.array([1, 2, 3, None, 5], dtype="Int32"),
            "B": pd.array([1, None, None, 4, 5], dtype="Int64"),
            "C": pd.array([None, -2.0, 3.0, None, 5.25], dtype="Float32"),
            "D": pd.array([-15.0, 2.0, 3.0, 4.0, 11.5], dtype="Float64"),
            "E": np.array([None, None, None, None, Decimal("5.0")]),
            "F": pd.array(
                [Decimal("123.0"), None, Decimal("0.051"), None, Decimal("980.604")],
                dtype=pd.ArrowDtype(pa.decimal128(6, 3)),
            ),
            "G": pd.array(
                [Decimal(f"{i}.{i}") for i in range(3, 8)],
                dtype=pd.ArrowDtype(pa.decimal128(2, 1)),
            ),
        }
    )
    create_table_jit(df, table_name, conn, db_schema)
    expected_value_counts = {"A": 5, "B": 5, "C": 5, "D": 5, "E": 5, "F": 5, "G": 5}
    expected_null_counts = {"A": 1, "B": 2, "C": 2, "D": 0, "E": 4, "F": 2, "G": 0}
    # Note: Since the tests care about endianness we manually convert
    # to bytes to ensure portability.
    expected_lower_bounds = {
        "A": b"\x01\x00\x00\x00",
        "B": b"\x01\x00\x00\x00\x00\x00\x00\x00",
        # Note: Float values are converted using a IEEE 754 converter
        # and remapped to little endian.
        "C": b"\x00\x00\x00\xc0",
        "D": b"\x00\x00\x00\x00\x00\x00\x2e\xc0",
        # This value is manually calculated for Decimal(38, 18) by computing
        # 5 * 10^18 and converting to a 128 bit representation.
        "E": b"\x00\x00\x00\x00\x00\x00\x00\x00\x45\x63\x91\x82\x44\xf4\x00\x00",
        # This value is manually calculated for Decimal(6, 3) by computing
        # 51 and converting to a 32 bit representation (smallest # of bytes required).
        "F": b"\x00\x00\x00\x33",
        # This value is manually calculated for Decimal(2, 1) by computing
        # 33 and converting to an 8 bit representation (smallest # of bytes required).
        "G": b"\x21",
    }
    expected_upper_bounds = {
        "A": b"\x05\x00\x00\x00",
        "B": b"\x05\x00\x00\x00\x00\x00\x00\x00",
        # Note: Float values are converted using a IEEE 754 converter
        # and remapped to little endian.
        "C": b"\x00\x00\xa8\x40",
        "D": b"\x00\x00\x00\x00\x00\x00\x27\x40",
        # This value is manually calculated for Decimal(38, 18) by computing
        # 5 * 10^18 and converting to a 128 bit representation.
        "E": b"\x00\x00\x00\x00\x00\x00\x00\x00\x45\x63\x91\x82\x44\xf4\x00\x00",
        # This value is manually calculated for Decimal(6, 3) by computing
        # 980604 and converting to a 32 bit representation (smallest # of bytes required).
        "F": b"\x00\x0e\xf6\x7c",
        # This value is manually calculated for Decimal(2, 1) by computing
        # 77 and converting to an 8 bit representation (smallest # of bytes required).
        "G": b"\x4d",
    }
    validate_metrics(
        warehouse_loc,
        db_schema,
        table_name,
        expected_value_counts,
        expected_null_counts,
        expected_lower_bounds,
        expected_upper_bounds,
    )


@pytest_mark_one_rank
def test_datetime_metrics(
    iceberg_database,
    iceberg_table_conn,
    memory_leak_check,
):
    table_name = "datetime_metrics_table"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)
    df = pd.DataFrame(
        {
            # Timestamp NTZ
            "A": pd.array(
                [
                    pd.Timestamp("2021-01-01 14:23:12.431"),
                    pd.Timestamp("2021-01-02 11:44:21"),
                    None,
                    pd.Timestamp("2021-01-04"),
                    pd.Timestamp("2021-01-05 00:00:00.000001"),
                ]
            ),
            # Timestamp LTZ
            "B": pd.array(
                [
                    pd.Timestamp("2021-01-01 14:23:12.431", tz="US/Pacific"),
                    pd.Timestamp("2021-01-02 11:44:21", tz="US/Pacific"),
                    None,
                    pd.Timestamp("2021-01-04", tz="US/Pacific"),
                    pd.Timestamp("2021-01-05 00:00:00.000001", tz="US/Pacific"),
                ]
            ),
            # Time
            "C": np.array(
                [
                    bodo.types.Time(2, 46, 40, precision=6),
                    bodo.types.Time(0, 59, 59, microsecond=11, precision=6),
                    None,
                    None,
                    bodo.types.Time(23, 45, 45, millisecond=948, precision=6),
                ]
            ),
            # Date
            "D": np.array(
                [datetime.date(2021, 1, 1), None, None, datetime.date(2021, 1, 4), None]
            ),
        }
    )
    create_table_jit(df, table_name, conn, db_schema)
    expected_value_counts = {"A": 5, "B": 5, "C": 5, "D": 5}
    expected_null_counts = {"A": 1, "B": 1, "C": 2, "D": 3}
    # Note: Since the tests care about endianness we manually convert
    # to bytes to ensure portability.
    expected_lower_bounds = {
        "A": b"\x98\x47\xf7\x7a\xd7\xb7\x05\x00",
        "B": b"\x98\x67\x94\x2f\xde\xb7\x05\x00",
        "C": b"\xcb\x61\x84\xd6\x00\x00\x00\x00",
        "D": b"\xc4\x48\x00\x00",
    }
    expected_upper_bounds = {
        "A": b"\x01\x00\x43\xe3\x1b\xb8\x05\x00",
        "B": b"\x01\x20\xe0\x97\x22\xb8\x05\x00",
        "C": b"\x60\x93\xef\xea\x13\x00\x00\x00",
        "D": b"\xc7\x48\x00\x00",
    }
    validate_metrics(
        warehouse_loc,
        db_schema,
        table_name,
        expected_value_counts,
        expected_null_counts,
        expected_lower_bounds,
        expected_upper_bounds,
    )


@pytest_mark_one_rank
def test_string_metrics(
    iceberg_database,
    iceberg_table_conn,
    memory_leak_check,
):
    table_name = "string_metrics_table"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)
    # Explicitly check for dictionary values that aren't in the indices.
    indices = pa.Int32Array.from_pandas(pd.array([1, 1, None, 2, 2]))
    dictionary = pa.StringArray.from_pandas(pd.array(["a", "b", "d", "e"]))
    dict_arr = pa.DictionaryArray.from_arrays(indices, dictionary)
    df = pd.DataFrame(
        {
            "A": pd.array([None, "AAC", "AALER", "AALR", None], dtype="string"),
            "B": dict_arr,
        }
    )
    try:
        # Disable dictionary encoding so every value we write manually sets dictionaries.
        orig_use_dict_str_type = bodo.hiframes.boxing._use_dict_str_type
        bodo.hiframes.boxing._use_dict_str_type = False
        create_table_jit(df, table_name, conn, db_schema)
    finally:
        bodo.hiframes.boxing._use_dict_str_type = orig_use_dict_str_type
    expected_value_counts = {"A": 5, "B": 5}
    expected_null_counts = {"A": 2, "B": 1}
    expected_lower_bounds = {
        "A": b"AAC",
        "B": b"b",
    }
    expected_upper_bounds = {
        # Note: The standard just defines non-zero but we always write 1.
        "A": b"AALR",
        "B": b"d",
    }
    validate_metrics(
        warehouse_loc,
        db_schema,
        table_name,
        expected_value_counts,
        expected_null_counts,
        expected_lower_bounds,
        expected_upper_bounds,
    )


@pytest_mark_one_rank
def test_boolean_binary_metrics(
    iceberg_database,
    iceberg_table_conn,
    memory_leak_check,
):
    table_name = "boolean_binary_metrics_table"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)

    df = pd.DataFrame(
        {
            "A": pd.array([True, False, False, True, None], dtype="boolean"),
            "B": np.array([b"\x07", None, None, b"\x05\x06", b"\x05"]),
        }
    )
    create_table_jit(df, table_name, conn, db_schema)
    expected_value_counts = {"A": 5, "B": 5}
    expected_null_counts = {"A": 1, "B": 2}
    expected_lower_bounds = {
        "A": b"\x00",
        "B": b"\x05",
    }
    expected_upper_bounds = {
        # Note: The standard just defines non-zero but we always write 1.
        "A": b"\x01",
        "B": b"\x07",
    }
    validate_metrics(
        warehouse_loc,
        db_schema,
        table_name,
        expected_value_counts,
        expected_null_counts,
        expected_lower_bounds,
        expected_upper_bounds,
    )


@pytest_mark_one_rank
def test_struct_metrics(
    iceberg_database,
    iceberg_table_conn,
    memory_leak_check,
):
    table_name = "struct_metrics_table"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)
    struct_array = pd.array(
        [
            {"f1": 1, "f2": None, "f3": datetime.date(2021, 1, 1)},
            {"f1": 2, "f2": "AAC", "f3": None},
            {"f1": 3, "f2": "AALER", "f3": None},
            {"f1": None, "f2": "AALR", "f3": datetime.date(2021, 1, 4)},
            {"f1": 5, "f2": None, "f3": None},
        ],
        dtype=pd.ArrowDtype(
            pa.struct([("f1", pa.int32()), ("f2", pa.string()), ("f3", pa.date32())])
        ),
    )
    df = pd.DataFrame({"A": struct_array})
    create_table_jit(df, table_name, conn, db_schema)
    expected_value_counts = {"A": 5, "A.f1": 5, "A.f2": 5, "A.f3": 5}
    expected_null_counts = {"A": 0, "A.f1": 1, "A.f2": 2, "A.f3": 3}
    expected_lower_bounds = {
        "A.f1": b"\x01\x00\x00\x00",
        "A.f2": b"AAC",
        "A.f3": b"\xc4\x48\x00\x00",
    }
    expected_upper_bounds = {
        "A.f1": b"\x05\x00\x00\x00",
        "A.f2": b"AALR",
        "A.f3": b"\xc7\x48\x00\x00",
    }
    validate_metrics(
        warehouse_loc,
        db_schema,
        table_name,
        expected_value_counts,
        expected_null_counts,
        expected_lower_bounds,
        expected_upper_bounds,
    )


@pytest_mark_one_rank
def test_list_metrics(
    iceberg_database,
    iceberg_table_conn,
    memory_leak_check,
):
    table_name = "list_metrics_table"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)
    list_int_array = pd.array(
        [
            [1, 2, None, 4],
            [3, 4],
            [None, 2, 7],
            None,
            [5, 6, 7],
        ],
        dtype=pd.ArrowDtype(pa.list_(pa.int32())),
    )
    list_string_array = pd.array(
        [
            ["a", "b"],
            ["c", "d", "e"],
            None,
            ["f", None],
            [None, None],
        ],
        dtype=pd.ArrowDtype(pa.large_list(pa.string())),
    )
    df = pd.DataFrame({"A": list_int_array, "B": list_string_array})
    create_table_jit(df, table_name, conn, db_schema)
    expected_value_counts = {"A": 5, "B": 5, "A.element": 12, "B.element": 9}
    expected_null_counts = {"A": 1, "B": 1, "A.element": 2, "B.element": 3}
    expected_lower_bounds = {}
    expected_upper_bounds = {}
    validate_metrics(
        warehouse_loc,
        db_schema,
        table_name,
        expected_value_counts,
        expected_null_counts,
        expected_lower_bounds,
        expected_upper_bounds,
    )


@pytest_mark_one_rank
def test_map_metrics(
    iceberg_database,
    iceberg_table_conn,
    memory_leak_check,
):
    table_name = "map_metrics_table"
    db_schema, warehouse_loc = iceberg_database(table_name)
    conn = iceberg_table_conn(table_name, db_schema, warehouse_loc, check_exists=False)
    map_str_int = pd.array(
        [
            {"a": 1, "b": None},
            {"c": 3, "d": None, "e": 5},
            {"a": None},
            None,
            {"x": 7},
        ],
        dtype=pd.ArrowDtype(pa.map_(pa.string(), pa.int32())),
    )
    df = pd.DataFrame({"A": map_str_int})
    create_table_jit(df, table_name, conn, db_schema)
    expected_value_counts = {"A": 5, "A.key": 7, "A.value": 7}
    expected_null_counts = {"A": 1, "A.key": 0, "A.value": 3}
    expected_lower_bounds = {}
    expected_upper_bounds = {}
    validate_metrics(
        warehouse_loc,
        db_schema,
        table_name,
        expected_value_counts,
        expected_null_counts,
        expected_lower_bounds,
        expected_upper_bounds,
    )
