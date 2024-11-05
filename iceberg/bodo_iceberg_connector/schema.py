"""
API used to translate a Java Schema object into various Pythonic
representations (Arrow and Bodo)
"""

import typing as pt
from collections import namedtuple

from py4j.protocol import Py4JError

from bodo_iceberg_connector.catalog_conn import (
    gen_table_loc,
    normalize_data_loc,
    parse_conn_str,
)
from bodo_iceberg_connector.errors import IcebergError, IcebergJavaError
from bodo_iceberg_connector.py4j_support import (
    get_catalog,
    launch_jvm,
)
from bodo_iceberg_connector.schema_helper import (
    arrow_schema_j2py,
    b_ICEBERG_FIELD_ID_MD_KEY,
    convert_arrow_schema_to_large_types,
)

if pt.TYPE_CHECKING:
    import pyarrow as pa


# Types I didn't figure out how to test with Spark:
#   FixedType
#   TimeType
#   UUIDType
# These should be possible to support based on this information
# https://iceberg.apache.org/spec/#parquet

# Create a named tuple for schema components
BodoIcebergSchema = namedtuple(
    "BodoIcebergSchema", "colnames coltypes field_ids is_required"
)


def get_typing_info(conn_str: str, schema: str, table: str):
    """
    Return information about an Iceberg Table needed at compile-time
    Primarily used for writing to Iceberg
    """

    (
        schema_id,
        warehouse,
        _,
        pyarrow_schema,
        iceberg_schema_str,
        partition_spec,
        sort_order,
    ) = get_iceberg_info(conn_str, schema, table, False)
    return (
        warehouse,
        schema_id,
        pyarrow_schema,
        iceberg_schema_str,
        partition_spec,
        sort_order,
    )


def get_iceberg_typing_schema(conn_str: str, schema: str, table: str):
    """
    Returns the table schema information for a given iceberg table
    used at typing. Also returns the pyarrow schema object.
    """
    # TODO: Combine with get_typing_info?
    _, _, schemas, pyarrow_schema, _, _, _ = get_iceberg_info(conn_str, schema, table)
    assert schemas is not None
    return (schemas.colnames, schemas.coltypes, pyarrow_schema)


def get_iceberg_runtime_schema(conn_str: str, schema: str, table: str):
    """
    Returns the table schema information for a given iceberg table
    used at runtime.
    """
    _, _, schemas, _, _, _, _ = get_iceberg_info(conn_str, schema, table)
    assert schemas is not None
    return (schemas.field_ids, schemas.coltypes)


def get_iceberg_info(conn_str: str, schema: str, table: str, error: bool = True):
    """
    Returns all of the necessary Bodo schemas for an iceberg table,
    both using field_id and names.

    Port is unused and kept in case we opt to switch back to py4j
    """
    # Parse conn_str to determine catalog type and warehouse location
    catalog_type, warehouse = parse_conn_str(conn_str)

    try:
        # Construct Table Reader
        bodo_iceberg_table_reader = get_catalog(conn_str, catalog_type)

        # Get Iceberg Schema Info
        java_table_info = bodo_iceberg_table_reader.getTableInfo(schema, table, error)
        if java_table_info is None:
            schema_id = None
            iceberg_schema_str = ""
            java_schema = None
            py_schema = None
            pyarrow_schema = None
            pyarrow_types = []
            iceberg_schema = None
            partition_spec = []
            sort_order = []

            if warehouse is None:
                raise IcebergError(
                    "`warehouse` parameter required in connection string"
                )
            table_loc = gen_table_loc(catalog_type, warehouse, schema, table)  # type: ignore

        else:
            schema_id: int | None = java_table_info.getSchemaID()
            iceberg_schema_str = str(java_table_info.getIcebergSchemaEncoding())
            # XXX Do we need this anymore if the Arrow schema has all the
            # details including the Iceberg Field IDs?
            java_schema = java_table_info.getIcebergSchema()
            py_schema = iceberg_schema_java_to_py(java_schema)

            pyarrow_schema: pa.Schema = arrow_schema_j2py(
                java_table_info.getArrowSchema()
            )
            pyarrow_schema = convert_arrow_schema_to_large_types(pyarrow_schema)
            assert (
                py_schema.colnames == pyarrow_schema.names
            ), "Iceberg Schema Field Names Should be Equal in PyArrow Schema"
            assert (
                py_schema.field_ids
                == [int(f.metadata[b_ICEBERG_FIELD_ID_MD_KEY]) for f in pyarrow_schema]
            ), "Iceberg Field IDs should match the IDs in the metadata of the PyArrow Schema's fields"
            assert (
                py_schema.is_required == [(not f.nullable) for f in pyarrow_schema]
            ), "Iceberg fields' 'required' property should match the nullability of the PyArrow Schema's fields"

            pyarrow_types = [
                pyarrow_schema.field(name) for name in pyarrow_schema.names
            ]

            iceberg_schema = BodoIcebergSchema(
                pyarrow_schema.names,
                pyarrow_types,
                py_schema.field_ids,
                py_schema.is_required,
            )
            # Create a map from Iceberg column field id to
            # column name
            field_id_to_col_name_map: dict[int, str] = {
                py_schema.field_ids[i]: py_schema.colnames[i]
                for i in range(len(py_schema.colnames))
            }

            # TODO: Override when warehouse is passed in?
            # Or move the table? Java API has ability to do so
            table_loc = java_table_info.getLoc()

            partition_spec = partition_spec_j2py(
                java_table_info.getPartitionFields(), field_id_to_col_name_map
            )
            sort_order = sort_order_j2py(
                java_table_info.getSortFields(), field_id_to_col_name_map
            )

    except Py4JError as e:
        raise IcebergJavaError.from_java_error(e)

    # TODO: Remove when sure that Iceberg's Schema and PyArrow's Schema always match
    # field_ids are necessary
    # pyarrow_types are directly from pyarrow schema so not necessary
    # Unsure about colnames
    # is_required == nullable from pyarrow
    return (
        schema_id,
        normalize_data_loc(table_loc),
        iceberg_schema,
        # This has the Iceberg Field IDs embedded in the
        # fields' metadata:
        pyarrow_schema,
        iceberg_schema_str,
        partition_spec,
        sort_order,
    )


def iceberg_schema_java_to_py(java_schema):
    """
    Converts an Iceberg Java schema object to a Python equivalent.
    """
    # List of column names
    colnames = []
    # List of column types
    coltypes = []
    # List of field ids
    field_ids = []
    # List of if each column is required
    is_required_lst = []
    # Get a list of Java objects for the columns
    field_objects = java_schema.columns()
    for field in field_objects:
        # This should be a Python string.
        name = str(field.name())
        # This should be a Python Integer
        field_id = int(field.fieldId())
        # This should be a Python Boolean
        is_required = bool(field.isRequired())
        # This should be a Java object
        iceberg_type = field.type()
        type_val = str(iceberg_type.toString())
        colnames.append(name)
        coltypes.append(type_val)
        field_ids.append(field_id)
        is_required_lst.append(is_required)

    return BodoIcebergSchema(colnames, coltypes, field_ids, is_required_lst)


def partition_spec_j2py(
    partition_list, field_id_to_col_name_map: dict[int, str]
) -> list[tuple[str, str, int, str]]:
    """
    Generate python representation of partition spec which is
    a tuple containing the column name, the name of the transform,
    the argument for the transform, and the name of the transformed
    column.
    field_id_to_col_name_map is a map of field id of the columns in
    table to its name.
    """
    return [
        (
            field_id_to_col_name_map[spec.sourceId()],
            *get_transform_info(spec.transform()),
            spec.name(),
        )
        for spec in partition_list
    ]


def sort_order_j2py(
    sort_list, field_id_to_col_name_map: dict[int, str]
) -> list[tuple[str, str, int, bool, bool]]:
    """
    Generate python representation of sort order which is
    a tuple containing the column name, the name of the transform,
    the argument for the transform, whether to sort in an ascending
    order and whether to put nulls last when sorting.
    field_id_to_col_name_map is a map of field id of the columns in
    table to its name.
    """
    gateway = launch_jvm()
    return [
        (
            field_id_to_col_name_map[order.sourceId()],
            *get_transform_info(order.transform()),
            order.direction() == gateway.jvm.org.apache.iceberg.SortDirection.ASC,  # type: ignore
            order.nullOrder() == gateway.jvm.org.apache.iceberg.NullOrder.NULLS_LAST,  # type: ignore
        )
        for order in sort_list
    ]


def get_transform_info(transform) -> tuple[str, int]:
    name = transform.toString()
    if name.startswith("truncate["):
        return "truncate", int(name[(len("truncate") + 1) : -1])
    elif name.startswith("bucket["):
        return "bucket", int(name[(len("bucket") + 1) : -1])
    else:
        return name, -1
