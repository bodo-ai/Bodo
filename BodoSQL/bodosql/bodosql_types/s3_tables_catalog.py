"""Python and JIT class for describing a S3 Tables Iceberg catalog. A S3 Tables
catalog contains all information needed to connect to S3 Tables for organizing and modifying tables.
"""

import numba
from numba.core import cgutils, types
from numba.extending import (
    NativeValue,
    box,
    intrinsic,
    make_attribute_wrapper,
    models,
    overload,
    register_model,
    typeof_impl,
    unbox,
)

from bodo.io.iceberg import IcebergConnectionType
from bodo.utils.typing import get_literal_value, raise_bodo_error
from bodosql import DatabaseCatalog, DatabaseCatalogType
from bodosql.imported_java_classes import JavaEntryPoint


def _create_java_s3_tables_catalog(warehouse: str):
    """
    Create a Java BodoS3Tables object.
    Args:
        warehouse (str): The warehouse to connect to.
    Returns:
        JavaObject: A Java BodoS3Tables object.
    """
    return JavaEntryPoint.buildBodoS3TablesCatalog(warehouse)


class S3TablesCatalog(DatabaseCatalog):
    """
    Python class for storing the information
        needed to connect to a S3 Tables Iceberg catalog.
    """

    def __init__(self, warehouse: str):
        """
        Create a S3 Tables catalog from a connection string to a S3 Tables catalog.
        Args:
            warehouse (str): The warehouse to connect to.
        """
        self.warehouse = warehouse

    def get_java_object(self):
        return _create_java_s3_tables_catalog(self.warehouse)

    def __eq__(self, other):
        if not isinstance(other, S3TablesCatalog):
            return False
        return self.warehouse == other.warehouse


@overload(S3TablesCatalog, no_unliteral=True)
def overload_s3_tables_catalog_constructor(warehouse: str):
    raise_bodo_error("S3TablesCatalog: Cannot be created in JIT mode.")


class S3TablesCatalogType(DatabaseCatalogType):
    def __init__(self, warehouse: str):
        self.warehouse = warehouse
        super().__init__(name=f"S3TablesCatalog({self.warehouse=})")

    def get_java_object(self):
        return _create_java_s3_tables_catalog(self.warehouse)

    @property
    def key(self):
        return self.warehouse


@typeof_impl.register(S3TablesCatalog)
def typeof_s3_tables_catalog(val, c):
    return S3TablesCatalogType(warehouse=val.warehouse)


register_model(S3TablesCatalogType)(models.OpaqueModel)


@box(S3TablesCatalogType)
def box_s3_tables_catalog_type(typ, val, c):
    """
    Box a S3 Tables Catalog native representation into a Python object. We populate
    the contents based on typing information.
    """
    warehouse_obj = c.pyapi.from_native_value(
        types.unicode_type,
        c.context.get_constant_generic(c.builder, types.unicode_type, typ.warehouse),
        c.env_manager,
    )
    s3_tables_catalog_obj = c.pyapi.unserialize(
        c.pyapi.serialize_object(S3TablesCatalog)
    )
    res = c.pyapi.call_function_objargs(s3_tables_catalog_obj, (warehouse_obj,))
    c.pyapi.decref(warehouse_obj)
    c.pyapi.decref(s3_tables_catalog_obj)
    return res


@unbox(S3TablesCatalogType)
def unbox_s3_tables_catalog_type(typ, val, c):
    """
    Unbox a S3 Tables Catalog Python object into its native representation.
    Since the actual model is opaque we can just generate a dummy.
    """
    return NativeValue(c.context.get_dummy_value())


@numba.jit(types.unicode_type(types.unicode_type))
def get_conn_str(warehouse):
    """Get the connection string for a S3 Tables Iceberg catalog."""
    return "iceberg+" + warehouse


class S3TablesConnectionType(IcebergConnectionType):
    """
    Python class for storing the information needed to connect to a S3 Tables Iceberg catalog.
    The compiler can get a connection string using the get_conn_str function.
    The runtime can get a connection string using the conn_str attribute.
    """

    def __init__(self, warehouse):
        self.warehouse = warehouse
        self.conn_str = get_conn_str(warehouse)

        super().__init__(name=f"S3TablesConnectionType({warehouse=})")


@intrinsic
def _get_s3_tables_connection(typingctx, warehouse, conn_str):
    """Create a struct model for a  S3TablesonnectionType from a warehouse and connection string."""
    literal_warehouse = get_literal_value(warehouse)
    s3_tables_connection_type = S3TablesConnectionType(literal_warehouse)

    def codegen(context, builder, sig, args):
        """lowering code to initialize a S3TablesConnectionType"""
        s3_tables_connection_type = sig.return_type
        s3_tables_connection_struct = cgutils.create_struct_proxy(
            s3_tables_connection_type
        )(context, builder)
        context.nrt.incref(builder, sig.args[1], args[1])
        s3_tables_connection_struct.conn_str = args[1]
        return s3_tables_connection_struct._getvalue()

    return s3_tables_connection_type(warehouse, conn_str), codegen


def get_s3_tables_connection(warehouse: str):
    pass


@overload(get_s3_tables_connection, no_unliteral=True)
def overload_get_s3_tables_connection(warehouse: str):
    """Overload for get_s3_tables_connection that creates a S3TablesConnectionType."""

    def impl(warehouse: str):  # pragma: no cover
        conn_str = get_conn_str(warehouse)
        return _get_s3_tables_connection(warehouse, conn_str)

    return impl


@register_model(S3TablesConnectionType)
class S3TablesConnectionModel(models.StructModel):
    """Model for S3TablesConnectionType that has one member, conn_str."""

    def __init__(self, dmm, fe_type):
        members = [
            ("conn_str", types.unicode_type),
        ]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(S3TablesConnectionType, "conn_str", "conn_str")
