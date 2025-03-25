"""Python and JIT class for describing a Glue Iceberg catalog. A Glue
catalog contains all information needed to connect use Glue Iceberg catalog for organizing and modifying tables.
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


def _create_java_glue_catalog(warehouse: str):
    """
    Create a Java BodoGlueCatalog object.
    Args:
        warehouse (str): The warehouse to connect to.
    Returns:
        JavaObject: A Java GlueCatalog object.
    """
    return JavaEntryPoint.buildBodoGlueCatalog(warehouse)


class GlueCatalog(DatabaseCatalog):
    """
    Python class for storing the information
        needed to connect to a Glue Iceberg catalog.
    """

    def __init__(self, warehouse: str):
        """
        Create a Glue catalog from a connection string to a glue catalog.
        Args:
            warehouse (str): The warehouse to connect to.
        """
        self.warehouse = warehouse

    def get_java_object(self):
        return _create_java_glue_catalog(self.warehouse)

    def __eq__(self, other):
        if not isinstance(other, GlueCatalog):
            return False
        return self.warehouse == other.warehouse


@overload(GlueCatalog, no_unliteral=True)
def overload_snowflake_catalog_constructor(warehouse: str):
    raise_bodo_error("GlueCatalog: Cannot be created in JIT mode.")


class GlueCatalogType(DatabaseCatalogType):
    def __init__(self, warehouse: str):
        """
        Create a glue catalog type from a connection string to a glue catalog.
        Args:
            warehouse (str): The warehouse to connect to.
        """
        self.warehouse = warehouse

        super().__init__(name=f"GlueCatalogType({self.warehouse=})")

    def get_java_object(self):
        return _create_java_glue_catalog(self.warehouse)

    @property
    def key(self):
        return self.warehouse


@typeof_impl.register(GlueCatalog)
def typeof_glue_catalog(val, c):
    return GlueCatalogType(warehouse=val.warehouse)


register_model(GlueCatalogType)(models.OpaqueModel)


@box(GlueCatalogType)
def box_glue_catalog(typ, val, c):
    """
    Box a Glue Catalog native representation into a Python object. We populate
    the contents based on typing information.
    """
    warehouse_obj = c.pyapi.from_native_value(
        types.unicode_type,
        c.context.get_constant_generic(c.builder, types.unicode_type, typ.warehouse),
        c.env_manager,
    )

    glue_catalog_obj = c.pyapi.unserialize(c.pyapi.serialize_object(GlueCatalog))
    res = c.pyapi.call_function_objargs(glue_catalog_obj, (warehouse_obj,))
    c.pyapi.decref(warehouse_obj)
    c.pyapi.decref(glue_catalog_obj)
    return res


@unbox(GlueCatalogType)
def unbox_glue_catalog(typ, val, c):
    """
    Unbox a Glue Catalog Python object into its native representation.
    Since the actual model is opaque we can just generate a dummy.
    """
    return NativeValue(c.context.get_dummy_value())


@numba.jit
def get_conn_str(warehouse):
    """Get the connection string for a Glue Iceberg catalog."""
    return f"iceberg+glue?warehouse={warehouse}"


class GlueConnectionType(IcebergConnectionType):
    """
    Python class for storing the information needed to connect to a Glue Iceberg catalog.
    The compiler can get a connection string using the get_conn_str function.
    The runtime can get a connection string using the conn_str attribute.
    """

    def __init__(self, warehouse):
        self.warehouse = warehouse
        self.conn_str = get_conn_str(warehouse)

        super().__init__(name=f"GlueConnectionType({warehouse=})")


@intrinsic(prefer_literal=True)
def _get_glue_connection(typingctx, warehouse, conn_str):
    """Create a struct model for a  GlueConnectionType from a warehouse and connection string."""
    literal_warehouse = get_literal_value(warehouse)
    glue_connection_type = GlueConnectionType(literal_warehouse)

    def codegen(context, builder, sig, args):
        """lowering code to initialize a GlueConnectionType"""
        glue_connection_type = sig.return_type
        glue_connection_struct = cgutils.create_struct_proxy(glue_connection_type)(
            context, builder
        )
        context.nrt.incref(builder, sig.args[1], args[1])
        glue_connection_struct.conn_str = args[1]
        return glue_connection_struct._getvalue()

    return glue_connection_type(warehouse, conn_str), codegen


def get_glue_connection(warehouse: str):
    pass


@overload(get_glue_connection, no_unliteral=True)
def overload_get_glue_connection(warehouse: str):
    """Overload for get_glue_connection that creates a GlueConnectionType."""

    def impl(warehouse: str):  # pragma: no cover
        conn_str = get_conn_str(warehouse)
        conn = _get_glue_connection(warehouse, conn_str)
        return conn

    return impl


@register_model(GlueConnectionType)
class GlueConnectionTypeModel(models.StructModel):
    """Model for GlueConnectionType has one member, conn_str."""

    def __init__(self, dmm, fe_type):
        members = [
            ("conn_str", types.unicode_type),
        ]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(GlueConnectionType, "conn_str", "conn_str")
