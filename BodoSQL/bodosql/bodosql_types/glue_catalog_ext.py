"""JIT extensions for GlueCatalog"""

from __future__ import annotations

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

from bodo.ir.iceberg_ext import IcebergConnectionType
from bodo.utils.typing import get_literal_value, raise_bodo_error
from bodosql.bodosql_types.database_catalog_ext import DatabaseCatalogType
from bodosql.bodosql_types.glue_catalog import GlueCatalog, _create_java_glue_catalog


@overload(GlueCatalog, no_unliteral=True)
def overload_glue_catalog_constructor(warehouse: str, region: str = ""):
    raise_bodo_error("GlueCatalog: Cannot be created in JIT mode.")


class GlueCatalogType(DatabaseCatalogType):
    def __init__(self, warehouse: str, region: str = ""):
        """
        Create a glue catalog type from a connection string to a glue catalog.
        Args:
            warehouse (str): The warehouse to connect to.
        """
        self.warehouse = warehouse
        self.region = region

        super().__init__(name=f"GlueCatalogType({self.warehouse=}, {self.region=})")

    def get_java_object(self):
        return _create_java_glue_catalog(self.warehouse, self.region)

    @property
    def key(self):
        return self.warehouse, self.region


@typeof_impl.register(GlueCatalog)
def typeof_glue_catalog(val, c):
    return GlueCatalogType(warehouse=val.warehouse, region=val.region)


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
    region_obj = c.pyapi.from_native_value(
        types.unicode_type,
        c.context.get_constant_generic(c.builder, types.unicode_type, typ.region),
        c.env_manager,
    )

    glue_catalog_obj = c.pyapi.unserialize(c.pyapi.serialize_object(GlueCatalog))
    res = c.pyapi.call_function_objargs(glue_catalog_obj, (warehouse_obj, region_obj))
    c.pyapi.decref(warehouse_obj)
    c.pyapi.decref(region_obj)
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
def get_conn_str(warehouse, region):
    """Get the connection string for a Glue Iceberg catalog."""
    con_str = f"iceberg+glue?warehouse={warehouse}"
    if region:
        con_str += f"&client.region={region}"
    return con_str


class GlueConnectionType(IcebergConnectionType):
    """
    Python class for storing the information needed to connect to a Glue Iceberg catalog.
    The compiler can get a connection string using the get_conn_str function.
    The runtime can get a connection string using the conn_str attribute.
    """

    def __init__(self, warehouse, region=""):
        self.warehouse = warehouse
        self.region = region
        self.conn_str = get_conn_str(warehouse, region)

        super().__init__(name=f"GlueConnectionType({warehouse=}, {region=})")


@intrinsic(prefer_literal=True)
def _get_glue_connection(typingctx, warehouse, region, conn_str):
    """Create a struct model for a  GlueConnectionType from a warehouse, region, and connection string."""
    literal_warehouse = get_literal_value(warehouse)
    literal_region = get_literal_value(region)
    glue_connection_type = GlueConnectionType(literal_warehouse, literal_region)

    def codegen(context, builder, sig, args):
        """lowering code to initialize a GlueConnectionType"""
        glue_connection_type = sig.return_type
        glue_connection_struct = cgutils.create_struct_proxy(glue_connection_type)(
            context, builder
        )
        context.nrt.incref(builder, sig.args[2], args[2])
        glue_connection_struct.conn_str = args[2]
        return glue_connection_struct._getvalue()

    return glue_connection_type(warehouse, region, conn_str), codegen


def get_glue_connection(warehouse: str, region: str = ""):
    pass


@overload(get_glue_connection, no_unliteral=True)
def overload_get_glue_connection(warehouse: str, region: str = ""):
    """Overload for get_glue_connection that creates a GlueConnectionType."""

    def impl(warehouse: str, region: str = ""):  # pragma: no cover
        conn_str = get_conn_str(warehouse, region)
        conn = _get_glue_connection(warehouse, region, conn_str)
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
