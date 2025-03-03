"""Python and JIT class for describing a REST Iceberg catalog. A REST
catalog contains all information needed to connect and use REST Iceberg catalog for organizing and modifying tables.
"""

import os

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

import bodo
from bodo.io.iceberg import IcebergConnectionType
from bodo.io.iceberg.catalog import conn_str_to_catalog
from bodo.utils.typing import get_literal_value, raise_bodo_error
from bodo.utils.utils import run_rank0
from bodosql import DatabaseCatalog, DatabaseCatalogType
from bodosql.imported_java_classes import JavaEntryPoint


def _create_java_REST_catalog(
    rest_uri: str,
    warehouse: str,
    token: str | None,
    credential: str | None,
    scope: str | None = None,
    default_schema: str | None = None,
):
    """
    Create a Java RESTCatalog object.
    Args:
        warehouse (str): The warehouse to connect to.
        rest_uri (str): The URI of the REST server.
        token (str): The token to use for authentication.
        credential (str): The credential to use for authentication.
        scope (str): The scope to use for authentication.
        default_schema (str): The default schema to use.
    Returns:
        JavaObject: A Java RESTCatalog object.
    """
    return JavaEntryPoint.buildIcebergRESTCatalog(
        rest_uri,
        warehouse,
        token,
        credential,
        scope,
        default_schema,
    )


class RESTCatalog(DatabaseCatalog):
    """
    Python class for storing the information
        needed to connect to a REST Iceberg catalog.
    """

    def __init__(
        self,
        warehouse: str,
        rest_uri: str,
        token: str | None = None,
        credential: str | None = None,
        scope: str | None = None,
        default_schema: str | None = None,
    ):
        """
        Create a REST catalog from a connection string to a REST catalog.
        Either a token or a credential must be provided.
        Args:
            warehouse (str): The warehouse to connect to.
            rest_uri (str): The URI of the REST server.
            token (str): The token to use for authentication.
            credential (str): The credential to use for authentication.
            scope (str): The scope to use for authentication.
            default_schema (str): The default schema to use.
        """
        self.warehouse = warehouse
        self.rest_uri = rest_uri
        self.token = token
        self.credential = credential
        self.scope = scope
        self.default_schema = default_schema
        if self.token is None:
            self.token = self.get_token()

        # Set the token as an environment variable so that it can be accessed at runtime
        # Used by the RESTConnectionType
        os.environ["__BODOSQL_REST_TOKEN"] = self.token

    def get_java_object(self):
        return _create_java_REST_catalog(
            self.rest_uri,
            self.warehouse,
            self.token,
            self.credential,
            self.scope,
            self.default_schema,
        )

    @run_rank0
    def get_token(self):
        """
        Get the token for the REST catalog from a pyiceberg catalog.
        """
        con_str = get_conn_str(
            self.rest_uri, self.warehouse, scope=self.scope, credential=self.credential
        )
        py_catalog = conn_str_to_catalog(con_str)
        return py_catalog.properties["token"]

    def __eq__(self, other):
        if not isinstance(other, RESTCatalog):
            return False
        return self.warehouse == other.warehouse


@overload(RESTCatalog, no_unliteral=True)
def overload_REST_catalog_constructor(
    warehouse: str, token: str | None = None, credential: str | None = None
):
    raise_bodo_error("RESTCatalog: Cannot be created in JIT mode.")


class RESTCatalogType(DatabaseCatalogType):
    def __init__(
        self,
        warehouse: str,
        rest_uri: str,
        token: str | None = None,
        credential: str | None = None,
        scope: str | None = None,
        default_schema: str | None = None,
    ):
        """
        Create a REST catalog type from a connection string to a REST catalog.
        Args:
            warehouse (str): The warehouse to connect to.
            rest_uri (str): The URI of the REST server.
            token (str): The token to use for authentication.
            credential (str): The credential to use for authentication.
            scope (str): The scope to use for authentication.
            default_schema (str): The default schema to use.
        """
        self.warehouse = warehouse
        self.rest_uri = rest_uri
        self.token = token
        self.credential = credential
        self.scope = scope
        self.default_schema = default_schema

        super().__init__(
            name=f"RESTCatalogType({self.warehouse=},{self.rest_uri},{'token' if self.token is not None else 'credential'}=*****)",
        )

    def get_java_object(self):
        return _create_java_REST_catalog(
            self.rest_uri,
            self.warehouse,
            self.token,
            self.credential,
            self.scope,
            self.default_schema,
        )

    @property
    def key(self):
        return self.warehouse, self.rest_uri


@typeof_impl.register(RESTCatalog)
def typeof_REST_catalog(val, c):
    return RESTCatalogType(
        warehouse=val.warehouse,
        rest_uri=val.rest_uri,
        token=val.token,
        credential=val.credential,
        scope=val.scope,
        default_schema=val.default_schema,
    )


register_model(RESTCatalogType)(models.OpaqueModel)


@box(RESTCatalogType)
def box_REST_catalog(typ, val, c):
    """
    Box a REST Catalog native representation into a Python object. We populate
    the contents based on typing information.
    """
    warehouse_obj = c.pyapi.from_native_value(
        types.unicode_type,
        c.context.get_constant_generic(c.builder, types.unicode_type, typ.warehouse),
        c.env_manager,
    )
    rest_uri_obj = c.pyapi.from_native_value(
        types.unicode_type,
        c.context.get_constant_generic(c.builder, types.unicode_type, typ.rest_uri),
        c.env_manager,
    )
    if typ.token is not None:
        token_obj = c.pyapi.from_native_value(
            types.unicode_type,
            c.context.get_constant_generic(c.builder, types.unicode_type, typ.token),
            c.env_manager,
        )
    else:
        token_obj = c.pyapi.make_none()

    if typ.credential is not None:
        credential_obj = c.pyapi.from_native_value(
            types.unicode_type,
            c.context.get_constant_generic(
                c.builder, types.unicode_type, typ.credential
            ),
            c.env_manager,
        )
    else:
        credential_obj = c.pyapi.make_none()

    if typ.scope is not None:
        scope_obj = c.pyapi.from_native_value(
            types.unicode_type,
            c.context.get_constant_generic(c.builder, types.unicode_type, typ.scope),
            c.env_manager,
        )
    else:
        scope_obj = c.pyapi.make_none()

    if typ.default_schema is not None:
        default_schema_obj = c.pyapi.from_native_value(
            types.unicode_type,
            c.context.get_constant_generic(
                c.builder, types.unicode_type, typ.default_schema
            ),
            c.env_manager,
        )
    else:
        default_schema_obj = c.pyapi.make_none()

    REST_catalog_obj = c.pyapi.unserialize(c.pyapi.serialize_object(RESTCatalog))
    res = c.pyapi.call_function_objargs(
        REST_catalog_obj,
        (
            warehouse_obj,
            rest_uri_obj,
            token_obj,
            credential_obj,
            scope_obj,
            default_schema_obj,
        ),
    )
    c.pyapi.decref(warehouse_obj)
    c.pyapi.decref(rest_uri_obj)
    c.pyapi.decref(token_obj)
    c.pyapi.decref(credential_obj)
    c.pyapi.decref(scope_obj)
    c.pyapi.decref(default_schema_obj)
    c.pyapi.decref(REST_catalog_obj)
    return res


@unbox(RESTCatalogType)
def unbox_REST_catalog(typ, val, c):
    """
    Unbox a REST Catalog Python object into its native representation.
    Since the actual model is opaque we can just generate a dummy.
    """
    return NativeValue(c.context.get_dummy_value())


@numba.jit
def get_conn_str(rest_uri, warehouse, scope=None, token=None, credential=None):
    """Get the connection string for a REST Iceberg catalog."""
    conn_str = f"iceberg+{rest_uri}?warehouse={warehouse}"
    if scope is not None:
        conn_str += f"&scope={scope}"
    if token is not None:
        conn_str += f"&token={token}"
    if credential is not None:
        conn_str += f"&credential={credential}"
    return conn_str


class RESTConnectionType(IcebergConnectionType):
    """
    Python class for storing the information
        needed to connect to a REST Iceberg catalog.
    Token is read from an environment variable so that it can be accessed at runtime.
    The compiler can get a connection string using the get_conn_str function.
    The runtime can get a connection string using the conn_str attribute.
    """

    def __init__(self, rest_uri, warehouse, scope):
        self.warehouse = warehouse
        token = os.getenv("__BODOSQL_REST_TOKEN")
        assert token is not None, (
            "RESTConnectionType: Expected __BODOSQL_REST_TOKEN to be defined"
        )

        self.conn_str = get_conn_str(rest_uri, warehouse, token=token, scope=scope)

        super().__init__(
            name=f"RESTConnectionType({warehouse=}, {rest_uri=}, conn_str=*********, {scope=})",
        )


@intrinsic(prefer_literal=True)
def _get_REST_connection(typingctx, rest_uri, warehouse, scope, conn_str):
    """Create a struct model for a RESTConnectionType from a uri, warehouse and connection string."""
    literal_rest_uri = get_literal_value(rest_uri)
    literal_warehouse = get_literal_value(warehouse)
    literal_scope = get_literal_value(scope)
    REST_connection_type = RESTConnectionType(
        literal_rest_uri, literal_warehouse, literal_scope
    )

    def codegen(context, builder, sig, args):
        """lowering code to initialize a RESTConnectionType"""
        REST_connection_type = sig.return_type
        REST_connection_struct = cgutils.create_struct_proxy(REST_connection_type)(
            context, builder
        )
        context.nrt.incref(builder, sig.args[3], args[3])
        REST_connection_struct.conn_str = args[3]
        return REST_connection_struct._getvalue()

    return REST_connection_type(rest_uri, warehouse, scope, conn_str), codegen


def get_REST_connection(rest_uri: str, warehouse: str, scope: str):
    pass


@overload(get_REST_connection)
def overload_get_REST_connection(rest_uri: str, warehouse: str, scope: str):
    """Overload for get_REST_connection that creates a RESTConnectionType."""

    def impl(rest_uri: str, warehouse: str, scope: str):  # pragma: no cover
        with bodo.no_warning_objmode(token="unicode_type"):
            token = os.getenv("__BODOSQL_REST_TOKEN", "")
        assert token != "", (
            "get_REST_connection: Expected __BODOSQL_REST_TOKEN to be defined"
        )
        conn_str = get_conn_str(rest_uri, warehouse, token=token, scope=scope)
        conn = _get_REST_connection(rest_uri, warehouse, scope, conn_str)
        return conn

    return impl


@register_model(RESTConnectionType)
class RESTConnectionTypeModel(models.StructModel):
    """Model for RESTConnectionType has one member, conn_str."""

    def __init__(self, dmm, fe_type):
        members = [
            ("conn_str", types.unicode_type),
        ]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(RESTConnectionType, "conn_str", "conn_str")
