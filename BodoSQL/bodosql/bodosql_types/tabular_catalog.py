"""Python and JIT class for describing a Tabular Iceberg catalog. A Tabular 
catalog contains all information needed to connect use Tabular Iceberg catalog for organizing and modifying tables.
"""
# Copyright (C) 2024 Bodo Inc. All rights reserved.
from numba.core import types
from numba.extending import (
    NativeValue,
    box,
    models,
    register_model,
    typeof_impl,
    unbox,
)

from bodosql import DatabaseCatalog, DatabaseCatalogType
from bodosql.imported_java_classes import TabularCatalogClass


def _create_java_tabular_catalog(
    warehouse: str, token: str | None, credential: str | None
):
    """
    Create a Java TabularCatalog object.
    Args:
        warehouse (str): The warehouse to connect to.
        token (str): The token to use for authentication.
        credential (str): The credential to use for authentication.
    Returns:
        JavaObject: A Java TabularCatalog object.
    """
    return TabularCatalogClass(
        warehouse,
        token,
        credential,
    )


class TabularCatalog(DatabaseCatalog):
    """
    Python class for storing the information
        needed to connect to a Tabular Iceberg catalog.
    """

    def __init__(
        self,
        warehouse: str,
        token: str | None = None,
        credential: str | None = None,
    ):
        """
        Create a tabular catalog from a connection string to a tabular catalog.
        Either a token or a credential must be provided.
        Args:
            warehouse (str): The warehouse to connect to.
            token (str): The token to use for authentication.
            credential (str): The credential to use for authentication.
        """
        self.warehouse = warehouse
        self.token = token
        self.credential = credential

    def get_java_object(self):
        return _create_java_tabular_catalog(self.warehouse, self.token, self.credential)

    def __eq__(self, other):
        if not isinstance(other, TabularCatalog):
            return False
        return self.warehouse == other.warehouse


class TabularCatalogType(DatabaseCatalogType):
    def __init__(
        self, warehouse: str, token: str | None = None, credential: str | None = None
    ):
        """
        Create a tabular catalog type from a connection string to a tabular catalog.
        Args:
            warehouse (str): The warehouse to connect to.
            token (str): The token to use for authentication.
            credential (str): The credential to use for authentication.
        """
        self.warehouse = warehouse
        self.token = token
        self.credential = credential

        super(TabularCatalogType, self).__init__(
            name=f"TabularCatalogType({self.warehouse=},{'token' if self.token is not None else 'credential'}=*****)",
        )

    def get_java_object(self):
        return _create_java_tabular_catalog(self.warehouse, self.token, self.credential)

    @property
    def key(self):
        return self.warehouse


@typeof_impl.register(TabularCatalog)
def typeof_tabular_catalog(val, c):
    return TabularCatalogType(
        warehouse=val.warehouse, token=val.token, credential=val.credential
    )


register_model(TabularCatalogType)(models.OpaqueModel)


@box(TabularCatalogType)
def box_tabular_catalog(typ, val, c):
    """
    Box a Tabular Catalog native representation into a Python object. We populate
    the contents based on typing information.
    """
    warehouse_obj = c.pyapi.from_native_value(
        types.unicode_type,
        c.context.get_constant_generic(c.builder, types.unicode_type, typ.warehouse),
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

    tabular_catalog_obj = c.pyapi.unserialize(c.pyapi.serialize_object(TabularCatalog))
    res = c.pyapi.call_function_objargs(
        tabular_catalog_obj,
        (
            warehouse_obj,
            token_obj,
            credential_obj,
        ),
    )
    c.pyapi.decref(warehouse_obj)
    c.pyapi.decref(token_obj)
    c.pyapi.decref(credential_obj)
    c.pyapi.decref(tabular_catalog_obj)
    return res


@unbox(TabularCatalogType)
def unbox_tabular_catalog(typ, val, c):
    """
    Unbox a Tabular Catalog Python object into its native representation.
    Since the actual model is opaque we can just generate a dummy.
    """
    return NativeValue(c.context.get_dummy_value())
