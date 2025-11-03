"""JIT extension for Snowflake catalog type."""

from __future__ import annotations

from numba.core import types
from numba.core.imputils import lower_constant
from numba.core.typing import signature
from numba.extending import (
    NativeValue,
    box,
    intrinsic,
    models,
    overload,
    register_model,
    typeof_impl,
    unbox,
)

from bodo.utils.typing import (
    get_literal_value,
    is_literal_type,
    is_overload_none,
    raise_bodo_error,
)
from bodosql.bodosql_types.database_catalog_ext import DatabaseCatalogType
from bodosql.bodosql_types.snowflake_catalog import (
    SnowflakeCatalog,
    _create_java_snowflake_catalog,
    _validate_constructor_args,
)


class SnowflakeCatalogType(DatabaseCatalogType):
    """JIT class for storing the account information
    needed to connect to a remote Snowflake account from
    Java.
    """

    def __init__(
        self,
        username,
        password,
        account,
        warehouse,
        database,
        connection_params,
        iceberg_volume,
    ):
        _validate_constructor_args(
            username,
            password,
            account,
            warehouse,
            database,
            connection_params,
            iceberg_volume,
        )
        if connection_params is None:
            connection_params = {}
        self.connection_params = connection_params
        self.username = username
        self.password = password
        self.account = account
        self.warehouse = warehouse
        self.database = database
        self.connection_params = connection_params
        self.iceberg_volume = iceberg_volume
        super().__init__(
            # We omit the password in case the type is printed.
            name=f"SnowflakeCatalogType(username={username}, password=*******, account={account}, warehouse={warehouse}, database={database}, connection_params={connection_params}, iceberg_volume={iceberg_volume})"
        )

    @property
    def key(self):
        """Key used for caching. We use this because the password is omitted from
        the name.
        """
        return (
            self.username,
            self.password,
            self.account,
            self.warehouse,
            self.database,
            tuple(self.connection_params.items()),
            self.iceberg_volume,
        )

    def get_java_object(self):
        return _create_java_snowflake_catalog(
            self.username,
            self.password,
            self.account,
            self.warehouse,
            self.database,
            self.connection_params,
            self.iceberg_volume,
        )


@typeof_impl.register(SnowflakeCatalog)
def typeof_snowflake_catalog(val, c):
    return SnowflakeCatalogType(
        val.username,
        val.password,
        val.account,
        val.warehouse,
        val.database,
        val.connection_params,
        val.iceberg_volume,
    )


# Define the data model for the SnowflakeCatalog as opaque.
register_model(SnowflakeCatalogType)(models.OpaqueModel)


@box(SnowflakeCatalogType)
def box_snowflake_catalog(typ, val, c):
    """
    Box a snowflake catalog into a Python object. We populate
    the contents based on typing information.
    """
    # Load constants from the type.
    username_obj = c.pyapi.from_native_value(
        types.unicode_type,
        c.context.get_constant_generic(c.builder, types.unicode_type, typ.username),
        c.env_manager,
    )
    password_obj = c.pyapi.from_native_value(
        types.unicode_type,
        c.context.get_constant_generic(c.builder, types.unicode_type, typ.password),
        c.env_manager,
    )
    account_obj = c.pyapi.from_native_value(
        types.unicode_type,
        c.context.get_constant_generic(c.builder, types.unicode_type, typ.account),
        c.env_manager,
    )
    warehouse_obj = c.pyapi.from_native_value(
        types.unicode_type,
        c.context.get_constant_generic(c.builder, types.unicode_type, typ.warehouse),
        c.env_manager,
    )
    database_obj = c.pyapi.from_native_value(
        types.unicode_type,
        c.context.get_constant_generic(c.builder, types.unicode_type, typ.database),
        c.env_manager,
    )
    if typ.iceberg_volume is not None:
        iceberg_volume_obj = c.pyapi.from_native_value(
            types.unicode_type,
            c.context.get_constant_generic(
                c.builder, types.unicode_type, typ.iceberg_volume
            ),
            c.env_manager,
        )
    else:
        iceberg_volume_obj = c.pyapi.make_none()

    # Lowering a constant dictionary doesn't appear to work, so we construct
    # the dictionary in Python from the key + value pairs.
    keys = list(typ.connection_params.keys())
    values = list(typ.connection_params.values())
    keys_obj = c.pyapi.from_native_value(
        types.List(types.unicode_type),
        c.context.get_constant_generic(c.builder, types.List(types.unicode_type), keys),
        c.env_manager,
    )
    values_obj = c.pyapi.from_native_value(
        types.List(types.unicode_type),
        c.context.get_constant_generic(
            c.builder, types.List(types.unicode_type), values
        ),
        c.env_manager,
    )

    zip_obj = c.pyapi.unserialize(c.pyapi.serialize_object(zip))

    zipped_obj = c.pyapi.call_function_objargs(zip_obj, (keys_obj, values_obj))
    dict_obj = c.pyapi.unserialize(c.pyapi.serialize_object(dict))
    connection_params_obj = c.pyapi.call_function_objargs(dict_obj, (zipped_obj,))

    snowflake_catalog_obj = c.pyapi.unserialize(
        c.pyapi.serialize_object(SnowflakeCatalog)
    )
    res = c.pyapi.call_function_objargs(
        snowflake_catalog_obj,
        (
            username_obj,
            password_obj,
            account_obj,
            warehouse_obj,
            database_obj,
            connection_params_obj,
            iceberg_volume_obj,
        ),
    )
    c.pyapi.decref(snowflake_catalog_obj)
    c.pyapi.decref(username_obj)
    c.pyapi.decref(password_obj)
    c.pyapi.decref(account_obj)
    c.pyapi.decref(warehouse_obj)
    c.pyapi.decref(database_obj)
    c.pyapi.decref(iceberg_volume_obj)
    c.pyapi.decref(connection_params_obj)
    c.pyapi.decref(dict_obj)
    c.pyapi.decref(zipped_obj)
    c.pyapi.decref(zip_obj)
    c.pyapi.decref(values_obj)
    c.pyapi.decref(keys_obj)
    return res


@unbox(SnowflakeCatalogType)
def unbox_snowflake_catalog(typ, val, c):
    """
    Unbox a Snowflake Catalog Python object into its native representation.
    Since the actual model is opaque we can just generate a dummy.
    """
    return NativeValue(c.context.get_dummy_value())


@lower_constant(SnowflakeCatalogType)
def lower_constant_table_path(context, builder, ty, pyval):
    """
    Support lowering a SnowflakeCatalog as a constant. Since
    the actual model is opaque we can just generate a dummy.
    """
    return context.get_dummy_value()


# Create the JIT constructor
@overload(SnowflakeCatalog, no_unliteral=True)
def overload_snowflake_catalog_constructor(
    username,
    password,
    account,
    warehouse,
    database,
    connection_params=None,
    iceberg_volume=None,
):
    """
    SnowflakeCatalog Constructor to enable creating the catalog directly
    inside JIT code. This is not the intended usage but should still be supported
    for parity.
    """

    def impl(
        username,
        password,
        account,
        warehouse,
        database,
        connection_params=None,
        iceberg_volume=None,
    ):  # pragma: no cover
        return init_snowflake_connector(
            username,
            password,
            account,
            warehouse,
            database,
            connection_params,
            iceberg_volume,
        )

    return impl


@intrinsic(prefer_literal=True)
def init_snowflake_connector(
    typingctx,
    username,
    password,
    account,
    warehouse,
    database,
    connection_params,
    iceberg_volume,
):
    """
    Intrinsic used to actually construct the SnowflakeCatalog from the constructor.
    """
    # Check for literals
    if not is_literal_type(username):
        raise_bodo_error(
            "bodosql.SnowflakeCatalog(): 'username' must be a constant string"
        )
    if not is_literal_type(password):
        raise_bodo_error(
            "bodosql.SnowflakeCatalog(): 'password' must be a constant string"
        )
    if not is_literal_type(account):
        raise_bodo_error(
            "bodosql.SnowflakeCatalog(): 'account' must be a constant string"
        )
    if not is_literal_type(warehouse):
        raise_bodo_error(
            "bodosql.SnowflakeCatalog(): 'warehouse' must be a constant string"
        )
    if not is_literal_type(database):
        raise_bodo_error(
            "bodosql.SnowflakeCatalog(): 'database' must be a constant string"
        )
    if not is_literal_type(connection_params):
        raise_bodo_error(
            "bodosql.SnowflakeCatalog(): 'connection_params' must be a constant Dict[str, str] if provided"
        )
    if not (is_overload_none(iceberg_volume) or is_literal_type(iceberg_volume)):
        raise_bodo_error(
            "bodosql.SnowflakeCatalog(): 'iceberg_volume' must be None or a constant string"
        )

    # Extract the literal values
    username_lit = get_literal_value(username)
    password_lit = get_literal_value(password)
    account_lit = get_literal_value(account)
    warehouse_lit = get_literal_value(warehouse)
    database_lit = get_literal_value(database)
    connection_params_lit = get_literal_value(connection_params)
    iceberg_volume_lit = get_literal_value(iceberg_volume)

    # Construct the output type
    ret_type = SnowflakeCatalogType(
        username_lit,
        password_lit,
        account_lit,
        warehouse_lit,
        database_lit,
        connection_params_lit,
        iceberg_volume_lit,
    )

    def codegen(context, builder, signature, args):  # pragma: no cover
        # We just return a dummy since this has an opaque model.
        return context.get_dummy_value()

    return (
        signature(
            ret_type,
            username,
            password,
            account,
            warehouse,
            database,
            connection_params,
            iceberg_volume,
        ),
        codegen,
    )
