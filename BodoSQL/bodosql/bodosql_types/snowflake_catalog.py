"""Python and JIT class for describing a Snowflake catalog. A snowflake
catalog contains all information needed to connect to a Snowflake account
in Java and load relevant schema information.
"""

from __future__ import annotations

from copy import deepcopy

from bodo.io.snowflake import parse_conn_str
from bodosql import DatabaseCatalog
from bodosql.imported_java_classes import JavaEntryPoint, build_java_properties


def _validate_constructor_args(
    username: str,
    password: str,
    account: str,
    warehouse: str,
    database: str,
    connection_params: dict[str, str] | None,
    iceberg_volume: str | None,
):
    """Validate

    Args:
        username (str): Snowflake username
        password (str): Snowflake password
        account (str): Snowflake account
        warehouse (str): Snowflake warehouse type
        database (str): Snowflake database name to use.
        connection_params (Optional[Dict[str, str]]): Any additional connection parameters to provide.
        iceberg_volume (Optional[str]): Snowflake external volume (e.g. S3/ADLS) for writing Iceberg data if available
    """
    if not isinstance(username, str):
        raise ValueError(
            f"SnowflakeCatalog(): 'username' argument must be a constant string. Found {type(username)}."
        )
    if not isinstance(password, str):
        raise ValueError(
            f"SnowflakeCatalog(): 'password' argument must be a constant string. Found {type(password)}."
        )
    if not isinstance(account, str):
        raise ValueError(
            f"SnowflakeCatalog(): 'account' argument must be a constant string. Found {type(account)}."
        )
    if not isinstance(warehouse, str):
        raise ValueError(
            f"SnowflakeCatalog(): 'warehouse' argument must be a constant string. Found {type(warehouse)}."
        )
    if not isinstance(database, str):
        raise ValueError(
            f"SnowflakeCatalog(): 'database' argument must be a constant string. Found {type(database)}."
        )
    is_str_dict = isinstance(connection_params, dict) and all(
        isinstance(k, str) and isinstance(v, str) for k, v in connection_params.items()
    )
    if not (connection_params is None or is_str_dict):
        raise ValueError(
            "SnowflakeCatalog(): 'connection_params' argument must be a Dict[str, str] if provided."
        )
    if iceberg_volume is not None and not isinstance(iceberg_volume, str):
        raise ValueError(
            f"SnowflakeCatalog(): 'iceberg_volume' argument must be a constant string. Found {type(iceberg_volume)}."
        )


def _create_java_snowflake_catalog(
    username: str,
    password: str,
    account: str,
    warehouse: str,
    database: str,
    connection_params: dict[str, str],
    iceberg_volume: str | None,
):
    """Create a SnowflakeCatalog Java object
    from the given parameters.

    Args:
        username (str): Snowflake username
        password (str): Snowflake password
        account (str): Snowflake account
        warehouse (str): Snowflake warehouse
        database (str): Snowflake database to use.
        connection_params (Dict[str, str]): Any optional connection parameters
        to pass.
        iceberg_volume (Optional[str]): Snowflake external volume (e.g. S3/ADLS) for writing Iceberg data if available
    """
    # Create a properties object to pass parameters. Account
    # and database are not included because they are needed
    # directly in the Java Snowflake Catalog constructor.
    properties = build_java_properties(connection_params)
    # Create the Snowflake catalog
    return JavaEntryPoint.buildSnowflakeCatalog(
        username, password, account, database, warehouse, properties, iceberg_volume
    )


class SnowflakeCatalog(DatabaseCatalog):
    """Python class for storing the account information
    needed to connect to a remote Snowflake account from
    Java.
    """

    def __init__(
        self,
        username: str,
        password: str,
        account: str,
        warehouse: str,
        database: str,
        connection_params: dict[str, str] | None = None,
        iceberg_volume: str | None = None,
    ):
        """Constructor for the Snowflake catalog. The required arguments
        are based on the information that should be made available when
        registering with the Bodo platform. The design is described here:

        https://bodo.atlassian.net/wiki/spaces/BodoSQL/pages/1097859073/Bodo+Design+Changes
        """
        _validate_constructor_args(
            username,
            password,
            account,
            warehouse,
            database,
            connection_params,
            iceberg_volume,
        )
        self.username = username
        self.password = password
        self.account = account
        self.warehouse = warehouse
        self.database = database
        if connection_params is None:
            connection_params = {}
        else:
            # Create a deepcopy to prevent any unexpected changes
            # after validation.
            connection_params = deepcopy(connection_params)
        self.connection_params = connection_params
        self.iceberg_volume = iceberg_volume

    @classmethod
    def from_conn_str(cls, conn_str: str) -> SnowflakeCatalog:
        conn_contents = parse_conn_str(conn_str, strict_parsing=True)
        ref_str = "See https://docs.snowflake.com/developer-guide/python-connector/sqlalchemy#connection-parameters for constructing a connection URL."

        # Parse Required Parameters Out of conn_contents
        # TODO: Output of parse_conn_str is better as NamedTuple
        # But what argument are required for Snowflake SQLAlchemy
        # Snowflake Docs have more details
        if (username := conn_contents.pop("user", None)) is None:
            raise ValueError(
                f"SnowflakeCatalog.from_conn_str: `conn_str` must contain a user login name. {ref_str}"
            )

        if (password := conn_contents.pop("password", None)) is None:
            password = ""
        if (account := conn_contents.pop("account", None)) is None:
            raise ValueError(
                f"SnowflakeCatalog.from_conn_str: `conn_str` must contain an an account identifier or URL. {ref_str}"
            )
        if (warehouse := conn_contents.pop("warehouse", None)) is None:
            raise ValueError(
                f"SnowflakeCatalog.from_conn_str: `conn_str` must contain a warehouse name as an additional connection parameter. {ref_str}"
            )
        if (database := conn_contents.pop("database", None)) is None:
            raise ValueError(
                f"SnowflakeCatalog.from_conn_str: `conn_str` must contain a database name in the URI path. {ref_str}"
            )

        iceberg_volume = conn_contents.pop("iceberg_volume", None)

        # Remaining parameters in conn_contents are still passed in
        # Example: schema, role_name, etc
        # TODO: Does BodoSQL support session_parameters (its a dict, but can it be flattened?)
        return cls(
            username,
            password,
            account,
            warehouse,
            database,
            connection_params=conn_contents,
            iceberg_volume=iceberg_volume,
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

    # Define == for testing
    def __eq__(self, other: object) -> bool:
        if isinstance(other, SnowflakeCatalog):
            return (
                self.username == other.username
                and self.password == other.password
                and self.account == other.account
                and self.warehouse == other.warehouse
                and self.database == other.database
                and self.connection_params == other.connection_params
                and self.iceberg_volume == other.iceberg_volume
            )
        return False
