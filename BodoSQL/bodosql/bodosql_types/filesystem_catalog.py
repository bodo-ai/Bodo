"""Python and JIT class for describing a Filesystem catalog. A Filesystem
catalog contains all information needed to connect use the Hadoop, POSIX or S3 Filesystem
for organizing tables.
"""
# Copyright (C) 2024 Bodo Inc. All rights reserved.

from numba.core import types
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

from bodosql import DatabaseCatalog, DatabaseCatalogType
from bodosql.imported_java_classes import FileSystemCatalogClass, WriteTargetEnum


def _create_java_filesystem_catalog(connection_string: str, default_write_format: str):
    """
    Create a Java FileSystemCatalog object.

    Args:
        connection_string (str): The connection string to the file system.
        default_write_format (str): What should be the default write format?

    Returns:
        JavaObject: A Java FileSystemCatalog object.
    """
    return FileSystemCatalogClass(
        connection_string, WriteTargetEnum.fromString(default_write_format)
    )


class FileSystemCatalog(DatabaseCatalog):
    """Python class for storing the information
    needed to treat a Hadoop, POSIX or S3 Filesystem as a catalog.
    """

    def __init__(self, connection_string: str, default_write_format: str = "iceberg"):
        """
        Create a filesystem catalog from a connection string to a file system
        and an indication of how to write files.

        Args:
            connection_string (str): The connection string to the file system.
                Right now with the given constraints this can be a local file
                system or S3.
            default_write_format (str, optional): What should be the default write
                format? This can be either "parquet"/"pq" or "iceberg". Defaults to "iceberg".
        """
        self.connection_string = connection_string
        self.default_write_format = self.standardize_write_format(default_write_format)

    @staticmethod
    def standardize_write_format(write_format: str) -> str:
        """
        Convert a write format to a standard format.

        Args:
            write_format (str): The desired output write format.

        Returns:
            str: The write format converted to a standard format.
        """
        write_format = write_format.lower()
        if write_format in ["parquet", "pq"]:
            return "parquet"
        elif write_format == "iceberg":
            return "iceberg"
        else:
            raise ValueError(f"Unknown write format {write_format}")

    def get_java_object(self):
        return _create_java_filesystem_catalog(
            self.connection_string, self.default_write_format
        )

    # Define == for testing
    def __eq__(self, other: object) -> bool:
        if isinstance(other, FileSystemCatalog):
            return (
                self.connection_string == other.connection_string
                and self.default_write_format == other.default_write_format
            )
        return False


class FileSystemCatalogType(DatabaseCatalogType):
    """JIT class for storing the information
    needed to treat a Hadoop/POSIX/S3 Filesystem as a catalog.
    """

    def __init__(self, connection_string: str, default_write_format: str):
        """
        Create a filesystem catalog from a connection string to a file system
        and an indication of how to write files.

        Args:
            connection_string (str): The connection string to the file system.
                Right now with the given constraints this can be a local file
                system or S3.
            default_write_format (str, optional): What should be the default write
                format? This should already be standardized by Python.
        """
        self.connection_string = connection_string
        self.default_write_format = default_write_format
        super(FileSystemCatalogType, self).__init__(
            name=f"FileSystemCatalogType(connection_string={connection_string}, default_write_format={default_write_format})"
        )

    def get_java_object(self):
        return _create_java_filesystem_catalog(
            self.connection_string, self.default_write_format
        )


@typeof_impl.register(FileSystemCatalog)
def typeof_snowflake_catalog(val, c):
    return FileSystemCatalogType(val.connection_string, val.default_write_format)


# Define the data model for the FileSystemCatalog as opaque.
register_model(FileSystemCatalogType)(models.OpaqueModel)


@box(FileSystemCatalogType)
def box_filesystem_catalog(typ, val, c):
    """
    Box a FileSystem catalog into a Python object. We populate
    the contents based on typing information.
    """
    # Load constants from the type.
    connection_string_obj = c.pyapi.from_native_value(
        types.unicode_type,
        c.context.get_constant_generic(
            c.builder, types.unicode_type, typ.connection_string
        ),
        c.env_manager,
    )
    default_write_format_obj = c.pyapi.from_native_value(
        types.unicode_type,
        c.context.get_constant_generic(
            c.builder, types.unicode_type, typ.default_write_format
        ),
        c.env_manager,
    )

    filesystem_catalog_obj = c.pyapi.unserialize(
        c.pyapi.serialize_object(FileSystemCatalog)
    )
    res = c.pyapi.call_function_objargs(
        filesystem_catalog_obj,
        (
            connection_string_obj,
            default_write_format_obj,
        ),
    )
    c.pyapi.decref(filesystem_catalog_obj)
    c.pyapi.decref(connection_string_obj)
    c.pyapi.decref(default_write_format_obj)
    return res


@unbox(FileSystemCatalogType)
def unbox_filesystem_catalog(typ, val, c):
    """
    Unbox a FileSystem Catalog Python object into its native representation.
    Since the actual model is opaque we can just generate a dummy.
    """
    return NativeValue(c.context.get_dummy_value())
