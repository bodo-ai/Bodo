"""Python and JIT class for describing a Filesystem catalog. A Filesystem
catalog contains all information needed to connect use the Hadoop, POSIX or S3 Filesystem
for organizing tables.
"""

from bodosql import DatabaseCatalog
from bodosql.imported_java_classes import JavaEntryPoint


def _create_java_filesystem_catalog(
    connection_string: str, default_write_format: str, default_schema: str
):
    """
    Create a Java FileSystemCatalog object.

    Args:
        connection_string (str): The connection string to the file system.
        default_write_format (str): What should be the default write format?
        default_schema (str): The default schema(s) to use the for the catalog.

    Returns:
        JavaObject: A Java FileSystemCatalog object.
    """
    return JavaEntryPoint.buildFileSystemCatalog(
        connection_string,
        default_write_format,
        default_schema,
    )


class FileSystemCatalog(DatabaseCatalog):
    """Python class for storing the information
    needed to treat a Hadoop, POSIX or S3 Filesystem as a catalog.
    """

    def __init__(
        self,
        connection_string: str,
        default_write_format: str = "iceberg",
        default_schema: str = ".",
    ):
        """
        Create a filesystem catalog from a connection string to a file system
        and an indication of how to write files.

        Args:
            connection_string (str): The connection string to the file system.
                Right now with the given constraints this can be a local file
                system or S3.
            default_write_format (str, optional): What should be the default write
                format? This can be either "parquet"/"pq" or "iceberg". Defaults to "iceberg".
            default_schema (str, optional): The default schema(s) to use the for the catalog.
                This value should be written as a dot separated compound identifier as if it was written
                directly in the SQL (e.g. schema1."schema2") and follows BodoSQL identifier rules
                on casing/quote escapes.
        """
        self.connection_string = connection_string
        self.default_write_format = self.standardize_write_format(default_write_format)
        self.default_schema = default_schema

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
            self.connection_string, self.default_write_format, self.default_schema
        )

    # Define == for testing
    def __eq__(self, other: object) -> bool:
        if isinstance(other, FileSystemCatalog):
            return (
                self.connection_string == other.connection_string
                and self.default_write_format == other.default_write_format
                and self.default_schema == other.default_schema
            )
        return False
