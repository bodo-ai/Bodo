"""Python and JIT class for describing a S3 Tables Iceberg catalog. A S3 Tables
catalog contains all information needed to connect to S3 Tables for organizing and modifying tables.
"""

from bodosql import DatabaseCatalog
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
