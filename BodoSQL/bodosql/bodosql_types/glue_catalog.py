"""Python and JIT class for describing a Glue Iceberg catalog. A Glue
catalog contains all information needed to connect use Glue Iceberg catalog for organizing and modifying tables.
"""

from __future__ import annotations

from bodosql import DatabaseCatalog
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
