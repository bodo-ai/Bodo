from abc import ABC, abstractmethod

import pandas as pd

from bodo.spawn.utils import run_rank0
from bodo.tests.utils import gen_unique_table_id


class DDLTestHarness(ABC):
    """
    Abstract base class for DDL test harnesses.

    This class defines the interface for implementing DDL test harnesses for different catalogs.
    Subclasses should implement the abstract methods to provide catalog-specific functionality.

    """

    #######################
    #   Abstract methods  #
    #######################

    """
    These methods are catalog specific and should be implemented in the specific test harness.
    """

    @abstractmethod
    def run_bodo_query(self, query: str) -> pd.DataFrame:
        """
        Executes a query on the Bodo catalog and returns the output as a pandas dataframe.
        """
        pass

    @abstractmethod
    def run_spark_query(self, query: str) -> pd.DataFrame:
        """
        Executes a query on the Spark catalog and returns the output as a pandas dataframe.
        """
        pass

    @abstractmethod
    def get_table_identifier(
        self, table_name: str, db_schema: str | None = None
    ) -> str:
        """Converts a table name into a table identifier string.
           Exact format differs per catalog.

        Args:
            table_name (str): The name of the table.
            db_schema (str, optional): Name of schema that table is in. If not specified, defaults to catalog's default schema.

        Returns:
            str: Table identifier string (e.g. "schema"."table_name")
        """
        pass

    @abstractmethod
    def create_test_table(self, table_identifier: str) -> None:
        """
        Create a test table with the given table identifier.

        Args:
            table_identifier (str): The identifier for the test table.
        """
        pass

    @abstractmethod
    def drop_test_table(self, table_identifier: str) -> None:
        """
        Drops the specified test table.

        Args:
            table_identifier (str): The identifier of the table to be dropped.

        Returns:
            None
        """
        pass

    @abstractmethod
    def check_table_exists(self, table_identifier: str) -> bool:
        """
        Checks if a table exists in the database.

        Args:
            table_identifier (str): The identifier of the table to check.

        Returns:
            bool: True if the table exists, False otherwise.
        """
        pass

    @abstractmethod
    def show_table_properties(self, table_identifier: str) -> pd.DataFrame:
        """
        Retrieves the properties of a table using `SHOW TBLPROPERTIES`.

        Args:
            table_identifier (str): The identifier of the table.

        Returns:
            pd.DataFrame: A dataframe containing the properties of the table.
        """
        pass

    @abstractmethod
    def describe_table_extended(self, table_identifier: str) -> pd.DataFrame:
        """
        Retrieves extended information about a table using `DESCRIBE TABLE EXTENDED`.
        Refreshes the table metadata before describing the table.

        Args:
            table_identifier (str): The identifier of the table.

        Returns:
            pd.DataFrame: A dataframe containing the extended information about the table.
        """
        pass

    @abstractmethod
    def refresh_table(self, table_identifier: str) -> None:
        """
        Refreshes the metadata of a table using `spark.catalog.refreshTable`.

        Args:
            table_identifier (str): The identifier of the table.

        Returns:
            None
        """
        pass

    @abstractmethod
    def describe_table(self, table_identifier: str, spark=False) -> pd.DataFrame:
        """
        Retrieves the description of a table using `DESCRIBE TABLE`.
        Refreshes the table metadata before describing the table.

        Args:
            table_identifier (str): The identifier of the table.
            spark (bool): If True, use Spark to describe the table. Otherwise, use Bodo.

        Returns:
            pd.DataFrame: A dataframe containing the description of the table.
        """
        pass

    @abstractmethod
    def create_test_view(self, view_identifier: str) -> None:
        """
        Create a test view with the given view identifier.

        Args:
            view_identifier (str): The identifier for the test view.
        """
        pass

    @abstractmethod
    def drop_test_view(self, view_identifier: str) -> None:
        """
        Drops the specified test view.

        Args:
            view_identifier (str): The identifier of the view to be dropped.

        Returns:
            None
        """
        pass

    @abstractmethod
    def check_view_exists(self, table_name: str) -> bool:
        """
        Checks if a view exists in the database.

        Args:
            table_name (str): The name of the view to check.

        Returns:
            bool: True if the view exists, False otherwise.
        """
        pass

    @abstractmethod
    def check_schema_exists(self, schema_name: str) -> bool:
        """
        Checks if a schema exists in the database.

        Args:
            schema_name (str): The name of the schema to check.

        Returns:
            bool: True if the schema exists, False otherwise.
        """
        pass

    #######################
    #   Helper functions  #
    #######################

    def check_row_exists(self, output, row):
        """
        Helper function to check if a row exists in the output dataframe.
        """
        return ((output["key"] == row["key"]) & (output["value"] == row["value"])).any()

    @run_rank0
    def gen_unique_id(self, prefix: str):
        """
        Generates a unique id by appending a unique number to the prefix.

        Args:
            prefix (str): The prefix for the unique id.

        Returns:
            str: The generated unique id.
        """
        return gen_unique_table_id(prefix)
