from __future__ import annotations

import typing as pt

import pyarrow as pa
from pyiceberg.catalog import Catalog, PropertiesUpdateSummary
from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table import CommitTableResponse, CreateTableTransaction, Table
from pyiceberg.table.sorting import UNSORTED_SORT_ORDER, SortOrder
from pyiceberg.table.update import (
    TableRequirement,
    TableUpdate,
)
from pyiceberg.typedef import EMPTY_DICT, Identifier, Properties

from bodo_iceberg_connector.py4j_support import get_table_id

if pt.TYPE_CHECKING:
    from py4j.java_gateway import JavaObject


class JavaTable(Table):
    def __init__(
        self, identifier: Identifier, catalog: JavaCatalog, java_table: JavaObject
    ):
        # TODO: Construct temporary metadata and metadata_location and IO how?
        super().__init__(identifier, None, None, None, catalog)
        # Any functions need overloading
        self.java_table = java_table


class JavaCatalog(Catalog):
    """
    Wrapper class to jump from Python to Java for the class API
    """

    def __init__(self, java_obj: JavaObject):
        super().__init__("java_catalog")
        self.java_obj = java_obj

    def _wrap_identifier(self, identifier: str | Identifier) -> JavaObject:
        parsed_id = self.identifier_to_tuple(identifier)
        TableIdentifier = get_table_id()
        return pt.cast(JavaObject, TableIdentifier.of(*parsed_id))

    def load_table(self, identifier: str | Identifier) -> Table:
        java_id = self._wrap_identifier(identifier)
        java_table = self.java_obj.loadTable(java_id)
        return JavaTable(self.identifier_to_tuple(identifier), self, java_table)

    def table_exists(self, identifier: str | Identifier) -> bool:
        java_id = self._wrap_identifier(identifier)
        return self.java_obj.tableExists(java_id)

    def create_table(
        self,
        identifier: str | Identifier,
        schema: pa.Schema | Schema,
        location: str | None = None,
        partition_spec: PartitionSpec = UNPARTITIONED_PARTITION_SPEC,
        sort_order: SortOrder = UNSORTED_SORT_ORDER,
        properties: Properties = EMPTY_DICT,
    ) -> Table:
        pass

    def create_table_transaction(
        self,
        identifier: str | Identifier,
        schema: Schema | pa.Schema,
        location: str | None = None,
        partition_spec: PartitionSpec = UNPARTITIONED_PARTITION_SPEC,
        sort_order: SortOrder = UNSORTED_SORT_ORDER,
        properties: Properties = EMPTY_DICT,
    ) -> CreateTableTransaction:
        pass

    def register_table(
        self, identifier: str | Identifier, metadata_location: str
    ) -> Table:
        pass

    def drop_table(self, identifier: str | Identifier) -> None:
        pass

    def purge_table(self, identifier: str | Identifier) -> None:
        pass

    def rename_table(
        self, from_identifier: str | Identifier, to_identifier: str | Identifier
    ) -> Table:
        pass

    def commit_table(
        self,
        table: Table,
        requirements: tuple[TableRequirement, ...],
        updates: tuple[TableUpdate, ...],
    ) -> CommitTableResponse:
        pass

    def create_namespace(
        self, namespace: str | Identifier, properties: Properties = EMPTY_DICT
    ) -> None:
        pass

    def drop_namespace(self, namespace: str | Identifier) -> None:
        pass

    def list_tables(self, namespace: str | Identifier) -> list[Identifier]:
        pass

    def list_namespaces(self, namespace: str | Identifier = ()) -> list[Identifier]:
        pass

    def list_views(self, namespace: str | Identifier) -> list[Identifier]:
        pass

    def load_namespace_properties(self, namespace: str | Identifier) -> Properties:
        pass

    def update_namespace_properties(
        self,
        namespace: str | Identifier,
        removals: set[str] | None = None,
        updates: Properties = EMPTY_DICT,
    ) -> PropertiesUpdateSummary:
        pass

    def drop_view(self, identifier: str | Identifier) -> None:
        pass
