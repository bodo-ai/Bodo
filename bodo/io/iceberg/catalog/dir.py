"""
A simpler implementation of the Iceberg Java Hadoop Catalog.
Only supports loading, creating/replacing/appending to tables
from a directory structure.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from urllib.parse import urlparse

import pyarrow as pa
from pyiceberg.catalog import (
    WAREHOUSE_LOCATION,
    Catalog,
    MetastoreCatalog,
    PropertiesUpdateSummary,
)
from pyiceberg.exceptions import NoSuchTableError
from pyiceberg.io import load_file_io
from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.serializers import FromInputFile
from pyiceberg.table import (
    CommitTableResponse,
    CreateTableTransaction,
    StagedTable,
    Table,
    sorting,
)
from pyiceberg.table.metadata import new_table_metadata
from pyiceberg.table.update import (
    TableRequirement,
    TableUpdate,
    update_table_metadata,
)
from pyiceberg.typedef import EMPTY_DICT, Identifier, Properties


class DirCatalog(Catalog):
    def __init__(self, name: str, **properties: str):
        super().__init__(name, **properties)
        if WAREHOUSE_LOCATION not in self.properties:
            raise ValueError(f"Missing {WAREHOUSE_LOCATION} property")

    @cached_property
    def warehouse_path(self) -> str:
        path = self.properties[WAREHOUSE_LOCATION]
        # Convert relative paths to absolute paths if path is a local file path
        if urlparse(path).scheme == "":
            path = str(Path(path).resolve())
        return path

    def _table_path(self, identifier: Identifier) -> str:
        wh_path = self.warehouse_path.removesuffix("/")
        return f"{wh_path}/{'/'.join(identifier)}"

    def _load_table_and_version(
        self, identifier: str | Identifier
    ) -> tuple[Table, int]:
        iden = self.identifier_to_tuple(identifier)
        table_dir_path = self._table_path(iden)
        io = load_file_io(properties=self.properties)

        # Get metadata JSON file id from version-hint.text file
        version_hint_file = f"{table_dir_path}/metadata/version-hint.text"
        try:
            with io.new_input(version_hint_file).open(seekable=False) as f:
                version_hint = int(f.read().strip().decode())
        except FileNotFoundError as e:
            raise NoSuchTableError(
                f"No table with identifier {identifier} exists."
            ) from e

        # Load Table from metadata JSON file path
        metadata_loc = f"{table_dir_path}/metadata/v{version_hint}.metadata.json"
        metadata = FromInputFile.table_metadata(io.new_input(metadata_loc))

        return Table(
            identifier=iden,
            metadata=metadata,
            metadata_location=metadata_loc,
            io=load_file_io({**self.properties, **metadata.properties}),
            catalog=self,
        ), version_hint

    def load_table(self, identifier: str | Identifier) -> Table:
        table, _ = self._load_table_and_version(identifier)
        return table

    def table_exists(self, identifier: str | Identifier) -> bool:
        iden = self.identifier_to_tuple(identifier)
        table_dir_path = self._table_path(iden)
        io = load_file_io(properties=self.properties)

        version_hint_file = f"{table_dir_path}/metadata/version-hint.text"
        return io.new_input(version_hint_file).exists()

    def create_namespace(
        self, namespace: str | Identifier, properties: Properties = EMPTY_DICT
    ) -> None:
        # We don't support namespace properties
        if properties:
            raise NotImplementedError("Namespace properties are not supported")

        # Namespaces are just directories, so we don't need to do anything
        return

    def register_table(
        self, identifier: str | Identifier, metadata_location: str
    ) -> Table:
        # TODO
        raise NotImplementedError("register_table is not implemented yet")

    def create_table(
        self,
        identifier: str | Identifier,
        schema: Schema | pa.Schema,
        location: str | None = None,
        partition_spec: PartitionSpec = UNPARTITIONED_PARTITION_SPEC,
        sort_order: sorting.SortOrder = sorting.UNSORTED_SORT_ORDER,
        properties: Properties = EMPTY_DICT,
    ) -> Table:
        if location:
            raise NotImplementedError(
                "DirCatalog does not support user specified table locations."
            )
        ice_schema = self._convert_schema_if_needed(schema)
        iden = self.identifier_to_tuple(identifier)
        table_dir_path = self._table_path(iden)

        metadata = new_table_metadata(
            location=table_dir_path,
            schema=ice_schema,
            partition_spec=partition_spec,
            sort_order=sort_order,
            properties=properties,
        )

        # Write metadata file
        metadata_loc = f"{table_dir_path}/metadata/v1.metadata.json"
        io = load_file_io(self.properties, metadata_loc)
        MetastoreCatalog._write_metadata(metadata, io, metadata_loc)

        # Write version hint file
        version_hint_file = io.new_output(
            f"{table_dir_path}/metadata/version-hint.text"
        )
        with version_hint_file.create(overwrite=True) as f:
            f.write(b"1")

        return self.load_table(identifier=identifier)

    def drop_table(self, identifier: str | Identifier) -> None:
        # TODO
        raise NotImplementedError("drop_table is not implemented yet")

    def purge_table(self, identifier: str | Identifier) -> None:
        # TODO
        raise NotImplementedError("purge_table is not implemented yet")

    def commit_table(
        self,
        table: Table,
        requirements: tuple[TableRequirement, ...],
        updates: tuple[TableUpdate, ...],
    ) -> CommitTableResponse:
        # Get the current table metadata
        iden = table._identifier
        table_path = self._table_path(iden)
        try:
            current_table, curr_version = self._load_table_and_version(iden)
        except NoSuchTableError:
            current_table, curr_version = None, 0

        # Update and stage the table
        # Note, this is copied from the MetadataCatalog implementation
        # But DirCatalog should not inherit from MetadataCatalog
        for requirement in requirements:
            requirement.validate(current_table.metadata if current_table else None)

        updated_metadata = update_table_metadata(
            base_metadata=current_table.metadata
            if current_table
            else MetastoreCatalog._empty_table_metadata(),
            updates=updates,
            enforce_validation=current_table is None,
            metadata_location=current_table.metadata_location
            if current_table
            else None,
        )

        new_version = curr_version + 1
        new_metadata_location = f"{table_path}/metadata/v{new_version}.metadata.json"
        new_io = self._load_file_io(
            properties=updated_metadata.properties, location=new_metadata_location
        )
        updated_staged_table = StagedTable(
            identifier=iden,
            metadata=updated_metadata,
            metadata_location=new_metadata_location,
            io=new_io,
            catalog=self,
        )

        if current_table and updated_staged_table.metadata == current_table.metadata:
            # no changes, do nothing
            return CommitTableResponse(
                metadata=current_table.metadata,
                metadata_location=current_table.metadata_location,
            )

        # Write the updated metadata
        MetastoreCatalog._write_metadata(
            metadata=updated_staged_table.metadata,
            io=updated_staged_table.io,
            metadata_path=updated_staged_table.metadata_location,
        )

        # Update the version hint file
        version_file = new_io.new_output(f"{table_path}/metadata/version-hint.text")
        with version_file.create(overwrite=True) as f:
            f.write(str(new_version).encode())

        return CommitTableResponse(
            metadata=updated_staged_table.metadata,
            metadata_location=updated_staged_table.metadata_location,
        )

    # All of the following functions are impossible to implement with just a FileIO

    def list_tables(self, namespace: str | Identifier) -> list[Identifier]:
        raise NotImplementedError

    def list_namespaces(self, namespace: str | Identifier = ()) -> list[Identifier]:
        raise NotImplementedError

    def load_namespace_properties(self, namespace: str | Identifier) -> Properties:
        raise NotImplementedError

    def namespace_exists(self, namespace: str | Identifier) -> bool:
        raise NotImplementedError

    def drop_view(self, identifier: str | Identifier) -> None:
        raise NotImplementedError

    def rename_table(
        self, from_identifier: str | Identifier, to_identifier: str | Identifier
    ) -> Table:
        raise NotImplementedError

    def drop_namespace(self, namespace: str | Identifier) -> None:
        raise NotImplementedError

    def update_namespace_properties(
        self,
        namespace: str | Identifier,
        removals: set[str] | None = None,
        updates: Properties = EMPTY_DICT,
    ) -> PropertiesUpdateSummary:
        raise NotImplementedError

    def create_table_transaction(
        self,
        identifier: str | Identifier,
        schema: Schema | pa.Schema,
        location: str | None = None,
        partition_spec: PartitionSpec = UNPARTITIONED_PARTITION_SPEC,
        sort_order: sorting.SortOrder = sorting.UNSORTED_SORT_ORDER,
        properties: Properties = EMPTY_DICT,
    ) -> CreateTableTransaction:
        if location:
            raise NotImplementedError(
                "DirCatalog does not support user specified table locations."
            )
        ice_schema = self._convert_schema_if_needed(schema)
        iden = self.identifier_to_tuple(identifier)
        table_dir_path = self._table_path(iden)

        metadata = new_table_metadata(
            location=table_dir_path,
            schema=ice_schema,
            partition_spec=partition_spec,
            sort_order=sort_order,
            properties=properties,
        )

        # Write metadata file
        metadata_loc = f"{table_dir_path}/metadata/v1.metadata.json"
        io = load_file_io(self.properties, metadata_loc)

        return CreateTableTransaction(
            StagedTable(
                identifier=iden,
                metadata=metadata,
                metadata_location=metadata_loc,
                io=io,
                catalog=self,
            )
        )

    def list_views(self, namespace: str | Identifier) -> list[Identifier]:
        raise NotImplementedError("Views are not supported in DirCatalog")

    def view_exists(self, identifier: str | Identifier) -> bool:
        raise NotImplementedError("Views are not supported in DirCatalog")
