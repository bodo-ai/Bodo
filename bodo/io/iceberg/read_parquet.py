"""
Operations related to reading a Parquet dataset from an Iceberg table.
For example, getting Parquet metadata, applying pushdowns, estimating
the size for distributed read planning, and actually reading the dataset.
"""

from __future__ import annotations

import os
import re
import time
import typing as pt
import warnings
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from mpi4py import MPI

import bodo
from bodo import BodoWarning
from bodo.io import arrow_cpp
from bodo.io.iceberg.common import (
    FieldID,
    FieldIDs,
    FieldName,
    FieldNames,
    IcebergParquetInfo,
    SchemaGroupIdentifier,
    b_ICEBERG_FIELD_ID_MD_KEY,
)
from bodo.io.iceberg.read_schema_evo import (
    validate_file_schema_compatible_with_read_schema,
)
from bodo.io.parquet_pio import (
    REMOTE_FILESYSTEMS,
    filter_row_groups_from_start_of_dataset,
    filter_row_groups_from_start_of_dataset_heuristic,
    schema_with_dict_cols,
)

if pt.TYPE_CHECKING:  # pragma: no cover
    import pyarrow.fs as pa_fs
    from pyarrow._dataset import Dataset
    from pyarrow._fs import PyFileSystem


def sanitize_col_name(col_name: str) -> str:  # pragma: no cover
    """
    Sanitize a column name to remove
    any spaces, quotes, etc.
    Ref: https://stackoverflow.com/questions/3303312/how-do-i-convert-a-string-to-a-valid-variable-name-in-python.
    Essentially turns a string to a valid
    Python variable name, which is sufficient for our purposes.
    Note that these are not guaranteed to be 1 to 1, i.e.
    two inputs could produce the same output.

    Args:
        col_name (str): String to sanitize.

    Returns:
        str: Sanitized string
    """
    return re.sub(r"\W|^(?=\d)", "_", col_name)


@dataclass
class IcebergPiece:
    """
    A simple dataclass representing a parquet
    file to read during an Iceberg table read.
    These are used in 'IcebergParquetDataset'
    to store information about the files to read.
    """

    # Path to the file. This may or may not
    # include the protocol prefix.
    path: str
    # Index of the schema group that this
    # file belongs to. This corresponds to
    # the schema groups in
    # IcebergParquetDataset.schema_groups.
    # -1 if we're setting the schema_group_identifier instead.
    schema_group_idx: int
    # Schema group identifier. This is used when
    # the schema groups haven't been created yet.
    schema_group_identifier: SchemaGroupIdentifier | None
    # Number of rows to read from this file
    # In case of row-level filtering, this is the count
    # after applying the filters. In case of piece-level filtering,
    # this is the total number of rows in the piece.
    _bodo_num_rows: int


class IcebergSchemaGroup:
    """
    Class to store the details about one "Schema Group"
    during Iceberg read. A schema group is a group of files
    where their schemas are "similar" in terms of the
    Iceberg fields they contain and the names of those
    fields. Therefore, a schema group is identified
    by two ordered tuples:
    1. The Iceberg Field IDs
    2. The corresponding fields' names.
    The idea is that we can read these files as one
    Arrow dataset. This is useful since Arrow can do
    async read-ahead on the files in a dataset, which
    improves performance.
    The shared Arrow expression filter applied to these
    files is generated based on the mapping of column names
    in the "final_schema" (the schema of the table we want
    to read the dataset *as*) to the names in the "read_schema"
    (i.e. the intermediate schema that we will give to Arrow
    during read so that it can perform the filters correctly
    and fill in nulls for columns where they don't exist).
    """

    def __init__(
        self,
        iceberg_field_ids: FieldIDs,
        parquet_field_names: FieldNames,
        final_schema: pa.Schema,
        expr_filter_f_str: str | None = None,
        filter_scalars: list[tuple[str, pt.Any]] | None = None,
    ):
        """
        Construct a new Schema Group.

        Args:
            iceberg_field_ids (tuple[int]): Ordered tuple of Iceberg field
                IDs of the top-level columns (i.e. field IDs of the nested
                fields is not included in these).
            parquet_field_names (tuple[str]): Ordered tuple of the field
                names for the fields corresponding to those in
                'iceberg_field_ids'.
            final_schema (pa.Schema): The 'final_schema' that will be used
                for generating the read-schema.
            expr_filter_f_str (Optional[str], optional): An f-string
                representation of the Arrow expression filter to use to generate
                the filter expression. See description of 'generate_expr_filter'
                for more details. Defaults to None.
            filter_scalars (Optional[list[tuple[str, Any]]], optional): List of
                tuples with the variable names and values of the scalars
                present in the expr_filter_f_str. See description of
                'generate_expr_filter'for more details. Defaults to None.
        """
        assert len(iceberg_field_ids) == len(parquet_field_names)
        self.iceberg_field_ids = iceberg_field_ids
        self.parquet_field_names = parquet_field_names
        self.final_schema: pa.Schema = final_schema
        self.read_schema: pa.Schema = self.gen_read_schema(
            self.iceberg_field_ids, self.parquet_field_names, self.final_schema
        )
        self.expr_filter: pc.Expression | None = None
        if (expr_filter_f_str is not None) and (len(expr_filter_f_str) > 0):
            filter_scalars = [] if filter_scalars is None else filter_scalars
            col_rename_map: dict[str, str] = {
                self.final_schema.field(i).name: self.read_schema.field(i).name
                for i in range(len(self.final_schema.names))
            }
            self.expr_filter = generate_expr_filter(
                expr_filter_f_str, filter_scalars, col_rename_map
            )

    @property
    def group_identifier(self) -> SchemaGroupIdentifier:
        """
        The tuple that uniquely identifies a Schema Group.

        Returns:
            SchemaGroupIdentifier
        """
        return (self.iceberg_field_ids, self.parquet_field_names)

    @staticmethod
    def gen_read_field(
        iceberg_field_ids: FieldID,
        parquet_field_names: FieldName,
        final_field: pa.Field,
        field_name_for_err_msg: str,
    ) -> pa.Field:
        """
        Recursive helper for gen_read_schema to generate
        the Iceberg Schema Group's read field.

        Args:
            iceberg_field_ids (int | tuple): Iceberg Field ID
                of this field in the parquet file. In the
                semi-structured type case, this will be a tuple
                where the first element is the Field ID of the
                semi-structured type itself and the rest will
                be Field IDs of it sub-fields (each of these may
                also be tuples since they might be semi-structured
                themselves).
            parquet_field_names (str | tuple): Corresponding
                fields' names.
            final_field (pa.Field): The target field.
            field_name_for_err_msg (str): Since this function is
                called recursively, we use this to build up
                a more meaningful name for any error messages that
                we raise.

        Returns:
            pa.Field: Field to use when reading the files in this
                schema group.
        """

        assert final_field.metadata is not None, (
            f"Field {field_name_for_err_msg} does not have metadata! This is most likely a bug in Bodo."
        )
        assert b_ICEBERG_FIELD_ID_MD_KEY in final_field.metadata, (
            f"Field {field_name_for_err_msg} does not have the Iceberg Field ID in its metadata. "
            f"Metadata:\n{final_field.metadata}\nThis is most likely a bug in Bodo."
        )
        iceberg_field_id = int(final_field.metadata[b_ICEBERG_FIELD_ID_MD_KEY])

        if isinstance(iceberg_field_ids, int):
            assert isinstance(parquet_field_names, str)
            if iceberg_field_id != iceberg_field_ids:
                raise RuntimeError(
                    f"Field {field_name_for_err_msg} does not have the expected Iceberg Field ID! "
                    f"Expected {iceberg_field_id} but got {iceberg_field_ids} instead."
                )
            if (
                pa.types.is_map(final_field.type)
                or pa.types.is_list(final_field.type)
                or pa.types.is_large_list(final_field.type)
                or pa.types.is_fixed_size_list(final_field.type)
                or pa.types.is_struct(final_field.type)
            ):
                raise RuntimeError(
                    f"Expected field type for Iceberg Field ID {iceberg_field_id} ({field_name_for_err_msg}) "
                    f"to be a nested type ({final_field.type}), but it was a primitive type instead!"
                )
            # Note that "final_field" is assumed to be "compatible",
            # i.e. the same or less strict (in terms of type and
            # nullability) than whatever is in the files. The actual
            # validation is performed at read-time at a
            # per-file basis.
            # During the read, we will use the "old" name for this
            # file.
            return final_field.with_name(parquet_field_names)
        else:
            assert isinstance(iceberg_field_ids, tuple)
            assert isinstance(parquet_field_names, tuple)
            assert len(iceberg_field_ids) == len(parquet_field_names)
            assert isinstance(iceberg_field_ids[0], int)
            assert isinstance(parquet_field_names[0], str)
            if iceberg_field_id != iceberg_field_ids[0]:
                raise RuntimeError(
                    f"Field {field_name_for_err_msg} does not have the expected Iceberg Field ID! "
                    f"Expected {iceberg_field_id} but got {iceberg_field_ids[0]} instead."
                )
            if not (
                pa.types.is_map(final_field.type)
                or pa.types.is_list(final_field.type)
                or pa.types.is_large_list(final_field.type)
                or pa.types.is_fixed_size_list(final_field.type)
                or pa.types.is_struct(final_field.type)
            ):
                raise RuntimeError(
                    f"Expected field type for Iceberg Field ID {iceberg_field_id} ({field_name_for_err_msg}) "
                    f"to be a primitive type ({final_field.type}), but it was a nested type instead!"
                )

            if pa.types.is_struct(final_field.type):
                # Struct is the tricky case where we must handle
                # evolution ourselves at read time. Unlike
                # top-level fields, Arrow doesn't like it if
                # we add extra sub-fields to the struct (to fill
                # them with nulls). It also doesn't like it if we
                # don't provide the fields in the same order as they
                # will appear in the parquet files. However, it
                # does allow "skipping" sub-fields as long as the order
                # of the rest of the fields is consistent with that in the
                # parquet file. It can also still
                # perform nullability/type promotion (including multiple
                # levels down in its sub-fields). Therefore, we will
                # only include the sub-fields that exist in the parquet
                # file and will maintain the order.
                # If a sub-field from the final_field doesn't exist in the parquet file,
                # we won't add it to the read_schema at this point. We
                # will add it later (see 'EvolveRecordBatch' in 'iceberg_parquet_reader.cpp').
                # We will keep the fields in the same original order and with
                # the same names. We will do the re-ordering and renaming later
                # as part of 'EvolveRecordBatch'.
                # We will however skip the fields that we no longer need.
                # We will also perform the type/nullability promotion
                # as per the final_field.

                final_sub_fields_iceberg_field_id_to_idx: dict[int, int] = {
                    int(
                        final_field.type.field(i).metadata[b_ICEBERG_FIELD_ID_MD_KEY]
                    ): i
                    for i in range(final_field.type.num_fields)
                }

                read_fields: list[pa.Field] = []
                iceberg_field_ids_in_schema_group_sub_fields: set[int] = set()
                # Sub-fields start at index 1.
                for i in range(1, len(iceberg_field_ids)):
                    sub_field_iceberg_field_id: int = (
                        iceberg_field_ids[i]
                        if isinstance(iceberg_field_ids[i], int)
                        else iceberg_field_ids[i][0]
                    )
                    iceberg_field_ids_in_schema_group_sub_fields.add(
                        sub_field_iceberg_field_id
                    )
                    if (
                        sub_field_iceberg_field_id
                        in final_sub_fields_iceberg_field_id_to_idx
                    ):
                        final_sub_field_: pa.Field = final_field.type.field(
                            final_sub_fields_iceberg_field_id_to_idx[
                                sub_field_iceberg_field_id
                            ]
                        )
                        read_schema_sub_field = IcebergSchemaGroup.gen_read_field(
                            iceberg_field_ids[i],
                            parquet_field_names[i],
                            final_sub_field_,
                            field_name_for_err_msg=f"{field_name_for_err_msg}.{final_sub_field_.name}",
                        )
                        read_fields.append(read_schema_sub_field)

                # Verify that all the required sub fields in the final field exist
                # in the schema group field.
                for i in range(final_field.type.num_fields):
                    final_sub_field: pa.Field = final_field.type.field(i)
                    assert final_sub_field.metadata is not None
                    assert b_ICEBERG_FIELD_ID_MD_KEY in final_sub_field.metadata
                    final_sub_field_iceberg_field_id = int(
                        final_sub_field.metadata[b_ICEBERG_FIELD_ID_MD_KEY]
                    )
                    if (not final_sub_field.nullable) and (
                        final_sub_field_iceberg_field_id
                        not in iceberg_field_ids_in_schema_group_sub_fields
                    ):
                        raise RuntimeError(
                            f"Non-nullable field '{field_name_for_err_msg}.{final_sub_field.name}' "
                            f"(Iceberg Field ID: {final_sub_field_iceberg_field_id}) not found in "
                            "the schema group!"
                        )

                return final_field.with_type(pa.struct(read_fields)).with_name(
                    parquet_field_names[0]
                )
            elif pa.types.is_large_list(final_field.type):
                assert len(iceberg_field_ids) == len(parquet_field_names) == 2
                read_value_field = IcebergSchemaGroup.gen_read_field(
                    iceberg_field_ids[1],
                    parquet_field_names[1],
                    final_field.type.value_field,
                    field_name_for_err_msg=f"{field_name_for_err_msg}.element",
                )
                return final_field.with_type(pa.large_list(read_value_field)).with_name(
                    parquet_field_names[0]
                )
            else:
                assert pa.types.is_map(final_field.type)
                assert len(iceberg_field_ids) == len(parquet_field_names) == 3
                read_key_field = IcebergSchemaGroup.gen_read_field(
                    iceberg_field_ids[1],
                    parquet_field_names[1],
                    final_field.type.key_field,
                    field_name_for_err_msg=f"{field_name_for_err_msg}.key",
                )
                read_item_field = IcebergSchemaGroup.gen_read_field(
                    iceberg_field_ids[2],
                    parquet_field_names[2],
                    final_field.type.item_field,
                    field_name_for_err_msg=f"{field_name_for_err_msg}.value",
                )
                return final_field.with_type(
                    pa.map_(read_key_field, read_item_field)
                ).with_name(parquet_field_names[0])

    @staticmethod
    def gen_read_schema(
        iceberg_field_ids: FieldIDs,
        parquet_field_names: FieldNames,
        final_schema: pa.Schema,
    ) -> pa.Schema:
        """
        Generate the "read_schema", i.e. the schema given
        to Arrow Dataset Scanners when reading the files
        belonging to this schema group, from the final/target
        schema.
        The "read_schema" will have the same number of
        fields as the "final_schema" and the fields
        corresponding to the "final_schema" will be
        in the same order as the "final_schema".
        The "final_schema" must have Iceberg Field IDs
        in the metadata of the fields.
        Nested fields are handled by calling 'gen_read_field'
        on them recursively.

        Args:
            iceberg_field_ids (tuple[int | tuple]): Iceberg field IDs
                of the fields in the schema of the files in
                the schema-group.
            parquet_field_names (tuple[str | tuple]): The corresponding
                field names.
            final_schema (pa.Schema): The target schema.

        Returns:
            pa.Schema: 'read_schema' for the schema group.
        """

        # Create a map from Iceberg Field Id to the column index for the
        # top-level fields.
        schema_group_field_id_to_schema_group_col_idx: dict[int, int] = {}
        for i in range(len(iceberg_field_ids)):
            if isinstance(iceberg_field_ids[i], int):
                assert isinstance(parquet_field_names[i], str)
                schema_group_field_id_to_schema_group_col_idx[iceberg_field_ids[i]] = i
            else:
                assert isinstance(iceberg_field_ids[i], tuple)
                assert isinstance(parquet_field_names[i], tuple)
                assert isinstance(iceberg_field_ids[i][0], int)
                assert isinstance(parquet_field_names[i][0], str)
                schema_group_field_id_to_schema_group_col_idx[
                    iceberg_field_ids[i][0]
                ] = i

        read_schema_fields: list[pa.Field] = []
        for i, field in enumerate(final_schema):
            assert field.metadata is not None, (
                f"Target schema field doesn't have metadata! This is most likely a bug in Bodo. Field:\n{field}."
            )
            assert b_ICEBERG_FIELD_ID_MD_KEY in field.metadata, (
                f"Target schema field metadata doesn't have the required Iceberg Field ID. "
                f"This is most likely a bug in Bodo.\nField: {field}\nField metadata: {field.metadata}."
            )
            iceberg_field_id = int(field.metadata[b_ICEBERG_FIELD_ID_MD_KEY])
            if iceberg_field_id in schema_group_field_id_to_schema_group_col_idx:
                # If this field exists in the file:
                schema_group_field_idx = schema_group_field_id_to_schema_group_col_idx[
                    iceberg_field_id
                ]
                read_schema_field = IcebergSchemaGroup.gen_read_field(
                    iceberg_field_ids[schema_group_field_idx],
                    parquet_field_names[schema_group_field_idx],
                    field,
                    field_name_for_err_msg=field.name,
                )
                read_schema_fields.append(read_schema_field)
            else:
                # Field is not in the file, i.e. it was added to the table
                # after these files were written or the column in optional
                # and the original writer chose not to write them.
                # To avoid name conflicts, we will use a unique name.
                # The column will automatically be filled with nulls at read
                # time.
                if not field.nullable:
                    raise RuntimeError(
                        f"Non-nullable field '{field.name}' (Field ID: "
                        f"{iceberg_field_id}) not found in the schema group!"
                    )
                sanitized_field_name = sanitize_col_name(field.name)
                _uniq_name = f"_BODO_TEMP_{iceberg_field_id}_{sanitized_field_name}"
                assert _uniq_name not in parquet_field_names, (
                    f"Generated unique name for Iceberg field already exists in the file! "
                    f"This is most likely a bug in Bodo.\n{_uniq_name=}\n{parquet_field_names=}"
                )
                read_schema_fields.append(field.with_name(_uniq_name))

        return pa.schema(read_schema_fields)


@dataclass
class IcebergPqDatasetMetrics:
    """
    Metrics for the get_iceberg_pq_dataset step.
    All timers are in microseconds.
    """

    file_list_time: int = 0
    file_to_schema_time_us: int = 0
    get_fs_time: int = 0
    n_files_analyzed: int = 0
    file_frags_creation_time: int = 0
    get_sg_id_time: int = 0
    sort_by_sg_id_time: int = 0
    nunique_sgs_seen: int = 0
    exact_row_counts_time: int = 0
    get_row_counts_nrgs: int = 0
    get_row_counts_nrows: int = 0
    get_row_counts_total_bytes: int = 0
    pieces_allgather_time: int = 0
    sort_all_pieces_time: int = 0
    assemble_ds_time: int = 0


@dataclass
class IcebergParquetDataset:
    """
    Store dataset info in the way expected by Arrow reader in C++.
    """

    # Whether this is a row-level or piece-level read.
    # In case of a row-level read, we get the exact row counts
    # for each piece after applying filters. In the piece-level read
    # case, we only prune out pieces based on metadata and report
    # the row count of the entire piece.
    row_level: bool
    # This is the PyArrow schema object for the final/target schema to read
    # the table as. This is obtained from Iceberg at compile time, i.e. the
    # expected final schema. It must have Iceberg Field IDs in the metadata
    # of its fields.
    pa_table_schema: pa.Schema
    # List of files exactly as given by Iceberg. This is used for operations like delete/merge.
    # There files are likely the relative paths to the Iceberg table for local files.
    # For example if the absolute path was /Users/bodo/iceberg_db/my_table/part01.pq
    # and the iceberg directory is iceberg_db, then the path in the list would be
    # iceberg_db/my_table/part01.pq.
    file_list: list[str]
    # Snapshot id. This is used for operations like delete/merge.
    snapshot_id: int
    # Filesystem can be None when there are no files to read.
    filesystem: pa_fs.FileSystem | None
    # Parquet files to read ordered by the schema group
    # they belong to. We order them this way so that when
    # we split this list between ranks for the actual read,
    # each rank will (ideally) only need to handle a subset
    # of the schema groups (and hence minimize the number of
    # Arrow scanners/record-batch-readers that it needs
    # to create).
    pieces: list[IcebergPiece]
    # Ordered list of schema groups. These are all the
    # different schemas we will need to handle during
    # the actual read for handling schema evolution.
    schema_groups: list[IcebergSchemaGroup]
    # Total number of rows that we will read (globally).
    _bodo_total_rows: int
    # Metrics
    metrics: IcebergPqDatasetMetrics


def warn_if_non_ideal_io_parallelism(
    g_total_rgs: int, g_total_size_bytes: int, protocol: str
) -> None:
    """
    Helper function for raising warnings on rank-0 when
    the file properties are not ideal for effective
    parallelism.

    Args:
        g_total_rgs (int): Total number of row groups (global).
        g_total_size_bytes (int): Total size of all row groups
            to read from all files (global).
        protocol (str): Filesystem protocol. This is used
            to determine if we're reading from a remote
            filesystem.
    """
    if bodo.get_rank() == 0 and g_total_rgs < bodo.get_size() and g_total_rgs != 0:
        warnings.warn(
            BodoWarning(
                f"Total number of row groups in Iceberg dataset ({g_total_rgs}) is too small for effective IO parallelization."
                f"For best performance the number of row groups should be greater than the number of workers ({bodo.get_size()}). "
                "For more details, refer to https://docs.bodo.ai/latest/file_io/#parquet-section."
            )
        )
    # Print a warning if average row group size < 1 MB and reading from remote filesystem
    if g_total_rgs == 0:
        avg_row_group_size_bytes = 0
    else:
        avg_row_group_size_bytes = g_total_size_bytes // g_total_rgs
    if (
        bodo.get_rank() == 0
        and g_total_size_bytes >= (20 * 1024 * 1024)
        and avg_row_group_size_bytes < (1024 * 1024)
        and protocol in REMOTE_FILESYSTEMS
    ):
        warnings.warn(
            BodoWarning(
                f"Parquet (Iceberg) average row group size is small ({avg_row_group_size_bytes} bytes) "
                "and can have negative impact on performance when reading from remote sources."
            )
        )


def get_schema_group_identifier_from_pa_field(
    field: pa.Field,
    field_name_for_err_msg: str,
) -> tuple[FieldID, FieldName]:
    """
    Recursive helper for 'get_schema_group_identifier_from_pa_schema'
    to get the schema group identifier for a specific
    field (or sub-field of a nested field). These will
    then be stitched back together to form the
    full schema group identifier.

    Args:
        field (pa.Field): The field to generate the group
            identifier based off of. This could be
            a nested field.
        field_name_for_err_msg (str): Since this function
            is called recursively, we use this field to
            have a more meaningful field name that can be
            used in the error messages.

    Returns:
        SchemaGroupIdentifier: Schema group identifier
            for this field.
    """
    field_type = field.type

    if (field.metadata is None) or (b_ICEBERG_FIELD_ID_MD_KEY not in field.metadata):
        raise RuntimeError(
            f"Field {field_name_for_err_msg} does not have an Iceberg Field ID!"
        )

    iceberg_field_id: int = int(field.metadata[b_ICEBERG_FIELD_ID_MD_KEY])

    if pa.types.is_struct(field_type):
        sub_field_schema_group_identifiers = [
            get_schema_group_identifier_from_pa_field(
                field_type.field(i),
                f"{field_name_for_err_msg}.{field_type.field(i).name}",
            )
            for i in range(field_type.num_fields)
        ]
        field_ids = [iceberg_field_id] + [
            x[0] for x in sub_field_schema_group_identifiers
        ]
        field_names = [field.name] + [x[1] for x in sub_field_schema_group_identifiers]
        return tuple(field_ids), tuple(field_names)

    elif pa.types.is_map(field_type):
        key_field_schema_group_identifier = get_schema_group_identifier_from_pa_field(
            field_type.key_field, f"{field_name_for_err_msg}.key"
        )
        item_field_schema_group_identifier = get_schema_group_identifier_from_pa_field(
            field_type.item_field, f"{field_name_for_err_msg}.value"
        )
        return (
            iceberg_field_id,
            key_field_schema_group_identifier[0],
            item_field_schema_group_identifier[0],
        ), (
            field.name,
            key_field_schema_group_identifier[1],
            item_field_schema_group_identifier[1],
        )
    elif (
        pa.types.is_list(field_type)
        or pa.types.is_large_list(field_type)
        or pa.types.is_fixed_size_list(field_type)
    ):
        value_field_schema_group_identifier = get_schema_group_identifier_from_pa_field(
            field_type.value_field, f"{field_name_for_err_msg}.element"
        )
        return (iceberg_field_id, value_field_schema_group_identifier[0]), (
            field.name,
            value_field_schema_group_identifier[1],
        )
    else:
        return (iceberg_field_id, field.name)


def get_schema_group_identifier_from_pa_schema(
    schema: pa.Schema,
) -> SchemaGroupIdentifier:
    """
    Generate the schema group identifier from
    the schema of a parquet file. The schema group
    identifier is a tuple of tuples. The first
    is a tuple of Iceberg Field IDs and the second
    is a tuple of the corresponding field names
    in the Parquet file. Nested fields are represented
    by nested tuples within the top-level tuples.

    Args:
        schema (pa.Schema): Schema to generate the
            schema group identifier based on.

    Returns:
        SchemaGroupIdentifier: The schema group identifier.
    """
    field_identifiers = [
        get_schema_group_identifier_from_pa_field(f, f.name) for f in schema
    ]
    iceberg_field_ids = tuple(x[0] for x in field_identifiers)
    pq_field_names = tuple(x[1] for x in field_identifiers)
    return iceberg_field_ids, pq_field_names


def generate_expr_filter(
    expr_filter_f_str: str,
    filter_scalars: list[tuple[str, pt.Any]],
    col_rename_map: dict[str, str],
) -> pc.Expression:
    """
    Helper function to dynamically generate the Arrow expressions
    for filtering at runtime.
    The 'expr_filter_f_str' is generated by
    'bodo.ir.connector.generate_arrow_filters' by setting the
    'output_expr_filters_as_f_string' parameter.
    'filter_scalars' is generated using 'get_filter_scalars_pyobject'.
    See '_gen_sql_reader_py' and '_gen_iceberg_reader_chunked_py'
    in 'iceberg_ext.py' for more details on how these
    are generated.
    Note that we don't cache this computation, so it's best
    to call it once per IcebergSchemaGroup.


    Args:
        expr_filter_f_str (str): An f-string version of the
            final expression. We will populate the templated
            variables in this f-string using the col_rename_map.
        filter_scalars (list[tuple[str, Any]]): List of tuples
            of the form ('fXX', Any). The first element is the
            name of the variable in the expr_filter_f_str
            that will assume this value and the second element
            is the actual value itself. This value can be
            any Python object (e.g. string, int, list[Any], etc.)
        col_rename_map (dict[str, str]): Column rename map
            used to populate the templated variables in
            expr_filter_f_str. This is the mapping of the
            column names from the 'final_schema' to the
            'read_schema' of an IcebergSchemaGroup.

    Returns:
        pc.Expression: Generated expression object that can
            be used to filter the table during read.
    """

    # Fill in the templated column names.
    expr_filter_str = expr_filter_f_str.format(**col_rename_map)
    # Produce the parameters for the function.
    # e.g. 'f0, f1, f2'
    input_vars_str = ",".join([x[0] for x in filter_scalars])
    glbs = globals()
    glbs["ds"] = ds
    glbs["pc"] = pc
    glbs["pa"] = pa
    loc_vars = {}
    # By passing in the scalars as arguments, they will
    # get mapped correctly in the expr_filter_str.
    func_text = f"def impl({input_vars_str}):\n  return {expr_filter_str}"
    input_vars = [x[1] for x in filter_scalars]
    exec(func_text, glbs, loc_vars)
    expr_filter = loc_vars["impl"](*input_vars)
    return expr_filter


def distribute_pieces(pieces: list[IcebergPiece]) -> list[IcebergPiece]:
    """
    Distribute Iceberg File pieces between all ranks so that all ranks
    have to read roughly the same number of rows.
    To do this, we use a greedy algorithm described here:
    https://www.cs.cmu.edu/~15451-f23/lectures/lecture19-approx.pdf.
    The algorithm is deterministic, i.e. should yield the same result
    on all ranks. Therefore, no synchronization is performed.

    Args:
        pieces (list[IcebergPiece]): List of file pieces to
            distribute between all ranks. This must be the global
            list of pieces and must be ordered the same on all ranks.

    Returns:
        list[IcebergPiece]: List of pieces assigned that this
            rank should read. This will be ordered by the
            schema_group_idx so that all files in the same SchemaGroup
            are next to each other.
    """

    # Use a simple greedy algorithm to assign pieces to respective ranks.
    # Sort the pieces from the largest to smallest.
    # Iterate through the pieces and assign it to the rank with the
    # fewest rows.
    # XXX There's a concern that if the piece is at a row-group level,
    # row groups from the same file may get assigned to different
    # ranks, which can lead to wasted IO. To alleviate this, we could
    # first do this algorithm on the file-level pieces. If there's
    # significant skew, we can break up the files-level pieces into
    # row-group-level pieces and repeat the algorithm.

    import heapq

    comm = MPI.COMM_WORLD
    myrank: int = comm.Get_rank()
    n_pes: int = comm.Get_size()

    # Sort the pieces
    sorted_pieces: list[IcebergPiece] = sorted(
        pieces, key=lambda p: (p._bodo_num_rows, p.path)
    )

    pieces_myrank: list[IcebergPiece] = []

    # To assign the pieces, we iterate through the pieces and assign the piece
    # to the rank with the least rows already assigned to it. To keep track
    # of the ranks and how many rows they're assigned, we use a heap
    # where each element is of the form (num_rows, rank). This allows us
    # to get the min rank in logarithmic time in each iteration.
    rank_heap: list[tuple[int, int]] = [(0, i) for i in range(n_pes)]
    heapq.heapify(rank_heap)

    for piece in sorted_pieces:
        piece_nrows = piece._bodo_num_rows
        least_rows, rank = heapq.heappop(rank_heap)
        if rank == myrank:
            pieces_myrank.append(piece)
        heapq.heappush(rank_heap, (least_rows + piece_nrows, rank))

    # Sort by schema_group_idx before returning
    pieces_myrank = sorted(pieces_myrank, key=lambda p: (p.schema_group_idx, p.path))

    return pieces_myrank


def get_dataset_for_schema_group(
    schema_group: IcebergSchemaGroup,
    files: list[str],
    files_rows_to_read: list[int],
    final_schema: pa.Schema,
    str_as_dict_cols: list[str],
    filesystem: pa_fs.FileSystem,
    start_offset: int,
    len_all_fpaths: int,
) -> tuple[Dataset, pa.Schema, int]:
    """
    Create an Arrow Dataset for files belonging
    to the same Iceberg Schema Group.
    Args:
        schema_group (IcebergSchemaGroup): Schema Group for
            the files.
        files (list[str]): List of files.
        files_rows_to_read (list[int]): Number of rows to
            read from each of the files.
        final_schema (pa.Schema): Target schema for the final
            Iceberg table. This is used for certain column
            renaming during the read.
        str_as_dict_cols (list[str]): List of column names
            that must be read with dictionary encoding.
        filesystem (pa.fs.FileSystem): Filesyste to use for reading the files.
        start_offset (int): The starting row offset to read from
            in the first file.
        len_all_fpaths (int): Total number of files across all schema
            groups that this rank will read. This is used in some
            heuristics to decide whether or not to split the
            file-level dataset into row-group-level dataset.
    Returns:
        tuple[Dataset, pa.Schema, int]:
            - Arrow Dataset for the files in the
            schema group.
            - The schema that the Dataset will use
            while reading the file(s). This may be slightly
            different than the read_schema of the schema-group
            since some columns may be dict-encoded.
            - Updated start_offset.
    """
    read_schema: pa.Schema = schema_group.read_schema

    # Create a ParquetFileFormat where we specify the columns to
    # dict encode.
    col_rename_map: dict[str, str] = {
        final_schema.field(i).name: read_schema.field(i).name
        for i in range(len(final_schema.names))
    }
    schema_group_str_as_dict_cols: list[str] = [
        col_rename_map[f] for f in str_as_dict_cols
    ]
    pq_format = ds.ParquetFileFormat(dictionary_columns=schema_group_str_as_dict_cols)

    # Set columns to be read as dictionary encoded in the read schema
    read_schema = schema_with_dict_cols(read_schema, schema_group_str_as_dict_cols)

    dataset = ds.dataset(
        files,
        filesystem=filesystem,
        schema=read_schema,
        format=pq_format,
    )

    # For the first schema group, prune out row groups if it could be beneficial:
    if (start_offset > 0) and filter_row_groups_from_start_of_dataset_heuristic(
        # We will consider number of files across all schema groups and not just
        # this one to determine whether or not to prune row groups.
        len_all_fpaths,
        start_offset,
        schema_group.expr_filter,
    ):
        # The starting offset the Parquet reader knows about is from the first
        # file, not the first row group, so we need to communicate this back to C++
        dataset, start_offset = filter_row_groups_from_start_of_dataset(
            dataset,
            start_offset,
            sum(files_rows_to_read),
            pq_format,
        )

    return dataset, read_schema, start_offset


def get_pyarrow_datasets(
    fpaths: list[str],
    file_nrows_to_read: list[int],
    file_schema_group_idxs: list[int],
    schema_groups: list[IcebergSchemaGroup],
    avg_num_pieces: float,
    is_parallel: bool,
    filesystem: pa_fs.FileSystem,
    str_as_dict_cols: list[str],
    start_offset: int,
    final_schema: pa.Schema,
) -> tuple[list[Dataset], list[pa.Schema], list[pc.Expression], int]:
    """
    Get the PyArrow Datasets for the given files.
    This will return one Dataset for every unique schema
    group that these filepaths will use.
    This will also return an updated offset for the file/piece.
    Args:
        fpaths (list[str]): List of files to read from. The files
            must be ordered by their corresponding schema group.
        file_nrows_to_read (list[int]): Total number of rows this
            process is going to read from each of these files.
            From the first file, it will read starting from 'start_offset'.
        file_schema_group_idxs (list[int]): Index of the schema group
            that each of these files belongs to.
        schema_groups (list[IcebergSchemaGroup]): List of all the schema
            groups.
        avg_num_pieces (float): Average number of pieces that every
            rank will read. If a rank is going to read many more
            files than average, we assign it more IO threads.
        is_parallel (bool): Whether this is being called in parallel
            across all ranks.
        filesystem (pa.fs.FileSystem): Filesystem to use for reading the files.
        str_as_dict_cols (list[str]): List of column names
            that must be read with dictionary encoding.
        start_offset (int): The starting row offset to read from
            in the first piece. This is only applicable when reading at
            a row-level. If reading at a piece-level, this should be
            set to 0.
        final_schema (pa.Schema): Target schema for the final
            Iceberg table. This is used for certain column
            renaming during the read.
    Returns:
        tuple[list["Dataset"], list[pa.Schema], list[pc.Expression], int]:
            - List of Arrow Datasets. There will be one
            per Schema Group that this rank will end up reading from.
            - List of the corresponding read_schema for each of
            the datasets (based on the schema group the dataset belongs to).
            - List of the corresponding filter to apply for each of
            the datasets (based on the schema group the dataset belongs to).
            - Update row offset into the first file/piece. Only applicable
            in the row-level read case.
    """
    cpu_count = os.cpu_count()
    if cpu_count is None or cpu_count == 0:
        cpu_count = 2
    default_io_threads = min(int(os.environ.get("BODO_MIN_IO_THREADS", 4)), cpu_count)
    max_io_threads = min(int(os.environ.get("BODO_MAX_IO_THREADS", 16)), cpu_count)
    # Assign more threads to ranks that have to read many more files
    # than the others.
    # TODO Unset this after the read??
    if (
        is_parallel
        and len(fpaths) > max_io_threads
        and len(fpaths) / avg_num_pieces >= 2.0
    ):
        pa.set_io_thread_count(max_io_threads)
    else:
        pa.set_io_thread_count(default_io_threads)

    if len(fpaths) == 0:
        return [], [], [], start_offset

    datasets: list[Dataset] = []
    dataset_read_schemas: list[pa.Schema] = []
    dataset_expr_filters: list[pc.Expression] = []

    # Assuming the files are ordered by their corresponding
    # schema group index, we can iterate and group files that way.
    curr_file_idx = 0
    curr_schema_group_idx = file_schema_group_idxs[0]
    while curr_file_idx < len(fpaths):
        # Accumulate files for the group:
        curr_schema_group_files: list[str] = []
        curr_schema_group_files_rows_to_read: list[int] = []
        while (curr_file_idx < len(fpaths)) and (
            file_schema_group_idxs[curr_file_idx] == curr_schema_group_idx
        ):
            curr_schema_group_files.append(fpaths[curr_file_idx])
            curr_schema_group_files_rows_to_read.append(
                file_nrows_to_read[curr_file_idx]
            )
            curr_file_idx += 1

        # Get schema group for this set of files.
        schema_group: IcebergSchemaGroup = schema_groups[curr_schema_group_idx]
        # Get the row counts for these files.
        dataset, dataset_read_schema, new_start_offset = get_dataset_for_schema_group(
            schema_group,
            curr_schema_group_files,
            curr_schema_group_files_rows_to_read,
            final_schema,
            str_as_dict_cols,
            filesystem,
            # 'start_offset' is only applicable to the
            # first schema-group. It's 0 for the rest.
            start_offset if len(datasets) == 0 else 0,
            len(fpaths),
        )
        # Update the overall start_offset if this is the first
        # schema-group.
        start_offset = new_start_offset if len(datasets) == 0 else start_offset
        datasets.append(dataset)
        dataset_read_schemas.append(dataset_read_schema)
        dataset_expr_filters.append(schema_group.expr_filter)

        # Update the schema group index.
        if curr_file_idx < len(fpaths):
            curr_schema_group_idx = file_schema_group_idxs[curr_file_idx]

    return datasets, dataset_read_schemas, dataset_expr_filters, start_offset


def get_pieces_with_exact_row_counts(
    schema_group: IcebergSchemaGroup,
    schema_group_identifier: SchemaGroupIdentifier,
    pq_infos: list[IcebergParquetInfo],
    fs: PyFileSystem | pa_fs.FileSystem,
    final_schema: pa.Schema,
    str_as_dict_cols: list[str],
    metrics: IcebergPqDatasetMetrics,
) -> list[IcebergPiece]:
    """
    Helper function for 'get_row_counts_for_schema_group' to get pieces with
    the exact row counts for a list of files (after applying filters)
    which all belong to the same schema group.
    NOTE: The file fragments are expected to have their metadata already
    populated.

    Args:
        schema_group (IcebergSchemaGroup): SchemaGroup that the files
            belong to.
        schema_group_identifier (SchemaGroupIdentifier):
            Group identifier. This is a tuple of two ordered tuples.
            The first is an ordered tuple of Iceberg Field IDs and
            the second is an ordered tuple of the corresponding
            field names.
        pq_file_fragments (list[IcebergParquetInfo]): List of files
            to get the row counts for.
        fs (pa.fs.FileSystem): Filesystem to use for accessing the files and getting the row count
            and metadata information.
            NOTE: This is only used when there are dict-encoded
            columns and we need to re-create the fragments from a
            new ParquetFileFormat which sets the dict-encoded
            columns correctly.
        final_schema (pa.Schema): Target schema for the Iceberg table.
        str_as_dict_cols (list[str]): List of column names (in final schema)
            that will be read with dictionary encoding.
        metrics (IcebergPqDatasetMetrics): Metrics to update in place.

    Returns:
        list[IcebergPiece]: Pieces with exact row count information.
    """

    # For frag.scanner().count_rows(), we use the expected schema instead
    # of the file schema. This schema should be a less-restrictive
    # superset of the file schema, so it should be valid.
    read_schema: pa.Schema = schema_group.read_schema

    # When using frag.scanner().count_rows(),
    # we need to use the schema with pa.dictionary fields for the
    # dictionary encoded fields. This is important for correctness
    # since this is what we will do during the actual read
    # (see 'get_dataset_for_schema_group'). Without this,
    # the row count may be inaccurate (potentially due to bugs in Arrow).
    # See [BSE-2790] for more context.

    # Create a ParquetFileFormat where we specify the columns to
    # dict encode.
    col_rename_map: dict[str, str] = {
        final_schema.field(i).name: read_schema.field(i).name
        for i in range(len(final_schema.names))
    }
    schema_group_str_as_dict_cols: list[str] = [
        col_rename_map[f] for f in str_as_dict_cols
    ]
    pq_file_format = ds.ParquetFileFormat(
        dictionary_columns=schema_group_str_as_dict_cols
    )
    # Set columns to be read as dictionary encoded in the read schema
    read_schema = schema_with_dict_cols(read_schema, schema_group_str_as_dict_cols)

    # Create ParquetFileFragments for parallel row-count calculation
    start = time.monotonic()
    pq_file_fragments: list[ds.ParquetFileFragment] = []
    for pq_info in pq_infos:
        pq_file_fragments.append(
            pq_file_format.make_fragment(pq_info.sanitized_path, fs)
        )
    metrics.file_frags_creation_time += int((time.monotonic() - start) * 1_000_000)

    pieces: list[IcebergPiece] = []

    # Determine the row counts for each file fragment in parallel.
    # Presumably the work is partitioned more or less equally among ranks,
    # and we are mostly (or just) reading metadata, so we assign four IO
    # threads to every rank.
    # XXX Use a separate env var for this?
    nthreads = min(int(os.environ.get("BODO_MIN_IO_THREADS", 4)), 4)
    pa_orig_io_thread_count = pa.io_thread_count()
    pa.set_io_thread_count(nthreads)
    pa_orig_cpu_thread_count = pa.cpu_count()
    pa.set_cpu_count(nthreads)
    try:
        t0: float = time.monotonic()
        file_row_counts = arrow_cpp.fetch_parquet_frag_row_counts(
            pq_file_fragments, schema_group.expr_filter, read_schema
        )
        for frag, file_row_count in zip(pq_file_fragments, file_row_counts):
            pieces.append(
                IcebergPiece(frag.path, -1, schema_group_identifier, file_row_count)
            )
            metrics.get_row_counts_nrows += file_row_count
            metrics.get_row_counts_nrgs += frag.num_row_groups
            metrics.get_row_counts_total_bytes += sum(
                rg.total_byte_size for rg in frag.row_groups
            )
        metrics.exact_row_counts_time += int((time.monotonic() - t0) * 1_000_000)
    finally:
        # Restore pyarrow default IO thread count
        pa.set_io_thread_count(pa_orig_io_thread_count)
        pa.set_cpu_count(pa_orig_cpu_thread_count)

    return pieces


def get_row_counts_for_schema_group(
    schema_group_identifier: SchemaGroupIdentifier,
    pq_infos: list[IcebergParquetInfo],
    fs: PyFileSystem | pa_fs.FileSystem,
    final_schema: pa.Schema,
    str_as_dict_cols: list[str],
    metrics: IcebergPqDatasetMetrics,
    row_level: bool = False,
    expr_filter_f_str: str | None = None,
    filter_scalars: list[tuple[str, pt.Any]] | None = None,
) -> list[IcebergPiece]:
    """
    Get the row counts for files belonging to the same
    Schema Group. Note that this is the row count
    after applying the provided filters in the row_level=True
    case. In the row_level=False case, we only apply the filters
    at the row group metadata level and hence the row counts
    are simply the number of rows in the row groups that weren't
    entirely pruned out.
    Note that this also validates the schemas of the files
    to ensure that they are compatible with this schema
    group.
    NOTE: This function is completely local and doesn't
    do any synchronization. It may raise Exceptions.
    The caller is expected to handle the error-synchronization.

    NOTE: The file fragments are expected to have their metadata already
    populated.

    Args:
        schema_group_identifier (SchemaGroupIdentifier):
            Group identifier. This is a tuple of two ordered tuples.
            The first is an ordered tuple of Iceberg Field IDs and
            the second is an ordered tuple of the corresponding
            field names.
        pq_infos (list[IcebergFileInfo]): List of files
            to get the row counts for.
        fs (pa.fs.FileSystem): Filesystem to use for accessing the files and getting the row count
             and metadata information.
             NOTE: This is only used in the row_level=True case when
             there are dict-encoded columns and we need to re-create the
             fragments from a new ParquetFileFormat which sets the
             dict-encoded columns correctly.
        final_schema (pa.Schema): Target schema for the Iceberg table.
            This will be used to generate a "read_schema" for this
            schema group.
        str_as_dict_cols (list[str]): List of column names
            that will be read with dictionary encoding.
        metrics: (IcebergPqDatasetMetrics): Metrics to update in place.
        row_level (bool): Whether the row counts need to be done with
            row-level filtering or if row-group level filtering
            is sufficient.
        expr_filter_f_str (str, optional): f-string to use for
            generating the filter. See description
            of 'generate_expr_filter' for more details. Defaults to None.
        filter_scalars (list[tuple[str, Any]], optional): The scalars
            to use for generating the filter. See description
            of 'generate_expr_filter' for more details. Defaults to None.

    Returns:
        list[IcebergPiece]: List of 'IcebergPiece's.
            In the row_level=False case, this includes details about
            the selected row groups within the files.
    """

    # Create a temporary IcebergSchemaGroup.
    schema_group: IcebergSchemaGroup = IcebergSchemaGroup(
        iceberg_field_ids=schema_group_identifier[0],
        parquet_field_names=schema_group_identifier[1],
        final_schema=final_schema,
        expr_filter_f_str=expr_filter_f_str,
        filter_scalars=filter_scalars,
    )

    ## 1. Validate that the file schemas are all compatible.
    # This will incur an expensive metadata read, so its behind a debug flag
    if bodo.check_parquet_schema:
        pq_file_format = ds.ParquetFileFormat()
        for pq_info in pq_infos:
            frag = pq_file_format.make_fragment(pq_info.sanitized_path, fs)
            file_schema = frag.metadata.schema.to_arrow_schema()
            try:
                # We use the original read-schema from the schema group
                # here (i.e. without the dictionary types) since that's
                # what the file is supposed to contain.
                validate_file_schema_compatible_with_read_schema(
                    file_schema, schema_group.read_schema
                )
            except Exception as e:
                msg = f"Schema of file {pq_info.path} is not compatible.\n{str(e)}"
                # TODO: raise BodoError in case of compiler (not dataframe library)
                raise ValueError(msg)

    ## 2. Perform filtering to get row counts and construct the IcebergPieces.
    pieces: list[IcebergPiece] = []
    if row_level:
        ## 2.1 If we need to get exact row counts, we will use the dataset
        # scanner API and apply the filter. Arrow will try to calculate this by
        # by reading only the file's metadata, and if it needs to
        # access data it will read as little as possible (only the
        # required columns and only subset of row groups if possible).
        pieces = get_pieces_with_exact_row_counts(
            schema_group,
            schema_group_identifier,
            pq_infos,
            fs,
            final_schema,
            str_as_dict_cols,
            metrics,
        )
    else:
        ## 2.2 If we are only doing piece level filtering, we can reuse Iceberg-level
        # row counts for the estimates. This skips row-group filtering
        pieces: list[IcebergPiece] = []
        for pq_info in pq_infos:
            pieces.append(
                IcebergPiece(
                    pq_info.sanitized_path,
                    -1,
                    schema_group_identifier,
                    pq_info.row_count,
                )
            )
            metrics.get_row_counts_nrows += pq_info.row_count

    return pieces
