import pyiceberg.io
import pyiceberg.io.pyarrow
import pyiceberg.schema
from pyiceberg.schema import Accessor, Schema, SchemaVisitor, visit
from pyiceberg.types import (
    IcebergType,
    ListType,
    MapType,
    NestedField,
    PrimitiveType,
    StructType,
)

from .file_io import BodoPyArrowFileIO

Position = int


class _BuildPositionAccessors(SchemaVisitor[dict[Position, Accessor]]):
    """
    Monkey-patch from pyiceberg.schema._BuildPositionAccessors to fix a bug
    with nested fields and making sure it's available.
    """

    def schema(
        self, schema: Schema, struct_result: dict[Position, Accessor]
    ) -> dict[Position, Accessor]:
        return struct_result

    def struct(
        self, struct: StructType, field_results: list[dict[Position, Accessor]]
    ) -> dict[Position, Accessor]:
        result = {}

        for position, field in enumerate(struct.fields):
            if field_results[position]:
                for inner_field_id, acc in field_results[position].items():
                    result[inner_field_id] = Accessor(position, inner=acc)
            result[field.field_id] = Accessor(position)

        return result

    def field(
        self, field: NestedField, field_result: dict[Position, Accessor]
    ) -> dict[Position, Accessor]:
        return field_result

    def list(
        self, list_type: ListType, element_result: dict[Position, Accessor]
    ) -> dict[Position, Accessor]:
        return {}

    def map(
        self,
        map_type: MapType,
        key_result: dict[Position, Accessor],
        value_result: dict[Position, Accessor],
    ) -> dict[Position, Accessor]:
        return {}

    def primitive(self, primitive: PrimitiveType) -> dict[Position, Accessor]:
        return {}


def build_position_accessors(
    schema_or_type: Schema | IcebergType,
) -> dict[int, Accessor]:
    """
    Monkey-patched version of pyiceberg.schema.build_position_accessors to
    use the new _BuildPositionAccessors class that supports top-level nested fields.
    """
    return visit(schema_or_type, _BuildPositionAccessors())


# Monkey-patch the original function
pyiceberg.schema.build_position_accessors = build_position_accessors

# Moneky-patch inferred IO to support Bodo's
pyiceberg.io.pyarrow.PyArrowFileIO = BodoPyArrowFileIO
# Monkey-patch to use patched ArrowFileIO for abfs and abfss
pyiceberg.io.SCHEMA_TO_FILE_IO["abfs"].insert(0, pyiceberg.io.ARROW_FILE_IO)
pyiceberg.io.SCHEMA_TO_FILE_IO["abfss"].insert(0, pyiceberg.io.ARROW_FILE_IO)
