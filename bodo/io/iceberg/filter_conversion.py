"""Convert Iceberg expressions to PyArrow expressions and scalars for filtering.
This module imports pyicberg so it shouldn't be imported unless PyIceberg is installed."""

from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
from pyiceberg.expressions import BoundTerm
from pyiceberg.expressions.literals import Literal
from pyiceberg.expressions.visitors import BoundBooleanExpressionVisitor


# Adapted from https://github.com/apache/iceberg-python/blob/3070e7a2b8d681cd02f753b3a46e6ff1b27b76cf/pyiceberg/io/pyarrow.py#L702
# The changes are that we return a string expression and a list of name, scalar pairs instead of a PyArrow expression.
# This is so we can format the expression string with remapped column names from schema evolution.
class _ConvertToArrowExpressionStringAndScalar(
    BoundBooleanExpressionVisitor[tuple[str, list[tuple[str, Any]]]]
):
    """Visitor to convert a bound Iceberg expression to a PyArrow expression string and associated scalars.
    Creates expr_filter_f_str and filter_scalars for use in other functions like get_iceberg_pq_dataset."""

    def make_names_unique(
        self,
        left_child: tuple[str, list[tuple[str, Any]]],
        right_child: tuple[str, list[tuple[str, Any]]],
    ) -> tuple[tuple[str, list[tuple[str, Any]]], tuple[str, list[tuple[str, Any]]]]:
        """Make sure the names in the left and right child results are unique, we assume that all names are of form fn where n is a number and
        they are increasing with index."""
        left_names = {name for name, _ in left_child[1]}
        right_names = {name for name, _ in right_child[1]}

        # If there is no overlap, we can just return the children as is
        if left_names.isdisjoint(right_names):
            return left_child, right_child

        # This is a simple implementation that just sets the name to the lowest unused number
        max_left_name = left_child[1][-1][0] if left_child[1] else "f0"
        next_name = int(max_left_name[1:]) + 1
        right_rename_map = {
            name: f"f{next_name + i}" for i, (name, _) in enumerate(right_child[1])
        }

        # Rename the right child names in the scalar tuple
        right_child_scalars = [
            (right_rename_map[name], value) for name, value in right_child[1]
        ]
        # Rename the right child names in the expression string
        right_child_expr = right_child[0]
        for old_name, new_name in right_rename_map.items():
            right_child_expr = right_child_expr.replace(old_name, new_name)
        right_child = (right_child_expr, right_child_scalars)

        return left_child, right_child

    def visit_in(
        self, term: BoundTerm[Any], literals: set[Any]
    ) -> tuple[str, list[tuple[str, Any]]]:
        from pyiceberg.io.pyarrow import schema_to_pyarrow

        type = schema_to_pyarrow(term.ref().field.field_type)
        field_name = term.ref().field.name
        array_literals = pa.array(literals, type=type)
        return f"pc.field('{{{field_name}}}').isin(f0)", [("f0", array_literals)]

    def visit_not_in(
        self, term: BoundTerm[Any], literals: set[Any]
    ) -> tuple[str, list[tuple[str, Any]]]:
        from pyiceberg.io.pyarrow import schema_to_pyarrow

        type = schema_to_pyarrow(term.ref().field.field_type)
        field_name = term.ref().field.name
        array_literals = pa.array(literals, type=type)
        return f"~pc.field('{{{field_name}}}').isin(f0)", [("f0", array_literals)]

    def visit_is_nan(self, term: BoundTerm[Any]) -> tuple[str, list[tuple[str, Any]]]:
        ref = pc.field(term.ref().field.name)
        return f"pc.is_nan('{{{ref}}}')", []

    def visit_not_nan(self, term: BoundTerm[Any]) -> tuple[str, list[tuple[str, Any]]]:
        ref = pc.field(term.ref().field.name)
        return f"~pc.is_nan('{{{ref}}}')", []

    def visit_is_null(self, term: BoundTerm[Any]) -> tuple[str, list[tuple[str, Any]]]:
        return f"pc.field('{{{term.ref().field.name}}}').is_null(nan_is_null=False)", []

    def visit_not_null(self, term: BoundTerm[Any]) -> tuple[str, list[tuple[str, Any]]]:
        return f"pc.field('{{{term.ref().field.name}}}').is_valid()", []

    def visit_equal(
        self, term: BoundTerm[Any], literal: Literal[Any]
    ) -> tuple[str, list[tuple[str, Any]]]:
        from pyiceberg.io.pyarrow import _convert_scalar

        scalar = _convert_scalar(literal.value, term.ref().field.field_type)
        return f"pc.field('{{{term.ref().field.name}}}') == f0", [("f0", scalar)]

    def visit_not_equal(
        self, term: BoundTerm[Any], literal: Literal[Any]
    ) -> tuple[str, list[tuple[str, Any]]]:
        from pyiceberg.io.pyarrow import _convert_scalar

        scalar = _convert_scalar(literal.value, term.ref().field.field_type)
        return f"pc.field('{{{term.ref().field.name}}}') != f0", [("f0", scalar)]

    def visit_greater_than_or_equal(
        self, term: BoundTerm[Any], literal: Literal[Any]
    ) -> tuple[str, list[tuple[str, Any]]]:
        from pyiceberg.io.pyarrow import _convert_scalar

        scalar = _convert_scalar(literal.value, term.ref().field.field_type)
        return f"pc.field('{{{term.ref().field.name}}}') >= f0", [("f0", scalar)]

    def visit_greater_than(
        self, term: BoundTerm[Any], literal: Literal[Any]
    ) -> tuple[str, list[tuple[str, Any]]]:
        from pyiceberg.io.pyarrow import _convert_scalar

        scalar = _convert_scalar(literal.value, term.ref().field.field_type)
        return f"pc.field('{{{term.ref().field.name}}}') > f0", [("f0", scalar)]

    def visit_less_than(
        self, term: BoundTerm[Any], literal: Literal[Any]
    ) -> tuple[str, list[tuple[str, Any]]]:
        from pyiceberg.io.pyarrow import _convert_scalar

        scalar = _convert_scalar(literal.value, term.ref().field.field_type)
        return f"pc.field('{{{term.ref().field.name}}}') < f0", [("f0", scalar)]

    def visit_less_than_or_equal(
        self, term: BoundTerm[Any], literal: Literal[Any]
    ) -> tuple[str, list[tuple[str, Any]]]:
        from pyiceberg.io.pyarrow import _convert_scalar

        scalar = _convert_scalar(literal.value, term.ref().field.field_type)
        return f"pc.field('{{{term.ref().field.name}}}') <= f0", [("f0", scalar)]

    def visit_starts_with(
        self, term: BoundTerm[Any], literal: Literal[Any]
    ) -> tuple[str, list[tuple[str, Any]]]:
        return f"pc.starts_with(pc.field('{{{term.ref().field.name}}}'), f0)", [
            ("f0", literal.value)
        ]

    def visit_not_starts_with(
        self, term: BoundTerm[Any], literal: Literal[Any]
    ) -> tuple[str, list[tuple[str, Any]]]:
        return f"~pc.starts_with(pc.field('{{{term.ref().field.name}}}'), f0)", [
            ("f0", literal.value)
        ]

    def visit_true(self) -> tuple[str, list[tuple[str, Any]]]:
        return "pc.scalar(True)", []

    def visit_false(self) -> tuple[str, list[tuple[str, Any]]]:
        return "pc.scalar(False)", []

    def visit_not(
        self, child_result: tuple[str, list[tuple[str, Any]]]
    ) -> tuple[str, list[tuple[str, Any]]]:
        return f"~({child_result[0]})", child_result[1]

    def visit_and(
        self,
        left_result: tuple[str, list[tuple[str, Any]]],
        right_result: tuple[str, list[tuple[str, Any]]],
    ) -> tuple[str, list[tuple[str, Any]]]:
        left_result, right_result = self.make_names_unique(left_result, right_result)
        return f"({left_result[0]}) & ({right_result[0]})", left_result[
            1
        ] + right_result[1]

    def visit_or(
        self,
        left_result: tuple[str, list[tuple[str, Any]]],
        right_result: tuple[str, list[tuple[str, Any]]],
    ) -> tuple[str, list[tuple[str, Any]]]:
        left_result, right_result = self.make_names_unique(left_result, right_result)
        return f"({left_result[0]}) | ({right_result[0]})", left_result[
            1
        ] + right_result[1]
