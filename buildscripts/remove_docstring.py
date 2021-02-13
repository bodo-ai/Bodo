"""Remove docstrings from a python file and replace it
"""
import sys
import ast
import astunparse
from pathlib import Path


class RemoveDocstring(ast.NodeTransformer):
    """assumes the first string in function definition, class, or module is a
    docstring and removes it.
    """

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        return self._remove_docstring(node)

    def visit_ClassDef(self, node):
        self.generic_visit(node)
        return self._remove_docstring(node)

    def visit_Module(self, node):
        self.generic_visit(node)
        return self._remove_docstring(node)

    def _remove_docstring(self, node):
        if len(node.body) == 0:
            return node

        first_node = node.body[0]
        if isinstance(first_node, ast.Expr) and isinstance(first_node.value, ast.Str):
            node.body = node.body[1:]
            if len(node.body) == 0:
                # add a pass statement in case the function/class/module has
                # only a docstring
                node.body.append(ast.Pass())
        return node


if __name__ == "__main__":
    all_files = list(Path("bodo").glob("**/*.py"))

    for filename in all_files:
        with open(filename, "r") as f:
            code = f.read()
        node = ast.parse(code)
        node = RemoveDocstring().visit(node)
        new_code = astunparse.unparse(node)
        with open(filename, "w") as f:
            f.write(new_code)
