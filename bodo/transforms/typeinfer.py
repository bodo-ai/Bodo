import numba
from numba.core.typing.templates import (
    AbstractTemplate,
    Registry,
)
from numba.extending import models, register_model
from numba.types import ExternalFunction, Function


class BodoFunction(Function):
    """
    Type class for builtin functions implemented by Bodo.
    """

    def get_impl_key(self, sig):
        return self.templates[0].key


register_model(BodoFunction)(models.OpaqueModel)


class ExternalFunctionErrorChecked(ExternalFunction):
    """Same as Numba's ExternalFunction, but lowering checks for Python exceptions"""

    pass


register_model(ExternalFunctionErrorChecked)(models.OpaqueModel)


class BodoRegistry(Registry):
    """Registry of functions typed by Bodo's native typer to plug into Numba"""

    def __init__(self):
        from bodo.libs.memory_budget import register_operator

        super().__init__()

        class BodoTemplate(AbstractTemplate):
            key = register_operator
            path = b"bodo.libs.memory_budget.register_operator"

        self.globals.append((register_operator, BodoFunction(BodoTemplate)))


bodo_registry = BodoRegistry()
numba.core.registry.cpu_target.typing_context.install_registry(bodo_registry)
