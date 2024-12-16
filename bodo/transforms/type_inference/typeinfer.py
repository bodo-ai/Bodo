import numba
from numba.core.typing.templates import (
    AbstractTemplate,
    Registry,
)
from numba.extending import models, register_model
from numba.types import Function


class BodoFunction(Function):
    """
    Type class for builtin functions implemented by Bodo.
    """

    def get_impl_key(self, sig):
        return self.templates[0].key


register_model(BodoFunction)(models.OpaqueModel)


class BodoRegistry(Registry):
    """Registry of functions typed by Bodo's native typer to plug into Numba"""

    def __init__(self):
        from bodo.libs.distributed_api import get_rank
        from bodo.libs.memory_budget import register_operator

        super().__init__()
        # Used by native type inferrer to box BodoFunction types.
        # Should match CallTyperRegistry.callTypers.
        self.function_map = {}

        # bodo.libs.memory_budget.register_operator
        class BodoTemplateRegisterOpterator(AbstractTemplate):
            key = register_operator
            path = b"bodo.libs.memory_budget.register_operator"

        func_type = BodoFunction(BodoTemplateRegisterOpterator)
        self.globals.append((register_operator, func_type))
        self.function_map["bodo.libs.memory_budget.register_operator"] = func_type

        # bodo.libs.distributed_api.get_rank
        class BodoTemplateGetRank(AbstractTemplate):
            key = get_rank
            path = b"bodo.libs.distributed_api.get_rank"

        func_type = BodoFunction(BodoTemplateGetRank)
        self.globals.append((get_rank, func_type))
        self.function_map["bodo.libs.distributed_api.get_rank"] = func_type
        self.function_map["bodo.get_rank"] = func_type


bodo_registry = BodoRegistry()
numba.core.registry.cpu_target.typing_context.install_registry(bodo_registry)
