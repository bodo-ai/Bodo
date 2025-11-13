"""JIT class for DatabaseCatalog"""

from abc import ABC, abstractmethod

from numba.core import types


class DatabaseCatalogType(types.Type, ABC):
    """
    JIT abstract class for storing information for connecting
    to a remote DataBase from Java. Since JIT doesn't have true
    abstract classes we use this type simply for checking
    supported types, but there should be no way to construct
    this type.

    As a result, boxing/unboxing, typeof_implm and lower_constant
    are all not implemented for this class. In addition, no model
    is registered for this class.
    """

    @abstractmethod
    def get_java_object(self):
        """Convert the Python catalog object into its Java
        representation. Each implementing class should have its
        own corresponding java class.
        """
        return NotImplemented
