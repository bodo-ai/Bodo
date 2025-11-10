"""Python and JIT class info for describing a connection to a remote Database.
Each implementing class should contain the relevant information to
construct a Java object that can be used to load schema + table
information from Java code.
"""

from abc import ABC, abstractmethod


class DatabaseCatalog(ABC):
    """
    Python Abstract Class for storing information for connecting
    to a remote DataBase from Java.
    """

    @abstractmethod
    def get_java_object(self):
        """Convert the Python catalog object into its Java
        representation. Each implementing class should have its
        own corresponding java class.
        """
        return NotImplemented
