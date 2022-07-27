from py4j.protocol import Py4JJavaError


class IcebergError(Exception):
    """General Exception from Bodo Iceberg Connector"""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class IcebergJavaError(IcebergError):
    """
    Exception raised when Java Exception is Captured from Iceberg Connector Code

    Arguments:
        message (str): Python message to always include and pass to Bodo
        java_error (JException): Reference to Java exception including Java traceback
    """

    def __init__(self, message: str, java_error: str):
        super().__init__(message)
        self.message = message
        self.java_error = java_error

    @classmethod
    def from_java_error(cls, e: Py4JJavaError):
        # TODO: figure out how to get this to work with subclasses
        if (
            str(e.java_exception.getClass())
            == "class org.apache.iceberg.exceptions.NoSuchTableException"
        ):
            return cls("No such Iceberg table found", str(e.java_exception))
        elif (
            str(e.java_exception.getClass())
            == "class org.apache.iceberg.exceptions.RuntimeIOException"
        ):
            return cls("Unable to find Iceberg table", str(e.java_exception))
        else:
            return cls("Unknown Iceberg Error", str(e.java_exception))
