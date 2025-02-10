from py4j.protocol import Py4JError, Py4JJavaError, Py4JNetworkError


class IcebergWarning(Warning):
    pass


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

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    @classmethod
    def from_java_error(cls, e: Py4JError):
        # TODO: Include stacktrace in dev mode?
        if isinstance(e, Py4JJavaError):
            if (
                str(e.java_exception.getClass())
                == "class org.apache.iceberg.exceptions.NoSuchTableException"
            ):
                return cls("No such Iceberg table found: " + str(e.java_exception))
            elif (
                str(e.java_exception.getClass())
                == "class org.apache.iceberg.exceptions.RuntimeIOException"
            ):
                return cls("Unable to access Iceberg table: " + str(e.java_exception))
            else:
                return cls("Unknown Iceberg Java Error: " + str(e.java_exception))
        elif isinstance(e, Py4JNetworkError):
            return cls("Unexpected Py4J Network Error: " + str(e))
        else:
            assert isinstance(e, Py4JError), (
                "from_java_error(): Only Py4JErrors should be handled."
            )
            return cls("Unexpected Py4J Error: " + str(e))
