from jpype import JException


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

    def __init__(self, message: str, java_error: JException):
        super().__init__(message)
        self.message = message
        self.java_error = java_error

    @classmethod
    def from_java_error(cls, e: JException):
        if e.__class__.__name__ == "org.apache.iceberg.exceptions.NoSuchTableException":
            return cls("No such Iceberg table found", e)
        elif e.__class__.__name__ == "org.apache.iceberg.exceptions.RuntimeIOException":
            return cls("Unable to find Iceberg table", e)
        else:
            return cls("Unknown Iceberg Error", e)
