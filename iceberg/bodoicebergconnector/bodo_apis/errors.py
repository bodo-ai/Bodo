import jpype


class IcebergError(Exception):
    """
    Exception raised when Java Exception is Captured from Iceberg Connector Code

    Attributes:
        exception: Java Exception that was captured
        message: Explaination, either with the Java Exception or Custom
    """

    def __init__(self, exception: str, message: str):
        self.exception = exception
        self.message = message
        super().__init__(
            f"Error Reported by Iceberg Connector ({self.exception}): {self.message}"
        )

    @classmethod
    def from_java_exception(cls, e: jpype.JException):
        return cls(e.__class__.__name__, e.message())
