from IPython.core.magic import (
    Magics,
    magics_class,
)
import sys


@magics_class
class PlatformMagicsBase(Magics):
    """
    Base class with some utility functions, etc.
    """

    def __display_usage_error__(self, err_msg):
        """
        Display a usage error message
        """
        self.shell.show_usage_error(err_msg)

    def __display_error_message__(self, err_msg):
        """
        Display an error message
        """
        print("Error: %s" % err_msg, file=sys.stderr)
