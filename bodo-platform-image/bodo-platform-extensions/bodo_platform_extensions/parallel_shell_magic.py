from IPython.core.magic import (
    line_magic,
    magics_class,
)
from .magics_base import PlatformMagicsBase
from .helper import execute_on_all_nodes


@magics_class
class ParallelShellMagics(PlatformMagicsBase):
    """
    Magic to run any shell command on all nodes.
    Under the hood this just runs the command
    on all hosts using mpiexec.
    """

    @line_magic
    def psh(self, line):
        """
        Execute a shell command on all nodes using mpiexec.
        """
        full_command = line.strip()
        # Execute on all nodes (in verbose mode, without timeout)
        returncode, timed_out = execute_on_all_nodes(
            full_command,
            None,
            True,
        )
        # Handle output
        if timed_out:
            self.__display_error_message__("Timed out!")
        print("returncode: ", returncode)
        if returncode == 0:
            print("\nSuccessfully executed on all nodes.")
