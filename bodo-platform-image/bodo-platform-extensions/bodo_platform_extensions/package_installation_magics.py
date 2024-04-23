from IPython.core.magic import (
    line_magic,
    magics_class,
)
from IPython.core.magics.packaging import (
    CONDA_COMMANDS_REQUIRING_PREFIX,
    CONDA_COMMANDS_REQUIRING_YES,
    CONDA_ENV_FLAGS,
    CONDA_YES_FLAGS,
    _get_conda_executable,
)
import shlex
import sys
from IPython.core import magic_arguments
from .magics_base import PlatformMagicsBase
from .helper import execute_on_all_nodes


@magics_class
class PackageInstallationMagics(PlatformMagicsBase):
    """
    Magics to run conda and pip commands
    on all nodes in a cluster.
    """

    @line_magic
    def pconda(self, line):
        """
        Inspired by IPython's own conda install magic
        https://github.com/ipython/ipython/blob/96617c61df3edba5f723c7a154f5f58abf513c7f/IPython/core/magics/packaging.py#L80
        """
        conda = _get_conda_executable()  # Needs testing, else use /opt/conda/bin/conda
        args = shlex.split(line)
        command = args[0] if len(args) > 0 else ""
        args = args[1:] if len(args) > 1 else [""]

        extra_args = []
        # we need to insert --yes in the argument list for some commands
        needs_yes = command in CONDA_COMMANDS_REQUIRING_YES
        has_yes = set(args).intersection(CONDA_YES_FLAGS)
        if needs_yes and not has_yes:
            extra_args.append("--yes")

        # Add --prefix to point conda installation to the current environment
        needs_prefix = command in CONDA_COMMANDS_REQUIRING_PREFIX
        has_prefix = set(args).intersection(CONDA_ENV_FLAGS)
        if needs_prefix and not has_prefix:
            extra_args.extend(["--prefix", sys.prefix])

        full_command = f"sudo {' '.join([conda, command] + extra_args + args)}"
        full_command = full_command.strip()
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
            print(
                "\nSuccessfully executed on all nodes. Note: you may need to restart the kernel to use updated packages."
            )
        # TODO Remove mpich/mpi if it was accidentally installed

    @line_magic
    def ppip(self, line):
        """
        Inspired by IPython's own conda install magic
        https://github.com/ipython/ipython/blob/96617c61df3edba5f723c7a154f5f58abf513c7f/IPython/core/magics/packaging.py#L63
        """
        python = sys.executable
        python = shlex.quote(python)
        full_command = " ".join([python, "-m", "pip", line])
        full_command = full_command.strip()
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
            print(
                "\nSuccessfully executed on all nodes. Note: you may need to restart the kernel to use updated packages."
            )
