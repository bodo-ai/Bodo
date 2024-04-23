from IPython.core.magic import (
    line_magic,
    magics_class,
)
from IPython.core import magic_arguments

from .magics_base import PlatformMagicsBase
from .helper import execute_shell
from .platform_utils import PLATFORM_HOSTFILE_UPDATE_SCRIPT_LOC


@magics_class
class HostfileUpdateMagic(PlatformMagicsBase):
    """
    Magic to update hostfile.
    """

    @line_magic
    @magic_arguments.magic_arguments()
    @magic_arguments.argument(
        "--timeout", type=int, default=None, help="number of seconds to timeout after."
    )
    @magic_arguments.argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Print information such as the shell commands being run and its output.",
    )
    def update_hostfile(self, line="", local_ns=None):
        """
        Bodo IPython Magic to update the cluster's hostfile.
        Simply calls the /tmp/update_hostfile.sh script.
        """

        args = magic_arguments.parse_argstring(self.update_hostfile, line)

        cmd = f"/bin/sh {PLATFORM_HOSTFILE_UPDATE_SCRIPT_LOC}"

        returncode, timed_out = execute_shell(cmd, args.timeout, args.verbose)
        if timed_out:
            self.__display_error_message__("Timed out!")
        if args.verbose:
            print("returncode: ", returncode)

        # If successful
        if returncode == 0:
            print("Successfully updated hostfile. Please restart the kernel.")
