from collections import Counter
from typing import List
import re

from ipyparallel.cluster.launcher import SlurmEngineSetLauncher, MPIEngineSetLauncher
from .utils import BColor, colorize_output
import os
import os.path


# check if we are using a new image that has Slurm installed, otherwise fall back to
# regular MPI engines
slurm_installed = os.path.isfile("/etc/slurm/cgroup.conf")
# Feature flag for testing Slurm. This will be set during instance initialization (userdata).
# Default: `false` until the feature is stabilized.
use_slurm_if_installed = os.environ.get("BODO_CONF_USE_SLURM", "false").lower() == "true"
if slurm_installed and use_slurm_if_installed:
    EngineSetLauncher = SlurmEngineSetLauncher
else:
    EngineSetLauncher = MPIEngineSetLauncher

class BodoPlatformMPIEngineSetLauncher(EngineSetLauncher):
    def _get_error_message_from_exit_code(self, exit_code: str):
        message_dict = {
            "9": "This usually indicates an out-of-memory (OOM) issue. Try with a larger cluster?",
            "11": "This is usually indicative of memory issues, or a bug in the application.",
        }

        if exit_code in message_dict:
            return colorize_output(BColor.FAIL, f"Detected exit code {exit_code}. {message_dict[exit_code]}\n")

        return colorize_output(BColor.FAIL, f"Execution failed. Detected exit code: {exit_code}\n")

    # This function is called each time there's an mpiexec error. We are overriding the
    # implementation of the parent class here with a custom implementation that logs
    # a more helpful error message based on heuristics (like log aggregation from engines)
    def _log_output(self, stop_data):
        """Logs mpiexec error output, if any.

        Parameters
        ----------
            stop_dict : dict
                contains metadata about error (exit code, etc)
        """
        # Get the output lines from the engines
        output = self.get_output(remove=False)

        # List to store process terminal signals across all engines
        process_termination_signals: List[str] = []

        # Iterate over each line of the output to aggregate exit codes of the processes
        for line in output.splitlines(True):
            m = re.match(f'.*(KILLED BY SIGNAL:\s(\d+)).*', line)
            if m is not None:
                process_termination_signals.append(m.group(2))

        if len(process_termination_signals) > 0:
            # Get the most common signal
            exit_signal = Counter(
                process_termination_signals).most_common(1)[0][0]
            exit_message = self._get_error_message_from_exit_code(exit_signal)
            self.log.error(exit_message)

        super()._log_output(stop_data)
