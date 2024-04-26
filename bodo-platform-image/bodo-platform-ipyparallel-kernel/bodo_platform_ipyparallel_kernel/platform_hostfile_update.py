from .helpers import execute_shell
import os

# Check in multiple locations (for backward compatibility)
POTENTIAL_UPDATE_HOSTFILE_SCRIPT_PATHS = [
    os.path.expanduser("~/update_hostfile.sh"),
    "/home/bodo/update_hostfile.sh",
    "/tmp/update_hostfile.sh",
]


def update_hostfile(logger):
    update_hostfile_script_found = False
    for path in POTENTIAL_UPDATE_HOSTFILE_SCRIPT_PATHS:
        if os.path.exists(path):
            logger.info(f"Found file: {path}. Running...")
            update_hostfile_script_found = True
            stdout_, stderr_, returncode, timed_out = execute_shell(
                f"sh {path}", timeout=30, verbose=False
            )
            if returncode == 0:
                logger.info(f"Update Hostfile: SUCCESS")
            else:
                logger.warning(f"Update Hostfile: FAILED")
            logger.debug(f"Update Hostfile returncode: {returncode}")
            logger.debug(f"Update Hostfile timed_out: {timed_out}")
            logger.debug(f"Update Hostfile STDOUT:\n{stdout_}")
            logger.debug(f"Update Hostfile STDERR:\n{stderr_}")
            break
    # Log a warning in case none of the files were found
    if not update_hostfile_script_found:
        logger.warning(
            f"Could not update hostfile. Couldn't find any of these files: {','.join(POTENTIAL_UPDATE_HOSTFILE_SCRIPT_PATHS)}."
        )
