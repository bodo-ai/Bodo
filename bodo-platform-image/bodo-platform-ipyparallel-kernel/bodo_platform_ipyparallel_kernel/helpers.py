import subprocess


def execute_shell(cmd, timeout, verbose=False):

    if isinstance(cmd, str):
        cmd = cmd.split()

    if verbose:
        print("Command: ", " ".join(cmd))

    stdout_ = ""
    stderr_ = ""
    returncode = None
    timed_out = False

    try:
        res = subprocess.run(cmd, capture_output=True, timeout=timeout)
        stdout_ = res.stdout.decode("utf-8")
        stderr_ = res.stderr.decode("utf-8")
        returncode = res.returncode
    except subprocess.TimeoutExpired as e:
        stdout_ = e.stdout.decode("utf-8") if e.stdout else e.stdout
        stderr_ = e.stderr.decode("utf-8") if e.stderr else e.stderr
        returncode = 124  # Usual shell error code for timeouts
        timed_out = True
    # TODO Handle other types of exceptions

    return stdout_, stderr_, returncode, timed_out
