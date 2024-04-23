import subprocess
from .platform_utils import get_cluster_hosts


def execute_shell(cmd, timeout, verbose=False):

    if isinstance(cmd, str):
        cmd = cmd.split()

    if verbose:
        print("Command: ", " ".join(cmd))

    returncode = None
    timed_out = False

    try:
        p = subprocess.Popen(cmd)
        returncode = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired as e:
        p.kill()  # Kill the process if it has timed out
        returncode = 124  # Usual shell error code for timeouts
        timed_out = True

    return returncode, timed_out


def execute_on_all_nodes(command: str, timeout=None, verbose=False):
    """
    Execute the given command on all nodes in the cluster
    via mpiexec.
    """

    args = [
        "mpiexec",
        "-prepend-rank",  # To make the output easier to parse through
        "-ppn",  # Run once per node
        "1",
    ]
    if hosts := get_cluster_hosts(verbose=verbose):
        args += [
            "-hosts",
            ",".join(hosts),
        ]

    args += command.split()

    return execute_shell(args, timeout=timeout, verbose=verbose)


def copy_file_to_all_nodes(source, dest, verbose=False, timeout=None):
    hosts = get_cluster_hosts()

    num_timed_out = 0
    num_errored_out = 0

    processes = []
    for host in hosts:
        cmd = ["scp", source, f"{host}:{dest}"]
        if verbose:
            print("Running: ", " ".join(cmd))
        p = subprocess.Popen(cmd)
        processes.append(p)

    for host, p in zip(hosts, processes):
        try:
            # TODO This is flawed logic since we're giving each host
            # timeout seconds.
            rc = p.wait(timeout=timeout)
            if rc != 0:
                raise subprocess.CalledProcessError(rc, cmd)
        except subprocess.TimeoutExpired as e:
            p.kill()  # Kill the process if it has timed out
            num_timed_out += 1
            if verbose:
                print(f"Timed out on {host}")
        except subprocess.CalledProcessError as e:
            num_errored_out += 1
            if verbose:
                print(f"Errored out on {host}. Exit Code: {rc}")

    return num_timed_out, num_errored_out, len(hosts)
