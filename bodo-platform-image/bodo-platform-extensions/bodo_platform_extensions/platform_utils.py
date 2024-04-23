import os

PLATFORM_HOSTFILE_UPDATE_SCRIPT_LOC = "/tmp/update_hostfile.sh"


USUAL_PLATFORM_HOSTFILE_LOCATIONS = list(
    set(
        [
            "/home/bodo/hostfile",
            "/home/bodo/machinefile",
            os.path.expanduser("~/hostfile"),
            os.path.expanduser("~/machinefile"),
        ]
    )
)


def get_hosts_from_hostfile(hostfile_path):
    with open(hostfile_path, "r") as f:
        lines = f.readlines()
        lines = list(map(str.strip, lines))
    # If they're of the form host:cores, get just the hosts
    hosts = list(map(lambda x: x.split(":")[0], lines))
    # Remove empty lines
    hosts = list(filter(lambda x: len(x) > 0, hosts))
    # Remove any duplicates
    hosts = list(set(hosts))
    return hosts


def get_cluster_hosts(verbose=False):
    hosts = []
    potential_hostfile_paths = []

    # If I_MPI_HYDRA_HOST_FILE is defined, it takes precendence
    if f := os.environ.get("I_MPI_HYDRA_HOST_FILE"):
        potential_hostfile_paths.append(f)
    # Look for hostfile in these locations next
    potential_hostfile_paths.extend(USUAL_PLATFORM_HOSTFILE_LOCATIONS)

    # Check if these paths exists. The first one is assumed to be correct.
    for path in potential_hostfile_paths:
        if os.path.exists(path):
            if verbose:
                print(f"Reading from hostfile: {path}")
            hosts = get_hosts_from_hostfile(path)
            break
    if (not hosts) and verbose:
        print("No hostfile found. Looked at following locations:\n\t", end="")
        print("\n\t".join(potential_hostfile_paths))
    return hosts
