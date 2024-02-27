import psutil
import os

USUAL_PLATFORM_HOSTFILE_LOCATIONS = [
    "/home/bodo/hostfile",
    "/home/bodo/machinefile",
    os.path.expanduser("~/hostfile"),
    os.path.expanduser("~/machinefile"),
]


def get_hosts():
    hosts = []
    potential_hostfile_paths = []
    # hostfile required for setting up Slurm below
    hostfile = None

    # If I_MPI_HYDRA_HOST_FILE is defined, it takes precendence
    if f := os.environ.get("I_MPI_HYDRA_HOST_FILE"):
        potential_hostfile_paths.append(f)
    # Similarly also check HYDRA_HOST_FILE
    if f := os.environ.get("HYDRA_HOST_FILE"):
        potential_hostfile_paths.append(f)
    # Look for hostfile in these locations next
    potential_hostfile_paths.extend(USUAL_PLATFORM_HOSTFILE_LOCATIONS)
    # Remove duplicates
    potential_hostfile_paths = list(set(potential_hostfile_paths))

    # Check if these paths exists. The first one is assumed to be correct.
    for path in potential_hostfile_paths:
        if os.path.exists(path):
            with open(path, "r") as f:
                lines = f.readlines()
                lines = list(map(str.strip, lines))
            # If they're of the form host:cores, get just the hosts
            hosts = list(map(lambda x: x.split(":")[0], lines))
            # Remove empty lines
            hosts = list(filter(lambda x: len(x) > 0, hosts))
            # Remove any duplicates
            hosts = list(set(hosts))
            hostfile = path
            break
    if not hosts:
        print("No hostfile found. Looked at following locations:\n\t", end="")
        print("\n\t".join(potential_hostfile_paths))
    return hosts, hostfile


c.Cluster.engine_launcher_class = "bodo"

hosts, hostfile = get_hosts()
cores_per_node = psutil.cpu_count(logical=False)
n_nodes = max(len(hosts), 1)
n = cores_per_node * n_nodes
c.Cluster.n = n

mpi_args = ["-ppn", "1"]  ## For round-robin placement
# mpi_args = ["-ppn", f"{cores_per_node}"]  ## For block-placement
if hosts:
    mpi_args.extend(["-hosts", ",".join(hosts)])
c.MPIEngineSetLauncher.mpi_args = mpi_args

c.SlurmEngineSetLauncher.batch_template = f"""#!/bin/sh
#SBATCH --export=ALL
#SBATCH --job-name=ipengine
#SBATCH --ntasks={{n}}
#SBATCH --cpus-per-task=2
export I_MPI_PMI_LIBRARY="/usr/lib64/libpmi.so"
srun --cpus-per-task=2 --mpi=pmi2 --cpu-bind=cores --distribution=cyclic {{program_and_args}}
"""
SLURM_SETUP_TIMEOUT_SECONDS = 30
# check if we are using a new image that has Slurm installed, otherwise fall back to
# regular MPI engines
slurm_installed = os.path.isfile("/etc/slurm/cgroup.conf")
use_slurm_if_installed = (
    os.environ.get("BODO_CONF_USE_SLURM", "false").lower() == "true"
)

if slurm_installed and use_slurm_if_installed:
    from bodo_platform_ipyparallel_kernel.helpers import execute_shell

    os.environ["I_MPI_PMI_LIBRARY"] = "/usr/lib64/libpmi.so"
    # configure and start Slurm
    stdout_, stderr_, returncode, timed_out = execute_shell(
        f"sh /home/bodo/setup_slurm.sh {hostfile}",
        timeout=SLURM_SETUP_TIMEOUT_SECONDS,
    )

    if returncode != 0:
        print(f"Slurm setup failed\n{stdout_}\n{stderr_}")

    # Cancel other running notebooks to avoid waiting for resources
    stdout_, stderr_, returncode, timed_out = execute_shell(
        "scancel -n ipengine",
        timeout=SLURM_SETUP_TIMEOUT_SECONDS,
    )

    if returncode != 0:
        print(f"Slurm setup failed (scancel)\n{stdout_}\n{stderr_}")


c.Cluster.controller_ip = "*"
c.Cluster.controller_args = ["--nodb"]
