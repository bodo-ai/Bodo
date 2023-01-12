"""
 To run: python restart_efa.py hostfile
 ON EFA clusters, run this script when you get this error
 Abort(1615503) on node 157 (rank 157 in comm 0): Fatal error in PMPI_Init_thread: Other MPI error, error stack:
 MPIR_Init_thread(136)........:
 MPID_Init(904)...............:
 MPIDI_OFI_mpi_init_hook(1421):
 MPIDU_bc_table_create(338)...: Missing hostname or invalid host/port description in business card
 Error in system call pthread_mutex_destroy: Device or resource busy
    ../../src/mpi/init/init_thread_cs.c:60
"""
import subprocess
import sys

from common import parse_machinefile

machinefile = sys.argv[1]
hosts = parse_machinefile(machinefile)


def restart(hosts):
    processes = []
    for host in hosts:
        cmd = ["ssh", f"{host}", f"pkill -u bodo"]
        p = subprocess.Popen(cmd)
        processes.append(p)
    for p in processes:
        rc = p.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)


restart(hosts)
