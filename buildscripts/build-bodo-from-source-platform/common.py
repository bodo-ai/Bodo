import subprocess


def send_file(hosts, fname, folder=False):
    cmd_template = ["scp"]
    if folder:
        cmd_template.append("-r")
    cmd_template.append(fname)

    processes = []
    for host in hosts:
        cmd = cmd_template + [f"{host}:~"]
        p = subprocess.Popen(cmd)
        processes.append(p)
    for p in processes:
        rc = p.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)


def run_script(hosts, script_name):
    processes = []
    for host in hosts:
        cmd = ["ssh", f"{host}", f"bash {script_name}"]
        p = subprocess.Popen(cmd)
        processes.append(p)
    for p in processes:
        rc = p.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)


def parse_machinefile(fname):
    hosts = []
    with open(fname, "r") as f:
        for l in f:
            hosts.append(l.rstrip("\n"))
    return hosts
