import sys

from common import parse_machinefile, run_script, send_file

machinefile = sys.argv[1]
hosts = parse_machinefile(machinefile)

send_file(hosts, "bodo.tar.gz")
# send_file(hosts, "test.py")
send_file(hosts, "build.sh")
run_script(hosts, "build.sh")
