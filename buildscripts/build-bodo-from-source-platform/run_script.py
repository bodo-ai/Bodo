import sys

from common import parse_machinefile, run_script

machinefile = sys.argv[1]
file_to_run = sys.argv[2]
hosts = parse_machinefile(machinefile)

run_script(hosts, file_to_run)
