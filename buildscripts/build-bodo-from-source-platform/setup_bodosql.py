import sys

from common import parse_machinefile, run_script, send_file

machinefile = sys.argv[1]
hosts = parse_machinefile(machinefile)

send_file(hosts, "update_bodosql_dev_env.sh")
run_script(hosts, "update_bodosql_dev_env.sh")
