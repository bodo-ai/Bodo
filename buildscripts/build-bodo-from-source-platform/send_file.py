import sys

from common import parse_machinefile, send_file

machinefile = sys.argv[1]
file_to_send = sys.argv[2]
hosts = parse_machinefile(machinefile)

send_file(hosts, file_to_send)
