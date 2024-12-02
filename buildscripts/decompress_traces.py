import sys
import zlib

# Decompress JSON trace file compressed with zlib and no zlib header.
# Usage: python decompress_traces.py in_file out_file

in_fname = sys.argv[1]
out_fname = sys.argv[2]

with open(in_fname, "rb") as f:
    json_str = zlib.decompress(f.read(), wbits=-15).decode()

with open(out_fname, "w") as f:
    f.write(json_str)
