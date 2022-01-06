""" This is a script which invokes mpiexec binary (which needs
    to be in the same folder) and passes through any command line
    arguments to it """
import sys
import os
import os.path


def main(args=None):
    import subprocess

    if args is None:
        args = sys.argv[1:]

    cmd = [os.path.join(os.path.dirname(__file__), 'mpiexec')]
    cmd.extend(args)
    try:
        return subprocess.call(cmd)
    except Exception as e:
        print(e)
        return 1


if __name__ == '__main__':
    sys.exit(main())
