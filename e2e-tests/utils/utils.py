import subprocess


def run_cmd(cmd, print_output=True):
    # TODO: specify a timeout to check_output or will the CI handle this? (e.g.
    # situations where Bodo hangs for some reason)
    # stderr=subprocess.STDOUT to also capture stderr in the result
    try:
        output = subprocess.check_output(
            cmd, stderr=subprocess.STDOUT, text=True, errors="replace"
        )
    except subprocess.CalledProcessError as e:
        print(e.output)
        raise
    if print_output:
        print(output)
    return output
