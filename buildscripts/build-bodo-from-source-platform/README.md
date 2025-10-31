# build-bodo-from-source-platform

## Set up DEV environment on platform

> [!NOTE]
> If the node is too small, it may run out of memory during the build process.
> In that case, you can try to build on a larger node or use environment variable `CMAKE_BUILD_PARALLEL_LEVEL`
> to manually control the number of processes used.
> Try setting it to a single process using `export CMAKE_BUILD_PARALLEL_LEVEL=1`

> [!NOTE]
> When running step 3 (or any other similar building and cloning instructions) below, please avoid building in `/bodofs`. Its usage has to be limited to storing basic files and nothing that requires high disk performance or full POSIX.

1. SSH into the cluster node (`My Notebook` -> `BODO CLUSTERS` -> `Terminal` beside the running cluster).
1. Copy `install.sh` to the node.
1. Run the script: `bash -i install.sh` (recommended in directory `/home/bodo`).
1. Enter your GitHub PAT to checkout when prompted (alternatively,
   `GITHUB_TOKEN` can be set as environment variable)
1. Run `pixi shell -e platform-dev` to enter the environment created by the install
   script.

> [!NOTE]
> Modify time for bodosqlwrapper.py has to be the same when installed from source on different nodes since files having different timestamps breaks Numba caching.
> Therefore, after any source code changes to `bodosqlwrapper.py`, run:
> `psh touch -am -t 202401010000 ~/Bodo/bodo-platform-image/bodo-platform-utils/bodo_platform_utils/bodosqlwrapper.py`
> See https://bodo.atlassian.net/browse/BSE-3582

> [!NOTE]
> Do not run programs in a `/bodofs` working directory since some Bodo internals may be using current working directory and `/bodofs` is slow.
> We have seen significant Snowflake write and compilation slow down when current working directory is on `/bodofs`.
> Both local working directory and nfs seem fine to use. For example, `cd ~; px python ...` works.

## Workflow

It's recommended to create a working branch while testing. You can push changes to this branch from your
personal machine, and then pull the changes on the nodes and rebuild. e.g.

1. Checkout the branch (only required once): `psh git checkout working-branch`.
1. Pull latest changes (do this after every push): `psh git pull`.
1. Re-build: `BODO_SKIP_CPP_TESTS=1 psh pip install --no-deps --no-build-isolation -ve . && cd BodoSQL && psh python -m pip install --no-deps --no-build-isolation -ve . && cd ..`.

## Using Bodo/BodoSQL

1. Drop into the `DEV` environment by running `source activate DEV`
1. Make sure you have set your AWS access credentials if you need access to s3. E.g.

   ```
   export AWS_ACCESS_KEY_ID="xyz"
   export AWS_SECRET_ACCESS_KEY="xyz"
   ```

   Use an [instance-role](https://docs.bodo.ai/latest/installation_and_setup/bodo_platform/#instance_role_cluster) whenever possible, to avoid having to set these.

1. Edit the `hostfile` to ensure the node you are currently connected to is first.
   This is useful/necessary because IO happens to the node that's first in the hostfile.
   For example, if you are generating trace data it will always get dumped to the node that's first in the `hostfile`

1. Run jobs, but be aware that anything you're doing on the node does not count as activity, so auto-shutdown might be invoked if you don't have a notebook connected.

1. Ensure that you are using efa. This can be done by executing: `I_MPI_DEBUG=4 px python -u -c "import bodo; print(bodo.__version__)"` and checking that `libfabric provider: efa` appears at the start of the logs.

1. You can use the `BodoSQLWrapper.py` script to execute a SQL query from a `.sql` file directly:
   `px python -u BodoSQLWrapper.py -c snowflake_creds.json -f query.sql -p query_pandas.py -w <BODO_WH> -d <BODO_DB> -o bodo_out.pq 2>&1 | tee logs-$(date '+%Y-%m-%d-%H:%M:%S').txt`.
   `2>&1 | tee log.txt` sends both stdout and stderr to both the terminal and the log file.
   `$(date '+%Y-%m-%d-%H:%M:%S')` appends the execution time to the name of the log file.
   This is useful to ensure that all log files are retained and not overwritten by future executions.

## Useful notes

- To get print statements to show up immediately, set `export PYTHONUNBUFFERED=1`
- Put your test files in `/bodofs`, but do not run programs in a `/bodofs` working directory (some Bodo internals may be using current working directory).
- Sometimes error messages aren't helpful because they're treated as internal errors. In that case set `export NUMBA_DEVELOPER_MODE=1`.
- To get numba caching information, set `export NUMBA_DEBUG_CACHE=1`.
- If you set `export BODO_SKIP_CPP_TESTS=1` you can skip compiling the C++ tests, which can take a long time to build.

## Troubleshooting
If you encounter any issues, [check this confluence document](https://bodo.atlassian.net/wiki/spaces/B/pages/1894416388/Troubleshooting+build+bodo+from+source+on+platform).
