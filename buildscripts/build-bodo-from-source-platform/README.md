# build-bodo-from-source-platform

## Set up DEV environment on platform

1. SSH into any of the cluster nodes.
1. Set your [Github token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token#creating-a-personal-access-token-classic) as an environment variable: `export GITHUB_TOKEN=<token>`
1. Clone Bodo repository on all machines using a Github token: `psh git clone https://$GITHUB_TOKEN@github.com/Bodo-inc/Bodo.git`.
   This will also set the token in the Git Remote Origin URL, and therefore future git actions won't ask for credentials.
1. Install `conda-lock` on all nodes: `psh sudo /opt/conda/bin/mamba install conda-lock -c conda-forge -n base --yes`
1. Remove `mpi` and `mpich` on all nodes: `psh sudo /opt/conda/bin/conda remove mpi mpich -n base --force --yes`
1. Navigate to the folder with the environment lock file: `cd Bodo/buildscripts/envs`
1. Create a DEV environment from the lock file: `psh conda-lock install --dev --mamba -n DEV conda-lock.yml`
1. Activate the environment: `conda activate DEV`
1. Remove `mpi` and `mpich` on all nodes for the DEV environment. `psh conda remove mpi mpich --force --yes`
1. Navigate to base folder of Bodo repo: `cd ~/Bodo`
1. Build Bodo: `psh python setup.py develop`
1. Build BodoSQL: `cd BodoSQL && psh python setup.py develop && cd ..`

It's recommended to create a working branch while testing. You can push changes to this branch from your
personal machine, and then pull the changes on the nodes and rebuild. e.g.

1. Checkout the branch (only required once): `psh git checkout working-branch`.
1. Pull latest changes (do this after every push): `psh git pull`.
1. Re-build: `psh python setup.py develop && cd BodoSQL && psh python setup.py develop && cd ..`.

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
   On AWS workspaces, running `update_hostfile` from the terminal should update the activity timer.

1. Ensure that you are using efa. This can be done by executing: `I_MPI_DEBUG=4 px python -u -c "import bodo; print(bodo.__version__)"` and checking that `libfabric provider: efa` appears at the start of the logs.

1. You can use the `BodoSQLWrapper.py` script to execute a SQL query from a `.sql` file directly:
   `px python -u BodoSQLWrapper.py -c snowflake_creds.json -f query.sql -p query_pandas.py -w <BODO_WH> -d <BODO_DB> -o bodo_out.pq 2>&1 | tee logs-$(date '+%Y-%m-%d-%H:%M:%S').txt`.
   `2>&1 | tee log.txt` sends both stdout and stderr to both the terminal and the log file.
   `$(date '+%Y-%m-%d-%H:%M:%S')` appends the execution time to the name of the log file.
   This is useful to ensure that all log files are retained and not overwritten by future executions.

## Useful notes

- To get print statements to show up immediately, set `export PYTHONUNBUFFERED=1`
- Put your test files in `/shared` (or `/bodofs` based on workspace version).
- Sometimes error messages aren't helpful because they're treated as internal errors. In that case set `export NUMBA_DEVELOPER_MODE=1`.
- To get numba caching information, set `export NUMBA_DEBUG_CACHE=1`.
