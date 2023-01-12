# build-bodo-from-source-platform

## Building Bodo/BodoSQL

1. On your local machine (or notebook instance), clone Bodo, change to the needed branch and/or change source code as needed
2. Create bodo.tar.gz (`tar -czf bodo.tar.gz Bodo`)
3. Copy the tarball to platform using `scp ./bodo.tar.gz bodo@<IP-ADDRESS>`
4. Untar on the platform (`tar -xzf bodo.tar.gz`)
5. scp this (build-bodo-from-source-platform) repo into your home directory on the platform
6. Copy/move the files out of the repo: `cp ./build-bodo-from-source-platform/* .`
7. Run `cp /shared/.hostfile-<UIUD> hostfile` where `<UIUD>` is the UIUD of the cluster
8. Run `python setup.py hostfile`
9. Run `python build.py hostfile`

## Using Bodo/BodoSQL

1. Drop into the `DEV` environment by running `source activate DEV`
2. Make sure you have set your AWS access credentials if you need access to s3. E.g.

    ```
    export AWS_ACCESS_KEY_ID="xyz"
    export AWS_SECRET_ACCESS_KEY="xyz"
    ```
3. Edit the `hostfile` to ensure the node your currently connected to is first. This is useful/necessary because IO happens to the node that's first in the hostfile. For example, if you are generating trace data it will always get dumped to the node that's first in the `hostfile`
4. Transfer any files you need to other nodes, eg if you want to run `queries.py` you can do `python ./send_file.py hostfile queries.py` so that MPI can find the file on each node. Sharing a file on `/shared` can cause race conditions with the python/JIT caching.
5. Run jobs, but be aware that anything you're doing on the node does not count as activity, so autoshutdown might be invoked if you don't have a notebook connected.


## Useful notes
- To get print statements to show up immediately, set `export PYTHONUNBUFFERED=1`
- You have to either copy scripts like `queries.py` to all nodes or put them in `/shared` if they're small enough.
- Sometimes error messages aren't helpful because they're treated as internal errors. In that case set `export NUMBA_DEVELOPER_MODE=1`
- When you get this error messge
      `Missing hostname or invalid host/port description on busuiness card`
  Run `python restart_efa.py hostfile`. This wil kill all PID in the cluster and close your current connection. You can `ssh` again and resume your work.
