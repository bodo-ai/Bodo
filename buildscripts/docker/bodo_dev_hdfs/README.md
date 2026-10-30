# Running Bodo HDFS Tests

Run from the Bodo repository root on Linux.

## Build Docker image

```bash
docker build -t bodo_dev_hdfs \
  -f buildscripts/docker/bodo_dev_hdfs/Dockerfile .
```

## Start container

```bash
docker run --rm -it --network host \
  -v "$PWD:/Bodo" \
  -v bodo_hdfs_pixi:/Bodo/.pixi \
  -w /Bodo bodo_dev_hdfs bash
```

The named volume preserves the container's Pixi environment
between runs without conflicting with the host's environment.

## Initialize and start HDFS

Inside the container:

```bash
pixi install
pixi run build-bodo

pixi shell

# Configure Hadoop JNI
export CLASSPATH="$($HADOOP_HOME/bin/hdfs classpath --glob)"

# Initialize a fresh, disposable HDFS cluster
hdfs namenode -format -force -nonInteractive

# Start services
hdfs --daemon start namenode
hdfs --daemon start datanode

# Verify one live DataNode
hdfs dfsadmin -report
```

Only format the NameNode for a fresh cluster.
Do not reformat an existing cluster containing data.

## Run tests

```bash
BODO_SPAWN_MODE=0 pytest -svWignore bodo/tests/test_hdfs.py
```

The HDFS NameNode and DataNode run inside the container.
On subsequent runs, the Pixi environment is preserved, but
the disposable HDFS cluster must be initialized again.
