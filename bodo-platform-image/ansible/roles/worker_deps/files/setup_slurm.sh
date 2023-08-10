set -eo pipefail

if [ -z "$1" ]; then
    echo "setup_slurm.sh: need to pass hostfile"
    exit 1
fi
export HOST_FILE=$1
# need to write into random place in case of two jobs running at once
TMP_CONF=/tmp/slurm${RANDOM}.conf
TMP_DBCONF=/tmp/slurmdb${RANDOM}.conf
cp /home/bodo/slurm.conf.template TMP_CONF
cp /home/bodo/slurmdb.conf.template TMP_DBCONF

# add controller hostname to conf files
export SLURMCTLD_HOST=`hostname`
sed -i "s/# SlurmctldHost=/SlurmctldHost=$SLURMCTLD_HOST/g" TMP_CONF
sed -i "s/# DbdHost=/DbdHost=$SLURMCTLD_HOST/g" TMP_DBCONF
sudo mv TMP_DBCONF /etc/slurm/slurmdbd.conf
sudo chown slurm /etc/slurm/slurmdbd.conf
sudo chmod 600 /etc/slurm/slurmdbd.conf

# assuming the cluster is homogeneous and controller is one of the nodes
export LSCPU_OUT=$(lscpu)
export THREADS_PER_CORE=$(echo "$LSCPU_OUT" | grep 'Thread(s) per core:' | awk '{print $4}')
export VCPU_COUNT=$(echo "$LSCPU_OUT" | grep 'CPU(s):' | grep -v NUMA | awk '{print $2}')


# add compute nodes to slurm.conf
export NODE_COUNTER=0
for NODE in $(cat $HOST_FILE | sort | uniq); do
  export NODE_HOSTNAME=$(ssh -q -o "StrictHostKeyChecking no" $NODE "hostname")
  # Should be same as $NODE, getting it again in case hostfile format changed
  export NODE_ADDR=$(ssh -q -o "StrictHostKeyChecking no" $NODE "hostname -I | awk '{print \$1}'")
  echo "NodeName=Node$NODE_COUNTER NodeHostName=$NODE_HOSTNAME NodeAddr=$NODE_ADDR CPUs=$VCPU_COUNT ThreadsPerCore=$THREADS_PER_CORE State=UNKNOWN" >> TMP_CONF
  let NODE_COUNTER=$NODE_COUNTER+1
done

# add partition info after compute info
echo "" >> TMP_CONF
echo "PartitionName=JobPartition Nodes=ALL Default=YES MaxTime=INFINITE State=UP OverSubscribe=NO" >> TMP_CONF

# start slurmd on all nodes
for NODE in $(cat $HOST_FILE | sort | uniq); do
  scp -o "StrictHostKeyChecking no" TMP_CONF $NODE:TMP_CONF
  ssh -o "StrictHostKeyChecking no" $NODE sudo cp TMP_CONF /etc/slurm/slurm.conf
  ssh -o "StrictHostKeyChecking no" $NODE sudo systemctl stop slurmd.service
  ssh -o "StrictHostKeyChecking no" $NODE sudo systemctl start slurmd.service
  ssh -o "StrictHostKeyChecking no" $NODE sudo systemctl status slurmd.service
done

sudo systemctl stop slurmdbd
sudo systemctl start slurmdbd
sudo systemctl status slurmdbd

# wait till slurm db be ready to accept connections
while ! nc -z 127.0.0.1 6819; do   sleep 1; done

sudo systemctl stop slurmctld.service
sudo systemctl start slurmctld.service
sudo systemctl status slurmctld.service
