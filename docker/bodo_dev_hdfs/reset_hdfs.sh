cd /opt/hadoop-3.2.1
sbin/stop-dfs.sh
rm -rf /tmp/hadoop-root
bin/hdfs namenode -format
sbin/start-dfs.sh
jps
