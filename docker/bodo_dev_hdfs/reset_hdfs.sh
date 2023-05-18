cd /opt/hadoop-3.3.2
sbin/stop-dfs.sh
rm -rf /tmp/hadoop-root
bin/hdfs namenode -format
sbin/start-dfs.sh
jps
