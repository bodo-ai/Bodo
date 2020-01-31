#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
cores=16
today=$(TZ=":US/Eastern" date +%Y_%m_%d)
filename="/home/ubuntu/benchmark_logs/2020/benchmark_log.$today.txt"

eval "$(conda shell.bash hook)"
conda activate Bodo

cd /home/ubuntu/
cd tpch
rm -rf tpch-dbgen
rm -rf /tmp/data1
yes | scripts/generateData.sh /tmp/data1 1.0
chmod -R u+r /tmp/data1

echo >> $filename
echo ------------------Running TPCH-H Benchmark------------------ >> $filename
for i in {1,3,4,5,6,9,10,12,14,18,19,20}
do
    echo ------------------Query $i, Num Proc: $cores------------------ >> $filename
    mpiexec -n $cores python main.py --folder=/tmp/data1 -p=bodo --query $i -l=0 >> $filename
done
echo ------------------TPCH-H Benchmark Done------------------ >> $filename
echo >> $filename

echo ------------------Running Custom Queries Benchmark------------------ >> $filename
echo ------------------Running Q-GroupBy, Num Proc: $cores------------------ >> $filename
mpiexec -n $cores python main.py -f q_groupby --folder=/tmp/data1 -p=bodo -l=0 >> $filename
echo ------------------Custom Queries Benchmark Done------------------ >> $filename
echo >> $filename
