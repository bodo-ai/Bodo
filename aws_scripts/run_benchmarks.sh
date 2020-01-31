#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
cores=16
today=$(TZ=":US/Eastern" date +%Y_%m_%d)
filename="/home/ubuntu/benchmark_logs/2020/benchmark_log.$today.txt"

cd /home/ubuntu/
eval "$(conda shell.bash hook)"
conda activate Bodo
cd benchmarks

echo ------------------Spec------------------ > $filename
echo "EC2 Instance Type: m5.8xlarge" >> $filename
echo "# of Physical Cores: 16" >> $filename
echo "# of vCPUs: 32" >> $filename
echo >> $filename


echo ------------------Running Cal-pi Benchmark------------------ >> $filename
for i in {1,2}
do
    echo ------------------Cal-pi prob-size multiplier: $i, Num Proc: $cores------------------ >> $filename
    mpiexec -n $cores python calc-pi.py bodo $i $cores >> $filename
done

echo ------------------Cal-pi Benchmark Done------------------ >> $filename
echo >> $filename
