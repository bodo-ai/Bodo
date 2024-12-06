# Running this benchmakr

Install Modin/ray

``` shell
cd benchmarks/modin-ray
# create ray cluster
ray up modin-cluster.yaml -y
# Forward dashboard to local
ray dashboard modin-cluster.yaml
# ssh into head node
ray attach modin-cluster.yaml
# end ssh session
exit
# run benchmark
ray submit modin-cluster.yaml nyc_taxi_preciptation.py
# tear down cluster
ray down modin-cluster.yaml -y
```