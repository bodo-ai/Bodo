# Running this benchmark


``` shell
cd benchmarks/modin-ray
# create ray cluster
ray up modin-cluster.yaml -y
# (optional) Forward dashboard to local
ray dashboard modin-cluster.yaml
# (in a separate terminal) (optional) ssh into head node
ray attach modin-cluster.yaml
# (in a separate terminal) run benchmark
ray submit modin-cluster.yaml nyc_taxi_preciptation.py
# run benchmark again to make sure all nodes have had a chance to start up
ray submit modin-cluster.yaml nyc_taxi_preciptation.py
# tear down cluster
ray down modin-cluster.yaml -y
# end ssh session
exit
```