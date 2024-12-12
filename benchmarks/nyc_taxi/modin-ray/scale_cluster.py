"""Request more cpus from the autoscaler and wait until they are available
for compute tasks.

usage:
    ray submit modin-cluster.yaml scale_cluster.py
"""

import time

import ray
from ray.autoscaler.sdk import request_resources

ray.init(address="auto")
cpu_count = ray.cluster_resources()["CPU"]
print("RAY CPU COUNT: ", cpu_count)

request_resources(num_cpus=256)

while cpu_count < 256:
    cpu_count = ray.cluster_resources()["CPU"]
    time.sleep(1)
