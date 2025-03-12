"""Request more cpus from the autoscaler and wait until they are available
for compute tasks.

usage:
    ray submit modin-cluster.yaml scale_cluster.py NUM_VCPU
"""

import sys
import time

import ray
from ray.autoscaler.sdk import request_resources

if __name__ == "__main__":
    num_vcpu = int(sys.argv[1])

    ray.init(address="auto")
    cpu_count = ray.cluster_resources()["CPU"]
    print("RAY CPU COUNT: ", cpu_count)

    request_resources(num_cpus=num_vcpu)

    while cpu_count < num_vcpu:
        cpu_count = ray.cluster_resources()["CPU"]
        time.sleep(1)
