# Deploying Bodo with Kubernetes {#kubernetes}

 This section demonstrates an example showing how to deploy a Bodo application with Kubernetes.
 We deploy Bodo with the [Kubeflow MPI-Operator](https://github.com/kubeflow/mpi-operator){target="blank"}. 


## Setting Up

You need the following to deploy your Bodo application using Kubernetes:

- **Access to a Kubernetes cluster.**

    For this example, we'll use kops on AWS. See the section below on [creating a Kubernetes cluster][kops] to see how we set it up.

- **A Docker image containing the Bodo application scripts and their intended Bodo version made available on a Docker registry, so that Kubernetes can pull it.**

    For this example, we created a [Docker image](https://hub.docker.com/r/bodoaidocker/bodo-kubernetes/tags){target="blank"} using [this Dockerfile](https://github.com/bodo-ai/Bodo-examples/blob/08c2b8991b5626473a0a7541411ac9d268191892/08-Kubernetes/Dockerfile){target="blank"} and uploaded it to Docker Hub. It includes a Bodo application called `pi.py` that calculates the value of pi using the Monte Carlo method, and can be used to validate your setup.

    You can use this as a base image for your own Docker image. If you want to use a private registry, you can follow the instructions [here](https://kubernetes.io/docs/tasks/configure-pod-container/pull-image-private-registry/){target="blank"}.


!!! warning
    Make sure to provide the correct CPU and Memory requests in the YAML file for your Bodo jobs. If correct
    values are not provided or the cluster doesn't have sufficient CPU or Memory required for the job, the job will be
    terminated and worker pods may keep respawning. You can get a good estimate of the CPU and Memory requirements
    by extrapolation from running the job locally on a smaller dataset.


### Creating a Kubernetes Cluster using KOPS {#kops}

Here are the steps create an AWS EKS cluster using KOPS.

-   Install KOPS on your local machine:

    ```shell
    # Mac
    brew install kops

    # Linux
    curl -LO https://github.com/kubernetes/kops/releases/download/$(curl -s https://api.github.com/repos/kubernetes/kops/releases/latest | grep tag_name | cut -d '"' -f 4)/kops-linux-amd64
    chmod +x kops-linux-amd64
    sudo mv kops-linux-amd64 /usr/local/bin/kops
    ```

-   Create a location to store your cluster configuration:

    First you need to create an S3 bucket to use as your `KOPS_STATE_STORE`.

    ```shell
    export KOPS_CLUSTER_NAME=imesh.k8s.local
    export KOPS_STATE_STORE=s3://<your S3 bucket name>
    ```

-  Create your cluster:

    The following code block creates a cluster of 2 nodes each with 4 cores .
    You can modify the `node-count` argument to change the number of instances.
    To change the number of worker nodes, update `node-size`. You can deploy the cluster
    in a different AWS region and availability zone by modifying the `zones` argument.

    ```
    kops create cluster \
    --node-count=2 \
    --node-size=c5.2xlarge \
    --control-plane-size=c5.large \
    --zones=us-east-2c \
    --name=${KOPS_CLUSTER_NAME}
    ```

    !!! tip

        The parameter `control-plane-size`
        refers to the leader that manages K8s but doesn’t do any Bodo computation,
        so you should keep the instance size small.

-   Finish creating the cluster with the following command.

    ```
    kops update cluster --name $KOPS_CLUSTER_NAME --yes --admin
    ```
    !!! note
        This might take several minutes to finish.

-   Verify that the cluster setup is finished by running:

    ```
    kops validate cluster
    ```


[//]: # (## Deploying Bodo on a Kubernetes Cluster Using Helm)

[//]: # ()
[//]: # (You can deploy Bodo applications on a Kubernetes cluster using Helm by following the steps outlined below.)

[//]: # ()
[//]: # (1.   **Clone the Bodo examples git repository.**)

[//]: # ()
[//]: # (    ```shell)

[//]: # (    git clone https://github.com/bodo-ai/Bodo-examples.git)

[//]: # (    ```)

[//]: # ()
[//]: # (1.   **Set up Helm and run your application.**)

[//]: # ()
[//]: # (    First, check if Helm is already installed in your system using)

[//]: # (    ```shell)

[//]: # (    helm version)

[//]: # (    ```)

[//]: # (    If it is not installed, then you can use [this guide]&#40;https://helm.sh/docs/intro/install/&#41;{target="blank"} to set up Helm in your system.)

[//]: # (    To run the example Bodo application, navigate to the repository and run the following command)

[//]: # (    ```shell)

[//]: # (    helm install <release-name> <helm-directory-path>)

[//]: # (    ```)

[//]: # (    For instance, you can run the following for the `chicago_crimes` example.)

[//]: # (    ```shell)

[//]: # (    helm install bodo-job-chicago-crime Kubernetes/helm)

[//]: # (    ```)

[//]: # (    This command will install the MPI-Job CRD and deploy a MPIJob which runs the Chicago Crimes Example.)

[//]: # ()
[//]: # ()
[//]: # (    To run the chart with your job specification, navigate to the Kubernetes/helm folder, edit the `values.yaml` file, and run the above command.)

[//]: # ()
[//]: # (1.   **Retrieve the results.**)

[//]: # ()
[//]: # (    When the job finishes running, your launcher pod will change its status to `completed`, aand any information published to stdout)

[//]: # (    can be found in the logs of the launcher pod:)

[//]: # ()
[//]: # (    ```shell)

[//]: # (    PODNAME=$&#40;kubectl get pods -o=name&#41;)

[//]: # (    kubectl logs -f ${PODNAME})

[//]: # (    ```)

[//]: # ()
[//]: # (### Teardown)

[//]: # ()
[//]: # (-   If you want to remove the job, run)

[//]: # (    ```shell)

[//]: # (    helm uninstall <release-name>)

[//]: # (    ```)

[//]: # (-   If you want to delete the MPI-Operator CRD, run the command)

[//]: # (    ```shell)

[//]: # (    kubectl delete -f Kubernetes/helm/crds/mpi-operator.yaml)

[//]: # (    ```)


## Deploying Bodo on a Kubernetes Cluster Manually

### Install MPIJob Custom Resource Definitions(CRD)

The most up-to-date installation guide is available at [MPI-Operator Github](https://github.com/kubeflow/mpi-operator). This example was tested using [v0.4.0](https://github.com/kubeflow/mpi-operator/tree/v0.4.0), as shown below:

```shell
git clone https://github.com/kubeflow/mpi-operator --branch v0.4.0
cd mpi-operator
kubectl apply -f deploy/v2beta1/mpi-operator.yaml
```

You can check whether the MPI Job custom resource is installed via:

```shell
kubectl get crd
```

The output should include `mpijobs.kubeflow.org` similar to:

```console
NAME                   CREATED AT
mpijobs.kubeflow.org    2024-04-02T19:43:04Z
```

### Run your Bodo application

1. Define a kubernetes resource for your Bodo workload, such as the one defined in [`mpijob.yaml`](https://github.com/bodo-ai/Bodo-examples/blob/08c2b8991b5626473a0a7541411ac9d268191892/08-Kubernetes/mpijob.yaml){target="blank"}
that runs the [pi](https://github.com/bodo-ai/Bodo-examples/blob/08c2b8991b5626473a0a7541411ac9d268191892/08-Kubernetes/pi.py){target="blank"} example. You can modify it based on your cluster configuration:
   1. Update `spec.slotsPerWorker` with the number of physical cores (_not_ vCPUs) on each node
   2. Set `spec.mpiReplicaSpecs.Worker.replicas` to the number of worker nodes in your cluster.
   3. Build the image using the Dockerfile or use `bodoaidocker/bodo-kubernetes` and replace the image at
   `spec.mpiReplicaSpecs.Launcher.template.spec.containers.image` and `spec.mpiReplicaSpecs.Worker.template.spec.containers.image`.
   4. Check the container arguments is referring to the python file you have intended to run
        ```shell
         args:
            - mpirun
            - -n
            - "8"
            - python
            - /home/mpiuser/pi.py
        ```
3. Lastly, make sure `-n` is equal to `spec.mpiReplicaSpecs.Worker.replicas` multiplied by `spec.slotsPerWorker`, i.e. the total number of physical cores on your worker nodes.
4. Run the example by deploying it in your cluster with `kubectl create -f mpijob.yaml`. This should add 1 pod to each worker and a launcher pod to your master node.
5. View the generated pods by this deployment with `kubectl get pods`. You may inspect any logs by looking at the individual pod's logs.

### Retrieve the Results

    When the job finishes running, your launcher pod will change its status to completed and any stdout information can be found in the logs of the launcher pod:

    ```shell
    PODNAME=$(kubectl get pods -o=name)
    kubectl logs -f ${PODNAME}
    ```

## Teardown

- When a job has finished running, you can remove it by running `kubectl delete -f mpijob.yaml`.
- If you want to delete the MPI-Operator crd, please follow the steps on the [MPI-Operator Github repository](https://github.com/kubeflow/mpi-operator){target="blank"}.


