# AWS PrivateLink {#aws_private_link}
:fontawesome-brands-aws: Supported on AWS ·

AWS PrivateLink is a service provided by Amazon Web Services (AWS) that enables secure,
private connectivity between your virtual private cloud (VPC) and on-premises networks to AWS services.
It ensures that traffic between these resources never traverses the public internet, enhancing security, reducing
exposure to internet-based threats, and providing low-latency connectivity.

### Bodo Cluster AWS PrivateLink
AWS PrivateLink for Bodo Platform enables private connectivity between Bodo Platform and client clusters.
With this feature, you do not need to use an internet gateway or NAT to allow communication with the Bodo Platform from Bodo clusters in your VPC subnets.


### Bodo Cluster Endpoint Services

List of supported endpoint services:

| Region    | Service Name                                            | Supported AZs |
|-----------|---------------------------------------------------------|---------------|
| us-east-1 | com.amazonaws.vpce.us-east-1.vpce-svc-0b0c6643c1f764b62 | az1, az4, az6 | 
| us-east-2 | com.amazonaws.vpce.us-east-2.vpce-svc-0c49909796ff87d5b | az1, az2, az3 | 
| us-west-1 | com.amazonaws.vpce.us-west-1.vpce-svc-0ce34162b3b8f1eaa | az1, az3      |
| us-west-2 | com.amazonaws.vpce.us-west-2.vpce-svc-026e1758e07ba65d5 | az1, az2, az3 | 
| eu-west-1 | com.amazonaws.vpce.eu-west-1.vpce-svc-05352b1056a782d85 | az1, az2, az3 |


### Configure Bodo Workspace with PrivateLink

This section explains how to configure [Customer Management VPC](customer_managed_vpc.md) to use AWS Private link, 
so the connection between the Bodo Platform and Bodo clusters will be made in the AWS internal network.

#### Configuration

1. Create an interface endpoint that points to the specific Bodo Cluster Endpoint Service, depending on the region:
![Bodo-Cluster-Interface-Endpoint](../../platform2-gifs/bodo-cluster-interface-endpoint.gif#center)

2. Once the endpoint is available, modify the private DNS name:
![Bodo-Cluster-Interface-Endpoint-DNS](../../platform2-gifs/bodo-cluster-interface-endpoint-dns.gif#center)

3. Create an S3 gateway endpoint if it does not already exist in the VPC (required for access workspace S3 storage):
![AWS-S3-Gateway](../../platform2-gifs/s3-gateway.gif#center)

4. Create an SSM interface endpoint if it does not already exist in the VPC (required for Bodo clusters to read workspace SSM parameters):
![AWS-SSM-Interface](../../platform2-gifs/ssm-interface-endpoint.gif#center)


!!! info "Important"
        For interface endpoints, you don't need to select all the subnets used by the workers; you just need to select at least one.
        For the S3 gateway, you need to select all route tables associated with subnets used by Bodo clusters.
