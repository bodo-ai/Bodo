# Using your own Instance Role for a Cluster {#instance_role_cluster}

:fontawesome-brands-aws: Supported on AWS only·
In cases where you want to access additional AWS resources from Bodo clusters e.g., S3 buckets,
you can create an IAM Role in your AWS account and then register it as an Instance Role on the Bodo Platform, which will allow you to access those resources from Bodo clusters without using AWS keys.

![View-Instance-Roles](../../platform2-gifs/instance-role-list.gif#center)

Note that, by default, Bodo creates an IAM role with the necessary policies for each cluster. When you register your own role with the Bodo Platform, it will automatically attach the other required policies to this role.

Here, we walk through setting up an IAM Role in AWS and then registering it as an Instance Role on the Bodo Platform. For this example, we will be creating a role with access to an S3 bucket in your AWS account:

Step 1: Create an AWS IAM Role on the [AWS Management Console](https://aws.amazon.com/console/):

1. Go to the IAM service.

![AWS-IAM](../../platform2-screenshots/aws_iam.png#center)

2. In the left sidebar click on Roles.

![AWS-IAM-ROLE](../../platform2-screenshots/aws_iam_role.png#center)

3. Click on the button `Create role`, then select:
   - Trusted entity type: **AWS service**
   - Common use cases: **EC2**

![AWS-IAM-Role-Form](../../platform2-screenshots/aws_iam_role_form.png#center)

4. Click next, and then create a new policy that will be attached to this role:
   - json policy:

```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::<private-s3-bucket-name>"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject",
                "s3:DeleteObject",
                "s3:PutObjectAcl"
            ],
            "Resource": [
                "arn:aws:s3:::<private-s3-bucket-name>/*"
            ]
        }
    ]
}
```

5. Go back to Create role, refresh the list of policies, and add the policy that you created.
1. Click **Next**, then in the Role name field, type a role name and click **Create role**.
1. Copy the Role ARN from the role summary.

![AWS-IAM-Role-ARN](../../platform2-screenshots/aws_iam_role_arn.png#center)

Step 2: Register your AWS IAM Role on the Bodo Platform as a new Instance Role:

1. Click on the **CREATE INSTANCE ROLE** button and in the creation form, fill the following fields:
   - Name: Name for the Instance Role
   - Role ARN: AWS Role ARN from Step 1
   - Description: Short description for Instance Role

![Create-Instance-Role](../../platform2-gifs/instance-role-form.gif#center)

2. Click on the **Create** button.

The Instance Role will now be registered on the Platform. It can have one of two status-es:

- **Active**: Instance Role is ready to use
- **Failed**: Something went wrong while registering the Instance Role and it cannot be used. Some possible problems could be:
  - The Platform wasn't able to find the specified role.
  - The Platform was not able to attach additional Bodo polices that are required for normal cluster operations.
