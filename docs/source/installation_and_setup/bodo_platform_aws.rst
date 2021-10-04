.. _bodo_platform_aws:

Bodo Managed Cloud Platform on AWS
===================================


- :ref:`registration`
- :ref:`setting_aws_credentials`
- :ref:`creating_clusters`
- :ref:`attaching_notebook_to_cluster`
- :ref:`connecting_to_a_cluster`
- :ref:`running_a_job`
- :ref:`resources_created_in_aws_env`
- :ref:`aws_account_cleanup`
- :ref:`aws_billing`

.. _registration:

Registration
------------

a. Subscribe through the `AWS Marketplace <https://aws.amazon.com/marketplace/pp/B08NY29SMQ>`_.
#. After confirming your subscription, you'll be directed to Bodo Platform's registration page.
#. Fill out the fields with your information. If this is your individual account,
   use a unique name such as `firstname_lastname` for the **Organization Name** field.
#. Check the box for accepting terms and conditions and click on ``SIGN UP``:

    .. image:: ../platform_onboarding_screenshots/signup.png
        :align: center
        :alt: Signup-Page

#. A page confirming that an activation link was sent to your email will appear.
   Please open the email and click on the activation link:

    .. image:: ../platform_onboarding_screenshots/signup-conf.png
        :align: center
        :alt: Signup-Page-Confirmation

   Clicking on the confirmation link will take you to the bodo platform page
   where you can use your newly created credentials to sign in:

        .. image:: ../platform_onboarding_screenshots/login.png
            :align: center
            :alt: Login-Page

.. _setting_aws_credentials:

Setting AWS Credentials
-----------------------

To use Bodo on AWS, you need to link your AWS account to the Bodo platform. This can be done either using the *Settings* page
in the left bar or the first item in the *Onboarding* list highlighted in green as shown in the picture below:

.. image:: ../platform_onboarding_screenshots/dashboard.png
    :align: center
    :alt: Dashboard

To be able to use the Bodo Platform to launch clusters and notebooks, you must grant it permission to access your AWS account and provision the
required resources in it. This can be done through an `AWS Cross Account IAM Role <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-roles-for-amazon-ec2.html>`_ for the Bodo Platform.

.. _create_iam_role:

Create a Cross-Account IAM Role
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two ways to create such an IAM Role, (a) you can create it manually, or (b) you can provide us with `Access Keys <https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html>`_
and we can create an IAM role in your AWS account. We provide directions for both these methods below.

.. _create_iam_role_manually:

Create the IAM Role Manually
****************************

#. Log in to the `AWS Management Console <https://aws.amazon.com/console/>`_  and navigate to the **IAM** Service.
#. Select the *Roles* tab in the sidebar, and click ``Create Role``.
#. In **Select type of trusted entity**, select ``Another AWS Account``.
#. Enter the Bodo Platform Account ID `481633624848` in the **Account ID** field.
#. Check the ``Require external ID`` option.

    .. image:: ../create_role_screenshots/create_iam_role_manual_form_step1.png
        :align: center
        :alt: Create Role Form Step 1

    In the **External ID** field, copy over the External ID from the *Settings* page on the Bodo Platform.    

    .. image:: ../create_role_screenshots/create_iam_role_manual_externalId.png
        :align: center
        :alt: External ID Platform

#. Click the ``Next: Permissions`` button.
#. Click the ``Next: Tags`` button.
#. Click the ``Next: Review`` button.
#. In the **Role name** field, enter a role name, e.g. `BodoPlatformUser`.

    .. image:: ../create_role_screenshots/create_iam_role_manual_review_step.png
        :align: center
        :alt: Create Role Form Review

#. Click ``Create Role``. You will be taken back to the list of IAM Roles in your account.
#. In the list of IAM Roles, click on the role you just created.
#. Click on ``Add inline policy``.

    .. image:: ../create_role_screenshots/create_role_manual_summary_page.png
        :align: center
        :alt: Create Role Summary Page

#. Click the ``JSON`` tab.

    .. image:: ../create_role_screenshots/create_iam_role_manual_policy_editor.png
        :align: center
        :alt: Create Role Manual Policy Editor

#. Bodo Cloud Platform requires a specific set of AWS permissions which are documented in :download:`Bodo-Platform Policy <downloadables/bodo-platform.json>`. 
   Paste the contents of the linked JSON file into the policy editor. 
#. Click on ``Review policy``.
#. In the *Name* field, add a policy name, e.g. `Bodo-Platform-User-Policy`. Click on ``Create policy``. You will be taken back to the Role Summary.
#. From the role summary, copy the ``Role ARN``. This is the value that you will enter into the **Role ARN** field on the *Setting* Page on the Bodo Platform.

    .. image:: ../create_role_screenshots/create_iam_role_manual_final_summary.png
        :align: center
        :alt: Create Role Final Summary

.. _create_iam_role_using_platform:

Let the Bodo Platform create the IAM Role
*****************************************

#. Follow the instructions from `AWS Account and Access Keys guide <https://docs.aws.amazon.com/powershell/latest/userguide/pstools-appendix-sign-up.html>`_
   to create/retrieve your AWS access key ID and secret access key.
#. Click on ``Create Role For Me`` below the **Role ARN** field on the Setting page. This will open up a panel.

    .. image:: ../create_role_screenshots/create_role_via_platform_create_role_button.png
        :align: center
        :alt: Create Role Button on Platform

#. Enter the Access Keys created in step 1 in the form and click on ``CREATE ROLE``.

    .. image:: ../create_role_screenshots/create_role_via_platform_enter_access_keys.png
        :align: center
        :alt: Enter Access Keys to create role on Platform
    
   **NOTE**: We will **not** save the provided Access Keys for security reasons.

#. Click ``OK`` on the popup confirmation box.
#. We will use the provided Access Keys to create an IAM Role in your AWS Account. 
#. The created Role ARN will be displayed on the same form.

    .. image:: ../create_role_screenshots/create_role_via_platform_generated_role.png
        :align: center
        :alt: Role ARN generated on the Platform

#. Copy the generated Role ARN. This is the value that you will enter into the **Role ARN** field on the *Setting* Page on the Bodo Platform.
#. In some cases, this role creation might fail. This could happen due to various reasons.
   
   a. A role already exists: In this case, please open the `AWS Management Console <https://aws.amazon.com/console/>`_, and navigate to the *IAM* Service. 
      Click on *Roles* in the sidebar. Look for a Role named `BodoPlatformUser`. Click on the role, and copy over the
      **Role ARN** from the role summary. Alternatively, you can delete the existing role from the AWS Console and then try to create
      an IAM role again via the Bodo Platform. This will ensure you have the role set up with the correct permissions.
      
      **Note**: If this is a shared AWS Account, ensure that no one else is actively using this IAM Role before deleting it.
   #. Provided access keys are not valid: Please ensure that valid access keys are provided.
   #. Provided access keys don't have the right permissions to create a role: Please ensure that the provided access keys have the permissions required
      to create an IAM Role.
   
   If none of these work, try creating the IAM Role manually as described in :ref:`create_iam_role_manually`.


Once you have generated an IAM Role using either of the methods described above, you are now ready to fill the Settings Form on the Bodo Platform.

#. Enter the Role ARN created using one of the above options into the **Role ARN** field in the Settings Form.

#. Select a **Region** from the dropdown list. This is the region that your resources will be deployed in by default.

#. Click on ``SAVE``.

You can see the progress on granting `AMI <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AMIs.html>`_
launch permissions to your account ID in the **AMI Share Status** field. Your account is ready when it turns green.

**Note:** We grant AMI launch permissions to your account in the following AWS regions: us-east-1, us-east-2, us-west-1 & us-west-2.

**Note:** It is highly recommended that you ensure sufficient limits on your AWS account to launch
resources. See :ref:`resources_created_in_aws_env` for the resources required for Bodo Cloud Platform.


.. seealso:: :ref:`bodo_platform`


.. _resources_created_in_aws_env:

Resources Created in Your AWS Environment
-----------------------------------------

Bodo deploys cluster/notebook resources in your own AWS environment to ensure
security of your data.
Below is a list of AWS resources
that the Bodo Platform creates in your account to enable clusters and notebooks.


.. list-table::
  :header-rows: 1

  * - AWS Service
    - Purpose
  * - `EC2 Instances <https://aws.amazon.com/ec2/>`_
    - Cluster/notebook workers
  * - `EFS <https://aws.amazon.com/efs/>`_
    - Shared file system for clusters
  * - `VPC <https://aws.amazon.com/vpc/>`_, `Subnets <https://docs.aws.amazon.com/vpc/latest/userguide/VPC_Subnets.html>`_,
      `NAT Gateway <https://docs.aws.amazon.com/vpc/latest/userguide/vpc-nat-gateway.html>`_,
      `Elastic IP <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/elastic-ip-addresses-eip.html>`_,
      `ENI <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-eni.html>`_,
      `Security Groups <https://docs.aws.amazon.com/vpc/latest/userguide/VPC_SecurityGroups.html>`_, ...
    - Secure networking for clusters/notebooks
  * - `S3 <https://aws.amazon.com/s3/>`_ and `Dynamo DB <https://aws.amazon.com/dynamodb/>`_
    - Resource states
  * - `AWS Systems Manager <https://aws.amazon.com/systems-manager/>`_
    - Managing EC2 instances
  * - `KMS <https://aws.amazon.com/kms/>`_
    - Cluster secrets (e.g. SSH keys)
  * - `IAM Role <https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html>`_ for Clusters
    - Allow cluster workers to access resources above

.. note::

    These resources incur additional AWS infrastructure charges and are not included in the Bodo Platform charges.

.. _aws_account_cleanup:

AWS Account Cleanup
-------------------

As explained in :ref:`resources_created_in_aws_env`, the platform creates two types of resources in the users' AWS environments: 
organization level resources and cluster specific resources. The organization level resources are created by the platform to set 
up shared resources (such as a VPC, an EFS Mount, etc) that are used later by all created resources. The cluster specific resources 
(such as EC2 instances, ENIs, etc) are created by the platform to host/manage a specific cluster. This includes notebooks and 
corresponding resources as well.
The cluster specific resources are removed when you request a cluster to be removed.
The organization level resources persist in the user account so they can be used by clusters deployed in the future.
However, if you need to remove these resources for any reason (AWS limits, etc.), an option to do so is provided.
Navigate to the *Settings* page and click on ``Show Advanced`` in the bottom-right corner. 

.. image:: ../platform_onboarding_screenshots/settings-ac-cleanup.png
    :align: center
    :alt: Settings-Account-Cleanup


This will bring up a section called *AWS Resource Cleanup*. 

.. image:: ../platform_onboarding_screenshots/settings-adv-ac-cleanup.png
    :align: center
    :alt: Advanced-Settings-Account-Cleanup


Select the region from which you would like to remove these resources
(i.e. the region in which the resources you want to delete have been created), and click ``CLEANUP AWS RESOURCES``.
Note that this will only work if you don't have any active clusters in that region deployed through the platform.
Else, the request will be rejected, and you'll be asked to remove all clusters in that region before trying again.
Removing active clusters (including clusters with a *FAILED* status) is necessary because 
this process will make them inaccessible to the platform.

.. seealso:: :ref:`troubleshootingaws`

.. _aws_billing:

Billing
-------

Users subscribed to the Bodo Platform through the AWS Marketplace will be charged for their use of the platform as part
of their regular AWS bill. The platform charges are based on the type of instances deployed and the duration of their usage (to the nearest minute).
The hourly rate for the supported instance types can be found on our `website <https://bodo.ai/pricing-aws>`_.
For any cluster deployed through the platform, users are charged starting from when the cluster has been successfully deployed, 
until the time the user requests the cluster to be removed. 

**Note:** Users are not charged in case of failures in cluster creation.

As mentioned previously in :ref:`resources_created_in_aws_env`, the AWS resources set up by the platform in your AWS environment
incur additional AWS infrastructure charges, and are not included in the Bodo Platform charges.