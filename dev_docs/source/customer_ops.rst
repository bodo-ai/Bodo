.. _customer_ops:

Generating Binaries for Customers and the Platform
--------------------------------------------------
We use the `Azure build obfuscated pipeline <https://dev.azure.com/bodo-inc/Bodo/_build?definitionId=5&_a=summary>`_ to generate Bodo binaries. By default, the pipeline triggers for releases and publishes the binaries to the ``bodo.ai`` conda channel on artifactory.

Here are the expected behaviors for manual triggers:

- If you trigger the pipeline on a tag/release commit with license check enabled, the pipeline will generate a binary and push it to the official ``bodo.ai`` conda channel on Artifactory. It will also automatically trigger a build with no license check and push it to the ``bodo.ai-platform`` channel and trigger an AMI update.

- If you trigger the pipeline on a regular commit with no tags with license check enabled, the pipeline will generate a binary and push it to the dev ``bodo.ai-dev`` conda channel on Artifactory. Make note of the version of the binary and follow the instructions below to share this particular dev release with a customer.

- If you trigger the pipeline with no license check, the pipeline will generate a binary and push it to the internal ``bodo-binary`` conda channel on Artifactory.

Giving Customers Access to Binaries
-----------------------------------
By default, we give customers access to our ``bodo.ai`` and ``bodo.ai-dev`` conda channels on Artifactory which hold the official and engineering releases respectively.

- To generate a customer token, trigger the `CodeBuild pipeline <https://us-east-2.console.aws.amazon.com/codesuite/codebuild/427443013497/projects/generate-customer-token>`_ passing in the environment variable ``conda_username`` the customer username for which you would like to generate a token.

  Once the pipeline completes, you can copy to token from the end of the log.


- To install Bodo from the releases channel, execute the following command::

     export BODO_TOKEN=<token>
     conda install -c https://<username>:"$BODO_TOKEN"@bodo.jfrog.io/artifactory/api/conda/bodo.ai -c conda-forge bodo


- To install Bodo from the dev releases channel, execute the following command::


     export BODO_TOKEN=<token>
     conda install -c https://<username>:<token>@bodo.jfrog.io/artifactory/api/conda/bodo.ai-dev -c conda-forge bodo

where ``<username>`` and ``<token>`` are the customer username and tokens generated in the step above.

The above command installs the latest dev release. Usually, to point the customer to the specific dev release you created for them, specify the bodo version. The command would be in this case::

     export BODO_TOKEN=<token>
     conda install -c https://<username>:<token>@bodo.jfrog.io/artifactory/api/conda/bodo.ai-dev -c conda-forge bodo=<DEV_VERSION>

where ``<version>`` is the full version name of the built created.

Generating Customer License Keys
--------------------------------
To generate a customer license, you can trigger the
`Azure License-Generation pipeline <https://dev.azure.com/bodo-inc/Bodo/_build?definitionId=9>`_.

When you run the pipeline you need to specify the number cores and when the license terminates
in the `Variables` section under `Advanced Options`. The number of cores is specified by giving
a value for ``MAX_CORES`` and for license termination you can either specify the number of days
for which the license is valid in ``TRIAL_DAYS`` or the day upon which the license expires via
``EXPIRATION_DATE``.

You must provide at least one of ``TRIAL_DAYS`` and ``EXPIRATION_DATE`` and
if you provide both the license will by generated for the ``EXPIRATION_DATE``.


Generating AWS Platform Private Offers for Customers
----------------------------------------------------

To generate a private offer for a customer, you will need their AWS Account ID. Once they have provided this to you, follow the steps outlined below.

Log onto the `AWS Marketplace <https://aws.amazon.com/marketplace>`_ using the Platform-Seller Account (929854676641) and navigate to the *Marketplace Management Portal*.
If you don't have access, ask the Platform Team to create an IAM user for you with the appropriate permissions.

Once logged in, click on **Offers** as shown below.

     .. figure:: ../figs/aws-private-offer-console.png
        :alt: AWS Marketplace Console

Then, click on **Create an Offer**.

     .. figure:: ../figs/aws-private-offer-create.png
        :alt: AWS Offers Management Create Offer

You will be presented with a form as shown below. Select the Bodo Cloud Platform for the **Product** field, and enter the AWS Account ID
of the customer in the **Buyer account id(s)** field. Click on **Next**.

     .. figure:: ../figs/aws-private-offer-account-id.png
        :alt: AWS Offer Enter Buyer Account ID

On the next page, select an appropriate rate for the **bodo_unit** pricing dimension. The regular rate is 1 bodo_unit = $0.001.
So, for instance, set it to 0.0 for a free-trial, 0.0005 for a 50% discount, etc.
Check *I want to enable zero dollar prices* if providing a free-trial.

     .. figure:: ../figs/aws-private-offer-set-rate.png
        :alt: AWS Offer Set Rate

Select *Public Offer EULA* in the **EULA Version** field. This is our custom EULA.
Set appropriate values in the **Offer expiry date** and **Subscription end date** fields.
Once all the fields are filled, click on **Review Offer**.

     .. figure:: ../figs/aws-private-offer-set-eula-dates.png
        :alt: AWS Offer Set EULA and Dates

On the **Review Offer** page, verify that the details look correct, and if they do, click on **Extend Offer**.

     .. figure:: ../figs/aws-private-offer-review-offer.png
        :alt: AWS Offer Review

You will see a confirmation message like the one below. Click on **Return to Manage Offers**.

     .. figure:: ../figs/aws-private-offer-confirmation.png
        :alt: AWS Offer Confirmation

It can take up to 10 minutes for the offer to get generated. Once it is generated, it'll show up on the list of offers (identifiable by the Buyer's AWS Account ID).
Select the offer, click on **Copy Offer url** and share the URL with the customer so they can accept the offer.

     .. figure:: ../figs/aws-private-offer-copy-url.png
        :alt: AWS Offer Copy URL
