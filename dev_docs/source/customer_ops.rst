.. _customer_ops:

Generating Binaries for Customers and the Platform
--------------------------------------------------
We use the `Azure build obfuscated pipeline <https://dev.azure.com/bodo-inc/Bodo/_build?definitionId=5&_a=summary>`_ to generate Bodo binaries. By default, the pipeline triggers for releases and publishes the binaries to the ``bodo.ai`` conda channel on artifactory.

Here are the expected behaviors for manual triggers:

- If you trigger the pipeline on a tag/release commit with license check enabled, the pipeline will generate a binary and push it to the official ``bodo.ai`` conda channel on Artifactory. It will also automatically trigger a build with no license check and push it to the ``bodo.ai-platform`` channel and trigger an AMI update.

- If you trigger the pipeline on a regular commit with no tags with license check enabled, the pipeline will generate a binary and push it to the dev ``bodo.ai-dev`` conda channel on Artifactory. Make note of the version of the binary and follow the instructions below to share this particular dev release with a customer.

- If you trigger the pipeline with no license check, the pipeline will generate a binary and push it to the internal ``bodo-binary`` conda channel on Artifactory.

Giving Customers Access to Binaries
---------------------------------
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
<tbd>
