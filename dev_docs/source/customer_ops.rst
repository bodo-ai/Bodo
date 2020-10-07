.. _customer_ops:

Give Customers Access to Binaries
---------------------------------
By default, we give customers access to our ``bodo.ai`` and ``bodo.ai-dev`` conda channels on Artifactory which hold the official and engineering releases respectively.

- To generate a customer token, trigger the `codebuild pipeline <https://us-east-2.console.aws.amazon.com/codesuite/codebuild/427443013497/projects/generate-customer-token>`_ passing in the environment variable ``conda_username`` the customer username for which you would like to generate a token.

  Once the pipeline completes, you can copy to token from the end of the log.


- To install Bodo from the releases channel, execute the following command::

     export BODO_TOKEN=<token>
     conda install -c https://<username>:"$BODO_TOKEN"@bodo.jfrog.io/artifactory/api/conda/bodo.ai -c conda-forge bodo


- To install Bodo from the dev releases channel, execute the following command::


     export BODO_TOKEN=<token>
     conda install -c https://<username>:<token>@bodo.jfrog.io/artifactory/api/conda/bodo.ai-dev -c conda-forge bodo

where ``<username>`` and ``<token>`` are the customer username and tokens generated in the step above.

The above command installs the latest dev release. Usually, we give customers the link to the subchannel that contains the specific dev release created for them. The command would be in this case::

     export BODO_TOKEN=<token>
     conda install -c https://<username>:<token>@bodo.jfrog.io/artifactory/api/conda/bodo.ai-dev/bodo-<version>.tar.bz2 -c conda-forge bodo

where ``<version>`` is the full version name of the built created.

Generating Customer License Keys
--------------------------------
<tbd>
