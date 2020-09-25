.. _customer_ops:

Give Customers Access to Binaries
---------------------------------
By default, we give customers access to our ``bodo.ai`` and ``bodo.ai-dev`` conda channels on Artifactory which hold the official and engineering releases respectively.

- Use the following command from your command line to generate a token for the customer username <customer_user>::

     curl -u<admin_username>:<admin_password> -XPOST "https://bodo.jfrog.io/artifactory/api/security/token" -d "username=<customer_username>" -d "scope=member-of-groups:Customers" -d "expires_in=0"

You will need your admin credentials to perform this operation. Those are stored in the team's 1Password shared vault.

- To install Bodo, execute the following command::

     conda install -c https://<username>:<token>@bodo.jfrog.io/artifactory/api/conda/bodo.ai -c conda-forge bodo

where ``<username>`` and ``<token>`` are the customer username and tokens generated in the step above.

Generating Customer License Keys
--------------------------------
<tbd>
