.. _customer_ops:

Give Customers Access to Binaries
---------------------------------
By default, we give customers access to our `bodo.ai` and `bodo.ai-dev` conda channels on artifactory which hold the official and minor releases respectively.
Use the following command from your command line to generate a token for the customer username <customer_user>.
You will need your admin credentials to perform this operation.
`curl -u<admin_username>:<admin_password> -XPOST "https://bodo.jfrog.io/artifactory/api/security/token" -d "username=<customer_username>" -d "scope=member-of-groups:Customers"`

Generating Customer License Keys
--------------------------------
<tbd>
