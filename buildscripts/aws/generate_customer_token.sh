#!/bin/bash
set -exo pipefail

pip install credstash
ADMIN_USERNAME=`credstash -r us-east-2 get artifactory.admin.username`
ADMIN_TOKEN=`credstash -r us-east-2 get artifactory.admin.token`

curl -u"$ADMIN_USERNAME":"$ADMIN_TOKEN" -XPOST "https://bodo.jfrog.io/artifactory/api/security/token" -d "username=$conda_username" -d "scope=member-of-groups:Customers" -d "expires_in=0"
