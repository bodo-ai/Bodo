# Instructions to Build

1) Create file `terraform.tfvars` in the following format:

```
github_app_id     = "12345"
github_key_base64 = "base64-encoded-private-key"
```

Fill in the actual values. See `variables.tf` for more details.

2) Install Terraform (instructions online)

3) Run the following in this folder:

```
terraform init
terraform apply
```

For more details, see: https://bodo.atlassian.net/wiki/spaces/B/pages/1558577174/Self-Hosted+GitHub+Actions
