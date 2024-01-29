# Details about the VPC Structure
output "vpc" {
  value = module.vpc
}

# URL Endpoint for Lambda to Use in Github App's Webhook
output "webhook_endpoint" {
  value = module.runners.webhook.endpoint
}

# Secret Key for Lambda to Use in Github App's Webhook
output "webhook_secret" {
  sensitive = true
  value     = random_password.random.result
}
