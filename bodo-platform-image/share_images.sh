#!/bin/bash

set -eo pipefail

if [ -z "${AUTH_SERVICE_URL:+1}" ]; then
  echo "AUTH_SERVICE_URL is not set"
  exit 1
fi


if [ -z "${BOT_PLATFORM_USERNAME:+1}" ]; then
  echo "BOT_PLATFORM_USERNAME is not set"
  exit 1
fi

if [ -z "${BOT_PLATFORM_PASSWORD:+1}" ]; then
  echo "BOT_PLATFORM_PASSWORD is not set"
  exit 1
fi

if [ -z "${BACKEND_SERVICE_URL:+1}" ]; then
  echo "BACKEND_SERVICE_URL is not set"
  exit 1
fi

# Login using the bot and get access_token
echo "Log into the platform using herman_bot account and get an access token..."

auth_response=$(python3 scripts/common.py $AUTH_SERVICE_URL $BOT_PLATFORM_USERNAME $BOT_PLATFORM_PASSWORD)
# Check that the response was not an error code
if [ 0 -ne $? ]; then 
    echo "Login Failed."
    echo "Received response: $auth_response"
    exit 1
fi;

# Send the image information to backend
echo "Triggering bodoImage endpoint on backend with the access token and image entries..."
azure_backend_response=$(curl -f --location --request POST "$BACKEND_SERVICE_URL/api/image/bodoImage" \
--header 'Content-Type: application/json' \
--header "Authorization: Bearer $auth_response" \
-d @img_share_requests.json)
# Check that the response was not an error code
if [ 0 -ne $? ]; then 
    echo "bodoImage call to the backend service failed"
    echo "Response received from backend: $backend_response"
    exit 1
fi;