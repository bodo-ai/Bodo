import errno
import os
import requests
from scripts.common import login

assert "AUTH_SERVICE_URL" in os.environ
assert "BOT_PLATFORM_USERNAME" in os.environ
assert "BOT_PLATFORM_PASSWORD" in os.environ
assert "BACKEND_SERVICE_URL" in os.environ
assert "BODO_VERSION" in os.environ
assert "BODO_CONDA_INSTALL_LINK" in os.environ
assert "ORGANIZATION_UUID" in os.environ

print("Log into the platform using herman_bot account and get an access token...")
try:
    access_token = login(os.environ["AUTH_SERVICE_URL"], os.environ["BOT_PLATFORM_USERNAME"], os.environ["BOT_PLATFORM_PASSWORD"])
except Exception as e:
    print("Login Failed")
    print("Received Response: " + e)
    exit(1)

print("Login successful...")

print("Triggering bodoVersion endpoint on backend with the access token...")
header = {'Authorization': 'Bearer ' + access_token}
bodoVersionData = {
    "bodoVersion": "%s" % os.environ["BODO_VERSION"],
    "condaInstallLink": "%s" % os.environ["BODO_CONDA_INSTALL_LINK"],
    "supported": "true"
}

bodo_version_url = os.path.join(os.environ["BACKEND_SERVICE_URL"], "api", "image", "bodoVersion")

try:
    response = requests.post(bodo_version_url, json=bodoVersionData, headers=header)
except requests.exceptions.RequestException as e:
    print("bodoVersion call to the backend service failed")
    print("Response received from backend: " + e)
    exit(1)

print("Successfully triggered backend service...")
print("Response received from backend: " + str(response))
