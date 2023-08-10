import os
import requests
import json

from scripts.common import login


PR_NUM_VERSIONS = 1
MASTER_NUM_VERSIONS = 3

assert "BACKEND_SERVICE_URL" in os.environ
assert "CI_EVENT" in os.environ
assert "AUTH_SERVICE_URL" in os.environ
assert "BOT_PLATFORM_USERNAME" in os.environ
assert "BOT_PLATFORM_PASSWORD" in os.environ

BACKEND_SERVICE_URL = os.environ["BACKEND_SERVICE_URL"]
CI_EVENT = os.environ["CI_EVENT"]
AUTH_SERVICE_URL = os.environ["AUTH_SERVICE_URL"]
BOT_PLATFORM_USERNAME = os.environ["BOT_PLATFORM_USERNAME"]
BOT_PLATFORM_PASSWORD = os.environ["BOT_PLATFORM_PASSWORD"]

# Login using bot_herman and get access token
access_token = login(AUTH_SERVICE_URL, BOT_PLATFORM_USERNAME, BOT_PLATFORM_PASSWORD)

# Trigger the backend to get supported bodo versions list
print("Triggering bodoVersion endpoint on backend with the access token...")

headers = {"Authorization": f"Bearer {access_token}"}
if CI_EVENT == 'release':
    get_bodo_versions_response = requests.get(
        f"{BACKEND_SERVICE_URL}/api/image/bodoVersion", headers=headers
    )
else:
    num_versions = PR_NUM_VERSIONS if CI_EVENT == 'pull_request' else MASTER_NUM_VERSIONS
    get_bodo_versions_response = requests.get(
        f"{BACKEND_SERVICE_URL}/api/image/bodoVersion/{num_versions}", headers=headers
    )


# Check that the response was not an error code
if get_bodo_versions_response.status_code != 200:
    print("bodoVersion call to the backend service failed")
    print(f"Response received from backend: {get_bodo_versions_response}")
    print(
        f"Response content received from backend: {get_bodo_versions_response.content.decode('utf-8')}"
    )
    exit(1)

print("Successfully triggered backend service...")
print(f"Response received from backend: {get_bodo_versions_response}")  # DEBUG_STMT
print(
    f"Response content received from backend: {get_bodo_versions_response.content.decode('utf-8')}"
)  # DEBUG_STMT

bodo_versions = json.loads(get_bodo_versions_response.content.decode("utf-8"))

print("bodo_versions: \n", bodo_versions)

bodo_versions_file_lines = []
for bodo_version_entry in bodo_versions:
    _bodo_version = bodo_version_entry["bodoVersion"]
    bodo_versions_file_lines.append(f"{_bodo_version}\n")

print("bodo_versions_file_lines: \n", bodo_versions_file_lines)

print("Writing response to bodo_versions.txt")
with open("bodo_versions.txt", "w") as f:
    f.write("".join(bodo_versions_file_lines))

print("Written to bodo_versions.txt")
