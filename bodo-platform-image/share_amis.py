import os
import json
import requests
import argparse

from scripts.common import login
import subprocess


# Environment Variables
assert "BACKEND_SERVICE_URL" in os.environ
assert "AUTH_SERVICE_URL" in os.environ
assert "BOT_PLATFORM_USERNAME" in os.environ
assert "BOT_PLATFORM_PASSWORD" in os.environ

BACKEND_SERVICE_URL = os.environ["BACKEND_SERVICE_URL"]
AUTH_SERVICE_URL = os.environ["AUTH_SERVICE_URL"]
BOT_PLATFORM_USERNAME = os.environ["BOT_PLATFORM_USERNAME"]
BOT_PLATFORM_PASSWORD = os.environ["BOT_PLATFORM_PASSWORD"]


# Argument Parsing
parser = argparse.ArgumentParser(description='Share AMIs to Amazon')
parser.add_argument('-r', '--regions', dest='regions', action='extend', 
                    nargs='+', type=str, help='Regions to share AMI with')
parser.add_argument('-i', '--input', dest='input', action='store', type=str,
                    default='ami_share_requests.json',
                    help='JSON File Output of AMI CI Build')

args = parser.parse_args()


# Parse AMI entries JSON file
ami_entries = []
with open(args.input, 'r') as f:
  ami_entries = json.load(f)['amiAddAndShareRequests']

if args.regions is not None:
  ami_entries = [entry for entry in ami_entries if entry['region'] in args.regions]

print('AMI Entries:\n')
for entry in ami_entries:
  for (key, value) in entry.items():
    print(f'{key:<24}{value}')
  print()


# Login using the bot_herman account and get access token
access_token = login(AUTH_SERVICE_URL, BOT_PLATFORM_USERNAME, BOT_PLATFORM_PASSWORD)

# Add AMI entry to backend
print("Adding AMI entries to backend...")
headers = {"Authorization": f"Bearer {access_token}"}
cmd = ['aws', 'ec2', 'modify-image-attribute', '--launch-permission', 'Add=[{UserId=427443013497}]']

for entry in ami_entries:
  res = requests.post(f"{BACKEND_SERVICE_URL}/api/image/bodoAmi", 
                      headers=headers, data=entry)
  
  print(f"Added AMI in Region {entry['region']} to DB: Status {res.status_code}")

  process = subprocess.run(cmd + ['--region', entry['region'], '--image-id', entry['workerImageId']])
  print(f"Shared Worker AMI: Status {process.returncode}")
  process = subprocess.run(cmd + ['--region', entry['region'], '--image-id', entry['jupyterImageId']])
  print(f"Shared Jupyter AMI: Status {process.returncode}")
