import os
import sys

import requests
import logging


def login(auth_url, bot_username, bot_password):
    """
    Function which login user to platform.
    :param auth_url: login endpoint for each environment,
    :param bot_username: email of bot used for login,
    :param bot_password: password of bot used for login

    :return: access_token - token from auth service
    """
    response = requests.post(
        "{}/identity/resources/auth/v1/user".format(auth_url),
        json={
            "email": bot_username,
            "password": bot_password,
        },
    )

    if response.status_code != 200:
        logging.info(response.json())
        raise Exception(f"Login to Bodo Platform failed: {response.status_code}")

    # use organization id to switch tenant with superadmin capability
    assert "ORGANIZATION_UUID" in os.environ
    logging.info("ORGANIZATION_UUID: %s", os.environ["ORGANIZATION_UUID"])
    payload = {"tenantId": os.environ["ORGANIZATION_UUID"]}
    url = f"{auth_url}/identity/resources/users/v1/tenant"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": response.json()["accessToken"],
    }

    # Put to switch tenant
    requests.put(url, headers=headers, json=payload)

    # Log back in to receive access token with superadmin capability
    response = requests.post(
        "{}/identity/resources/auth/v1/user".format(auth_url),
        json={
            "email": bot_username,
            "password": bot_password,
        },
    )

    if response.status_code != 200:
        logging.info(response.json())
        raise Exception(f"Switch to organization failed: {response.status_code}")

    return response.json()["accessToken"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    if len(sys.argv) == 4:
        # Only print here as the bash scripts captures the access token to use it for authentication
        print(login(sys.argv[1], sys.argv[2], sys.argv[3]))
        sys.exit(0)
    else:
        print(
            "Missing arguments, please provide the authentication endpoint, username and password"
        )
        sys.exit(1)
