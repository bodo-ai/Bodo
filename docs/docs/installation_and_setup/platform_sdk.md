# Bodo Platform SDK

Bodo Cloud Platform provides a simple SDK that can be integrated in CI/CD pipelines easily.
For example, compute jobs can be orchestrated
easily.

<!-- List of contents: -->

<!-- - [Getting Started](#getting-started) -->

## Getting started {#getting-started}

Install the latest Bodo SDK using:

```console
pip install bodosdk
```

The first step is to create an *API Token* in the Bodo Platform for
Bodo SDK authentication.
Navigate to *API Tokens* in the Admin Console to generate a token.
Copy and save the token's *Client ID* and *Secret Key* and use them for BodoClient
definition:

```python
from bodosdk.models import WorkspaceKeys
from bodosdk.client import get_bodo_client

keys = WorkspaceKeys(
    client_id='XYZ',
    secret_key='XYZ'
)
client = get_bodo_client(keys)
```

To learn more about Bodo SDK, refer to the docs on the [PyPi page for BodoSDK](https://pypi.org/project/bodosdk/)
