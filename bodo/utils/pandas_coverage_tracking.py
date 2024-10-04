# Copyright (C) 2024 Bodo Inc. All rights reserved.
""" Get pandas apis from a list of pre-specified URLs and use apis to find
methods/attributes we are currently not supporting that lack proper error messages.
This script will be run during the import bodo step if specified by _check_pandas_docs
"""
import numba
import requests
from numba.core.target_extension import dispatcher_registry

from bodo.pandas_compat import _check_pandas_change
from bodo.utils import search_templates

# URL of the Pandas API reference pages
urls = [
    "https://pandas.pydata.org/docs/reference/io.html",
    "https://pandas.pydata.org/docs/reference/general_functions.html",
    "https://pandas.pydata.org/docs/reference/series.html",
    "https://pandas.pydata.org/docs/reference/frame.html",
    "https://pandas.pydata.org/docs/reference/arrays.html",
    "https://pandas.pydata.org/docs/reference/indexing.html",
    "https://pandas.pydata.org/docs/reference/offset_frequency.html",
    "https://pandas.pydata.org/docs/reference/window.html",
    "https://pandas.pydata.org/docs/reference/groupby.html",
    "https://pandas.pydata.org/docs/reference/resampling.html",
    "https://pandas.pydata.org/docs/reference/style.html",
    "https://pandas.pydata.org/docs/reference/plotting.html",
    "https://pandas.pydata.org/docs/reference/options.html",
    "https://pandas.pydata.org/docs/reference/extensions.html",
    "https://pandas.pydata.org/docs/reference/testing.html",
    "https://pandas.pydata.org/docs/reference/missing_value.html",
]


def get_pandas_apis_from_url(url):
    from bs4 import BeautifulSoup

    result = []
    # Send a GET request to the URL
    response = requests.get(url)

    # Parse the page content with BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")

    # Find all 'a' tags with the class 'reference internal'
    api_refs = soup.find_all("a", class_="reference internal", href=True)

    # Filter and print only those links that contain a <code> tag
    # with class 'py-obj' (to avoid links in description to other APIs those have 'py-func')
    for ref in api_refs:
        if ref.find("code", {"class": "py-obj"}):
            result.append(ref.text.strip())

    return result


def get_all_pandas_apis():
    """Get all pandas api's as a list

    Returns:
        List[str]: Get all pandas api's as a list of paths (excluding the pd. part)
    """
    result = []
    for url in urls:
        result.extend(get_pandas_apis_from_url(url))

    return result


if _check_pandas_change:
    print("Checking Pandas API's for new methods/attributes, this may take a minute...")
    pandas_apis = get_all_pandas_apis()
    # We only target the CPU. The other option is Numba ufuncs
    disp = dispatcher_registry[numba.core.target_extension.CPU]
    typing_ctx = disp.targetdescr.typing_context
    # Probably not necessary to refresh
    typing_ctx.refresh()

    search_templates.lookup_all(
        pandas_apis,
        typing_ctx,
        types_dict=search_templates.bodo_pd_types_dict,
        keys=[
            "Series",
        ],
    )
