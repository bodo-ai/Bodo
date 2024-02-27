import json
import pytest


@pytest.fixture(scope="module")
def pip_packages(host):
    out = host.run("pip list --format=json")
    assert out.rc == 0, "Error while executing pip list"
    pkgs = {}
    for pkg in json.loads(out.stdout):
        pkgs[pkg["name"]] = {"version": pkg["version"]}
    return pkgs


@pytest.fixture(scope="module")
def conda_packages(host):
    out = host.run("conda list --json")
    assert out.rc == 0, "Error while executing pip list"
    pkgs = {}
    for pkg in json.loads(out.stdout):
        pkgs[pkg["name"]] = {"version": pkg["version"]}
    return pkgs


@pytest.fixture(scope="module")
def remote_dir():
    return "/var"
