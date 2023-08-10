#### Package tests ################
import pytest
import time


@pytest.mark.parametrize(
    "package, version",
    [
        ("notebook", ""),
        ("jupyter_server", ""),
        ("jupyter_client", ""),
        ("jupyterlab", ""),
        ("jupyter_core", ""),
        ("nbconvert", ""),
        ("ipython", ""),
        ("ipykernel", ""),
        ("ipywidgets", "7.7.0"),
        ("bodo-jupyterlab", "1.9.0"),
        ("httpie", ""),
    ],
)
def test_jupyter_ipy_packages(pip_packages, conda_packages, package, version, host):
    assert (package in conda_packages) or (
        package in pip_packages
    ), f"Package {package} is not installed!"
    if package in conda_packages and version != "":
        assert (
            conda_packages[package]["version"] == version
        ), f"Package {package} is installed with version {version}. Version {conda_packages[package]['version']} expected"
    if package in pip_packages and version != "":
        assert (
            pip_packages[package]["version"] == version
        ), f"Package {package} is installed with version {version}. Version {pip_packages[package]['version']} expected"

    if package == "httpie":

        def test_httpie(host):
            r = host.run("http --help")
            assert r.succeeded

        test_httpie(host)
    
    if package == "jupyter_client":
        # Hacky way to convert string versions to numberic once.
        # Split by . -> analyzes major, minor, patch (number bw 0-99)
        # Multiplies and sums to an integer value of all 3 parts
        def version_test():
            curr_version = pip_packages[package]["version"]
            l = [int(x, 10) for x in curr_version.split('.')]
            l.reverse()
            version_processed = sum(x * (100 ** i) for i, x in enumerate(l))
            assert version_processed >= 70402, "An unsupported, buggy version of jupyter-client was installed! Make sure to install a version >= 7.4.2"
        
        version_test()



def test_kernel_specs(host):
    assert host.run("jupyter kernelspec list | grep bodo_platform_dummy_kernel").succeeded, "Dummy kernel not installed"

    # For some reason python3 always shows up in kernelspec list even when it's not installed
    # The install path contains site-packages/ipykernel/resources only when it's "ghost"
    assert "site-packages/ipykernel/resources" in host.run("jupyter kernelspec list | grep python3").stdout.strip(), "Python3 kernel installed"

def test_nginx(host):
    run = host.run("sudo nginx -T")
    out = run.stdout.strip()
    assert run.succeeded, "Nginx config file validation failed!"
    assert (
        "127.0.0.1:8080" in out
    ), "Nginx server running on the wrong port, check the config file"


def test_jupyter_commands(host):
    assert host.run(
        "jupyter server list"
    ).succeeded, (
        "Jupyter server list command couldnt run, check jupyter server installation!"
    )
    assert host.run(
        "jupyter notebook list"
    ).succeeded, (
        "Jupyter notebook list command couldnt run, check jupyter server installation!"
    )


def test_jupyter_start(host, conda_packages):
    # Why are we doing this? "jupyter lab" is a not-terminating command, so since we cannot keep it running,
    # We use the timeout linux command to forcefuly send a termination signal after x seconds of it running,
    # Enough for us to analyze is jupyter is working correctly
    run = host.run("timeout -s SIGTERM --preserve-status 5s jupyter lab --allow-root")
    time.sleep(5)
    # why stderr? because we are manually sending a sigterm to cancel this run, so the output shows up at stderr
    out = run.stderr
    server_version = conda_packages["jupyter_server"]["version"]
    check_string = f"Jupyter Server {server_version} is running at:"
    assert run.succeeded, "Jupyter is not starting, check the installation!"
    assert check_string in out, "Jupyter server not running!"
