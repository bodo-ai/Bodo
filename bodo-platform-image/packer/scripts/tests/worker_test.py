#### Package tests ################
import os
import time

import pytest


@pytest.mark.parametrize(
    "package",
    [
        "bodo",
        "snowflake-connector-python",
        "bodo-platform-utils",
        "bodo-platform-extensions",
    ],
)
def test_packages(package, pip_packages, conda_packages, remote_dir, host):
    assert (
        package in conda_packages or package in pip_packages
    ), f"Packages {package} not installed!"

    if package == "bodo-platform-utils":

        def test_bodo_platform_utils(host):
            r = host.run(
                "python " + remote_dir + "/tests/test_files/test_bodo_utils.py"
            )
            assert r.succeeded

        test_bodo_platform_utils(host)


@pytest.mark.parametrize(
    "package, version",
    [
        ("bodo-iceberg-connector", os.environ["ICEBERG_VERSION"] if not None else ""),
        ("bodosql", os.environ["BODOSQL_VERSION"] if not None else ""),
    ],
)
def test_optional_packages(package, version, pip_packages, conda_packages):
    if version is None or version == "":
        pytest.skip(f"Image does not support {package}")
    else:
        assert (
            package in conda_packages or package in pip_packages
        ), f"Optional package {package} not installed!"


@pytest.mark.parametrize("package", ["mpi4py"])
def test_mpi4py(host, package, pip_packages, conda_packages, remote_dir):
    assert (
        package in conda_packages or package in pip_packages
    ), f"Package {package} not installed!"

    def test_mpi_run(host):
        mpi = host.run("mpiexec --version")
        output = mpi.stdout
        assert mpi.succeeded, "mpiexec failed!"
        assert output.startswith(
            "Intel(R) MPI Library"
        ), "Wrong MPI version downloaded, We need Intels MPI!"

    def test_mpiexec(host):
        r = host.run(
            "mpiexec -n 4 python " + remote_dir + "/tests/test_files/test_mpi.py"
        )
        assert (
            r.succeeded
        ), "Sample mpiexec with a python/numpy operation script failed!"

    def test_allgatherv(host):
        r = host.run(
            "mpiexec -n 4 python "
            + remote_dir
            + "/tests/test_files/test_mpi_allgatherv.py"
        )
        assert r.succeeded, "Sample mpiexec with a python allgather script failed!"

    test_mpi_run(host)
    test_mpiexec(host)
    test_allgatherv(host)


@pytest.mark.parametrize("package", ["mpich", "mpi"])
def test_mpich_not_installed(package, pip_packages, conda_packages):
    # For performance reason, we only want to be working with intels MPI packages
    # Which is specifically installed from the source binary and not other package managers
    # Since we have seen errors in the past with various MPI distributions available on the machine
    # We want to avoid this condition
    assert not (
        package in conda_packages or package in pip_packages
    ), "Extra MPI packages installed, DELETE them!"


def test_ucx_not_installed(pip_packages, conda_packages):
    # UCX can interfere with Intel MPI, causing it to use Mellanox for
    # communication, particularly on Azure. For now, since we don't
    # install the Mellanox drivers, we should remove UCX entirely
    # so Intel MPI doesn't find it
    assert not (
        "ucx" in conda_packages or "ucx" in pip_packages
    ), "UCX Package Installed in Python, DELETE them!"


def test_bodo_arrow_fork_installed(conda_packages):
    pkgs = [
        "libarrow",
        "libarrow-acero",
        "libarrow-dataset",
        "libarrow-substrait",
        "libparquet",
        "pyarrow-core",
        "pyarrow",
    ]

    for pkg in pkgs:
        assert pkg in conda_packages, f"Package {pkg} not installed in AMI"
        assert (
            conda_packages[pkg]["channel"] == "bodo.ai"
        ), f"Package {pkg} not installed from `bodo.ai` channel"


@pytest.mark.parametrize(
    "package",
    [
        ("ipython"),
        ("bodo-platform-ipyparallel-kernel"),
        ("ipykernel"),
        ("ipyparallel"),
        ("ipywidgets"),
    ],
)
def test_ipy_packages(host, package, pip_packages, conda_packages, remote_dir):
    assert (
        package in conda_packages or package in pip_packages
    ), f"Package{package} not installed!"
    if package == "ipython":

        def test_ipyparallel_run(host):
            r = host.run(
                "python " + remote_dir + "/tests/test_files/test_ipyparallel.py"
            )
            assert r.succeeded

        def test_mock_engine_start(host):
            run = host.run(
                "timeout -s SIGTERM --preserve-status 55s ipcluster start -n 2"
            )
            time.sleep(60)
            # why stderr? because we are manually sending a sigterm to cancel this run, so the output shows up at stderr
            out = run.stderr
            print(out)
            assert run.succeeded, "Ipcluster with mpi failed to start engines!!"
            assert "Engine Connected: 0" in out, "Engine 0 Could not start!"
            assert "Engine Connected: 1" in out, "Engine 1 could not start!"

        test_ipyparallel_run(host)
        test_mock_engine_start(host)


@pytest.mark.parametrize("package", ["nbconvert"])
@pytest.mark.skip(reason="test is not working on conda=4.12 python<3.10 image")
def test_nbconvert(host, package, pip_packages, conda_packages, remote_dir):
    assert (
        package in conda_packages or package in pip_packages
    ), "nbconvert not found in either package managers!!"

    def test_nbconvert_run(host):
        r = host.run(
            "jupyter nbconvert --ExecutePreprocessor.kernel_name=bodo_platform_ipyparallel_kernel\
        --to notebook --execute "
            + remote_dir
            + "/tests/test_files/test_nbconvert.ipynb --debug --allow-errors --stdout"
        )
        assert r.succeeded, "Nbconvert run failed, check installation on worker image!"
        out = r.stdout
        out = out.strip()
        # The below check works correctly to catch erroring out code,
        # commenting out since this causes a license error now,
        # Re-enable once Sahil changes the license check
        # print(out)
        # assert "error" not in out, "There was an error in execution! Check logs"

    test_nbconvert_run(host)


@pytest.mark.parametrize(
    "connector",
    [
        "snowflake-sqlalchemy",
        "snowflake-connector-python",
        "sqlalchemy",
        "psycopg2",
        "pymysql",
    ],
)
def test_connectors(connector, pip_packages, conda_packages):
    assert (
        connector in conda_packages or connector in pip_packages
    ), f"Database connector {connector} not installed!"


@pytest.mark.parametrize(
    "package, expected_version",
    [
        ("scikit-learn", "1.4"),
        ("bodo", os.environ["BODO_VERSION"]),
        ("bodo-iceberg-connector", os.environ["ICEBERG_VERSION"]),
        ("bodosql", os.environ["BODOSQL_VERSION"]),
        ("bodo-platform-ipyparallel-kernel", "2.0.0"),
        ("ipyparallel", "8.6.1"),
        ("ipywidgets", "8.1.1"),
        ("bodo-azurefs-sas-token-provider", "1.0"),
        ("pandas", "2.2.3"),
    ],
)
def test_versions(package, expected_version, conda_packages):
    if expected_version and len(expected_version) != 0:
        assert package in conda_packages, f"Package {package} not in conda_packages."
        actual_version = conda_packages[package]["version"]
        assert_msg = f"Unsupported package version installed for package {package}!\nExpected:{expected_version}\nFound:{actual_version}\n"
        assert actual_version.startswith(expected_version), assert_msg


def test_bodo_azurefs_sas_token_provider(host, remote_dir):
    def test_jar_in_CLASSPATH(host):
        r = host.run(
            "python "
            + remote_dir
            + "/tests/test_files/test_bodo_azurefs_sas_token_provider.py"
        )
        assert (
            r.succeeded
        ), "Error while checking for bodo_azurefs_sas_token_provider jar in CLASSPATH"

    test_jar_in_CLASSPATH(host)


# def test_slurm_install(host):
#    """make sure Slurm is installed properly"""
#    SLURM_VERSION = os.environ["SLURM_VERSION"]
#    # make sure slurmctld and slurmd are available
#    slurm = host.package("slurm")
#    assert slurm.is_installed, "slurm not installed properly"
#    assert slurm.version == SLURM_VERSION, "invalid slurm version"
#    slurmctld = host.package("slurm-slurmctld")
#    assert slurmctld.is_installed, "slurmctld not installed properly"
#    assert slurmctld.version == SLURM_VERSION, "invalid slurmctld version"


def test_goofys_install(host):
    """make sure goofys is installed properly"""

    r = host.run("which goofys")
    assert r.succeeded, "Goofys not installed"
