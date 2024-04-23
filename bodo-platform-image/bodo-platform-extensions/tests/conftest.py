import pytest
from IPython.testing.globalipapp import start_ipython


@pytest.fixture(scope="session")
def session_ipython():
    return start_ipython()


@pytest.fixture(scope="function")
def ipython(session_ipython):
    session_ipython.run_line_magic(magic_name="load_ext", line="bodo_platform_extensions")
    yield session_ipython
    # session_ip.run_line_magic(magic_name="unload_ext", line="bodo_platform_extensions")
    session_ipython.run_line_magic(magic_name="reset", line="-f")
