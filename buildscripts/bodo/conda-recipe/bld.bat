echo "building bodo for Windows OS!!"

set SETUPTOOLS_SCM_PRETEND_VERSION="%PKG_VERSION%"

set CONDA_PREFIX="%PREFIX%"
set CMAKE_GENERATOR="Visual Studio 17 2022"

:: Build using pip and CMake
"%PYTHON%" -m pip install --no-deps --no-build-isolation -vv ^
    --config-settings=build.verbose=true ^
    --config-settings=logging.level="DEBUG" .