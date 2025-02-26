echo "building bodo for Windows OS!!"

echo %VSINSTALLDIR%

set "SETUPTOOLS_SCM_PRETEND_VERSION=%PKG_VERSION%"

set "CONDA_PREFIX=%PREFIX%"
set "CMAKE_GEN=Visual Studio 17 2022"

:: Build using pip and CMake
"%PYTHON%" -m pip install --no-deps --no-build-isolation -vv ^
    --config-settings=build.verbose=true ^
    --config-settings=logging.level="DEBUG" .


@REM conda build . -c conda-forge --no-verify --no-anaconda-upload --croot C:\Users\owner\dev\conda-bld --build-id-pat bb --python 3.12
