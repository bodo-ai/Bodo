echo "building bodo for Windows OS!!"

:: Build using pip and CMake
"%PYTHON%" -m pip install --no-deps --no-build-isolation -vv ^
    --config-settings=build.verbose=true ^
    --config-settings=logging.level="DEBUG" ^ .