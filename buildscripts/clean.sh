#!/bin/sh
echo "Removing the __pycache__"
find . -name "__pycache__" | xargs rm -rf
echo "Removing the build directory"
rm -rf build
echo "Removing bodo/io/csv_json_reader.cpp"
find . -name "csv_json_reader.cpp" | xargs rm -f
echo "Removing bodo/io/pyarrow_wrappers.cpp"
find . -name "pyarrow_wrappers.cpp" | xargs rm -f
echo "Removing bodo/utils/tracing.c"
find . -name "tracing.c" | xargs rm -f
echo "Removing bodo/memory.cpp"
find . -name "memory.cpp" | xargs rm -f
echo "Removing bodo/tests/memory_tester.cpp"
find . -name "memory_tester.cpp" | xargs rm -f
echo "Removing bodo/transforms/type_inference/native_typer.cpp"
find . -name "native_typer.cpp" | xargs rm -f
echo "Removing bodo/pandas/plan_optimizer.cpp"
find . -name "plan_optimizer.cpp" | xargs rm -f
echo "Removing DuckDB build"
rm -rf bodo/pandas/vendor/duckdb/build

