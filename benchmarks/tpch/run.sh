#!/bin/sh

FOLDER=${1:-"/Users/scottroutledge/dev/bodo-org/tpch_benchmark/tpch/SF10"}
POLARS_DIR=${2:-"/Users/scottroutledge/dev/bodo-org/tpch_benchmark/polars-benchmark"}

touch out.txt
echo "" > out.txt

# echo "Running Pandas"
# echo "### Pandas ###" > out.txt
# python dataframe_lib.py --folder "$FOLDER" --queries 1 2 3 4 5 6 7 8 9 10 --scale_factor 10 --backend pandas >> out.txt


# echo "Running Bodo"
# echo "### BODO ###" > out.txt
# python dataframe_lib.py --folder "$FOLDER" --queries 1 2 3 4 5 6 7 8 9 10 --scale_factor 10 >> out.txt


echo "Running Polars"
echo "### POLARS ###" >> out.txt
export PYTHONPATH=$POLARS_DIR
pip install polars linetimer pydantic-settings
python run_polars.py --folder "$FOLDER" --queries 1 2 3 4 5 6 7 8 9 10 --scale_factor 10 >> out.txt


# echo "Running PySpark"
# echo "### PYSPARK ###" >> out.txt
# python run_pyspark.py --folder "$FOLDER" --queries 1 2 3 4 5 6 7 8 9 10 --scale_factor 10 >> out.txt


# Parse output and plot
python gen_plot.py