#!/bin/bash
# Usage: ./run_all_queries.sh
# Runs dataframe_queries.py for queries 1..22, moves rank files, extracts async stats, and generates graphs.

PYTHON_CMD="python dataframe_queries.py"
FOLDER="~/tpch/from_s3/SF100"
SCALE_FLAG="--scale_factor 100"
WORKERS_FLAG="--queries"   # note: this script will append the query number after this flag
#ENV_PREFIX="BODO_GPU_STREAMING_BATCH_SIZE_DIVISOR=4000 BODO_DATAFRAME_LIBRARY_DUMP_PLANS=1 BODO_GPU=1 BODO_DATAFRAME_LIBRARY_RUN_PARALLEL=1 BODO_NUM_WORKERS=2"
export BODO_GPU_STREAMING_BATCH_SIZE_DIVISOR=2000
export BODO_DATAFRAME_LIBRARY_DUMP_PLANS=1
export BODO_GPU=1
export BODO_DATAFRAME_LIBRARY_RUN_PARALLEL=1
export BODO_NUM_WORKERS=2
TIMEOUT_SECONDS=120
KILL_AFTER_SECONDS=10

for q in {1..22}; do
  echo "=== Running query ${q} ==="

  OUTFILE="memory_gpu.q${q}.txt"

   # Run the python command with environment variables and a timeout.
  # Redirect stdout+stderr to the per-query log file.
  # timeout exit codes: 124 = command timed out, 137 = killed (SIGKILL), other non-zero = error.
  echo "Running: ${ENV_PREFIX} ${PYTHON_CMD} --folder ${FOLDER} ${SCALE_FLAG} --queries ${q}"
  #timeout --kill-after=${KILL_AFTER_SECONDS}s ${TIMEOUT_SECONDS}s \
  #  bash -c "${ENV_PREFIX} ${PYTHON_CMD} --folder \"${FOLDER}\" ${SCALE_FLAG} --no_warmup --queries ${q}" &> "${OUTFILE}"
  #rc=$?
  timeout --kill-after=${KILL_AFTER_SECONDS}s ${TIMEOUT_SECONDS}s python dataframe_queries.py --folder ~/tpch/from_s3/SF100 --scale_factor 100  --no_warmup --queries ${q} >& "${OUTFILE}"

  if [[ ${rc} -eq 0 ]]; then
    echo "Query ${q} finished within ${TIMEOUT_SECONDS}s (exit 0)."
  elif [[ ${rc} -eq 124 ]]; then
    echo "Query ${q} timed out after ${TIMEOUT_SECONDS}s (exit 124). Moving on to next query."
  elif [[ ${rc} -eq 137 ]]; then
    echo "Query ${q} was killed (exit 137). Moving on to next query."
  else
    echo "Query ${q} exited with code ${rc}. Continuing to post-run steps and next query."
  fi

  # Move rank files if they exist
  if [[ -f "rank0" ]]; then
    mv -f "rank0" "memory_gpu.q${q}.rank0"
    echo "Moved rank0 -> memory_gpu.q${q}.rank0"
  else
    echo "Warning: rank0 not found for query ${q}"
  fi

  if [[ -f "rank1" ]]; then
    mv -f "rank1" "memory_gpu.q${q}.rank1"
    echo "Moved rank1 -> memory_gpu.q${q}.rank1"
  else
    echo "Warning: rank1 not found for query ${q}"
  fi

  # Extract async stats from the rank0 file (if present)
  RANK0_FILE="memory_gpu.q${q}.rank0"
  ASYNC_STATS_FILE="memory_gpu.q${q}.rank0_async_stats"
  if [[ -f "${RANK0_FILE}" ]]; then
    # grep the lines containing GPUMem, take the last field, split on '/', and output first 4 parts as CSV
    grep GPUMem "${RANK0_FILE}" | awk '{print $NF}' | awk -F'/' '{print $1 "," $2 "," $3 "," $4}' > "${ASYNC_STATS_FILE}" || true
    echo "Wrote async stats to ${ASYNC_STATS_FILE}"
  else
    echo "Skipping async stats extraction: ${RANK0_FILE} not found"
  fi

  # Generate graph from the async stats file (if created)
  if [[ -s "${ASYNC_STATS_FILE}" ]]; then
    python gen_async_stats_graph.py "${ASYNC_STATS_FILE}"
    echo "Generated graph for query ${q} from ${ASYNC_STATS_FILE}"
  else
    echo "Skipping graph generation: ${ASYNC_STATS_FILE} missing or empty"
  fi

  # Extract async stats from the rank1 file (if present)
  RANK1_FILE="memory_gpu.q${q}.rank1"
  ASYNC_STATS_FILE="memory_gpu.q${q}.rank1_async_stats"
  if [[ -f "${RANK1_FILE}" ]]; then
    # grep the lines containing GPUMem, take the last field, split on '/', and output first 4 parts as CSV
    grep GPUMem "${RANK1_FILE}" | awk '{print $NF}' | awk -F'/' '{print $1 "," $2 "," $3 "," $4}' > "${ASYNC_STATS_FILE}" || true
    echo "Wrote async stats to ${ASYNC_STATS_FILE}"
  else
    echo "Skipping async stats extraction: ${RANK1_FILE} not found"
  fi

  # Generate graph from the async stats file (if created)
  if [[ -s "${ASYNC_STATS_FILE}" ]]; then
    python gen_async_stats_graph.py "${ASYNC_STATS_FILE}"
    echo "Generated graph for query ${q} from ${ASYNC_STATS_FILE}"
  else
    echo "Skipping graph generation: ${ASYNC_STATS_FILE} missing or empty"
  fi
  echo "=== Finished query ${q} ==="
  echo
done

echo "All queries complete."

