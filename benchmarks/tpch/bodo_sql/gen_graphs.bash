#!/bin/bash

for q in {1..22}; do
  echo "=== Running query ${q} ==="

  OUTFILE="memory_gpu.q${q}.txt"

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
    python ../gen_async_stats_graph.py "${ASYNC_STATS_FILE}" ${q}
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
    python ../gen_async_stats_graph.py "${ASYNC_STATS_FILE}" ${q}
    echo "Generated graph for query ${q} from ${ASYNC_STATS_FILE}"
  else
    echo "Skipping graph generation: ${ASYNC_STATS_FILE} missing or empty"
  fi
  echo "=== Finished query ${q} ==="
  echo
done

echo "All queries complete."

