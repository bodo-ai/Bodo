#!/bin/bash

# This script generates bodo/libs/_mpi.h which adds [[nodiscard]] to every MPI
# method that is not in the allowlist below. The goal is to make sure
# error checking is not missed for MPI calls.

# The return values of these methods are allowed to be discarded
allowlist=$(cat <<EOF
MPI_Initialized
MPI_Finalized
MPI_Comm_rank
MPI_Comm_size
MPI_Comm_free
MPI_Error_string
MPI_Error_class
MPI_Wait
# The following methods are not used anywhere in our code but are also not
# defined on platform, so to prevent compilation issues we skip adding
# [[no discard]].
MPI_T_source_get_num
MPI_T_source_get_timestamp
MPI_T_source_get_info
MPI_T_event_set_dropped_handler
MPI_T_event_register_callback
MPI_T_event_read
MPI_T_event_handle_set_info
MPI_T_event_handle_get_info
MPI_T_event_handle_free
MPI_T_event_handle_alloc
MPI_T_event_get_timestamp
MPI_T_event_get_source
MPI_T_event_get_num
MPI_T_event_get_info
MPI_T_event_get_index
MPI_T_event_copy
MPI_T_event_callback_set_info
MPI_T_event_callback_get_info
MPI_T_category_get_num_events
MPI_Psend_init
MPI_Pready_range
MPI_Pready_list
MPI_Pready
MPI_Parrived
MPI_Isendrecv_replace_c
MPI_Isendrecv_replace
MPI_Isendrecv_c
MPI_Intercomm_create_from_groups
MPI_Info_get_string
MPI_Comm_idup_with_info
MPI_Info_create_env
MPI_Isendrecv
MPI_Precv_init
MPI_T_category_get_events
EOF
)

# Location of the MPI dylib in the current conda environment
MPI_DYLIB=$CONDA_PREFIX/lib/libmpi.12.dylib

list_mpi_methods() {
  # List all MPI methods in the MPI dylib
  nm -gU $MPI_DYLIB \
    | grep "T _MPI" \
    | awk '{print $3}' \
    | sed 's/_MPI/MPI/g' \
    | grep "^MPI_"
}

filtered_mpi_methods() {
  # Get the list of MPI methods that are not in the allowlist
  for mpi_method in $(list_mpi_methods); do
    skip=0
    for allowed in $allowlist; do
      if [ "$mpi_method" == "$allowed" ]; then
        skip=1
        break
      fi
    done

    if [ $skip -eq 1 ]; then
      continue
    fi
    echo "$mpi_method"
  done
}

cat <<EOF
/**
 * This file is generated by buildscripts/generate_mpi_wrapper.sh DO NOT EDIT MANUALLY
 * To regenerate this file, run buildscripts/generate_mpi_wrapper.sh > bodo/libs/_mpi.h
 *
 * This file adds [[nodiscard]] to most MPI methods, but otherwise just includes mpi.h.
 * The goal is to make sure error checking is not missed for MPI calls.
 */

#include <mpi.h>
#define NODISCARD(fn) [[nodiscard]] decltype(fn) fn

EOF

for mpi_method in $(filtered_mpi_methods); do
  echo "NODISCARD($mpi_method);"
done
