// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include "_distributed.h"
#include <ctime>

PyMODINIT_FUNC PyInit_hdist(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "hdist", "No docs", -1, NULL,
    };
    m = PyModule_Create(&moduledef);
    if (m == NULL) return NULL;

#ifdef TRIAL_PERIOD
    // printf("trial period %d\n", TRIAL_PERIOD);
    // printf("trial start %d\n", TRIAL_START);
    // check expiration date
    std::time_t curr_time = std::time(0);  // get time of now
    std::time_t start_time = TRIAL_START;
    double time_diff = std::difftime(curr_time, start_time) / (60 * 60 * 24);
    if (time_diff > TRIAL_PERIOD) {
        PyErr_SetString(PyExc_RuntimeError, "Bodo trial period has expired!");
        return NULL;
    }
#endif

    // make sure MPI is initialized, assuming this will be called
    // on all processes
    int is_initialized;
    MPI_Initialized(&is_initialized);
    if (!is_initialized) MPI_Init(NULL, NULL);

#ifdef MAX_CORE_COUNT
    int num_pes;
    MPI_Comm_size(MPI_COMM_WORLD, &num_pes);
    // printf("max core count %d\n", MAX_CORE_COUNT);
    // printf("number of processors %d\n", num_pes);
    if (num_pes>MAX_CORE_COUNT) {
        PyErr_SetString(PyExc_RuntimeError, "Exceeded the max core count!");
          return NULL;
    }
#endif

    // initialize decimal_mpi_type
    // TODO: free when program exits
    if (decimal_mpi_type == MPI_DATATYPE_NULL) {
        MPI_Type_contiguous(2, MPI_LONG_LONG_INT, &decimal_mpi_type);
        MPI_Type_commit(&decimal_mpi_type);
    }

    int decimal_bytes;
    MPI_Type_size(decimal_mpi_type, &decimal_bytes);
    // decimal_value should be exactly 128 bits to match Python
    if (decimal_bytes != 16)
        std::cerr << "invalid decimal mpi type size" << std::endl;

    PyObject_SetAttrString(m, "dist_get_rank",
                           PyLong_FromVoidPtr((void *)(&dist_get_rank)));
    PyObject_SetAttrString(m, "dist_get_size",
                           PyLong_FromVoidPtr((void *)(&dist_get_size)));
    PyObject_SetAttrString(m, "dist_get_start",
                           PyLong_FromVoidPtr((void *)(&dist_get_start)));
    PyObject_SetAttrString(m, "dist_get_end",
                           PyLong_FromVoidPtr((void *)(&dist_get_end)));
    PyObject_SetAttrString(
        m, "dist_get_node_portion",
        PyLong_FromVoidPtr((void *)(&dist_get_node_portion)));
    PyObject_SetAttrString(m, "dist_get_time",
                           PyLong_FromVoidPtr((void *)(&dist_get_time)));
    PyObject_SetAttrString(m, "get_time",
                           PyLong_FromVoidPtr((void *)(&get_time)));
    PyObject_SetAttrString(m, "barrier",
                           PyLong_FromVoidPtr((void *)(&barrier)));

    PyObject_SetAttrString(m, "dist_reduce",
                           PyLong_FromVoidPtr((void *)(&dist_reduce)));
    PyObject_SetAttrString(m, "dist_exscan",
                           PyLong_FromVoidPtr((void *)(&dist_exscan)));
    PyObject_SetAttrString(m, "dist_arr_reduce",
                           PyLong_FromVoidPtr((void *)(&dist_arr_reduce)));
    PyObject_SetAttrString(m, "dist_irecv",
                           PyLong_FromVoidPtr((void *)(&dist_irecv)));
    PyObject_SetAttrString(m, "dist_isend",
                           PyLong_FromVoidPtr((void *)(&dist_isend)));
    PyObject_SetAttrString(m, "dist_recv",
                           PyLong_FromVoidPtr((void *)(&dist_recv)));
    PyObject_SetAttrString(m, "dist_send",
                           PyLong_FromVoidPtr((void *)(&dist_send)));
    PyObject_SetAttrString(m, "dist_wait",
                           PyLong_FromVoidPtr((void *)(&dist_wait)));
    PyObject_SetAttrString(
        m, "dist_get_item_pointer",
        PyLong_FromVoidPtr((void *)(&dist_get_item_pointer)));
    PyObject_SetAttrString(m, "get_dummy_ptr",
                           PyLong_FromVoidPtr((void *)(&get_dummy_ptr)));
    PyObject_SetAttrString(m, "c_gather_scalar",
                           PyLong_FromVoidPtr((void *)(&c_gather_scalar)));
    PyObject_SetAttrString(m, "c_gatherv",
                           PyLong_FromVoidPtr((void *)(&c_gatherv)));
    PyObject_SetAttrString(m, "c_allgatherv",
                           PyLong_FromVoidPtr((void *)(&c_allgatherv)));
    PyObject_SetAttrString(m, "c_bcast",
                           PyLong_FromVoidPtr((void *)(&c_bcast)));
    PyObject_SetAttrString(m, "c_alltoallv",
                           PyLong_FromVoidPtr((void *)(&c_alltoallv)));
    PyObject_SetAttrString(m, "c_alltoall",
                           PyLong_FromVoidPtr((void *)(&c_alltoall)));
    PyObject_SetAttrString(m, "allgather",
                           PyLong_FromVoidPtr((void *)(&allgather)));
    PyObject_SetAttrString(m, "comm_req_alloc",
                           PyLong_FromVoidPtr((void *)(&comm_req_alloc)));
    PyObject_SetAttrString(m, "req_array_setitem",
                           PyLong_FromVoidPtr((void *)(&req_array_setitem)));
    PyObject_SetAttrString(m, "dist_waitall",
                           PyLong_FromVoidPtr((void *)(&dist_waitall)));
    PyObject_SetAttrString(m, "comm_req_dealloc",
                           PyLong_FromVoidPtr((void *)(&comm_req_dealloc)));

    PyObject_SetAttrString(m, "finalize",
                           PyLong_FromVoidPtr((void *)(&finalize)));
    PyObject_SetAttrString(m, "oneD_reshape_shuffle",
                           PyLong_FromVoidPtr((void *)(&oneD_reshape_shuffle)));
    PyObject_SetAttrString(m, "permutation_int",
                           PyLong_FromVoidPtr((void *)(&permutation_int)));
    PyObject_SetAttrString(
        m, "permutation_array_index",
        PyLong_FromVoidPtr((void *)(&permutation_array_index)));

    // add actual int value to module
    PyObject_SetAttrString(m, "mpi_req_num_bytes",
                           PyLong_FromSize_t(get_mpi_req_num_bytes()));
    PyObject_SetAttrString(m, "ANY_SOURCE",
                           PyLong_FromLong((long)MPI_ANY_SOURCE));
    return m;
}
