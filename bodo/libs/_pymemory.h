#include <Python.h>

/**
 * @brief Initializes the memory_cpp module
 * primarily used to set the BufferPool singleton as
 * Arrow's memory pool in Cython, as well as some helper functions
 * for testing purposes
 */
PyMODINIT_FUNC PyInit_memory_cpp(void);
