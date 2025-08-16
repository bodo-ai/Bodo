#include "_pymemory.h"
#include "_memory.h"

extern "C" {

/// @brief Returns the pointer to the default BufferPool
PyObject* default_buffer_pool_ptr(PyObject* self, PyObject* Py_UNUSED(args)) {
    bodo::BufferPool* pool = bodo::BufferPool::DefaultPtr();
    return PyLong_FromVoidPtr(pool);
}

/// @brief Get the smallest size class of the default BufferPool
PyObject* default_buffer_pool_smallest_size_class(PyObject* self,
                                                  PyObject* Py_UNUSED(args)) {
    bodo::BufferPool* pool = bodo::BufferPool::DefaultPtr();
    return PyLong_FromLongLong(pool->GetSmallestSizeClassSize());
}

/// @brief Returns the number of bytes allocated by the default BufferPool
PyObject* default_buffer_pool_bytes_allocated(PyObject* self,
                                              PyObject* Py_UNUSED(args)) {
    bodo::BufferPool* pool = bodo::BufferPool::DefaultPtr();
    return PyLong_FromLongLong(pool->bytes_allocated());
}

/// @brief Returns the number of bytes pinned by the default BufferPool
PyObject* default_buffer_pool_bytes_pinned(PyObject* self,
                                           PyObject* Py_UNUSED(args)) {
    bodo::BufferPool* pool = bodo::BufferPool::DefaultPtr();
    return PyLong_FromUnsignedLongLong(pool->bytes_pinned());
}

/// @brief Performs cleanup on the default BufferPool
PyObject* default_buffer_pool_cleanup(PyObject* self,
                                      PyObject* Py_UNUSED(args)) {
    bodo::BufferPool* pool = bodo::BufferPool::DefaultPtr();
    pool->Cleanup();
    Py_RETURN_NONE;
}

PyMODINIT_FUNC PyInit_memory_cpp(void) {
    static PyMethodDef SpamMethods[] = {
        {"default_buffer_pool_ptr", default_buffer_pool_ptr, METH_NOARGS,
         "No Docs"},
        {"default_buffer_pool_smallest_size_class",
         default_buffer_pool_smallest_size_class, METH_NOARGS, "No Docs"},
        {"default_buffer_pool_bytes_allocated",
         default_buffer_pool_bytes_allocated, METH_NOARGS, "No Docs"},
        {"default_buffer_pool_bytes_pinned", default_buffer_pool_bytes_pinned,
         METH_NOARGS, "No Docs"},
        {"default_buffer_pool_cleanup", default_buffer_pool_cleanup,
         METH_NOARGS, "No Docs"},

        // Sentinel required, otherwise segfault?
        {nullptr, nullptr, 0, nullptr}};
    static struct PyModuleDef moduledef = {.m_base = PyModuleDef_HEAD_INIT,
                                           .m_name = "memory_cpp",
                                           .m_doc = "No Docs",
                                           .m_size = -1,
                                           .m_methods = SpamMethods};
    PyObject* m = PyModule_Create(&moduledef);
    if (m == nullptr) {
        return nullptr;
    }

    // Not calling bodo_common_init() since memory_cpp is used inside of it

    return m;
}

}  // extern "C"
