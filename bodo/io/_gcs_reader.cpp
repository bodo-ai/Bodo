// Copyright (C) 2021 Bodo Inc. All rights reserved.
#include <Python.h>
#include <iostream>

#include "../libs/_bodo_common.h"
#include "_bodo_file_reader.h"
#include "arrow/filesystem/filesystem.h"
#include "arrow/io/interfaces.h"
#include "arrow/python/filesystem.h"
#include "arrow/result.h"
#include "arrow/status.h"
#include "pyfs.cpp"

// if status of arrow::Result is not ok, form an err msg and raise a
// runtime_error with it.
#define CHECK_ARROW(expr, msg /*, s3_fs_region*/)                            \
    if (!(expr.ok())) {                                                      \
        std::string err_msg = std::string("Error in arrow[gcsfs]: ") + msg + \
                              " " + expr.ToString() + ".\n";                 \
        throw std::runtime_error(err_msg);                                   \
    }

// if status of arrow::Result is not ok, form an err msg and raise a
// runtime_error with it. If it is ok, get value using ValueOrDie
// and assign it to lhs using std::move
#define CHECK_ARROW_AND_ASSIGN(res, msg, lhs /*, s3_fs_region*/) \
    CHECK_ARROW(res.status(), msg /*, s3_fs_region*/)            \
    lhs = std::move(res).ValueOrDie();

// a global singleton instance of PyFileSystem that is
// initialized the first time it is needed and reused afterwards
static PyObject *pyfs;  // owns the arrow::py::fs::PyFileSystem object
static bool is_fs_initialized = false;

std::shared_ptr<arrow::py::fs::PyFileSystem> get_gcs_fs() {
    // TODO: allow passing options to gcsfs like project, token, etc.
    // TODO are there regions to handle in GCS?
    // TODO: error checking for CPython API calls

    if (!is_fs_initialized) {
        // Python:
        //
        // import gcsfs
        // fs = gcsfs.GCSFileSystem(token='anon')
        // import pyarrow.fs
        // import bodo.io.pyfs
        // pyfs = bodo.io.pyfs.PyFileSystemBodo(pyarrow.fs.FSSpecHandler(fs))
        //
        // In C++ we get pointer to arrow::py::fs::PyFileSystem by calling
        // get_cpp_fs(pyfs) which we defined in pyfs.pyx

        // import gcsfs
        PyObject *gcsfs_mod = PyImport_ImportModule("gcsfs");
        // fs = gcsfs.GCSFileSystem(token=None)
        PyObject *GCSFileSystem =
            PyObject_GetAttrString(gcsfs_mod, "GCSFileSystem");
        Py_DECREF(gcsfs_mod);
        PyObject *args = PyTuple_New(0);  // no positional args
        PyObject *kwargs = Py_BuildValue("{s:s}", "token", NULL);
        PyObject *fs = PyObject_Call(GCSFileSystem, args, kwargs);
        Py_DECREF(args);
        Py_DECREF(kwargs);
        Py_DECREF(GCSFileSystem);

        // import pyarrow.fs
        PyObject *pyarrow_fs_mod = PyImport_ImportModule("pyarrow.fs");
        // handler = pyarrow.fs.FSSpecHandler(fs)
        PyObject *handler =
            PyObject_CallMethod(pyarrow_fs_mod, "FSSpecHandler", "O", fs);
        Py_DECREF(pyarrow_fs_mod);
        Py_DECREF(fs);

        // import bodo.io.pyfs
        PyObject *bodo_pyfs_mod = PyImport_ImportModule("bodo.io.pyfs");
        // pyfs = bodo.io.pyfs.PyFileSystemBodo(handler)
        pyfs = PyObject_CallMethod(bodo_pyfs_mod, "PyFileSystemBodo", "O",
                                   handler);
        Py_DECREF(bodo_pyfs_mod);
        Py_DECREF(handler);

        is_fs_initialized = true;
    }

    std::shared_ptr<arrow::py::fs::PyFileSystem> pyfs_cpp =
        std::dynamic_pointer_cast<arrow::py::fs::PyFileSystem>(
            get_cpp_fs((c_PyFileSystemBodo *)pyfs));
    return pyfs_cpp;
}

std::shared_ptr<arrow::py::fs::PyFileSystem> get_fsspec_fs() {
    if (!is_fs_initialized) {
        PyObject *fsspec = PyImport_ImportModule("fsspec");
        PyObject *fsspec_filesystem =
            PyObject_GetAttrString(fsspec, "filesystem");
        Py_DECREF(fsspec);
        PyObject *args = PyTuple_New(0);
        PyObject *kwargs = Py_BuildValue("{s:s}", "protocol", "http");
        PyObject *fs = PyObject_Call(fsspec_filesystem, args, kwargs);
        Py_DECREF(args);
        Py_DECREF(kwargs);
        Py_DECREF(fsspec_filesystem);
        PyObject *pyarrow_fs = PyImport_ImportModule("pyarrow.fs");
        PyObject *handler =
            PyObject_CallMethod(pyarrow_fs, "FSSpecHandler", "O", fs);
        Py_DECREF(pyarrow_fs);
        Py_DECREF(fs);
        PyObject *bodo_pyfs = PyImport_ImportModule("bodo.io.pyfs");
        pyfs = PyObject_CallMethod(bodo_pyfs, "PyFileSystemBodo", "O", handler);
        Py_DECREF(bodo_pyfs);
        Py_DECREF(handler);
        is_fs_initialized = true;
    }
    std::shared_ptr<arrow::py::fs::PyFileSystem> pyfs_cpp =
        std::dynamic_pointer_cast<arrow::py::fs::PyFileSystem>(
            get_cpp_fs((c_PyFileSystemBodo *)pyfs));
    return pyfs_cpp;
}

static int finalize_gcs() {
    if (is_fs_initialized) {
        Py_DECREF(pyfs);  // note that this will delete the underlying Arrow C++
                          // objects as well
        pyfs = nullptr;
        is_fs_initialized = false;
    }
    return 0;
}

void gcs_get_fs(std::shared_ptr<arrow::py::fs::PyFileSystem> *fs/*,
                std::string bucket_region*/) {
    try {
        *fs = get_gcs_fs(/*bucket_region*/);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    }
}

void gcs_open_file(const char *fname,
                   std::shared_ptr<::arrow::io::RandomAccessFile> *file/*,
                   const char *bucket_region*/) {
    std::shared_ptr<arrow::py::fs::PyFileSystem> fs =
        get_gcs_fs(/*bucket_region*/);
    arrow::Result<std::shared_ptr<::arrow::io::RandomAccessFile>> result;
    result = fs->OpenInputFile(std::string(fname));
    CHECK_ARROW_AND_ASSIGN(result, "fs->OpenInputFile",
                           *file /*, fs->region()*/)
}

void fsspec_open_file(std::string fname,
                      std::shared_ptr<::arrow::io::RandomAccessFile> *file) {
    std::shared_ptr<arrow::py::fs::PyFileSystem> fs = get_fsspec_fs();
    arrow::Result<std::shared_ptr<::arrow::io::RandomAccessFile>> result;
    result = fs->OpenInputFile(fname);
    CHECK_ARROW_AND_ASSIGN(result, "fs->OpenInputFile", *file)
}

PyMODINIT_FUNC PyInit_gcs_reader(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "gcs_reader", "No docs", -1, NULL,
    };
    m = PyModule_Create(&moduledef);
    if (m == NULL) return NULL;

    // Only ever called from C++
    PyObject_SetAttrString(m, "gcs_open_file",
                           PyLong_FromVoidPtr((void *)(&gcs_open_file)));
    // Only ever called from C++
    PyObject_SetAttrString(m, "gcs_get_fs",
                           PyLong_FromVoidPtr((void *)(&gcs_get_fs)));
    PyObject_SetAttrString(m, "finalize_gcs",
                           PyLong_FromVoidPtr((void *)(&finalize_gcs)));
    return m;
}

#undef CHECK_ARROW
#undef CHECK_ARROW_AND_ASSIGN
