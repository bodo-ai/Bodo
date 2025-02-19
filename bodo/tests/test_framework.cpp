#include <sstream>
#include "../libs/_bodo_common.h"
#include "../libs/_distributed.h"
#include "./test.hpp"
#include "mpi.h"

/// Python CAPI type that is constructed for each test case.
///
/// These test cases are available in the bodo.ext.test_cpp module (if built).
///
/// If you have an instance of a 'PyTestCase', you can call it to run the test,
/// or access location and provenance information using the filename, name,
/// line, etc attributes.
struct PyTestCase {
    // All python compatible objects must have this as the head
    PyObject_HEAD

    PyTestCase(const std::string &filenm, const std::string &nm,
               bodo::tests::test_case tc)
        : filenm_(filenm),
          nm_(nm),
          lineno_(tc.lineno_),
          func_(tc.func_),
          markers_(tc.markers_) {
        PyObject_Init(reinterpret_cast<PyObject *>(this), &PyTestCase::TYPE);
        Py_INCREF(reinterpret_cast<PyObject *>(this));

        nmstr_ = PyUnicode_DecodeLocaleAndSize(nm.data(), nm.size(), nullptr);
        if (nmstr_ == nullptr) {
            throw std::runtime_error("PyTestCase fails");
        }

        filenmstr_ = PyUnicode_DecodeLocaleAndSize(filenm.data(), filenm.size(),
                                                   nullptr);
        if (filenmstr_ == nullptr) {
            throw std::runtime_error("PyTestCase fails");
        }

        markers_list_ = PyList_New(markers_.size());
        if (markers_list_ == nullptr) {
            throw std::runtime_error("PyTestCase fails");
        }
        for (size_t i = 0; i < markers_.size(); ++i) {
            std::string_view marker = markers_[i];
            PyObject *markerstr = PyUnicode_DecodeLocaleAndSize(
                marker.data(), marker.size(), nullptr);
            if (markerstr == nullptr) {
                throw std::runtime_error("PyTestCase fails");
            }
            PyList_SetItem(markers_list_, i, markerstr);
        }
    }

    ~PyTestCase() {
        Py_DECREF(nmstr_);
        Py_DECREF(filenmstr_);
        Py_DECREF(markers_list_);
    }

    PyObject *operator()(PyObject *args, PyObject *kwargs) {
        try {
            func_();
        } catch (std::exception &e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
            return nullptr;
        } catch (std::string &e) {
            PyErr_SetString(PyExc_RuntimeError, e.c_str());
            return nullptr;
        }

        Py_INCREF(Py_None);
        return Py_None;
    }

    PyObject *as_str() {
        Py_INCREF(nmstr_);
        return nmstr_;
    }

    static void destroy(PyObject *testcase_) {
        delete reinterpret_cast<PyTestCase *>(testcase_);
    }

    PyObject *get_attr(const char *attr) {
        if (strcmp(attr, "filename") == 0) {
            Py_INCREF(filenmstr_);
            return filenmstr_;
        } else if (strcmp(attr, "name") == 0) {
            Py_INCREF(nmstr_);
            return nmstr_;
        } else if (strcmp(attr, "lineno") == 0) {
            return PyLong_FromLong(lineno_);

        } else if (strcmp(attr, "markers") == 0) {
            Py_INCREF(markers_list_);
            return markers_list_;
        } else {
            return nullptr;
        }
    }

    static PyTypeObject TYPE;

   private:
    std::string filenm_, nm_;
    PyObject *filenmstr_, *nmstr_, *markers_list_;
    int lineno_;
    std::function<void()> func_;
    std::vector<std::string> markers_;
};

PyObject *PyTestCase_as_str(PyObject *tc) {
    return reinterpret_cast<PyTestCase *>(tc)->as_str();
}

PyObject *PyTestCase_getattr(PyObject *tc, char *attr) {
    return reinterpret_cast<PyTestCase *>(tc)->get_attr(attr);
}

PyObject *PyTestCase_call(PyObject *tc, PyObject *args, PyObject *kwargs) {
    return (*reinterpret_cast<PyTestCase *>(tc))(args, kwargs);
}

PyTypeObject PyTestCase::TYPE = {
    PyVarObject_HEAD_INIT(NULL, 0)

        "TestCase",                   /* tp_name */
    sizeof(PyTestCase),               /* tp_basicsize */
    0,                                /* tp_itemsize */
    (destructor)&PyTestCase::destroy, /* tp_dealloc */
    0,                                /* tp_print */
    &PyTestCase_getattr,              /* tp_getattr */
    nullptr,                          /* tp_setattr */
    nullptr,                          /* tp_compare */
    &PyTestCase_as_str,               /* tp_repr */
    nullptr,                          /* tp_as_number */
    nullptr,                          /* tp_as_sequence */
    nullptr,                          /* tp_as_mapping */
    nullptr,                          /* tp_hash  */
    &PyTestCase_call,                 /* tp_call */
    &PyTestCase_as_str,               /* tp_str */
    nullptr,                          /* tp_getattro */
    nullptr,                          /* tp_setattro */
    nullptr,                          /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,               /* tp_flags */
    "PyTest C++ test case",           /* tp_doc */
    nullptr,                          /* tp_traverse */
    nullptr,                          /* tp_clear */
    nullptr,                          /* tp_richcompare */
    0,                                /* tp_weaklistoffset */
    nullptr,                          /* tp_iter */
    nullptr,                          /* tp_iternext */
    nullptr,                          /* tp_methods */
    nullptr,                          /* tp_members */
    nullptr,                          /* tp_getset */
    nullptr,                          /* tp_base */
    nullptr,                          /* tp_dict */
    nullptr,                          /* tp_descr_get */
    nullptr,                          /* tp_descr_set */
    0,                                /* tp_dictoffset */
    nullptr,                          /* tp_init */
    nullptr,                          /* tp_alloc */
    nullptr,                          /* tp_new */
};

PyMODINIT_FUNC PyInit_test_cpp(void) {
    PyObject *m;
    MOD_DEF(m, "test_cpp", "No docs", nullptr);
    if (m == nullptr) {
        return nullptr;
    }

    if (PyType_Ready(&PyTestCase::TYPE) != 0) {
        Py_DECREF(m);
        return nullptr;
    }

    PyObject *test_list = PyList_New(0);
    if (!test_list) {
        throw std::runtime_error("Could not create list");
    }

    auto suites(bodo::tests::suite::get_all());
    for (auto suite : suites) {
        for (auto [test_nm, test] : suite->tests()) {
            if (PyList_Append(test_list,
                              reinterpret_cast<PyObject *>(new PyTestCase(
                                  suite->name(), test_nm, test))) != 0) {
                throw std::runtime_error("Could not append to list");
            }
        }
    }

    PyObject_SetAttrString(m, "tests", test_list);
    return m;
}

static bodo::tests::suite *s_current = nullptr;

/// @brief This is the main list that ends up exposed to the python side
static std::vector<bodo::tests::suite *> s_suites;

void bodo::tests::suite::set_current(bodo::tests::suite *n) {
    s_current = n;
    s_suites.push_back(n);
}

bodo::tests::suite *bodo::tests::suite::get_current() { return s_current; }

const std::vector<bodo::tests::suite *> &bodo::tests::suite::get_all() {
    return s_suites;
}

void bodo::tests::check(bool b, std::source_location loc) {
    if (b) {
        return;
    }
    std::stringstream error;
    error << "Assertion failed at " << loc.file_name() << ":" << loc.line()
          << "," << loc.column();

    check(b, error.str().c_str(), loc);
}

void bodo::tests::check(bool b, const char *msg, std::source_location loc) {
    if (b) {
        return;
    }

    std::cout << "Assertion failed: " << msg << std::endl;
    throw std::runtime_error("Check failure");
}

void bodo::tests::check_exception(std::function<void()> f,
                                  const char *expected_msg_start) {
    bool caught = false;
    std::string msg;

    try {
        f();
    } catch (std::runtime_error &e) {
        msg = std::string(e.what());
        caught = true;
    }

    if (!caught) {
        std::cout << "No exception was thrown. Expected: " << expected_msg_start
                  << std::endl;
        throw std::runtime_error("check_exception failure");
    }

    if (!std::string_view(msg).starts_with(expected_msg_start)) {
        std::cout << "Exception message did not match."
                  << "\nExpected: " << expected_msg_start << "\nActual: " << msg
                  << std::endl;
        throw std::runtime_error("check_exception failure");
    }
}

void bodo::tests::check_parallel(bool b, std::source_location loc) {
    bool global_b = b;
    CHECK_MPI(MPI_Allreduce(&b, &global_b, 1, MPI_UNSIGNED_CHAR, MPI_LAND,
                            MPI_COMM_WORLD),
              "bodo::tests::check_parallel: MPI error on MPI_Allreduce:");
    if (global_b) {
        return;
    }
    std::stringstream error;
    error << "Assertion failed at " << loc.file_name() << ":" << loc.line()
          << "," << loc.column();

    check(global_b, error.str().c_str(), loc);
}
