#include <sstream>
#include "../libs/_bodo_common.h"
#include "./test.hpp"

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
        : filenm_(filenm), nm_(nm), lineno_(tc.lineno_), func_(tc.func_) {
        PyObject_Init(reinterpret_cast<PyObject *>(this), &PyTestCase::TYPE);
        Py_INCREF(reinterpret_cast<PyObject *>(this));

        nmstr_ = PyUnicode_DecodeLocaleAndSize(nm.data(), nm.size(), nullptr);
        if (nmstr_ == nullptr)
            throw std::runtime_error("PyTestCase fails");

        filenmstr_ = PyUnicode_DecodeLocaleAndSize(filenm.data(), filenm.size(),
                                                   nullptr);
        if (filenmstr_ == nullptr)
            throw std::runtime_error("PyTestCase fails");
    }

    ~PyTestCase() {
        Py_DECREF(nmstr_);
        Py_DECREF(filenmstr_);
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
        } else {
            return nullptr;
        }
    }

    static PyTypeObject TYPE;

   private:
    std::string filenm_, nm_;
    PyObject *filenmstr_, *nmstr_;
    int lineno_;
    std::function<void()> func_;
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
    0,                                /* tp_setattr */
    0,                                /* tp_compare */
    &PyTestCase_as_str,               /* tp_repr */
    0,                                /* tp_as_number */
    0,                                /* tp_as_sequence */
    0,                                /* tp_as_mapping */
    0,                                /* tp_hash  */
    &PyTestCase_call,                 /* tp_call */
    &PyTestCase_as_str,               /* tp_str */
    0,                                /* tp_getattro */
    0,                                /* tp_setattro */
    0,                                /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,               /* tp_flags */
    "PyTest C++ test case",           /* tp_doc */
    0,                                /* tp_traverse */
    0,                                /* tp_clear */
    0,                                /* tp_richcompare */
    0,                                /* tp_weaklistoffset */
    0,                                /* tp_iter */
    0,                                /* tp_iternext */
    0,                                /* tp_methods */
    0,                                /* tp_members */
    0,                                /* tp_getset */
    0,                                /* tp_base */
    0,                                /* tp_dict */
    0,                                /* tp_descr_get */
    0,                                /* tp_descr_set */
    0,                                /* tp_dictoffset */
    0,                                /* tp_init */
    0,                                /* tp_alloc */
    0,                                /* tp_new */
};

PyMODINIT_FUNC PyInit_test_cpp(void) {
    PyObject *m;
    MOD_DEF(m, "test_cpp", "No docs", NULL);
    if (m == NULL)
        return NULL;

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
    if (b)
        return;
    std::stringstream error;
    error << "Assertion failed at " << loc.file_name() << ":" << loc.line()
          << "," << loc.column();

    check(b, error.str().c_str(), loc);
}

void bodo::tests::check(bool b, const char *msg, std::source_location loc) {
    if (b)
        return;

    std::cerr << "Assertion failed: " << msg << std::endl;
    throw std::runtime_error("Check failure");
}

void bodo::tests::check_exception(std::function<void()> f,
                                  const char *expected_msg_start) {
    bool caught = false;
    const char *msg = nullptr;

    try {
        f();
    } catch (std::runtime_error &e) {
        msg = e.what();
        caught = true;
    }

    if (!caught) {
        std::cerr << "No exception was thrown. Expected: " << expected_msg_start
                  << std::endl;
        throw std::runtime_error("check_exception failure");
    }

    if (!std::string_view(msg).starts_with(expected_msg_start)) {
        std::cerr << "Exception message did not match."
                  << "\nExpected: " << expected_msg_start << "\nActual: " << msg
                  << std::endl;
        throw std::runtime_error("check_exception failure");
    }
}
