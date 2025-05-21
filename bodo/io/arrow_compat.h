#pragma once
#include <arrow/compute/expression.h>
#include <arrow/dataset/dataset.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/python/pyarrow.h>

namespace arrow::py {
#define DECLARE_WRAP_FUNCTIONS(FUNC_SUFFIX, TYPE_NAME)                 \
    ARROW_PYTHON_EXPORT bool is_##FUNC_SUFFIX(PyObject*);              \
    ARROW_PYTHON_EXPORT arrow::Result<TYPE_NAME> unwrap_##FUNC_SUFFIX( \
        PyObject*);                                                    \
    ARROW_PYTHON_EXPORT PyObject* wrap_##FUNC_SUFFIX(const TYPE_NAME&);

#define DEFINE_WRAP_FUNCTIONS(FUNC_SUFFIX, TYPE_NAME, IS_VALID_CHECK)        \
    bool is_##FUNC_SUFFIX(PyObject* obj) {                                   \
        return ::pyarrow_is_##FUNC_SUFFIX(obj) != 0;                         \
    }                                                                        \
                                                                             \
    PyObject* wrap_##FUNC_SUFFIX(const TYPE_NAME& src) {                     \
        return ::pyarrow_wrap_##FUNC_SUFFIX(src);                            \
    }                                                                        \
    arrow::Result<TYPE_NAME> unwrap_##FUNC_SUFFIX(PyObject* obj) {           \
        auto out = ::pyarrow_unwrap_##FUNC_SUFFIX(obj);                      \
        if (IS_VALID_CHECK) {                                                \
            return std::move(out);                                           \
        } else {                                                             \
            return arrow::Status::TypeError("Could not unwrap ", #TYPE_NAME, \
                                            " from Python object of type '", \
                                            Py_TYPE(obj)->tp_name, "'");     \
        }                                                                    \
    }

}  // namespace arrow::py

extern "C++" {

/**
 * @brief Unwrap PyArrow Schema PyObject and return the C++ value
 * NOTE: Not calling arrow::py::unwrap_schema() in ArrowReader constructor
 * directly due to a segfault with pip. See:
 * https://bodo.atlassian.net/browse/BSE-2925
 *
 * @param pyarrow_schema input PyArrow Schema
 * @return std::shared_ptr<arrow::Schema> C++ Schema
 */
std::shared_ptr<arrow::Schema> unwrap_schema(PyObject* pyarrow_schema);

namespace arrow::py {
int import_pyarrow_wrappers();
DECLARE_WRAP_FUNCTIONS(dataset, std::shared_ptr<arrow::dataset::Dataset>);
DECLARE_WRAP_FUNCTIONS(fragment, std::shared_ptr<arrow::dataset::Fragment>);
DECLARE_WRAP_FUNCTIONS(expression, arrow::compute::Expression);
DECLARE_WRAP_FUNCTIONS(filesystem, std::shared_ptr<arrow::fs::FileSystem>);
}  // namespace arrow::py

}  // extern "C++"
