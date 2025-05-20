#include "arrow_compat.h"
#include "../libs/_bodo_common.h"
#include "arrow/dataset/dataset.h"

// Silence warnings from including generated code
#if defined(__GNUC__) || defined(__clang__)
PUSH_IGNORED_COMPILER_ERROR("-Wreturn-type-c-linkage")
PUSH_IGNORED_COMPILER_ERROR("-Wunused-variable")
PUSH_IGNORED_COMPILER_ERROR("-Wunused-function")
#endif
#include "pyarrow_wrappers_api.h"

#if defined(__GNUC__) || defined(__clang__)
POP_IGNORED_COMPILER_ERROR()
#endif

namespace arrow::py {

int import_pyarrow_wrappers() { return ::import_bodo__io__pyarrow_wrappers(); }
DEFINE_WRAP_FUNCTIONS(dataset, std::shared_ptr<arrow::dataset::Dataset>, out);
DEFINE_WRAP_FUNCTIONS(fragment, std::shared_ptr<arrow::dataset::Fragment>, out);
DEFINE_WRAP_FUNCTIONS(expression, arrow::compute::Expression, out.is_valid());
DEFINE_WRAP_FUNCTIONS(filesystem, std::shared_ptr<arrow::fs::FileSystem>, out);

}  // namespace arrow::py

std::shared_ptr<arrow::Schema> unwrap_schema(PyObject* pyarrow_schema) {
    if (arrow::py::import_pyarrow() != 0) {
        throw std::runtime_error("importing pyarrow failed");
    }

    auto res = arrow::py::unwrap_schema(pyarrow_schema);
    if (!(res.ok())) {
        std::string err_msg =
            "Unwrapping Arrow Schema from Python Object Failed: " +
            res.status().ToString();
        throw std::runtime_error(err_msg);
    }
    return res.ValueOrDie();
}
