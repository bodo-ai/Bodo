#include "arrow_compat.h"
#include "../libs/_bodo_common.h"
#include "arrow/dataset/dataset.h"

// Silence warnings from including generated code
PUSH_IGNORED_COMPILER_ERROR("-Wreturn-type-c-linkage")
PUSH_IGNORED_COMPILER_ERROR("-Wunused-variable")
PUSH_IGNORED_COMPILER_ERROR("-Wunused-function")
#include "pyarrow_wrappers_api.h"
POP_IGNORED_COMPILER_ERROR()

namespace arrow {
namespace py {

int import_pyarrow_wrappers() { return ::import_bodo__io__pyarrow_wrappers(); }
DEFINE_WRAP_FUNCTIONS(dataset, std::shared_ptr<arrow::dataset::Dataset>, out);
DEFINE_WRAP_FUNCTIONS(fragment, std::shared_ptr<arrow::dataset::Fragment>, out);
DEFINE_WRAP_FUNCTIONS(expression, arrow::compute::Expression, out.is_valid());
DEFINE_WRAP_FUNCTIONS(filesystem, std::shared_ptr<arrow::fs::FileSystem>, out);

}  // namespace py
}  // namespace arrow
