// Copyright (C) 2024 Bodo Inc. All rights reserved.
#include "fast_typer.h"
#include <functional>
#include <unordered_map>

// Declaring unbox_args from Cython manually to work around Cython not
// generating native_typer.h file (TODO: fix)
std::vector<std::shared_ptr<bodo::Type>> unbox_args(PyObject*);

namespace bodo {

/**
 * @brief type inference function for bodo.libs.memory_budget.register_operator
 *
 * @param args argument types (tuple object of Type objects)
 * @return std::shared_ptr<Signature> call type signature
 */
std::shared_ptr<Signature> register_operator_infer(PyObject* args) {
    static std::shared_ptr<Signature> sig =
        std::make_shared<Signature>(NoneType::getInstance(), unbox_args(args));
    return sig;
}

/**
 * @brief singleton registry of type inference functions
 *
 */
class CallTyperRegistry {
   public:
    typedef std::function<std::shared_ptr<Signature>(PyObject*)> InferFunc;
    const std::unordered_map<std::string, InferFunc> callTypers = {
        {"bodo.libs.memory_budget.register_operator", register_operator_infer}};
    CallTyperRegistry() {}
    static CallTyperRegistry& getInstance() {
        static CallTyperRegistry instance;
        return instance;
    }
    CallTyperRegistry(CallTyperRegistry const&) = delete;
    void operator=(CallTyperRegistry const&) = delete;
};

/**
 * @brief call type resolver called from Cython
 *
 * @param func_path full module path of the function to type
 * @param args argument types (tuple object of Type objects)
 * @return std::shared_ptr<Signature> call type signature
 */
std::shared_ptr<Signature> resolve_call_type(char* func_path, PyObject* args) {
    return CallTyperRegistry::getInstance().callTypers.at(func_path)(args);
}

}  // namespace bodo
