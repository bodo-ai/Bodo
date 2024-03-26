#include "_bodo_common.h"
#include "include/v8-context.h"
#include "include/v8-persistent-handle.h"
#include "include/v8-script.h"

struct JavaScriptFunction {
    // The type of the return array
    const std::unique_ptr<bodo::DataType> return_type;
    // The names of the arguments
    const std::vector<std::string> arg_names;
    // The context to execute the function in
    const v8::Global<v8::Context> context;
    // This is a global function so we can call it from any context, contains
    // the passed in body
    v8::Global<v8::Function> v8_func;
    // The event to be used for tracing
    tracing::ResumableEvent tracing_event;
    static std::unique_ptr<JavaScriptFunction> create(
        std::string _body, std::vector<std::string> _arg_names,
        std::unique_ptr<bodo::DataType> _return_type);
    // Marked as friend to allow create_javascript_udf_py_entry to call the
    // constructor, avoids using create which returns a unique pointer when we
    // need a raw pointer
    friend JavaScriptFunction *create_javascript_udf_py_entry(
        char *body, int32_t body_len, array_info *arg_names,
        int8_t *return_array_type, int8_t *return_array_c_type,
        int32_t size_return_array_type);

   private:
    // This is a random function name to avoid collisions with other functions
    const size_t size_rand_names = 32;
    std::string random_function_name;
    /**
     * Private because it needs an active handle_scope, create() should be used
     * instead
     * @param _body The JavaScript code as a string, will be wrapped in a
     * function
     * @param _arg_names The names of the arguments
     * @param _return_type The type of the return array
     */
    JavaScriptFunction(std::string _body, std::vector<std::string> _arg_names,
                       std::unique_ptr<bodo::DataType> _return_type);
};

/**
 * @brief Idempotent function to initialize V8
 */
void init_v8();

/**
 * @brief Python entrypoint to create a JavaScript UDF from a string and return
 * type. This function just wraps body into a string and deserializes the return
 * array type before passing to the constructor.
 * @param body The JavaScript code
 * @param body_len The length of the JavaScript code
 * @param arg_names The names of the arguments, should be a string array_info
 * with one element per argument
 * @param return_array_type The type of the return array in serialized form
 * @param return_array_c_type The dtype of the return array in serialized form
 * @return The JavaScript UDF
 */
JavaScriptFunction *create_javascript_udf_py_entry(
    char *body, int32_t body_len, array_info *arg_names,
    int8_t *return_array_type, int8_t *return_array_c_type,
    int32_t size_return_array_type);

/**
 * @brief Delete a JavaScript UDF
 * @param func The JavaScript UDF
 */
void delete_javascript_udf_py_entry(JavaScriptFunction *func);

/**
 * @brief Python entry point to execute a JavaScript UDF
 * This function just wraps raw pointers before passing to
 * execute_javascript_udf
 * @param f The JavaScript UDF
 * @param args The arguments, must be the same number as f->arg_names.size()
 * @return The resulting array_info
 */
array_info *execute_javascript_udf_py_entry(JavaScriptFunction *f,
                                            array_info **args);

/**
 * @brief Calls the JavaScriptFunction with the supplied args and returns the
 * result
 * @param func JavaScriptFunction to execute
 * @param args The arguments to call the function with, each array should be the
 * same length, or length 1 for scalars.
 * @return Output array, each element is cast to func->return_type. It's of
 * length args[0] or 1 if there are no args
 */
std::shared_ptr<array_info> execute_javascript_udf(
    JavaScriptFunction *func, std::vector<std::unique_ptr<array_info>> args);
