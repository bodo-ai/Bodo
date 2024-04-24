from numba.core import cgutils, types
from numba.extending import (
    NativeValue,
    box,
    make_attribute_wrapper,
    models,
    register_model,
    typeof_impl,
    unbox,
)


def install_py_obj_class(
    types_name, module, python_type=None, class_name=None, model_name=None
):
    """
    Helper for generating Python Object types with opaque
    models. This dynamically generates the class,
    creates a single instance of the type, and
    registers the type inside Numba with proper argument
    registering, boxing, and unboxing.
    We also create a custom memory model for better memory management
    to avoid memory leaks (see notes below).
    We create these classes and set them in the provided module.
    For convenience, we also return a reference to the new type class.
    @params:
    types_name: Name of the type to register inside numba.core.types
    module: The module to declare the new classes in.
    python_type (optional): The actual python type for registering types
    of arguments to functions. We skip the `typeof_impl.register` step
    when not provided
    class_name (optional): Name of the class for the generated type class.
    model_name (optional): Name of the class for the generated model class.
    @returns: The created type class.
    """
    # If not provided, create class_name by converting the types_name to camel_case
    class_name = (
        "".join(map(str.title, types_name.split("_")))
        if class_name is None
        else class_name
    )
    # If not provided, create model_name by adding 'Model' to class_name
    model_name = f"{class_name}Model" if model_name is None else model_name

    class_text = f"class {class_name}(types.Opaque):\n"
    class_text += f"    def __init__(self):\n"
    class_text += f"       types.Opaque.__init__(self, name='{class_name}')\n"
    # Implement the reduce method for pickling
    # https://stackoverflow.com/questions/11658511/pickling-dynamically-generated-classes
    class_text += f"    def __reduce__(self):\n"
    class_text += f"        return (types.Opaque, ('{class_name}',), self.__dict__)\n"

    locs = {}
    exec(class_text, {"types": types, "models": models}, locs)
    class_value = locs[class_name]
    # Register the class in the given module for outside use
    setattr(module, class_name, class_value)
    class_instance = class_value()
    # Register the type in numba.core.types
    setattr(types, types_name, class_instance)

    # creating a wrapper meminfo around the Python object that holds and manages a reference
    # to avoid memory leaks (see [BE-2825]).
    # See boxing and unboxing for Numpy arrays in Numba as an example:
    # https://github.com/numba/numba/blob/496bc20d91485affa842a63173522a6afef453b6/numba/core/runtime/_nrt_python.c#L332
    # https://github.com/numba/numba/blob/496bc20d91485affa842a63173522a6afef453b6/numba/core/runtime/_nrt_python.c#L310
    # https://github.com/numba/numba/blob/496bc20d91485affa842a63173522a6afef453b6/numba/core/runtime/_nrt_python.c#L248
    # https://github.com/numba/numba/blob/496bc20d91485affa842a63173522a6afef453b6/numba/core/runtime/_nrt_python.c#L34
    class_text = f"class {model_name}(models.StructModel):\n"
    class_text += f"    def __init__(self, dmm, fe_type):\n"
    class_text += f"        members = [\n"
    class_text += f"            ('meminfo', types.MemInfoPointer({types_name})),\n"
    class_text += f"            ('pyobj', types.voidptr),\n"
    class_text += f"        ]\n"
    class_text += f"        models.StructModel.__init__(self, dmm, fe_type, members)\n"

    exec(
        class_text, {"types": types, "models": models, types_name: class_instance}, locs
    )

    # Register the model
    model_value = locs[model_name]
    setattr(module, model_name, model_value)
    register_model(class_value)(model_value)
    make_attribute_wrapper(class_value, "pyobj", "_pyobj")

    if python_type is not None:
        typeof_impl.register(python_type)(lambda val, c: class_instance)
    unbox(class_value)(unbox_py_obj)
    box(class_value)(box_py_obj)

    # We return the class for convenience (better IDE compatibility).
    # If the function is called from the module specified in 'module',
    # this is essentially a no-op.
    return class_value


def box_py_obj(typ, val, c):
    struct_proxy = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)
    obj = struct_proxy.pyobj
    c.pyapi.incref(obj)
    c.context.nrt.decref(c.builder, typ, val)
    return obj


def unbox_py_obj(typ, obj, c):
    return NativeValue(
        create_struct_from_pyobject(typ, obj, c.context, c.builder, c.pyapi)
    )


def create_struct_from_pyobject(typ, obj, context, builder, pyapi):
    """
    Helper function to wrap a regular Python Object pointer into a version that
    borrows and manages a reference to the object. This is useful for passing Python
    objects allocated in C++ into object mode.
    """
    struct_proxy = cgutils.create_struct_proxy(typ)(context, builder)
    # borrows and manages a reference for obj (see data model comments above)
    struct_proxy.meminfo = pyapi.nrt_meminfo_new_from_pyobject(
        context.get_constant_null(types.voidptr), obj
    )
    struct_proxy.pyobj = obj
    return struct_proxy._getvalue()
