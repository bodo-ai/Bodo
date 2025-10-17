# Bodo Error Messages {#bodoerrors}

This page lists some of the compilation error messages you may
encounter with your jitted functions, reasons for them and suggestions
on how to proceed with resolving them.

## Unsupported Bodo Functionality

-  `BodoError: <functionality> not supported yet`

    As the error states, this message is encountered when you are
    attempting to call an as yet unsupported API within a jit
    function. For example :

    ```py
    @bodo.jit
    def unsupported_func(pd_str_series):
        return pd_str_series.str.casefold()
    ```

    would result in an unsupported `BodoError` as follows:

    ```console
    BodoError: Series.str.casefold not supported yet
    ```

    Please submit a request for us to support your required
    functionality [here](https://github.com/bodo-ai/feedback). Also
    consider joining our [community slack](https://join.slack.com/t/bodocommunity/shared_invite/zt-qwdc8fad-6rZ8a1RmkkJ6eOX1X__knA),
    where you can interact directly with fellow Bodo users to find a
    workaround for your requirements. For longer and more detailed
    discussions, please join our [discourse](https://discourse.bodo.ai).

    !!! seealso "See Also"
        [@bodo.wrap_python][objmode] can be used to switch to
        Python interpreted context to be able to run your workload, but we
        strongly recommend trying to find a Bodo-native workaround.


-   `BodoError: <operation> : <parameter_name> parameter only supports default value`

    Certain methods only support default parameter values for some of
    their parameters. Please see [supported Pandas API][pandas] for
    a list of supported pandas functionality and their respective
    parameters. We also have a list of supported [Numpy][numpy] , as well as
    [ML][ml] operations.

## Typing Errors

-   `BodoError: <operation>: <operand> must be a compile time constant`

    Bodo needs certain arguments to be known at compile time to
    produce an optimized binary. Please refer to the documentation on
    [compile time constants][require_constants] for more
    details.

-   `BodoError: dtype <DataType> cannot be stored in arrays`

    This error message is encountered when Bodo is unable to assign a
    supported type to elements of an array.

    Example:
    
    ```py
    @bodo.jit
    def obj_in_array():
        df = pd.DataFrame({'col1': ["1", "2"], 'col2': [3, 4]})
        return df.select_dtypes(include='object')

    a = obj_in_array()
    print(a)
    ```
    
    Error:
    ```console
    BodoError: dtype pyobject cannot be stored in arrays
    ```
    
    In this example, we get this error because we attempted to get
    Bodo to recognize `col1` as a column with the datatype `object`,
    and the `object` type is too generic for Bodo. A workaround for
    this specific example would be to return
    `df.select_dtypes(exclude='int')`.

- `Invalid Series.dt/Series.cat/Series.str, cannot handle conditional yet`

    This error is encountered when there are conditional assignments
    of series functions `Series.dt`, `Series.cat` or `Series.str`,
    which Bodo cannot handle yet.

    Example:

    ```py
    @bodo.jit
    def conditional_series_str(flag):
        s = pd.Series(["Str_Series"])
        s1 = pd.Series(["Str_Series_1"]).str
        if flag:
            s1 = s.str
        else:
            s1 = s1
        return s1.split("_")
    ```
  
    Error:

    ```console
    BodoError: ...
              Invalid Series.str, cannot handle conditional yet
    ```
  
    When using these operations, you need to include the function
    and accessor together inside the control flow if it is
    absolutely necessary. For this specific case, we simply compute
    the `str.split` within the conditional:

    ```py
    @bodo.jit
    def test_category(flag):
        s = pd.Series(["A_Str_Series"])
        s1 = pd.Series(["test_series"]).str
        s2 = None
        if flag:
            s2 = s.str.split("_")
        else:
            s2 = s1.split("_")
        return s2
    ```
  
## Unsupported Numba Errors

-   `numba.core.errors.TypingError: Compilation error`

    This is likely due to unsupported functionality. If you encounter
    this error, please provide us a minimum reproducer for this error
    [here](https://github.com/bodo-ai/feedback).

-   `numba.core.errors.TypingError: Unknown attribute <attribute> of type`

    This is an uncaught error due to unsupported functionality. If
    you encounter this error, please provide us a minimum reproducer
    for this error [here](https://github.com/bodo-ai/feedback).

