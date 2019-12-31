// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include "_datetime_ext.h"

extern "C" {

PyMODINIT_FUNC PyInit_hdatetime_ext(void) {
    PyObject *m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "hdatetime_ext", "No docs", -1, NULL,
    };
    m = PyModule_Create(&moduledef);
    if (m == NULL) return NULL;

    // init numpy
    import_array();
    //
    // PyObject_SetAttrString(m, "dt_to_timestamp",
    //                         PyLong_FromVoidPtr((void*)(&dt_to_timestamp)));

    PyObject_SetAttrString(
        m, "parse_iso_8601_datetime",
        PyLong_FromVoidPtr((void *)(&parse_iso_8601_datetime)));
    PyObject_SetAttrString(
        m, "convert_datetimestruct_to_datetime",
        PyLong_FromVoidPtr((void *)(&convert_datetimestruct_to_datetime)));

    return m;
}

}  // extern "C"
