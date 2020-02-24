// Copyright (C) 2019 Bodo Inc. All rights reserved.
/**
 * @file _common_defs.h
 * @author Mathieu Dutour Sikiric (mathieu@bodo.ai)
 * @brief Tool for printing error messages to std::cout before
 *    doing the PyErr_SetString.
 *    The reasoning is that once the PyErr_SetString is set, Python
 *    will crash but the error message will not be printed.
 * @date 20 February 2020
 */

#include <Python.h>
#include<iostream>

void Bodo_PyErr_SetString(PyObject *type, const char *message)
{
  std::cerr << "BodoRuntimeCppError, setting PyErr_SetString to " << message << "\n";
  PyErr_SetString(type, message);
}
