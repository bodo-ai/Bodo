// Copyright (C) 2021 Bodo Inc. All rights reserved.
#include <Python.h>
#include <string>
#include <unordered_map>

#include <arrow/io/api.h>

extern std::unordered_map<std::string, PyObject *> pyfs;

void fsspec_open_file(const std::string &fname, const std::string &protocol,
                      std::shared_ptr<::arrow::io::RandomAccessFile> *file);
