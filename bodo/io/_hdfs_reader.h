// Copyright (C) 2021 Bodo Inc. All rights reserved.

#include <memory>

#include <arrow/io/api.h>

void hdfs_open_file(const char *fname,
                    std::shared_ptr<::arrow::io::RandomAccessFile> *file);
