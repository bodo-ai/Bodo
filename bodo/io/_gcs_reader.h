// Copyright (C) 2021 Bodo Inc. All rights reserved.
#include <arrow/io/api.h>

void gcs_open_file(const char *fname,
                   std::shared_ptr<::arrow::io::RandomAccessFile> *file);

void fsspec_open_file(std::string fname,
                      std::shared_ptr<::arrow::io::RandomAccessFile> *file);
