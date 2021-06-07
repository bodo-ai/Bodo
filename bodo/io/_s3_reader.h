// Copyright (C) 2019 Bodo Inc. All rights reserved.

#include <memory>

#include <arrow/io/api.h>

void s3_open_file(const char *fname,
                  std::shared_ptr<::arrow::io::RandomAccessFile> *file,
                  const char *bucket_region, bool anonymous);
