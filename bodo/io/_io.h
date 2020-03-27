// Copyright (C) 2019 Bodo Inc. All rights reserved.
#ifndef _IO_H_INCLUDED
#define _IO_H_INCLUDED
#include <cstdint>

extern "C" {
void file_write_parallel(char* file_name, char* buff, int64_t start,
                         int64_t count, int64_t elem_size);
void file_write(char* file_name, void* buff, int64_t size);
}  // extern "C"

#endif  // _IO_H_INCLUDED
