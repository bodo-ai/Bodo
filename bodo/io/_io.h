// Copyright (C) 2019 Bodo Inc. All rights reserved.
//-----------------------------------------------------------------------------
// functions here are used in np.tofile() and np.fromfile()
// file_write_parallel is also used for df.to_csv() distributed writing to posix

#ifndef _IO_H_INCLUDED
#define _IO_H_INCLUDED
#include <cstdint>

extern "C" {
uint64_t get_file_size(char* file_name);
void file_read(char* file_name, void* buff, int64_t size);
void file_write(char* file_name, void* buff, int64_t size);
void file_read_parallel(char* file_name, char* buff, int64_t start,
                        int64_t count);
void file_write_parallel(char* file_name, char* buff, int64_t start,
                         int64_t count, int64_t elem_size);
}  // extern "C"

#endif  // _IO_H_INCLUDED
