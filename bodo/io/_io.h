//-----------------------------------------------------------------------------
// functions here are used in np.tofile() and np.fromfile()
// file_write_parallel is also used for df.to_csv() distributed writing to posix

#pragma once

#include <cstdint>

extern "C" {
uint64_t get_file_size(const char* file_name);
void file_read(const char* file_name, void* buff, int64_t size, int64_t offset);
void file_write(const char* file_name, void* buff, int64_t size);
void file_write_py_entrypt(const char* file_name, void* buff, int64_t size);
void file_read_parallel(const char* file_name, char* buff, int64_t start,
                        int64_t count);
void file_write_parallel(const char* file_name, char* buff, int64_t start,
                         int64_t count, int64_t elem_size);
void file_write_parallel_py_entrypt(char* file_name, char* buff, int64_t start,
                                    int64_t count, int64_t elem_size);
}  // extern "C"
