
#include <memory>

#include <arrow/io/interfaces.h>

void hdfs_open_file(const char *fname,
                    std::shared_ptr<::arrow::io::RandomAccessFile> *file);
