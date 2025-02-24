
#include <memory>

#include <arrow/io/interfaces.h>

void s3_open_file(const char *fname,
                  std::shared_ptr<::arrow::io::RandomAccessFile> *file,
                  const char *bucket_region, bool anonymous);
