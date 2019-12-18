#ifndef _BODO_FILE_READER_H_INCLUDED
#define _BODO_FILE_READER_H_INCLUDED
#include <cstdint>


// File reader abstraction to enable pluging in multiple data sources and file formats
class FileReader
{
public:
    const char *fname;
    FileReader(const char *_fname) : fname(_fname) {}
    virtual uint64_t getSize() = 0;
    virtual bool seek(int64_t pos) = 0;
    virtual bool ok() = 0;
    virtual bool read(char *s, int64_t size) = 0;
};

#endif  // _BODO_FILE_READER_H_INCLUDED
