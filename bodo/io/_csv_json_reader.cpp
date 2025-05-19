/*
  SPMD Stream, CSV reader, and JSON reader.

  We provide a Python object that is file-like in the Pandas sense
  and can so be used as the input argument to pandas CSV & JSON read.
  When called in a parallel/distributed setup, each process owns a
  chunk of the csv/json file only. The chunks are balanced by number of
  rows of DataFrame (not necessarily number of bytes), determined by number
  of lines(csv & json(orient = 'records', lines=True)) or
  number of objects (json(orient = 'records', lines=False)) . The actual file
  read is done lazily in the objects read method.
*/
#include "_csv_json_reader.h"
#include <Python.h>
#include <mpi.h>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <arrow/filesystem/filesystem.h>
#include <arrow/io/compressed.h>
#include <arrow/memory_pool.h>
#include <arrow/util/compression.h>

#include "../libs/_distributed.h"
#include "../libs/_stl_allocator.h"
#include "_bodo_file_reader.h"

#include "csv_json_reader.h"

// lines argument of read_json(lines = json_lines)
// when json_lines=true, we are reading Json line format where each
// DataFrame row/json record takes up exactly one line, ended with '\n'
// when json_lines=false. we are reading a multi line json format where each
// DataFrame row/json record takes up more than one lines, separated by '},'

#undef CHECK
#define CHECK(expr, msg)                                    \
    if (!(expr)) {                                          \
        std::cerr << "Error in read: " << msg << std::endl; \
    }

// if status of arrow::Result is not ok, form an err msg and raise a
// runtime_error with it
#define CHECK_ARROW(res, funcname, msg)                                        \
    if (!(res.status().ok())) {                                                \
        std::string err_msg = std::string("Error in _csv_json_reader.cpp::") + \
                              funcname + ": " + msg + " " +                    \
                              res.status().ToString();                         \
        throw std::runtime_error(err_msg);                                     \
    }

// In cases where only one process is executing code that might crash, we use
// a slightly different strategy than raising errors directly as in the macro
// above, since in these cases that would cause hangs. In these cases, we
// instead use the macro below to set a bool variable true if an error must be
// raised, and form an appropriate err msg and set it to a passed in std::string
// variable.
#define CHECK_ARROW_ONE_PROC(res, funcname, msg, raise_err_bool, err_msg_var) \
    if (!(res.status().ok())) {                                               \
        err_msg_var = std::string("Error in _csv_json_reader.cpp::") +        \
                      funcname + ": " + msg + " " + res.status().ToString();  \
        raise_err_bool = true;                                                \
    }

// It is expected that the user will then check this boolean and immediately
// go to a part of code that is run by all processes. We then use the macro
// below to bcast this err bool and err msg and raise a runtime error (if
// needed) with this err msg on all the processes, ensuring there wouldn't be
// hangs.

#define ARROW_ONE_PROC_ERR_SYNC_USING_MPI(raise_err_bool, err_msg_var)       \
    CHECK_MPI(MPI_Bcast(&raise_err_bool, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD),  \
              "ARROW_ONE_PROC_ERR_SYNC_USING_MPI: MPI error on MPI_Bcast:"); \
    if (raise_err_bool) {                                                    \
        int err_msg_size = err_msg_var.size();                               \
        CHECK_MPI(                                                           \
            MPI_Bcast(&err_msg_size, 1, MPI_INT, 0, MPI_COMM_WORLD),         \
            "ARROW_ONE_PROC_ERR_SYNC_USING_MPI: MPI error on MPI_Bcast:");   \
        err_msg_var.resize(err_msg_size);                                    \
        CHECK_MPI(                                                           \
            MPI_Bcast(const_cast<char *>(err_msg_var.data()), err_msg_size,  \
                      MPI_CHAR, 0, MPI_COMM_WORLD),                          \
            "ARROW_ONE_PROC_ERR_SYNC_USING_MPI: MPI error on MPI_Bcast:");   \
        throw std::runtime_error(err_msg_var);                               \
    }

// Copied from _bodo_common.cpp to avoid importing it here
void Bodo_PyErr_SetString(PyObject *type, const char *message) {
    PyErr_SetString(type, message);
    throw std::runtime_error(message);
}

// Buffer pool pointer that points to the central buffer pool from the main
// module. Necessary since csv_json_reader is a separate module from bodo.ext.
static arrow::MemoryPool *memory_pool = nullptr;

void init_buffer_pool_ptr(int64_t buffer_pool_ptr) {
    memory_pool = reinterpret_cast<arrow::MemoryPool *>(buffer_pool_ptr);
}

// STL allocator that plugs in the central buffer pool from memory_pool above.
// Cannot use DefaultSTLBufferPoolAllocator since it's defined in the bodo.ext
// module.
template <class T>
class ReaderSTLBufferPoolAllocator : public bodo::STLBufferPoolAllocator<T> {
   public:
    template <class U>
    struct rebind {
        using other = ReaderSTLBufferPoolAllocator<U>;
    };

    template <class U>
    ReaderSTLBufferPoolAllocator(
        const ReaderSTLBufferPoolAllocator<U> &other) noexcept
        : ReaderSTLBufferPoolAllocator(other.pool(), other.size()) {}

    template <class U>
    ReaderSTLBufferPoolAllocator(ReaderSTLBufferPoolAllocator<U> &&other)
        : ReaderSTLBufferPoolAllocator(other.pool(), other.size()) {}

    template <class U>
    ReaderSTLBufferPoolAllocator &operator=(
        const ReaderSTLBufferPoolAllocator<U> &other) {
        this->pool_ = other.pool();
        this->size_ = other.size();
    }

    template <class U>
    ReaderSTLBufferPoolAllocator &operator=(
        ReaderSTLBufferPoolAllocator<U> &&other) {
        this->pool_ = other.pool();
        this->size_ = other.size();
    }

    ReaderSTLBufferPoolAllocator(arrow::MemoryPool *pool, size_t size) noexcept
        : bodo::STLBufferPoolAllocator<T>(pool, size) {}

    ReaderSTLBufferPoolAllocator(arrow::MemoryPool *pool) noexcept
        : ReaderSTLBufferPoolAllocator(pool, 0) {}

    ReaderSTLBufferPoolAllocator() noexcept
        : bodo::STLBufferPoolAllocator<T>(memory_pool) {}
};

// std vector with buffer pool allocator plugged in
template <typename T, class Allocator = ReaderSTLBufferPoolAllocator<T>>
using bodo_vector = std::vector<T, Allocator>;

/**
 * A SkiprowsListInfo object collects information about the skiprows with list
 * to avoid reading certain rows. It's needed when low_memory path is used.
 * With skipping rows, the read operation is not contiguous anymore and the data
 * to read is split into sections. This object stores start/end offsets to read
 * for each section and
 * - Flag whether it's a list or just one element
 * - Length of the list
 * - skiprows list values
 * In addition, since the read of the file can happen in chunks
 * (if file is too large or chunksize is used), we need to keep track of
 * - current index of skiprows list
 * - current index of how many rows are scanned so far.
 */
class SkiprowsListInfo {
   public:
    SkiprowsListInfo(bool is_list, int64_t len, int64_t *_skiprows)
        : is_skiprows_list(is_list),
          num_skiprows(len),
          skiprows(_skiprows, _skiprows + len) {
        // add 1 to account for last section to read.
        // e.g with 1 row to skip, read from start to beginning of skipped row
        // next section from after skipped rows to the end.
        read_start_offset.resize(len + 1);
        read_end_offset.resize(len + 1);
    }
    ~SkiprowsListInfo() = default;

    bool is_skiprows_list;
    int64_t num_skiprows;
    std::vector<int64_t> skiprows;
    int64_t skiprows_list_idx = 0;
    int64_t rows_read = 0;
    std::vector<int64_t> read_start_offset;
    std::vector<int64_t> read_end_offset;
};

// ***********************************************************************************
// C interface for getting the file-like chunk reader
// ***********************************************************************************

// ----------------------------------------------------------------------------
/**
 * A PathInfo object obtains and stores information about the csv/json files
 * to be read, including:
 * - Compression scheme when inferred from file name (extension)
 * - Determining if path is a single file or a directory
 * - Getting list of files to read and their sizes
 * All of this information is obtained only on rank 0 and broadcasted to other
 * ranks (to reduce requests on filesystem, especially remote filesystems).
 */
class PathInfo {
   public:
    /**
     * @param file_path : path passed in pd.read_csv/pd.read_json call
     */
    PathInfo(const char *file_path, const std::string &compression_pyarg,
             const char *bucket_region, bool is_anon) {
        get_read_path_info(file_path, compression_pyarg, is_anon,
                           this->is_remote_fs, this->compression,
                           this->file_names, this->file_sizes, this->fs);

        if (PyErr_Occurred()) {
            is_valid = false;
            return;
        }

        /// sum of all file sizes
        this->total_ds_size =
            std::reduce(this->file_sizes.begin(), this->file_sizes.end());
        ;
    }
    ~PathInfo() = default;

    /// get the compression scheme used by this file(s)
    const std::string &get_compression_scheme() const { return compression; }

    /**
     * Get file names for this path.
     * If the path is a directory, it only considers files of size greater than
     * zero which are not one of the excluded paths (like hidden files)
     */
    const std::vector<std::string> &get_file_names() const {
        return file_names;
    }

    /**
     * Get file sizes for this path. Item in position i corresponds
     * to the size of file name in position i returned by get_file_names.
     * If the path is a directory, it only considers files of size greater than
     * zero which are not one of the excluded paths (like hidden files)
     */
    const std::vector<int64_t> &get_file_sizes() const { return file_sizes; }

    /**
     * Return name of first file in path. If path is a single file then just
     * return the name of that file.
     */
    const std::string get_first_file() const { return file_names[0]; }

    /**
     * Return total size of all files in path.
     */
    int64_t get_size() const { return total_ds_size; }

    /**
     * Return if the filesystem being used is a remote-filesystem.
     * At the moment we only support S3 and HDFS.
     */
    bool get_is_remote_fs() const { return this->is_remote_fs; }

    /**
     * Get arrow::fs::FileSystem object necessary to read data from this path.
     */
    std::shared_ptr<arrow::fs::FileSystem> get_fs() { return this->fs; }

    /**
     * @brief return whether the provided path is valid or not
     *
     * @return true
     * @return false
     */
    bool is_path_valid() { return is_valid; }

   private:
    bool is_valid = true;
    bool is_remote_fs = false;  // like S3 or HDFS
    std::string compression = "UNKNOWN";
    std::vector<std::string> file_names;
    std::vector<int64_t> file_sizes;
    std::shared_ptr<arrow::fs::FileSystem> fs;
    /// sum of all file sizes
    int64_t total_ds_size = -1;
};

// ***********************************************************************************
// Our file-like object for reading chunks in a std::istream
// ***********************************************************************************

using stream_reader = struct {
    PyObject_HEAD
        /* Your internal buffer, size and pos */
        FileReader *ifs;  // input stream
    size_t chunk_start;   // start of our chunk
    size_t chunk_size;    // size of our chunk
    size_t chunk_pos;     // current position in our chunk

    std::shared_ptr<bodo_vector<char>>
        buf;  // internal buffer for converting stream input to Unicode object

    // The following attributes are needed for chunksize Iterator support
    bool first_read;     // Flag used by iterator to track if the first read was
                         // performed.
    int64_t first_pos;   // global first offset byte for the first chunk. Use to
                         // account for skipped bytes if skiprows is used
                         // (excluding header_size_bytes)
    int64_t global_end;  // global end offset byte for the current chunk
    int64_t g_total_bytes;      // Total number of bytes for the whole file(s)
    int64_t chunksize_rows;     // chunksize (num. of rows per chunk for chunk
                                // iterator)
    class PathInfo *path_info;  // PathInfo struct
    bool is_parallel;  // Flag to say whether data is distributed or replicated.
    int64_t header_size_bytes;  // store header size to avoid computing
                                // for every chunk/iterator read
    // needed for use of skiprows as list with chunksize
    class SkiprowsListInfo *skiprows_list_info;
    bool csv_header;  // whether file(s) has header or not (needed for knowing
                      // whether skiprows are 1-based or 0-based indexing)
};

static void stream_reader_dealloc(stream_reader *self) {
    // we own the stream!
    if (self->ifs) {
        delete self->ifs;
    }
    if (self->path_info) {
        delete self->path_info;
    }
    if (self->skiprows_list_info) {
        delete self->skiprows_list_info;
    }
    Py_TYPE(self)->tp_free(self);
}

// Needed to return size of data to Python code
static PyObject *stream_reader_get_chunk_size(stream_reader *self) {
    return PyLong_FromSsize_t(self->chunk_size);
}

// Needed to return parallel value to Python code
static PyObject *stream_reader_is_parallel(stream_reader *self) {
    return PyBool_FromLong(self->is_parallel);
}

// alloc a HPTAIO object
static PyObject *stream_reader_new(PyTypeObject *type, PyObject *args,
                                   PyObject *kwds) {
    stream_reader *self = (stream_reader *)type->tp_alloc(type, 0);
    if (PyErr_Occurred()) {
        PyErr_Print();
        return nullptr;
    }
    self->ifs = nullptr;
    self->chunk_start = 0;
    self->chunk_size = 0;
    self->chunk_pos = 0;
    self->first_pos = 0;
    self->global_end = 0;
    self->g_total_bytes = 0;
    self->chunksize_rows = 0;
    self->is_parallel = false;
    self->header_size_bytes = 0;
    self->path_info = nullptr;
    self->first_read = false;
    self->buf = std::make_shared<bodo_vector<char>>(0);

    return (PyObject *)self;
}
// we provide this mostly for testing purposes
// users are not supposed to use this
static int stream_reader_pyinit(PyObject *self, PyObject *args,
                                PyObject *kwds) {
    // char* str = NULL;
    // Py_ssize_t count = 0;

    // if(!PyArg_ParseTuple(args, "|z#", &str, &count) || str == NULL) {
    //     if(PyErr_Occurred()) PyErr_Print();
    //     return 0;
    // }

    // ((stream_reader*)self)->chunk_start = 0;
    // ((stream_reader*)self)->chunk_pos = 0;
    // ((stream_reader*)self)->ifs = new std::istringstream(str);
    // if(!((stream_reader*)self)->ifs->good()) {
    //     std::cerr << "Could not create istrstream from string.\n";
    //     ((stream_reader*)self)->chunk_size = 0;
    //     return -1;
    // }
    // ((stream_reader*)self)->chunk_size = count;

    return 0;
}

/**
 * We use this (and not the above) from C to init our StreamReader object
 * Will seek to chunk beginning and store metadata needed for chunksize_iterator
 * reads
 * @param[out] self: stream_reader wrapper used by Pandas to read data
 * @param[in] ifs: MemReader that has the data
 * @param[in] start: start reading position
 * @param[in] sz: size of the data read (in bytes)
 * These are used only with chunksize mode
 * @param[in] skipped_bytes: number of bytes skipped (if using skiprows)
 * @param[in] global_end: end of current chunk across all ranks
 * @param[in] g_total_bytes: size of total data to read
 * @param[in] chunksize: number of rows per chunk
 * @param[in] is_parallel: whether data is distributed or replicated
 * @param[in] path_info: information about file(s) (file_names, file_sizes, ...)
 * @param[in] header_size_bytes: number of bytes in the header
 */
static void stream_reader_init(stream_reader *self, FileReader *ifs,
                               size_t start, size_t sz, int64_t skipped_bytes,
                               int64_t global_end, int64_t g_total_bytes,
                               int64_t chunksize, bool is_parallel,
                               PathInfo *path_info, int64_t header_size_bytes,
                               SkiprowsListInfo *skiprows_list_info,
                               bool csv_header) {
    if (!ifs) {
        std::cerr << "Can't handle NULL pointer as input stream.\n";
        return;
    }
    self->ifs = ifs;
    if (!self->ifs->ok()) {
        std::cerr << "Got bad istream in initializing StreamReader object."
                  << std::endl;
        return;
    }
    // seek to our chunk beginning
    // only if sz > 0
    bool ok = true;
    if (sz > 0) {
        ok = self->ifs->seek(start);
    }
    if (!ok) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "Could not seek to start position");
        return;
    }
    self->chunk_start = start;
    self->chunk_size = sz;
    self->chunk_pos = 0;
    self->first_pos = skipped_bytes;
    self->global_end = global_end;
    self->g_total_bytes = g_total_bytes;
    self->chunksize_rows = chunksize;
    self->is_parallel = is_parallel;
    self->path_info = path_info;
    self->header_size_bytes = header_size_bytes;
    self->skiprows_list_info = skiprows_list_info;
    self->csv_header = csv_header;
}

// read given number of bytes from our chunk and return a Unicode Object
// returns NULL if an error occured.
// does not read beyond end of our chunk (even if file continues)
static PyObject *stream_reader_read(stream_reader *self, PyObject *args) {
    // partially copied from from CPython's stringio.c
    if (self->ifs == nullptr) {
        PyErr_SetString(PyExc_ValueError,
                        "I/O operation on uninitialized StreamReader object");
        return nullptr;
    }
    Py_ssize_t size, n;

    PyObject *arg = Py_None;
    if (!PyArg_ParseTuple(args, "|O:read", &arg)) {
        return nullptr;
    }
    if (PyNumber_Check(arg)) {
        size = PyNumber_AsSsize_t(arg, PyExc_OverflowError);
        if (size == -1 && PyErr_Occurred()) {
            return nullptr;
        }
    } else if (arg == Py_None) {
        /* Read until EOF is reached, by default. */
        size = -1;
    } else {
        PyErr_Format(PyExc_TypeError, "integer argument expected, got '%s'",
                     Py_TYPE(arg)->tp_name);
        return nullptr;
    }
    /* adjust invalid sizes */
    n = self->chunk_size - self->chunk_pos;
    if (size < 0 || size > n) {
        size = n;
        if (size < 0) {
            size = 0;
        }
    }

    self->buf->resize(size);
    bool ok = self->ifs->read(self->buf->data(), size);
    self->chunk_pos += size;
    if (!ok) {
        std::cerr << "Failed reading " << size << " bytes" << std::endl;
        return nullptr;
    }
    // buffer_rd_bytes() function of pandas expects a Bytes object
    // using PyUnicode_FromStringAndSize is wrong since 'size'
    // may end up in the middle a multi-byte UTF-8 character
    return PyBytes_FromStringAndSize(self->buf->data(), size);
}

// Needed to make Pandas accept it, never used
static PyObject *stream_reader_iternext(PyObject *self) {
    std::cerr << "iternext not implemented";
    return nullptr;
};

// Update stream_reader with next chunk data and update its metadata
// returns false if EOF
static bool stream_reader_update_reader(stream_reader *self);

static PyMethodDef stream_reader_methods[] = {
    {
        "read",
        (PyCFunction)stream_reader_read,
        METH_VARARGS,
        "Read at most n characters, returned as a unicode.",
    },
    {
        "get_chunk_size",
        (PyCFunction)stream_reader_get_chunk_size,
        METH_VARARGS,
        "Return size of the chunk in bytes.",
    },
    {
        "is_parallel",
        (PyCFunction)stream_reader_is_parallel,
        METH_VARARGS,
        "Return the stored parallel flag for the reader.",
    },
    {
        "update_reader",
        (PyCFunction)stream_reader_update_reader,
        METH_VARARGS,
        "Update reader with next chunk info.",
    },
    {nullptr} /* Sentinel */
};

// the actual Python type class
static PyTypeObject stream_reader_type = {
    PyVarObject_HEAD_INIT(NULL, 0) "bodo.libs.StreamReader", /*tp_name*/
    sizeof(stream_reader),                                   /*tp_basicsize*/
    0,                                                       /*tp_itemsize*/
    (destructor)stream_reader_dealloc,                       /*tp_dealloc*/
    0,                                                       /*tp_print*/
    nullptr,                                                 /*tp_getattr*/
    nullptr,                                                 /*tp_setattr*/
    nullptr,                                                 /*tp_compare*/
    nullptr,                                                 /*tp_repr*/
    nullptr,                                                 /*tp_as_number*/
    nullptr,                                                 /*tp_as_sequence*/
    nullptr,                                                 /*tp_as_mapping*/
    nullptr,                                                 /*tp_hash */
    nullptr,                                                 /*tp_call*/
    nullptr,                                                 /*tp_str*/
    nullptr,                                                 /*tp_getattro*/
    nullptr,                                                 /*tp_setattro*/
    nullptr,                                                 /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,                /*tp_flags*/
    "stream_reader objects",                                 /* tp_doc */
    nullptr,                                                 /* tp_traverse */
    nullptr,                                                 /* tp_clear */
    nullptr,                /* tp_richcompare */
    0,                      /* tp_weaklistoffset */
    stream_reader_iternext, /* tp_iter */
    stream_reader_iternext, /* tp_iternext */
    stream_reader_methods,  /* tp_methods */
    nullptr,                /* tp_members */
    nullptr,                /* tp_getset */
    nullptr,                /* tp_base */
    nullptr,                /* tp_dict */
    nullptr,                /* tp_descr_get */
    nullptr,                /* tp_descr_set */
    0,                      /* tp_dictoffset */
    stream_reader_pyinit,   /* tp_init */
    nullptr,                /* tp_alloc */
    stream_reader_new,      /* tp_new */
};

void init_stream_reader_type() {
    if (PyType_Ready(&stream_reader_type) < 0) {
        PyErr_SetString(PyExc_RuntimeError,
                        "stream_reader_type is not initialized properly");
    }
}

/**
 * Get size of header in bytes. Header is understood to be the first row of the
 * file (up to and including the first row_separator). This is only used for
 * CSV right now.
 * The filesystem is accessed only on rank 0, result is communicated to other
 * processes using MPI.
 */
int64_t get_header_size(const std::string &fname, int64_t file_size,
                        PathInfo &path_info, char row_separator) {
    int64_t header_size = 0;

    bool raise_runtime_err = false;
    std::string runtime_err_msg = "";
    if (dist_get_rank() == 0) {
        std::shared_ptr<arrow::fs::FileSystem> fs = path_info.get_fs();
        arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>>
            file_result = fs->OpenInputFile(fname);
        CHECK_ARROW_ONE_PROC(file_result, "get_header_size",
                             "fs->OpenInputFile", raise_runtime_err,
                             runtime_err_msg);
        if (!raise_runtime_err) {
            std::shared_ptr<arrow::io::RandomAccessFile> file =
                file_result.ValueOrDie();
#define BUF_SIZE 1024
            std::vector<char> data(BUF_SIZE);
#undef BUF_SIZE
            bool header_found = false;
            int64_t seen_size = 0;
            while (!header_found && seen_size < file_size) {
                // TODO check status
                int64_t read_size =
                    std::min(int64_t(data.size()), file_size - seen_size);
                arrow::Result<int64_t> res = file->Read(read_size, data.data());
                for (int64_t i = 0; i < read_size; i++, header_size++) {
                    if (data[i] == row_separator) {
                        header_found = true;
                        break;
                    }
                }
                seen_size += read_size;
            }
            // TODO error if !header_found
        }
    }

    ARROW_ONE_PROC_ERR_SYNC_USING_MPI(raise_runtime_err, runtime_err_msg);

    header_size += 1;
    CHECK_MPI(MPI_Bcast(&header_size, 1, MPI_INT64_T, 0, MPI_COMM_WORLD),
              "get_header_size: MPI error on MPI_Bcast:");
    return header_size;
}

/**
 * This FileReader is intended to be passed to pandas in order to parse
 * CSV/JSON text content that is stored in memory.
 * The MemReader can read (un)compressed csv/json into memory and essentially
 * just stores csv/json data.
 */
class MemReader : public FileReader {
   public:
    /// starting offset of data
    int64_t start = 0;
    /// current read position
    int64_t pos = start;
    /// data stored by this MemReader
    bodo_vector<char> data;
    /// character that constitutes the row separator for this data
    char row_separator;
    /// true if the content refers to JSON where records span multiple lines
    bool json_multi_line = false;
    /// starting offset of each row
    bodo_vector<int64_t> row_offsets;
    /// FileReader status
    bool status_ok = true;

    /**
     * @param row_separator : character that constitutes the row separator
     * @param json_multi_line : true if the content refers to JSON where
     *                          records span multiple lines
     */
    MemReader(char row_separator, bool json_multi_line = false)
        // note that none of the parameters passed to super class affect
        // MemReader
        : FileReader("", false, !json_multi_line),
          row_separator(row_separator),
          json_multi_line(json_multi_line) {}

    /**
     * @param size : total size (in bytes) to reserve for data
     * @param row_separator : character that constitutes the row separator
     * @param json_multi_line : true if the content refers to JSON where
     *                          records span multiple lines
     */
    MemReader(size_t size, char row_separator, bool json_multi_line = false)
        // note that none of the parameters passed to super class affect
        // MemReader
        : FileReader("", false, !json_multi_line),
          row_separator(row_separator),
          json_multi_line(json_multi_line) {
        data.reserve(size);
    }

    ~MemReader() override = default;

    /**
     * Return total size of data.
     */
    uint64_t getSize() override { return data.size() - start; }

    /**
     * Read size bytes into given buffer s (from current position)
     */
    bool read(char *s, int64_t size) override {
        memcpy(s, data.data() + pos, size);
        pos += size;
        return true;
    };

    /**
     * Seek pos_req bytes into data.
     */
    bool seek(int64_t pos_req) override {
        pos = pos_req + start;
        status_ok = pos >= start && pos < int64_t(data.size());
        return status_ok;
    }

    /**
     * Returns reader status.
     */
    bool ok() override { return status_ok; }

    /// not used by MemReader
    bool read_to_buff(char *s, int64_t size) { return false; }

    /// not used by MemReader
    bool skipHeaderRows() override { return this->csv_header; };

    /**
     * Calculate row offsets for current data (fills row_offsets attribute).
     */
    void calc_row_offsets() {
        row_offsets.clear();
        row_offsets.push_back(start);  // first row starts at 'start'
        if (data.size() > 0) {
            for (int64_t i = start; i < int64_t(data.size()); i++) {
                if (data[i] == row_separator) {
                    if (i > start && data[i - 1] == row_separator) {
                        // ignore empty row
                        continue;
                    }
                    row_offsets.push_back(i + 1);  // next row starts at i + 1
                }
            }
        }
    }

    /**
     * Get number of rows in data.
     */
    int64_t get_num_rows() { return row_offsets.size() - 1; }

    /**
     * Replace data with new_data (MemReader takes ownership of it).
     */
    void set_data(bodo_vector<char> &new_data) {
        data = std::move(new_data);
        start = pos = 0;
        row_offsets.clear();
    }

    /**
     * Read portion of uncompressed file.
     * @param fname : file name
     * @param file_start : read starting position
     * @param file_end : read end position
     * @param fs : File system to read from
     */
    void read_uncompressed_file(const std::string &fname, size_t file_start,
                                size_t file_end,
                                std::shared_ptr<arrow::fs::FileSystem> fs) {
        int64_t read_size = file_end - file_start;
        size_t cur_data_size = data.size();
        data.resize(cur_data_size + read_size);
        arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>>
            file_result = fs->OpenInputFile(fname);
        CHECK_ARROW(file_result, "read_uncompressed_file", "fs->OpenInputFile")
        std::shared_ptr<arrow::io::RandomAccessFile> file =
            file_result.ValueOrDie();
        arrow::Status status;
        status = file->Seek(file_start);
        // TODO check status
        arrow::Result<int64_t> res =
            file->Read(read_size, data.data() + cur_data_size);
    }

    /**
     * Read whole compressed file (decompressed into memory).
     * @param fname : file name
     * @param fs : File system to read from
     * @param compression : compression scheme used (Arrow string
     *                      representation, see arrow/util/compression.h/cc)
     * @param skip_header : true if want to skip header (will skip first row)
     */
    void read_compressed_file(const std::string &fname,
                              std::shared_ptr<arrow::fs::FileSystem> fs,
                              const std::string &compression,
                              bool skip_header) {
        arrow::Result<std::shared_ptr<arrow::io::InputStream>>
            raw_istream_result = fs->OpenInputStream(fname);
        CHECK_ARROW(raw_istream_result, "read_compressed_file",
                    "fs->OpenInputStream")
        std::shared_ptr<arrow::io::InputStream> raw_istream =
            raw_istream_result.ValueOrDie();

        arrow::Result<arrow::Compression::type> compression_type_result =
            arrow::util::Codec::GetCompressionType(compression);
        CHECK_ARROW(compression_type_result, "read_compressed_file",
                    "arrow::util::Codec::GetCompressionType")
        arrow::Compression::type compression_type =
            std::move(compression_type_result).ValueOrDie();

        arrow::Result<std::unique_ptr<arrow::util::Codec>> codec_result =
            arrow::util::Codec::Create(compression_type);
        CHECK_ARROW(codec_result, "read_compressed_file",
                    "arrow::util::Codec::Create")
        std::unique_ptr<arrow::util::Codec> codec =
            std::move(codec_result).ValueOrDie();

        arrow::Result<std::shared_ptr<arrow::io::CompressedInputStream>>
            istream_result = arrow::io::CompressedInputStream::Make(
                codec.get(), raw_istream, memory_pool);
        CHECK_ARROW(istream_result, "read_compressed_file",
                    "arrow::io::CompressedInputStream::Make")
        std::shared_ptr<arrow::io::CompressedInputStream> istream =
            std::move(istream_result).ValueOrDie();
#define READ_SIZE 8192
        int64_t actual_size = int64_t(data.size());
        bool skipped_header = !skip_header;
        while (true) {
            // read a chunk of READ_SIZE bytes
            data.resize(data.size() + READ_SIZE);
            arrow::Result<int64_t> bytes_read_result =
                istream->Read(READ_SIZE, data.data() + actual_size);
            CHECK_ARROW(bytes_read_result, "read_compressed_file",
                        "istream->Read")
            int64_t bytes_read = bytes_read_result.ValueOrDie();
            if (bytes_read == 0) {
                break;
            }
            if (!skipped_header) {
                // check for row_separator in new data read
                for (int64_t i = actual_size; i < actual_size + bytes_read;
                     i++) {
                    if (data[i] == row_separator) {
                        // found row separator, move data after it to position 0
                        size_t good_size = actual_size + bytes_read - i - 1;
                        memmove(data.data() + actual_size, data.data() + i + 1,
                                good_size);
                        actual_size += good_size;
                        skipped_header = true;
                        bytes_read = 0;
                        break;
                    }
                }
            }
            actual_size += bytes_read;
        }
        data.resize(actual_size);
    }
#undef READ_SIZE

    /**
     * Do some post-processing of data before passing to pandas.
     * Currently this makes sure that, for JSON multi line record file, data
     * correctly starts with '[' and ends with ']'
     */
    void finalize() {
        if (json_multi_line) {
            if (data.size() == 0) {
                data.push_back('[');
                data.push_back(']');
                data.push_back('\n');
                start = pos = 0;
            } else {
                // every rank must have a self-contained piece of JSON content
                // that starts with [ and ends with ]
                // Ranks other than 0 can have '[' or ',' as first character
                // (',' after doing data_row_correction()). in this case, just
                // replace it with '['
                data[start] = '[';
                // ranks can have '}' or ']' as last character. JSON multi-line
                // rows end with }, so ranks will end up with '}' as last
                // character after data_row_correction()
                if (data.back() == '}') {
                    data.push_back(']');
                }
            }
        }
    }
};

/**
 * Corrects data in MemReader of all ranks so that MemReaders contain complete
 * rows.
 * The MemReader can be in any state, but if readers already contain complete
 * rows this function will do unnecessary work and communication.
 * @param reader : MemReader containing data to correct
 * @param row_separator : character delimiting rows
 */
void data_row_correction(MemReader *reader, char row_separator) {
    // A more efficient algorithm is possible (that sends less messages and
    // data) but at the cost of code complexity. This part is not performance
    // critical so we favor a simple algorithm.

    int rank = dist_get_rank();
    int num_ranks = dist_get_size();

    bodo_vector<char> &data = reader->data;
    if (rank < num_ranks - 1) {  // receive chunk from right, append to my data
        size_t cur_data_size = data.size();
        MPI_Status status;
        // probe for incoming message from rank + 1
        CHECK_MPI(MPI_Probe(rank + 1, 0, MPI_COMM_WORLD, &status),
                  "data_row_correction: MPI error on MPI_Probe:");
        // when probe returns, the status object has the message size
        int recv_size;
        CHECK_MPI(MPI_Get_count(&status, MPI_CHAR, &recv_size),
                  "data_row_correction: MPI error on MPI_Get_count:");
        data.resize(cur_data_size + recv_size);
        CHECK_MPI(MPI_Recv(data.data() + cur_data_size, recv_size, MPI_CHAR,
                           rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE),
                  "data_row_correction: MPI error on MPI_Recv:");
    }
    if (rank >
        0) {  // send to left (rank - 1): my data up to the first separator
        int64_t sep_idx = -1;
        for (int64_t i = reader->start; i < int64_t(data.size()); i++) {
            if (data[i] == row_separator) {
                sep_idx = i;
                break;
            }
        }
        if (sep_idx != -1) {
            CHECK_MPI(MPI_Send(data.data(), sep_idx + 1, MPI_CHAR, rank - 1, 0,
                               MPI_COMM_WORLD),
                      "data_row_correction: MPI error on MPI_Send:");
            reader->start = sep_idx + 1;
        } else {
            // I have no separator. Send all my data
            CHECK_MPI(MPI_Send(data.data() + reader->start,
                               data.size() - reader->start, MPI_CHAR, rank - 1,
                               0, MPI_COMM_WORLD),
                      "data_row_correction: MPI error on MPI_Send:");
            data.clear();
            reader->start = 0;
        }
    }
}

/**
 * In case of low_memory=False (i.e. whole dataset is loaded in memory first)
 * Remove skipped rows from the data buffer by finding the rank where
 * the skiprow number lies. Then, compute what is the index of this row in
 * respect to rank's row numbers and copy the following rows over the skipped
 * one. Keep track of number of bytes skipped and at the end resize the data
 * buffer. balance_rows() is called afterwards since number of rows can change
 * in each rank
 * @param[in,out] mem_reader : MemReader whose data we want to modify
 * @param[in] skiprows: list of rows to skip (in global dataset)
 * @param[in] skiprows_list_len: number of rows to skip
 * @param[in] is_parallel: indicates if data is distributed or replicated
 * @param[in] csv_header: true if file contains header
 */
void set_skiprows_list(MemReader *mem_reader, int64_t *skiprows,
                       int64_t skiprows_list_len, bool is_parallel,
                       bool csv_header) {
    tracing::Event ev("set_skiprows_list", is_parallel);
    // we need to know the row offsets to skip rows
    mem_reader->calc_row_offsets();
    int64_t rows_skipped_bytes = 0;
    int64_t skip_row;
    if (is_parallel) {
        int my_rank = dist_get_rank();
        int num_ranks = dist_get_size();
        // first allgather the number of rows on every rank
        int64_t num_rows = mem_reader->get_num_rows();
        std::vector<int64_t> num_rows_ranks(
            num_ranks);  // number of rows in each rank
        CHECK_MPI(
            MPI_Allgather(&num_rows, 1, MPI_INT64_T, num_rows_ranks.data(), 1,
                          MPI_INT64_T, MPI_COMM_WORLD),
            "set_skiprows_list: MPI error on MPI_Allgather:");

        // determine the start/end row number of each rank
        int64_t rank_global_start, rank_global_end;

        for (int rank = 0; rank < num_ranks; rank++) {
            if (rank == my_rank) {
                rank_global_end =
                    std::accumulate(num_rows_ranks.begin(),
                                    num_rows_ranks.begin() + rank + 1, 0) -
                    1;
                rank_global_start = rank_global_end - num_rows + 1;
                // Loop backwards to be able to copy without overriding the next
                // data.
                for (int i = skiprows_list_len - 1; i >= 0; i--) {
                    // skiprows is 1-index based if file contains a header
                    skip_row = skiprows[i] - csv_header;
                    // needed to handle case skiprows=[0] and file has header
                    if (skip_row == -1) {
                        skip_row = 0;
                    }
                    // Finished rows that are in this rank's range
                    if (skip_row < rank_global_start) {
                        break;
                    }
                    if (skip_row >= rank_global_start &&
                        skip_row <= rank_global_end) {
                        // Copy from next row to end on this position
                        int64_t row_pos = skip_row - rank_global_start;
                        if (row_pos < num_rows) {
                            int64_t bytes_to_move =
                                mem_reader->data.size() -
                                mem_reader->row_offsets[row_pos + 1];
                            memmove(mem_reader->data.data() +
                                        mem_reader->row_offsets[row_pos],
                                    mem_reader->data.data() +
                                        mem_reader->row_offsets[row_pos + 1],
                                    bytes_to_move);
                            rows_skipped_bytes +=
                                mem_reader->row_offsets[row_pos + 1] -
                                mem_reader->row_offsets[row_pos];
                        } else {
                            // special case when row to skip is the last row.
                            // No need to copy, just add its size to the total
                            // skipped bytes
                            rows_skipped_bytes +=
                                mem_reader->data.size() -
                                mem_reader->row_offsets[row_pos];
                        }
                    }
                }
            }
        }
    } else {
        int64_t num_rows = mem_reader->get_num_rows();
        for (int i = skiprows_list_len - 1; i >= 0; i--) {
            skip_row = skiprows[i] - csv_header;
            if (skip_row >= num_rows) {
                continue;
            }
            if (skip_row == -1) {
                skip_row = 0;
            }
            if (skip_row < num_rows) {
                int64_t bytes_to_move = mem_reader->data.size() -
                                        mem_reader->row_offsets[skip_row + 1];
                memmove(
                    mem_reader->data.data() + mem_reader->row_offsets[skip_row],
                    mem_reader->data.data() +
                        mem_reader->row_offsets[skip_row + 1],
                    bytes_to_move);
                rows_skipped_bytes += mem_reader->row_offsets[skip_row + 1] -
                                      mem_reader->row_offsets[skip_row];
            } else {
                rows_skipped_bytes +=
                    mem_reader->data.size() - mem_reader->row_offsets[skip_row];
            }
        }
    }
    // Resize
    mem_reader->data.resize(mem_reader->data.size() - rows_skipped_bytes);
}
/**
 * Skip first n rows of the global dataset. Note that if data is distributed,
 * this may require skipping a variable number of rows on multiple ranks.
 * The MemReaders can be in any state for this function to work, but it should
 * only be called once, and rows should be balanced afterwards with
 * balance_rows() because this function can modify a variable number of rows on
 * multiple ranks.
 * @param reader : MemReader whose data we want to modify
 * @param skiprows: number of rows to skip (in global dataset)
 * @param is_parallel: indicates if data is distributed or replicated
 */
void skip_rows(MemReader *reader, int64_t skiprows, bool is_parallel) {
    tracing::Event ev("skip_rows", is_parallel);
    if (skiprows <= 0) {
        return;
    }
    // we need to know the row offsets to skip rows
    reader->calc_row_offsets();
    if (is_parallel) {
        int my_rank = dist_get_rank();
        int num_ranks = dist_get_size();
        // first allgather the number of rows on every rank
        int64_t num_rows = reader->get_num_rows();
        std::vector<int64_t> num_rows_ranks(
            num_ranks);  // number of rows in each rank
        CHECK_MPI(
            MPI_Allgather(&num_rows, 1, MPI_INT64_T, num_rows_ranks.data(), 1,
                          MPI_INT64_T, MPI_COMM_WORLD),
            "skip_rows: MPI error on MPI_Allgather:");

        // determine the number of rows we need to skip on each rank,
        // and modify starting offset of data accordingly
        for (int rank = 0; rank < num_ranks; rank++) {
            int64_t rank_skip = std::min(skiprows, num_rows_ranks[rank]);
            if (rank == my_rank) {
                reader->start = reader->row_offsets[rank_skip];
                return;
            }
            skiprows -= rank_skip;
            if (skiprows == 0) {
                break;
            }
        }
    } else {
        // data is replicated, so skip same number of rows on every rank
        // we just need to modify the starting offset
        reader->start = reader->row_offsets[skiprows];
    }
}

/**
 * Read first n rows of the global dataset. Note that if data is distributed,
 * this may require reading a variable number of rows on multiple ranks.
 * The MemReaders can be in any state for this function to work, but it should
 * only be called once, and rows should be balanced afterwards with
 * balance_rows() because this function can modify a variable number of rows on
 * multiple ranks.
 * @param reader : MemReader whose data we want to modify
 * @param nrows: number of rows to read (in global dataset)
 * @param is_parallel: indicates if data is distributed or replicated
 */
void set_nrows(MemReader *reader, int64_t nrows, bool is_parallel) {
    tracing::Event ev("set_nrows", is_parallel);
    if (nrows <= 0) {
        return;
    }
    // we need to know the row offsets to set specific number of rows to read
    reader->calc_row_offsets();
    if (is_parallel) {
        int my_rank = dist_get_rank();
        int num_ranks = dist_get_size();
        // first allgather the number of rows on every rank
        int64_t num_rows = reader->get_num_rows();
        std::vector<int64_t> num_rows_ranks(
            num_ranks);  // number of rows in each rank
        CHECK_MPI(
            MPI_Allgather(&num_rows, 1, MPI_INT64_T, num_rows_ranks.data(), 1,
                          MPI_INT64_T, MPI_COMM_WORLD),
            "set_nrows: MPI error on MPI_Allgather:");

        // determine the number of rows we need to read on each rank,
        // and modify data accordingly
        for (int rank = 0; rank < num_ranks; rank++) {
            int64_t local_nrows = std::min(nrows, num_rows_ranks[rank]);
            if (rank == my_rank) {
                int64_t total_bytes = reader->row_offsets[local_nrows];
                reader->data.resize(total_bytes);
                return;
            }
            nrows -= local_nrows;
            // Remove data from rest of ranks that has nrows==0
            if (nrows == 0) {
                int64_t total_bytes = 0;
                reader->data.resize(total_bytes);
                break;
            }
        }
    } else {
        // data is replicated, so read same number of rows on every rank
        // we just need to modify the end offset/remove data after end offset
        // use `min` to handle case where user gives nrows larger than total
        // rows in file.
        int64_t updated_rows = std::min(nrows, reader->get_num_rows());
        reader->data.resize(reader->row_offsets[updated_rows]);
        return;
    }
}
/**
 * For balance_rows function below, calculate the number of rows that
 * this rank will send to other ranks.
 * The calculation ensures that each rank has
 * dist_get_node_portion(total_rows, num_ranks, rank) rows, and is calculated
 * in such a way that rank 0 sends rows in order to rank 0,1,... as needed,
 * rank 1 sends rows in order to rank 0,1,... as needed (taking into account
 * what rank 0 sends), and so on.
 * @param num_rows : vector containing the number of rows of every rank
 * @param total_rows : total number of rows in global dataset
 * @return to_send : vector of (rank, num_rows) tuples, meaning send 'num_rows'
 *                   from this rank to 'rank'. Only needs to contain values
 *                   for num_rows > 0 cases.
 */
void calc_row_transfer(const std::vector<int64_t> &num_rows, int64_t total_rows,
                       std::vector<std::pair<int, int64_t>> &to_send) {
    int myrank = dist_get_rank();
    int num_ranks = dist_get_size();

    int64_t start_row_global = 0;  // index of my first row in global dataset
    for (int i = 0; i < myrank; i++) {
        start_row_global += num_rows[i];
    }

    // rows_left tracks how many rows I have left to send
    int64_t rows_left = num_rows[myrank];
    if (rows_left == 0) {
        return;
    }

    using range = std::pair<int64_t, int64_t>;
    // my current row range (in global dataset)
    range my_cur_range(start_row_global, start_row_global + num_rows[myrank]);

    for (int rank = 0; rank < num_ranks; rank++) {
        // get required range for rank 'rank'
        int64_t row_start = dist_get_start(total_rows, num_ranks, rank);
        int64_t row_end = dist_get_end(total_rows, num_ranks, rank);
        range range_other(row_start, row_end);
        // get overlap of my current range with required range for other rank
        range overlap(
            std::max(std::get<0>(my_cur_range), std::get<0>(range_other)),
            std::min(std::get<1>(my_cur_range), std::get<1>(range_other)));
        // the overlap determines how many rows I have to send to other rank
        int64_t rows_to_send = 0;
        if (std::get<0>(overlap) <= std::get<1>(overlap)) {
            rows_to_send = std::get<1>(overlap) - std::get<0>(overlap);
        }
        if (rows_to_send > 0) {
            to_send.emplace_back(rank, rows_to_send);
        }
        rows_left -= rows_to_send;
        if (rows_left == 0) {
            break;
        }
    }
}

/**
 * Redistribute rows across MemReaders of all ranks to ensure that the number
 * of rows is "balanced" (matches dist_get_node_portion(total_rows, num_ranks,
 * rank) for each rank). See calc_row_transfer for details on how row transfer
 * is calculated.
 * IMPORTANT: this assumes that the data in each MemReader consists of complete
 * rows.
 * This can be called multiple times as needed.
 */
void balance_rows(MemReader *reader) {
    int num_ranks = dist_get_size();
    // need the row offsets to balance rows
    reader->calc_row_offsets();

    // first allgather the number of rows on every rank
    int64_t num_rows = reader->get_num_rows();
    std::vector<int64_t> num_rows_ranks(
        num_ranks);  // number of rows in each rank
    CHECK_MPI(MPI_Allgather(&num_rows, 1, MPI_INT64_T, num_rows_ranks.data(), 1,
                            MPI_INT64_T, MPI_COMM_WORLD),
              "balance_rows: MPI error on MPI_Allgather:");

    // check that all ranks have same number of rows. in that case there is no
    // need to do anything
    auto result = std::ranges::minmax_element(num_rows_ranks);
    int64_t min = *result.min;
    int64_t max = *result.max;
    if (min == max) {
        return;  // already balanced
    }

    // get total number of rows in global dataset
    int64_t total_rows = std::accumulate(num_rows_ranks.begin(),
                                         num_rows_ranks.end(), int64_t(0));

    // by default don't send or receive anything. this is changed below as
    // needed
    std::vector<int64_t> sendcounts(num_ranks, 0);
    std::vector<int64_t> recvcounts(num_ranks, 0);
    std::vector<int64_t> sdispls(num_ranks, 0);
    std::vector<int64_t> rdispls(num_ranks, 0);

    // calc send counts
    std::vector<std::pair<int, int64_t>>
        to_send;  // vector of (rank, nrows)  meaning send nrows to rank
    calc_row_transfer(num_rows_ranks, total_rows, to_send);
    int64_t cur_offset = 0;
    int64_t cur_row = 0;
    for (auto rank_rows : to_send) {
        int rank = rank_rows.first;
        int64_t rows = rank_rows.second;
        int64_t num_bytes = 0;
        for (int i = cur_row; i < cur_row + rows; i++) {
            num_bytes += reader->row_offsets[i + 1] - reader->row_offsets[i];
        }
        sendcounts[rank] = num_bytes;
        sdispls[rank] = cur_offset;
        cur_offset += num_bytes;
        cur_row += rows;
    }

    // get recv count
    CHECK_MPI(MPI_Alltoall(sendcounts.data(), 1, MPI_INT64_T, recvcounts.data(),
                           1, MPI_INT64_T, MPI_COMM_WORLD),
              "balance_rows: MPI error on MPI_Alltoall:");

    // have to receive rows from other processes
    int64_t total_recv_size = 0;
    cur_offset = 0;
    for (int rank = 0; rank < num_ranks; rank++) {
        rdispls[rank] = cur_offset;
        cur_offset += recvcounts[rank];
        total_recv_size += recvcounts[rank];
    }
    bodo_vector<char> recvbuf(total_recv_size);
    char *sendbuf = reader->data.data() + reader->start;

    bodo_alltoallv(sendbuf, sendcounts, sdispls, MPI_CHAR, recvbuf.data(),
                   recvcounts, rdispls, MPI_CHAR, MPI_COMM_WORLD);

    reader->set_data(recvbuf);  // mem reader takes ownership of the buffer
}
/**
 * Function to scan portion of data from file and computes
 * global dataset start and end position required to identify actual portion of
 * the file(s) to load into stream read buffer. This is needed when user-code
 * uses skiprows (one element), nrows and/or chunksize.
 * global_offset computed depends on type of rows we're scanning.
 * For skiprows: it refers to global_start. It'll skip
 * rows until it reaches end of rows to skip. In this case, we return
 * current_pos we stopped at in the last local_data to
 * to signal start scanning from this postion for nrows/chunksize.
 * For nrows: it refers to global_end.
 * Start scanning until it reaches total number of rows to read.
 * For chunksize: it refers to chunksize_global_end.
 * Start scanning until it reaches total number of rows to read for
 * chunk-iterator.
 * Since whole dataset is loaded in section, this function needs to
 * keep track of total_bytes scanned so far across all these sections.
 * total_bytes is used to set final global offset value for the whole dataset.
 *
 * @param[in] i_start: start reading position for loaded data
 * @param[in] bytes_read: size of actual data loaded into buffer used to scan
 * for rows
 * @param[in] local_data: buffer that contains current section of data read from
 * file
 * @param[in] row_separator: character used to signal end of line
 * @param[in, out] rows_count: how many rows to scan (it can be skiprows, nrows,
 * chunksize)
 * @param[out] global_offset: global dataset offset to compute
 * (skiprows->global_start, nrows->global_end, chunksize->chunksize_global_end)
 * @param[in] global_start: global dataset start offset to add to global_end
 * @param[in, out] total_bytes: keep track of how many bytes scanned so far
 * across all 8MB data portion.
 * **These information are needed to keep track of rows to skip when computing
 * nrows/chunksize. If skiprows is a list, we need to skip over these rows and
 * avoid counting them towards nrows/chunksize total rows.
 * @param[in] is_skiplist: True if skiprows is a list
 * @param[in] skiprows: array of row numbers to skip
 * cur_row and skiprow_num are needed to keep track of where the scanning
 * process is across the multiple chunks read.
 * @param[in, out] cur_row: current row position to read
 * @param[in, out] skiprow_num: current index for skiprows to skip
 * @param[in] skiprows_list_len: size of skiprows array
 * @return current_pos: current position we reached while scanning this section
 * of local_data. Needed to identify i_start for nrows/chunksize-iterator case.
 */
static int64_t compute_offsets(int64_t i_start, int64_t bytes_read,
                               const bodo_vector<char> &local_data,
                               char row_separator, int64_t &rows_count,
                               int64_t &global_offset, int64_t global_start,
                               int64_t &total_bytes, bool is_skiplist,
                               int64_t *skiprows, int64_t &cur_row,
                               int64_t &skiprow_num,
                               int64_t skiprows_list_len) {
    int64_t current_pos = 0;
    for (int64_t i = i_start; i < bytes_read; i++) {
        if (local_data[i] == row_separator) {
            // skip empty lines
            if (i > i_start && local_data[i - 1] == row_separator) {
                continue;
            }
            cur_row++;
            // avoid counting the skipped row.
            if (is_skiplist && skiprow_num < skiprows_list_len &&
                cur_row == skiprows[skiprow_num]) {
                skiprow_num++;
                continue;
            }
            rows_count--;

            if (!rows_count) {
                // In skiprows case:
                // keep track of current_pos in this local_data read to continue
                // reading from this position for nrows/chunksize.
                current_pos = i + 1 - i_start;
                global_offset = global_start + total_bytes + current_pos;
                break;
            }
        }
    }
    // If local_data is read and we still have rows remaining, add the
    // bytes_read to add this whole section to the total_bytes scanned so far.
    if (rows_count) {
        total_bytes += (bytes_read - i_start);
        // Otherwise, add the part of local_data that was scanned to the
        // total_bytes
    } else {
        total_bytes += current_pos;
    }
    return current_pos;  // needed for determining skipped_start_pos
}
/**
 * Function to scan portion of data from file and computes
 * global dataset start and end position required for each section of data in
 * the file(s) to load into stream read buffer.
 * With skiprows as a list, data is not contiguous and will be read in sections
 * identified by rows to skip.
 * Example:
 * read from beginning upto 1st row to skip is 1st section.
 * read from next row (after skipped one) until the 2nd skipped rows.
 * @param[in] i_start: start reading position for loaded data
 * @param[in] bytes_read: size of actual data loaded into buffer used to scan
 * for rows
 * @param[in] local_data: buffer that contains current section of data read from
 * file
 * @param[in] row_separator: character used to signal end of line
 * @param[in,out] skiprows_list_info: information about skiprows list
 * @param[in, out] skiplist_idx: current index for skiprow element
 * @param[in, out] rows_count: global number of rows scanned so far
 * @param[in, out] total_bytes: keep track of how many bytes scanned so far
 * @param[in] csv_header: True is file contains header
 *
 */
static void compute_skiprows_list_offsets(
    int64_t i_start, int64_t bytes_read, const bodo_vector<char> &local_data,
    char row_separator, SkiprowsListInfo *skiprows_list_info,
    int64_t &skiplist_idx, int64_t &rows_count, int64_t &total_bytes,
    bool csv_header) {
    // Treat 0 like 1, only diff in header which is already
    // handled by in the Python side.
    if (skiprows_list_info->skiprows[0] == 0 && csv_header) {
        skiprows_list_info->skiprows[0]++;
    }
    for (int64_t i = i_start; i < bytes_read; i++) {
        if (local_data[i] == row_separator) {
            // skip empty lines
            if (i > i_start && local_data[i - 1] == row_separator) {
                continue;
            }
            // this is last row to read, next will be skipped
            // compute and store end of row to read
            if (rows_count == skiprows_list_info->skiprows[skiplist_idx] - 1) {
                skiprows_list_info->read_end_offset[skiplist_idx] =
                    total_bytes + i + 1 - i_start;
            }
            // reached end of skipped rows
            // compute and store start of next row to read after skipping
            // current skiprow number
            else if (rows_count == skiprows_list_info->skiprows[skiplist_idx]) {
                skiprows_list_info->read_start_offset[skiplist_idx + 1] =
                    total_bytes + i + 1 - i_start;
                skiplist_idx++;
            }
            rows_count++;
        }
        if (skiplist_idx >= skiprows_list_info->num_skiprows) {
            break;
        }
    }
    // If local_data is read and we still have rows to skip, add the
    // bytes_read to the total_bytes scanned so far.
    if (skiplist_idx < skiprows_list_info->num_skiprows) {
        total_bytes += (bytes_read - i_start);
    }
}
/**
 * Read uncompressed file(s) in chunks to find actual start and end of the data
 * to be loaded based on skiprows and nrows values.
 * Plus, compute and update chunksize-iterator information
 * Compute nrows/skiprows information first time only (when called from
 * file_chunk_reader). Next calls form stream_reader_update, ignores
 * nrows/skiprows and updates chunksize-iterator information only.
 * @param file_names: list of file names
 * @param file_sizes: list of file sizes in bytes
 * @param header_size_bytes: csv header bytes
 * @param fs : File system to read from
 * @param skiprows: number of rows to skip reading from the beginning
 * @param nrows: total number of rows to read from beginning or after skiprows
 * if requested
 * @param global_start[out]: global dataset starting position
 * @param global_end[out]: global dataset end position
 * @param chunkisze[in]: number of rows per chunk (for chunksize-iterator)
 * @param chunksize_global_end[out]: global end position for requested chunksize
 * @param chunksize_bytes[out]: number of bytes for requested chunksize across
 * all ranks
 * @param first_read[in]: flag to identify whether it's first time to compute
 * @param[in,out] skiprows_list_info: information about skiprows list.
 * @param[in] csv_header: True if file(s) has header.
 * nrows/skiprows, or we only need to care about chunksize
 */
void read_file_info(const std::vector<std::string> &file_names,
                    const std::vector<int64_t> &file_sizes,
                    size_t header_size_bytes,
                    std::shared_ptr<arrow::fs::FileSystem> fs,
                    int64_t *skiprows, int64_t nrows, int64_t &global_start,
                    int64_t &global_end, char row_separator, int64_t chunksize,
                    int64_t &chunksize_global_end, int64_t &chunksize_bytes,
                    bool first_read, SkiprowsListInfo *skiprows_list_info,
                    bool csv_header) {
    tracing::Event ev("read_file_info", false);

    // TODO Tune (https://bodo.atlassian.net/browse/BE-2600)
    constexpr int64_t CHUNK_SIZE = 1024 * 1024;
    bodo_vector<char> local_data(CHUNK_SIZE);
    // Bytes skipped from the beginning based on how many rows to skip when
    // using skiprows
    int64_t total_skipped = 0;
    // Bytes to read based on how many rows to read when using nrows
    int64_t nrows_bytes = 0;
    // Start byte position to start read requested data.
    int64_t skipped_start_pos = 0;
    int64_t cur_size = 0;
    int64_t skiplist_idx = 0;  // current index in skiprows list
    // If file has header, start count from 1 as elements in skiprows will be
    // 1-index based.
    // Otherwise, start from 0.
    int64_t rows_count =
        csv_header;  // number of rows scanned when computing skiprows list info
    int64_t total_read =
        0;  // number of bytes scanned  when computing skiprows list info
    int64_t nrows_cur_row =
        0;  // current row scanned when computing nrows with skiprows list
    int64_t nrows_skiprow_idx =
        0;  // current skiprow index while computing nrows
    // number of bytes to add to chunk_start based on global_start value for the
    // whole dataset.
    int64_t remainder = global_start;
    for (size_t file_i = 0; file_i < file_sizes.size(); file_i++) {
        // to handle folder of csvs
        // If global_start's value is still set in the previous file, set
        // remainder to 0 to start reading from top of current file. Otherwise,
        // start reading current file from (global_start-files_read_so_far)
        if (global_start < cur_size) {
            remainder = 0;
        } else {
            remainder = global_start - cur_size;
        }
        int64_t fsize = file_sizes[file_i] - header_size_bytes;
        int64_t bytes_left_to_read = fsize - remainder;
        int64_t chunk_start = remainder + header_size_bytes;
        arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>>
            file_result = fs->OpenInputFile(file_names[file_i]);
        CHECK_ARROW(file_result, "read_file_info", "fs->OpenInputFile")
        std::shared_ptr<arrow::io::RandomAccessFile> file =
            file_result.ValueOrDie();
        arrow::Status status;
        // Read file in chunks of CHUNK_SIZE bytes
        // Scan for row_separator to identify position of the row to skip and/or
        // read (and end of chunksize if requested)
        while ((chunk_start - header_size_bytes) < uint64_t(fsize)) {
            int64_t chunk_end =
                chunk_start + std::min(CHUNK_SIZE, bytes_left_to_read);
            int64_t read_size = chunk_end - chunk_start;
            status = file->Seek(chunk_start);
            arrow::Result<int64_t> bytes_read_result =
                file->Read(read_size, local_data.data());
            CHECK_ARROW(bytes_read_result, "read_file_info", "file->Read")
            int64_t bytes_read = bytes_read_result.ValueOrDie();
            if (bytes_read == 0) {
                break;
            }
            bytes_left_to_read -= read_size;
            // 1. skiprows
            // Compute if skiprows is list or just a number > 0
            if ((skiprows_list_info->is_skiprows_list || skiprows[0] > 0) &&
                first_read) {
                // If skiprows is a list
                // get offsets for rows to skip.
                if (skiprows_list_info->is_skiprows_list) {
                    tracing::Event ev_skiprows_list(
                        "compute_skiprows_list_offsets", false);
                    ev_skiprows_list.add_attribute("skiprows", skiprows[0]);
                    compute_skiprows_list_offsets(
                        0, bytes_read, local_data, row_separator,
                        skiprows_list_info, skiplist_idx, rows_count,
                        total_read, csv_header);
                    // reset index position to signal end of computing skiprows
                    // and start scanning data for nrows/chunksize case below
                    if (skiplist_idx >= skiprows_list_info->num_skiprows) {
                        skiplist_idx = 0;
                    }
                    ev_skiprows_list.finalize();
                }
                // check for row_separator in new data read
                // keep track how of many bytes to skip (total_skipped) from the
                // beginning of the whole dataset and update global_start offset
                else {
                    int64_t dummy_count = 0;
                    tracing::Event ev_skiprows("compute_offsets_ev_skiprows",
                                               false);
                    ev_skiprows.add_attribute("skiprows", skiprows[0]);
                    skipped_start_pos = compute_offsets(
                        0, bytes_read, local_data, row_separator, skiprows[0],
                        global_start, 0, total_skipped,
                        // skiprows_list tracking info. Not needed in this case
                        // but provided for consistency. See compute_offsets
                        false, skiprows, dummy_count, dummy_count, 0);
                    ev_skiprows.finalize();
                }
            }
            // 2. nrows
            // Start nrows count only after skiprows reach 0
            if (!(skiplist_idx && skiprows[0]) && nrows > 0 && first_read) {
                // Start reading rest of current chunk (skipped_start_pos = new
                // char after skipped bytes) Then, continue reading whole chunks
                // (skipped_start_pos=0)
                // check for row_separator in new data read
                // keep track how of many bytes to read (nrows_bytes) after the
                // skipped bytes of the whole dataset and update global_end
                // offset
                // nrows need to know skiprows to avoid counting them towards
                // nrows total count
                tracing::Event ev_nrows("compute_offsets_nrows", false);
                ev_nrows.add_attribute("nrows", nrows);
                compute_offsets(
                    skipped_start_pos, bytes_read, local_data, row_separator,
                    nrows, global_end, global_start, nrows_bytes,
                    skiprows_list_info->is_skiprows_list,
                    skiprows_list_info->skiprows.data(), nrows_cur_row,
                    nrows_skiprow_idx, skiprows_list_info->skiprows.size());
                ev_nrows.finalize();
            }
            // 3. chunksize
            // start from end of skiprows
            if (!(skiplist_idx && skiprows[0]) && chunksize > 0) {
                // check for row_separator in new data read
                // keep track how of many bytes to read (chunksize_bytes) for
                // this chunk after the skipped bytes of the whole dataset and
                // update chunksize_global_end offset
                // If skiprows is a list, don't count skipped rows towards
                // chunksize keep track of how many skipped rows are found while
                // reading chunks. Since compute offset is generalized, I can't
                // just send skiprows_list_info as rows_read and
                // skiprows_list_idx has different meaning for each computation
                // case
                tracing::Event ev_chunksize("compute_offsets_chunksize", false);
                ev_chunksize.add_attribute("chunksize", chunksize);
                compute_offsets(skipped_start_pos, bytes_read, local_data,
                                row_separator, chunksize, chunksize_global_end,
                                global_start, chunksize_bytes,
                                skiprows_list_info->is_skiprows_list,
                                skiprows_list_info->skiprows.data(),
                                skiprows_list_info->rows_read,
                                skiprows_list_info->skiprows_list_idx,
                                skiprows_list_info->num_skiprows);
                ev_chunksize.finalize();
            }
            chunk_start = chunk_end;
            // Reset at end of loop since the first chunk
            // nrows/chunksize-iterator start reading could be continuing a read
            // of a chunk skiprows read part of it. Next chunk(s) to read will
            // be read exclusively for nrows/chunksize-iterator so we set it to
            // 0.
            skipped_start_pos = 0;
            // If the code already scanned through all skiprows and nrows,
            // it got the required information (global_start/global_end) and
            // no need to read more chunks in this file.
            // NOTE: if chunksize is not used it'll be -1. If used, we decrease
            // until it reaches 0. See
            // https://github.com/bodo-ai/Bodo/blob/1476d4812a2131603ef633ef97f0e4e36f05dc02/bodo/ir/csv_ext.py#L986
            // Hence the check:
            if (!(skiprows_list_info->is_skiprows_list &&
                  skiprows_list_info->skiprows[0]) &&
                !nrows && (chunksize <= 0)) {
                break;
            }
        }  // end-while-true
        // If scan reached start and end position, no need to read more files.
        // NOTE: if chunksize is not used it'll be -1. If used, we decrease
        // until it reaches 0. See
        // https://github.com/bodo-ai/Bodo/blob/1476d4812a2131603ef633ef97f0e4e36f05dc02/bodo/ir/csv_ext.py#L986
        // Hence the check:
        if (!(skiprows_list_info->is_skiprows_list &&
              skiprows_list_info->skiprows[0]) &&
            !nrows && ((chunksize <= 0))) {
            break;
        }
        cur_size += fsize;
    }  // end-for-loop-all-files
    // last chunk could have less rows than chunksize
    // In this case the end position is start + chunk_size
    // which handles an edge case if the edge chunk is also
    // the first chunk.
    if (chunksize_global_end <= global_start) {
        chunksize_global_end = global_start + chunksize_bytes;
    }
}
/**
 * Read chunk of the data (could be whole chunk of dataset or just chunksize
 * rows), load it into the mem_reader buffer and communicate row data if data is
 * distributed.
 * @param[in, out] mem_reader: reader
 * @param[in] path_info: information about the csv file(s)
 * @param[in] header_size_bytes: number of bytes in the header
 * @param[in] start_global: global start position of the chunk for each rank
 * @param[in] to_read: number of bytes to read for this chunk
 * @param[in] row_separator: symbol to identify end of row
 * @param[in] is_parallel: whether data is distributed or replicated
 * @param[in] do_correction: whether to data communication correction after read
 * or no. needed for skiplist reading (as it needs to do data correction only
 * after all sections are read)
 */
void read_chunk_data(MemReader *mem_reader, PathInfo *path_info,
                     int64_t header_size_bytes, int64_t start_global,
                     int64_t to_read, char row_separator, bool is_parallel,
                     bool do_correction = true) {
    tracing::Event ev("read_chunk_data", is_parallel);
    ev.add_attribute("to_read", to_read);
    const std::vector<std::string> &file_names = path_info->get_file_names();
    const std::vector<int64_t> &file_sizes = path_info->get_file_sizes();
    int64_t cur_size = 0;
    // find first file to read from and read from it
    int64_t cur_file_idx = 0;
    for (size_t i = 0; i < file_sizes.size(); i++) {
        int64_t fsize = file_sizes[i] - header_size_bytes;
        if (cur_size + fsize > start_global) {
            int64_t file_start = start_global - cur_size + header_size_bytes;
            int64_t file_end =
                file_start +
                std::min(to_read, fsize + header_size_bytes - file_start);
            mem_reader->read_uncompressed_file(file_names[i], file_start,
                                               file_end, path_info->get_fs());
            to_read -= (file_end - file_start);
            cur_file_idx = i + 1;
            break;
        }
        cur_size += fsize;
    }
    // read from subsequent files
    while (to_read > 0) {
        int64_t f_to_read =
            std::min(file_sizes[cur_file_idx] - header_size_bytes, to_read);
        mem_reader->read_uncompressed_file(
            file_names[cur_file_idx], header_size_bytes,
            f_to_read + header_size_bytes, path_info->get_fs());
        to_read -= f_to_read;
        cur_file_idx += 1;
    }
    // correct data so that each rank only has complete rows
    if (is_parallel && do_correction) {
        data_row_correction(mem_reader, row_separator);
    }
}
/**
 * In case of using skiprows list
 * Read chunk of the data in sections according to skiprows boundaries.
 * Section to read is all the bytes between two skipped rows.
 * If only one row is skipped, data is read from beginning to start of skipped
 * rows and then from end of the skipped row to the end. Load it into the
 * mem_reader buffer and communicate row data after reading all sections if data
 * is distributed.
 * @param[in, out] mem_reader: reader that will contain the data.
 * @param[in] path_info: information about the csv file(s).
 * @param[in] header_size_bytes: number of bytes in the header.
 * @param[in] start_global: global start position of the chunk for each rank.
 * @param[in] row_separator: symbol to identify end of row.
 * @param[in] is_parallel: whether data is distributed or replicated.
 * @param[in] main_end_global: global end position of the chunk for each rank.
 * @param[in] read_start_offset: array of start offset for sections to read.
 * @param[in] read_end_offset: array of end offset for sections to read.
 * @param[in] skiprows_list_len: number of rows to skip.
 */
void read_chunk_data_skiplist(MemReader *mem_reader, PathInfo *path_info,
                              int64_t header_size_bytes, int64_t start_global,
                              char row_separator, bool is_parallel,
                              int64_t main_end_global,
                              const std::vector<int64_t> &read_start_offset,
                              const std::vector<int64_t> &read_end_offset,
                              int64_t skiprows_list_len) {
    tracing::Event ev("read_chunk_data_skiplist", is_parallel);
    ev.add_attribute("to_read", main_end_global - start_global);
    int64_t end_global;
    // Add 1 to account for final section (from end of skipped rows to end of
    // chunk)
    for (int skiplist_i = 0; skiplist_i < skiprows_list_len + 1; skiplist_i++) {
        // next read is not in my chunk.
        if (start_global >= main_end_global) {
            break;
        }
        int64_t local_to_read = 0;
        // Check if the rank has data to read in this section.
        // 1. Compute start and end offsets to read
        // start_global is maximum of rank's main starting point and section's
        // start offset. end_global is minimum of rank's main_end and section's
        // end offset.
        // 2. Only read if start and my whole chunk's global end are inside the
        // reading section. start_global is updated with each section so we need
        // to check that the main global end is after the section's start. Also,
        // this handle case where my entire chunk to read is skipped (i.e. skip
        // from 80-100 and my read is from 82-90).
        if (start_global < read_end_offset[skiplist_i] &&
            main_end_global >= read_start_offset[skiplist_i]) {
            // Handle edge case where rank's main start offset is in a row that
            // we don't want to read e.g. skiprow from 81-102 and rank starting
            // position is 100
            start_global =
                std::max(start_global, read_start_offset[skiplist_i]);
            // Handle edge case where rank's end offset is in a row that we
            // don't want to read
            end_global = std::min(main_end_global, read_end_offset[skiplist_i]);
            local_to_read = end_global - start_global;
            // Do data correction only after loading all the data.
            read_chunk_data(mem_reader, path_info, header_size_bytes,
                            start_global, local_to_read, row_separator,
                            is_parallel, false);
        }
        // Set start_global to start of next section
        // start_global==0 (to handle case where first row is skipped so
        // local_read=0)
        if ((local_to_read || start_global == 0) &&
            (size_t)(skiplist_i + 1) < read_start_offset.size()) {
            start_global = read_start_offset[skiplist_i + 1];
        }
    }
    if (is_parallel) {
        data_row_correction(mem_reader, row_separator);
    }
}
// If skiprows and/or nrows are used
// This is the threshold used to determine whether there's enough memory for
// the whole dataset to be loaded first, then compute nrows/skiprows.
// If datasize is more than 60% of the memory size of all nodes,
// then we opt to scan data first
#define MEMORY_LOAD_FACTOR 0.6  // TODO: tune-it

// If nrows is used, and we're reading from a remote file-system like
// S3 or HDFS:
// This is the file-size threshold for the low_memory path.
// (100MB for now)
// TODO: tune-it (https://bodo.atlassian.net/browse/BE-2599)
constexpr int64_t REMOTE_FS_READ_ENTIRE_FILE_THRESHOLD = 100 * 1024 * 1024;

/**
 * Each rank: read my chunk of CSV/JSON dataset.
 * Without chunksize: chunk means total data to load from file for each rank.
 * In case of using chunksize, the chunk refers to first chunksize rows only:
 *  Compute total bytes needed for the whole requested dataset but load only
 * up to chunk bytes. Store information about total size, chunksize bytes, where
 * it ends, and whether data is replicated or distributed. Next, chunks are read
 * when requested by Python iterator in stream_reader_update. It uses the stored
 * information to determine next chunk to read and signal to Python's iterator
 * end of read.
 * For more information: see chunksize-iterator design in confluence
 * (Pandas Read CSV Iterator Design)
 * skiprows can be a single element or a list.
 * Single element: means skip n rows from the beginning.
 * In case of list: means skip specific row numbers.
 *  Note: The skiprows is sorted in Python part.
 *  low_memory path:
 *      rank 0 will compute start/end offset for each row to skip
 *      and broadcast it. Then, each rank will identify its global start and end
 * position and uses the skipped rows offset information to read its chunk in
 * sections.
 *
 *  default path: after loading the whole dataset, each rank will check if the
 * skipped rows is within its chunk and if so, it'll copy the next rows over
 * this skipped row and keep track of number of bytes skipped. Then, resize data
 * buffer accordingly.
 *
 *
 *
 * @param[in] fname : path specifying *all* CSV/JSON file(s) to read (not
 *                just my files)
 * @param[in] suffix : "csv" or "json"
 * @param[in is_parallel : indicates whether data is distributed or replicated
 * @param[in] skiprows : number of rows to skip or list of row number to skip
 * in global dataset
 * @param[in] nrows: number of rows to read in global dataset
 * @param[in] json_lines : true if JSON file is in JSON Lines format (one
 * row/record per line)
 * @param[in] csv_header : true if CSV files contain headers
 * @param[in] compression_pyarg : compression scheme
 * @param[in] bucket_region: s3 bucket server location (needed if data is in s3
 * path)
 * @param[in] chunksize: number of rows per chunk
 * This is needed because behavior of skiprows=4 is different from skiprows=[4]
 * The former means skip 4 rows from the beginning. Later means skip the 4th
 * row.
 * @param[in] is_skiprows_list: Flag to know whether skiprows is a single
 * element or a list or rows.
 * @param[in] skiprows_list_len: number of rows to skip
 * @param[in] pd_low_memory: flag whether user explicitly requested low_memory
 * mode.
 */
extern "C" PyObject *file_chunk_reader(
    const char *fname, const char *suffix, bool is_parallel, int64_t *skiprows,
    int64_t nrows, bool json_lines, bool csv_header,
    const char *compression_pyarg, const char *bucket_region,
    PyObject *storage_options, int64_t chunksize = 0,
    bool is_skiprows_list = false, int64_t skiprows_list_len = 0,
    bool pd_low_memory = false) {
    try {
        CHECK(fname != nullptr, "NULL filename provided.");
        tracing::Event ev("file_chunk_reader", is_parallel);
        if (storage_options == Py_None) {
            throw std::runtime_error("ParquetReader: storage_options is None");
        }

        // Extract values from the storage_options dict
        // Check that it's a dictionary, else throw an error
        bool is_anon = false;
        if (PyDict_Check(storage_options)) {
            // Get value of "anon". Returns NULL if it doesn't exist in the
            // dict. No need to decref s3fs_anon_py, PyDict_GetItemString
            // returns borrowed ref
            PyObject *s3fs_anon_py =
                PyDict_GetItemString(storage_options, "anon");
            if (s3fs_anon_py != nullptr && s3fs_anon_py == Py_True) {
                is_anon = true;
            }
        } else {
            throw std::runtime_error(
                "_csv_json_reader.cpp::file_chunk_reader: storage_options is "
                "not a python dictionary.");
        }
        Py_DECREF(storage_options);

        // TODO right now we get the list of file names and file sizes on rank 0
        // and broadcast to every process. This is potentially not scalable
        // (think many millions of potentially long file names) and not really
        // necessary (we could scatter instead of broadcast). But this doesn't
        // seem like something that should worry us right now

        char row_separator = '\n';
        if ((strcmp(suffix, "json") == 0) && !json_lines) {
            row_separator = '}';
        }

        int rank = dist_get_rank();
        int num_ranks = dist_get_size();
        MemReader *mem_reader = nullptr;

        PathInfo *path_info =
            new PathInfo(fname, compression_pyarg, bucket_region, is_anon);
        if (!path_info->is_path_valid()) {
            delete path_info;
            return nullptr;
        }
        const std::string compression = path_info->get_compression_scheme();
        SkiprowsListInfo *skiprows_list_info =
            new SkiprowsListInfo(is_skiprows_list, skiprows_list_len, skiprows);
        // This flag is used to indicate whether the data is larger than memory
        // available
        // We use it to determine which path to take for computing start
        // position and size of data to read. If data is too large to fit, we
        // should read the file(s) in chunks and compute the offset and size of
        // data to read (if skiprows and/or nrows is used) 2nd path, load the
        // whole data and then compute offset and size of data to read

        bool is_low_memory = false;
        int64_t g_total_bytes = 0;  // total number of bytes of all the data to
                                    // read across all ranks
        int64_t end_global =
            0;  // global end byte position for the whole dataset
        int64_t header_size_bytes = 0;  // size of header in bytes
        int64_t chunksize_global_end =
            0;                          // first chunk global end byte position
        int64_t g_chunksize_bytes = 0;  // size of chunk in bytes
        // 0= total_size, 1=start_pos, 2=end_pos
        std::vector<int64_t> bytes_meta_data(3);
        if (compression == "uncompressed") {
            const std::vector<std::string> &file_names =
                path_info->get_file_names();
            const std::vector<int64_t> &file_sizes =
                path_info->get_file_sizes();
            if (csv_header) {
                header_size_bytes = get_header_size(
                    file_names[0], file_sizes[0], *path_info, '\n');
            }
            // total_size: total size excluding headers
            bytes_meta_data[0] =
                path_info->get_size() - (file_names.size() * header_size_bytes);
            g_total_bytes = bytes_meta_data[0];
            // now determine which files to read from and what portion from each
            // file this is based on partitioning of global dataset based on
            // bytes. As such, this first phase can end up with rows of data in
            // multiple processes
            int64_t start_global =
                0;  // start global offset for the whole dataset
            end_global = bytes_meta_data[0];
            // Find memory across all nodes.
            // Assumes cluster is uniform (i.e. same node type for all clusters)
            // TODO: Non-uniform cluster
            // Allreduce(first rank/node, SUM(node_sys_memory))
            size_t cluster_sys_memory =
                get_total_node_memory() * dist_get_node_count();
            // Checks if memory size is low or user explicitly requested
            // low_memory.
            is_low_memory = (((double)bytes_meta_data[0] / cluster_sys_memory) >
                             MEMORY_LOAD_FACTOR) ||
                            pd_low_memory;
            // If nrows is specified, we should try to avoid reading the entire
            // file from a remote file system if the file-size is >
            // REMOTE_FS_READ_ENTIRE_FILE_THRESHOLD. Instead we should take the
            // low-memory-path, scan the file in chunks and then only read as
            // much as required.
            is_low_memory = is_low_memory ||
                            ((nrows > 0) && (path_info->get_is_remote_fs()) &&
                             ((double)bytes_meta_data[0] >
                              REMOTE_FS_READ_ENTIRE_FILE_THRESHOLD));
            // If one of the ranks has is_low_memory true,
            // all ranks should switch to scanning first.
            CHECK_MPI(MPI_Allreduce(MPI_IN_PLACE, &is_low_memory, 1, MPI_C_BOOL,
                                    MPI_LOR, MPI_COMM_WORLD),
                      "file_chunk_reader: MPI error on MPI_Allreduce:");

            // If skiprows and/or nrows are used and we decided to scan first,
            // start_global and/or end_global will be shifted.
            // Rank 0 computes these new offsets and
            // stores starting offset based on skiprows in start_pos. Default 0
            // stores ending offset based on nrows in end_pos. Default end of
            // file(s) computes new total_size start_pos
            // global_start
            bytes_meta_data[1] = start_global;
            // global_end
            bytes_meta_data[2] = end_global;

            // Special case when nrows=0, we don't read from disk
            if (nrows == 0) {
                bytes_meta_data[0] = 0;
                bytes_meta_data[1] = 0;
                bytes_meta_data[2] = 0;
            }
            // Cases to use low_memory path
            // 1. user used pd.read_csv(..., low_memory=True)
            // 2. file(s) > 60% of memory && nrows/skiprows is used
            // 3. chunksize is used
            // 4. nrows is used &&
            //    file(s) > REMOTE_FS_READ_ENTIRE_FILE_THRESHOLD &&
            //    reading from a remote filesystem like S3 or HDFS
            // i.e. we shouldn't attempt to load all data and then do
            // nrows/skiprows
            if (((is_skiprows_list || skiprows[0] || (nrows > 0)) &&
                 is_low_memory) ||
                (chunksize > 0)) {
                // rank 0 finds update start, end offsets, and total size.
                if (nrows > 0 && chunksize > nrows) {
                    chunksize = nrows;
                }
                // rank 0 computes bytes information and broadcasts it.
                if (rank == 0) {
                    read_file_info(file_names, file_sizes, header_size_bytes,
                                   path_info->get_fs(), skiprows, nrows,
                                   bytes_meta_data[1], bytes_meta_data[2],
                                   row_separator, chunksize,
                                   chunksize_global_end, g_chunksize_bytes,
                                   true, skiprows_list_info, csv_header);
                    bytes_meta_data[0] =
                        bytes_meta_data[2] - bytes_meta_data[1];
                }
                if (is_skiprows_list) {
                    // Brodacast skiprows_list_info object
                    CHECK_MPI(
                        MPI_Bcast(skiprows_list_info->read_start_offset.data(),
                                  skiprows_list_info->read_start_offset.size(),
                                  MPI_INT64_T, 0, MPI_COMM_WORLD),
                        "file_chunk_reader: MPI error on MPI_Bcast:");
                    CHECK_MPI(
                        MPI_Bcast(skiprows_list_info->read_end_offset.data(),
                                  skiprows_list_info->read_end_offset.size(),
                                  MPI_INT64_T, 0, MPI_COMM_WORLD),
                        "file_chunk_reader: MPI error on MPI_Bcast:");
                    CHECK_MPI(MPI_Bcast(&skiprows_list_info->rows_read, 1,
                                        MPI_INT64_T, 0, MPI_COMM_WORLD),
                              "file_chunk_reader: MPI error on MPI_Bcast:");
                    CHECK_MPI(MPI_Bcast(&skiprows_list_info->skiprows_list_idx,
                                        1, MPI_INT64_T, 0, MPI_COMM_WORLD),
                              "file_chunk_reader: MPI error on MPI_Bcast:");
                }
                CHECK_MPI(MPI_Bcast(bytes_meta_data.data(), 3, MPI_INT64_T, 0,
                                    MPI_COMM_WORLD),
                          "file_chunk_reader: MPI error on MPI_Bcast:");
                CHECK_MPI(MPI_Bcast(&chunksize_global_end, 1, MPI_INT64_T, 0,
                                    MPI_COMM_WORLD),
                          "file_chunk_reader: MPI error on MPI_Bcast:");
                CHECK_MPI(MPI_Bcast(&g_chunksize_bytes, 1, MPI_INT64_T, 0,
                                    MPI_COMM_WORLD),
                          "file_chunk_reader: MPI error on MPI_Bcast:");
                // chunksize iterator case
                // only read size upto number of bytes in the chunk
                if (chunksize > 0) {
                    // Total bytes to read from file(s)
                    // could be updated if skiprows/nrows are used in low_memory
                    // mode
                    g_total_bytes = bytes_meta_data[0];
                    bytes_meta_data[0] = g_chunksize_bytes;
                    bytes_meta_data[2] = chunksize_global_end;
                }
            }
            if (is_parallel) {
                start_global =
                    bytes_meta_data[1] +
                    dist_get_start(bytes_meta_data[0], num_ranks, rank);
                end_global = bytes_meta_data[1] +
                             dist_get_end(bytes_meta_data[0], num_ranks, rank);
            } else {
                start_global = bytes_meta_data[1];
                end_global = bytes_meta_data[2];
            }
            if (!is_skiprows_list) {
                int64_t to_read = end_global - start_global;
                mem_reader = new MemReader(to_read, row_separator, !json_lines);
                read_chunk_data(mem_reader, path_info, header_size_bytes,
                                start_global, to_read, row_separator,
                                is_parallel);
            } else {
                int64_t main_end_global = end_global;
                int64_t to_read = end_global - start_global;
                mem_reader = new MemReader(to_read, row_separator, !json_lines);
                skiprows_list_info->read_start_offset[0] =
                    start_global;  // global first read portion
                skiprows_list_info->read_end_offset[skiprows_list_len] =
                    bytes_meta_data[2];  // global end read portion
                read_chunk_data_skiplist(
                    mem_reader, path_info, header_size_bytes, start_global,
                    row_separator, is_parallel, main_end_global,
                    skiprows_list_info->read_start_offset,
                    skiprows_list_info->read_end_offset, skiprows_list_len);
            }
        } else {
            // if files are compressed, one rank will be responsible for
            // decompressing a whole file into memory. Data will later be
            // redistributed (see balance_rows below)
            const std::vector<std::string> &file_names =
                path_info->get_file_names();
            mem_reader = new MemReader(row_separator, !json_lines);
            int64_t num_files = file_names.size();
            if (is_parallel && num_files < num_ranks) {
                // try to space the read across nodes to avoid memory issues
                // if decompressing huge files
                int64_t ppf = num_ranks / num_files;
                if (rank % ppf == 0) {
                    // I read a file
                    int64_t my_file = rank / ppf;
                    if (my_file < num_files) {
                        mem_reader->read_compressed_file(
                            file_names[my_file], path_info->get_fs(),
                            compression, csv_header);
                    }
                }
            } else {
                int64_t start = 0;
                int64_t end = num_files;
                if (is_parallel) {
                    start = dist_get_start(num_files, num_ranks, rank);
                    end = dist_get_end(num_files, num_ranks, rank);
                }
                for (int64_t i = start; i < end; i++) {
                    // note that multiple calls to read_compressed_file append
                    // to existing data
                    mem_reader->read_compressed_file(file_names[i],
                                                     path_info->get_fs(),
                                                     compression, csv_header);
                }
            }
        }

        // Execute these only if weren't done earlier (based on memory
        // availability and use of chunksize)
        if (!is_low_memory && (chunksize <= 0)) {
            // skip rows if requested
            // list path
            if (is_skiprows_list) {
                set_skiprows_list(mem_reader, skiprows, skiprows_list_len,
                                  is_parallel, csv_header);
            } else {
                skip_rows(mem_reader, skiprows[0], is_parallel);
            }

            // read nrows if requested
            set_nrows(mem_reader, nrows, is_parallel);
        }

        // shuffle data so that each rank has required number of rows
        if (is_parallel) {
            balance_rows(mem_reader);
        }

        // prepare data for pandas
        mem_reader->finalize();

        // compressed case does not handle nrows with low_memory
        // TODO: chunksize with compressed (compute offsets)

        // now create a stream reader PyObject that wraps MemReader, to be read
        // from pandas
        auto gilstate = PyGILState_Ensure();
        PyObject *reader =
            PyObject_CallFunctionObjArgs((PyObject *)&stream_reader_type, NULL);
        PyGILState_Release(gilstate);
        if (reader == nullptr || PyErr_Occurred()) {
            PyErr_Print();
            std::cerr << "Could not create chunk reader object" << std::endl;
            if (reader) {
                delete reader;
            }
            reader = nullptr;
        } else {
            stream_reader_init(
                reinterpret_cast<stream_reader *>(reader), mem_reader, 0,
                mem_reader->getSize(), bytes_meta_data[1], chunksize_global_end,
                g_total_bytes, chunksize, is_parallel, path_info,
                header_size_bytes, skiprows_list_info, csv_header);
        }
        return reader;
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}
#undef MEMORY_LOAD_FACTOR

extern "C" PyObject *csv_file_chunk_reader(
    const char *fname, bool is_parallel, int64_t *skiprows, int64_t nrows,
    bool header, const char *compression, const char *bucket_region,
    PyObject *storage_options, int64_t chunksize, bool is_skiprows_list,
    int64_t skiprows_list_len, bool pd_low_memory) {
    return file_chunk_reader(fname, "csv", is_parallel, skiprows, nrows, true,
                             header, compression, bucket_region,
                             storage_options, chunksize, is_skiprows_list,
                             skiprows_list_len, pd_low_memory);
}

extern "C" PyObject *json_file_chunk_reader(const char *fname, bool lines,
                                            bool is_parallel, int64_t nrows,
                                            const char *compression,
                                            const char *bucket_region,
                                            PyObject *storage_options) {
    // TODO nrows not used??
    int64_t skiprows = 0;
    return file_chunk_reader(fname, "json", is_parallel, &skiprows, nrows,
                             lines, false, compression, bucket_region,
                             storage_options);
}

// Update reader. returns true if reader has data
extern "C" bool update_csv_reader(PyObject *reader) {
    stream_reader *reader_obj = reinterpret_cast<stream_reader *>(reader);
    // If data is empty return false.
    if (reader_obj->g_total_bytes <= 0) {
        return false;
    }
    bool initial_read = reader_obj->first_read;
    // First call will already have data in buffer.
    // data already in buffer, no need to update.
    if (initial_read) {
        reader_obj->first_read = false;
        return true;
    }
    // Next calls will update the data and return false when EOF or
    // no more data nrows to read.
    return stream_reader_update_reader(reader_obj);
}

// Initialize reader
extern "C" void initialize_csv_reader(PyObject *reader) {
    stream_reader *reader_obj = reinterpret_cast<stream_reader *>(reader);
    reader_obj->first_read = true;
}

// NOTE: some old testing code that is commented out due to
// introduction of FileReader interface.
// TODO: update testing code
// taking a string to create a istream and calling csv_chunk_reader
// TODO: develop a StringReader class for testing
// extern "C" PyObject* csv_string_chunk_reader(const std::string * str, bool
// is_parallel)
// {
//     CHECK(str != NULL, "NULL string provided.");
//     // get total file-size
//     std::istringstream * f = new std::istringstream(*str);
//     CHECK(f->good(), "could not create istrstream from string.");
//     return csv_chunk_reader(f, str->size(), is_parallel, 0, -1);
// }
/**
 * Update reader with data for next chunk iterator
 * Return false when no more data to read
 * Computes chunksize total bytes and global end position for this chunk
 * Then, each rank loads its chunk and create corresponding mem_reader
 * to be wrapped and passed back to Python
 * @param[in, out] self: stream_reader passed from Python
 */
static bool stream_reader_update_reader(stream_reader *self) {
    // Check if reached end of requested data to read (could be EOF or nrows)
    // global_end is global byte offset (includes skipped bytes)
    // If skiprows is used, add total skipped bytes to size of total data to
    // read
    if (self->global_end >= (self->g_total_bytes + self->first_pos)) {
        return false;
    }
    const std::vector<std::string> &file_names =
        self->path_info->get_file_names();
    const std::vector<int64_t> &file_sizes = self->path_info->get_file_sizes();
    int rank = dist_get_rank();
    int num_ranks = dist_get_size();
    // Set new start to be end of previous chunk
    int64_t l_start_global = self->global_end;
    // store start of the new chunk to use in computing new local start/end
    // offsets
    int64_t tmp = l_start_global;
    // Store size of the new chunk.
    int64_t g_chunksize_bytes = 0;
    // Rank's own end
    int64_t l_end_global = l_start_global;
    // rank 0: compute new chunk size and global end. Then, broadcast them.
    if (rank == 0) {
        read_file_info(file_names, file_sizes, self->header_size_bytes,
                       self->path_info->get_fs(),
                       self->skiprows_list_info->skiprows.data(), 0,
                       l_start_global, l_end_global, '\n', self->chunksize_rows,
                       self->global_end, g_chunksize_bytes, false,
                       self->skiprows_list_info, self->csv_header);
    }
    CHECK_MPI(MPI_Bcast(&self->global_end, 1, MPI_INT64_T, 0, MPI_COMM_WORLD),
              "stream_reader_update_reader: MPI error on MPI_Bcast:");
    CHECK_MPI(MPI_Bcast(&g_chunksize_bytes, 1, MPI_INT64_T, 0, MPI_COMM_WORLD),
              "stream_reader_update_reader: MPI error on MPI_Bcast:");
    CHECK_MPI(MPI_Bcast(&self->skiprows_list_info->rows_read, 1, MPI_INT64_T, 0,
                        MPI_COMM_WORLD),
              "stream_reader_update_reader: MPI error on MPI_Bcast:");
    CHECK_MPI(MPI_Bcast(&self->skiprows_list_info->skiprows_list_idx, 1,
                        MPI_INT64_T, 0, MPI_COMM_WORLD),
              "stream_reader_update_reader: MPI error on MPI_Bcast:");
    // Each rank computes its start and end position to read
    if (self->is_parallel) {
        l_start_global =
            tmp + dist_get_start(g_chunksize_bytes, num_ranks, rank);
        l_end_global = tmp + dist_get_end(g_chunksize_bytes, num_ranks, rank);
    }  // if-parallel
    else {
        l_end_global += g_chunksize_bytes;
    }
    // Identify how many bytes each rank should read.
    int64_t to_read = l_end_global - l_start_global;
    MemReader *mem_reader = new MemReader(to_read, '\n', false);
    // load data in a new mem_reader
    if (!self->skiprows_list_info->is_skiprows_list) {
        read_chunk_data(mem_reader, self->path_info, self->header_size_bytes,
                        l_start_global, to_read, '\n', self->is_parallel);
    } else {
        int64_t main_end_global = l_end_global;  // my end.
        self->skiprows_list_info->read_start_offset[0] =
            tmp;  // chunk global start
        self->skiprows_list_info
            ->read_end_offset[self->skiprows_list_info->num_skiprows] =
            self->global_end;  // chunk global end
        read_chunk_data_skiplist(mem_reader, self->path_info,
                                 self->header_size_bytes, l_start_global, '\n',
                                 self->is_parallel, main_end_global,
                                 self->skiprows_list_info->read_start_offset,
                                 self->skiprows_list_info->read_end_offset,
                                 self->skiprows_list_info->num_skiprows);
    }
    // shuffle data so that each rank has required number of rows
    if (self->is_parallel) {
        balance_rows(mem_reader);
    }
    // prepare data for pandas
    mem_reader->pos = mem_reader->start;
    mem_reader->finalize();
    // update reader with new one.
    delete self->ifs;
    self->ifs = mem_reader;
    // Update chunk_size after data is balanced
    self->chunk_size = mem_reader->getSize();
    self->chunk_pos = 0;

    return true;
}

#undef CHECK
