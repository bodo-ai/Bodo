// Copyright (C) 2019 Bodo Inc. All rights reserved.
/*
  SPMD Stream, CSV reader, and JSON reader.

  We provide a Python object that is file-like in the Pandas sense
  and can so be used as the input argument to pandas CSV & JSON read.
  When called in a parallel/distributed setup, each process owns a
  chunk of the csv/json file only. The chunks are balanced by number of
  rows of dataframe (not necessarily number of bytes), determined by number
  of lines(csv & json(orient = 'records', lines=True)) or
  number of objects (json(orient = 'records', lines=False)) . The actual file
  read is done lazily in the objects read method.
*/
#include "_csv_json_reader.h"
#include <Python.h>
#include <mpi.h>
#include <algorithm>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/tokenizer.hpp>
#include <cinttypes>
#include <ciso646>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "../libs/_bodo_common.h"
#include "../libs/_distributed.h"
#include "_bodo_file_reader.h"
#include "_fs_io.h"
#include "arrow/filesystem/localfs.h"
#include "arrow/io/compressed.h"
#include "arrow/util/compression.h"
#include "structmember.h"

// lines argument of read_json(lines = json_lines)
// when json_lines=true, we are reading Json line format where each
// dataframe row/json record takes up exactly one line, ended with '\n'
// when json_lines=false. we are reading a multi line json format where each
// dataframe row/json record takes up more than one lines, separated by ',\n'

#define CHECK(expr, msg)                                    \
    if (!(expr)) {                                          \
        std::cerr << "Error in read: " << msg << std::endl; \
    }

// read local files
// currently using ifstream, TODO: benchmark Arrow's LocalFS
class LocalFileReader : public SingleFileReader {
   public:
    std::ifstream *fstream;
    LocalFileReader(const char *_fname, const char *f_type, bool csv_header,
                    bool json_lines)
        : SingleFileReader(_fname, f_type, csv_header, json_lines) {
        this->fstream = new std::ifstream(fname);
        CHECK(fstream->good() && !fstream->eof() && fstream->is_open(),
              "could not open file.");
    }
    uint64_t getSize() { return boost::filesystem::file_size(fname); }
    bool seek(int64_t pos) {
        this->fstream->seekg(pos + this->csv_header_bytes, std::ios_base::beg);
        return this->ok();
    }
    bool ok() { return (this->fstream->good() and !this->fstream->eof()); }
    bool read_to_buff(char *s, int64_t size) {
        this->fstream->read(s, size);
        return this->ok();
    }
    virtual ~LocalFileReader() {
        if (fstream) delete fstream;
    }
};

class LocalDirectoryFileReader : public DirectoryFileReader {
   public:
    path_vec
        file_paths;  // sorted paths of each csv/json file inside the directory
    LocalDirectoryFileReader(const char *_dirname, const char *f_type,
                             bool csv_header, bool json_lines)
        : DirectoryFileReader(_dirname, f_type, csv_header, json_lines) {
        // only keep the files that are csv/json files
        std::string dirname(_dirname);
        const char *suffix = this->f_type_to_string();

        auto nonEmptyEndsWithSuffix =
            [suffix](const boost::filesystem::path &path) -> bool {
            return (boost::filesystem::is_regular_file(path) &&
                    boost::ends_with(path.string(), suffix) &&
                    boost::filesystem::file_size(path) > 0);
        };

        std::copy_if(boost::filesystem::directory_iterator(_dirname),
                     boost::filesystem::directory_iterator(),
                     std::back_inserter(this->file_paths),
                     nonEmptyEndsWithSuffix);

        if ((this->file_paths).size() == 0) {
            Bodo_PyErr_SetString(
                PyExc_RuntimeError,
                ("No valid file to read from directory: " + dirname).c_str());
        }
        // sort all files in directory
        std::sort(this->file_paths.begin(), this->file_paths.end());

        // find & set all file name
        for (auto it = this->file_paths.begin(); it != this->file_paths.end();
             ++it) {
            (this->file_names).push_back((*it).string());
        }

        // find and set header row size in bytes
        this->findHeaderRowSize();

        // find dir_size and construct file_sizes
        // assuming the directory contains files only, i.e. no subdirectory
        this->dir_size = 0;
        for (auto it = this->file_paths.begin(); it != this->file_paths.end();
             ++it) {
            (this->file_sizes).push_back(this->dir_size);
            this->dir_size += boost::filesystem::file_size(*it);
            this->dir_size -= this->csv_header_bytes;
        }
        this->file_sizes.push_back(this->dir_size);
    };

    void initFileReader(const char *fname) {
        this->f_reader =
            new LocalFileReader(fname, this->f_type_to_string(),
                                this->csv_header, this->json_lines);
        this->f_reader->csv_header_bytes = this->csv_header_bytes;
    };
};
#undef CHECK

// ***********************************************************************************
// Our file-like object for reading chunks in a std::istream
// ***********************************************************************************

typedef struct {
    PyObject_HEAD
        /* Your internal buffer, size and pos */
        FileReader *ifs;  // input stream
    size_t chunk_start;   // start of our chunk
    size_t chunk_size;    // size of our chunk
    size_t chunk_pos;     // current position in our chunk
    std::vector<char>
        buf;  // internal buffer for converting stream input to Unicode object
} stream_reader;

static void stream_reader_dealloc(stream_reader *self) {
    // we own the stream!
    if (self->ifs) delete self->ifs;
    Py_TYPE(self)->tp_free(self);
}

// alloc a HPTAIO object
static PyObject *stream_reader_new(PyTypeObject *type, PyObject *args,
                                   PyObject *kwds) {
    stream_reader *self = (stream_reader *)type->tp_alloc(type, 0);
    if (PyErr_Occurred()) {
        PyErr_Print();
        return NULL;
    }
    self->ifs = NULL;
    self->chunk_start = 0;
    self->chunk_size = 0;
    self->chunk_pos = 0;

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

// We use this (and not the above) from C to init our StreamReader object
// Will seek to chunk beginning
static void stream_reader_init(stream_reader *self, FileReader *ifs,
                               size_t start, size_t sz) {
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
    bool ok = self->ifs->seek(start);
    if (!ok) {
        Bodo_PyErr_SetString(PyExc_RuntimeError,
                             "Could not seek to start position");
        return;
    }
    self->chunk_start = start;
    self->chunk_size = sz;
    self->chunk_pos = 0;
}

// read given number of bytes from our chunk and return a Unicode Object
// returns NULL if an error occured.
// does not read beyond end of our chunk (even if file continues)
static PyObject *stream_reader_read(stream_reader *self, PyObject *args) {
    // partially copied from from CPython's stringio.c
    if (self->ifs == NULL) {
        PyErr_SetString(PyExc_ValueError,
                        "I/O operation on uninitialized StreamReader object");
        return NULL;
    }
    Py_ssize_t size, n;

    PyObject *arg = Py_None;
    if (!PyArg_ParseTuple(args, "|O:read", &arg)) {
        return NULL;
    }
    if (PyNumber_Check(arg)) {
        size = PyNumber_AsSsize_t(arg, PyExc_OverflowError);
        if (size == -1 && PyErr_Occurred()) {
            return NULL;
        }
    } else if (arg == Py_None) {
        /* Read until EOF is reached, by default. */
        size = -1;
    } else {
        PyErr_Format(PyExc_TypeError, "integer argument expected, got '%s'",
                     Py_TYPE(arg)->tp_name);
        return NULL;
    }
    /* adjust invalid sizes */
    n = self->chunk_size - self->chunk_pos;
    if (size < 0 || size > n) {
        size = n;
        if (size < 0) size = 0;
    }
    self->buf.resize(size);
    bool ok = self->ifs->read(self->buf.data(), size);
    self->chunk_pos += size;
    if (!ok) {
        std::cerr << "Failed reading " << size << " bytes" << std::endl;
        return NULL;
    }
    // buffer_rd_bytes() function of pandas expects a Bytes object
    // using PyUnicode_FromStringAndSize is wrong since 'size'
    // may end up in the middle a multi-byte UTF-8 character
    return PyBytes_FromStringAndSize(self->buf.data(), size);
}

// Needed to make Pandas accept it, never used
static PyObject *stream_reader_iternext(PyObject *self) {
    std::cerr << "iternext not implemented";
    return NULL;
};

// our class has only one method
static PyMethodDef stream_reader_methods[] = {
    {
        "read",
        (PyCFunction)stream_reader_read,
        METH_VARARGS,
        "Read at most n characters, returned as a unicode.",
    },
    {NULL} /* Sentinel */
};

// the actual Python type class
static PyTypeObject stream_reader_type = {
    PyObject_HEAD_INIT(NULL) "bodo.libs.hio.StreamReader", /*tp_name*/
    sizeof(stream_reader),                                 /*tp_basicsize*/
    0,                                                     /*tp_itemsize*/
    (destructor)stream_reader_dealloc,                     /*tp_dealloc*/
    0,                                                     /*tp_print*/
    0,                                                     /*tp_getattr*/
    0,                                                     /*tp_setattr*/
    0,                                                     /*tp_compare*/
    0,                                                     /*tp_repr*/
    0,                                                     /*tp_as_number*/
    0,                                                     /*tp_as_sequence*/
    0,                                                     /*tp_as_mapping*/
    0,                                                     /*tp_hash */
    0,                                                     /*tp_call*/
    0,                                                     /*tp_str*/
    0,                                                     /*tp_getattro*/
    0,                                                     /*tp_setattro*/
    0,                                                     /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,              /*tp_flags*/
    "stream_reader objects",                               /* tp_doc */
    0,                                                     /* tp_traverse */
    0,                                                     /* tp_clear */
    0,                                                     /* tp_richcompare */
    0,                      /* tp_weaklistoffset */
    stream_reader_iternext, /* tp_iter */
    stream_reader_iternext, /* tp_iternext */
    stream_reader_methods,  /* tp_methods */
    0,                      /* tp_members */
    0,                      /* tp_getset */
    0,                      /* tp_base */
    0,                      /* tp_dict */
    0,                      /* tp_descr_get */
    0,                      /* tp_descr_set */
    0,                      /* tp_dictoffset */
    stream_reader_pyinit,   /* tp_init */
    0,                      /* tp_alloc */
    stream_reader_new,      /* tp_new */
};

// at module load time we need to make our type known ot Python
extern "C" void PyInit_csv(PyObject *m) {
    if (PyType_Ready(&stream_reader_type) < 0) return;
    Py_INCREF(&stream_reader_type);
    PyModule_AddObject(m, "StreamReader", (PyObject *)&stream_reader_type);
    PyObject_SetAttrString(
        m, "csv_file_chunk_reader",
        PyLong_FromVoidPtr((void *)(&csv_file_chunk_reader)));
    // NOTE: old testing code that is commented out due to
    // introduction of FileReader interface.
    // TODO: update testing code
    // PyObject_SetAttrString(m, "csv_string_chunk_reader",
    //                        PyLong_FromVoidPtr((void*)(&csv_string_chunk_reader)));
}

extern "C" void PyInit_json(PyObject *m) {
    if (PyType_Ready(&stream_reader_type) < 0) return;
    Py_INCREF(&stream_reader_type);
    PyModule_AddObject(m, "StreamReader", (PyObject *)&stream_reader_type);
    PyObject_SetAttrString(
        m, "json_file_chunk_reader",
        PyLong_FromVoidPtr((void *)(&json_file_chunk_reader)));
}

// ***********************************************************************************
// C interface for getting the file-like chunk reader
// ***********************************************************************************

#define CHECK(expr, msg)                                    \
    if (!(expr)) {                                          \
        std::cerr << "Error in read: " << msg << std::endl; \
        return NULL;                                        \
    }

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
    PathInfo(const char *file_path, const std::string &compression_pyarg) : file_path(file_path) {
        // obtain path info on rank 0, broadcast to other ranks.
        // this sets PathInfo attributes on all ranks
        obtain_is_directory();
        obtain_file_names_and_sizes();
        obtain_compression_scheme(compression_pyarg);
    }

    /// get the compression scheme used by this file(s)
    const std::string &get_compression_scheme() const { return compression; }

    /// true if path refers to a directory
    bool is_directory() const { return is_dir; }

    /**
     * Get file names for this path.
     * If the path is a directory, it only considers files of size greater than
     * zero whose name is not "_SUCCESS" or ends in ".crc".
     */
    const std::vector<std::string> &get_file_names() const {
        return file_names;
    }

    /**
     * Get file sizes for this path. Item in position i corresponds
     * to the size of file name in position i returned by get_file_names.
     * If the path is a directory, it only considers files of size greater than
     * zero whose name is not "_SUCCESS" or ends in ".crc".
     */
    const std::vector<int64_t> &get_file_sizes() const {
        return file_sizes;
    }

    /**
     * Return name of first file in path. If path is a single file then just
     * return the name of that file.
     */
    const std::string get_first_file() const {
        return file_names[0];
    }

    /**
     * Return total size of all files in path.
     */
    int64_t get_size() const {
        return total_ds_size;
    }

    /**
     * Get arrow::fs::FileSystem object necessary to read data from this path.
     */
    std::shared_ptr<arrow::fs::FileSystem> get_fs() {
        if (!fs) {
            bool is_hdfs = boost::starts_with(file_path, "hdfs://");
            bool is_s3 = boost::starts_with(file_path, "s3://");
            if (is_s3 || is_hdfs) {
                arrow::internal::Uri uri;
                uri.Parse(file_path);
                PyObject *fs_mod = nullptr;
                PyObject *func_obj = nullptr;
                if (is_s3) {
                    import_fs_module(Bodo_Fs::s3, "", fs_mod);
                    get_get_fs_pyobject(Bodo_Fs::s3, "", fs_mod, func_obj);
                    s3_get_fs_t s3_get_fs =
                        (s3_get_fs_t)PyNumber_AsSsize_t(func_obj, NULL);
                    std::shared_ptr<arrow::fs::S3FileSystem> s3_fs;
                    s3_get_fs(&s3_fs);
                    fs = s3_fs;
                    // remove s3:// prefix from file_path
                    arrow::fs::S3Options::FromUri(uri, &file_path);
                } else if (is_hdfs) {
                    import_fs_module(Bodo_Fs::hdfs, "", fs_mod);
                    get_get_fs_pyobject(Bodo_Fs::hdfs, "", fs_mod, func_obj);
                    hdfs_get_fs_t hdfs_get_fs =
                        (hdfs_get_fs_t)PyNumber_AsSsize_t(func_obj, NULL);
                    std::shared_ptr<::arrow::fs::HadoopFileSystem> hdfs_fs;
                    hdfs_get_fs(file_path, &hdfs_fs);
                    fs = hdfs_fs;
                    // remove hdfs://host:port prefix from file_path
                    file_path = uri.path();
                }
                Py_DECREF(fs_mod);
                Py_DECREF(func_obj);
            } else {
                fs = std::make_shared<arrow::fs::LocalFileSystem>();
            }
        }
        return fs;
    }

   private:

    /**
     * Determines if path is a directory or a single file.
     * The filesystem is accessed only on rank 0, result is communicated to
     * other processes using MPI.
     */
    void obtain_is_directory() {
        int c_is_dir = 0;
        if (dist_get_rank() == 0) {
            std::shared_ptr<arrow::fs::FileSystem> fs = get_fs();
            arrow::fs::FileInfo file_stat =
                fs->GetFileInfo(file_path).ValueOrDie();
            if (file_stat.IsDirectory())
                c_is_dir = 1;
            else if (file_stat.IsFile())
                c_is_dir = 0;
            else
                Bodo_PyErr_SetString(
                    PyExc_RuntimeError,
                    "Error in PathInfo::is_directory: invalid path");
        }
        MPI_Bcast(&c_is_dir, 1, MPI_INT, 0, MPI_COMM_WORLD);
        is_dir = bool(c_is_dir);
    }

    /**
     * Obtain and store the vectors of file names and sizes for this path.
     * If the path is a directory, it only considers files of size greater than
     * zero whose name is not "_SUCCESS" or ends in ".crc".
     * The filesystem is accessed only on rank 0, result is communicated to
     * other processes using MPI.
     */
    void obtain_file_names_and_sizes() {
        file_names.clear();
        file_sizes.clear();
        total_ds_size = 0;  // this attribute tracks the sum of all file sizes
        if (dist_get_rank() == 0) {
            std::shared_ptr<arrow::fs::FileSystem> fs = get_fs();
            int64_t total_len = 0;  // total length of file names (including
                                    // null bytes at end of each)
            if (!is_directory()) {
                // we always send file name to other ranks, even if just a
                // single file, because it might be modified by
                // PathInfo::get_fs(), which initially is only done on rank 0
                file_names.push_back(file_path);
                const arrow::fs::FileInfo &fi =
                    fs->GetFileInfo(file_path).ValueOrDie();
                file_sizes.push_back(fi.size());
                total_ds_size += fi.size();
                total_len = int64_t(file_path.size() + 1);
            } else {
                arrow::fs::FileSelector dir_selector;
                // initialize dir_selector
                dir_selector.base_dir = file_path;
                // FileInfo used to determine names and sizes
                std::vector<arrow::fs::FileInfo> file_infos =
                    fs->GetFileInfo(dir_selector).ValueOrDie();
                // sort files by name
                std::sort(file_infos.begin(), file_infos.end(),
                          arrow::fs::FileInfo::ByPath{});

                for (auto &fi : file_infos) {
                    const std::string &path = fi.path();
                    const std::string &fname = fi.base_name();
                    int64_t fsize = fi.size();
                    // skip 0 size files and those ending in .crc or named
                    // _SUCCESS (auxiliary files generated by Spark)
                    if (fsize <= 0) continue;
                    if (boost::ends_with(fname, ".crc")) continue;
                    if (fname == "_SUCCESS") continue;
                    total_len += int64_t(path.size() + 1);
                    total_ds_size += fi.size();
                    // NOTE: this gives the full path and file name.
                    // we might just want to get the file name
                    file_names.push_back(path);
                    file_sizes.push_back(fsize);
                }
            }
            if (dist_get_size() > 1) {
                // send file names to other ranks
                std::vector<char> str_data(total_len);
                char *str_data_ptr = str_data.data();
                for (auto &fname : file_names) {
                    memcpy(str_data_ptr, fname.c_str(), fname.size());
                    str_data_ptr[fname.size()] = 0;  // null terminate string
                    str_data_ptr += fname.size() + 1;
                }
                MPI_Bcast(&total_len, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
                MPI_Bcast(str_data.data(), str_data.size(), MPI_CHAR, 0,
                          MPI_COMM_WORLD);
            }
        } else {
            // receive file names from rank 0
            int64_t recv_size;
            MPI_Bcast(&recv_size, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
            std::vector<char> str_data(recv_size);
            MPI_Bcast(str_data.data(), str_data.size(), MPI_CHAR, 0,
                      MPI_COMM_WORLD);
            char *cur_str = str_data.data();
            while (cur_str < str_data.data() + recv_size) {
                file_names.push_back(cur_str);
                cur_str += file_names.back().size() + 1;
            }
            file_sizes.resize(file_names.size());
        }
        // communicate file sizes to all processes
        MPI_Bcast(file_sizes.data(), file_sizes.size(), MPI_INT64_T, 0,
                  MPI_COMM_WORLD);
        MPI_Bcast(&total_ds_size, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
    }

    /**
     * Given the compression argument received from pd.read_csv()/pd.read_json
     * set and return the compression scheme used.
     * @param compression_pyarg : compression argument of read_csv/read_json
     * @return the compression scheme used, in Arrow string representation
     * (see arrow/util/compression.h/cc)
     */
    void obtain_compression_scheme(const std::string &compression_pyarg) {
        if (compression == "UNKNOWN") {
            compression = compression_pyarg;
            std::string fname;
            if (compression_pyarg == "infer") {
                if (!is_directory())
                    fname = file_path;
                else
                    // infer compression scheme from the name of the first file
                    fname = get_first_file();
                if (boost::ends_with(fname, ".gz"))
                    compression = "GZIP";  // using arrow-cpp's representation
                else if (boost::ends_with(fname, ".bz2"))
                    compression = "BZ2";  // using arrow-cpp's representation
                // ... TODO: more compression formats
                else
                    compression =
                        "UNCOMPRESSED";  // using arrow-cpp's representation
            }
        }
    }

    /// original file path passed through read_csv/read_json
    std::string file_path;
    bool is_dir;
    std::string compression = "UNKNOWN";
    std::vector<std::string> file_names;
    std::vector<int64_t> file_sizes;
    std::shared_ptr<arrow::fs::FileSystem> fs;
    /// sum of all file sizes
    int64_t total_ds_size = -1;
};

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
    if (dist_get_rank() == 0) {
        std::shared_ptr<arrow::fs::FileSystem> fs = path_info.get_fs();
        std::shared_ptr<arrow::io::RandomAccessFile> file =
            fs->OpenInputFile(fname).ValueOrDie();
#define BUF_SIZE 1024
        std::vector<char> data(BUF_SIZE);
#undef BUF_SIZE
        bool header_found = false;
        int64_t seen_size = 0;
        while (!header_found && seen_size < file_size) {
            // TODO check status
            int64_t read_size = std::min(int64_t(data.size()), file_size - seen_size);
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
    header_size += 1;
    MPI_Bcast(&header_size, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
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
    std::vector<char> data;
    /// character that constitutes the row separator for this data
    char row_separator;
    /// true if the content refers to JSON where records span multiple lines
    bool json_multi_line = false;
    /// starting offset of each row
    std::vector<int64_t> row_offsets;
    /// FileReader status
    bool status_ok = true;

    /**
     * @param row_separator : character that constitutes the row separator
     * @param json_multi_line : true if the content refers to JSON where
     *                          records span multiple lines
     */
    MemReader(char row_separator, bool json_multi_line = false)
        // note that none of the parameters passed to super class affect MemReader
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
        // note that none of the parameters passed to super class affect MemReader
        : FileReader("", false, !json_multi_line),
          row_separator(row_separator),
          json_multi_line(json_multi_line) {
        data.reserve(size);
    }

    virtual ~MemReader() {}

    /**
     * Return total size of data.
     */
    uint64_t getSize() {
        return data.size() - start;
    }

    /**
     * Read size bytes into given buffer s (from current position)
     */
    bool read(char *s, int64_t size) {
        memcpy(s, data.data() + pos, size);
        pos += size;
        return true;
    };

    /**
     * Seek pos_req bytes into data.
     */
    bool seek(int64_t pos_req) {
        pos = pos_req + start;
        status_ok = pos >= start && pos < int64_t(data.size());
        return status_ok;
    }

    /**
     * Returns reader status.
     */
    bool ok() { return status_ok; }

    /// not used by MemReader
    bool read_to_buff(char *s, int64_t size) { return false; }

    /// not used by MemReader
    bool skipHeaderRows() { return this->csv_header; };

    /**
     * Calculate row offsets for current data (fills row_offsets attribute).
     */
    void calc_row_offsets() {
        row_offsets.clear();
        row_offsets.push_back(start);  // first row starts at 'start'
        if (data.size() > 0) {
            for (int64_t i = start; i < int64_t(data.size()); i++) {
                if (data[i] == row_separator) {
                    if (i > start && data[i - 1] == row_separator)
                        // ignore empty row
                        continue;
                    row_offsets.push_back(i + 1);  // next row starts at i + 1
                }
            }
        }
    }

    /**
     * Get number of rows in data.
     */
    int64_t get_num_rows() {
        return row_offsets.size() - 1;
    }

    /**
     * Replace data with new_data (MemReader takes ownership of it).
     */
    void set_data(std::vector<char> &new_data) {
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
        std::shared_ptr<arrow::io::RandomAccessFile> file =
            fs->OpenInputFile(fname).ValueOrDie();
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
        std::shared_ptr<arrow::io::InputStream> raw_istream =
            fs->OpenInputStream(fname).ValueOrDie();
        arrow::Compression::type compression_type =
            arrow::util::Codec::GetCompressionType(compression).ValueOrDie();
        std::unique_ptr<arrow::util::Codec> codec =
            arrow::util::Codec::Create(compression_type).ValueOrDie();
        std::shared_ptr<arrow::io::CompressedInputStream> istream =
            arrow::io::CompressedInputStream::Make(codec.get(), raw_istream)
                .ValueOrDie();
#define READ_SIZE 8192
        int64_t actual_size = int64_t(data.size());
        bool skipped_header = !skip_header;
        while (true) {
            // read a chunk of READ_SIZE bytes
            data.resize(data.size() + READ_SIZE);
            int64_t bytes_read =
                istream->Read(READ_SIZE, data.data() + actual_size).ValueOrDie();
            if (bytes_read == 0) break;
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
                if (data.back() == '}') data.push_back(']');
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

    std::vector<char> &data = reader->data;
    if (rank < num_ranks - 1) {  // receive chunk from right, append to my data
        size_t cur_data_size = data.size();
        MPI_Status status;
        // probe for incoming message from rank + 1
        MPI_Probe(rank + 1, 0, MPI_COMM_WORLD, &status);
        // when probe returns, the status object has the message size
        int recv_size;
        MPI_Get_count(&status, MPI_CHAR, &recv_size);
        data.resize(cur_data_size + recv_size);
        MPI_Recv(data.data() + cur_data_size, recv_size, MPI_CHAR, rank + 1, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
            MPI_Send(data.data(), sep_idx + 1, MPI_CHAR, rank - 1, 0,
                     MPI_COMM_WORLD);
            reader->start = sep_idx + 1;
        } else {
            // I have no separator. Send all my data
            MPI_Send(data.data() + reader->start, data.size() - reader->start,
                     MPI_CHAR, rank - 1, 0, MPI_COMM_WORLD);
            data.clear();
            reader->start = 0;
        }
    }
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
    if (skiprows <= 0) return;
    // we need to know the row offsets to skip rows
    reader->calc_row_offsets();
    if (is_parallel) {
        int my_rank = dist_get_rank();
        int num_ranks = dist_get_size();
        // first allgather the number of rows on every rank
        int64_t num_rows = reader->get_num_rows();
        std::vector<int64_t> num_rows_ranks(
            num_ranks);  // number of rows in each rank
        MPI_Allgather(&num_rows, 1, MPI_INT64_T, num_rows_ranks.data(), 1,
                      MPI_INT64_T, MPI_COMM_WORLD);

        // determine the number of rows we need to skip on each rank,
        // and modify starting offset of data accordingly
        for (int rank = 0; rank < num_ranks; rank++) {
            int64_t rank_skip =
                std::min(skiprows, num_rows_ranks[rank]);
            if (rank == my_rank) {
                reader->start = reader->row_offsets[rank_skip];
                return;
            }
            skiprows -= rank_skip;
            if (skiprows == 0) break;
        }
    } else {
        // data is replicated, so skip same number of rows on every rank
        // we just need to modify the starting offset
        reader->start = reader->row_offsets[skiprows];
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
void calc_row_transfer(const std::vector<int64_t> &num_rows,
                       int64_t total_rows,
                       std::vector<std::pair<int, int64_t>> &to_send) {
    int myrank = dist_get_rank();
    int num_ranks = dist_get_size();

    int64_t start_row_global = 0;  // index of my first row in global dataset
    for (int i = 0; i < myrank; i++) {
        start_row_global += num_rows[i];
    }

    // rows_left tracks how many rows I have left to send
    int64_t rows_left = num_rows[myrank];
    if (rows_left == 0) return;

    typedef std::pair<int64_t, int64_t> range;
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
        if (std::get<0>(overlap) <= std::get<1>(overlap))
            rows_to_send = std::get<1>(overlap) - std::get<0>(overlap);
        if (rows_to_send > 0) to_send.emplace_back(rank, rows_to_send);
        rows_left -= rows_to_send;
        if (rows_left == 0) break;
    }
}

/**
 * Redistribute rows across MemReaders of all ranks to ensure that the number
 * of rows is "balanced" (matches dist_get_node_portion(total_rows, num_ranks, rank)
 * for each rank). See calc_row_transfer for details on how row transfer
 * is calculated.
 * IMPORTANT: this assumes that the data in each MemReader consists of complete
 * rows.
 * This can be called multiple times as needed.
 */
void balance_rows(MemReader *reader) {
    int myrank = dist_get_rank();
    int num_ranks = dist_get_size();

    // need the row offsets to balance rows
    reader->calc_row_offsets();

    // first allgather the number of rows on every rank
    int64_t num_rows = reader->get_num_rows();
    std::vector<int64_t> num_rows_ranks(
        num_ranks);  // number of rows in each rank
    MPI_Allgather(&num_rows, 1, MPI_INT64_T, num_rows_ranks.data(), 1,
                  MPI_INT64_T, MPI_COMM_WORLD);

    // check that all ranks have same number of rows. in that case there is no
    // need to do anything
    auto result = std::minmax_element(num_rows_ranks.begin(), num_rows_ranks.end());
    int64_t min = *result.first;
    int64_t max = *result.second;
    if (min == max) return;  // already balanced

    // get total number of rows in global dataset
    int64_t total_rows =
        std::accumulate(num_rows_ranks.begin(), num_rows_ranks.end(), 0);

    // by default don't send or receive anything. this is changed below as
    // needed
    std::vector<int> sendcounts(num_ranks, 0);
    std::vector<int> recvcounts(num_ranks, 0);
    std::vector<int> sdispls(num_ranks, 0);
    std::vector<int> rdispls(num_ranks, 0);

    // calc send counts
    std::vector<std::pair<int, int64_t>>
        to_send;  // vector of (rank, nrows)  meaning send nrows to rank
    calc_row_transfer(num_rows_ranks, total_rows, to_send);
    int cur_offset = 0;
    int64_t cur_row = 0;
    for (auto rank_rows : to_send) {
        int rank = rank_rows.first;
        int64_t rows = rank_rows.second;
        int num_bytes = 0;
        for (int i = cur_row; i < cur_row + rows; i++)
            num_bytes += reader->row_offsets[i + 1] - reader->row_offsets[i];
        sendcounts[rank] = num_bytes;
        sdispls[rank] = cur_offset;
        cur_offset += num_bytes;
        cur_row += rows;
    }

    // get recv count
    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT,
                 MPI_COMM_WORLD);

    // have to receive rows from other processes
    int64_t total_recv_size = 0;
    cur_offset = 0;
    for (int rank = 0; rank < num_ranks; rank++) {
        rdispls[rank] = cur_offset;
        cur_offset += recvcounts[rank];
        total_recv_size += recvcounts[rank];
    }
    std::vector<char> recvbuf(total_recv_size);
    char *sendbuf = reader->data.data() + reader->start;
    MPI_Alltoallv(sendbuf, sendcounts.data(), sdispls.data(), MPI_CHAR,
                  recvbuf.data(), recvcounts.data(), rdispls.data(), MPI_CHAR,
                  MPI_COMM_WORLD);

    reader->set_data(recvbuf);  // mem reader takes ownership of the buffer
}

/**
 * Read my chunk of CSV/JSON dataset.
 * @param fname : path specifying *all* CSV/JSON file(s) to read (not
 *                just my files)
 * @param suffix : "csv" or "json"
 * @param is_parallel : indicates whether data is distributed or replicated
 * @param skiprows : number of rows to skip in global dataset
 * @param json_lines : true if JSON file is in JSON Lines format (one row/record per line)
 * @param csv_header : true if CSV files contain headers
 * @param compression_pyarg : compression scheme
 */
extern "C" PyObject *file_chunk_reader(const char *fname, const char *suffix,
                                       bool is_parallel, int64_t skiprows,
                                       int64_t nrows, bool json_lines,
                                       bool csv_header,
                                       const char *compression_pyarg) {
    // TODO nrows looks like it is not used. remove

    // TODO check that skiprows >= 0

    CHECK(fname != NULL, "NULL filename provided.");

    // TODO right now we get the list of file names and file sizes on rank 0
    // and broadcast to every process. This is potentially not scalable (think
    // many millions of potentially long file names) and not really necessary
    // (we could scatter instead of broadcast). But this doesn't seem like
    // something that should worry us right now

    char row_separator = '\n';
    if (strcmp(suffix, "json") == 0) row_separator = '}';

    int rank = dist_get_rank();
    int num_ranks = dist_get_size();
    MemReader *mem_reader = nullptr;

    PathInfo path_info(fname, compression_pyarg);
    const std::string compression = path_info.get_compression_scheme();

    if (compression == "UNCOMPRESSED") {
        const std::vector<std::string> &file_names = path_info.get_file_names();
        const std::vector<int64_t> &file_sizes = path_info.get_file_sizes();
        int64_t header_size_bytes = 0;
        if (csv_header)
            header_size_bytes =
                get_header_size(file_names[0], file_sizes[0], path_info, '\n');
        // total size excluding headers
        int64_t total_size =
            path_info.get_size() - (file_names.size() * header_size_bytes);
        // now determine which files to read from and what portion from each file
        // this is based on partitioning of global dataset based on bytes.
        // As such, this first phase can end up with rows of data in multiple
        // processes
        int64_t start_global = 0;
        int64_t end_global = total_size;
        if (is_parallel) {
            start_global = dist_get_start(total_size, num_ranks, rank);
            end_global = dist_get_end(total_size, num_ranks, rank);
        }
        int64_t to_read = end_global - start_global;
        mem_reader = new MemReader(to_read, row_separator, !json_lines);
        int64_t cur_size = 0;
        // find first file to read from and read from it
        int64_t cur_file_idx = 0;
        for (size_t i = 0; i < file_sizes.size(); i++) {
            int64_t fsize = file_sizes[i] - header_size_bytes;
            if (cur_size + fsize > start_global) {
                int64_t file_start =
                    start_global - cur_size + header_size_bytes;
                int64_t file_end =
                    file_start +
                    std::min(to_read, fsize + header_size_bytes - file_start);
                mem_reader->read_uncompressed_file(
                    file_names[i], file_start, file_end, path_info.get_fs());
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
                f_to_read + header_size_bytes, path_info.get_fs());
            to_read -= f_to_read;
            cur_file_idx += 1;
        }
        // correct data so that each rank only has complete rows
        if (is_parallel) data_row_correction(mem_reader, row_separator);
    } else {
        // if files are compressed, one rank will be responsible for decompressing
        // a whole file into memory. Data will later be redistributed (see
        // balance_rows below)
        const std::vector<std::string> &file_names = path_info.get_file_names();
        mem_reader = new MemReader(row_separator, !json_lines);
        int64_t num_files = file_names.size();
        if (is_parallel && num_files < num_ranks) {
            // try to space the read across nodes to avoid memory issues
            // if decompressing huge files
            int64_t ppf = num_ranks / num_files;
            if (rank % ppf == 0) {
                // I read a file
                int64_t my_file = rank / ppf;
                if (my_file < num_files)
                    mem_reader->read_compressed_file(file_names[my_file],
                                                     path_info.get_fs(),
                                                     compression, csv_header);
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
                mem_reader->read_compressed_file(
                    file_names[i], path_info.get_fs(), compression, csv_header);
            }
        }
    }

    // skip rows if requested
    skip_rows(mem_reader, skiprows, is_parallel);

    // shuffle data so that each rank has required number of rows
    if (is_parallel) balance_rows(mem_reader);

    // prepare data for pandas
    mem_reader->finalize();

    // now create a stream reader PyObject that wraps MemReader, to be read from
    // pandas
    auto gilstate = PyGILState_Ensure();
    PyObject *reader =
        PyObject_CallFunctionObjArgs((PyObject *)&stream_reader_type, NULL);
    PyGILState_Release(gilstate);
    if (reader == NULL || PyErr_Occurred()) {
        PyErr_Print();
        std::cerr << "Could not create chunk reader object" << std::endl;
        if (reader) delete reader;
        reader = NULL;
    } else {
        stream_reader_init(reinterpret_cast<stream_reader *>(reader),
                           mem_reader, 0, mem_reader->getSize());
    }
    return reader;
}

extern "C" PyObject *csv_file_chunk_reader(const char *fname, bool is_parallel,
                                           int64_t skiprows, int64_t nrows,
                                           bool header, const char *compression) {
    // TODO nrows not used??
    return file_chunk_reader(fname, "csv", is_parallel, skiprows, nrows, true,
                             header, compression);
}

extern "C" PyObject *json_file_chunk_reader(const char *fname, bool lines,
                                            bool is_parallel, int64_t nrows,
                                            const char *compression) {
    // TODO nrows not used??
    return file_chunk_reader(fname, "json", is_parallel, 0, nrows, lines,
                             false, compression);
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

#undef CHECK
