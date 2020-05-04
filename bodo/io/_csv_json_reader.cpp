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
    LocalFileReader(const char *_fname, const char *f_type)
        : SingleFileReader(_fname, f_type) {
        this->fstream = new std::ifstream(fname);
        CHECK(fstream->good() && !fstream->eof() && fstream->is_open(),
              "could not open file.");
    }
    uint64_t getSize() { return boost::filesystem::file_size(fname); }
    bool seek(int64_t pos) {
        this->fstream->seekg(pos, std::ios_base::beg);
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
    LocalDirectoryFileReader(const char *_dirname, const char *f_type)
        : DirectoryFileReader(_dirname, f_type) {
        // only keep the files that are csv/json files
        std::string dirname(_dirname);
        const char *suffix = this->f_type_to_stirng();

        auto endsWithSuffix =
            [suffix](const boost::filesystem::path &path) -> bool {
            return (boost::filesystem::is_regular_file(path) &&
                    boost::ends_with(path.string(), suffix));
        };

        std::copy_if(boost::filesystem::directory_iterator(_dirname),
                     boost::filesystem::directory_iterator(),
                     std::back_inserter(this->file_paths), endsWithSuffix);

        if ((this->file_paths).size() == 0) {
            Bodo_PyErr_SetString(
                PyExc_RuntimeError,
                ("No valid file to read from directory: " + dirname).c_str());
        }
        // sort all files in directory
        std::sort(this->file_paths.begin(), this->file_paths.end());

        // find dir_size and construct file_sizes
        // assuming the directory contains files only, i.e. no subdirectory
        this->dir_size = 0;
        for (auto it = this->file_paths.begin(); it != this->file_paths.end();
             ++it) {
            (this->file_sizes).push_back(this->dir_size);
            (this->file_names).push_back((*it).string());
            this->dir_size += boost::filesystem::file_size(*it);
        }
        this->file_sizes.push_back(this->dir_size);
    };

    void initFileReader(const char *fname) {
        this->f_reader = new LocalFileReader(fname, this->f_type_to_stirng());
        this->f_reader->json_lines = this->json_lines;
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

// 5MB buffer size to read from sources like AWS S3 in case they don't have
// buffering
static constexpr size_t BUFF_SIZE = 5 * 1024 * 1024;

// return vector of offsets of separaters in first n bytes of given stream
// csv & json(orient = 'recort', lines=True): separator is "\n"
// json(orient = 'recort', lines=False): separator is "},"
//                                       and "}]" indicated end of file
static std::vector<size_t> count_entries(FileReader *f, size_t n) {
    std::vector<size_t> pos;
    char separator;
    size_t i = 0;
    char *buffer = new char[BUFF_SIZE];
    size_t rank = dist_get_rank();
    bool quot = false;
    bool linebreak = false;
    // set separator based on file type
    if (f->f_type == File_Type::csv)
        separator = '\n';
    else {
        assert(f->f_type == File_Type::json);
        if (f->json_lines)
            separator = '\n';
        else
            separator = '}';
    }
    while (i < n) {
        size_t n_read = std::min(n - i, BUFF_SIZE);
        bool ok = f->read(buffer, n_read);
        if (!ok) break;
        for (size_t j = 0; j < n_read; j++) {
            if (buffer[j] == '\"' && separator == '\n') quot = !quot;
            if (buffer[j] == separator) {
                if (quot) linebreak = true;
                if (separator == '}') {
                    // last character is }
                    if (j >= n_read - 1) {
                        Bodo_PyErr_SetString(PyExc_RuntimeError,
                                             "unexpected ending in json file");
                    }
                    if (buffer[j + 1] != ',' && buffer[j + 1] != ']') {
                        Bodo_PyErr_SetString(
                            PyExc_RuntimeError,
                            "unexpected format in json file, '}' is not "
                            "followed by a ',' or a ']'");
                    }
                    j++;  // for ',' or ']'
                    if (buffer[j + 1] < n_read && buffer[j + 1] == '\n') {
                        j++;
                    }
                }
                pos.push_back(i + j);
            }
            // case where json output does not end with a new line
            // i.e. pandas.to_json output
            if (rank == dist_get_size() - 1 && (i + j == n - 1) &&
                buffer[j] != separator && separator == '\n') {
                pos.push_back(i + j);
            }
        }
        i += n_read;
    }
    if (rank == 0 && linebreak)
        std::cerr
            << "Line break within the columns of CSV/JSON file in distributed "
               "mode is NOT supported"
            << std::endl;

    if (i < n)
        std::cerr << "Warning, read only " << i << " bytes out of " << n
                  << "requested\n";
    delete[] buffer;
    return pos;
}

/**
 * Split stream into chunks and return a file-like object per rank. The returned
 *object
 * represents the data to be read on each process.
 *
 * We evenly distribute by number of lines by working on byte-chunks in parallel
 *   * counting new-lines and allreducing and exscaning numbers
 *   * computing start/end points of desired chunks-of-lines and sending them to
 *corresponding ranks.
 * Using dist_get_size and dist_get_start to compute chunk start/end/size as
 *well as
 * the final chunking of lines.
 *
 * @param[in]  f   the input stream
 * @param[in]  fsz total number of bytes in stream
 * @return     StreamReader file-like object to read the owned chunk through
 *pandas.read_csv, pandas.read_json
 **/
static PyObject *chunk_reader(FileReader *f, size_t fsz, bool is_parallel,
                              int64_t skiprows, int64_t nrows) {
    if (skiprows < 0) {
        std::cerr << "Invalid skiprows argument: " << skiprows << std::endl;
        return NULL;
    }
    // printf("rank %d skiprows %d nrows %d\n", dist_get_rank(), skiprows,
    // nrows);

    size_t nranks = dist_get_size();
    size_t my_off_start = 0;
    size_t my_off_end = fsz;

    if (is_parallel && nranks > 1) {
        size_t rank = dist_get_rank();

        // seek to our chunk
        size_t byte_offset = dist_get_start(fsz, nranks, rank);
        f->seek(byte_offset);
        if (!f->ok()) {
            Bodo_PyErr_SetString(PyExc_RuntimeError,
                                 "Could not seek to start position");
            return NULL;
        }
        // We evenly distribute the 'data' byte-wise
        // count number of lines in chunk
        // TODO: count only until nrows
        std::vector<size_t> line_offset =
            count_entries(f, dist_get_node_portion(fsz, nranks, rank));
        size_t no_lines = line_offset.size();
        // get total number of lines using allreduce
        size_t tot_no_lines(0);

        dist_reduce(reinterpret_cast<char *>(&no_lines),
                    reinterpret_cast<char *>(&tot_no_lines),
                    HPAT_ReduceOps::SUM, Bodo_CTypes::UINT64);

        // Now we need to communicate the distribution as we really want it
        // First determine which is our first line (which is the sum of previous
        // lines)
        size_t byte_first_line(0);
        dist_exscan(reinterpret_cast<char *>(&no_lines),
                    reinterpret_cast<char *>(&byte_first_line),
                    HPAT_ReduceOps::SUM, Bodo_CTypes::UINT64);
        size_t byte_last_line = byte_first_line + no_lines;

        // We now determine the chunks of lines that begin and end in our
        // byte-chunk

        // issue IRecv calls, eventually receiving start and end offsets of our
        // line-chunk
        const int START_OFFSET = 47011;
        const int END_OFFSET = 47012;
        std::vector<MPI_Request> mpi_reqs;
        mpi_reqs.push_back(dist_irecv(&my_off_start, 1, Bodo_CTypes::UINT64,
                                      MPI_ANY_SOURCE, START_OFFSET,
                                      (rank > 0 || skiprows > 0)));
        mpi_reqs.push_back(dist_irecv(&my_off_end, 1, Bodo_CTypes::UINT64,
                                      MPI_ANY_SOURCE, END_OFFSET,
                                      ((rank < (nranks - 1)) || nrows != -1)));

        // check nrows argument
        if (nrows != -1 && (nrows < 0 || size_t(nrows) > tot_no_lines)) {
            std::cerr << "Invalid nrows argument: " << nrows
                      << " for total number of lines: " << tot_no_lines
                      << std::endl;
            return NULL;
        }

        // number of lines that actually needs to be parsed
        size_t n_lines_to_read = nrows != -1 ? nrows : tot_no_lines - skiprows;
        // TODO skiprows and nrows need testing
        // send start offset of rank 0
        std::vector<size_t> list_off(2 + nranks);
        size_t* list_off_ptr = list_off.data();
        size_t idx_off=0;
        if (size_t(skiprows) > byte_first_line &&
            size_t(skiprows) <= byte_last_line) {
            size_t i_off = byte_offset +
                           line_offset[skiprows - byte_first_line - 1] +
                           1;  // +1 to skip/include leading/trailing newline
            list_off[idx_off] = i_off;
            size_t* i_ptr = list_off_ptr + idx_off;
            mpi_reqs.push_back(dist_isend(i_ptr, 1, Bodo_CTypes::UINT64, 0,
                                          START_OFFSET, true));
            idx_off++;
        }

        // send end offset of rank n-1
        if (size_t(nrows) > byte_first_line &&
            size_t(nrows) <= byte_last_line) {
            size_t i_off = byte_offset +
                           line_offset[nrows - byte_first_line - 1] +
                           1;  // +1 to skip/include leading/trailing newline
            list_off[idx_off] = i_off;
            size_t* i_ptr = list_off_ptr + idx_off;
            mpi_reqs.push_back(dist_isend(i_ptr, 1, Bodo_CTypes::UINT64,
                                          nranks - 1, END_OFFSET, true));
            idx_off++;
        }

        // We iterate through chunk boundaries (defined by line-numbers)
        // we start with boundary 1 as 0 is the beginning of file
        for (int i = 1; i < int(nranks); ++i) {
            size_t i_bndry =
                skiprows + dist_get_start(n_lines_to_read, (int)nranks, i);
            // Note our line_offsets mark the end of each line!
            // we check if boundary is on our byte-chunk
            if (i_bndry > byte_first_line && i_bndry <= byte_last_line) {
                // if so, send stream-offset to ranks which start/end here
                size_t i_off =
                    byte_offset + line_offset[i_bndry - byte_first_line - 1] +
                    1;  // +1 to skip/include leading/trailing newline
                list_off[idx_off] = i_off;
                size_t* i_ptr = list_off_ptr + idx_off;
                // send to rank that starts at this boundary: i
                mpi_reqs.push_back(dist_isend(i_ptr, 1, Bodo_CTypes::UINT64, i,
                                              START_OFFSET, true));
                // send to rank that ends at this boundary: i-1
                mpi_reqs.push_back(dist_isend(i_ptr, 1, Bodo_CTypes::UINT64,
                                              i - 1, END_OFFSET, true));
                idx_off++;
            } else {
                // if not and we past our chunk -> we stop
                if (i_bndry > byte_last_line) break;
            }  // else we are before our chunk -> continue iteration
        }
        // before reading, make sure we received our start/end offsets
        dist_waitall(mpi_reqs.size(), mpi_reqs.data());
    }  // if is_parallel
    else if (skiprows > 0 || nrows != -1) {
        f->seek(0);
        if (!f->ok()) {
            Bodo_PyErr_SetString(PyExc_RuntimeError,
                                 "Could not seek to start position");
        }
        std::vector<size_t> line_offset = count_entries(f, fsz);
        if (skiprows > 0) my_off_start = line_offset[skiprows - 1] + 1;
        if (nrows != -1) my_off_end = line_offset[nrows - 1] + 1;
    }

    // Here we now know exactly what chunk to read: [my_off_start,my_off_end[
    // let's create our file-like reader
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
        stream_reader_init(reinterpret_cast<stream_reader *>(reader), f,
                           my_off_start, my_off_end - my_off_start);
    }
    return reader;
}

typedef FileReader *(*s3_reader_init_t)(const char *, const char *);
typedef FileReader *(*hdfs_reader_init_t)(const char *, const char *);

// taking a file to create a istream and calling chunk_reader
extern "C" PyObject *file_chunk_reader(const char *fname, const char *suffix,
                                       bool is_parallel, int64_t skiprows,
                                       int64_t nrows, bool json_lines) {
    CHECK(fname != NULL, "NULL filename provided.");
    FileReader *f_reader;
    PyObject *f_mod;
    PyObject *func_obj;
    uint64_t fsz = -1;

    if (strncmp("s3://", fname, 5) == 0) {
        // load s3_reader module if path starts with s3://
        import_fs_module(Bodo_Fs::s3, suffix, f_mod);
        get_fs_reader_pyobject(Bodo_Fs::s3, suffix, f_mod, func_obj);

        s3_reader_init_t func =
            (s3_reader_init_t)PyNumber_AsSsize_t(func_obj, NULL);
        f_reader = func(fname + 5, suffix);

        Py_DECREF(f_mod);
        Py_DECREF(func_obj);
    } else if (strncmp("hdfs://", fname, 7) == 0) {
        // load hdfs_reader module if path starts with hdfs://
        import_fs_module(Bodo_Fs::hdfs, suffix, f_mod);
        get_fs_reader_pyobject(Bodo_Fs::hdfs, suffix, f_mod, func_obj);

        hdfs_reader_init_t func =
            (hdfs_reader_init_t)PyNumber_AsSsize_t(func_obj, NULL);
        f_reader = func(fname, suffix);

        Py_DECREF(f_mod);
        Py_DECREF(func_obj);
    } else {
        if (boost::filesystem::is_directory(fname)) {
            f_reader = new LocalDirectoryFileReader(fname, suffix);
        } else {
            f_reader = new LocalFileReader(fname, suffix);
        }
        CHECK(f_reader->ok(), "could not open file.");
    }
    fsz = f_reader->getSize();
    f_reader->json_lines = json_lines;
    return chunk_reader(f_reader, fsz, is_parallel, skiprows, nrows);
}

extern "C" PyObject *csv_file_chunk_reader(const char *fname, bool is_parallel,
                                           int64_t skiprows, int64_t nrows) {
    return file_chunk_reader(fname, "csv", is_parallel, skiprows, nrows, true);
}

extern "C" PyObject *json_file_chunk_reader(const char *fname, bool lines,
                                            bool is_parallel, int64_t nrows) {
    return file_chunk_reader(fname, "json", is_parallel, 0, nrows, lines);
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
