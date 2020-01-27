// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include <Python.h>
#include <iostream>
#include "_meminfo.h"


#ifndef Py_UNREACHABLE
#define Py_UNREACHABLE() abort()
#endif


// ******** ported from CPython 31e8d69bfe7cf5d4ffe0967cb225d2a8a229cc97

typedef struct {
    NRT_MemInfo *buffer;
    void *data;
    enum PyUnicode_Kind kind;
    int is_ascii;
    Py_UCS4 maxchar;
    Py_ssize_t size;
    Py_ssize_t pos;

    /* minimum number of allocated characters (default: 0) */
    Py_ssize_t min_length;

    /* minimum character (default: 127, ASCII) */
    Py_UCS4 min_char;

    /* If non-zero, overallocate the buffer (default: 0). */
    unsigned char overallocate;

    /* If readonly is 1, buffer is a shared string (cannot be modified)
       and size is set to 0. */
    unsigned char readonly;
} _C_UnicodeWriter;


void
_C_UnicodeWriter_Init(_C_UnicodeWriter *writer)
{
    memset(writer, 0, sizeof(*writer));

    /* ASCII is the bare minimum */
    writer->min_char = 127;

    /* use a value smaller than PyUnicode_1BYTE_KIND() so
       _C_UnicodeWriter_PrepareKind() will copy the buffer. */
    writer->kind = PyUnicode_WCHAR_KIND;
    writer->is_ascii = 0;
    assert(writer->kind <= PyUnicode_1BYTE_KIND);
}

#ifdef MS_WINDOWS
   /* On Windows, overallocate by 50% is the best factor */
#  define OVERALLOCATE_FACTOR 2
#else
   /* On Linux, overallocate by 25% is the best factor */
#  define OVERALLOCATE_FACTOR 4
#endif

/* Maximum code point of Unicode 6.0: 0x10ffff (1,114,111) */
#define MAX_UNICODE 0x10ffff



#define KIND_MAX_CHAR_VALUE(kind) \
      (kind == PyUnicode_1BYTE_KIND ?                                   \
       (0xffU) :                                                        \
       (kind == PyUnicode_2BYTE_KIND ?                                  \
        (0xffffU) :                                                     \
        (0x10ffffU)))

#include "stringlib/bytesobject.cpp"

#include "stringlib/asciilib.h"
#include "stringlib/codecs.h"
#include "stringlib/undef.h"

#include "stringlib/ucs1lib.h"
#include "stringlib/codecs.h"
#include "stringlib/undef.h"

#include "stringlib/ucs2lib.h"
#include "stringlib/codecs.h"
#include "stringlib/undef.h"

#include "stringlib/ucs4lib.h"
#include "stringlib/codecs.h"
#include "stringlib/undef.h"


int64_t unicode_to_utf8(char* out_data, char* data, int64_t size, int kind)
{
    //
    switch (kind) {
    default:
        Py_UNREACHABLE();
    case PyUnicode_1BYTE_KIND:
        return ucs1lib_utf8_encoder(out_data, (Py_UCS1 *)data, size);
    case PyUnicode_2BYTE_KIND:
        return ucs2lib_utf8_encoder(out_data, (Py_UCS2 *)data, size);
    case PyUnicode_4BYTE_KIND:
        return ucs4lib_utf8_encoder(out_data, (Py_UCS4 *)data, size);
    }
}
