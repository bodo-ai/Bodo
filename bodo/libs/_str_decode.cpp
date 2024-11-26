// Copyright (C) 2019 Bodo Inc. All rights reserved.
#include <Python.h>
#include "_meminfo.h"

#ifndef Py_UNREACHABLE
#define Py_UNREACHABLE() abort()
#endif

// Python 3.12 removed PyUnicode_WCHAR_KIND so replacing PyUnicode_Kind to
// keep existing code working
// https://github.com/python/cpython/blob/cce6ba91b3a0111110d7e1db828bd6311d58a0a7/Include/cpython/unicodeobject.h#L304
enum _PyUnicode_Kind {
    /* String contains only wstr byte characters.  This is only possible
       when the string was created with a legacy API and _PyUnicode_Ready()
       has not been called yet.  */
    _PyUnicode_WCHAR_KIND = 0,
    /* Return values of the PyUnicode_KIND() function: */
    _PyUnicode_1BYTE_KIND = 1,
    _PyUnicode_2BYTE_KIND = 2,
    _PyUnicode_4BYTE_KIND = 4
};

// ******** ported from CPython 31e8d69bfe7cf5d4ffe0967cb225d2a8a229cc97

typedef struct {
    NRT_MemInfo *buffer;
    void *data;
    enum _PyUnicode_Kind kind;
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

void _C_UnicodeWriter_Init(_C_UnicodeWriter *writer) {
    memset(writer, 0, sizeof(*writer));

    /* ASCII is the bare minimum */
    writer->min_char = 127;

    /* use a value smaller than _PyUnicode_1BYTE_KIND() so
       _C_UnicodeWriter_PrepareKind() will copy the buffer. */
    writer->kind = _PyUnicode_WCHAR_KIND;
    writer->is_ascii = 0;
    assert(writer->kind <= _PyUnicode_1BYTE_KIND);
}

#ifdef MS_WINDOWS
/* On Windows, overallocate by 50% is the best factor */
#define OVERALLOCATE_FACTOR 2
#else
/* On Linux, overallocate by 25% is the best factor */
#define OVERALLOCATE_FACTOR 4
#endif

/* Maximum code point of Unicode 6.0: 0x10ffff (1,114,111) */
#define MAX_UNICODE 0x10ffff

#define KIND_MAX_CHAR_VALUE(kind)  \
    (kind == _PyUnicode_1BYTE_KIND \
         ? (0xffU)                 \
         : (kind == _PyUnicode_2BYTE_KIND ? (0xffffU) : (0x10ffffU)))

// clang-format off
#include "vendored/stringlib/bytesobject.cpp"

#include "vendored/stringlib/asciilib.h"
#include "vendored/stringlib/codecs.h"
#include "vendored/stringlib/undef.h"

#include "vendored/stringlib/ucs1lib.h"
#include "vendored/stringlib/codecs.h"
#include "vendored/stringlib/undef.h"

#include "vendored/stringlib/ucs2lib.h"
#include "vendored/stringlib/codecs.h"
#include "vendored/stringlib/undef.h"

#include "vendored/stringlib/ucs4lib.h"
#include "vendored/stringlib/codecs.h"
#include "vendored/stringlib/undef.h"

// clang-format on
int64_t unicode_to_utf8(char *out_data, char *data, int64_t size, int kind) {
    //
    switch (kind) {
        default:
            Py_UNREACHABLE();
        case _PyUnicode_1BYTE_KIND:
            return ucs1lib_utf8_encoder(out_data, (Py_UCS1 *)data, size);
        case _PyUnicode_2BYTE_KIND:
            return ucs2lib_utf8_encoder(out_data, (Py_UCS2 *)data, size);
        case _PyUnicode_4BYTE_KIND:
            return ucs4lib_utf8_encoder(out_data, (Py_UCS4 *)data, size);
    }
}
