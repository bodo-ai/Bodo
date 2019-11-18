// Copyright (C) 2019 Bodo Inc. All rights reserved.
/*
 * Implementation of dictionaries using std::unordered_map.
 *
 * Provides most common maps of simple data types:
 *   {int*, double, float, string} -> {int*, double, float, string}
 * C-Functions are exported as Python module, types are part of their names
 *   dict_<key-type>_<value-type_{init, setitem, getitem, in}.
 * Also provides a dict which maps a byte-array to a int64.
 *
 * We define our own dictionary template class.
 * To get external C-functions per key/value-type we use a macro-factory
 * which generates C-Functions calling our C++ dictionary.
 */

#include <Python.h>
#include <boost/preprocessor/stringize.hpp>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <unordered_map>

// declaration of dict functions in python module
#define DEC_MOD_METHOD(func)                            \
    PyObject_SetAttrString(m, BOOST_PP_STRINGIZE(func), \
                           PyLong_FromVoidPtr((void*)(&func)))



// multimap for hash join
typedef std::unordered_multimap<int64_t, int64_t> multimap_int64_t;
typedef std::pair<multimap_int64_t::iterator, multimap_int64_t::iterator>
    multimap_int64_it_t;

multimap_int64_t* multimap_int64_init() { return new multimap_int64_t(); }

void multimap_int64_insert(multimap_int64_t* m, int64_t k, int64_t v) {
    m->insert(std::make_pair(k, v));
    return;
}

multimap_int64_it_t* multimap_int64_equal_range(multimap_int64_t* m,
                                                int64_t k) {
    return new multimap_int64_it_t(m->equal_range(k));
}

multimap_int64_it_t* multimap_int64_equal_range_alloc() {
    return new multimap_int64_it_t;
}

void multimap_int64_equal_range_dealloc(multimap_int64_it_t* r) { delete r; }

void multimap_int64_equal_range_inplace(multimap_int64_t* m, int64_t k,
                                        multimap_int64_it_t* r) {
    *r = m->equal_range(k);
}

// auto range = map.equal_range(1);
// for (auto it = range.first; it != range.second; ++it) {
//     std::cout << it->first << ' ' << it->second << '\n';
// }

bool multimap_int64_it_is_valid(multimap_int64_it_t* r) {
    return r->first != r->second;
}

int64_t multimap_int64_it_get_value(multimap_int64_it_t* r) {
    return (r->first)->second;
}

void multimap_int64_it_inc(multimap_int64_it_t* r) {
    (r->first)++;
    return;
}

// module initiliziation
// make our C-functions available
PyMODINIT_FUNC PyInit_hdict_ext(void) {
    PyObject* m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "hdict_ext", "No docs", -1, NULL,
    };
    m = PyModule_Create(&moduledef);
    if (m == NULL) return NULL;

    DEC_MOD_METHOD(multimap_int64_init);
    DEC_MOD_METHOD(multimap_int64_insert);
    DEC_MOD_METHOD(multimap_int64_equal_range);
    DEC_MOD_METHOD(multimap_int64_it_is_valid);
    DEC_MOD_METHOD(multimap_int64_it_get_value);
    DEC_MOD_METHOD(multimap_int64_it_inc);
    DEC_MOD_METHOD(multimap_int64_equal_range_alloc);
    DEC_MOD_METHOD(multimap_int64_equal_range_dealloc);
    DEC_MOD_METHOD(multimap_int64_equal_range_inplace);
    return m;
}
