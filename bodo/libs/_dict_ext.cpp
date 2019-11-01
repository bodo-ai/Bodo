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
#include <algorithm>
#include <boost/functional/hash/hash.hpp>
#include <boost/preprocessor/list/for_each_product.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <random>
#include <unordered_map>
#include <vector>

// we need a few typedefs to make our macro factory work
// It requires types to end with '_t'
typedef std::string unicode_type_t;
typedef bool bool_t;
typedef int int_t;
typedef float float32_t;
typedef double float64_t;


template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
    if (!v.empty()) {
        out << '[';
        std::copy(v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
        out << "\b\b]";
    }
    return out;
}

// Type trait to allow by pointer/reference/value distinction for different
// types.
// Some keys/values are passed by value, others by reference(pointer!)
// This template struct defines how the dict keys/values appear on the dict
// interface
// Generic struct defines the defaults as by-value
template <typename T>
struct IFTYPE {
    typedef T in_t;
    typedef T out_t;
    static out_t out(T& o) { return o; }
};

// strings appear by reference/pointer on the interface
template <>
struct IFTYPE<std::string> {
    typedef std::string& in_t;
    typedef std::string* out_t;
    static out_t out(std::string& o) { return &o; }
};


// Generic template dict class
template <typename IDX, typename VAL>
class dict {
   private:
    std::unordered_map<IDX, VAL> m_dict;

   public:
    typedef typename IFTYPE<IDX>::in_t idx_in_t;
    typedef typename IFTYPE<IDX>::out_t idx_out_t;
    typedef typename IFTYPE<VAL>::in_t val_in_t;
    typedef typename IFTYPE<VAL>::out_t val_out_t;

    dict() : m_dict() {}

    // sets given value for given index
    void setitem(idx_in_t index, val_in_t value) {
        m_dict[index] = value;
        return;
    }

    // @return value for given index, entry must exist
    val_out_t getitem(const idx_in_t index) {
        return IFTYPE<VAL>::out(m_dict.at(index));
    }

    // @return true if given index is found in dict, false otherwise
    bool in(const idx_in_t index) {
        return (m_dict.find(index) != m_dict.end());
    }

    // print the entire dict
    void print() {
        // TODO: return python string and print in native mode
        for (auto& x : m_dict) {
            std::cout << x.first << ": " << x.second << std::endl;
        }
        return;
    }

    // @return value for given index or default_val if not in dict
    val_out_t get(const idx_in_t index, val_in_t default_val) {
        auto val = m_dict.find(index);
        if (val == m_dict.end()) return IFTYPE<VAL>::out(default_val);
        return IFTYPE<VAL>::out((*val).second);
    }

    // deletes entry from dict
    // @return value for given index
    val_out_t pop(const idx_in_t index) {
        auto val = IFTYPE<VAL>::out(m_dict.at(index));
        m_dict.erase(index);
        return val;
    }

    void* keys() {
        // TODO: return actual iterator
        return this;
    }

    // @return maximum value (not key!) in dict
    val_out_t min() {
        // TODO: use actual iterator
        auto res = std::numeric_limits<VAL>::max();
        typename std::unordered_map<IDX, VAL>::iterator it = m_dict.end();
        for (typename std::unordered_map<IDX, VAL>::iterator x = m_dict.begin();
             x != m_dict.end(); ++x) {
            if (x->second < res) {
                res = x->second;
                it = x;
            }
        }
        return IFTYPE<VAL>::out(it->second);
    }

    // @return maximum value (not key!) in dict
    val_out_t max() {
        // TODO: use actual iterator
        auto res = std::numeric_limits<VAL>::min();
        typename std::unordered_map<IDX, VAL>::iterator it = m_dict.end();
        for (typename std::unordered_map<IDX, VAL>::iterator x = m_dict.begin();
             x != m_dict.end(); ++x) {
            if (x->second > res) {
                res = x->second;
                it = x;
            }
        }
        return IFTYPE<VAL>::out(it->second);
    }

    // @return true if dict is not empty, false otherwise
    bool not_empty() { return !m_dict.empty(); }
};

// macro expanding to C-functions
#define DEF_DICT(_IDX_, _VAL_)                                                \
    dict<_IDX_##_t, _VAL_##_t>* dict_##_IDX_##_##_VAL_##_init() {             \
        return new dict<_IDX_##_t, _VAL_##_t>();                              \
    }                                                                         \
    void dict_##_IDX_##_##_VAL_##_setitem(dict<_IDX_##_t, _VAL_##_t>* m,      \
                                          IFTYPE<_IDX_##_t>::in_t index,      \
                                          IFTYPE<_VAL_##_t>::in_t value) {    \
        m->setitem(index, value);                                             \
    }                                                                         \
    IFTYPE<_VAL_##_t>::out_t dict_##_IDX_##_##_VAL_##_getitem(                \
        dict<_IDX_##_t, _VAL_##_t>* m, const IFTYPE<_IDX_##_t>::in_t index) { \
        return m->getitem(index);                                             \
    }                                                                         \
    bool dict_##_IDX_##_##_VAL_##_in(dict<_IDX_##_t, _VAL_##_t>* m,           \
                                     const IFTYPE<_IDX_##_t>::in_t index) {   \
        return m->in(index);                                                  \
    }                                                                         \
    void dict_##_IDX_##_##_VAL_##_print(dict<_IDX_##_t, _VAL_##_t>* m) {      \
        m->print();                                                           \
    }                                                                         \
    IFTYPE<_VAL_##_t>::out_t dict_##_IDX_##_##_VAL_##_get(                    \
        dict<_IDX_##_t, _VAL_##_t>* m, const IFTYPE<_IDX_##_t>::in_t index,   \
        IFTYPE<_VAL_##_t>::in_t default_val) {                                \
        return m->get(index, default_val);                                    \
    }                                                                         \
    IFTYPE<_VAL_##_t>::out_t dict_##_IDX_##_##_VAL_##_pop(                    \
        dict<_IDX_##_t, _VAL_##_t>* m, const IFTYPE<_IDX_##_t>::in_t index) { \
        return m->pop(index);                                                 \
    }                                                                         \
    IFTYPE<_VAL_##_t>::out_t dict_##_IDX_##_##_VAL_##_min(                    \
        dict<_IDX_##_t, _VAL_##_t>* m) {                                      \
        return m->min();                                                      \
    }                                                                         \
    IFTYPE<_VAL_##_t>::out_t dict_##_IDX_##_##_VAL_##_max(                    \
        dict<_IDX_##_t, _VAL_##_t>* m) {                                      \
        return m->max();                                                      \
    }                                                                         \
    bool dict_##_IDX_##_##_VAL_##_not_empty(dict<_IDX_##_t, _VAL_##_t>* m) {  \
        return m->not_empty();                                                \
    }                                                                         \
    void* dict_##_IDX_##_##_VAL_##_keys(dict<_IDX_##_t, _VAL_##_t>* m) {      \
        return m->keys();                                                     \
    }


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

// all the types that we support for keys and values
#define TYPES                                                               \
    BOOST_PP_TUPLE_TO_LIST(                                                 \
        12, (int, int8, int16, int32, int64, uint8, uint16, uint32, uint64, \
             bool, float32, float64, unicode_type))


// Now use some macro-magic from boost to support dicts for above types
#define APPLY_DEF_DICT(r, product) DEF_DICT product
BOOST_PP_LIST_FOR_EACH_PRODUCT(APPLY_DEF_DICT, 2, (TYPES, TYPES))

// declaration of dict functions in python module
#define DEC_MOD_METHOD(func)                            \
    PyObject_SetAttrString(m, BOOST_PP_STRINGIZE(func), \
                           PyLong_FromVoidPtr((void*)(&func)))
#define DEC_DICT_MOD(_IDX_, _VAL_)                    \
    DEC_MOD_METHOD(dict_##_IDX_##_##_VAL_##_init);    \
    DEC_MOD_METHOD(dict_##_IDX_##_##_VAL_##_setitem); \
    DEC_MOD_METHOD(dict_##_IDX_##_##_VAL_##_getitem); \
    DEC_MOD_METHOD(dict_##_IDX_##_##_VAL_##_in);      \
    DEC_MOD_METHOD(dict_##_IDX_##_##_VAL_##_print);   \
    DEC_MOD_METHOD(dict_##_IDX_##_##_VAL_##_get);     \
    DEC_MOD_METHOD(dict_##_IDX_##_##_VAL_##_pop);     \
    DEC_MOD_METHOD(dict_##_IDX_##_##_VAL_##_keys);    \
    DEC_MOD_METHOD(dict_##_IDX_##_##_VAL_##_min);     \
    DEC_MOD_METHOD(dict_##_IDX_##_##_VAL_##_max);     \
    DEC_MOD_METHOD(dict_##_IDX_##_##_VAL_##_not_empty);

// module initiliziation
// make our C-functions available
PyMODINIT_FUNC PyInit_hdict_ext(void) {
    PyObject* m;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT, "hdict_ext", "No docs", -1, NULL,
    };
    m = PyModule_Create(&moduledef);
    if (m == NULL) return NULL;

// And all the other speicialized dicts
#define APPLY_DEC_DICT_MOD(r, product) DEC_DICT_MOD product
    BOOST_PP_LIST_FOR_EACH_PRODUCT(APPLY_DEC_DICT_MOD, 2, (TYPES, TYPES));

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
