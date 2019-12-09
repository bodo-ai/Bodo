/*
 * C++ implementation of timsort
 *
 * ported from Python's and OpenJDK's:
 * - http://svn.python.org/projects/python/trunk/Objects/listobject.c
 * - http://cr.openjdk.java.net/~martin/webrevs/openjdk7/timsort/raw_files/new/src/share/classes/java/util/TimSort.java
 *
 * Copyright (c) 2011 Fuji, Goro (gfx) <gfuji@cpan.org>.
 * Copyright (c) 2019 Morwenn.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef GFX_TIMSORT_HPP
#define GFX_TIMSORT_HPP

#include <algorithm>
#include <functional>
#include <iterator>
#include <vector>

// Semantic versioning macros

#define GFX_TIMSORT_VERSION_MAJOR 1
#define GFX_TIMSORT_VERSION_MINOR 1
#define GFX_TIMSORT_VERSION_PATCH 0

// Diagnostic selection macros

#ifdef GFX_TIMSORT_ENABLE_ASSERT
#   include <cassert>
#   define GFX_TIMSORT_ASSERT(expr) assert(expr)
#else
#   define GFX_TIMSORT_ASSERT(expr) ((void)0)
#endif

#ifdef GFX_TIMSORT_ENABLE_LOG
#   include <iostream>
#   define GFX_TIMSORT_LOG(expr) (std::clog << "# " << __func__ << ": " << expr << std::endl)
#else
#   define GFX_TIMSORT_LOG(expr) ((void)0)
#endif

// If GFX_TIMSORT_USE_STD_MOVE is not defined, try to define it as follows:
// - Check standard feature-testing macro
// - Check non-standard feature-testing macro
// - Check C++ standard (disable if < C++11)
// - Check compiler-specific versions known to support move semantics

#ifndef GFX_TIMSORT_USE_STD_MOVE
#   if defined(__cpp_rvalue_references)
#       define GFX_TIMSORT_USE_STD_MOVE 1
#   elif defined(__has_feature)
#       if __has_feature(cxx_rvalue_references)
#           define GFX_TIMSORT_USE_STD_MOVE 1
#       else
#           define GFX_TIMSORT_USE_STD_MOVE 0
#       endif
#   elif !(defined(__cplusplus) && __cplusplus >= 201103L)
#       define GFX_TIMSORT_USE_STD_MOVE 0
#   elif defined(_MSC_VER) && _MSC_VER >= 1700
#       define GFX_TIMSORT_USE_STD_MOVE 1
#   elif defined(__GNUC__) && (__GNUC__ >= 5 || (__GNUC__ >= 4 && __GNUC_MINOR__ >= 6))
#       define GFX_TIMSORT_USE_STD_MOVE 1
#   else
#       define GFX_TIMSORT_USE_STD_MOVE 0
#   endif
#endif

#if GFX_TIMSORT_USE_STD_MOVE
	#include <utility>
	#define GFX_TIMSORT_MOVE(x) std::move(x)
	#define GFX_TIMSORT_MOVE_RANGE(in1, in2, out) std::move((in1), (in2), (out));
	#define GFX_TIMSORT_MOVE_BACKWARD(in1, in2, out) std::move_backward((in1), (in2), (out));
#else
	#define GFX_TIMSORT_MOVE(x) (x)
	#define GFX_TIMSORT_MOVE_RANGE(in1, in2, out) std::copy((in1), (in2), (out));
	#define GFX_TIMSORT_MOVE_BACKWARD(in1, in2, out) std::copy_backward((in1), (in2), (out));
#endif


namespace gfx {

// ---------------------------------------
// Implementation details
// ---------------------------------------

namespace detail {

// Equivalent to C++20 std::identity
struct identity {
#if GFX_TIMSORT_USE_STD_MOVE
    template <typename T>
    T&& operator()(T&& value) const
    {
        return std::forward<T>(value);
    }
#else
    template <typename T>
    T& operator()(T& value) const {
        return value;
    }

    template <typename T>
    T const& operator()(T const& value) const {
        return value;
    }
#endif
};

// Merge a predicate and a projection function
template <typename Compare, typename Projection>
struct projection_compare {
    projection_compare(Compare comp, Projection proj) : compare(comp), projection(proj) {
    }

#if GFX_TIMSORT_USE_STD_MOVE
    template <typename T, typename U>
    bool operator()(T &&lhs, U &&rhs) {
        return static_cast<bool>(compare(
            projection(std::forward<T>(lhs)),
            projection(std::forward<U>(rhs))
        ));
    }
#else
    template <typename T, typename U>
    bool operator()(T &lhs, U &rhs) {
        return static_cast<bool>(compare(projection(lhs), projection(rhs)));
    }
#endif

    Compare compare;
    Projection projection;
};

template <typename Iterator> struct run {
    typedef typename std::iterator_traits<Iterator>::difference_type diff_t;

    Iterator base;
    diff_t len;

    run(Iterator b, diff_t l) : base(b), len(l) {
    }
};

template <typename RandomAccessIterator, typename Compare> class TimSort {
    typedef RandomAccessIterator iter_t;
    typedef typename std::iterator_traits<iter_t>::value_type value_t;
    typedef typename std::iterator_traits<iter_t>::reference ref_t;
    typedef typename std::iterator_traits<iter_t>::difference_type diff_t;

    static const int MIN_MERGE = 32;
    static const int MIN_GALLOP = 7;

    int minGallop_; // default to MIN_GALLOP

    std::vector<value_t> tmp_; // temp storage for merges
    typedef typename std::vector<value_t>::iterator tmp_iter_t;

    std::vector<run<RandomAccessIterator> > pending_;

    static void binarySort(iter_t const lo, iter_t const hi, iter_t start, Compare compare) {
        GFX_TIMSORT_ASSERT(lo <= start);
        GFX_TIMSORT_ASSERT(start <= hi);
        if (start == lo) {
            ++start;
        }
        for (; start < hi; ++start) {
            GFX_TIMSORT_ASSERT(lo <= start);
            value_t pivot = GFX_TIMSORT_MOVE(*start);

            iter_t const pos = std::upper_bound(lo, start, pivot, compare);
            for (iter_t p = start; p > pos; --p) {
                *p = GFX_TIMSORT_MOVE(*(p - 1));
            }
            *pos = GFX_TIMSORT_MOVE(pivot);
        }
    }

    static diff_t countRunAndMakeAscending(iter_t const lo, iter_t const hi, Compare compare) {
        GFX_TIMSORT_ASSERT(lo < hi);

        iter_t runHi = lo + 1;
        if (runHi == hi) {
            return 1;
        }

        if (compare(*(runHi++), *lo)) { // decreasing
            while (runHi < hi && compare(*runHi, *(runHi - 1))) {
                ++runHi;
            }
            std::reverse(lo, runHi);
        } else { // non-decreasing
            while (runHi < hi && !compare(*runHi, *(runHi - 1))) {
                ++runHi;
            }
        }

        return runHi - lo;
    }

    static diff_t minRunLength(diff_t n) {
        GFX_TIMSORT_ASSERT(n >= 0);

        diff_t r = 0;
        while (n >= MIN_MERGE) {
            r |= (n & 1);
            n >>= 1;
        }
        return n + r;
    }

    TimSort() : minGallop_(MIN_GALLOP) {
    }

    // Silence GCC -Winline warning
    ~TimSort() {}

    void pushRun(iter_t const runBase, diff_t const runLen) {
        pending_.push_back(run<iter_t>(runBase, runLen));
    }

    void mergeCollapse(Compare compare) {
        while (pending_.size() > 1) {
            diff_t n = pending_.size() - 2;

            if ((n > 0 && pending_[n - 1].len <= pending_[n].len + pending_[n + 1].len) ||
                (n > 1 && pending_[n - 2].len <= pending_[n - 1].len + pending_[n].len)) {
                if (pending_[n - 1].len < pending_[n + 1].len) {
                    --n;
                }
                mergeAt(n, compare);
            } else if (pending_[n].len <= pending_[n + 1].len) {
                mergeAt(n, compare);
            } else {
                break;
            }
        }
    }

    void mergeForceCollapse(Compare compare) {
        while (pending_.size() > 1) {
            diff_t n = pending_.size() - 2;

            if (n > 0 && pending_[n - 1].len < pending_[n + 1].len) {
                --n;
            }
            mergeAt(n, compare);
        }
    }

    void mergeAt(diff_t const i, Compare compare) {
        diff_t const stackSize = pending_.size();
        GFX_TIMSORT_ASSERT(stackSize >= 2);
        GFX_TIMSORT_ASSERT(i >= 0);
        GFX_TIMSORT_ASSERT(i == stackSize - 2 || i == stackSize - 3);

        iter_t base1 = pending_[i].base;
        diff_t len1 = pending_[i].len;
        iter_t base2 = pending_[i + 1].base;
        diff_t len2 = pending_[i + 1].len;

        GFX_TIMSORT_ASSERT(len1 > 0);
        GFX_TIMSORT_ASSERT(len2 > 0);
        GFX_TIMSORT_ASSERT(base1 + len1 == base2);

        pending_[i].len = len1 + len2;

        if (i == stackSize - 3) {
            pending_[i + 1] = pending_[i + 2];
        }

        pending_.pop_back();

        diff_t const k = gallopRight(*base2, base1, len1, 0, compare);
        GFX_TIMSORT_ASSERT(k >= 0);

        base1 += k;
        len1 -= k;

        if (len1 == 0) {
            return;
        }

        len2 = gallopLeft(*(base1 + (len1 - 1)), base2, len2, len2 - 1, compare);
        GFX_TIMSORT_ASSERT(len2 >= 0);
        if (len2 == 0) {
            return;
        }

        if (len1 <= len2) {
            mergeLo(base1, len1, base2, len2, compare);
        } else {
            mergeHi(base1, len1, base2, len2, compare);
        }
    }

    template <typename Iter>
    diff_t gallopLeft(ref_t key, Iter const base, diff_t const len, diff_t const hint, Compare compare) {
        GFX_TIMSORT_ASSERT(len > 0);
        GFX_TIMSORT_ASSERT(hint >= 0);
        GFX_TIMSORT_ASSERT(hint < len);

        diff_t lastOfs = 0;
        diff_t ofs = 1;

        if (compare(*(base + hint), key)) {
            diff_t const maxOfs = len - hint;
            while (ofs < maxOfs && compare(*(base + (hint + ofs)), key)) {
                lastOfs = ofs;
                ofs = (ofs << 1) + 1;

                if (ofs <= 0) { // int overflow
                    ofs = maxOfs;
                }
            }
            if (ofs > maxOfs) {
                ofs = maxOfs;
            }

            lastOfs += hint;
            ofs += hint;
        } else {
            diff_t const maxOfs = hint + 1;
            while (ofs < maxOfs && !compare(*(base + (hint - ofs)), key)) {
                lastOfs = ofs;
                ofs = (ofs << 1) + 1;

                if (ofs <= 0) {
                    ofs = maxOfs;
                }
            }
            if (ofs > maxOfs) {
                ofs = maxOfs;
            }

            diff_t const tmp = lastOfs;
            lastOfs = hint - ofs;
            ofs = hint - tmp;
        }
        GFX_TIMSORT_ASSERT(-1 <= lastOfs);
        GFX_TIMSORT_ASSERT(lastOfs < ofs);
        GFX_TIMSORT_ASSERT(ofs <= len);

        return std::lower_bound(base + (lastOfs + 1), base + ofs, key, compare) - base;
    }

    template <typename Iter>
    diff_t gallopRight(ref_t key, Iter const base, diff_t const len, diff_t const hint, Compare compare) {
        GFX_TIMSORT_ASSERT(len > 0);
        GFX_TIMSORT_ASSERT(hint >= 0);
        GFX_TIMSORT_ASSERT(hint < len);

        diff_t ofs = 1;
        diff_t lastOfs = 0;

        if (compare(key, *(base + hint))) {
            diff_t const maxOfs = hint + 1;
            while (ofs < maxOfs && compare(key, *(base + (hint - ofs)))) {
                lastOfs = ofs;
                ofs = (ofs << 1) + 1;

                if (ofs <= 0) {
                    ofs = maxOfs;
                }
            }
            if (ofs > maxOfs) {
                ofs = maxOfs;
            }

            diff_t const tmp = lastOfs;
            lastOfs = hint - ofs;
            ofs = hint - tmp;
        } else {
            diff_t const maxOfs = len - hint;
            while (ofs < maxOfs && !compare(key, *(base + (hint + ofs)))) {
                lastOfs = ofs;
                ofs = (ofs << 1) + 1;

                if (ofs <= 0) { // int overflow
                    ofs = maxOfs;
                }
            }
            if (ofs > maxOfs) {
                ofs = maxOfs;
            }

            lastOfs += hint;
            ofs += hint;
        }
        GFX_TIMSORT_ASSERT(-1 <= lastOfs);
        GFX_TIMSORT_ASSERT(lastOfs < ofs);
        GFX_TIMSORT_ASSERT(ofs <= len);

        return std::upper_bound(base + (lastOfs + 1), base + ofs, key, compare) - base;
    }

    static void rotateLeft(iter_t first, iter_t last)
    {
        value_t tmp = GFX_TIMSORT_MOVE(*first);
        iter_t last_1 = GFX_TIMSORT_MOVE_RANGE(first + 1, last, first);
        *last_1 = GFX_TIMSORT_MOVE(tmp);
    }

    static void rotateRight(iter_t first, iter_t last)
    {
        iter_t last_1 = last - 1;
        value_t tmp = GFX_TIMSORT_MOVE(*last_1);
        GFX_TIMSORT_MOVE_BACKWARD(first, last_1, last);
        *first = GFX_TIMSORT_MOVE(tmp);
    }


    void mergeLo(iter_t const base1, diff_t len1, iter_t const base2, diff_t len2, Compare compare) {
        GFX_TIMSORT_ASSERT(len1 > 0);
        GFX_TIMSORT_ASSERT(len2 > 0);
        GFX_TIMSORT_ASSERT(base1 + len1 == base2);

        if (len1 == 1) {
            return rotateLeft(base1, base2 + len2);
        }
        if (len2 == 1) {
            return rotateRight(base1, base2 + len2);
        }

        copy_to_tmp(base1, len1);

        tmp_iter_t cursor1 = tmp_.begin();
        iter_t cursor2 = base2;
        iter_t dest = base1;

        *(dest++) = GFX_TIMSORT_MOVE(*(cursor2++));
        --len2;

        int minGallop(minGallop_);

        // outer:
        while (true) {
            diff_t count1 = 0;
            diff_t count2 = 0;

            do {
                GFX_TIMSORT_ASSERT(len1 > 1);
                GFX_TIMSORT_ASSERT(len2 > 0);

                if (compare(*cursor2, *cursor1)) {
                    *(dest++) = GFX_TIMSORT_MOVE(*(cursor2++));
                    ++count2;
                    count1 = 0;
                    if (--len2 == 0) {
                        goto epilogue;
                    }
                } else {
                    *(dest++) = GFX_TIMSORT_MOVE(*(cursor1++));
                    ++count1;
                    count2 = 0;
                    if (--len1 == 1) {
                        goto epilogue;
                    }
                }
            } while ((count1 | count2) < minGallop);

            do {
                GFX_TIMSORT_ASSERT(len1 > 1);
                GFX_TIMSORT_ASSERT(len2 > 0);

                count1 = gallopRight(*cursor2, cursor1, len1, 0, compare);
                if (count1 != 0) {
                    GFX_TIMSORT_MOVE_BACKWARD(cursor1, cursor1 + count1, dest + count1);
                    dest += count1;
                    cursor1 += count1;
                    len1 -= count1;

                    if (len1 <= 1) {
                        goto epilogue;
                    }
                }
                *(dest++) = GFX_TIMSORT_MOVE(*(cursor2++));
                if (--len2 == 0) {
                    goto epilogue;
                }

                count2 = gallopLeft(*cursor1, cursor2, len2, 0, compare);
                if (count2 != 0) {
                    GFX_TIMSORT_MOVE_RANGE(cursor2, cursor2 + count2, dest);
                    dest += count2;
                    cursor2 += count2;
                    len2 -= count2;
                    if (len2 == 0) {
                        goto epilogue;
                    }
                }
                *(dest++) = GFX_TIMSORT_MOVE(*(cursor1++));
                if (--len1 == 1) {
                    goto epilogue;
                }

                --minGallop;
            } while ((count1 >= MIN_GALLOP) | (count2 >= MIN_GALLOP));

            if (minGallop < 0) {
                minGallop = 0;
            }
            minGallop += 2;
        } // end of "outer" loop

        epilogue: // merge what is left from either cursor1 or cursor2

        minGallop_ = (std::min)(minGallop, 1);

        if (len1 == 1) {
            GFX_TIMSORT_ASSERT(len2 > 0);
            GFX_TIMSORT_MOVE_RANGE(cursor2, cursor2 + len2, dest);
            *(dest + len2) = GFX_TIMSORT_MOVE(*cursor1);
        } else {
            GFX_TIMSORT_ASSERT(len1 != 0 && "Comparison function violates its general contract");
            GFX_TIMSORT_ASSERT(len2 == 0);
            GFX_TIMSORT_ASSERT(len1 > 1);
            GFX_TIMSORT_MOVE_RANGE(cursor1, cursor1 + len1, dest);
        }
    }

    void mergeHi(iter_t const base1, diff_t len1, iter_t const base2, diff_t len2, Compare compare) {
        GFX_TIMSORT_ASSERT(len1 > 0);
        GFX_TIMSORT_ASSERT(len2 > 0);
        GFX_TIMSORT_ASSERT(base1 + len1 == base2);

        if (len1 == 1) {
            return rotateLeft(base1, base2 + len2);
        }
        if (len2 == 1) {
            return rotateRight(base1, base2 + len2);
        }

        copy_to_tmp(base2, len2);

        iter_t cursor1 = base1 + len1;
        tmp_iter_t cursor2 = tmp_.begin() + (len2 - 1);
        iter_t dest = base2 + (len2 - 1);

        *(dest--) = GFX_TIMSORT_MOVE(*(--cursor1));
        --len1;

        int minGallop(minGallop_);

        // outer:
        while (true) {
            diff_t count1 = 0;
            diff_t count2 = 0;

            // The next loop is a hot path of the algorithm, so we decrement
            // eagerly the cursor so that it always points directly to the value
            // to compare, but we have to implement some trickier logic to make
            // sure that it points to the next value again by the end of said loop
            --cursor1;

            do {
                GFX_TIMSORT_ASSERT(len1 > 0);
                GFX_TIMSORT_ASSERT(len2 > 1);

                if (compare(*cursor2, *cursor1)) {
                    *(dest--) = GFX_TIMSORT_MOVE(*cursor1);
                    ++count1;
                    count2 = 0;
                    if (--len1 == 0) {
                        goto epilogue;
                    }
                    --cursor1;
                } else {
                    *(dest--) = GFX_TIMSORT_MOVE(*(cursor2--));
                    ++count2;
                    count1 = 0;
                    if (--len2 == 1) {
                        ++cursor1; // See comment before the loop
                        goto epilogue;
                    }
                }
            } while ((count1 | count2) < minGallop);
            ++cursor1; // See comment before the loop

            do {
                GFX_TIMSORT_ASSERT(len1 > 0);
                GFX_TIMSORT_ASSERT(len2 > 1);

                count1 = len1 - gallopRight(*cursor2, base1, len1, len1 - 1, compare);
                if (count1 != 0) {
                    dest -= count1;
                    cursor1 -= count1;
                    len1 -= count1;
                    GFX_TIMSORT_MOVE_BACKWARD(cursor1, cursor1 + count1, dest + (1 + count1));

                    if (len1 == 0) {
                        goto epilogue;
                    }
                }
                *(dest--) = GFX_TIMSORT_MOVE(*(cursor2--));
                if (--len2 == 1) {
                    goto epilogue;
                }

                count2 = len2 - gallopLeft(*(cursor1 - 1), tmp_.begin(), len2, len2 - 1, compare);
                if (count2 != 0) {
                    dest -= count2;
                    cursor2 -= count2;
                    len2 -= count2;
                    GFX_TIMSORT_MOVE_RANGE(cursor2 + 1, cursor2 + (1 + count2), dest + 1);
                    if (len2 <= 1) {
                        goto epilogue;
                    }
                }
                *(dest--) = GFX_TIMSORT_MOVE(*(--cursor1));
                if (--len1 == 0) {
                    goto epilogue;
                }

                minGallop--;
            } while ((count1 >= MIN_GALLOP) | (count2 >= MIN_GALLOP));

            if (minGallop < 0) {
                minGallop = 0;
            }
            minGallop += 2;
        } // end of "outer" loop

        epilogue: // merge what is left from either cursor1 or cursor2

        minGallop_ = (std::min)(minGallop, 1);

        if (len2 == 1) {
            GFX_TIMSORT_ASSERT(len1 > 0);
            dest -= len1;
            GFX_TIMSORT_MOVE_BACKWARD(cursor1 - len1, cursor1, dest + (1 + len1));
            *dest = GFX_TIMSORT_MOVE(*cursor2);
        } else {
            GFX_TIMSORT_ASSERT(len2 != 0 && "Comparison function violates its general contract");
            GFX_TIMSORT_ASSERT(len1 == 0);
            GFX_TIMSORT_ASSERT(len2 > 1);
            GFX_TIMSORT_MOVE_RANGE(tmp_.begin(), tmp_.begin() + len2, dest - (len2 - 1));
        }
    }

    void copy_to_tmp(iter_t const begin, diff_t len) {
#if GFX_TIMSORT_USE_STD_MOVE
        tmp_.assign(std::make_move_iterator(begin),
                    std::make_move_iterator(begin + len));
#else
        tmp_.assign(begin, begin + len);
#endif
    }

public:

    static void sort(iter_t const lo, iter_t const hi, Compare compare) {
        GFX_TIMSORT_ASSERT(lo <= hi);

        diff_t nRemaining = (hi - lo);
        if (nRemaining < 2) {
            return; // nothing to do
        }

        if (nRemaining < MIN_MERGE) {
            diff_t const initRunLen = countRunAndMakeAscending(lo, hi, compare);
            GFX_TIMSORT_LOG("initRunLen: " << initRunLen);
            binarySort(lo, hi, lo + initRunLen, compare);
            return;
        }

        TimSort ts;
        diff_t const minRun = minRunLength(nRemaining);
        iter_t cur = lo;
        do {
            diff_t runLen = countRunAndMakeAscending(cur, hi, compare);

            if (runLen < minRun) {
                diff_t const force = (std::min)(nRemaining, minRun);
                binarySort(cur, cur + force, cur + runLen, compare);
                runLen = force;
            }

            ts.pushRun(cur, runLen);
            ts.mergeCollapse(compare);

            cur += runLen;
            nRemaining -= runLen;
        } while (nRemaining != 0);

        GFX_TIMSORT_ASSERT(cur == hi);
        ts.mergeForceCollapse(compare);
        GFX_TIMSORT_ASSERT(ts.pending_.size() == 1);

        GFX_TIMSORT_LOG("size: " << (hi - lo) << " tmp_.size(): " << ts.tmp_.size()
                                 << " pending_.size(): " << ts.pending_.size());
    }
};

} // namespace detail

// ---------------------------------------
// Public interface implementation
// ---------------------------------------

/**
 * Stably sorts a range with a comparison function and a projection function.
 */
template <typename RandomAccessIterator, typename Compare, typename Projection>
void timsort(RandomAccessIterator const first, RandomAccessIterator const last,
             Compare compare, Projection projection) {
    typedef detail::projection_compare<Compare, Projection> compare_t;
    compare_t comp(compare, projection);
    detail::TimSort<RandomAccessIterator, compare_t>::sort(first, last, comp);
}

/**
 * Same as std::stable_sort(first, last, compare).
 */
template <typename RandomAccessIterator, typename Compare>
void timsort(RandomAccessIterator const first, RandomAccessIterator const last, Compare compare) {
    gfx::timsort(first, last, compare, detail::identity());
}

/**
 * Same as std::stable_sort(first, last).
 */
template <typename RandomAccessIterator>
void timsort(RandomAccessIterator const first, RandomAccessIterator const last) {
    typedef typename std::iterator_traits<RandomAccessIterator>::value_type value_type;
    gfx::timsort(first, last, std::less<value_type>(), detail::identity());
}

} // namespace gfx

#undef GFX_TIMSORT_ENABLE_ASSERT
#undef GFX_TIMSORT_ASSERT
#undef GFX_TIMSORT_ENABLE_LOG
#undef GFX_TIMSORT_LOG
#undef GFX_TIMSORT_MOVE
#undef GFX_TIMSORT_MOVE_RANGE
#undef GFX_TIMSORT_MOVE_BACKWARD
#undef GFX_TIMSORT_USE_STD_MOVE

#endif // GFX_TIMSORT_HPP
