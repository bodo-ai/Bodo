/// bodo::pinnable tests
#include <set>
#include <sstream>

#include "../libs/_bodo_common.h"
#include "../libs/_bodo_to_arrow.h"
#include "../libs/_memory.h"
#include "../libs/_pinnable.h"
#include "./test.hpp"

#if defined(_WIN32)
#define SKIP_IF_WINDOWS return
#else
#define SKIP_IF_WINDOWS
#endif

// Needed for equality_check's failure case below
namespace std {
template <typename X, typename Y>
std::ostream &operator<<(std::ostream &out, const std::pair<X, Y> &p) {
    out << "(" << p.first << ", " << p.second << ")";
    return out;
}
}  // namespace std

template <typename Iterator1, typename Iterator2>
void equality_check(const Iterator1 &begin1, const Iterator1 &end1,
                    const Iterator2 &begin2, const char *message) {
    int ix = 0, diffs = 0;
    bool differs = false, elided = false;

    auto i(begin1);
    auto j(begin2);

    for (; i != end1; ++i, ++j, ++ix) {
        if (*i != *j) {
            if (!differs) {
                std::cerr << "Array equality check fails: " << message
                          << std::endl;
            }
            differs = true;
            if (diffs < 50) {
                std::cerr << "  Mismatch at index " << ix << " expected: " << *i
                          << ", got: " << *j << std::endl;
            } else {
                elided = true;
            }
            diffs++;
        }
    }

    if (elided) {
        std::cerr << "in total, " << diffs
                  << " positions were different, but some were elided"
                  << std::endl;
    }

    if (differs) {
        std::cerr << "  Expected: ";
        std::copy(begin1, end1,
                  std::ostream_iterator<
                      typename std::iterator_traits<Iterator1>::value_type>(
                      std::cerr, ","));
        std::cerr << std::endl;

        std::cerr << "  Got:      ";
        std::copy(begin2, begin2 + std::distance(begin1, end1),
                  std::ostream_iterator<
                      typename std::iterator_traits<Iterator2>::value_type>(
                      std::cerr, ","));
        std::cerr << std::endl;

        throw std::runtime_error(message);
    }
}

// This is a test harness that takes a templatized functor class and
// a pinnable type. The functor is constructed with the given
// arguments.
//
// The functor objects are invoked with one argument of the template
// parameter type. The objects are of the same kind as T. So if T is
// bodo::vector, then the function objects receive a bodo::vector-like
// object.
//
// The functor will be invoked three times. First, it is invoked
// directly on type T.
//
// Then it is invoked on T where the allocator has been changed to the
// pinnable allocator.
//
// Finally, it is invoked on a pinned version of a pinnable<T>.
//
// The test simulates a spill and reload by physically moving the
// memory of the pinnable object.
//
// Lastly, all three versions of these objects are compared for
// equality and any mismatches are reported.
//
// This tests for multiple things:
//
//   1. Both the pinnable allocator and normal bodo allocator behave
//   similarly
//
//   2. The pinnable allocator can handle memory being spilled and
//   reloaded (potentially to a different spot in memory)
//
//   3. All three versions agree on the order and presence of elements.
//
// For convenience, the random number generator is seeded arbitrarily
// but identically for each of the three invocations of Func, so that
// randomized testing is deterministic.
template <typename T, template <typename> typename Func, typename... Args>
void simple_pin(const std::string &nm, Args &&...args) {
    Func<T> normal(std::forward<Args>(args)...);
    Func<typename bodo::pinnable<T>::element_type> spilling(
        std::forward<Args>(args)...);
    bodo::tests::test("simple_pin:" + nm, [normal, spilling] {
        bodo::pinnable<T> pinnable;
        T ref;

        auto seed(rand());

        srand(seed);  // Need to keep the random seed the same for all three
                      // tests

        normal(ref);
        {
            auto pinned(bodo::pin(pinnable));

            typename bodo::pinnable<T>::element_type nonspilling;
            srand(seed);
            spilling(nonspilling);

            srand(seed);
            spilling(*pinned);

            equality_check(std::begin(nonspilling), std::end(nonspilling),
                           std::begin(*pinned),
                           "Bodo array and pining array differ before spill");
            equality_check(std::begin(ref), std::end(ref), std::begin(*pinned),
                           "Arrays differ before spill");

            T ref2;
            normal(ref2);
        }

        // Emulate a spill
        pinnable.spillAndMove();

        auto pinned(bodo::pin(pinnable));

        equality_check(std::begin(ref), std::end(ref), std::begin(*pinned),
                       "Arrays differ after spill");
        bodo::tests::check(
            std::equal(std::begin(ref), std::end(ref), std::begin(*pinned)),
            "Arrays differ after spill");
    });
}

// Simple test case that pushes all elements between begin and end, in order
template <typename Vector>
struct push_back_test {
    typename Vector::value_type begin_;
    typename Vector::value_type end_;

    push_back_test(typename Vector::value_type begin,
                   typename Vector::value_type end)
        : begin_(begin), end_(end) {}

    void operator()(Vector &v) const {
        for (auto i(begin_); i != end_; ++i) {
            v.push_back(i);
        }
    }
};

// Push count elements in random insertion order.
template <typename Vector>
struct insert_test {
    int count_;

    insert_test(int count) : count_(count) {}

    void operator()(Vector &val) const {
        val.push_back(0);
        for (int i = 0; i < count_; ++i) {
            size_t ix = (rand() % val.size());
            auto newval(rand());
            val.insert(std::begin(val) + ix, newval);
        }
    }
};

// Insert (and sometimes erase) randomly throughout a map
template <typename Map>
struct insert_map_test {
    int count_;
    bool erase_;

    insert_map_test(int count, bool erase = false)
        : count_(count), erase_(erase) {}

    void operator()(Map &map) const {
        std::set<typename Map::key_type> keys;
        for (auto i(0); i < count_; ++i) {
            bool tryErase = (erase_ ? rand() % 4 : 0) == 1;
            typename Map::key_type key = rand();
            typename Map::mapped_type value = rand();

            if (tryErase && keys.size() > 0) {
                // choose random key to erase
                auto seti(keys.begin());
                std::advance(seti, key % keys.size());
                auto i(map.find(*seti));
                if (i == map.end()) {
                    throw std::runtime_error(
                        "Map could not find key we inserted");
                }
                keys.erase(seti);
                map.erase(i);
            } else {
                keys.insert(key);
                map[key] = value;
            }
        }
    }
};

// Wrapper for push_back_test. Pushes all elements between begin and end in
// order
template <typename T>
void construct_and_move(typename T::value_type begin,
                        typename T::value_type end) {
    std::stringstream nm;
    nm << "construct_and_move<" << typeid(T).name() << ">(" << begin << ", "
       << end << ")";
    simple_pin<T, push_back_test>(nm.str(), begin, end);
}

// Does count random insertions to the vector and compares results
template <typename T>
void insert_and_move(int count) {
    std::stringstream nm;
    nm << "insert_and_move<" << typeid(T).name() << ">(" << count << ")";
    simple_pin<T, insert_test>(nm.str(), count);
}

// Random inserts in an unordered map
template <typename T>
void insert_and_move_map(int count) {
    std::stringstream nm;
    nm << "insert_and_move_map<" << typeid(T).name() << ">(" << count << ")";
    simple_pin<T, insert_map_test>(nm.str(), count);
}

// Random inserts and erasures in an unordered map
template <typename T>
void insert_erase_and_move_map(int count) {
    std::stringstream nm;
    nm << "insert_erase_and_move_map<" << typeid(T).name() << ">(" << count
       << ")";
    simple_pin<T, insert_map_test>(nm.str(), count, true);
}

static bodo::tests::suite tests([] {
    // TODO: [BSE-4151] Test segfaulting on PR CI
    bodo::tests::test("pinnable_vector_uint32_t", [] {
        // skip on Windows since buffer pool is disabled.
        SKIP_IF_WINDOWS;

        auto pool = bodo::BufferPool();
        auto allocator = bodo::PinnableAllocator<uint32_t>(&pool);

        bodo::pinnable<bodo::vector<std::uint32_t>>::element_type initial(
            allocator);
        for (std::uint32_t i = 0; i < 40000; ++i) {
            initial.push_back(i);
        };

        bodo::tests::check(
            pool.bytes_allocated() ==
            (int64_t)(pool.bytes_pinned()));  // Everything is pinned

        bodo::pinnable<bodo::vector<std::uint32_t>>::element_type expected(
            initial);

        bodo::pinnable<bodo::vector<std::uint32_t>> pinnable_ints(
            std::move(initial));

        bodo::tests::check(pool.bytes_allocated() >
                           (int64_t)(pool.bytes_pinned()));
        bodo::tests::check(*bodo::pin(pinnable_ints) == expected);
    });

    bodo::tests::test("simple_insert", [] {
        typename bodo::pinnable<bodo::vector<uint32_t>>::element_type
            nonspilling;
        // std::vector<uint32_t> non-spilling;
        for (int i = 0; i < 33; ++i) {
            nonspilling.push_back(0);
        }
        nonspilling.push_back(0);
    });

    // construct_and_move<bodo::vector<uint32_t>>(0, 100);
    // construct_and_move<bodo::vector<uint32_t>>(0, 100000);

    // insert_and_move<bodo::vector<uint32_t>>(100);
    // insert_and_move<bodo::vector<uint32_t>>(100000);

    // insert_and_move_map<bodo::unord_map_container<uint32_t, uint32_t>>(100);
    // insert_and_move_map<bodo::unord_map_container<uint32_t, uint32_t>>(1000);

    // insert_and_move_map<bodo::unord_map_container<uint32_t, uint32_t>>(100);
    // insert_and_move_map<bodo::unord_map_container<uint32_t, uint32_t>>(1000);
    // insert_and_move_map<bodo::unord_map_container<uint32_t,
    // uint32_t>>(10000);

    // insert_erase_and_move_map<bodo::unord_map_container<uint32_t, uint32_t>>(
    //     100);
    // insert_erase_and_move_map<bodo::unord_map_container<uint32_t, uint32_t>>(
    //     1000);
    // insert_erase_and_move_map<bodo::unord_map_container<uint32_t, uint32_t>>(
    //     10000);

    // construct_and_move_map<bodo::unord_map_container<uint32_t, uint32_t>>(0,
    // 1000000); construct_and_move_map<bodo::unord_map_container<uint32_t,
    // uint32_t>>(0, 10000000);

    bodo::tests::test("test_to_arrow_roundtrip_pinnable", [] {
        // skip on Windows since buffer pool is disabled.
        SKIP_IF_WINDOWS;

        auto do_roundtrip_test = [](std::shared_ptr<array_info> arr) {
            auto arrow_arr = to_arrow(arr);
            auto out_arr =
                arrow_array_to_bodo(arrow_arr, bodo::BufferPool::DefaultPtr());
            // Check that we don't crash when we attempt to pin/unpin
            out_arr->unpin();
            out_arr->pin();
        };

        const size_t n_elem = 100;
        std::shared_ptr<array_info> int_arr =
            alloc_numpy(n_elem, Bodo_CTypes::CTypeEnum::INT32);
        auto *data = reinterpret_cast<int32_t *>(
            int_arr->data1<bodo_array_type::NUMPY>());
        for (size_t i = 0; i < n_elem; i++) {
            data[i] = i;
        }

        std::shared_ptr<array_info> array_item_arr =
            alloc_array_item(10, int_arr);
        auto *offsets = reinterpret_cast<uint64_t *>(
            array_item_arr->data1<bodo_array_type::ARRAY_ITEM>());
        offsets[0] = 0;
        for (size_t idx = 1; idx <= 10; idx++) {
            offsets[idx] = 10 * idx;
        }

        do_roundtrip_test(int_arr);
        do_roundtrip_test(array_item_arr);
    });
});
