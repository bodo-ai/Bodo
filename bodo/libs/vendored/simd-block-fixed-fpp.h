// Source:
// https://github.com/FastFilter/fastfilter_cpp/commit/4264fbb307b2d6a5a3ada255d6f53978beb90dc9
// Bodo changes:
// - Added bloom_size_bytes() function to get the size of the bloom filter
//   given the number of unique elements I want to insert (before constructing).
// - Added num_elements_for_bytes() function to get the number of elements to
// specify
//   the number of elements that should be specified to get a bloom filter of
//   the given size (before constructing).
// - Added bloom_filter_supported() function to check AVX2 support at runtime.
// - Added SimdBlockFilterFixed::union_reduction() to do a global MPI bitwise OR
//   allreduce of the filter across ranks.
// - Added dummy SimdBlockFilterFixed class when AVX2 is not found, to appease
//   the compiler when building extension modules that don't need the
//   march=haswell flag because they don't use bloom filters. Note that we need
//   this because mpi_comm_info depends on SimdBlockFilterFixed and
//   mpi_comm_info is in _bodo_common.h :(

// Copied from Apache Impala (incubating), usable under the terms in the Apache
// License, Version 2.0.

// This is a block Bloom filter (from Putze et al.'s "Cache-, Hash- and
// Space-Efficient Bloom Filters") with some twists:
//
// 1. Each block is a split Bloom filter - see Section 2.1 of Broder and
// Mitzenmacher's "Network Applications of Bloom Filters: A Survey".
//
// 2. The number of bits set per Add() is constant in order to take advantage of
// SIMD instructions.

#pragma once

#include <algorithm>
#include <cstdlib>
#include <memory>

#include "../_mpi.h"
#include "../tracing.h"
#include "hashutil.h"

using uint32_t = ::std::uint32_t;
using uint64_t = ::std::uint64_t;

inline bool bloom_filter_supported() {
// TODO [BE-1372]: https://bodo.atlassian.net/browse/BE-1372
#if defined(__aarch64__)
    return true;
#elif defined(_MSC_VER) || defined(__APPLE__)
    return false;
#else
    return __builtin_cpu_supports("avx2");
#endif
}

#if !defined(_MSC_VER)
__attribute__((always_inline))
#endif
inline uint32_t
reduce(uint32_t hash, uint32_t n) {
    // http://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
    return (uint32_t)(((uint64_t)hash * n) >> 32);
}

// Bodo change: Rename from rotl64 to __rotl64 to avoid ambiguity issues in the
// compiler when importing the datasketches library.
static inline uint64_t __rotl64(uint64_t n, unsigned int c) {
    // assumes width is a power of 2
    const unsigned int mask = (CHAR_BIT * sizeof(n) - 1);
    // assert ( (c<=mask) &&"rotate by type width or more");
    c &= mask;
    return (n << c) | (n >> ((-c) & mask));
}

#ifdef __AVX2__
#include <x86intrin.h>

#define BLOOM_SIZEOF_BUCKET 32  // in bytes

static inline size_t bloom_size_bytes(size_t n_bits) {
    size_t bucketCount = ::std::max(size_t(1), n_bits / 24);
    return (bucketCount * BLOOM_SIZEOF_BUCKET);
}

static inline size_t num_elements_for_bytes(size_t n_bytes) {
    if (n_bytes < BLOOM_SIZEOF_BUCKET) {
        return 1;
    }
    size_t num_buckets = n_bytes / BLOOM_SIZEOF_BUCKET;
    return num_buckets * 24;
}

template <typename HashFamily = ::hashing::SimpleMixSplit>
class SimdBlockFilterFixed {
   private:
    // The filter is divided up into Buckets:
    using Bucket = uint32_t[8];

    const int bucketCount;

    Bucket* directory_;
    size_t alloc_size;

    HashFamily hasher_;

   public:
    // Consumes at most (1 << log_heap_space) bytes on the heap:
    // XXX bits as int is dangerous.
    // Right now we are fine because bloom_size_bytes() takes size_t and we use
    // that function to skip making very large bloom filters
    explicit SimdBlockFilterFixed(const int bits);
    ~SimdBlockFilterFixed() noexcept;
    void Add(const uint64_t key) noexcept;

    // Add multiple items to the filter.
    void AddAll(const std::shared_ptr<uint64_t[]>& data, const size_t start,
                const size_t end);
    void AddAll(const std::shared_ptr<uint32_t[]>& data, const size_t start,
                const size_t end);

    bool Find(const uint64_t key) const noexcept;
    uint64_t SizeInBytes() const { return sizeof(Bucket) * bucketCount; }

    void union_reduction() {
        tracing::Event ev("bloom_union_reduction");
        // disable use of MPI_IN_PLACE since it causes incorrect results
        // with I_MPI_ADJUST_ALLREDUCE=4
        // See [BE-2377]
        // MPI_Allreduce(MPI_IN_PLACE, directory_,
        // static_cast<int>(SizeInBytes() / sizeof(uint64_t)),
        //              MPI_UINT64_T, MPI_BOR, MPI_COMM_WORLD);
        Bucket* recv_directory_;
        const int recv_malloc_failed = posix_memalign(
            reinterpret_cast<void**>(&recv_directory_), 64, alloc_size);
        if (recv_malloc_failed)
            throw ::std::bad_alloc();
        int err =
            MPI_Allreduce(directory_, recv_directory_,
                          static_cast<int>(SizeInBytes() / sizeof(uint64_t)),
                          MPI_UINT64_T, MPI_BOR, MPI_COMM_WORLD);
        if (err) {
            throw std::runtime_error(
                "SimdBlockFilterFixed::union_reduction: MPI error on "
                "MPI_Allreduce:");
        }
        free(directory_);
        directory_ = recv_directory_;
    }

   private:
    // A helper function for Insert()/Find(). Turns a 32-bit hash into a 256-bit
    // Bucket with 1 single 1-bit set in each 32-bit lane.
    static __m256i MakeMask(const uint32_t hash) noexcept;

    void ApplyBlock(uint64_t* tmp, int block, int len);
};

template <typename HashFamily>
SimdBlockFilterFixed<HashFamily>::SimdBlockFilterFixed(const int bits)
    // bits / 16: fpp 0.1777%, 75.1%
    // bits / 20: fpp 0.4384%, 63.4%
    // bits / 22: fpp 0.6692%, 61.1%
    // bits / 24: fpp 0.9765%, 59.7% <<== seems to be best (1% fpp seems
    // important) bits / 26: fpp 1.3769%, 59.3% bits / 28: fpp 1.9197%, 60.3%
    // bits / 32: fpp 3.3280%, 63.0%
    : bucketCount(::std::max(1, bits / 24)), directory_(nullptr), hasher_() {
    if (!__builtin_cpu_supports("avx2")) {
        throw ::std::runtime_error(
            "SimdBlockFilterFixed does not work without AVX2 instructions");
    }
    alloc_size = bucketCount * sizeof(Bucket);
    const int malloc_failed =
        posix_memalign(reinterpret_cast<void**>(&directory_), 64, alloc_size);
    if (malloc_failed)
        throw ::std::bad_alloc();
    memset(directory_, 0, alloc_size);
}

template <typename HashFamily>
SimdBlockFilterFixed<HashFamily>::~SimdBlockFilterFixed() noexcept {
    free(directory_);
    directory_ = nullptr;
}

// The SIMD reinterpret_casts technically violate C++'s strict aliasing rules.
// However, we compile with -fno-strict-aliasing.
template <typename HashFamily>
[[gnu::always_inline]] inline __m256i
SimdBlockFilterFixed<HashFamily>::MakeMask(const uint32_t hash) noexcept {
    const __m256i ones = _mm256_set1_epi32(1);
    // Odd constants for hashing:
    const __m256i rehash =
        _mm256_setr_epi32(0x47b6137bU, 0x44974d91U, 0x8824ad5bU, 0xa2b7289dU,
                          0x705495c7U, 0x2df1424bU, 0x9efc4947U, 0x5c6bfb31U);
    // Load hash into a YMM register, repeated eight times
    __m256i hash_data = _mm256_set1_epi32(hash);
    // Multiply-shift hashing ala Dietzfelbinger et al.: multiply 'hash' by
    // eight different odd constants, then keep the 5 most significant bits from
    // each product.
    hash_data = _mm256_mullo_epi32(rehash, hash_data);
    hash_data = _mm256_srli_epi32(hash_data, 27);
    // Use these 5 bits to shift a single bit to a location in each 32-bit lane
    return _mm256_sllv_epi32(ones, hash_data);
}

template <typename HashFamily>
[[gnu::always_inline]] inline void SimdBlockFilterFixed<HashFamily>::Add(
    const uint64_t key) noexcept {
    const auto hash = hasher_(key);
    const uint32_t bucket_idx = reduce(__rotl64(hash, 32), bucketCount);
    const __m256i mask = MakeMask(hash);
    __m256i* const bucket = &reinterpret_cast<__m256i*>(directory_)[bucket_idx];
    _mm256_store_si256(bucket, _mm256_or_si256(*bucket, mask));
}

const int blockShift = 14;
const int blockLen = 1 << blockShift;

template <typename HashFamily>
void SimdBlockFilterFixed<HashFamily>::ApplyBlock(uint64_t* tmp, int block,
                                                  int len) {
    for (int i = 0; i < len; i += 2) {
        uint64_t hash = tmp[(block << blockShift) + i];
        uint32_t bucket_idx = tmp[(block << blockShift) + i + 1];
        const __m256i mask = MakeMask(hash);
        __m256i* const bucket =
            &reinterpret_cast<__m256i*>(directory_)[bucket_idx];
        _mm256_store_si256(bucket, _mm256_or_si256(*bucket, mask));
    }
}

template <typename HashFamily>
void SimdBlockFilterFixed<HashFamily>::AddAll(
    const std::shared_ptr<uint64_t[]>& keys, const size_t start,
    const size_t end) {
    int blocks = 1 + bucketCount / blockLen;
    uint64_t* tmp = new uint64_t[blocks * blockLen];
    int* tmpLen = new int[blocks]();
    for (size_t i = start; i < end; i++) {
        uint64_t key = keys[i];
        uint64_t hash = hasher_(key);
        uint32_t bucket_idx = reduce(__rotl64(hash, 32), bucketCount);
        int block = bucket_idx >> blockShift;
        int len = tmpLen[block];
        tmp[(block << blockShift) + len] = hash;
        tmp[(block << blockShift) + len + 1] = bucket_idx;
        tmpLen[block] = len + 2;
        if (len + 2 == blockLen) {
            ApplyBlock(tmp, block, len + 1);
            tmpLen[block] = 0;
        }
    }
    for (int block = 0; block < blocks; block++) {
        ApplyBlock(tmp, block, tmpLen[block]);
    }
    delete[] tmp;
    delete[] tmpLen;
}

template <typename HashFamily>
void SimdBlockFilterFixed<HashFamily>::AddAll(
    const std::shared_ptr<uint32_t[]>& keys, const size_t start,
    const size_t end) {
    int blocks = 1 + bucketCount / blockLen;
    uint64_t* tmp = new uint64_t[blocks * blockLen];
    int* tmpLen = new int[blocks]();
    for (size_t i = start; i < end; i++) {
        uint64_t key = static_cast<uint64_t>(keys[i]);
        uint64_t hash = hasher_(key);
        uint32_t bucket_idx = reduce(__rotl64(hash, 32), bucketCount);
        int block = bucket_idx >> blockShift;
        int len = tmpLen[block];
        tmp[(block << blockShift) + len] = hash;
        tmp[(block << blockShift) + len + 1] = bucket_idx;
        tmpLen[block] = len + 2;
        if (len + 2 == blockLen) {
            ApplyBlock(tmp, block, len + 1);
            tmpLen[block] = 0;
        }
    }
    for (int block = 0; block < blocks; block++) {
        ApplyBlock(tmp, block, tmpLen[block]);
    }
    delete[] tmp;
    delete[] tmpLen;
}

template <typename HashFamily>
[[gnu::always_inline]] inline bool SimdBlockFilterFixed<HashFamily>::Find(
    const uint64_t key) const noexcept {
    const auto hash = hasher_(key);
    const uint32_t bucket_idx = reduce(__rotl64(hash, 32), bucketCount);
    const __m256i mask = MakeMask(hash);
    const __m256i bucket = reinterpret_cast<__m256i*>(directory_)[bucket_idx];
    // We should return true if 'bucket' has a one wherever 'mask' does.
    // _mm256_testc_si256 takes the negation of its first argument and ands that
    // with its second argument. In our case, the result is zero everywhere iff
    // there is a one in 'bucket' wherever 'mask' is one. testc returns 1 if the
    // result is 0 everywhere and returns 0 otherwise.
    return _mm256_testc_si256(bucket, mask);
}

///////////////////////////////////////////////////////////////////
/// 64-byte version
///////////////////////////////////////////////////////////////////

struct mask64bytes {
    __m256i first;
    __m256i second;
};

typedef struct mask64bytes mask64bytes_t;

template <typename HashFamily = ::hashing::SimpleMixSplit>
class SimdBlockFilterFixed64 {
   private:
    // The filter is divided up into Buckets:
    using Bucket = mask64bytes_t;

    const int bucketCount;

    Bucket* directory_;

    HashFamily hasher_;

   public:
    // Consumes at most (1 << log_heap_space) bytes on the heap:
    explicit SimdBlockFilterFixed64(const int bits);
    ~SimdBlockFilterFixed64() noexcept;
    void Add(const uint64_t key) noexcept;

    bool Find(const uint64_t key) const noexcept;
    uint64_t SizeInBytes() const { return sizeof(Bucket) * bucketCount; }

   private:
    static mask64bytes_t MakeMask(const uint64_t hash) noexcept;
};

template <typename HashFamily>
SimdBlockFilterFixed64<HashFamily>::SimdBlockFilterFixed64(const int bits)

    : bucketCount(::std::max(1, bits / 50)), directory_(nullptr), hasher_() {
    if (!__builtin_cpu_supports("avx2")) {
        throw ::std::runtime_error(
            "SimdBlockFilterFixed64 does not work without AVX2 instructions");
    }
    const size_t alloc_size = bucketCount * sizeof(Bucket);
    const int malloc_failed =
        posix_memalign(reinterpret_cast<void**>(&directory_), 64, alloc_size);
    if (malloc_failed)
        throw ::std::bad_alloc();
    memset(directory_, 0, alloc_size);
}

template <typename HashFamily>
SimdBlockFilterFixed64<HashFamily>::~SimdBlockFilterFixed64() noexcept {
    free(directory_);
    directory_ = nullptr;
}

template <typename HashFamily>
[[gnu::always_inline]] inline mask64bytes_t
SimdBlockFilterFixed64<HashFamily>::MakeMask(const uint64_t hash) noexcept {
    const __m256i ones = _mm256_set1_epi64x(1);
    const __m256i rehash1 =
        _mm256_setr_epi32(0x47b6137bU, 0x44974d91U, 0x8824ad5bU, 0xa2b7289dU,
                          0x705495c7U, 0x2df1424bU, 0x9efc4947U, 0x5c6bfb31U);
    mask64bytes_t answer;
    __m256i hash_data = _mm256_set1_epi32(hash);
    __m256i h = _mm256_mullo_epi32(rehash1, hash_data);
    h = _mm256_srli_epi32(h, 26);
    answer.first = _mm256_unpackhi_epi32(h, _mm256_setzero_si256());
    answer.first = _mm256_sllv_epi64(ones, answer.first);
    answer.second = _mm256_unpacklo_epi32(h, _mm256_setzero_si256());
    answer.second = _mm256_sllv_epi64(ones, answer.second);
    return answer;
}

template <typename HashFamily>
[[gnu::always_inline]] inline void SimdBlockFilterFixed64<HashFamily>::Add(
    const uint64_t key) noexcept {
    const auto hash = hasher_(key);
    const uint32_t bucket_idx = reduce(__rotl64(hash, 32), bucketCount);
    mask64bytes_t mask = MakeMask(hash);
    mask64bytes_t* const bucket =
        &reinterpret_cast<mask64bytes_t*>(directory_)[bucket_idx];
    bucket->first = _mm256_or_si256(mask.first, bucket->first);
    bucket->second = _mm256_or_si256(mask.second, bucket->second);
}

template <typename HashFamily>
[[gnu::always_inline]] inline bool SimdBlockFilterFixed64<HashFamily>::Find(
    const uint64_t key) const noexcept {
    const auto hash = hasher_(key);
    const uint32_t bucket_idx = reduce(__rotl64(hash, 32), bucketCount);
    const mask64bytes_t mask = MakeMask(hash);
    const mask64bytes_t bucket =
        reinterpret_cast<mask64bytes_t*>(directory_)[bucket_idx];
    return _mm256_testc_si256(bucket.first, mask.first) &
           _mm256_testc_si256(bucket.second, mask.second);
}

// #endif //__AVX2__

///////////////////
// 16-byte version ARM
//////////////////
#elif defined(__aarch64__)
#include <arm_neon.h>

#define BLOOM_SIZEOF_BUCKET sizeof(uint16x8_t)

static inline size_t bloom_size_bytes(size_t n_bits) {
    size_t bucketCount = ::std::max(size_t(1), n_bits / 10);
    return (bucketCount * BLOOM_SIZEOF_BUCKET);
}

static inline size_t num_elements_for_bytes(size_t n_bytes) {
    if (n_bytes <= BLOOM_SIZEOF_BUCKET) {
        return 1;
    }
    size_t num_buckets = n_bytes / BLOOM_SIZEOF_BUCKET;
    return num_buckets * 10;
}

template <typename HashFamily = ::hashing::SimpleMixSplit>
class SimdBlockFilterFixed {
   private:
    // The filter is divided up into Buckets:
    using Bucket = uint16x8_t;

    const int bucketCount;

    Bucket* directory_;
    size_t alloc_size;

    HashFamily hasher_;

   public:
    // Consumes at most (1 << log_heap_space) bytes on the heap:
    explicit SimdBlockFilterFixed(const int bits);
    ~SimdBlockFilterFixed() noexcept;
    void Add(const uint64_t key) noexcept;

    void AddAll(const std::shared_ptr<uint32_t[]>& data, const size_t start,
                const size_t end);

    bool Find(const uint64_t key) const noexcept;
    uint64_t SizeInBytes() const { return sizeof(Bucket) * bucketCount; }

    void union_reduction() {
        tracing::Event ev("bloom_union_reduction");
        // disable use of MPI_IN_PLACE since it causes incorrect results
        // wiht I_MPI_ADJUST_ALLREDUCE=4
        // See [BE-2377]
        // MPI_Allreduce(MPI_IN_PLACE, directory_,
        // static_cast<int>(SizeInBytes() / sizeof(uint64_t)),
        //              MPI_UINT64_T, MPI_BOR, MPI_COMM_WORLD);
        Bucket* recv_directory_;
        const int recv_malloc_failed = posix_memalign(
            reinterpret_cast<void**>(&recv_directory_), 64, alloc_size);
        if (recv_malloc_failed)
            throw ::std::bad_alloc();
        int err =
            MPI_Allreduce(directory_, recv_directory_,
                          static_cast<int>(SizeInBytes() / sizeof(uint64_t)),
                          MPI_UINT64_T, MPI_BOR, MPI_COMM_WORLD);
        if (err) {
            throw std::runtime_error(
                "SimdBlockFilterFixed::union_reduction: MPI error on "
                "MPI_Allreduce:");
        }
        free(directory_);
        directory_ = recv_directory_;
    }

   private:
    // A helper function for Insert()/Find(). Turns a 32-bit hash into a 256-bit
    // Bucket with 1 single 1-bit set in each 32-bit lane.
    static Bucket MakeMask(const uint16_t hash) noexcept;

    void ApplyBlock(uint64_t* tmp, int block, int len);
};

template <typename HashFamily>
SimdBlockFilterFixed<HashFamily>::SimdBlockFilterFixed(const int bits)
    : bucketCount(::std::max(1, bits / 10)), directory_(nullptr), hasher_() {
    alloc_size = bucketCount * sizeof(Bucket);
    const int malloc_failed =
        posix_memalign(reinterpret_cast<void**>(&directory_), 64, alloc_size);
    if (malloc_failed)
        throw ::std::bad_alloc();
    memset(directory_, 0, alloc_size);
}

template <typename HashFamily>
SimdBlockFilterFixed<HashFamily>::~SimdBlockFilterFixed() noexcept {
    free(directory_);
    directory_ = nullptr;
}

template <typename HashFamily>
[[gnu::always_inline]] inline uint16x8_t
SimdBlockFilterFixed<HashFamily>::MakeMask(const uint16_t hash) noexcept {
    const uint16x8_t ones = {1, 1, 1, 1, 1, 1, 1, 1};
    const uint16x8_t rehash = {0x79d8, 0xe722, 0xf2fb, 0x21ec,
                               0x121b, 0x2302, 0x705a, 0x6e87};
    uint16x8_t hash_data = {hash, hash, hash, hash, hash, hash, hash, hash};
    uint16x8_t answer = vmulq_u16(hash_data, rehash);
    answer = vshrq_n_u16(answer, 12);
    answer = vshlq_u16(ones, vreinterpretq_s16_u16(answer));
    return answer;
}

template <typename HashFamily>
[[gnu::always_inline]] inline void SimdBlockFilterFixed<HashFamily>::Add(
    const uint64_t key) noexcept {
    const auto hash = hasher_(key);
    const uint32_t bucket_idx = reduce(__rotl64(hash, 32), bucketCount);
    const uint16x8_t mask = MakeMask(hash);
    uint16x8_t bucket = directory_[bucket_idx];
    directory_[bucket_idx] = vorrq_u16(mask, bucket);
}

template <typename HashFamily>
void SimdBlockFilterFixed<HashFamily>::AddAll(
    const std::shared_ptr<uint32_t[]>& keys, const size_t start,
    const size_t end) {
    for (size_t i = start; i < end; i++) {
        // TODO: more efficient implementation like the AVX version does?
        Add(static_cast<uint64_t>(keys[i]));
    }
}

template <typename HashFamily>
[[gnu::always_inline]] inline bool SimdBlockFilterFixed<HashFamily>::Find(
    const uint64_t key) const noexcept {
    const auto hash = hasher_(key);
    const uint32_t bucket_idx = reduce(__rotl64(hash, 32), bucketCount);
    const uint16x8_t mask = MakeMask(hash);
    const uint16x8_t bucket = directory_[bucket_idx];
    uint16x8_t an = vbicq_u16(mask, bucket);
    uint64x2_t v64 = vreinterpretq_u64_u16(an);
    uint32x2_t v32 = vqmovn_u64(v64);
    uint64x1_t result = vreinterpret_u64_u32(v32);
    return vget_lane_u64(result, 0) == 0;
}

// #endif // __aarch64__

///////////////////////////////////////////////////////////////////
/// 16-byte version (not very good)
///////////////////////////////////////////////////////////////////

#elif defined(__SSE4_1__)

#include <smmintrin.h>

template <typename HashFamily = ::hashing::SimpleMixSplit>
class SimdBlockFilterFixed16 {
   private:
    // The filter is divided up into Buckets:
    using Bucket = __m128i;

    const int bucketCount;

    Bucket* directory_;

    HashFamily hasher_;

   public:
    // Consumes at most (1 << log_heap_space) bytes on the heap:
    explicit SimdBlockFilterFixed16(const int bits);
    ~SimdBlockFilterFixed16() noexcept;
    void Add(const uint64_t key) noexcept;

    bool Find(const uint64_t key) const noexcept;
    uint64_t SizeInBytes() const { return sizeof(Bucket) * bucketCount; }

   private:
    static __m128i MakeMask(const uint64_t hash) noexcept;
};

template <typename HashFamily>
SimdBlockFilterFixed16<HashFamily>::SimdBlockFilterFixed16(const int bits)

    : bucketCount(::std::max(1, bits / 10)), directory_(nullptr), hasher_() {
    const size_t alloc_size = bucketCount * sizeof(Bucket);
    const int malloc_failed =
        posix_memalign(reinterpret_cast<void**>(&directory_), 64, alloc_size);
    if (malloc_failed)
        throw ::std::bad_alloc();
    memset(directory_, 0, alloc_size);
}

template <typename HashFamily>
SimdBlockFilterFixed16<HashFamily>::~SimdBlockFilterFixed16() noexcept {
    free(directory_);
    directory_ = nullptr;
}

template <typename HashFamily>
[[gnu::always_inline]] inline __m128i
SimdBlockFilterFixed16<HashFamily>::MakeMask(const uint64_t hash) noexcept {
    const __m128i rehash1 = _mm_setr_epi16(0x47b5, 0x4497, 0x8823, 0xa2b7,
                                           0x7053, 0x2df1, 0x9efc, 0x5c6b);
    __m128i hash_data = _mm_set1_epi32(hash);
    __m128i h = _mm_mulhi_epi16(rehash1, hash_data);
    return _mm_shuffle_epi8(_mm_set_epi8(1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4,
                                         8, 16, 32, 64, -128),
                            h);
}

template <typename HashFamily>
[[gnu::always_inline]] inline void SimdBlockFilterFixed16<HashFamily>::Add(
    const uint64_t key) noexcept {
    const auto hash = hasher_(key);
    const uint32_t bucket_idx = reduce(__rotl64(hash, 32), bucketCount);
    __m128i mask = MakeMask(hash);
    __m128i* const bucket = reinterpret_cast<__m128i*>(directory_) + bucket_idx;
    __m128i bucketvalue = _mm_loadu_si128(bucket);
    bucketvalue = _mm_or_si128(bucketvalue, mask);
    _mm_storeu_si128(bucket, bucketvalue);
}

template <typename HashFamily>
[[gnu::always_inline]] inline bool SimdBlockFilterFixed16<HashFamily>::Find(
    const uint64_t key) const noexcept {
    const auto hash = hasher_(key);
    const uint32_t bucket_idx = reduce(__rotl64(hash, 32), bucketCount);
    const __m128i mask = MakeMask(hash);
    __m128i* const bucket = reinterpret_cast<__m128i*>(directory_) + bucket_idx;
    __m128i bucketvalue = _mm_loadu_si128(bucket);
    return _mm_testc_si128(bucketvalue, mask);
}

// #endif // #ifdef __SSE4_1__
#else

static inline size_t bloom_size_bytes(size_t n_bits) { return 0; }

static inline size_t num_elements_for_bytes(size_t n_bytes) { return 0; }

template <typename HashFamily = ::hashing::SimpleMixSplit>
class SimdBlockFilterFixed {
   public:
    explicit SimdBlockFilterFixed(const int bits) {
        throw ::std::runtime_error("Dummy filter constructor called");
    }
    ~SimdBlockFilterFixed() noexcept {};
    void Add(const uint64_t key) noexcept {};

    // Add multiple items to the filter.
    void AddAll(const std::shared_ptr<uint64_t[]>&& data, const size_t start,
                const size_t end) {}
    void AddAll(const std::shared_ptr<uint32_t[]>& data, const size_t start,
                const size_t end) {}
    bool Find(const uint64_t key) const noexcept { return false; }
    uint64_t SizeInBytes() const { return 0; }
    void union_reduction() {}
};

#endif
