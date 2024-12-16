// Monkey-patch of the Arrow tdigest library
// https://github.com/apache/arrow/blob/5de56928e0fe43f02005552eee058de57ffb2682/cpp/src/arrow/util/tdigest.h
// https://github.com/apache/arrow/blob/apache-arrow-11.0.0/cpp/src/arrow/util/tdigest.h
#pragma once

#include <cmath>
#include <concepts>
#include <memory>
#include <vector>

#ifndef ARROW_PREDICT_FALSE
#define ARROW_PREDICT_FALSE(x) (__builtin_expect(!!(x), 0))
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class TDigest {
   public:
    explicit TDigest(uint32_t delta = 100, uint32_t buffer_size = 500);
    ~TDigest();
    TDigest(TDigest&&);
    TDigest& operator=(TDigest&&);

    // reset and re-use this tdigest
    void Reset();

    // buffer a single data point, consume internal buffer if full
    // this function is intensively called and performance critical
    // call it only if you are sure no NAN exists in input data
    void Add(double value) {
        if (ARROW_PREDICT_FALSE(input_.size() == input_.capacity())) {
            MergeInput();
        }
        input_.push_back(value);
    }

    // skip NAN on adding
    template <typename T>
        requires std::floating_point<T>
    void NanAdd(T value) {
        if (!std::isnan(value))
            Add(value);
    }

    template <typename T>
        requires std::integral<T>
    void NanAdd(T value) {
        Add(static_cast<double>(value));
    }

    // merge with other t-digests, called infrequently
    void Merge(const std::vector<TDigest>& others);
    void Merge(const TDigest& other);
    void MPI_Merge();  // Bodo change

    // calculate quantile (Bodo change: added the parallel argument)
    double Quantile(double q, bool parallel = false) const;

    double Min() const { return Quantile(0); }
    double Max() const { return Quantile(1); }
    double Mean() const;

    // check if this tdigest contains no valid data points
    bool is_empty() const;

   private:
    // merge input data with current tdigest
    void MergeInput() const;

    // input buffer, size = buffer_size * sizeof(double)
    mutable std::vector<double> input_;

    // hide other members with pimpl
    class TDigestImpl;
    std::unique_ptr<TDigestImpl> impl_;
};
