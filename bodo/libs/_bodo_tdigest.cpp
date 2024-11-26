/* Monkey-patch of the Arrow tdigest library:
 * Current link:
 * https://github.com/apache/arrow/blob/apache-arrow-11.0.0/cpp/src/arrow/util/tdigest.cc
 * Permalink at the time of the original copying:
 * https://github.com/apache/arrow/blob/5de56928e0fe43f02005552eee058de57ffb2682/cpp/src/arrow/util/tdigest.cc
 * With the following changes:
 * - Status class and Validate functions removed
 * - Checking commands removed
 * - Extra methods that facilitate merging information from TDigest objects
 *   across multiple ranks
 */
#include "_bodo_tdigest.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <tuple>
#include <vector>

#include "_bodo_common.h"
#include "_distributed.h"

namespace {

// a numerically stable lerp is unbelievably complex
// but we are *approximating* the quantile, so let's keep it simple
double Lerp(double a, double b, double t) { return a + t * (b - a); }

// histogram bin
struct Centroid {
    double mean;
    double weight;  // # data points in this bin

    // merge with another centroid
    void Merge(const Centroid& centroid) {
        weight += centroid.weight;
        mean += (centroid.mean - mean) * centroid.weight / weight;
    }
};

// scale function K0: linear function, as baseline
struct ScalerK0 {
    explicit ScalerK0(uint32_t delta) : delta_norm(delta / 2.0) {}

    double K(double q) const { return delta_norm * q; }
    double Q(double k) const { return k / delta_norm; }

    const double delta_norm;
};

// scale function K1
struct ScalerK1 {
    explicit ScalerK1(uint32_t delta) : delta_norm(delta / (2.0 * M_PI)) {}

    double K(double q) const { return delta_norm * std::asin(2 * q - 1); }
    double Q(double k) const { return (std::sin(k / delta_norm) + 1) / 2; }

    const double delta_norm;
};

// implements t-digest merging algorithm
template <class T = ScalerK1>
class TDigestMerger : private T {
   public:
    explicit TDigestMerger(uint32_t delta) : T(delta) { Reset(0, nullptr); }

    void Reset(double total_weight, std::vector<Centroid>* tdigest) {
        total_weight_ = total_weight;
        tdigest_ = tdigest;
        if (tdigest_) {
            tdigest_->resize(0);
        }
        weight_so_far_ = 0;
        weight_limit_ = -1;  // trigger first centroid merge
    }

    // merge one centroid from a sorted centroid stream
    void Add(const Centroid& centroid) {
        auto& td = *tdigest_;
        const double weight = weight_so_far_ + centroid.weight;
        if (weight <= weight_limit_) {
            td.back().Merge(centroid);
        } else {
            const double quantile = weight_so_far_ / total_weight_;
            const double next_weight_limit =
                total_weight_ * this->Q(this->K(quantile) + 1);
            // weight limit should be strictly increasing, until the last
            // centroid
            if (next_weight_limit <= weight_limit_) {
                weight_limit_ = total_weight_;
            } else {
                weight_limit_ = next_weight_limit;
            }
            td.push_back(centroid);  // should never exceed capacity and trigger
                                     // reallocation
        }
        weight_so_far_ = weight;
    }

   private:
    double total_weight_;   // total weight of this tdigest
    double weight_so_far_;  // accumulated weight till current bin
    double weight_limit_;   // max accumulated weight to move to next bin
    std::vector<Centroid>* tdigest_;
};

}  // namespace

class TDigest::TDigestImpl {
   public:
    explicit TDigestImpl(uint32_t delta)
        : delta_(delta > 10 ? delta : 10), merger_(delta_) {
        tdigests_[0].reserve(delta_);
        tdigests_[1].reserve(delta_);
        Reset();
    }

    // Bodo change: Dump method removed as it is uncessary and relies on
    // other PyArrow internals

    // Bodo change: Validate method removed as it is uncessary and relies on
    // other PyArrow internals

    void Reset() {
        tdigests_[0].resize(0);
        tdigests_[1].resize(0);
        current_ = 0;
        total_weight_ = 0;
        min_ = std::numeric_limits<double>::max();
        max_ = std::numeric_limits<double>::lowest();
        merger_.Reset(0, nullptr);
    }

    // merge with other tdigests
    void Merge(const std::vector<const TDigestImpl*>& tdigest_impls) {
        // current and end iterator
        using CentroidIter = std::vector<Centroid>::const_iterator;
        using CentroidIterPair = std::pair<CentroidIter, CentroidIter>;
        // use a min-heap to find next minimal centroid from all tdigests
        auto centroid_gt = [](const CentroidIterPair& lhs,
                              const CentroidIterPair& rhs) {
            return lhs.first->mean > rhs.first->mean;
        };
        using CentroidQueue =
            std::priority_queue<CentroidIterPair, std::vector<CentroidIterPair>,
                                decltype(centroid_gt)>;

        // trivial dynamic memory allocated at runtime
        std::vector<CentroidIterPair> queue_buffer;
        queue_buffer.reserve(tdigest_impls.size() + 1);
        PUSH_IGNORED_COMPILER_ERROR("-Waggressive-loop-optimizations");
        CentroidQueue queue(std::move(centroid_gt), std::move(queue_buffer));
        POP_IGNORED_COMPILER_ERROR();

        const auto& this_tdigest = tdigests_[current_];
        if (this_tdigest.size() > 0) {
            queue.emplace(this_tdigest.cbegin(), this_tdigest.cend());
        }
        for (const TDigestImpl* td : tdigest_impls) {
            const auto& other_tdigest = td->tdigests_[td->current_];
            if (other_tdigest.size() > 0) {
                queue.emplace(other_tdigest.cbegin(), other_tdigest.cend());
                total_weight_ += td->total_weight_;
                min_ = std::min(min_, td->min_);
                max_ = std::max(max_, td->max_);
            }
        }

        merger_.Reset(total_weight_, &tdigests_[1 - current_]);
        CentroidIter current_iter, end_iter;
        // do k-way merge till one buffer left
        while (queue.size() > 1) {
            std::tie(current_iter, end_iter) = queue.top();
            merger_.Add(*current_iter);
            queue.pop();
            if (++current_iter != end_iter) {
                queue.emplace(current_iter, end_iter);
            }
        }
        // merge last buffer
        if (!queue.empty()) {
            std::tie(current_iter, end_iter) = queue.top();
            while (current_iter != end_iter) {
                merger_.Add(*current_iter++);
            }
        }
        merger_.Reset(0, nullptr);

        current_ = 1 - current_;
    }

    /* Bodo change: Function to merge across ranks. Merge with other tdigests
     * using only the minimal set of information communicated between them
     * across multiple ranks:
     * - The sum of weights on each rank
     * - The min and max value on each rank
     * - The number of centroids in each rank
     * - The means and weights of each centroid in each rank
     * The merger destructively modifies the TDigestImpl object on the root
     * rank while leaving other ranks unaltered.
     */
    void MPI_Merge() {
        // Bodo changes
        int myrank, n_pes;
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

        tracing::Event ev("TDigestImpl_MPI_Merge", true);

        // Gather the total weight, minimum value, maximum value, and number
        // of centroids from each rank onto vectors in the root rank
        std::vector<double> each_weight(n_pes);
        std::vector<double> each_min(n_pes);
        std::vector<double> each_max(n_pes);
        std::vector<int32_t> lengths(n_pes);

        int length = tdigests_[current_].size();

        c_gather_scalar(&total_weight_, each_weight.data(),
                        Bodo_CTypes::FLOAT64, false, 0);
        c_gather_scalar(&min_, each_min.data(), Bodo_CTypes::FLOAT64, false, 0);
        c_gather_scalar(&max_, each_max.data(), Bodo_CTypes::FLOAT64, false, 0);
        c_gather_scalar(&length, lengths.data(), Bodo_CTypes::INT32, false, 0);

        // Create a vector of the means of each centroid on the current rank
        // and the weights of each centroid on the current rank
        std::vector<double> local_means(length);
        std::vector<double> local_weights(length);
        for (int i = 0; i < length; i++) {
            local_means[i] = tdigests_[current_][i].mean;
            local_weights[i] = tdigests_[current_][i].weight;
        }

        // Calculate the displacements based on the lengths of each rank.
        int total_length = 0;
        for (int l : lengths) {
            total_length += l;
        }
        std::vector<int32_t> displs(n_pes);
        calc_disp(displs, lengths);

        // Gather the means/weights from each rank into vectors on the root rank
        std::vector<double> means(total_length);
        std::vector<double> weights(total_length);
        c_gatherv(local_means.data(), length, means.data(), lengths.data(),
                  displs.data(), Bodo_CTypes::FLOAT64, false, 0);
        c_gatherv(local_weights.data(), length, weights.data(), lengths.data(),
                  displs.data(), Bodo_CTypes::FLOAT64, false, 0);

        // Since the merger only has to occur on the root rank, all other ranks
        // can remain unchanged after sending their data to the root.
        if (myrank != 0)
            return;

        Reset();

        // End of Bodo changes

        // current and end iterator
        using CentroidIter = std::vector<Centroid>::const_iterator;
        using CentroidIterPair = std::pair<CentroidIter, CentroidIter>;
        // use a min-heap to find next minimal centroid from all tdigests
        auto centroid_gt = [](const CentroidIterPair& lhs,
                              const CentroidIterPair& rhs) {
            return lhs.first->mean > rhs.first->mean;
        };
        using CentroidQueue =
            std::priority_queue<CentroidIterPair, std::vector<CentroidIterPair>,
                                decltype(centroid_gt)>;

        // trivial dynamic memory allocated at runtime
        std::vector<CentroidIterPair> queue_buffer;
        queue_buffer.reserve(n_pes);
        PUSH_IGNORED_COMPILER_ERROR("-Waggressive-loop-optimizations");
        CentroidQueue queue(std::move(centroid_gt), std::move(queue_buffer));
        POP_IGNORED_COMPILER_ERROR();

        // Bodo changes

        // Loop across each batch of means/weights in the arrays, convert them
        // explicitly to a vector of centroids, then batch-add them to the
        // priority queue.
        int start = 0;
        int stop = 0;
        std::vector<std::vector<Centroid>> cvs(n_pes);
        for (int i = 0; i < n_pes; i++) {
            total_weight_ += each_weight[i];
            min_ = std::min(min_, each_min[i]);
            max_ = std::max(max_, each_max[i]);
            stop = start + lengths[i];
            for (int j = start; j < stop; j++) {
                cvs[i].push_back({means[j], weights[j]});
            }
            queue.emplace(cvs[i].cbegin(), cvs[i].cend());
            start = stop;
        }

        // End of Bodo changes

        merger_.Reset(total_weight_, &tdigests_[1 - current_]);
        CentroidIter current_iter, end_iter;
        // do k-way merge till one buffer left
        while (queue.size() > 1) {
            std::tie(current_iter, end_iter) = queue.top();
            merger_.Add(*current_iter);
            queue.pop();
            if (++current_iter != end_iter) {
                queue.emplace(current_iter, end_iter);
            }
        }
        // merge last buffer
        if (!queue.empty()) {
            std::tie(current_iter, end_iter) = queue.top();
            while (current_iter != end_iter) {
                merger_.Add(*current_iter++);
            }
        }
        merger_.Reset(0, nullptr);

        current_ = 1 - current_;
    }

    // merge input data with current tdigest
    void MergeInput(std::vector<double>& input) {
        total_weight_ += input.size();

        std::ranges::sort(input);
        min_ = std::min(min_, input.front());
        max_ = std::max(max_, input.back());

        // pick next minimal centroid from input and tdigest, feed to merger
        merger_.Reset(total_weight_, &tdigests_[1 - current_]);
        const auto& td = tdigests_[current_];
        uint32_t tdigest_index = 0, input_index = 0;
        while (tdigest_index < td.size() && input_index < input.size()) {
            if (td[tdigest_index].mean < input[input_index]) {
                merger_.Add(td[tdigest_index++]);
            } else {
                merger_.Add(
                    Centroid{.mean = input[input_index++], .weight = 1});
            }
        }
        while (tdigest_index < td.size()) {
            merger_.Add(td[tdigest_index++]);
        }
        while (input_index < input.size()) {
            merger_.Add(Centroid{.mean = input[input_index++], .weight = 1});
        }
        merger_.Reset(0, nullptr);

        input.resize(0);
        current_ = 1 - current_;
    }

    // Bodo changes: new function that contains the code originally in
    // the Quantile method
    double QuantileHelper(double q) const {
        const auto& td = tdigests_[current_];

        if (q < 0 || q > 1 || td.size() == 0) {
            return NAN;
        }

        const double index = q * total_weight_;
        if (index <= 1) {
            return min_;
        } else if (index >= total_weight_ - 1) {
            return max_;
        }

        // find centroid contains the index
        uint32_t ci = 0;
        double weight_sum = 0;
        for (; ci < td.size(); ++ci) {
            weight_sum += td[ci].weight;
            if (index <= weight_sum) {
                break;
            }
        }

        // deviation of index from the centroid center
        double diff = index + td[ci].weight / 2 - weight_sum;

        // index happen to be in a unit weight centroid
        if (td[ci].weight == 1 && std::abs(diff) < 0.5) {
            return td[ci].mean;
        }

        // find adjacent centroids for interpolation
        uint32_t ci_left = ci, ci_right = ci;
        if (diff > 0) {
            if (ci_right == td.size() - 1) {
                // index larger than center of last bin
                const Centroid* c = &td[ci_right];
                return Lerp(c->mean, max_, diff / (c->weight / 2));
            }
            ++ci_right;
        } else {
            if (ci_left == 0) {
                // index smaller than center of first bin
                const Centroid* c = &td[0];
                return Lerp(min_, c->mean, index / (c->weight / 2));
            }
            --ci_left;
            diff += td[ci_left].weight / 2 + td[ci_right].weight / 2;
        }

        // interpolate from adjacent centroids
        diff /= (td[ci_left].weight / 2 + td[ci_right].weight / 2);
        return Lerp(td[ci_left].mean, td[ci_right].mean, diff);
    }

    // Bodo changes: this method is turned into a wrapper that invokes the
    // true Quantile calculation (moved to QuantileHelper) and broadcasts
    // the results using MPI. Note: assumes that MPI_Merge has already been
    // called. Also, added the parallel parameter
    double Quantile(double q, bool parallel) const {
        int myrank, n_pes;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &n_pes);
        tracing::Event ev("TDigestImpl_Quantile", parallel);
        double res = 0.0;
        if (myrank == 0 || !parallel) {
            res = QuantileHelper(q);
        }
        if (parallel) {
            MPI_Datatype mpi_typ = get_MPI_typ(Bodo_CTypes::FLOAT64);
            CHECK_MPI(
                MPI_Bcast(&res, 1, mpi_typ, 0, MPI_COMM_WORLD),
                "TDigest::TDigestImpl::Quantile: MPI error on MPI_Bcast:");
        }
        return res;
    }

    double Mean() const {
        double sum = 0;
        for (const auto& centroid : tdigests_[current_]) {
            sum += centroid.mean * centroid.weight;
        }
        return total_weight_ == 0 ? NAN : sum / total_weight_;
    }

    double total_weight() const { return total_weight_; }

   private:
    // must be delcared before merger_, see constructor initialization list
    const uint32_t delta_;

    TDigestMerger<> merger_;
    double total_weight_;
    double min_, max_;

    // ping-pong buffer holds two tdigests, size = 2 * delta * sizeof(Centroid)
    std::vector<Centroid> tdigests_[2];
    // index of active tdigest buffer, 0 or 1
    int current_;
};

TDigest::TDigest(uint32_t delta, uint32_t buffer_size)
    : impl_(new TDigestImpl(delta)) {
    input_.reserve(buffer_size);
    Reset();
}

TDigest::~TDigest() = default;
TDigest::TDigest(TDigest&&) = default;
TDigest& TDigest::operator=(TDigest&&) = default;

void TDigest::Reset() {
    input_.resize(0);
    impl_->Reset();
}

void TDigest::Merge(const std::vector<TDigest>& others) {
    MergeInput();

    std::vector<const TDigestImpl*> other_impls;
    other_impls.reserve(others.size());
    for (auto& other : others) {
        other.MergeInput();
        other_impls.push_back(other.impl_.get());
    }
    impl_->Merge(other_impls);
}

// Bodo change: Validate method removed as it is unnecessary and relies on
// other PyArrow internals

// Bodo change: new method
void TDigest::MPI_Merge() {
    MergeInput();
    impl_->MPI_Merge();
}

void TDigest::Merge(const TDigest& other) {
    MergeInput();
    other.MergeInput();
    impl_->Merge({other.impl_.get()});
}

// Bodo change: added the parallel argument
double TDigest::Quantile(double q, bool parallel) const {
    MergeInput();
    return impl_->Quantile(q, parallel);
}

double TDigest::Mean() const {
    MergeInput();
    return impl_->Mean();
}

bool TDigest::is_empty() const {
    return input_.size() == 0 && impl_->total_weight() == 0;
}

void TDigest::MergeInput() const {
    if (input_.size() > 0) {
        impl_->MergeInput(input_);  // will mutate input_
    }
}
