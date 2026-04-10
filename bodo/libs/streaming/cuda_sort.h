#pragma once

#include <mpi.h>
#include <cudf/table/table.hpp>
#include "../_bodo_common.h"
#include "../gpu_utils.h"

#ifdef USE_CUDF
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>

class CudaSortState {
    // Implements Parallel Sample Sort (PSRS) for distributed sorting of cudf
    // tables across GPU ranks.
   public:
    CudaSortState(std::shared_ptr<bodo::Schema> schema,
                  std::vector<cudf::size_type> const& key_indices,
                  std::vector<cudf::order> const& column_order,
                  std::vector<cudf::null_order> const& null_precedence,
                  int64_t limit = -1, int64_t offset = 0);

    /**
     * @brief Consume a batch of data to be sorted.
     * @param table Input batch.
     * @param input_se Stream and event for the input batch.
     */
    void ConsumeBatch(std::shared_ptr<cudf::table> table,
                      std::shared_ptr<StreamAndEvent> input_se);

    /**
     * @brief Finalize the accumulation phase and perform the distributed sort.
     * @param local_is_last Whether this is the last batch on this rank.
     * @param input_se Stream and event for synchronization.
     * @return Global is last flag.
     */
    bool FinalizeAccumulation(bool local_is_last,
                              std::shared_ptr<StreamAndEvent> input_se);

    /**
     * @brief Finalize the sort phase and compute limits/offsets across ranks.
     * This MUST be called exactly once by all ranks after
     * FinalizeAccumulation returns true and before any calls to
     * GetOutputBatch.
     */
    void FinalizeSort();

    /**
     * @brief Get the sorted output batch.
     * @param[out] out_is_last Whether this is the last output batch.
     * @param stream CUDA stream for execution.
     * @return Unique pointer to the sorted cudf table.
     */
    std::unique_ptr<cudf::table> GetOutputBatch(bool& out_is_last,
                                                rmm::cuda_stream_view stream);

    /**
     * @brief Get the underlying MPI communicator.
     * @return MPI_Comm
     */
    MPI_Comm get_mpi_comm() { return shuffle_manager.get_mpi_comm(); }

   private:
    std::shared_ptr<bodo::Schema> schema;
    std::vector<cudf::size_type> key_indices;
    std::vector<cudf::order> column_order;
    std::vector<cudf::null_order> null_precedence;
    const int64_t limit;
    const int64_t offset;

    // Accumulation buffer for local batches
    std::vector<std::shared_ptr<cudf::table>> accumulation_buffer;

    // Received tables from shuffle
    std::vector<std::shared_ptr<cudf::table>> received_tables;

    // Final merged result
    std::unique_ptr<cudf::table> final_result = nullptr;

    // Local slice indices (pre-calculated in FinalizeSort)
    int64_t local_slice_start = 0;
    int64_t local_slice_end = 0;

    GpuRangeShuffleManager shuffle_manager;
    GpuTableAllGatherManager sample_gatherer;

    enum class State {
        ACCUMULATING,
        GATHERING_SAMPLES,
        SHUFFLING,
        MERGING,
        DONE
    };
    State state = State::ACCUMULATING;

    // Locally sorted data waiting for pivots
    std::unique_ptr<cudf::table> local_table;
    // Local samples being broadcasted
    std::shared_ptr<cudf::table> local_samples;
    // Samples received from other ranks
    std::vector<std::shared_ptr<cudf::table>> received_samples;

    /**
     * @brief Step 1 of PSRS: Local sort, sample, and start broadcast.
     */
    void ExecutePsrsStep1(rmm::cuda_stream_view stream);

    /**
     * @brief Step 2 of PSRS: Collect samples, pick pivots, partition, and start
     * shuffle.
     */
    void ExecutePsrsStep2(rmm::cuda_stream_view stream);
};

#else
class CudaSortState {};
#endif
