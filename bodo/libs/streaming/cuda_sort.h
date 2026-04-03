#pragma once

#include <mpi.h>
#include <cudf/table/table.hpp>
#include "../_bodo_common.h"
#include "../gpu_utils.h"

#ifdef USE_CUDF
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>

/**
 * @brief State for distributed GPU PSRS sort.
 */
class CudaSortState {
   public:
    CudaSortState(std::shared_ptr<bodo::Schema> schema,
                  std::vector<cudf::size_type> const& key_indices,
                  std::vector<cudf::order> const& column_order,
                  std::vector<cudf::null_order> const& null_precedence);

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
     * @return Global is last flag.
     */
    bool FinalizeAccumulation(bool local_is_last);

    /**
     * @brief Get the sorted output batch.
     * @param[out] out_is_last Whether this is the last output batch.
     * @param stream CUDA stream for execution.
     * @return Unique pointer to the sorted cudf table.
     */
    std::unique_ptr<cudf::table> GetOutputBatch(bool& out_is_last,
                                                rmm::cuda_stream_view stream);

   private:
    std::shared_ptr<bodo::Schema> schema;
    std::shared_ptr<arrow::Schema> key_schema;
    std::vector<cudf::size_type> key_indices;
    std::vector<cudf::order> column_order;
    std::vector<cudf::null_order> null_precedence;

    // Accumulation buffer for local batches
    std::vector<std::shared_ptr<cudf::table>> accumulation_buffer;

    // Received tables from shuffle
    std::vector<std::shared_ptr<cudf::table>> received_tables;

    // Final merged result
    std::unique_ptr<cudf::table> final_result = nullptr;

    GpuRangeShuffleManager shuffle_manager;

    enum class State { ACCUMULATING, SHUFFLING, MERGING, DONE };
    State state = State::ACCUMULATING;

    /**
     * @brief Execute the PSRS algorithm after all data is accumulated.
     */
    void ExecutePsrs(rmm::cuda_stream_view stream);
};

#else
class CudaSortState {};
#endif
