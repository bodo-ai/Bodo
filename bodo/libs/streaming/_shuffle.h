#pragma once

#include <arrow/util/bit_util.h>
#include <unordered_set>
#include "../_bodo_common.h"
#include "../_distributed.h"
#include "../_query_profile_collector.h"
#include "../_table_builder.h"

#define DEFAULT_SHUFFLE_THRESHOLD 50 * 1024 * 1024  // 50MiB
#define MIN_SHUFFLE_THRESHOLD 50 * 1024 * 1024      // 50MiB
#define MAX_SHUFFLE_THRESHOLD 200 * 1024 * 1024     // 200MiB
#define DEFAULT_SHUFFLE_THRESHOLD_PER_MiB 12800     // 12.5KiB

// Factor in determining whether shuffle buffer is large enough to need cleared
constexpr float SHUFFLE_BUFFER_CUTOFF_MULTIPLIER = 3.0;

// Minimum utilization of shuffle buffer, used as a factor in determining when
// to clear
constexpr float SHUFFLE_BUFFER_MIN_UTILIZATION = 0.5;

// Tag for message sending the metadata of the outgoing arrays.
constexpr int SHUFFLE_METADATA_MSG_TAG = 0;

// Streaming batch size. The default of 4096 should match the default of
// bodosql_streaming_batch_size defined in __init__.py
static char* __env_streaming_batch_size_str =
    std::getenv("BODO_STREAMING_BATCH_SIZE");
const int STREAMING_BATCH_SIZE = __env_streaming_batch_size_str != nullptr
                                     ? std::stoi(__env_streaming_batch_size_str)
                                     : 32768;

#ifndef DEFAULT_SYNC_ITERS
// Default number of iterations between syncs
// NOTE: should be the same as default_stream_loop_sync_iters in __init__.py
#define DEFAULT_SYNC_ITERS 1000
#endif

#ifndef DEFAULT_SYNC_UPDATE_FREQ
// Update sync freq every 10 syncs by default.
#define DEFAULT_SYNC_UPDATE_FREQ 10
#endif

using comm_info_iter_t = std::vector<mpi_comm_info>::const_iterator;
using str_comm_info_iter_t = std::vector<mpi_str_comm_info>::const_iterator;
using len_iter_t = std::vector<uint64_t>::const_iterator;

/**
 * @brief Get the default shuffle threshold for all streaming operators. We
 * check the 'BODO_SHUFFLE_THRESHOLD' environment variable first. If it's not
 * set, we decide based on available memory per rank.
 *
 * @return int64_t Threshold (in bytes)
 */
int64_t get_shuffle_threshold();

/**
 * @brief Get the max allowed MPI tag value.
 * Ref:
 * https://stackoverflow.com/questions/61662466/can-the-tag-of-mpi-send-be-a-long-int,
 * https://www.intel.com/content/www/us/en/developer/articles/technical/large-mpi-tags-with-the-intel-mpi.html
 *
 * @return int
 */
int get_max_allowed_tag_value();

/**
 * @brief Find the next available starting message tag for sending concurrent
 * messages to the same rank.
 *
 * @param inflight_tags A set of tags currently in flight.
 * @return int. Starting tag for the async send. -1 if no available tag is
 * found. If a valid tag is returned, then the next 10,000 consecutive tags
 * should also be available.
 */
int get_next_available_tag(std::unordered_set<int>& inflight_tags);

/**
 * @brief Struct for Shuffle metrics.
 *
 */
class IncrementalShuffleMetrics {
   public:
    using stat_t = MetricBase::StatValue;
    using time_t = MetricBase::TimerValue;

    // Time spent appending to the shuffle buffer.
    time_t append_time = 0;
    // Time spent sending shuffle data.
    time_t shuffle_send_time = 0;
    // Time spent receiving shuffle data.
    time_t shuffle_recv_time = 0;
    // Time spent finalizing the shuffle send.
    time_t shuffle_send_finalization_time = 0;
    // Time spent finalizing the shuffle receive.
    time_t shuffle_recv_finalization_time = 0;
    // Total number of shuffles sent
    stat_t n_shuffle_send = 0;
    // Total number of shuffles received
    stat_t n_shuffle_recv = 0;
    // Time spent hashing rows for shuffle.
    time_t hash_time = 0;
    // Time spent unifying the dictionaries locally after the shuffle.
    time_t dict_unification_time = 0;
    // Total number of rows appended to the shuffle buffer.
    stat_t total_appended_nrows = 0;
    // Total number of rows sent to other ranks across all shuffles.
    stat_t total_sent_nrows = 0;
    // Total number of rows received from other ranks across all shuffles.
    stat_t total_recv_nrows = 0;
    // Approximate number of bytes sent to other ranks across all shuffles.
    stat_t total_approx_sent_size_bytes = 0;
    // Approximate number of bytes sent to other ranks across all shuffles for
    // dictionaries.
    stat_t approx_sent_size_bytes_dicts = 0;
    // Total number of bytes received from other ranks across all shuffles.
    stat_t total_recv_size_bytes = 0;
    // Total number of bytes received from other ranks across all shuffles for
    // dictionaries.
    stat_t approx_recv_size_bytes_dicts = 0;
    // Peak allocated size of the shuffle buffer.
    stat_t peak_capacity_bytes = 0;
    // Peak utilized size of the shuffle buffer.
    stat_t peak_utilization_bytes = 0;
    // Total number of times we reset the buffer since it grew too large.
    stat_t n_buffer_resets = 0;

    /**
     * @brief Helper function for exporting metrics during reporting steps in
     * Join/GroupBy.
     *
     * @param metrics Vector of metrics to append to.
     */
    void add_to_metrics(std::vector<MetricBase>& metrics);
};

/**
 * @brief Holds buffers and MPI requests for async shuffle sends for an array
 (MPI_Issend calls). Buffers cannot be freed until send is completed.
 *
 */
class AsyncShuffleSendState {
   public:
    /**
     * @brief Construct a new send state.
     *
     * @param starting_msg_tag Starting message tag to use for posting the
     * messages that send the data buffers.
     */
    explicit AsyncShuffleSendState(int starting_msg_tag_);

    /**
     * @brief Getter for starting_msg_tag.
     *
     * @return int
     */
    int get_starting_msg_tag() const { return this->starting_msg_tag; }

    /**
     * @brief Post async sends to shuffle a table to other ranks
     * @param in_table Input table
     * @param comm_info MPI communication information
     * @param shuffle_comm MPI communicator for shuffle
     */
    void send(const std::shared_ptr<table_info>& in_table,
              mpi_comm_info& comm_info, MPI_Comm shuffle_comm);

    /**
     * @brief Returns true if send is done which allows this state to be freed.
     *
     */
    bool sendDone() {
        int flag;
        CHECK_MPI_TEST_ALL(
            send_requests, flag,
            "[AsyncShuffleSendState::sendDone] MPI error on MPI_Testall: ")
        return flag;
    }

    /**
     * @brief Get total size of send buffers in flight
     *
     * @return int64_t total buffer sizes
     */
    int64_t GetTotalBufferSize() {
        int64_t total_size = 0;
        for (std::shared_ptr<array_info>& arr : send_arrs) {
            // Don't include dictionary or children in size
            // because they're included in send_arrs
            total_size += array_memory_size(arr, false, false);
        }
        return total_size;
    }

   private:
    using comm_infos_t =
        std::tuple<std::vector<mpi_comm_info>, std::vector<mpi_str_comm_info>>;
    // Arrays that are being sent, all from a single table.
    // Will be one array for simple types and more for nested/dictionary types.
    std::vector<std::shared_ptr<array_info>> send_arrs;

    std::vector<std::vector<uint64_t>> rank_to_metadata_vec;
    std::vector<MPI_Request> send_requests;
    // Starting message tag to use for posting the messages that send the data
    // buffers. This enables sending multiple tables to ranks (as long as the
    // sender, e.g. StreamSort, maintains sufficient state to not re-use tags
    // for concurrent sends).
    int starting_msg_tag = -1;

    /**
     * @brief Compute communication information for sending a table
     * @param in_table Input table
     * @param comm_info Base comminfo
     * @return comm_infos_t sub comm infos and string comm infos
     */
    comm_infos_t compute_comm_infos(const std::shared_ptr<table_info>& in_table,
                                    mpi_comm_info& comm_info);

    /**
     * @brief Send lengths for a table to each rank
     * @param in_table Input table
     * @param comm_info base comm info to get the lengths from
     * @param comm_infos sub comm infos and str comm infos to get the lengths
     * from
     * @param shuffle_comm MPI communicator for shuffle
     */
    void send_metadata(const std::shared_ptr<table_info>& in_table,
                       mpi_comm_info& comm_info, comm_infos_t& comm_infos,
                       MPI_Comm shuffle_comm);

    /**
     * @brief send array buffers for table to each rank
     * @param in_table Input table to send
     * @param comm_info base comm info to get the lengths  and offsets from
     * @param comm_infos sub comm infos and str comm infos to get the lengths
     * and offsets from
     * @param shuffle_comm MPI communicator for shuffle
     */
    void send_arrays(const std::shared_ptr<table_info>& in_table,
                     mpi_comm_info& comm_info, comm_infos_t& comm_infos,
                     MPI_Comm shuffle_comm);
    /**
     * @brief Add an array to the send state. This will allocate a send array,
     * fill it, add it to send_arrs, and return the send array info
     */
    std::shared_ptr<array_info> addArray(
        const std::shared_ptr<array_info>& _in_arr,
        const mpi_comm_info& comm_info, const mpi_str_comm_info& str_comm_info);

    /**
     * @brief send data to other ranks for shuffle
     * @tparam top_level whether this is a top level array
     * @param shuffle_comm MPI communicator for shuffle
     * @param comm_info MPI communication information
     * @param in_arr Input array
     * @param curr_tags Current tags
     * @param must_shuffle_to_rank Whether we must shuffle to the rank, this is
     * only true for ranks that are children of an array item that sent offsets
     * to this rank
     * @return AsyncShuffleSendState Send state
     */
    void send_shuffle_data_unknown_type(
        const MPI_Comm shuffle_comm, const mpi_comm_info& comm_info,
        comm_info_iter_t& comm_info_iter,
        str_comm_info_iter_t& str_comm_info_iter,
        const std::shared_ptr<array_info>& in_arr, std::vector<int>& curr_tags,
        std::vector<bool>& must_shuffle_to_rank);

    /**
     * Shuffle the null bitmask for an array
     */
    template <bodo_array_type::arr_type_enum arr_type>
    inline void send_shuffle_null_bitmask(
        const MPI_Comm shuffle_comm, const mpi_comm_info& comm_info,
        const std::shared_ptr<array_info>& send_arr,
        std::vector<int>& curr_tags, size_t p) {
        MPI_Datatype mpi_type_null = MPI_UNSIGNED_CHAR;
        const void* buf =
            send_arr->null_bitmask<arr_type>() +
            comm_info.send_disp_null[p] * numpy_item_size[Bodo_CTypes::UINT8];

        MPI_Request send_req_null;
        CHECK_MPI(
            MPI_Issend(buf, comm_info.send_count_null[p], mpi_type_null, p,
                       curr_tags[p]++, shuffle_comm, &send_req_null),
            "AsyncShuffleSendState::send_shuffle_null_bitmask: MPI error on "
            "MPI_Issend:");
        this->send_requests.push_back(send_req_null);
    }
    /**
     * Send the data1 buffer for a numpy array
     */
    template <bodo_array_type::arr_type_enum arr_type,
              Bodo_CTypes::CTypeEnum dtype>
        requires(arr_type == bodo_array_type::NUMPY)
    void send_shuffle_data(const MPI_Comm shuffle_comm, mpi_comm_info comm_info,
                           comm_info_iter_t& sub_comm_info_iter,
                           str_comm_info_iter_t& str_comm_info_iter,
                           const std::shared_ptr<array_info>& in_arr,
                           std::vector<int>& curr_tags,
                           std::vector<bool>& must_shuffle_to_rank) {
        const mpi_str_comm_info& str_comm_info = *(str_comm_info_iter++);
        std::shared_ptr<array_info> send_arr =
            this->addArray(in_arr, comm_info, str_comm_info);

        const MPI_Datatype mpi_type = get_MPI_typ<dtype>();
        for (size_t p = 0; p < static_cast<size_t>(comm_info.n_pes); p++) {
            // Skip ranks that don't have any data and don't need to be shuffled
            // to They need to be shuffled to if they are the child of an array
            // item that sent offsets to this rank
            if (comm_info.send_count[p] == 0 && !must_shuffle_to_rank[p]) {
                continue;
            }
            const void* buff =
                send_arr->template data1<arr_type>() +
                (numpy_item_size[dtype] * comm_info.send_disp[p]);
            MPI_Request send_req;
            CHECK_MPI(
                MPI_Issend(buff, comm_info.send_count[p], mpi_type, p,
                           curr_tags[p]++, shuffle_comm, &send_req),
                "AsyncShuffleSendState::send_shuffle_data[NUMPY]: MPI error on "
                "MPI_Issend:");
            this->send_requests.push_back(send_req);
        }
    }
    /**
     * Send the data1 buffer and null bitmask for a nullable array
     */
    template <bodo_array_type::arr_type_enum arr_type,
              Bodo_CTypes::CTypeEnum dtype>
        requires(arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
                 dtype != Bodo_CTypes::_BOOL)
    void send_shuffle_data(const MPI_Comm shuffle_comm, mpi_comm_info comm_info,
                           comm_info_iter_t& sub_comm_info_iter,
                           str_comm_info_iter_t& str_comm_info_iter,
                           const std::shared_ptr<array_info>& in_arr,
                           std::vector<int>& curr_tags,
                           std::vector<bool>& must_shuffle_to_rank) {
        const mpi_str_comm_info& str_comm_info = *(str_comm_info_iter++);
        std::shared_ptr<array_info> send_arr =
            this->addArray(in_arr, comm_info, str_comm_info);

        const MPI_Datatype mpi_type = get_MPI_typ<dtype>();
        for (size_t p = 0; p < static_cast<size_t>(comm_info.n_pes); p++) {
            // Skip ranks that don't have any data and don't need to be shuffled
            // to They need to be shuffled to if they are the child of an array
            // item that sent offsets to this rank
            if (comm_info.send_count[p] == 0 && !must_shuffle_to_rank[p]) {
                continue;
            }

            MPI_Request send_req;
            const void* buff =
                send_arr->template data1<arr_type>() +
                (numpy_item_size[dtype] * comm_info.send_disp[p]);
            CHECK_MPI(
                MPI_Issend(buff, comm_info.send_count[p], mpi_type, p,
                           curr_tags[p]++, shuffle_comm, &send_req),
                "AsyncShuffleSendState::send_shuffle_data[NULLABLE, !BOOL]: "
                "MPI error on MPI_Issend:");

            this->send_requests.push_back(send_req);
            this->send_shuffle_null_bitmask<arr_type>(shuffle_comm, comm_info,
                                                      send_arr, curr_tags, p);
        }
    }
    /**
     * Send the data1 buffer and null bitmask for a nullable boolean array.
     * Also sends the number of bits used in the last byte so the receiver knows
     * the array's length.
     */
    template <bodo_array_type::arr_type_enum arr_type,
              Bodo_CTypes::CTypeEnum dtype>
        requires(arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
                 dtype == Bodo_CTypes::_BOOL)
    void send_shuffle_data(const MPI_Comm shuffle_comm, mpi_comm_info comm_info,
                           comm_info_iter_t& sub_comm_info_iter,
                           str_comm_info_iter_t& str_comm_info_iter,
                           const std::shared_ptr<array_info>& in_arr,
                           std::vector<int>& curr_tags,
                           std::vector<bool>& must_shuffle_to_rank) {
        const mpi_str_comm_info& str_comm_info = *(str_comm_info_iter++);
        std::shared_ptr<array_info> send_arr =
            this->addArray(in_arr, comm_info, str_comm_info);
        const MPI_Datatype mpi_type = get_MPI_typ<dtype>();
        for (size_t p = 0; p < static_cast<size_t>(comm_info.n_pes); p++) {
            // Skip ranks that don't have any data and don't need to be shuffled
            // to They need to be shuffled to if they are the child of an array
            // item that sent offsets to this rank
            if (comm_info.send_count[p] == 0 && !must_shuffle_to_rank[p]) {
                continue;
            }

            // Send the data
            MPI_Request send_req;
            char* buff = send_arr->template data1<arr_type>() +
                         comm_info.send_disp_null[p];
            CHECK_MPI(
                MPI_Issend(buff, comm_info.send_count_null[p], mpi_type, p,
                           curr_tags[p]++, shuffle_comm, &send_req),
                "AsyncShuffleSendState::send_shuffle_data[NULLABLE][BOOL]: MPI "
                "error on MPI_Issend:");
            this->send_requests.push_back(send_req);

            this->send_shuffle_null_bitmask<arr_type>(shuffle_comm, comm_info,
                                                      send_arr, curr_tags, p);
        }
    }

    /**
     * Send the data1 buffer, data2 buffer, and null bitmask for a string array
     */
    template <bodo_array_type::arr_type_enum arr_type,
              Bodo_CTypes::CTypeEnum dtype>
        requires(arr_type == bodo_array_type::STRING)
    void send_shuffle_data(const MPI_Comm shuffle_comm, mpi_comm_info comm_info,
                           comm_info_iter_t& sub_comm_info_iter,
                           str_comm_info_iter_t& str_comm_info_iter,
                           const std::shared_ptr<array_info>& in_arr,
                           std::vector<int>& curr_tags,
                           std::vector<bool>& must_shuffle_to_rank) {
        const mpi_str_comm_info& str_comm_info = *(str_comm_info_iter++);
        std::shared_ptr<array_info> send_arr =
            this->addArray(in_arr, comm_info, str_comm_info);

        const MPI_Datatype data_mpi_type = MPI_UNSIGNED_CHAR;
        // Fill_send_array converts offsets to send lengths of type uint32_t
        const MPI_Datatype len_mpi_type = MPI_UINT32_T;
        for (int p = 0; p < comm_info.n_pes; p++) {
            // Skip ranks that don't have any data and don't need to be shuffled
            // to They need to be shuffled to if they are the child of an array
            // item that sent offsets to this rank
            if (comm_info.send_count[p] == 0 && !must_shuffle_to_rank[p]) {
                continue;
            }
            MPI_Request data_send_req;
            const void* data_buff = send_arr->template data1<arr_type>() +
                                    str_comm_info.send_disp_sub[p];
            CHECK_MPI(
                MPI_Issend(data_buff, str_comm_info.send_count_sub[p],
                           data_mpi_type, p, curr_tags[p]++, shuffle_comm,
                           &data_send_req),
                "AsyncShuffleSendState::send_shuffle_data[STRING]: MPI error "
                "on Issend:");
            this->send_requests.push_back(data_send_req);

            MPI_Request len_send_req;
            const void* len_buff = send_arr->data2<arr_type>() +
                                   (sizeof(uint32_t) * comm_info.send_disp[p]);

            CHECK_MPI(
                MPI_Issend(len_buff, comm_info.send_count[p], len_mpi_type, p,
                           curr_tags[p]++, shuffle_comm, &len_send_req),
                "AsyncShuffleSendState::send_shuffle_data[STRING]: MPI error "
                "on MPI_Issend:");
            this->send_requests.push_back(len_send_req);

            this->send_shuffle_null_bitmask<arr_type>(shuffle_comm, comm_info,
                                                      send_arr, curr_tags, p);
        }
    }

    /**
     * Send the child offsets and dictionary for a dictionary array
     * The dictionary is sent as a string array to all ranks so the offests
     * remain valid
     */
    template <bodo_array_type::arr_type_enum arr_type,
              Bodo_CTypes::CTypeEnum dtype>
        requires(arr_type == bodo_array_type::DICT)
    void send_shuffle_data(const MPI_Comm shuffle_comm, mpi_comm_info comm_info,
                           comm_info_iter_t& sub_comm_info_iter,
                           str_comm_info_iter_t& str_comm_info_iter,
                           const std::shared_ptr<array_info>& in_arr,
                           std::vector<int>& curr_tags,
                           std::vector<bool>& must_shuffle_to_rank) {
        str_comm_info_iter++;
        // Shuffle the indices
        this->send_shuffle_data<bodo_array_type::NULLABLE_INT_BOOL,
                                Bodo_CTypes::INT32>(
            shuffle_comm, comm_info, sub_comm_info_iter, str_comm_info_iter,
            in_arr->child_arrays[1], curr_tags, must_shuffle_to_rank);
        // Copy the dict array because the dictionary comes from a builder.
        // Builders can be appended to which can cause resizes invalidating the
        // pointer.
        std::shared_ptr<array_info> dict_arr =
            copy_array(in_arr->child_arrays[0]);
        const MPI_Datatype data_mpi_type = MPI_UNSIGNED_CHAR;
        const MPI_Datatype offset_mpi_type = MPI_UINT64_T;
        // Send the whole dict to each rank
        for (int p = 0; p < comm_info.n_pes; p++) {
            // Skip ranks that don't have any data and don't need to be shuffled
            // to They need to be shuffled to if they are the child of an array
            // item that sent offsets to this rank
            if (comm_info.send_count[p] == 0 && !must_shuffle_to_rank[p]) {
                continue;
            }
            MPI_Request data_send_req;
            const void* data_buff = dict_arr->data1<bodo_array_type::STRING>();
            CHECK_MPI(
                MPI_Issend(data_buff, dict_arr->n_sub_elems(), data_mpi_type, p,
                           curr_tags[p]++, shuffle_comm, &data_send_req),
                "AsyncShuffleSendState::send_shuffle_data[DICT]: MPI error on "
                "MPI_Issend:");
            this->send_requests.push_back(data_send_req);

            MPI_Request offset_send_req;
            const void* offset_buff =
                dict_arr->data2<bodo_array_type::STRING>();
            CHECK_MPI(
                MPI_Issend(offset_buff, dict_arr->length + 1, offset_mpi_type,
                           p, curr_tags[p]++, shuffle_comm, &offset_send_req),
                "AsyncShuffleSendState::send_shuffle_data[DICT]: MPI error on "
                "MPI_Issend:");
            this->send_requests.push_back(offset_send_req);

            MPI_Request null_send_req;
            const void* null_buff =
                dict_arr->null_bitmask<bodo_array_type::STRING>();
            CHECK_MPI(
                MPI_Issend(null_buff,
                           arrow::bit_util::BytesForBits(dict_arr->length),
                           MPI_UNSIGNED_CHAR, p, curr_tags[p]++, shuffle_comm,
                           &null_send_req),
                "AsyncShuffleSendState::send_shuffle_data[DICT] MPI error on "
                "MPI_Issend:");
            this->send_requests.push_back(null_send_req);
        }
        this->send_arrs.push_back(dict_arr);
    }

    /**
     * Send the data1 buffer, null bitmask, and child array for a list array
     */
    template <bodo_array_type::arr_type_enum arr_type,
              Bodo_CTypes::CTypeEnum dtype>
        requires(arr_type == bodo_array_type::ARRAY_ITEM)
    void send_shuffle_data(const MPI_Comm shuffle_comm, mpi_comm_info comm_info,
                           comm_info_iter_t& sub_comm_info_iter,
                           str_comm_info_iter_t& str_comm_info_iter,
                           const std::shared_ptr<array_info>& in_arr,
                           std::vector<int>& curr_tags,
                           std::vector<bool>& must_shuffle_to_rank) {
        const mpi_str_comm_info& str_comm_info = *(str_comm_info_iter++);
        std::shared_ptr<array_info> send_arr =
            this->addArray(in_arr, comm_info, str_comm_info);
        const MPI_Datatype lens_mpi_type = MPI_UINT32_T;
        std::vector<bool> must_shuffle_to_rank_inner(comm_info.n_pes, false);

        for (size_t p = 0; p < static_cast<size_t>(comm_info.n_pes); p++) {
            // Skip ranks that don't have any data and don't need to be shuffled
            // to They need to be shuffled to if they are the child of an array
            // item that sent offsets to this rank
            if (comm_info.send_count[p] == 0 && !must_shuffle_to_rank[p]) {
                continue;
            }
            must_shuffle_to_rank_inner[p] = true;

            // Shuffle the lengths
            MPI_Request send_req;
            const void* buff = send_arr->data1<arr_type>() +
                               (sizeof(uint32_t) * comm_info.send_disp[p]);
            CHECK_MPI(
                MPI_Issend(buff, comm_info.send_count[p], lens_mpi_type, p,
                           curr_tags[p]++, shuffle_comm, &send_req),
                "AsyncShuffleSendState::send_shuffle_data[ARRAY_ITEM]: MPI "
                "error on MPI_Issend:");

            // Shuffle the null bitmask
            this->send_requests.push_back(send_req);
            this->send_shuffle_null_bitmask<arr_type>(shuffle_comm, comm_info,
                                                      send_arr, curr_tags, p);
        }
        mpi_comm_info sub_comm_info = *sub_comm_info_iter++;
        this->send_shuffle_data_unknown_type(
            shuffle_comm, sub_comm_info, sub_comm_info_iter, str_comm_info_iter,
            in_arr->child_arrays[0], curr_tags, must_shuffle_to_rank_inner);
    }
    /**
     * Send the data1 buffer, data2 buffer, and null bitmask for a timestamptz
     * array
     */
    template <bodo_array_type::arr_type_enum arr_type,
              Bodo_CTypes::CTypeEnum dtype>
        requires(arr_type == bodo_array_type::TIMESTAMPTZ)
    void send_shuffle_data(const MPI_Comm shuffle_comm, mpi_comm_info comm_info,
                           comm_info_iter_t& sub_comm_info_iter,
                           str_comm_info_iter_t& str_comm_info_iter,
                           const std::shared_ptr<array_info>& in_arr,
                           std::vector<int>& curr_tags,
                           std::vector<bool>& must_shuffle_to_rank) {
        const mpi_str_comm_info& str_comm_info = *(str_comm_info_iter++);
        std::shared_ptr<array_info> send_arr =
            this->addArray(in_arr, comm_info, str_comm_info);

        constexpr Bodo_CTypes::CTypeEnum offset_type = Bodo_CTypes::INT16;
        const MPI_Datatype mpi_datetime_type =
            get_MPI_typ<Bodo_CTypes::TIMESTAMPTZ>();
        const MPI_Datatype mpi_tz_offset_type = get_MPI_typ<offset_type>();
        for (size_t p = 0; p < static_cast<size_t>(comm_info.n_pes); p++) {
            // Skip ranks that don't have any data and don't need to be shuffled
            // to They need to be shuffled to if they are the child of an array
            // item that sent offsets to this rank
            if (comm_info.send_count[p] == 0 && !must_shuffle_to_rank[p]) {
                continue;
            }

            MPI_Request datetime_send_req;
            const void* buff =
                send_arr->data1<arr_type>() +
                (numpy_item_size[dtype] * comm_info.send_disp[p]);
            CHECK_MPI(
                MPI_Issend(buff, comm_info.send_count[p], mpi_datetime_type, p,
                           curr_tags[p]++, shuffle_comm, &datetime_send_req),
                "AsyncShuffleSendState::send_shuffle_data[TIMESTAMPTZ]: MPI "
                "error on MPI_Issend:");
            this->send_requests.push_back(datetime_send_req);

            MPI_Request tz_offset_send_req;
            buff = send_arr->data2<arr_type>() +
                   (numpy_item_size[offset_type] * comm_info.send_disp[p]);
            CHECK_MPI(
                MPI_Issend(buff, comm_info.send_count[p], mpi_tz_offset_type, p,
                           curr_tags[p]++, shuffle_comm, &tz_offset_send_req),
                "AsyncShuffleSendState::send_shuffle_data[TIMESTAMPTZ]: MPI "
                "error on MPI_Issend:");
            this->send_requests.push_back(tz_offset_send_req);

            this->send_shuffle_null_bitmask<arr_type>(shuffle_comm, comm_info,
                                                      send_arr, curr_tags, p);
        }
    }
    /**
     * Send nested arrays and null bitmask for a struct array
     */
    template <bodo_array_type::arr_type_enum arr_type,
              Bodo_CTypes::CTypeEnum dtype>
        requires(arr_type == bodo_array_type::STRUCT)
    void send_shuffle_data(const MPI_Comm shuffle_comm, mpi_comm_info comm_info,
                           comm_info_iter_t& sub_comm_info_iter,
                           str_comm_info_iter_t& str_comm_info_iter,
                           const std::shared_ptr<array_info>& in_arr,
                           std::vector<int>& curr_tags,
                           std::vector<bool>& must_shuffle_to_rank) {
        const mpi_str_comm_info& str_comm_info = *(str_comm_info_iter++);
        std::shared_ptr<array_info> send_arr =
            this->addArray(in_arr, comm_info, str_comm_info);

        for (size_t i = 0; i < in_arr->child_arrays.size(); i++) {
            this->send_shuffle_data_unknown_type(
                shuffle_comm, comm_info, sub_comm_info_iter, str_comm_info_iter,
                in_arr->child_arrays[i], curr_tags, must_shuffle_to_rank);
        }

        for (size_t p = 0; p < static_cast<size_t>(comm_info.n_pes); p++) {
            // Skip ranks that don't have any data and don't need to be shuffled
            // to They need to be shuffled to if they are the child of an array
            // item that sent offsets to this rank
            if (comm_info.send_count[p] == 0 && !must_shuffle_to_rank[p]) {
                continue;
            }
            this->send_shuffle_null_bitmask<arr_type>(shuffle_comm, comm_info,
                                                      send_arr, curr_tags, p);
        }
    }
    /**
     * Send nested array(item) array for a map array
     */
    template <bodo_array_type::arr_type_enum arr_type,
              Bodo_CTypes::CTypeEnum dtype>
        requires(arr_type == bodo_array_type::MAP)
    void send_shuffle_data(const MPI_Comm shuffle_comm, mpi_comm_info comm_info,
                           comm_info_iter_t& sub_comm_info_iter,
                           str_comm_info_iter_t& str_comm_info_iter,
                           const std::shared_ptr<array_info>& in_arr,
                           std::vector<int>& curr_tags,
                           std::vector<bool>& must_shuffle_to_rank) {
        str_comm_info_iter++;
        // Map arrays are just wrappers around array item arrays of structs
        return send_shuffle_data<bodo_array_type::ARRAY_ITEM,
                                 Bodo_CTypes::LIST>(
            shuffle_comm, comm_info, sub_comm_info_iter, str_comm_info_iter,
            in_arr->child_arrays[0], curr_tags, must_shuffle_to_rank);
    }
};

/**
 * @brief Holds buffers and MPI requests for async shuffle recvs for a table
 (MPI_Irecv calls). Buffers cannot be used or freed until recvs are completed.
 *
 */
class AsyncShuffleRecvState {
   public:
    std::vector<std::shared_ptr<array_info>> out_arrs;
    std::vector<MPI_Request> recv_requests;
    int source;
    std::unique_ptr<bodo::Schema> schema;

    AsyncShuffleRecvState(int source_, std::unique_ptr<bodo::Schema> schema_)
        : source(source_), schema(std::move(schema_)) {}

    /**
     * @brief Returns a tuple of a boolean and a shared_ptr to a table_info. If
     * recvs of all arrays are done, the boolean will be true, and the table
     * will be non-null, otherwise the return value will be (false, NULL).
     * When the boolean is true this state can be freed.
     * @param dict_builders dictionary builders to use for assembling arrays.
     * @param[out] metrics metrics object to track stats about recv.
     */
    std::pair<bool, std::shared_ptr<table_info>> recvDone(
        const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders,
        MPI_Comm shuffle_comm, IncrementalShuffleMetrics& metrics);

    /**
     * @brief Post MPI_Imrecv call for metadata (starting msg tag and lengths of
     * incoming arrays)
     * @param status MPI status object for probe of length message
     * @param m MPI_Message to recv length from
     */
    void PostMetadataRecv(MPI_Status& status, MPI_Message& m);

    /**
     * @brief test if the request posted by PostMetadataRecv is complete and if
     * so, get the metadata and post receives for the data arrays.
     *
     * @param shuffle_comm MPI communicator for shuffle
     */
    void TryRecvMetadataAndAllocArrs(MPI_Comm& shuffle_comm);

    /**
     * @brief Get total memory size of recv buffers in flight
     *
     * @return int64_t total buffer sizes
     */
    int64_t GetTotalBufferSize() {
        int64_t total_size = 0;
        for (std::shared_ptr<array_info>& arr : out_arrs) {
            total_size += array_memory_size(arr, false);
        }
        return total_size;
    }

   private:
    /**
     * @brief Return the metadata for the incoming arrays. The metadata consists
     * of the starting MPI tag to post receives on as well as the lengths of the
     * buffers.
     * @param shuffle_comm MPI communicator for shuffle
     * note that this function should only be called once
     * since it clears the underlying buffer
     */
    std::optional<std::vector<uint64_t>> GetRecvMetadata(MPI_Comm shuffle_comm);

    std::shared_ptr<array_info> finalize_receive_array(
        const std::shared_ptr<array_info>& arr,
        const std::shared_ptr<DictionaryBuilder>& dict_builder,
        std::vector<uint32_t>& data_lens_vec,
        IncrementalShuffleMetrics& metrics);
    MPI_Request metadata_request = MPI_REQUEST_NULL;
    std::vector<uint64_t> metadata_vec;
};

/**
 * @brief If any recieve states are complete, erase them from the input and
 * append their out tables to out_builder
 */
void consume_completed_recvs(
    std::vector<AsyncShuffleRecvState>& recv_states, MPI_Comm shuffle_comm,
    const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders,
    IncrementalShuffleMetrics& metrics, TableBuildBuffer& out_builder);

/**
 * @brief Common hash-shuffle functionality for streaming operators such as
 * HashJoin and Groupby.
 *
 */
class IncrementalShuffleState {
   public:
    /// @brief Schema of the shuffle table
    const std::shared_ptr<bodo::Schema> schema;
    /// @brief Dictionary builder for the dictionary-encoded columns. Note that
    /// these are only for top-level dictionaries and not for dictionary-encoded
    /// fields within nested data types.
    std::vector<std::shared_ptr<DictionaryBuilder>> dict_builders;
    /// @brief Shuffle data buffer.
    std::unique_ptr<TableBuildBuffer> table_buffer;

    MPI_Comm shuffle_comm;

    /**
     * @brief Constructor for new IncrementalShuffleState
     *
     * @param schema_ Schema of the shuffle table
     * @param dict_builders_ Dictionary builders for the top level columns.
     * @param n_keys_ Number of key columns (to shuffle based off of).
     * @param curr_iter_ Reference to the iteration counter from parent
     * operator. e.g. In Groupby, this is 'build_iter'. For HashJoin, this could
     * be either 'build_iter' or 'probe_iter' based on whether it's the
     * build_shuffle_state or probe_shuffle_state, respectively.
     * @param sync_freq_ Reference to the synchronization frequency variable of
     * the parent state. This will be modified by this state adaptively (if
     * enabled).
     */
    IncrementalShuffleState(
        std::shared_ptr<bodo::Schema> schema_,
        const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders_,
        const uint64_t n_keys_, const uint64_t& curr_iter_, int64_t& sync_freq_,
        int64_t parent_op_id_);
    /**
     * @brief Constructor for new IncrementalShuffleState
     *
     * @param arr_c_types_ Array types of the shuffle table.
     * @param arr_array_types_ CTypes of the shuffle table.
     * @param dict_builders_ Dictionary builders for the top level columns.
     * @param n_keys_ Number of key columns (to shuffle based off of).
     * @param curr_iter_ Reference to the iteration counter from parent
     * operator. e.g. In Groupby, this is 'build_iter'. For HashJoin, this could
     * be either 'build_iter' or 'probe_iter' based on whether it's the
     * build_shuffle_state or probe_shuffle_state, respectively.
     * @param sync_freq_ Reference to the synchronization frequency variable of
     * the parent state. This will be modified by this state adaptively (if
     * enabled).
     */
    IncrementalShuffleState(
        const std::vector<int8_t>& arr_c_types_,
        const std::vector<int8_t>& arr_array_types_,
        const std::vector<std::shared_ptr<DictionaryBuilder>>& dict_builders_,
        const uint64_t n_keys_, const uint64_t& curr_iter_, int64_t& sync_freq_,
        int64_t parent_op_id_);

    /**
     * @brief Calculate initial synchronization frequency if syncing
     * adaptively.
     * This must be called in the first iteration. It estimates how many
     * iterations it will take to for the shuffle buffer size of any rank to be
     * larger than this->shuffle_threshold based on the size of the first input
     * batch. 'sync_freq' will be modified accordingly.
     *
     * @param sample_in_table_batch Input batch to use for estimating the
     * initial sync frequency.
     * @param is_parallel Parallel flag
     */
    void Initialize(const std::shared_ptr<table_info>& sample_in_table_batch,
                    bool is_parallel, MPI_Comm _shuffle_comm);

    /**
     * @brief Append a build batch to the shuffle table. Only the rows specified
     * in append_rows will be appended. Note that we will reserve enough space
     * to append the entire input batch.
     *
     * @param in_table Input batch to append rows from.
     * @param append_rows Bitmask specifying the rows to append.
     */
    void AppendBatch(const std::shared_ptr<table_info>& in_table,
                     const std::vector<bool>& append_rows);

    /**
     * @brief Shuffle the data in the shuffle-table if we're at a
     * synchronization iteration and the shuffle table is larger than 50MiB on
     * some rank, or if this is the last iteration. It will also update the
     * synchronization frequency if we're at an update-sync-freq iteration.
     * This must be called in every iteration, but only if shuffle is possible
     * (i.e. data is distributed and requires shuffling). If no shuffle was
     * done, we will not return anything. If we do shuffle, we will return the
     * output shuffle table. The DICT columns in the output will have
     * dictionaries unified with the dict-builders. We will also reset the
     * shuffle buffer after every shuffle.
     *
     * @param is_last Is the last iteration.
     * @return std::optional<std::shared_ptr<table_info>> Output shuffle table
     * if we did a shuffle.
     */
    std::optional<std::shared_ptr<table_info>> ShuffleIfRequired(
        const bool is_last);

    bool SendRecvEmpty();

    /**
     * @brief Finalize the shuffle state. This will free the shuffle buffer and
     * release memory. It will also release its references to the dictionary
     * builders.
     *
     */
    virtual void Finalize();

    /**
     * @brief Export shuffle metrics into the provided vector.
     *
     * @param[in, out] metrics Vector to append the metrics to.
     */
    virtual void ExportMetrics(std::vector<MetricBase>& metrics) {
        this->metrics.add_to_metrics(metrics);
    }

    /**
     * @brief Reset the metrics.
     *
     */
    virtual void ResetMetrics() { this->metrics = IncrementalShuffleMetrics(); }

    /**
     * @brief Return true if send/recv buffer sizes are over a threshold and
     this rank shouldn't get more input data to avoid potential OOM.
     */
    bool BuffersFull();

    /**
     * @brief Determine if our shuffle buffers exceed the shuffle size after
     * doing any operator specific processing to reduce the shuffle data. The
     * base class just checks the shuffle size based on the current shuffle
     * buffer size, but subclasses my introduce additional compute to check if
     * we can reduce the amount of data we are shuffling.
     * @param is_last Have we reached the last iteration and therefore must
     * shuffle if there is any data.
     * @return If our buffers exceed the shuffle size threshold after
     * processing.
     */
    virtual bool ShouldShuffleAfterProcessing(bool is_last) {
        return (is_last && (this->table_buffer->data_table->nrows() > 0)) ||
               (table_local_memory_size(this->table_buffer->data_table,
                                        /*include_dict_size*/ false) >=
                this->shuffle_threshold);
    }

   protected:
    // Keep track of inflight tags to avoid tag collisions. See:
    // https://github.com/bodo-ai/Bodo/blob/3d5621629e95486bbc9bd4e6b45f85f22835f515/bodo/libs/streaming/_sort.cpp#L1633
    // For more details.
    std::unordered_set<int> inflight_tags;

    /**
     * @brief Helper function for ShuffleIfRequired. In this base class,
     * this simply returns the shuffle-table and its hashes.
     * Child classes can modify this. e.g. Groupby may do a drop-duplicates on
     * the shuffle buffer in the nunique-only case.
     *
     * @return std::tuple<
     * std::shared_ptr<table_info>,
     * std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>,
     * std::shared_ptr<uint32_t[]>,
     * std::unique_ptr<uint8_t[]> shuffle_table, dict_hashes, shuffle_hashes
     * keep_row_bitmap. Only keep_row_bitmap is optional may be nullptr.
     */
    virtual std::tuple<
        /*shuffle_table*/ std::shared_ptr<table_info>,
        /*dict_hashes*/
        std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>,
        /*shuffle_hashes*/ std::shared_ptr<uint32_t[]>,
        /*keep_row_bitmap*/ std::unique_ptr<uint8_t[]>>
    GetShuffleTableAndHashes();

    /**
     * @brief Helper function for ShuffleIfRequired. This is called after every
     * shuffle. This resets the shuffle table buffer (which only resets the
     * size, without releasing any memory).
     *
     */
    virtual void ResetAfterShuffle();

    /**
     * @brief API to report the input and shuffle batch size at every iteration.
     * This is meant to be called either during AppendBatch or in
     * `GroupbyState::UpdateShuffleGroupsAndCombine`.
     *
     * @param batch_size Size of the input batch
     * @param shuffle_batch_size Size of the shuffle batch (out of the input
     * batch)
     */
    inline void UpdateAppendBatchSize(uint64_t batch_size,
                                      uint64_t shuffle_batch_size) {
        this->max_input_batch_size_since_prev_shuffle =
            std::max(this->max_input_batch_size_since_prev_shuffle, batch_size);
        this->max_shuffle_batch_size_since_prev_shuffle =
            std::max(this->max_shuffle_batch_size_since_prev_shuffle,
                     shuffle_batch_size);
    }

    /// @brief Number of shuffle keys.
    const uint64_t n_keys;
    /// @brief Threshold to use to decide when we should shuffle.
    const int64_t shuffle_threshold = DEFAULT_SHUFFLE_THRESHOLD;

   private:
    /// @brief Reference to the iteration counter of the parent operator /
    /// operator-stage.
    const uint64_t& curr_iter;
    /// @brief The iteration number of last shuffle (used for adaptive sync
    /// estimation)
    uint64_t prev_shuffle_iter = 0;
    /// @brief Max input batch size seen since the previous shuffle.
    uint64_t max_input_batch_size_since_prev_shuffle = 0;
    /// @brief Max shuffle batch size seen since the previous shuffle.
    uint64_t max_shuffle_batch_size_since_prev_shuffle = 0;
    /// @brief Number of ranks.
    int n_pes;
    /// @brief Estimated size of a row based on just the dtypes.
    const size_t row_bytes_guesstimate;
    /// @brief Operator ID of the parent operator. Used for debug prints.
    const int64_t parent_op_id = -1;
    /// @brief Number of syncs after which we should re-evaluate the sync
    /// frequency.
    int64_t sync_update_freq = DEFAULT_SYNC_UPDATE_FREQ;
    /// @brief Print information about the shuffle state during initialization,
    /// during every shuffle and after sync frequency is updated.
    bool debug_mode = false;
    /// @brief Metrics for query profile.
    IncrementalShuffleMetrics metrics;

    std::vector<AsyncShuffleSendState> send_states;
    std::vector<AsyncShuffleRecvState> recv_states;

    /**
     * @brief Helper function to get the dictionary hashes for key columns.
     *
     * @return
     * std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>
     */
    std::shared_ptr<bodo::vector<std::shared_ptr<bodo::vector<uint32_t>>>>
    get_dict_hashes_for_keys();

    /**
     * @brief Helper function to unify the dictionaries of the shuffle output
     * with the dictionaries in dictionary-builders.
     *
     * @param in_table Input table.
     * @return std::shared_ptr<table_info> Table with unified dictionaries.
     */
    std::shared_ptr<table_info> unify_table_dicts(
        const std::shared_ptr<table_info>& in_table);
};

/**
 * @brief Starts shuffle communication for input table and fills send states
 with send arrays and MPI requests. Send states have to be kept around until
 communication is done but other input can be deallocated. See
 https://bodo.atlassian.net/wiki/spaces/B/pages/1716617219/Async+Streaming+Loop+and+Shuffle+Design
 *
 * @param in_table input table to shuffle
 * @param hashes partition hashes for choosing target ranks
 * @param shuffle_comm MPI communicator for shuffle (each operator has to have
 its own)
 * @param starting_msg_tag Starting message tag to use for posting the messages
 that send the data buffers.
 */
AsyncShuffleSendState shuffle_issend(
    std::shared_ptr<table_info> in_table,
    const std::shared_ptr<uint32_t[]>& hashes, const uint8_t* keep_row_bitmask,
    MPI_Comm shuffle_comm,
    int starting_msg_tag = (SHUFFLE_METADATA_MSG_TAG + 1));

/**
 * @brief Checks for incoming shuffle messages using MPI probe and fills recieve
 states. The received data isn't ready to use consume until receive is fully
 done (see recvDone). See
 https://bodo.atlassian.net/wiki/spaces/B/pages/1716617219/Async+Streaming+Loop+and+Shuffle+Design
 *
 * @param in_table table to use for receive data's schema
 * @param shuffle_comm MPI communicator for shuffle (each operator has to have
 its own)
 * @param recv_states vector of recieve states to fill
 * @param max_recv_states maximum length recv_states is allowed to be before
 * returning (default is effectively unbounded)
 */
void shuffle_irecv(std::shared_ptr<table_info> in_table, MPI_Comm shuffle_comm,
                   std::vector<AsyncShuffleRecvState>& recv_states,
                   size_t max_recv_states = std::numeric_limits<size_t>::max());

/**
 * @brief receive data from other ranks for shuffle
 * @tparam top_level whether this is a top level array
 * @param data_type DataType of the array
 * @param shuffle_comm MPI communicator for shuffle
 * @param source Source rank
 * @param curr_tag Current tag
 * @param recv_state AsyncShuffleRecvState
 * @param metrics IncrementalShuffleMetrics to update.
 * @return std::unique_ptr<array_info> Received array
 */
template <bool top_level>
std::unique_ptr<array_info> recv_shuffle_data_unknown_type(
    const std::unique_ptr<bodo::DataType>& data_type,
    const MPI_Comm shuffle_comm, const int source, int& curr_tag,
    AsyncShuffleRecvState& recv_state, len_iter_t& lens_iter);

/**
 * Receive the null bitmask for an array
 */
template <bodo_array_type::arr_type_enum arr_type>
void recv_null_bitmask(std::unique_ptr<array_info>& out_arr,
                       const MPI_Comm shuffle_comm, const int source,
                       int& curr_tag, AsyncShuffleRecvState& recv_state) {
    const MPI_Datatype mpi_type_null = MPI_UNSIGNED_CHAR;
    int recv_size_null = arrow::bit_util::BytesForBits(out_arr->length);
    MPI_Request recv_req_null;
    CHECK_MPI(MPI_Irecv(out_arr->null_bitmask<arr_type>(), recv_size_null,
                        mpi_type_null, source, curr_tag++, shuffle_comm,
                        &recv_req_null),
              "recv_null_bitmask: MPI error on MPI_Irecv:");
    recv_state.recv_requests.push_back(recv_req_null);
}

/**
 * Receive the data1 buffer for a numpy array
 */
template <bodo_array_type::arr_type_enum arr_type, Bodo_CTypes::CTypeEnum dtype,
          bool top_level>
    requires(arr_type == bodo_array_type::NUMPY)
std::unique_ptr<array_info> recv_shuffle_data(
    const std::unique_ptr<bodo::DataType>& data_type,
    const MPI_Comm shuffle_comm, const int source, int& curr_tag,
    AsyncShuffleRecvState& recv_state, len_iter_t& lens_iter) {
    const MPI_Datatype mpi_type = get_MPI_typ<dtype>();

    uint64_t arr_len = *lens_iter++;

    std::unique_ptr<array_info> out_arr = alloc_array_top_level<arr_type>(
        arr_len, 0, 0, arr_type, dtype, -1, 0, 0);
    MPI_Request recv_req;
    CHECK_MPI(MPI_Irecv(out_arr->data1<arr_type>(), arr_len, mpi_type, source,
                        curr_tag++, shuffle_comm, &recv_req),
              "recv_shuffle_data[NUMPY]: MPI error on MPI_Irecv:");
    recv_state.recv_requests.push_back(recv_req);
    return out_arr;
}

/**
 * Receive the data1 buffer and null bitmask for a nullable array
 */
template <bodo_array_type::arr_type_enum arr_type, Bodo_CTypes::CTypeEnum dtype,
          bool top_level>
    requires(arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
             dtype != Bodo_CTypes::_BOOL)
std::unique_ptr<array_info> recv_shuffle_data(
    const std::unique_ptr<bodo::DataType>& data_type,
    const MPI_Comm shuffle_comm, const int source, int& curr_tag,
    AsyncShuffleRecvState& recv_state, len_iter_t& lens_iter) {
    const MPI_Datatype mpi_type = get_MPI_typ<dtype>();
    uint64_t arr_len = *lens_iter++;

    std::unique_ptr<array_info> out_arr = alloc_array_top_level<arr_type>(
        arr_len, 0, 0, arr_type, dtype, -1, 0, 0);

    MPI_Request recv_req;
    CHECK_MPI(MPI_Irecv(out_arr->data1<arr_type>(), arr_len, mpi_type, source,
                        curr_tag++, shuffle_comm, &recv_req),
              "recv_shuffle_data[NULLABLE][!BOOL]: MPI error on MPI_Irecv:");
    recv_state.recv_requests.push_back(recv_req);

    recv_null_bitmask<arr_type>(out_arr, shuffle_comm, source, curr_tag,
                                recv_state);
    return out_arr;
}

/**
 * Receive the data1 buffer and null bitmask for a nullable boolean array.
 * Also receives the number of bits used in the last byte so we can adjust
 * the array's length.
 */
template <bodo_array_type::arr_type_enum arr_type, Bodo_CTypes::CTypeEnum dtype,
          bool top_level>
    requires(arr_type == bodo_array_type::NULLABLE_INT_BOOL &&
             dtype == Bodo_CTypes::_BOOL)
std::unique_ptr<array_info> recv_shuffle_data(
    const std::unique_ptr<bodo::DataType>& data_type,
    const MPI_Comm shuffle_comm, const int source, int& curr_tag,
    AsyncShuffleRecvState& recv_state, len_iter_t& lens_iter) {
    const MPI_Datatype mpi_type = get_MPI_typ<dtype>();
    uint64_t arr_len = *lens_iter++;

    std::unique_ptr<array_info> out_arr = alloc_array_top_level<arr_type>(
        arr_len, 0, 0, arr_type, dtype, -1, 0, 0);

    MPI_Request recv_req;
    CHECK_MPI(MPI_Irecv(out_arr->data1<arr_type>(),
                        arrow::bit_util::BytesForBits(arr_len), mpi_type,
                        source, curr_tag++, shuffle_comm, &recv_req),
              "recv_shuffle_data[NULLABLE][BOOL]: MPI error on MPI_Irecv:");
    recv_state.recv_requests.push_back(recv_req);

    recv_null_bitmask<arr_type>(out_arr, shuffle_comm, source, curr_tag,
                                recv_state);
    return out_arr;
}

/**
 * Receive the data1 buffer, data2 buffer, and null bitmask for a string array
 */
template <bodo_array_type::arr_type_enum arr_type, Bodo_CTypes::CTypeEnum dtype,
          bool top_level>
    requires(arr_type == bodo_array_type::STRING)
std::unique_ptr<array_info> recv_shuffle_data(
    const std::unique_ptr<bodo::DataType>& data_type,
    const MPI_Comm shuffle_comm, const int source, int& curr_tag,
    AsyncShuffleRecvState& recv_state, len_iter_t& lens_iter) {
    const MPI_Datatype data_mpi_type = MPI_UNSIGNED_CHAR;
    // Fill_send_array converts offsets to lengths of type uint32_t
    const MPI_Datatype len_mpi_type = MPI_UINT32_T;

    uint64_t arr_len = *lens_iter++;
    uint64_t n_chars = *lens_iter++;

    std::unique_ptr<array_info> out_arr = alloc_array_top_level<arr_type>(
        arr_len, n_chars, 0, arr_type, dtype, -1, 0, 0);

    MPI_Request data_req;
    CHECK_MPI(MPI_Irecv(out_arr->data1<arr_type>(), n_chars, data_mpi_type,
                        source, curr_tag++, shuffle_comm, &data_req),
              "recv_shuffle_data[STRING]: MPI error on MPI_Irecv:");
    recv_state.recv_requests.push_back(data_req);

    MPI_Request len_req;
    // Receive the lens, we know we can fit them in the offset buffer because
    // sizeof(offset_t) >= sizeof(uint32_t)
    CHECK_MPI(
        MPI_Irecv(out_arr->data2<arr_type, offset_t>(), arr_len, len_mpi_type,
                  source, curr_tag++, shuffle_comm, &len_req),
        "recv_shuffle_data[STRING]: MPI error on MPI_Irecv:");
    recv_state.recv_requests.push_back(len_req);

    recv_null_bitmask<arr_type>(out_arr, shuffle_comm, source, curr_tag,
                                recv_state);
    return out_arr;
}

/**
 * Receive an offsets array and dictionary for a dictionary array
 */
template <bodo_array_type::arr_type_enum arr_type, Bodo_CTypes::CTypeEnum dtype,
          bool top_level>
    requires(arr_type == bodo_array_type::DICT)
std::unique_ptr<array_info> recv_shuffle_data(
    const std::unique_ptr<bodo::DataType>& data_type,
    const MPI_Comm shuffle_comm, const int source, int& curr_tag,
    AsyncShuffleRecvState& recv_state, len_iter_t& lens_iter) {
    // Receive the indices
    std::unique_ptr<array_info> indices =
        recv_shuffle_data<bodo_array_type::NULLABLE_INT_BOOL,
                          Bodo_CTypes::INT32, top_level>(
            std::make_unique<bodo::DataType>(bodo_array_type::NULLABLE_INT_BOOL,
                                             Bodo_CTypes::INT32),
            shuffle_comm, source, curr_tag, recv_state, lens_iter);

    uint64_t dict_len = *lens_iter++;
    uint64_t dict_data_len = *lens_iter++;

    // Allocate the dict array
    std::unique_ptr<array_info> dict_arr =
        alloc_string_array(Bodo_CTypes::STRING, dict_len, dict_data_len);
    // Receive the dict data, offsets and null bitmask
    MPI_Request data_req;
    CHECK_MPI(MPI_Irecv(dict_arr->data1<bodo_array_type::STRING>(),
                        dict_data_len, MPI_UNSIGNED_CHAR, source, curr_tag++,
                        shuffle_comm, &data_req),
              "recv_shuffle_data[DICT]: MPI error on MPI_Irecv:");
    recv_state.recv_requests.push_back(data_req);
    MPI_Request offset_req;
    CHECK_MPI(
        MPI_Irecv(dict_arr->data2<bodo_array_type::STRING>(), dict_len + 1,
                  MPI_UINT64_T, source, curr_tag++, shuffle_comm, &offset_req),
        "recv_shuffle_data[DICT]: MPI error on MPI_Irecv:");
    recv_state.recv_requests.push_back(offset_req);
    MPI_Request null_req;
    CHECK_MPI(MPI_Irecv(dict_arr->null_bitmask<bodo_array_type::STRING>(),
                        arrow::bit_util::BytesForBits(dict_arr->length),
                        MPI_UNSIGNED_CHAR, source, curr_tag++, shuffle_comm,
                        &null_req),
              "recv_shuffle_data[DICT]: MPI error on MPI_Irecv:");
    recv_state.recv_requests.push_back(null_req);

    return create_dict_string_array(std::move(dict_arr), std::move(indices));
}

/**
 * Receive the data1 buffer, null bitmask, and child array for a list array
 */
template <bodo_array_type::arr_type_enum arr_type, Bodo_CTypes::CTypeEnum dtype,
          bool top_level>
    requires(arr_type == bodo_array_type::ARRAY_ITEM)
std::unique_ptr<array_info> recv_shuffle_data(
    const std::unique_ptr<bodo::DataType>& data_type,
    const MPI_Comm shuffle_comm, const int source, int& curr_tag,
    AsyncShuffleRecvState& recv_state, len_iter_t& lens_iter) {
    const MPI_Datatype len_mpi_type = MPI_UINT32_T;
    int64_t arr_len = *lens_iter++;

    std::unique_ptr<array_info> arr = alloc_array_item(arr_len, nullptr);

    MPI_Request len_recv_req;
    CHECK_MPI(MPI_Irecv(arr->data1<arr_type>(), arr_len, len_mpi_type, source,
                        curr_tag++, shuffle_comm, &len_recv_req),
              "recv_shuffle_data[ARRAY_ITEM]: MPI error on MPI_Irecv:");
    recv_state.recv_requests.push_back(len_recv_req);

    recv_null_bitmask<arr_type>(arr, shuffle_comm, source, curr_tag,
                                recv_state);

    const bodo::ArrayType* array_type =
        static_cast<bodo::ArrayType*>(data_type.get());
    const std::unique_ptr<bodo::DataType>& value_type = array_type->value_type;
    arr->child_arrays[0] = recv_shuffle_data_unknown_type<false>(
        value_type, shuffle_comm, source, curr_tag, recv_state, lens_iter);

    return arr;
}

/**
 * Receive the data1 buffer, data2 buffer, and null bitmask for a timestamptz
 * array
 */
template <bodo_array_type::arr_type_enum arr_type, Bodo_CTypes::CTypeEnum dtype,
          bool top_level>
    requires(arr_type == bodo_array_type::TIMESTAMPTZ)
std::unique_ptr<array_info> recv_shuffle_data(
    const std::unique_ptr<bodo::DataType>& data_type,
    const MPI_Comm shuffle_comm, const int source, int& curr_tag,
    AsyncShuffleRecvState& recv_state, len_iter_t& lens_iter) {
    const MPI_Datatype mpi_datetime_type = get_MPI_typ<Bodo_CTypes::DATETIME>();
    const MPI_Datatype mpi_tz_offset_type = get_MPI_typ<Bodo_CTypes::INT16>();

    uint64_t arr_len = *lens_iter++;
    std::unique_ptr<array_info> out_arr = alloc_timestamptz_array(arr_len);

    MPI_Request datetime_req;
    CHECK_MPI(MPI_Irecv(out_arr->data1<arr_type>(), arr_len, mpi_datetime_type,
                        source, curr_tag++, shuffle_comm, &datetime_req),
              "recv_shuffle_data[TIMESTAMPTZ]: MPI error on MPI_Irecv:");
    recv_state.recv_requests.push_back(datetime_req);

    MPI_Request tz_offset_req;
    CHECK_MPI(MPI_Irecv(out_arr->data2<arr_type>(), arr_len, mpi_tz_offset_type,
                        source, curr_tag++, shuffle_comm, &tz_offset_req),
              "recv_shuffle_data[TIMESTAMPTZ]: MPI error on MPI_Irecv:");
    recv_state.recv_requests.push_back(tz_offset_req);

    recv_null_bitmask<arr_type>(out_arr, shuffle_comm, source, curr_tag,
                                recv_state);
    return out_arr;
}

/**
 * Receive nested arrays and null bitmask for a struct array
 */
template <bodo_array_type::arr_type_enum arr_type, Bodo_CTypes::CTypeEnum dtype,
          bool top_level>
    requires(arr_type == bodo_array_type::STRUCT)
std::unique_ptr<array_info> recv_shuffle_data(
    const std::unique_ptr<bodo::DataType>& data_type,
    const MPI_Comm shuffle_comm, const int source, int& curr_tag,
    AsyncShuffleRecvState& recv_state, len_iter_t& lens_iter) {
    std::vector<std::shared_ptr<array_info>> child_arrays;
    uint64_t arr_len = *lens_iter++;

    assert(data_type->is_struct());
    const bodo::StructType* struct_type =
        static_cast<bodo::StructType*>(data_type.get());

    for (size_t i = 0; i < struct_type->child_types.size(); i++) {
        // We can pass top_level through instead of setting it to false
        // because all child arrays shuold be the length of the struct array
        child_arrays.push_back(recv_shuffle_data_unknown_type<top_level>(
            struct_type->child_types[i], shuffle_comm, source, curr_tag,
            recv_state, lens_iter));
    }
    std::unique_ptr<array_info> out_arr = alloc_struct(arr_len, child_arrays);
    recv_null_bitmask<arr_type>(out_arr, shuffle_comm, source, curr_tag,
                                recv_state);
    return out_arr;
}

/**
 * Receive nested array(item) array for a map array
 */
template <bodo_array_type::arr_type_enum arr_type, Bodo_CTypes::CTypeEnum dtype,
          bool top_level>
    requires(arr_type == bodo_array_type::MAP)
std::unique_ptr<array_info> recv_shuffle_data(
    const std::unique_ptr<bodo::DataType>& data_type,
    const MPI_Comm shuffle_comm, const int source, int& curr_tag,
    AsyncShuffleRecvState& recv_state, len_iter_t& lens_iter) {
    lens_iter++;
    // Map arrays are just wrappers around array item arrays of structs
    assert(data_type->is_map());
    const bodo::MapType* map_type =
        static_cast<bodo::MapType*>(data_type.get());

    std::vector<std::unique_ptr<bodo::DataType>> struct_children;
    struct_children.emplace_back(map_type->key_type->copy());
    struct_children.emplace_back(map_type->value_type->copy());

    std::unique_ptr<bodo::StructType> struct_type =
        std::make_unique<bodo::StructType>(std::move(struct_children));
    std::unique_ptr<bodo::ArrayType> array_type =
        std::make_unique<bodo::ArrayType>(std::move(struct_type));

    std::unique_ptr<array_info> child =
        recv_shuffle_data<bodo_array_type::ARRAY_ITEM, Bodo_CTypes::LIST,
                          top_level>(std::move(array_type), shuffle_comm,
                                     source, curr_tag, recv_state, lens_iter);
    size_t child_len = child->length;
    return alloc_map(child_len, std::move(child));
}

/**
 * @brief State for non-blocking is_last synchronization using IBarrier.
 Used in streaming Iceberg and Parquet writes currently.
 *
 */
class IsLastState {
   public:
    // The IBarrier request used for is_last synchronization
    MPI_Request is_last_request = MPI_REQUEST_NULL;
    bool is_last_barrier_started = false;
    bool global_is_last = false;
    MPI_Comm is_last_comm;

    IsLastState() {
        CHECK_MPI(MPI_Comm_dup(MPI_COMM_WORLD, &this->is_last_comm),
                  "IsLastState: MPI error on MPI_Comm_dup:");
    }
    IsLastState(MPI_Comm parent_comm) {
        CHECK_MPI(MPI_Comm_dup(parent_comm, &this->is_last_comm),
                  "IsLastState: MPI error on MPI_Comm_dup:");
    }
    ~IsLastState() { MPI_Comm_free(&this->is_last_comm); }
};

/**
 * @brief Performs non-blocking synchronization of is_last flag
 *
 * @param state non-blocking synchronization state
 * @param local_is_last local is_last flag that needs synchronized
 * @return 1 if is_last is true on all ranks else 0
 */
int32_t sync_is_last_non_blocking(IsLastState* state, int32_t local_is_last);
