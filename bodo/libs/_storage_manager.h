// Copyright (C) 2023 Bodo Inc. All rights reserved.
#pragma once

#include <chrono>
#include <span>

#include <arrow/status.h>

// Forward declare boost::json::object to avoid including the entire header and
// increasing compile times
namespace boost {
namespace json {
class object;
}
}  // namespace boost

// TODO Tell the compiler that the branch is unlikely.
#define CHECK_ARROW_MEM(expr, msg)                            \
    {                                                         \
        arrow::Status __status = expr;                        \
        if (!__status.ok()) {                                 \
            std::string err_msg =                             \
                std::string(msg) + " " + __status.ToString(); \
            throw std::runtime_error(err_msg);                \
        }                                                     \
    }

#define CHECK_ARROW_MEM_RET(expr, msg)                                      \
    {                                                                       \
        auto stat = expr;                                                   \
        if (!stat.ok()) {                                                   \
            std::string err_msg = std::string(msg) + " " + stat.ToString(); \
            return stat.WithMessage(err_msg);                               \
        }                                                                   \
    }

using milli_double = std::chrono::duration<double, std::milli>;

namespace bodo {

// --------------------------- Storage Options --------------------------- //
/// @brief Enum to Indicate which Manager to Use
enum StorageType : uint8_t {
    Local = 0,
    S3 = 1,
    Azure = 2,
};

/// @brief Options for Storage Manager Implementations
struct StorageOptions {
    /// @brief Amount of bytes allowed to be spilled to
    /// storage location
    int64_t usable_size = 1024ll * 1024ll * 1024ll;

    /// @brief Location / folder to write block spill files
    std::string location;

    /// @brief Type of StorageManager to use
    StorageType type = StorageType::Local;

    /// @brief Enable Tracing Mode
    bool tracing_mode = false;

    static std::shared_ptr<StorageOptions> Defaults(uint8_t tier);
};

// --------------------------- Storage Metrics --------------------------- //
/// @brief Struct to track Storage Manager statistics
/// from the point of creation
struct StorageManagerStats {
    /// @brief Current # of bytes spilled to storage
    uint64_t curr_spilled_bytes = 0;

    /// @brief Current # of blocks spilled in storage
    uint64_t curr_num_blocks_spilled = 0;

    /// @brief Total # of blocks ever spilled from storage
    uint64_t total_num_blocks_spilled = 0;

    /// @brief Total # of blocks ever read from storage
    uint64_t total_num_blocks_read = 0;

    /// @brief Total # of calls to delete from storage
    /// Managers can choose to batch deletes, so this
    /// may not be equal to total_num_blocks_read
    uint64_t total_num_del_calls = 0;

    /// @brief Total # of bytes spilled to storage
    uint64_t total_spilled_bytes = 0;

    /// @brief Total # of bytes read from storage
    uint64_t total_read_bytes = 0;

    /// @brief Total # of bytes deleted from storage
    /// Managers can choose to batch deletes, so this
    /// may not be equal to total_read_bytes
    uint64_t total_bytes_del = 0;

    /// @brief Max # of bytes spilled to storage
    uint64_t max_spilled_bytes = 0;

    /// @brief Total time spent writing to storage
    milli_double total_write_time{};

    /// @brief Total time spent reading from storage
    milli_double total_read_time{};

    /// @brief Total time spent deleting from storage
    milli_double total_delete_time{};
};

// --------------------------- Storage Manager --------------------------- //
/// @brief Abstract Class / Interface for Storage Managers
/// Storage Managers manage the reading + writing of blocks
/// from a storage location as well as size limitations
class StorageManager {
   public:
    StorageManager(std::string storage_name,
                   const std::shared_ptr<StorageOptions> options);

    // Required otherwise won't compile
    // All cleanup operations occur within the Cleanup virtual function
    // so that exceptions can be safely raised
    // Cleanup runs at program exit. However, if we never want cleanup
    // to raise errors, we should move it back to the destructor
    // TODO: Revisit in the future
    virtual ~StorageManager() = default;

    /// @brief How many bytes are available to be stored in this
    /// storage location. -1 indicates unlimited
    inline int64_t usable_size() const { return options->usable_size; }

    /**
     * @brief Update the current number of spilled bytes to this
     * storage location by diff
     *
     * @param diff The number of bytes added or removed from this
     * storage location.
     */
    void UpdateSpilledBytes(int64_t diff);

    /// @brief Get the next available block id for this storage location
    int64_t GetNewBlockID() { return this->block_id_counter++; }

    // ------------------------- Virtual Functions ------------------------- //

    /**
     * @brief Is there space available in this storage location for
     * allocations to be spilled to it
     *
     * @param size_class_idx Idx of size class to spill
     * @return true If allocation can be spilled to it
     * @return false If allocation can not be spilled here
     */
    virtual bool CanSpillTo(uint8_t size_class_idx) = 0;

    /**
     * @brief Read a block with id block_id and size of n_bytes
     * from storage, write its contents to out_ptr, and delete
     * block from storage.
     *
     * @param[in] block_id Index of block to read from storage
     * @param[in] size_class_idx Idx of size class of block
     * @param[out] out_ptr Location to write contents to
     * @throws Potential filesystem error raised during read.
     * Exception is used because we stop execution on failure
     */
    virtual void ReadBlock(int64_t block_id, uint8_t size_class_idx,
                           uint8_t* out_ptr) = 0;

    /**
     * @brief Write the contents of a frame located at in_ptr
     * and size frame_size to storage, and return the new block id.
     *
     * @param in_ptr Location of frame to write
     * @param size_class_idx Idx of size class of frame
     * @return int64_t Block ID of contents if success.
     *  or -1 if we can't spill to frame
     * @throws std::runtime_error For filesystem error during write
     * Exception is used because we stop execution on failure
     */
    virtual int64_t WriteBlock(uint8_t* in_ptr, uint8_t size_class_idx) = 0;

    /**
     * @brief Delete a block with id block_id and size of n_bytes from
     * storage.
     *
     * @param block_id Index of block to delete from storage
     * @param n_bytes Size of block in bytes
     * @return arrow::Status Ok if success, otherwise a potential
     * filesystem error raised during delete.
     * arrow::Status is used because we often ignore failures
     */
    virtual arrow::Status DeleteBlock(int64_t block_id,
                                      uint8_t size_class_idx) = 0;

    /// @brief Cleanup any leftover spill files
    /// Expected to run during program exit and can throw
    /// an error on fail
    virtual void Cleanup() = 0;

    /// @brief Name of the manager
    /// What to refer to in logs
    std::string storage_name;

    /// @brief All current Storage Manager statistics
    StorageManagerStats stats_;

   protected:
    /// @brief Rank and process-unique identifier for
    /// spilling. Can be used for location handling
    std::string uuid;

    /// @brief Configuration Options
    std::shared_ptr<StorageOptions> options;

   private:
    /// @brief Increment every time we write a block to disk
    int64_t block_id_counter = 0;
};

/// @brief Factory Function to Create StorageManager based on StorageOptions
std::unique_ptr<StorageManager> MakeStorageManager(
    const std::shared_ptr<StorageOptions>& options,
    const std::span<const uint64_t> size_class_bytes);

/// @brief Print Storage Manager Statistics for All Managers
/// in an easy-to-read table format
void PrintStorageManagerStats(FILE* os,
                              const std::span<std::string_view> m_names,
                              const std::span<const StorageManagerStats> stats);

/// @brief Get Storage Manager Statistics for All Managers as JSON
boost::json::object GetStorageManagerStats(
    const std::span<std::string_view> m_names,
    const std::span<const StorageManagerStats> stats);

};  // namespace bodo
