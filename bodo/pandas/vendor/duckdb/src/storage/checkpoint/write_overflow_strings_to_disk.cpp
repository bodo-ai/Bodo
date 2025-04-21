#include "duckdb/storage/checkpoint/write_overflow_strings_to_disk.hpp"
#include "duckdb/storage/block_manager.hpp"
#include "duckdb/storage/buffer_manager.hpp"
#include "duckdb/storage/partial_block_manager.hpp"

namespace duckdb {

WriteOverflowStringsToDisk::WriteOverflowStringsToDisk(PartialBlockManager &partial_block_manager)
    : partial_block_manager(partial_block_manager), block_id(INVALID_BLOCK), offset(0) {
}

WriteOverflowStringsToDisk::~WriteOverflowStringsToDisk() {
	// verify that the overflow writer has been flushed
	D_ASSERT(Exception::UncaughtException() || offset == 0);
}

void WriteOverflowStringsToDisk::WriteString(string_t string,
                                             block_id_t &result_block, int32_t &result_offset) {
	auto &block_manager = partial_block_manager.GetBlockManager();
	auto &buffer_manager = block_manager.buffer_manager;
	if (!handle.IsValid()) {
		handle = buffer_manager.Allocate(MemoryTag::OVERFLOW_STRINGS, block_manager.GetBlockSize());
	}
	// first write the length of the string
	if (block_id == INVALID_BLOCK || offset + 2 * sizeof(uint32_t) >= GetStringSpace()) {
		// AllocateNewBlock(state, block_manager.GetFreeBlockId());
	}
	result_block = block_id;
	result_offset = UnsafeNumericCast<int32_t>(offset);

	// write the length field
	auto data_ptr = handle.Ptr();
	auto string_length = string.GetSize();
	Store<uint32_t>(UnsafeNumericCast<uint32_t>(string_length), data_ptr + offset);
	offset += sizeof(uint32_t);

	// now write the remainder of the string
	auto strptr = string.GetData();
	auto remaining = UnsafeNumericCast<uint32_t>(string_length);
	while (remaining > 0) {
		uint32_t to_write = MinValue<uint32_t>(remaining, UnsafeNumericCast<uint32_t>(GetStringSpace() - offset));
		if (to_write > 0) {
			memcpy(data_ptr + offset, strptr, to_write);

			remaining -= to_write;
			offset += to_write;
			strptr += to_write;
		}
		if (remaining > 0) {
			D_ASSERT(offset == GetStringSpace());
			// there is still remaining stuff to write
			// now write the current block to disk and allocate a new block
			// AllocateNewBlock(state, block_manager.GetFreeBlockId());
		}
	}
}

void WriteOverflowStringsToDisk::Flush() {
	if (block_id != INVALID_BLOCK && offset > 0) {
		// zero-initialize the empty part of the overflow string buffer (if any)
		if (offset < GetStringSpace()) {
			memset(handle.Ptr() + offset, 0, GetStringSpace() - offset);
		}
		// write to disk
		auto &block_manager = partial_block_manager.GetBlockManager();
		block_manager.Write(handle.GetFileBuffer(), block_id);
	}
	block_id = INVALID_BLOCK;
	offset = 0;
}

void WriteOverflowStringsToDisk::AllocateNewBlock(block_id_t new_block_id) {
	if (block_id != INVALID_BLOCK) {
		// there is an old block, write it first
		// write the new block id at the end of the previous block
		Store<block_id_t>(new_block_id, handle.Ptr() + GetStringSpace());
		Flush();
	}
	offset = 0;
	block_id = new_block_id;
}

idx_t WriteOverflowStringsToDisk::GetStringSpace() const {
	auto &block_manager = partial_block_manager.GetBlockManager();
	return block_manager.GetBlockSize() - sizeof(block_id_t);
}

} // namespace duckdb
