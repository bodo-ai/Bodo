#include "duckdb/common/pair.hpp"
#include "duckdb/function/compression_function.hpp"
#include "duckdb/main/config.hpp"

namespace duckdb {

typedef CompressionFunction (*get_compression_function_t)(PhysicalType type);
typedef bool (*compression_supports_type_t)(const PhysicalType physical_type);

struct DefaultCompressionMethod {
	CompressionType type;
	get_compression_function_t get_function;
	compression_supports_type_t supports_type;
};

static optional_ptr<CompressionFunction> FindCompressionFunction(CompressionFunctionSet &set, CompressionType type,
                                                                 const PhysicalType physical_type) {
	auto &functions = set.functions;
	auto comp_entry = functions.find(type);
	if (comp_entry != functions.end()) {
		auto &type_functions = comp_entry->second;
		auto type_entry = type_functions.find(physical_type);
		if (type_entry != type_functions.end()) {
			return &type_entry->second;
		}
	}
	return nullptr;
}

static optional_ptr<CompressionFunction> LoadCompressionFunction(CompressionFunctionSet &set, CompressionType type,
                                                                 const PhysicalType physical_type) {
	throw InternalException("Unsupported compression function type");
}

static void TryLoadCompression(DBConfig &config, vector<reference<CompressionFunction>> &result, CompressionType type,
                               const PhysicalType physical_type) {
	if (config.options.disabled_compression_methods.find(type) != config.options.disabled_compression_methods.end()) {
		// explicitly disabled
		return;
	}
	auto function = config.GetCompressionFunction(type, physical_type);
	if (!function) {
		return;
	}
	result.push_back(*function);
}

vector<reference<CompressionFunction>> DBConfig::GetCompressionFunctions(const PhysicalType physical_type) {
	vector<reference<CompressionFunction>> result;
	TryLoadCompression(*this, result, CompressionType::COMPRESSION_UNCOMPRESSED, physical_type);
	TryLoadCompression(*this, result, CompressionType::COMPRESSION_RLE, physical_type);
	TryLoadCompression(*this, result, CompressionType::COMPRESSION_BITPACKING, physical_type);
	TryLoadCompression(*this, result, CompressionType::COMPRESSION_DICTIONARY, physical_type);
	TryLoadCompression(*this, result, CompressionType::COMPRESSION_CHIMP, physical_type);
	TryLoadCompression(*this, result, CompressionType::COMPRESSION_PATAS, physical_type);
	TryLoadCompression(*this, result, CompressionType::COMPRESSION_ALP, physical_type);
	TryLoadCompression(*this, result, CompressionType::COMPRESSION_ALPRD, physical_type);
	TryLoadCompression(*this, result, CompressionType::COMPRESSION_FSST, physical_type);
	TryLoadCompression(*this, result, CompressionType::COMPRESSION_ZSTD, physical_type);
	TryLoadCompression(*this, result, CompressionType::COMPRESSION_ROARING, physical_type);
	return result;
}

optional_ptr<CompressionFunction> DBConfig::GetCompressionFunction(CompressionType type,
                                                                   const PhysicalType physical_type) {
	lock_guard<mutex> l(compression_functions->lock);

	// Check if the function is already loaded into the global compression functions.
	auto function = FindCompressionFunction(*compression_functions, type, physical_type);
	if (function) {
		return function;
	}

	// We could not find the function in the global compression functions,
	// so we attempt loading it.
	return LoadCompressionFunction(*compression_functions, type, physical_type);
}

} // namespace duckdb
