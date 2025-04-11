#include "duckdb/execution/operator/helper/physical_load.hpp"

namespace duckdb {


SourceResultType PhysicalLoad::GetData(ExecutionContext &context, DataChunk &chunk, OperatorSourceInput &input) const {
	return SourceResultType::FINISHED;
}

} // namespace duckdb
