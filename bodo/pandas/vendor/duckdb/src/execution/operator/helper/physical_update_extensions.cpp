#include "duckdb/execution/operator/helper/physical_update_extensions.hpp"
#include "duckdb/common/enums/operator_result_type.hpp"

namespace duckdb {

SourceResultType PhysicalUpdateExtensions::GetData(ExecutionContext &context, DataChunk &chunk,
                                                   OperatorSourceInput &input) const {
    return SourceResultType::FINISHED;
}

unique_ptr<GlobalSourceState> PhysicalUpdateExtensions::GetGlobalSourceState(ClientContext &context) const {
	auto res = make_uniq<UpdateExtensionsGlobalState>();
	return std::move(res);
}

} // namespace duckdb
