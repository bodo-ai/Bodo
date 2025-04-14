#include "duckdb/execution/operator/helper/physical_reset.hpp"

#include "duckdb/common/string_util.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/client_context.hpp"

namespace duckdb {

SourceResultType PhysicalReset::GetData(ExecutionContext &context, DataChunk &chunk, OperatorSourceInput &input) const {
	if (scope == SetScope::VARIABLE) {
		auto &client_config = ClientConfig::GetConfig(context.client);
		client_config.ResetUserVariable(name);
		return SourceResultType::FINISHED;
	}
	auto &config = DBConfig::GetConfig(context.client);
	config.CheckLock(name);
	auto option = DBConfig::GetOptionByName(name);
	if (!option) {
		// check if this is an extra extension variable
		return SourceResultType::FINISHED;
	}

	// Transform scope
	SetScope variable_scope = scope;
	if (variable_scope == SetScope::AUTOMATIC) {
		if (option->set_local) {
			variable_scope = SetScope::SESSION;
		} else {
			D_ASSERT(option->set_global);
			variable_scope = SetScope::GLOBAL;
		}
	}

	switch (variable_scope) {
	case SetScope::GLOBAL: {
		if (!option->set_global) {
			throw CatalogException("option \"%s\" cannot be reset globally", name);
		}
		auto &db = DatabaseInstance::GetDatabase(context.client);
		config.ResetOption(&db, *option);
		break;
	}
	case SetScope::SESSION:
		if (!option->reset_local) {
			throw CatalogException("option \"%s\" cannot be reset locally", name);
		}
		option->reset_local(context.client);
		break;
	default:
		throw InternalException("Unsupported SetScope for variable");
	}

	return SourceResultType::FINISHED;
}

} // namespace duckdb
