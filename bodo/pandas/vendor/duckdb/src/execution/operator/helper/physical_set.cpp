#include "duckdb/execution/operator/helper/physical_set.hpp"

#include "duckdb/common/string_util.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/client_context.hpp"

namespace duckdb {

SourceResultType PhysicalSet::GetData(ExecutionContext &context, DataChunk &chunk, OperatorSourceInput &input) const {
	auto &config = DBConfig::GetConfig(context.client);
	// check if we are allowed to change the configuration option
	config.CheckLock(name);
	auto option = DBConfig::GetOptionByName(name);
	if (!option) {
		return SourceResultType::FINISHED;
	}
	SetScope variable_scope = scope;
	if (variable_scope == SetScope::AUTOMATIC) {
		if (option->set_local) {
			variable_scope = SetScope::SESSION;
		} else {
			D_ASSERT(option->set_global);
			variable_scope = SetScope::GLOBAL;
		}
	}

	Value input_val = value.CastAs(context.client, DBConfig::ParseLogicalType(option->parameter_type));
	switch (variable_scope) {
	case SetScope::GLOBAL: {
		if (!option->set_global) {
			throw CatalogException("option \"%s\" cannot be set globally", name);
		}
		auto &db = DatabaseInstance::GetDatabase(context.client);
		auto &config = DBConfig::GetConfig(context.client);
		config.SetOption(&db, *option, input_val);
		break;
	}
	case SetScope::SESSION:
		if (!option->set_local) {
			throw CatalogException("option \"%s\" cannot be set locally", name);
		}
		option->set_local(context.client, input_val);
		break;
	default:
		throw InternalException("Unsupported SetScope for variable");
	}

	return SourceResultType::FINISHED;
}

} // namespace duckdb
