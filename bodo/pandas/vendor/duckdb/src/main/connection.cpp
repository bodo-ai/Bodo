#include "duckdb/main/connection.hpp"

#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/function/table/read_csv.hpp"
#include "duckdb/main/appender.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/connection_manager.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/query_profiler.hpp"
#include "duckdb/parser/parser.hpp"
#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {

Connection::Connection(DatabaseInstance &database)
    : context(make_shared_ptr<ClientContext>(database.shared_from_this())), warning_cb(nullptr) {
	auto &connection_manager = ConnectionManager::Get(database);
	connection_manager.AddConnection(*context);
	connection_manager.AssignConnectionId(*this);

#ifdef DEBUG
	EnableProfiling();
	context->config.emit_profiler_output = false;
#endif
}

Connection::Connection(DuckDB &database) : Connection(*database.instance) {
	// Initialization of warning_cb happens in the other constructor
}

Connection::Connection(Connection &&other) noexcept : warning_cb(nullptr) {
	std::swap(context, other.context);
	std::swap(warning_cb, other.warning_cb);
	std::swap(connection_id, other.connection_id);
}

Connection &Connection::operator=(Connection &&other) noexcept {
	std::swap(context, other.context);
	std::swap(warning_cb, other.warning_cb);
	std::swap(connection_id, other.connection_id);
	return *this;
}

Connection::~Connection() {
	if (!context) {
		return;
	}
	ConnectionManager::Get(*context->db).RemoveConnection(*context);
}

string Connection::GetProfilingInformation(ProfilerPrintFormat format) {
	auto &profiler = QueryProfiler::Get(*context);
	return profiler.ToString(format);
}

optional_ptr<ProfilingNode> Connection::GetProfilingTree() {
	auto &client_config = ClientConfig::GetConfig(*context);
	auto enable_profiler = client_config.enable_profiler;

	if (!enable_profiler) {
		throw Exception(ExceptionType::SETTINGS, "Profiling is not enabled for this connection");
	}
	auto &profiler = QueryProfiler::Get(*context);
	return profiler.GetRoot();
}

void Connection::Interrupt() {
	context->Interrupt();
}

void Connection::EnableProfiling() {
	context->EnableProfiling();
}

void Connection::DisableProfiling() {
	context->DisableProfiling();
}

void Connection::EnableQueryVerification() {
	ClientConfig::GetConfig(*context).query_verification_enabled = true;
}

void Connection::DisableQueryVerification() {
	ClientConfig::GetConfig(*context).query_verification_enabled = false;
}

void Connection::ForceParallelism() {
	ClientConfig::GetConfig(*context).verify_parallelism = true;
}

unique_ptr<QueryResult> Connection::SendQuery(const string &query) {
	return context->Query(query, true);
}

unique_ptr<MaterializedQueryResult> Connection::Query(const string &query) {
	auto result = context->Query(query, false);
	D_ASSERT(result->type == QueryResultType::MATERIALIZED_RESULT);
	return unique_ptr_cast<QueryResult, MaterializedQueryResult>(std::move(result));
}

unique_ptr<MaterializedQueryResult> Connection::Query(unique_ptr<SQLStatement> statement) {
	auto result = context->Query(std::move(statement), false);
	D_ASSERT(result->type == QueryResultType::MATERIALIZED_RESULT);
	return unique_ptr_cast<QueryResult, MaterializedQueryResult>(std::move(result));
}

unique_ptr<PendingQueryResult> Connection::PendingQuery(const string &query, bool allow_stream_result) {
	return context->PendingQuery(query, allow_stream_result);
}

unique_ptr<PendingQueryResult> Connection::PendingQuery(unique_ptr<SQLStatement> statement, bool allow_stream_result) {
	return context->PendingQuery(std::move(statement), allow_stream_result);
}

unique_ptr<PendingQueryResult> Connection::PendingQuery(const string &query,
                                                        case_insensitive_map_t<BoundParameterData> &named_values,
                                                        bool allow_stream_result) {
	return context->PendingQuery(query, named_values, allow_stream_result);
}

unique_ptr<PendingQueryResult> Connection::PendingQuery(unique_ptr<SQLStatement> statement,
                                                        case_insensitive_map_t<BoundParameterData> &named_values,
                                                        bool allow_stream_result) {
	return context->PendingQuery(std::move(statement), named_values, allow_stream_result);
}

static case_insensitive_map_t<BoundParameterData> ConvertParamListToMap(vector<Value> &param_list) {
	case_insensitive_map_t<BoundParameterData> named_values;
	for (idx_t i = 0; i < param_list.size(); i++) {
		auto &val = param_list[i];
		named_values[std::to_string(i + 1)] = BoundParameterData(val);
	}
	return named_values;
}

unique_ptr<PendingQueryResult> Connection::PendingQuery(const string &query, vector<Value> &values,
                                                        bool allow_stream_result) {
	auto named_params = ConvertParamListToMap(values);
	return context->PendingQuery(query, named_params, allow_stream_result);
}

unique_ptr<PendingQueryResult> Connection::PendingQuery(unique_ptr<SQLStatement> statement, vector<Value> &values,
                                                        bool allow_stream_result) {
	auto named_params = ConvertParamListToMap(values);
	return context->PendingQuery(std::move(statement), named_params, allow_stream_result);
}

unique_ptr<PreparedStatement> Connection::Prepare(const string &query) {
	return context->Prepare(query);
}

unique_ptr<PreparedStatement> Connection::Prepare(unique_ptr<SQLStatement> statement) {
	return context->Prepare(std::move(statement));
}

unique_ptr<QueryResult> Connection::QueryParamsRecursive(const string &query, vector<Value> &values) {
	auto named_params = ConvertParamListToMap(values);
	auto pending = PendingQuery(query, named_params, false);
	if (pending->HasError()) {
		return make_uniq<MaterializedQueryResult>(pending->GetErrorObject());
	}
	return pending->Execute();
}

unique_ptr<TableDescription> Connection::TableInfo(const string &database_name, const string &schema_name,
                                                   const string &table_name) {
	return context->TableInfo(database_name, schema_name, table_name);
}

unique_ptr<TableDescription> Connection::TableInfo(const string &schema_name, const string &table_name) {
	return TableInfo(INVALID_CATALOG, schema_name, table_name);
}

unique_ptr<TableDescription> Connection::TableInfo(const string &table_name) {
	return TableInfo(INVALID_CATALOG, DEFAULT_SCHEMA, table_name);
}

vector<unique_ptr<SQLStatement>> Connection::ExtractStatements(const string &query) {
	return context->ParseStatements(query);
}

unique_ptr<LogicalOperator> Connection::ExtractPlan(const string &query) {
	return context->ExtractPlan(query);
}

void Connection::Append(TableDescription &description, DataChunk &chunk) {
	if (chunk.size() == 0) {
		return;
	}
	ColumnDataCollection collection(Allocator::Get(*context), chunk.GetTypes());
	collection.Append(chunk);
	Append(description, collection);
}

void Connection::Append(TableDescription &description, ColumnDataCollection &collection) {
	context->Append(description, collection);
}

void Connection::BeginTransaction() {
	auto result = Query("BEGIN TRANSACTION");
	if (result->HasError()) {
		result->ThrowError();
	}
}

void Connection::Commit() {
	auto result = Query("COMMIT");
	if (result->HasError()) {
		result->ThrowError();
	}
}

void Connection::Rollback() {
	auto result = Query("ROLLBACK");
	if (result->HasError()) {
		result->ThrowError();
	}
}

void Connection::SetAutoCommit(bool auto_commit) {
	context->transaction.SetAutoCommit(auto_commit);
}

bool Connection::IsAutoCommit() {
	return context->transaction.IsAutoCommit();
}
bool Connection::HasActiveTransaction() {
	return context->transaction.HasActiveTransaction();
}

} // namespace duckdb
