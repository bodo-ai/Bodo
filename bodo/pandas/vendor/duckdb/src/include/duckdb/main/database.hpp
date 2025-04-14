//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/main/database.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/winapi.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/settings.hpp"
#include "duckdb/main/valid_checker.hpp"

namespace duckdb {
class BufferManager;
class DatabaseManager;
class StorageManager;
class Catalog;
class TransactionManager;
class ConnectionManager;
class FileSystem;
class TaskScheduler;
class ObjectCache;
struct AttachInfo;
struct AttachOptions;
class DatabaseFileSystem;
struct DatabaseCacheEntry;
class LogManager;


class DatabaseInstance : public enable_shared_from_this<DatabaseInstance> {
	friend class DuckDB;

public:
	DUCKDB_API DatabaseInstance();
	DUCKDB_API ~DatabaseInstance();

	DBConfig config;

public:
	BufferPool &GetBufferPool() const;
	DUCKDB_API SecretManager &GetSecretManager();
	DUCKDB_API BufferManager &GetBufferManager();
	DUCKDB_API const BufferManager &GetBufferManager() const;
	DUCKDB_API DatabaseManager &GetDatabaseManager();
	DUCKDB_API FileSystem &GetFileSystem();
	DUCKDB_API TaskScheduler &GetScheduler();
	DUCKDB_API ObjectCache &GetObjectCache();
	DUCKDB_API ConnectionManager &GetConnectionManager();
	DUCKDB_API ValidChecker &GetValidChecker();
	DUCKDB_API LogManager &GetLogManager() const;

	idx_t NumberOfThreads();

	DUCKDB_API static DatabaseInstance &GetDatabase(ClientContext &context);
	DUCKDB_API static const DatabaseInstance &GetDatabase(const ClientContext &context);

	DUCKDB_API SettingLookupResult TryGetCurrentSetting(const string &key, Value &result) const;

	unique_ptr<AttachedDatabase> CreateAttachedDatabase(ClientContext &context, const AttachInfo &info,
	                                                    const AttachOptions &options);

private:
	void Initialize(const char *path, DBConfig *config);
	void CreateMainDatabase();

	void Configure(DBConfig &config, const char *path);

private:
	shared_ptr<BufferManager> buffer_manager;
	unique_ptr<DatabaseManager> db_manager;
	unique_ptr<TaskScheduler> scheduler;
	unique_ptr<ObjectCache> object_cache;
	unique_ptr<ConnectionManager> connection_manager;
	ValidChecker db_validity;
	unique_ptr<DatabaseFileSystem> db_file_system;
	shared_ptr<LogManager> log_manager;
};

//! The database object. This object holds the catalog and all the
//! database-specific meta information.
class DuckDB {
public:
	DUCKDB_API explicit DuckDB(const char *path = nullptr, DBConfig *config = nullptr);
	DUCKDB_API explicit DuckDB(const string &path, DBConfig *config = nullptr);
	DUCKDB_API explicit DuckDB(DatabaseInstance &instance);

	DUCKDB_API ~DuckDB();

	//! Reference to the actual database instance
	shared_ptr<DatabaseInstance> instance;

public:
	DUCKDB_API FileSystem &GetFileSystem();

	DUCKDB_API idx_t NumberOfThreads();
	DUCKDB_API static const char *SourceID();
	DUCKDB_API static const char *LibraryVersion();
	DUCKDB_API static idx_t StandardVectorSize();
	DUCKDB_API static string Platform();
};

} // namespace duckdb
