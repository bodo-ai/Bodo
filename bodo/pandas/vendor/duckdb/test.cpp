#include <iostream>

#include "duckdb/main/client_context.hpp"
#include "duckdb/main/database.hpp"

duckdb::ClientContext &get_duckdb_context() {
    std::cerr << "get_duckdb_context" << '\n';
    duckdb::DuckDB db(nullptr);
    std::cerr << "get_duckdb_context context" << '\n';
    duckdb::ClientContext context(db.instance);
    std::cerr << "get_duckdb_context done" << '\n';
    return context;
}

int main() {
	duckdb::ClientContext &context = get_duckdb_context();
	std::cerr << std::boolalpha << "ClientContext initialized " << context.interrupted << '\n';
	return 0;
}
