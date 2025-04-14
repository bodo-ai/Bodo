#include <iostream>

#include "duckdb/main/client_context.hpp"
#include "duckdb/main/database.hpp"

int main() {
    duckdb::DuckDB db(nullptr);
    duckdb::ClientContext context(db.instance);
	std::cerr << std::boolalpha << "ClientContext initialized " << context.interrupted << '\n';
	return 0;
}
