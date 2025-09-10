//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/planner/operator/logical_materialized_cte.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#if 0
#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {

class LogicalMaterializedCTE : public LogicalOperator {
	explicit LogicalMaterializedCTE() : LogicalOperator(LogicalOperatorType::LOGICAL_MATERIALIZED_CTE) {
	}

public:
	static constexpr const LogicalOperatorType TYPE = LogicalOperatorType::LOGICAL_MATERIALIZED_CTE;

public:
	LogicalMaterializedCTE(string ctename_p, idx_t table_index, idx_t column_count, unique_ptr<LogicalOperator> cte,
	                       unique_ptr<LogicalOperator> child)
	    : LogicalOperator(LogicalOperatorType::LOGICAL_MATERIALIZED_CTE), table_index(table_index),
	      column_count(column_count), ctename(std::move(ctename_p)) {
		children.push_back(std::move(cte));
		children.push_back(std::move(child));
	}

	idx_t table_index;
	idx_t column_count;
	string ctename;

    CTEMaterialize materialize = CTEMaterialize::CTE_MATERIALIZE_DEFAULT;

public:
	InsertionOrderPreservingMap<string> ParamsToString() const override;
	vector<ColumnBinding> GetColumnBindings() override {
		return children[1]->GetColumnBindings();
	}

	void Serialize(Serializer &serializer) const override;
	static unique_ptr<LogicalOperator> Deserialize(Deserializer &deserializer);

	vector<idx_t> GetTableIndex() const override;

protected:
	void ResolveTypes() override {
		types = children[1]->types;
	}
};
} // namespace duckdb
#else
#include "duckdb/planner/operator/logical_cte.hpp"
#include "duckdb/planner/logical_operator.hpp"

namespace duckdb {

class LogicalMaterializedCTE : public LogicalCTE {
    explicit LogicalMaterializedCTE() : LogicalCTE(LogicalOperatorType::LOGICAL_MATERIALIZED_CTE) {
    }

public:
    static constexpr const LogicalOperatorType TYPE = LogicalOperatorType::LOGICAL_MATERIALIZED_CTE;

public:
    LogicalMaterializedCTE(string ctename_p, idx_t table_index, idx_t column_count, unique_ptr<LogicalOperator> cte,
                           unique_ptr<LogicalOperator> child, CTEMaterialize materialize = CTEMaterialize::CTE_MATERIALIZE_DEFAULT)
        : LogicalCTE(std::move(ctename_p), table_index, column_count, std::move(cte), std::move(child),
                     LogicalOperatorType::LOGICAL_MATERIALIZED_CTE),
          materialize(materialize) {
    }

    CTEMaterialize materialize = CTEMaterialize::CTE_MATERIALIZE_ALWAYS;

public:
    InsertionOrderPreservingMap<string> ParamsToString() const override;
    vector<ColumnBinding> GetColumnBindings() override {
        return children[1]->GetColumnBindings();
    }

    void Serialize(Serializer &serializer) const override;
    static unique_ptr<LogicalOperator> Deserialize(Deserializer &deserializer);

    vector<idx_t> GetTableIndex() const override;

protected:
    void ResolveTypes() override {
        types = children[1]->types;
    }
};
} // namespace duckdb
#endif
