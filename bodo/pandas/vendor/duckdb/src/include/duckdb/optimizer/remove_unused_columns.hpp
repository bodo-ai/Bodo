//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/remove_unused_columns.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/planner/logical_operator_visitor.hpp"
#include "duckdb/planner/column_binding_map.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/column_index.hpp"

namespace duckdb {
class Binder;
class BoundColumnRefExpression;
class ClientContext;

struct ReferencedColumn {
	vector<reference<BoundColumnRefExpression>> bindings;
	vector<ColumnIndex> child_columns;
};

class BaseColumnPruner : public LogicalOperatorVisitor {
protected:
	//! The map of column references
	column_binding_map_t<ReferencedColumn> column_references;

protected:
	void VisitExpression(unique_ptr<Expression> *expression) override;

	unique_ptr<Expression> VisitReplace(BoundColumnRefExpression &expr, unique_ptr<Expression> *expr_ptr) override;
	unique_ptr<Expression> VisitReplace(BoundReferenceExpression &expr, unique_ptr<Expression> *expr_ptr) override;

protected:
	//! Add a reference to the column in its entirey
	void AddBinding(BoundColumnRefExpression &col);
	//! Add a reference to a sub-section of the column
	void AddBinding(BoundColumnRefExpression &col, ColumnIndex child_column);
	//! Perform a replacement of the ColumnBinding, iterating over all the currently found column references and
	//! replacing the bindings
	void ReplaceBinding(ColumnBinding current_binding, ColumnBinding new_binding);

	bool HandleStructExtract(Expression &expr);

	bool HandleStructExtractRecursive(Expression &expr, optional_ptr<BoundColumnRefExpression> &colref,
	                                  vector<idx_t> &indexes);

    // Bodo Change: utility function to detect nested CTE case.
    bool HasColumnReferencesForTable(idx_t table_index) const;
    // Bodo Change End
};

// Bodo Change: new wrapper class for optimization pass to store one copy
// of information since the internal RemoveUnusedColumns original pass
// creates sub-instances of itself for various parts of the tree.
class RemoveUnusedColumns;

typedef unordered_map<idx_t, unordered_map<idx_t, idx_t>> CTEColMap;

class RemoveUnusedColumnsPass {
public:
	RemoveUnusedColumnsPass(Binder &binder, ClientContext &context)
	    : binder(binder), context(context) {}

	void VisitOperator(LogicalOperator &op);

    friend class RemoveUnusedColumns;
private:
	Binder &binder;
	ClientContext &context;

    // If key idx is a cte index then it is the union of all columns
    // needed by all the CTErefs for that CTE.
    // If key idx is a table index (currently only CTERef table indices
    // are stored) then it is the exact set of columns needed by that
    // table index.
    unordered_map<idx_t, unordered_set<idx_t>> cte_required_cols;
    // For a given table index (key), map old to new column indices.
    CTEColMap cte_col_map;
    // Make sure that CTERef nodes do not alias each other for the
    // same CTE index by storing the addresses of the CTERef nodes
    // that we've seen for each CTE index thus far.
    unordered_map<idx_t, unordered_set<void *>> cte_ref_check;
    vector<unique_ptr<BoundColumnRefExpression>> temp_column_refs;
};
// Bodo Change End

//! The RemoveUnusedColumns optimizer traverses the logical operator tree and removes any columns that are not required
class RemoveUnusedColumns : public BaseColumnPruner {
public:
    // Bodo Change: taking the reference to the new singleton outer pass
	RemoveUnusedColumns(Binder &binder, ClientContext &context, RemoveUnusedColumnsPass &ruc_pass, bool is_root = false)
	    : binder(binder), context(context), pass(ruc_pass), everything_referenced(is_root) {
	}
    // Bodo Change End

	void VisitOperator(LogicalOperator &op) override;
    // Bodo Change: second pass over the tree once we know union of CTE refs
	void CTERefVisitOperator(LogicalOperator &op);
    // Bodo Change End

private:
	Binder &binder;
	ClientContext &context;

    // Bodo Change: hold reference to the outer singleton pass
    RemoveUnusedColumnsPass &pass;
    // Bodo Change End

	//! Whether or not all the columns are referenced. This happens in the case of the root expression (because the
	//! output implicitly refers all the columns below it)
	bool everything_referenced;
private:
	template <class T>
	void ClearUnusedExpressions(vector<T> &list, idx_t table_idx, bool replace = true);
};
} // namespace duckdb
