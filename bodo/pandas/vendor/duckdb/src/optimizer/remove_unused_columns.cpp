#include "duckdb/optimizer/remove_unused_columns.hpp"

#include "duckdb/common/assert.hpp"
#include "duckdb/common/pair.hpp"
#include "duckdb/function/aggregate/distributive_functions.hpp"
#include "duckdb/function/function_binder.hpp"
#include "duckdb/parser/parsed_data/vacuum_info.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/column_binding_map.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_cteref.hpp"
#include "duckdb/planner/operator/logical_distinct.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_materialized_cte.hpp"
#include "duckdb/planner/operator/logical_order.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_set_operation.hpp"
#include "duckdb/planner/operator/logical_simple.hpp"
#include "duckdb/function/scalar/struct_utils.hpp"
#include "duckdb/function/scalar/variant_utils.hpp"
#include "duckdb/function/scalar/nested_functions.hpp"
#include <utility>

namespace duckdb {

idx_t BaseColumnPruner::ReplaceBinding(ColumnBinding current_binding, ColumnBinding new_binding) {
	auto colrefs = column_references.find(current_binding);
	if (colrefs == column_references.end()) {
		return 1;
	}

	auto &col = colrefs->second;
	idx_t created_bindings;
	if (!col.child_columns.empty() && col.supports_pushdown_extract == PushdownExtractSupport::ENABLED) {
		D_ASSERT(!col.unique_paths.empty());
		//! Pushdown extract is supported, so we are potentially creating multiple bindings, 1 for each unique extract
		//! path
		created_bindings = col.unique_paths.size();
	} else {
		//! No pushdown extract, just rewrite the existing bindings
		for (auto &colref_p : col.bindings) {
			auto &colref = colref_p.get();
			D_ASSERT(colref.binding == current_binding);
			colref.binding = new_binding;
		}
		created_bindings = 1;
	}
	return created_bindings;
}

template <class T>
void RemoveUnusedColumns::ClearUnusedExpressions(vector<T> &list, idx_t table_idx, bool replace) {
	idx_t offset = 0;
	idx_t new_col_idx = 0;
	for (idx_t col_idx = 0; col_idx < list.size(); col_idx++) {
		auto current_binding = ColumnBinding(table_idx, col_idx + offset);
		auto entry = column_references.find(current_binding);
		if (entry == column_references.end()) {
			// this entry is not referred to, erase it from the set of expressions
			list.erase_at(col_idx);
			offset++;
			col_idx--;
			continue;
		}
		if (!replace) {
			continue;
		}
		bool should_replace = false;
		if (col_idx + offset != new_col_idx) {
			should_replace = true;
		}
		if (!entry->second.child_columns.empty() &&
		    entry->second.supports_pushdown_extract == PushdownExtractSupport::ENABLED) {
			//! One or more children of this column are referenced, and pushdown-extract is enabled
			should_replace = true;
		}
		if (should_replace) {
			// column is used but the ColumnBinding has changed because of removed columns
			auto created_bindings = ReplaceBinding(current_binding, ColumnBinding(table_idx, new_col_idx));
			new_col_idx += created_bindings;
		} else {
			new_col_idx++;
		}
	}
}

// Bodo Change: overall control of singleton pass behavior. BindingRewriter changes column refs
// based on pruned column locations.  Add second pass CTERefVisitOperator that adds projections
// to CTERef nodes when needed and updates column refs based on CTE column pruning.
void RemoveUnusedColumnsPass::VisitOperator(LogicalOperator &op) {
    RemoveUnusedColumns ruc(binder, context, *this, true);
    ruc.VisitOperator(op);
    // Do fixups and add projections for CTE ref nodes.
    ruc.CTERefVisitOperator(op);
}

class BindingRewriter : public LogicalOperatorVisitor {
public:
    BindingRewriter(CTEColMap &_col_map) :
        col_map(_col_map) {}

    void Rewrite(LogicalOperator &op) {
        VisitOperatorExpressions(op);
    }

protected:
    // Called for every expression in the operator
    void VisitExpression(unique_ptr<Expression> *expr_ptr) override {
        auto &expr = **expr_ptr;
        if (expr.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
            auto &colref = expr.Cast<BoundColumnRefExpression>();
            auto table_it = col_map.find(colref.binding.table_index);
            if (table_it != col_map.end()) {
                auto &table_col_map = table_it->second;
                auto it = table_col_map.find(colref.binding.column_index);
                if (it != table_col_map.end()) {
                    colref.binding.column_index = it->second;
                } else {
					throw InternalException("BindingRewrite visit expression if table index found then column index should have been found but wasn't");
                }
            }
        }
        LogicalOperatorVisitor::VisitExpression(expr_ptr);
    }

private:
    CTEColMap &col_map;
};

void RemoveUnusedColumns::CTERefVisitOperator(LogicalOperator &op) {
    for (auto &child : op.children) {
        CTERefVisitOperator(*child);
    }
    switch (op.type) {
	case LogicalOperatorType::LOGICAL_CTE_REF: {
		auto &cteref = op.Cast<LogicalCTERef>();
        const idx_t cte_id = cteref.cte_index;
        // Get the union of columns needed from this CTE.
        auto &bucket = pass.cte_required_cols[cte_id];
        const idx_t cte_table_index = cteref.table_index;

        unordered_map<idx_t, idx_t> col_map;
        vector<LogicalType> new_chunk_types;
        vector<string> new_names;

        idx_t offset = 0;
        // Configure this CTE ref to produce the unioned set of columns from above.
        for (idx_t col_idx = 0; col_idx < cteref.bound_columns.size(); col_idx++) {
            auto current_binding = ColumnBinding(cte_table_index, col_idx + offset);
            if (bucket.count(col_idx + offset) <= 0) {
                // this entry is not referred to, erase it from the set of expressions
                cteref.bound_columns.erase_at(col_idx);
                cteref.chunk_types.erase_at(col_idx);
                offset++;
                col_idx--;
            } else if (offset > 0) {
                // column is used but the ColumnBinding has changed because of removed columns
                ColumnBinding new_binding = ColumnBinding(cte_table_index, col_idx);
                ReplaceBinding(current_binding, new_binding);
                col_map.emplace(col_idx + offset, col_idx);
            } else {
                col_map.emplace(col_idx, col_idx);
            }
        }
        // Record a column mapping for this CTE ref table index.
        pass.cte_col_map.emplace(cte_table_index, col_map);
    }
    default: {
        // For all other node types, iterate through all of its children.
        for (size_t i = 0; i < op.children.size(); ++i) {
            LogicalOperator &child = *op.children[i];
            // We have a CTE ref as a child.
            if (child.type == LogicalOperatorType::LOGICAL_CTE_REF) {
		        auto &cteref = child.Cast<LogicalCTERef>();
                const idx_t cte_id = cteref.cte_index;
                auto &bucket = pass.cte_required_cols[cte_id];
                // Save the original table_index of the CTERef.
                const idx_t cte_table_index = cteref.table_index;
                auto &actual_col_needed = pass.cte_required_cols[cte_table_index];
                // This CTERef instance uses less than than union of all columns
                // for all the CTERefs for the given CTE.
                // So, we put a projection between op (the parent) and the child CTERef that
                // projects only the columns actually needed.
                if (actual_col_needed.size() < bucket.size()) {
                    // Generate a new table index.
                    idx_t new_table_index = binder.GenerateTableIndex();
                    // We will introduce a new projection between whatever node "op" is
                    // and the CTERef.  The projection will get the original CTERef table
                    // index and the projection will have the CTERef as a child with the
                    // new table index.
                    cteref.table_index = new_table_index;
                    op.children[i]->ResolveOperatorTypes();
                    auto bindings = op.children[i]->GetColumnBindings();
                    vector<unique_ptr<Expression>> exprs;
                    exprs.reserve(actual_col_needed.size());
                    int j = 0;
                    unordered_map<idx_t, idx_t> col_map_to_use;
                    auto &cte_col_map = pass.cte_col_map[cte_table_index];
                    // Project out only the columns actually needed from this particular
                    // CTE ref.
                    for (auto &col_needed : actual_col_needed) {
                        idx_t col_needed_union = cte_col_map[col_needed];
                        exprs.push_back(make_uniq<BoundColumnRefExpression>(cteref.chunk_types[col_needed_union], bindings[col_needed_union]));
                        col_map_to_use.emplace(col_needed, j);
                        ++j;
                    }
                    auto proj = make_uniq<LogicalProjection>(cte_table_index, std::move(exprs));
                    if (child.has_estimated_cardinality) {
                        proj->SetEstimatedCardinality(child.estimated_cardinality);
                    }
                    proj->children.push_back(std::move(op.children[i]));

                    op.children[i] = std::move(proj);

                    pass.cte_col_map[new_table_index] = cte_col_map;
                    pass.cte_col_map[cte_table_index] = col_map_to_use;
                }
            }
        }
        // Update column ref expressions in this node to use the new column mappings.
        BindingRewriter rewrite(pass.cte_col_map);
        rewrite.Rewrite(op);
    }
    }
}

bool BaseColumnPruner::HasColumnReferencesForTable(idx_t table_index) const {
    for (auto &entry : column_references) {
        if (entry.first.table_index == table_index) {
            return true;
        }
    }
    return false;
}
// Bodo Change End

void RemoveUnusedColumns::VisitOperator(LogicalOperator &op) {
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: {
		// aggregate
		auto &aggr = op.Cast<LogicalAggregate>();
		if (!everything_referenced) {
			// FIXME: groups that are not referenced need to stay -> but they don't need to be scanned and output!
			ClearUnusedExpressions(aggr.expressions, aggr.aggregate_index);
			if (aggr.expressions.empty() && aggr.groups.empty()) {
				// removed all expressions from the aggregate: push a COUNT(*)
				auto count_star_fun = CountStarFun::GetFunction();
				FunctionBinder function_binder(context);
				aggr.expressions.push_back(
				    function_binder.BindAggregateFunction(count_star_fun, {}, nullptr, AggregateType::NON_DISTINCT));
			}
		}

		// then recurse into the children of the aggregate
		// Note: We allow all optimizations (join column replacement, column pruning) to run below ROLLUP
		// The duplicate groups optimizer will be responsible for not breaking ROLLUP by skipping when
		// multiple grouping sets are present
        // Bodo Change: pass "pass" to all sub-passes
		RemoveUnusedColumns remove(binder, context,pass, everything_referenced );
        // Bodo Change End
		remove.VisitOperatorExpressions(op);
		remove.VisitOperator(*op.children[0]);
		return;
	}
	case LogicalOperatorType::LOGICAL_ASOF_JOIN:
	case LogicalOperatorType::LOGICAL_DELIM_JOIN:
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
		if (everything_referenced) {
			break;
		}
		auto &comp_join = op.Cast<LogicalComparisonJoin>();

		if (comp_join.join_type != JoinType::INNER) {
			break;
		}
		// for inner joins with equality predicates in the form of (X=Y)
		// we can replace any references to the RHS (Y) to references to the LHS (X)
		// this reduces the amount of columns we need to extract from the join hash table
		// (except in the case of floating point numbers which have +0 and -0, equal but different).
		for (auto &cond : comp_join.conditions) {
			if (!cond.IsComparison() || cond.GetComparisonType() != ExpressionType::COMPARE_EQUAL) {
				continue;
			}
			if (cond.GetLHS().GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
				continue;
			}
			if (cond.GetRHS().GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
				continue;
			}
			if (cond.GetLHS().Cast<BoundColumnRefExpression>().return_type.IsFloating()) {
				continue;
			}
			if (cond.GetRHS().Cast<BoundColumnRefExpression>().return_type.IsFloating()) {
				continue;
			}
			// comparison join between two bound column refs
			// we can replace any reference to the RHS (build-side) with a reference to the LHS (probe-side)
			auto &lhs_col = cond.GetLHS().Cast<BoundColumnRefExpression>();
			auto &rhs_col = cond.GetRHS().Cast<BoundColumnRefExpression>();
			// if there are any columns that refer to the RHS,
			auto colrefs = column_references.find(rhs_col.binding);
			if (colrefs == column_references.end()) {
				continue;
			}
			for (auto &entry : colrefs->second.bindings) {
				auto &colref = entry.get();
				colref.binding = lhs_col.binding;
				AddBinding(colref);
			}
			column_references.erase(rhs_col.binding);
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_ANY_JOIN:
		break;
	case LogicalOperatorType::LOGICAL_UNION: {
		auto &setop = op.Cast<LogicalSetOperation>();
		if (setop.setop_all && !everything_referenced) {
			// for UNION we can remove unreferenced columns if union all is used
			// it's possible not all columns are referenced, but unreferenced columns in the union can
			// still have an affect on the result of the union
			vector<idx_t> entries;
			for (idx_t i = 0; i < setop.column_count; i++) {
				entries.push_back(i);
			}
			ClearUnusedExpressions(entries, setop.table_index);
			if (entries.size() >= setop.column_count) {
				return;
			}
			if (entries.empty()) {
				// no columns referenced: this happens in the case of a COUNT(*)
				// extract the first column
				entries.push_back(0);
			}
			// columns were cleared
			setop.column_count = entries.size();

			for (idx_t child_idx = 0; child_idx < op.children.size(); child_idx++) {
                // Bodo Change: pass "pass" to all sub-passes
				RemoveUnusedColumns remove(binder, context, pass, true);
                // Bodo Change End

				auto &child = op.children[child_idx];

				// we push a projection under this child that references the required columns of the union
				child->ResolveOperatorTypes();
				auto bindings = child->GetColumnBindings();
				vector<unique_ptr<Expression>> expressions;
				expressions.reserve(entries.size());
				for (auto &column_idx : entries) {
					expressions.push_back(
					    make_uniq<BoundColumnRefExpression>(child->types[column_idx], bindings[column_idx]));
				}
				auto new_projection = make_uniq<LogicalProjection>(binder.GenerateTableIndex(), std::move(expressions));
				if (child->has_estimated_cardinality) {
					new_projection->SetEstimatedCardinality(child->estimated_cardinality);
				}
				new_projection->children.push_back(std::move(child));
				op.children[child_idx] = std::move(new_projection);

				remove.VisitOperator(*op.children[child_idx]);
			}
			return;
		}
		for (auto &child : op.children) {
			RemoveUnusedColumns remove(binder, context, pass, true);
			remove.VisitOperator(*child);
		}
		return;
	}
	case LogicalOperatorType::LOGICAL_EXCEPT:
	case LogicalOperatorType::LOGICAL_INTERSECT: {
		// for INTERSECT/EXCEPT operations we can't remove anything, just recursively visit the children
		for (auto &child : op.children) {
            // Bodo Change: pass "pass" to all sub-passes
			RemoveUnusedColumns remove(binder, context, pass, true);
            // Bodo Change End
			remove.VisitOperator(*child);
		}
		return;
	}
	case LogicalOperatorType::LOGICAL_PROJECTION: {
		if (!everything_referenced) {
			auto &proj = op.Cast<LogicalProjection>();
			CheckPushdownExtract(op);
			auto old_expression_count = proj.expressions.size();
			ClearUnusedExpressions(proj.expressions, proj.table_index);
			RewriteExpressions(proj, old_expression_count);

			if (proj.expressions.empty()) {
				// nothing references the projected expressions
				// this happens in the case of e.g. EXISTS(SELECT * FROM ...)
				// in this case we only need to project a single constant
				proj.expressions.push_back(make_uniq<BoundConstantExpression>(Value::INTEGER(42)));
			}
		}
		// then recurse into the children of this projection
		RemoveUnusedColumns remove(binder, context, pass);
		remove.VisitOperatorExpressions(op);
		remove.VisitOperator(*op.children[0]);
		return;
	}
	case LogicalOperatorType::LOGICAL_INSERT:
	case LogicalOperatorType::LOGICAL_UPDATE:
	case LogicalOperatorType::LOGICAL_DELETE:
	case LogicalOperatorType::LOGICAL_MERGE_INTO: {
		//! When RETURNING is used, a PROJECTION is the top level operator for INSERTS, UPDATES, and DELETES
		//! We still need to project all values from these operators so the projection
		//! on top of them can select from only the table values being inserted.
		//! TODO: Push down the projections from the returning statement
		//! TODO: Be careful because you might be adding expressions when a user returns *
        // Bodo Change: pass "pass" to all sub-passes
		RemoveUnusedColumns remove(binder, context, pass, true);
        // Bodo Change End
		remove.VisitOperatorExpressions(op);
		remove.VisitOperator(*op.children[0]);
		return;
	}
	case LogicalOperatorType::LOGICAL_GET: {
		LogicalOperatorVisitor::VisitOperatorExpressions(op);
		auto &get = op.Cast<LogicalGet>();
		RemoveColumnsFromLogicalGet(get);
		if (!op.children.empty()) {
			// Some LOGICAL_GET operators (e.g., table in out functions) may have a
			// child operator. So we recurse into it if it exists.
			RemoveUnusedColumns remove(binder, context, pass, true);
			remove.VisitOperator(*op.children[0]);
		}
		return;
	}
	case LogicalOperatorType::LOGICAL_DISTINCT: {
		auto &distinct = op.Cast<LogicalDistinct>();
		if (distinct.distinct_type == DistinctType::DISTINCT_ON) {
			// distinct type references columns that need to be distinct on, so no
			// need to implicity reference everything.
			break;
		}
		// distinct, all projected columns are used for the DISTINCT computation
		// mark all columns as used and continue to the children
		// FIXME: DISTINCT with expression list does not implicitly reference everything
		everything_referenced = true;
		break;
	}
	case LogicalOperatorType::LOGICAL_COPY_TO_FILE:
	case LogicalOperatorType::LOGICAL_RECURSIVE_CTE: {
		everything_referenced = true;
		break;
	}
    // Bodo Change: enabling column pruning of basic CTE nodes
	case LogicalOperatorType::LOGICAL_MATERIALIZED_CTE: {
        /*
         * Here is a summary of how column pruning in the presence of CTEs
         * works. Previously remove_unused_columns did one traversal of the
         * plan tree but CTE column pruning requires two.  In the first pass,
         * the normal column pruning code encounters CTE ref nodes and knows
         * which columns are needed from each of those nodes.  The
         * LOGICAL_CTE_REF case below handles taking those needed columns and
         * unioning them with the columns needed by other CTE ref nodes for
         * the same CTE index. This union is the complete set of columns that
         * the CTE production side of the CTE needs to produce.  To re-use
         * the existing code, below we push that unioned set of required
         * columns down into the root node of the CTE production side of this
         * CTE node and then the normal code handles column pruning for that
         * entire sub-plan tree.
         *
         * In the second pass (CTERefVisitOperator), we do a post-order
         * traversal of the plan tree and do the following:
         * 1) When we encounter a CTE ref node, we can now prune its set of
         * columns back to the completed union of columns for the given CTE
         * index.
         * 2) For all other nodes, we first check if any of their children
         * are CTE ref nodes.  If so, and the columns needed for this CTE
         * ref is a strict subset of union of columns then we insert a
         * projection to remove the unneeded columns at this point.
         * 3) Steps 1, 2, and 3 maintain a data structure that says how
         * columns have been pruned and the mapping between old and new
         * column numbers for each table index.  For every node in the plan,
         * this step sees if it uses any column from any table index that
         * has had pruning done on it and if so updates its own sets of
         * column mappings for its outputs and rewrites any expressions
         * internal to the node that uses remapped column indices from its
         * inputs.
         *
         * For the outermost CTE, duckdb is already configured to assume
         * that the "using" side of the CTE requires all the columns
         * referenced in the root node of the using side.  However, for
         * nested CTEs, the root node of the using side of the nested CTE
         * should use the above union of required columns.  The
         * HasColumnReferencesForTable call below checks if the CTE node
         * table index has any recorded column prunings.  This will be
         * true now iff it is a nested CTE.  If it is a nested CTE, then
         * propagate the CTE's required columns to the root node of the using
         * side to initialize its set of required columns.
         */
		auto &cte = op.Cast<LogicalMaterializedCTE>();
        const idx_t cte_id = cte.table_index;

        // ----------------- CTE use side of CTE node --------------------
        auto &use_cte_op = *(cte.children[1]);
        // We only be true for now if-and-only-if in nested CTE.
        if (HasColumnReferencesForTable(cte_id)) {
            use_cte_op.ResolveOperatorTypes();

            vector<idx_t> use_cte_root_table_indices = use_cte_op.GetTableIndex();
            idx_t use_cte_root_table_index;
            if (use_cte_root_table_indices.size() != 1) {
                throw InternalException("child 1 don't yet know how to handle to case where CTE root node has multiple table indices! " + std::to_string(static_cast<int>(use_cte_op.type)));
            }

            // Prepare the CTE side for processing by setting the columns of the root
            // of the CTE equal to the union of all columns needed by all the CTERefs.
            use_cte_root_table_index = use_cte_root_table_indices[0];

            int count_col_found = 0;
            for (idx_t i = 0; i < use_cte_op.types.size(); ++i) {
                ColumnBinding cte_binding(cte_id, i);
                if (column_references.find(cte_binding) != column_references.end()) {
                    ColumnBinding child_binding(use_cte_root_table_index, i);
                    pass.temp_column_refs.push_back(make_uniq<BoundColumnRefExpression>(use_cte_op.types[i], child_binding));
                    AddBinding(*(pass.temp_column_refs.back()));
                    ++count_col_found;
                }
            }
        }

        // Process unique (non-CTE) side of the CTE.
        VisitOperator(use_cte_op);

        // ----------------- CTE side of CTE node ------------------------

        // Find the info on the union of the columns required by all the CTERef
        // with this cte index.
        auto cte_it = pass.cte_required_cols.find(cte_id);
        if (cte_it == pass.cte_required_cols.end()) {
	        throw InternalException("Should always find CTERef info!");
        }
        RemoveUnusedColumns cte_side(binder, context, pass);
        // Get the root of the CTE production side of this CTE node.
        auto &cte_op = *(cte.children[0]);
        cte_op.ResolveOperatorTypes();
        idx_t cte_root_table_index;
        // Propagate the required set of columns down to this root node.
        if (cte_op.type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY) {
			auto &aggr = cte_op.Cast<LogicalAggregate>();
            // For each required column index, figure out which binding domain it belongs to
            for (auto col_idx : cte_it->second) {
                ColumnBinding binding;
                if (col_idx < aggr.groups.size()) {
                    // This is a group column
                    binding = ColumnBinding(aggr.group_index, col_idx);
                } else {
                    // This is an aggregate expression
                    idx_t agg_col = col_idx - aggr.groups.size();
                    binding = ColumnBinding(aggr.aggregate_index, agg_col);
                }

                pass.temp_column_refs.push_back(make_uniq<BoundColumnRefExpression>(cte_op.types[col_idx], binding));
                cte_side.AddBinding(*(pass.temp_column_refs.back()));
            }
        } else {
            vector<idx_t> cte_root_table_indices = cte_op.GetTableIndex();
            if (cte_root_table_indices.size() != 1) {
                throw InternalException("child 0 don't yet know how to handle to case where CTE root node has multiple table indices!" + std::to_string(static_cast<int>(cte_op.type)));
            }
            // Prepare the CTE side for processing by setting the columns of the root
            // of the CTE equal to the union of all columns needed by all the CTERefs.
            cte_root_table_index = cte_root_table_indices[0];
            // For each column needed by a CTERef.
            for (auto col_idx : cte_it->second) {
                ColumnBinding binding(cte_root_table_index, col_idx);
                pass.temp_column_refs.push_back(make_uniq<BoundColumnRefExpression>(cte_op.types[col_idx], binding));
                cte_side.AddBinding(*(pass.temp_column_refs.back()));
            }
        }
        // Now process the duplicated (CTE) side of the CTE.
        cte_side.VisitOperator(cte_op);
        return;
	}
	case LogicalOperatorType::LOGICAL_CTE_REF: {
		auto &cteref = op.Cast<LogicalCTERef>();
        const idx_t cte_id = cteref.cte_index;          // stable identifier for the CTE
        const idx_t cte_table_index = cteref.table_index;

        if (pass.cte_ref_check.find(cte_id) != pass.cte_ref_check.end()) {
            pass.cte_ref_check[cte_id] = unordered_set<void*>();
        }
        unordered_set<void*> &ptr_set = pass.cte_ref_check[cte_id];

        if (ptr_set.find(&cteref) != ptr_set.end()) {
	        throw InternalException("CTERef nodes should not alias!");
        }

        // Look through column_references for bindings that belong to this CTERef
        unordered_set<idx_t> required_here;
        // Sanity check.
        if (cteref.chunk_types.size() != cteref.bound_columns.size()) {
	        throw InternalException("Size of chunk_types and colnames don't match.");
        }
        // For all columns in this CTERef.
        for (idx_t i = 0; i < cteref.bound_columns.size(); i++) {
            ColumnBinding binding(cte_table_index, i);
            // See if the column is in the column needed data structure (column_references).
            if (column_references.find(binding) != column_references.end()) {
                required_here.insert(i);
            }
        }

        // If nothing matched, conservatively keep all
        if (required_here.empty()) {
            for (idx_t i = 0; i < cteref.chunk_types.size(); i++) {
                required_here.insert(i);
            }
        }

        pass.cte_required_cols[cte_table_index] = required_here;

        // Union into global per-CTE requirement
        auto &bucket = pass.cte_required_cols[cte_id];
        bucket.insert(required_here.begin(), required_here.end());
		break;
	}
    // Bodo Change End
	case LogicalOperatorType::LOGICAL_PIVOT: {
		everything_referenced = true;
		break;
	}
	default:
		break;
	}
    LogicalOperatorVisitor::VisitOperatorExpressions(op);
    LogicalOperatorVisitor::VisitOperatorChildren(op);

	if (op.type == LogicalOperatorType::LOGICAL_ASOF_JOIN || op.type == LogicalOperatorType::LOGICAL_DELIM_JOIN ||
	    op.type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
		auto &comp_join = op.Cast<LogicalComparisonJoin>();
		// after removing duplicate columns we may have duplicate join conditions (if the join graph is cyclical)
		vector<JoinCondition> unique_conditions;
		for (auto &cond : comp_join.conditions) {
			bool found = false;
			for (auto &unique_cond : unique_conditions) {
				if (cond.IsComparison() && unique_cond.IsComparison()) {
					if (cond.GetComparisonType() == unique_cond.GetComparisonType() &&
					    cond.GetLHS().Equals(unique_cond.GetLHS()) && cond.GetRHS().Equals(unique_cond.GetRHS())) {
						found = true;
						break;
					}
				} else if (!cond.IsComparison() && !unique_cond.IsComparison()) {
					if (cond.GetJoinExpression().Equals(unique_cond.GetJoinExpression())) {
						found = true;
						break;
					}
				}
			}
			if (!found) {
				unique_conditions.push_back(std::move(cond));
			}
		}
		comp_join.conditions = std::move(unique_conditions);
	}
}

static idx_t GetColumnIdsIndexForFilter(vector<ColumnIndex> &column_ids, idx_t filter_idx) {
	// Find the index in the column_ids that contains the column referenced by the filter
	auto it = std::find_if(column_ids.begin(), column_ids.end(), [&filter_idx](const ColumnIndex &column_index) {
		return column_index.GetPrimaryIndex() == filter_idx;
	});
	if (it == column_ids.end()) {
		throw InternalException("Could not find column index for table filter");
	}
	return static_cast<idx_t>(std::distance(column_ids.begin(), it));
}

//! returns: found_path, depth of the found path
std::pair<column_index_set::iterator, idx_t> FindShortestMatchingPath(column_index_set &all_paths,
                                                                      const ColumnIndex &full_path) {
	idx_t depth = 0;
	column_index_set::iterator entry;

	ColumnIndex copy;
	if (full_path.HasPrimaryIndex()) {
		copy = ColumnIndex(full_path.GetPrimaryIndex());
	} else {
		copy = ColumnIndex(full_path.GetFieldName());
	}

	reference<const ColumnIndex> path_iter(full_path);
	reference<ColumnIndex> copy_iter(copy);
	while (true) {
		if (path_iter.get().HasType()) {
			copy_iter.get().SetType(path_iter.get().GetType());
		}
		entry = all_paths.find(copy);
		if (entry != all_paths.end()) {
			//! Path found, we're done
			return make_pair(entry, depth);
		}
		if (!path_iter.get().HasChildren()) {
			break;
		}
		path_iter = path_iter.get().GetChildIndex(0);

		ColumnIndex new_child;
		if (path_iter.get().HasPrimaryIndex()) {
			new_child = ColumnIndex(path_iter.get().GetPrimaryIndex());
		} else {
			new_child = ColumnIndex(path_iter.get().GetFieldName());
		}

		copy_iter.get().AddChildIndex(new_child);
		copy_iter = copy_iter.get().GetChildIndex(0);
		depth++;
	}
	return make_pair(all_paths.end(), depth);
}

void RemoveUnusedColumns::WritePushdownExtractColumns(
    const ColumnBinding &binding, ReferencedColumn &col, idx_t original_idx, const LogicalType &column_type,
    const std::function<idx_t(const ColumnIndex &extract_path, optional_ptr<const LogicalType> cast_type)> &callback) {
	//! For each struct extract, replace the expression with a BoundColumnRefExpression
	//! The expression references a binding created for the extracted path, 1 per unique path
	for (auto &struct_extract : col.struct_extracts) {
		//! Replace the struct extract expression at the right depth with a BoundColumnRefExpression
		auto &full_path = struct_extract.extract_path;

		auto res = FindShortestMatchingPath(col.unique_paths, full_path);
		auto entry = res.first;
		auto depth = res.second;
		if (entry == col.unique_paths.end()) {
			throw InternalException("This path wasn't found in the registered paths for this expression at all!?");
		}
		D_ASSERT(struct_extract.components.size() > depth);
		auto &component = struct_extract.components[depth];
		auto &expr = component.cast ? *component.cast : component.extract;

		optional_ptr<LogicalType> cast_type;
		auto return_type = expr->return_type;

		auto &colref = col.bindings[struct_extract.bindings_idx];
		auto colref_copy = colref.get().Copy();
		expr = std::move(colref_copy);
		auto &new_expr = expr->Cast<BoundColumnRefExpression>();
		new_expr.return_type = return_type;

		auto column_index = callback(*entry, component.cast ? &(*component.cast)->return_type : nullptr);
		new_expr.binding.column_index = column_index;
	}
}

static unique_ptr<Expression> ConstructStructExtractFromPath(ClientContext &context, unique_ptr<Expression> target,
                                                             const ColumnIndex &path) {
	auto extract_function = GetKeyExtractFunction();
	auto bind_callback = extract_function.GetBindCallback();

	auto &struct_type = target->return_type;
	D_ASSERT(struct_type.id() == LogicalTypeId::STRUCT);
	reference<const LogicalType> type_iter(struct_type);
	reference<const ColumnIndex> path_iter(path);
	while (true) {
		auto child_index = path_iter.get().GetPrimaryIndex();
		auto &child_types = StructType::GetChildTypes(type_iter.get());
		D_ASSERT(child_index < child_types.size());
		auto &key = child_types[child_index].first;
		type_iter = child_types[child_index].second;

		auto function = extract_function;
		vector<unique_ptr<Expression>> arguments(2);
		arguments[0] = (std::move(target));
		arguments[1] = (make_uniq<BoundConstantExpression>(Value(key)));
		auto bind_info = bind_callback(context, function, arguments);
		auto return_type = function.GetReturnType();
		target = make_uniq<BoundFunctionExpression>(return_type, std::move(function), std::move(arguments),
		                                            std::move(bind_info));
		if (!path_iter.get().HasChildren()) {
			break;
		}
		path_iter = path_iter.get().GetChildIndex(0);
	}
	return target;
}

void RemoveUnusedColumns::RewriteExpressions(LogicalProjection &proj, idx_t expression_count) {
	vector<unique_ptr<Expression>> expressions;
	auto &context = this->context;

	column_index_map<idx_t> new_bindings;
	idx_t expression_idx = 0;
	for (idx_t i = 0; i < expression_count; i++) {
		auto binding = ColumnBinding(proj.table_index, i);
		auto entry = column_references.find(binding);
		if (entry == column_references.end()) {
			//! Already removed by the call to ClearUnusedExpressions
			continue;
		}
		if (entry->second.child_columns.empty() ||
		    entry->second.supports_pushdown_extract != PushdownExtractSupport::ENABLED) {
			expressions.push_back(std::move(proj.expressions[expression_idx++]));
			continue;
		}
		auto &expr = *proj.expressions[expression_idx++];
		auto &colref = expr.Cast<BoundColumnRefExpression>();
		auto original_binding = colref.binding;
		auto &column_type = expr.return_type;
		idx_t start = expressions.size();
		//! Pushdown Extract is supported, emit a column for every field
		WritePushdownExtractColumns(
		    entry->first, entry->second, i, column_type,
		    [&](const ColumnIndex &extract_path, optional_ptr<const LogicalType> cast_type) -> idx_t {
			    auto target = make_uniq<BoundColumnRefExpression>(column_type, original_binding);
			    target->SetAlias(expr.GetAlias());
			    auto new_extract = ConstructStructExtractFromPath(context, std::move(target), extract_path);
			    if (cast_type) {
				    auto cast = BoundCastExpression::AddCastToType(context, std::move(new_extract), *cast_type);
				    new_extract = std::move(cast);
			    }
			    ColumnIndex full_path(i);
			    full_path.AddChildIndex(extract_path);
			    auto it = new_bindings.emplace(full_path, expressions.size()).first;
			    if (it->second == expressions.size()) {
				    expressions.push_back(std::move(new_extract));
			    }
			    return it->second;
		    });
		for (; start < expressions.size(); start++) {
			VisitExpression(&expressions[start]);
		}
	}
	proj.expressions = std::move(expressions);
}

void RemoveUnusedColumns::CheckPushdownExtract(LogicalOperator &op) {
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_GET: {
		auto &get = op.Cast<LogicalGet>();
		//! For all referenced struct fields, check if the scan supports pushing down the extract
		auto &column_ids = get.GetColumnIds();
		for (idx_t i = 0; i < column_ids.size(); i++) {
			auto entry = column_references.find(ColumnBinding(get.table_index, i));
			if (entry == column_references.end()) {
				//! Binding is not referenced, skip
				continue;
			}
			auto &col = entry->second;
			if (col.struct_extracts.empty()) {
				//! Either not a struct, or we're not using struct field projection pushdown - skip it
				continue;
			}
			if (col.supports_pushdown_extract == PushdownExtractSupport::DISABLED) {
				//! We're already not using pushdown extract for this column, no need to check with the scan
				continue;
			}
			auto logical_column_index = LogicalIndex(column_ids[i].GetPrimaryIndex());
			if (!get.function.supports_pushdown_extract || get.function.statistics) {
				//! Either 'statistics_extended' needs to be set or 'statistics' needs to be NULL
				col.supports_pushdown_extract = PushdownExtractSupport::DISABLED;
				continue;
			}
			D_ASSERT(get.bind_data);
			if (get.function.supports_pushdown_extract(*get.bind_data, logical_column_index)) {
				col.supports_pushdown_extract = PushdownExtractSupport::ENABLED;
			} else {
				col.supports_pushdown_extract = PushdownExtractSupport::DISABLED;
			}
		}
		return;
	}
	case LogicalOperatorType::LOGICAL_PROJECTION: {
		auto &proj = op.Cast<LogicalProjection>();
		for (idx_t idx = 0; idx < proj.expressions.size(); idx++) {
			auto &expr = *proj.expressions[idx];
			auto record = column_references.find(ColumnBinding(proj.table_index, idx));
			if (record == column_references.end()) {
				//! Not referenced, skip
				continue;
			}
			auto &col = record->second;
			auto &child_columns = col.child_columns;
			if (child_columns.empty()) {
				//! No children of this column are referenced, skip
				continue;
			}
			if (expr.type != ExpressionType::BOUND_COLUMN_REF) {
				//! Not a column reference, can't pull up the extract
				continue;
			}
			if (expr.return_type.id() != LogicalTypeId::STRUCT) {
				//! Extract pull up only supported for STRUCT currently
				continue;
			}
			if (col.supports_pushdown_extract == PushdownExtractSupport::DISABLED) {
				//! Already explicitly disabled, don't need to check
				continue;
			}
			col.supports_pushdown_extract = PushdownExtractSupport::ENABLED;
		}
		return;
	}
	default:
		throw InternalException("CheckPushdownExtract not supported for LogicalOperatorType::%s",
		                        EnumUtil::ToString(op.type));
	}
}

void RemoveUnusedColumns::RemoveColumnsFromLogicalGet(LogicalGet &get) {
	if (everything_referenced) {
		return;
	}
	if (!get.function.projection_pushdown) {
		return;
	}

	//! The existing column ids
	auto old_column_ids = get.GetColumnIds();
	//! The newly written column ids (same size as 'original_ids')
	vector<ColumnIndex> new_column_ids;
	//! The old index that the new one is based on (same size as 'new_column_ids')
	vector<idx_t> original_ids;
	//! Map of column index to binding, for pushed down struct extracts
	column_index_map<idx_t> child_map;
	//! Created bindings per original_column (for pushdown extract verification)
	unordered_map<idx_t, idx_t> created_bindings;

	// Create "selection vector" that contains all indices of the old 'column_ids' of the LogicalGet
	//! i.e: This contains all the columns of the table
	vector<idx_t> proj_sel;
	for (idx_t col_idx = 0; col_idx < old_column_ids.size(); col_idx++) {
		proj_sel.push_back(col_idx);
	}
	// Create a copy so we can later check the difference between these two:
	//! 1. The set of filtered ids containing only the columns referenced by the projection expressions (proj_sel)
	//! 2. The set of filtered ids containing columns referenced by either the projection or the filter expressions
	//! (col_sel)
	auto col_sel = proj_sel;
	// Clear unused ids, exclude filter columns that are projected out immediately
	ClearUnusedExpressions(proj_sel, get.table_index, false);

	//! FIXME: pushdown extract is disabled when a struct field is referenced by a filter,
	//! because that would involve rewriting the existing TableFilterSet
	SetMode(BaseColumnPrunerMode::DISABLE_PUSHDOWN_EXTRACT);

	// for every table filter, push a column binding into the column references map to prevent the column from
	// being projected out

	//! NOTE: This vector is required to keep the referenced Expressions alive
	vector<unique_ptr<Expression>> filter_expressions;
	for (auto &filter : get.table_filters.filters) {
		auto index = GetColumnIdsIndexForFilter(old_column_ids, filter.first);
		auto column_type = get.GetColumnType(ColumnIndex(filter.first));

		ColumnBinding filter_binding(get.table_index, index);
		auto column_ref = make_uniq<BoundColumnRefExpression>(std::move(column_type), filter_binding);
		//! Convert the filter to an expression, so we can visit it
		auto filter_expr = filter.second->ToExpression(*column_ref);
		if (filter_expr->IsScalar()) {
			filter_expr = std::move(column_ref);
		}
		filter_expressions.push_back(std::move(filter_expr));
		//! Now visit the filter to add to the 'column_references'
		VisitExpression(&filter_expressions.back());
	}

	//! Check with the LogicalGet whether pushdown-extract is supported
	CheckPushdownExtract(get);

	// Clear unused ids, include filter columns that are projected out immediately
	ClearUnusedExpressions(col_sel, get.table_index);

	// Now set the column ids in the LogicalGet using the "selection vector"
	for (auto &col_sel_idx : col_sel) {
		auto &column_type = get.GetColumnType(old_column_ids[col_sel_idx]);
		auto entry = column_references.find(ColumnBinding(get.table_index, col_sel_idx));
		if (entry == column_references.end()) {
			throw InternalException("RemoveUnusedColumns - could not find referenced column");
		}
		if (entry->second.child_columns.empty() ||
		    entry->second.supports_pushdown_extract != PushdownExtractSupport::ENABLED) {
			auto &logical_column_id = old_column_ids[col_sel_idx];

			original_ids.emplace_back(col_sel_idx);
			if (logical_column_id.IsPushdownExtract()) {
				//! RemoveUnusedColumns is also used by other optimizers,
				//! so we have to deal with this case and preserve the PushdownExtract we created earlier
				D_ASSERT(entry->second.child_columns.empty());
				new_column_ids.emplace_back(logical_column_id);
			} else {
				ColumnIndex new_index(logical_column_id.GetPrimaryIndex(), entry->second.child_columns);
				new_column_ids.emplace_back(std::move(new_index));
			}
			continue;
		}
		auto struct_column_index = old_column_ids[col_sel_idx].GetPrimaryIndex();

		//! Pushdown Extract is supported, emit a column for every field
		WritePushdownExtractColumns(
		    entry->first, entry->second, col_sel_idx, column_type,
		    [&](const ColumnIndex &extract_path, optional_ptr<const LogicalType> cast_type) -> idx_t {
			    ColumnIndex new_index(struct_column_index, {extract_path});
			    new_index.SetPushdownExtractType(column_type, cast_type);

			    auto column_binding_index = new_column_ids.size();
			    auto entry = child_map.find(new_index);
			    if (entry == child_map.end()) {
				    //! Adds the binding for the child only if it doesn't exist yet
				    entry = child_map.emplace(new_index, column_binding_index).first;
				    created_bindings[new_index.GetPrimaryIndex()]++;

				    new_column_ids.emplace_back(std::move(new_index));
				    original_ids.emplace_back(col_sel_idx);
			    }
			    return entry->second;
		    });
	}
	if (new_column_ids.empty()) {
		// this generally means we are only interested in whether or not anything exists in the table (e.g.
		// EXISTS(SELECT * FROM tbl)) in this case, we just scan the row identifier column as it means we do not
		// need to read any of the columns
		auto any_column = get.GetAnyColumn();
		original_ids.emplace_back(any_column);
		new_column_ids.emplace_back(any_column);
	}
	get.SetColumnIds(std::move(new_column_ids));

	if (!get.function.filter_prune) {
		return;
	}
	// Now set the projection cols by matching the "selection vector" that excludes filter columns
	// with the "selection vector" that includes filter columns
	idx_t col_idx = 0;
	get.projection_ids.clear();
	vector<idx_t> filtered_original_ids;
	//! Find matching indices between the proj_sel and the col_sel
	for (auto to_keep : proj_sel) {
		for (; col_idx < col_sel.size(); col_idx++) {
			if (to_keep == col_sel[col_idx]) {
				filtered_original_ids.push_back(to_keep);
				break;
			}
		}
	}
	col_idx = 0;
	for (auto col : filtered_original_ids) {
		for (; col_idx < original_ids.size(); col_idx++) {
			if (original_ids[col_idx] == col) {
				get.projection_ids.push_back(col_idx);
			} else if (original_ids[col_idx] > col) {
				break;
			}
		}
	}
}

void BaseColumnPruner::SetMode(BaseColumnPrunerMode mode) {
	this->mode = mode;
}

BaseColumnPrunerMode BaseColumnPruner::GetMode() const {
	return mode;
}

bool BaseColumnPruner::HandleStructExtract(unique_ptr<Expression> &expr_p,
                                           optional_ptr<BoundColumnRefExpression> &colref,
                                           reference<ColumnIndex> &path_ref,
                                           vector<ReferencedExtractComponent> &expressions) {
	auto &function = expr_p->Cast<BoundFunctionExpression>();
	auto &child = function.children[0];
	D_ASSERT(child->return_type.id() == LogicalTypeId::STRUCT);
	auto &bind_data = function.bind_info->Cast<StructExtractBindData>();
	// struct extract, check if left child is a bound column ref
	if (child->GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
		// column reference - check if it is a struct
		auto &ref = child->Cast<BoundColumnRefExpression>();
		if (ref.return_type.id() != LogicalTypeId::STRUCT) {
			return false;
		}
		colref = &ref;
		auto &path = path_ref.get();
		path.AddChildIndex(ColumnIndex(bind_data.index));
		path_ref = path.GetChildIndex(0);
		expressions.emplace_back(expr_p);
		return true;
	}
	// not a column reference - try to handle this recursively
	if (!HandleExtractRecursive(child, colref, path_ref, expressions)) {
		return false;
	}
	auto &path = path_ref.get();
	path.AddChildIndex(ColumnIndex(bind_data.index));
	path_ref = path.GetChildIndex(0);

	expressions.emplace_back(expr_p);
	return true;
}

bool BaseColumnPruner::HandleVariantExtract(unique_ptr<Expression> &expr_p,
                                            optional_ptr<BoundColumnRefExpression> &colref,
                                            reference<ColumnIndex> &path_ref,
                                            vector<ReferencedExtractComponent> &expressions) {
	auto &function = expr_p->Cast<BoundFunctionExpression>();
	auto &child = function.children[0];
	D_ASSERT(child->return_type.id() == LogicalTypeId::VARIANT);
	auto &bind_data = function.bind_info->Cast<VariantExtractBindData>();
	if (bind_data.component.lookup_mode != VariantChildLookupMode::BY_KEY) {
		//! We don't push down variant extract on ARRAY values
		return false;
	}
	// variant extract, check if left child is a bound column ref
	if (child->GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
		// column reference - check if it is a variant
		auto &ref = child->Cast<BoundColumnRefExpression>();
		if (ref.return_type.id() != LogicalTypeId::VARIANT) {
			return false;
		}
		colref = &ref;

		auto &path = path_ref.get();
		path.AddChildIndex(ColumnIndex(bind_data.component.key));
		path_ref = path.GetChildIndex(0);

		expressions.emplace_back(expr_p);
		return true;
	}
	// not a column reference - try to handle this recursively
	if (!HandleExtractRecursive(child, colref, path_ref, expressions)) {
		return false;
	}

	auto &path = path_ref.get();
	path.AddChildIndex(ColumnIndex(bind_data.component.key));
	path_ref = path.GetChildIndex(0);

	expressions.emplace_back(expr_p);
	return true;
}

bool BaseColumnPruner::HandleExtractRecursive(unique_ptr<Expression> &expr_p,
                                              optional_ptr<BoundColumnRefExpression> &colref,
                                              reference<ColumnIndex> &path_ref,
                                              vector<ReferencedExtractComponent> &expressions) {
	auto &expr = *expr_p;
	if (expr.GetExpressionClass() != ExpressionClass::BOUND_FUNCTION) {
		return false;
	}
	auto &function = expr.Cast<BoundFunctionExpression>();
	if (function.function.name != "struct_extract_at" && function.function.name != "struct_extract" &&
	    function.function.name != "array_extract" && function.function.name != "variant_extract") {
		return false;
	}
	if (!function.bind_info) {
		return false;
	}
	auto &child = function.children[0];
	auto child_type = child->return_type.id();
	switch (child_type) {
	case LogicalTypeId::STRUCT:
		return HandleStructExtract(expr_p, colref, path_ref, expressions);
	case LogicalTypeId::VARIANT:
		return HandleVariantExtract(expr_p, colref, path_ref, expressions);
	default:
		return false;
	}
}

bool BaseColumnPruner::HandleExtractExpression(unique_ptr<Expression> *expression,
                                               optional_ptr<unique_ptr<Expression>> cast_expression) {
	optional_ptr<BoundColumnRefExpression> colref;
	vector<ReferencedExtractComponent> expressions;

	ColumnIndex path(0);
	reference<ColumnIndex> path_ref(path);
	if (!HandleExtractRecursive(*expression, colref, path_ref, expressions)) {
		return false;
	}
	if (cast_expression) {
		auto &top_level = expressions.back();
		top_level.cast = cast_expression;
		path_ref.get().SetType((*cast_expression)->return_type);
	}

	AddBinding(*colref, path.GetChildIndex(0), expressions);
	return true;
}

void BaseColumnPruner::MergeChildColumns(vector<ColumnIndex> &current_child_columns, ColumnIndex &new_child_column) {
	if (current_child_columns.empty()) {
		// there's already a reference to the full column - we can't extract only a subfield
		// skip struct projection pushdown
		return;
	}
	// if we are already extract sub-fields, add it (if it is not there yet)
	for (auto &binding : current_child_columns) {
		if (binding.HasPrimaryIndex()) {
			if (!new_child_column.HasPrimaryIndex()) {
				continue;
			}
			if (binding.GetPrimaryIndex() != new_child_column.GetPrimaryIndex()) {
				continue;
			}
		} else {
			if (new_child_column.HasPrimaryIndex()) {
				continue;
			}
			if (binding.GetFieldName() != new_child_column.GetFieldName()) {
				continue;
			}
		}
		// found a match: sub-field is already projected
		// check if we have child columns
		auto &nested_child_columns = binding.GetChildIndexesMutable();
		if (!new_child_column.HasChildren()) {
			// new child is a reference to a full column - clear any existing bindings (if any)
			nested_child_columns.clear();
		} else {
			// new child has a sub-reference - merge recursively
			D_ASSERT(new_child_column.ChildIndexCount() == 1);
			MergeChildColumns(nested_child_columns, new_child_column.GetChildIndex(0));
		}
		return;
	}
	// this child column is not projected yet - add it in
	current_child_columns.push_back(std::move(new_child_column));
}

void BaseColumnPruner::AddBinding(BoundColumnRefExpression &col, ColumnIndex child_column) {
	auto entry = column_references.find(col.binding);
	if (entry == column_references.end()) {
		// column not referenced yet - add a binding to it entirely
		ReferencedColumn column;
		column.bindings.push_back(col);
		column.child_columns.push_back(std::move(child_column));
		entry = column_references.emplace(make_pair(col.binding, std::move(column))).first;
	} else {
		// column reference already exists - check add the binding
		auto &column = entry->second;
		column.bindings.push_back(col);

		MergeChildColumns(column.child_columns, child_column);
	}
	if (mode == BaseColumnPrunerMode::DISABLE_PUSHDOWN_EXTRACT) {
		//! Any child referenced after this mode is set disables PUSHDOWN_EXTRACT
		entry->second.supports_pushdown_extract = PushdownExtractSupport::DISABLED;
	}
}

void ReferencedColumn::AddPath(const ColumnIndex &path) {
	if (child_columns.empty()) {
		//! Full field already referenced, won't use struct field projection pushdown at all
		return;
	}
	path.VerifySinglePath();

	auto res = FindShortestMatchingPath(unique_paths, path);
	auto entry = res.first;
	if (entry != unique_paths.end()) {
		//! The parent path already exists, don't add the new path
		return;
	}

	//! No parent path exists, but child paths could already be added, remove them if they exist
	auto it = unique_paths.begin();
	for (; it != unique_paths.end();) {
		auto &unique_path = *it;
		if (unique_path.IsChildPathOf(path)) {
			auto current = it;
			it++;
			unique_paths.erase(current);
		} else {
			it++;
		}
	}
	//! Finally add the new path to the map
	unique_paths.emplace(path);
}

void BaseColumnPruner::AddBinding(BoundColumnRefExpression &col, ColumnIndex child_column,
                                  const vector<ReferencedExtractComponent> &parent) {
	AddBinding(col, child_column);
	auto entry = column_references.find(col.binding);
	if (entry == column_references.end()) {
		throw InternalException("ColumnBinding for the col was somehow not added by the previous step?");
	}
	auto &referenced_column = entry->second;
	//! Save a reference to the top-level struct extract, so we can potentially replace it later
	D_ASSERT(!referenced_column.bindings.empty());

	//! NOTE: this path does not contain the column index of the root,
	//! i.e 's.a' will just be a ColumnIndex with the index of 'a', without children
	referenced_column.AddPath(child_column);
	referenced_column.struct_extracts.emplace_back(parent, referenced_column.bindings.size() - 1,
	                                               std::move(child_column));
}

void BaseColumnPruner::AddBinding(BoundColumnRefExpression &col) {
	auto entry = column_references.find(col.binding);
	if (entry == column_references.end()) {
		// column not referenced yet - add a binding to it entirely
        // Bodo Change: making consistent with above function.
		ReferencedColumn column;
		column.bindings.push_back(col);
		column_references.insert(make_pair(col.binding, std::move(column)));
        // Bodo Change End
	} else {
		// column reference already exists - add the binding and clear any sub-references
		auto &column = entry->second;
		column.bindings.push_back(col);
		column.child_columns.clear();
	}
}

static bool TryGetCastChild(unique_ptr<Expression> &expr, optional_ptr<unique_ptr<Expression>> &child) {
	if (expr->type != ExpressionType::OPERATOR_CAST) {
		return false;
	}
	D_ASSERT(expr->GetExpressionClass() == ExpressionClass::BOUND_CAST);
	auto &cast = expr->Cast<BoundCastExpression>();
	if (cast.try_cast) {
		return false;
	}

	child = cast.child;
	return true;
}

void BaseColumnPruner::VisitExpression(unique_ptr<Expression> *expression) {
	//! Check if this is a struct extract wrapped in a cast
	optional_ptr<unique_ptr<Expression>> cast_child;
	if (TryGetCastChild(*expression, cast_child)) {
		if (HandleExtractExpression(cast_child.get(), expression)) {
			// already handled
			return;
		}
	}

	//! Check if this is a struct extract
	if (HandleExtractExpression(expression)) {
		// already handled
		return;
	}
	// recurse
	LogicalOperatorVisitor::VisitExpression(expression);
}

unique_ptr<Expression> BaseColumnPruner::VisitReplace(BoundColumnRefExpression &expr,
                                                      unique_ptr<Expression> *expr_ptr) {
	// add a reference to the entire column
	AddBinding(expr);
	return nullptr;
}

unique_ptr<Expression> BaseColumnPruner::VisitReplace(BoundReferenceExpression &expr,
                                                      unique_ptr<Expression> *expr_ptr) {
	// BoundReferenceExpression should not be used here yet, they only belong in the physical plan
	throw InternalException("BoundReferenceExpression should not be used here yet!");
}

} // namespace duckdb
