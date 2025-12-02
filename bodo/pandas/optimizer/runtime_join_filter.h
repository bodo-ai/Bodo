#pragma once
#include "duckdb/planner/logical_operator.hpp"

struct JoinColumnInfo {
    std::vector<int64_t> filter_columns;
    std::vector<bool> is_first_locations;
};

class RuntimeJoinFilterPushdownOptimizer {
   public:
    /**
     * @brief Visits a logical operator and creates an optimized version of the
     * child plan with join filters inserted where appropriate.
     *
     * @param op - the logical operator to visit
     * @return duckdb::unique_ptr<duckdb::LogicalOperator> - the optimized
     * logical operator
     */
    duckdb::unique_ptr<duckdb::LogicalOperator> VisitOperator(
        duckdb::unique_ptr<duckdb::LogicalOperator> &op);

   private:
    size_t cur_join_filter_id = 0;
    using JoinFilterProgramState = std::unordered_map<int, JoinColumnInfo>;
    JoinFilterProgramState join_state_map;

    /**
     * @brief Inserts join filters for all entries in join_state_map above the
     * given operator op. Returns the resulting top operator, either op or the
     * join filters. params
     *
     * @param op - the operator above which to insert join filters
     * @param join_state_map - mapping of join IDs to join column info to create
     * filters for
     * @return duckdb::unique_ptr<duckdb::LogicalOperator> - the new top
     * operator
     *
     */
    duckdb::unique_ptr<duckdb::LogicalOperator> insert_join_filters(
        duckdb::unique_ptr<duckdb::LogicalOperator> &op,
        JoinFilterProgramState &join_state_map);

    /**
     * @brief Visits a LogicalComparisonJoin operator, pushing down join filters
     * to its children where possible. This operator can add new join filters to
     * the program state if op is a right or inner join.
     *
     * @param op - the LogicalComparisonJoin operator to visit
     * @return duckdb::unique_ptr<duckdb::LogicalOperator> - the optimized
     * operator
     */
    duckdb::unique_ptr<duckdb::LogicalOperator> VisitCompJoin(
        duckdb::unique_ptr<duckdb::LogicalOperator> &op);

    /**
     * @brief Visits a LogicalProjection operator, remapping join filter columns
     * through the projection expressions and propagating the join filter
     * program state down to the child.
     *
     * @param op - the LogicalProjection operator to visit
     * @return duckdb::unique_ptr<duckdb::LogicalOperator> - the optimized
     * operator
     */
    duckdb::unique_ptr<duckdb::LogicalOperator> VisitProjection(
        duckdb::unique_ptr<duckdb::LogicalOperator> &op);
    duckdb::unique_ptr<duckdb::LogicalOperator> VisitFilter(
        duckdb::unique_ptr<duckdb::LogicalOperator> &op);
};
