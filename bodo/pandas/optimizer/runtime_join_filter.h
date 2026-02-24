#pragma once
#include "duckdb/planner/logical_operator.hpp"

struct JoinColumnInfo {
    std::vector<int64_t> filter_columns;
    std::vector<bool> is_first_locations;
    std::vector<int64_t> orig_build_key_cols;
};

using JoinFilterProgramState = std::unordered_map<int, JoinColumnInfo>;

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
    JoinFilterProgramState join_state_map;

    /**
     * @brief Inserts join filters for all entries in join_state_map above the
     * given operator op. Returns the resulting top operator, either op or the
     * join filters.
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

    /**
     * @brief Visits a LogicalFilter operator, remapping join filter columns
     * through the filter's columns and propagating the join filter program
     * state down to the child.
     *
     * @param op - the LogicalFilter operator to visit
     * @return duckdb::unique_ptr<duckdb::LogicalOperator> - the optimized
     * operator
     */
    duckdb::unique_ptr<duckdb::LogicalOperator> VisitFilter(
        duckdb::unique_ptr<duckdb::LogicalOperator> &op);

    /**
     * @brief Visits a LogicalAggregate operator, pushing columns that are
     * groupby keys, otherwise inserting join filters above the aggregate.
     *
     * @param op - the LogicalAggregate operator to visit
     * @return duckdb::unique_ptr<duckdb::LogicalOperator> - the optimized
     * operator
     */
    duckdb::unique_ptr<duckdb::LogicalOperator> VisitAggregate(
        duckdb::unique_ptr<duckdb::LogicalOperator> &op);

    /**
     * @brief Visits a LogicalCrossProduct operator, propagating join filter
     * program state down to both children. Very similar to VisitCompJoin but
     * without any ability to create new filters or push based on join keys
     * since there are no keys.
     * @param op - the LogicalCrossProduct operator to visit
     * @return duckdb::unique_ptr<duckdb::LogicalOperator> - the optimized
     * operator
     */
    duckdb::unique_ptr<duckdb::LogicalOperator> VisitCrossProduct(
        duckdb::unique_ptr<duckdb::LogicalOperator> &op);

    /**
     * @brief Visits a LogicalDistinct operator, propagating join filter
     * program state down to the child if the distinct keys cover all join
     * filters.
     * @param op - the LogicalDistinct operator to visit
     * @return duckdb::unique_ptr<duckdb::LogicalOperator> - the optimized
     * operator
     */
    duckdb::unique_ptr<duckdb::LogicalOperator> VisitDistinct(
        duckdb::unique_ptr<duckdb::LogicalOperator> &op);

    /**
     * @brief Visits a LogicalUnion operator, remapping join filter columns
     * through the union mappings and propagating the join filter program state
     * down to both children.
     *
     * @param op - the LogicalUnion operator to visit
     * @return duckdb::unique_ptr<duckdb::LogicalOperator> - the optimized
     * operator
     */
    duckdb::unique_ptr<duckdb::LogicalOperator> VisitUnion(
        duckdb::unique_ptr<duckdb::LogicalOperator> &op);

    /**
     * @brief Visits a LogicalGet operator, inserting a join filter node on top
     * and attaching join statistics to the LogicalGet for runtime filter.
     *
     * @param op - the LogicalGet operator to visit
     * @return duckdb::unique_ptr<duckdb::LogicalOperator> - the optimized
     * operator
     */
    duckdb::unique_ptr<duckdb::LogicalOperator> VisitGet(
        duckdb::unique_ptr<duckdb::LogicalOperator> &op);
};
