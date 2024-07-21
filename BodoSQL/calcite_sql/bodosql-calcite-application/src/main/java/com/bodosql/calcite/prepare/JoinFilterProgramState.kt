package com.bodosql.calcite.prepare

import com.google.common.annotations.VisibleForTesting

/**
 * Tracks join state needed to properly execute the join filter program.
 * This is split into its own file to allow more complex unit testing.
 */
@VisibleForTesting
internal class JoinFilterProgramState : Iterable<Pair<Int, JoinFilterProgramState.JoinColumnInfo>> {
    /**
     * The column information need for each join. Each join maintains 1
     * entry in the list per leftKey in the join. If that key is no longer available,
     * then the entry will be -1. In addition, we must maintain if this is the first location
     * at which the key is available for generating the minimum number of filters.
     */
    data class JoinColumnInfo(
        val filterColumns: List<Int>,
        val filterIsFirstLocations: List<Boolean>,
        val nonEqualityFilterColumns: List<NonEqualityJoinFilterColumnInfo>,
    )

    data class JoinOutputLists(
        val joinIDs: List<Int>,
        val equalityFilterColumns: List<List<Int>>,
        val equalityFilterIsFirstLocations: List<List<Boolean>>,
        val nonEqualityFilterColumns: List<List<NonEqualityJoinFilterColumnInfo>>,
    )

    class JoinFilterProgramStateIterator(joinStateMap: Map<Int, JoinColumnInfo>) : Iterator<Pair<Int, JoinColumnInfo>> {
        private val joinStateMapIterator = joinStateMap.iterator()

        override fun hasNext(): Boolean {
            return joinStateMapIterator.hasNext()
        }

        override fun next(): Pair<Int, JoinColumnInfo> {
            val (joinID, joinInfo) = joinStateMapIterator.next()
            return Pair(joinID, joinInfo)
        }
    }

    // Hold the join information for all live joins. Since we need to support
    // set operations across multiple join IDs (intersect + difference), we map
    // the join id to the entry information about the join.
    private val joinStateMap: MutableMap<Int, JoinColumnInfo> = mutableMapOf()

    /**
     * Add a new join to the state.
     */
    fun add(
        joinID: Int,
        filterColumns: List<Int>,
        filterIsFirstLocations: List<Boolean>,
        nonEqualityFilterInfo: List<NonEqualityJoinFilterColumnInfo>,
    ) {
        joinStateMap[joinID] = JoinColumnInfo(filterColumns, filterIsFirstLocations, nonEqualityFilterInfo)
    }

    /**
     * Return the result as a triple of lists. This is a format more compatible with
     * the RuntimeJoinFilter RelNodes.
     */
    fun flattenToLists(): JoinOutputLists {
        val joinIDs = mutableListOf<Int>()
        val equalityFilterColumns = mutableListOf<List<Int>>()
        val equalityFilterIsFirstLocations = mutableListOf<List<Boolean>>()
        val nonEqualityFilterColumns = mutableListOf<List<NonEqualityJoinFilterColumnInfo>>()
        for ((joinID, joinInfo) in joinStateMap) {
            joinIDs.add(joinID)
            equalityFilterColumns.add(joinInfo.filterColumns)
            equalityFilterIsFirstLocations.add(joinInfo.filterIsFirstLocations)
            nonEqualityFilterColumns.add(joinInfo.nonEqualityFilterColumns)
        }
        return JoinOutputLists(joinIDs, equalityFilterColumns, equalityFilterIsFirstLocations, nonEqualityFilterColumns)
    }

    fun isEmpty(): Boolean {
        return joinStateMap.isEmpty()
    }

    fun isNotEmpty(): Boolean {
        return joinStateMap.isNotEmpty()
    }

    /**
     * Returns an iterator over the elements of this object.
     */
    override fun iterator(): Iterator<Pair<Int, JoinColumnInfo>> {
        return JoinFilterProgramStateIterator(joinStateMap)
    }

    /**
     * Compute the if a particular column matches based on our definition of equality.
     * This is ordered, so we always output based on the first column. We control if we
     * produce output from matches or non-matches based on the emptyOnMatch parameter, so
     * this can be reused for both difference and intersection.
     *
     * @param col1 The column of the first join.
     * @param isFirst1 If the column is the first location of the join.
     * @param col2 The column of the second join.
     * @param isFirst2 If the column is the first location of the join.
     * @param emptyOnMatch If we should output an empty column on a match.
     * @return A pair of the column and if it is the first location of the join. This is
     * empty if we fail our match condition mixed with the emptyOnMatch parameter.
     */
    private fun columnMatchResult(
        col1: Int,
        isFirst1: Boolean,
        col2: Int,
        isFirst2: Boolean,
        emptyOnMatch: Boolean,
    ): Pair<Int, Boolean> {
        val matches = col1 != -1 && col1 == col2 && isFirst1 == isFirst2
        // We output the empty column if we match and emptyOnMatch is True
        // or if we don't match and emptyOnMatch is False.
        val outputEmpty = matches == emptyOnMatch
        return if (outputEmpty) {
            Pair(-1, false)
        } else {
            Pair(col1, isFirst1)
        }
    }

    /**
     * Compute the Set Difference of this - other. For difference purposes
     * we consider a join column to be a "match" if the value is the same in
     * the index and its isFirstLocation value is the same. In all practical uses
     * other should be a subset of this, so this definition is reasonable.
     *
     * Note: As an invariant this function assumes that for the same join ID
     * all lists are the same length because they logically refer to the same
     * join.
     *
     * @param other The other JoinFilterProgramState to compare against.
     * @return The set difference of this - other.
     */
    fun difference(other: JoinFilterProgramState): JoinFilterProgramState {
        // Even though other is probably smaller, since we need all entries
        // that aren't contained in other we need to iterate over THIS.
        val result = JoinFilterProgramState()
        for ((joinID, joinInfo) in joinStateMap) {
            val otherJoinInfo = other.joinStateMap[joinID]
            if (otherJoinInfo == null) {
                result.add(joinID, joinInfo.filterColumns, joinInfo.filterIsFirstLocations, joinInfo.nonEqualityFilterColumns)
            } else {
                // We need to actually compute the difference between the columns.
                var hasDifference = false
                val remainingColumns: MutableList<Int> = mutableListOf()
                val remainingIsFirstLocations: MutableList<Boolean> = mutableListOf()
                joinInfo.filterColumns.forEachIndexed { index, column ->
                    val (column, isFirstLocation) =
                        columnMatchResult(
                            column,
                            joinInfo.filterIsFirstLocations[index],
                            otherJoinInfo.filterColumns[index],
                            otherJoinInfo.filterIsFirstLocations[index],
                            true,
                        )
                    hasDifference = hasDifference || column != -1
                    remainingColumns.add(column)
                    remainingIsFirstLocations.add(isFirstLocation)
                }
                // Compute the intersection of non-equality filters.
                val firstNonEqualityFilters: MutableMap<Int, MutableSet<Pair<NonEqualityType, Int>>> = mutableMapOf()
                for (nonEqualityFilter in joinInfo.nonEqualityFilterColumns) {
                    if (nonEqualityFilter.columnIndex !in firstNonEqualityFilters) {
                        firstNonEqualityFilters[nonEqualityFilter.columnIndex] = mutableSetOf()
                    }
                    firstNonEqualityFilters[nonEqualityFilter.columnIndex]!!.add(
                        Pair(nonEqualityFilter.type, nonEqualityFilter.originalJoinBuildIndex),
                    )
                }
                val secondNonEqualityFilters: MutableMap<Int, MutableSet<Pair<NonEqualityType, Int>>> = mutableMapOf()
                for (nonEqualityFilter in otherJoinInfo.nonEqualityFilterColumns) {
                    if (nonEqualityFilter.columnIndex !in secondNonEqualityFilters) {
                        secondNonEqualityFilters[nonEqualityFilter.columnIndex] = mutableSetOf()
                    }
                    secondNonEqualityFilters[nonEqualityFilter.columnIndex]!!.add(
                        Pair(nonEqualityFilter.type, nonEqualityFilter.originalJoinBuildIndex),
                    )
                }
                // Compute the actual set level intersection.
                val newNonEqualityFilters: MutableList<NonEqualityJoinFilterColumnInfo> = mutableListOf()
                for ((columnIndex, columnInfo) in firstNonEqualityFilters) {
                    if (columnIndex in secondNonEqualityFilters) {
                        val difference = columnInfo.minus(secondNonEqualityFilters[columnIndex]!!)
                        for ((type, origBuildIndex) in difference) {
                            newNonEqualityFilters.add(NonEqualityJoinFilterColumnInfo(type, columnIndex, origBuildIndex))
                        }
                    } else {
                        for ((type, origBuildIndex) in columnInfo) {
                            newNonEqualityFilters.add(NonEqualityJoinFilterColumnInfo(type, columnIndex, origBuildIndex))
                        }
                    }
                }
                if (hasDifference || newNonEqualityFilters.isNotEmpty()) {
                    result.add(joinID, remainingColumns, remainingIsFirstLocations, newNonEqualityFilters)
                }
            }
        }
        return result
    }

    /**
     * Compute the Set Intersection of this ∩ other. For intersection purposes
     * we consider a join column to be a "match" if the value is the same in
     * the index and its isFirstLocation value is the same. In all practical uses
     * the columns being referenced should be the same because we are trying to
     * compute across matching locations, so this definition is reasonable.
     *
     * Note: As an invariant this function assumes that for the same join ID
     * all lists are the same length because they logically refer to the same
     * join.
     *
     * @param other The other JoinFilterProgramState to compare against.
     * @return The set intersection of this ∩ other.
     */
    fun intersection(other: JoinFilterProgramState): JoinFilterProgramState {
        // Iterate over the smaller of the two sets.
        val (smaller, larger) =
            if (joinStateMap.size < other.joinStateMap.size) {
                joinStateMap to other.joinStateMap
            } else {
                other.joinStateMap to joinStateMap
            }
        val result = JoinFilterProgramState()
        for ((joinID, joinInfo) in smaller) {
            val otherJoinInfo = larger[joinID]
            if (otherJoinInfo != null) {
                // We need to actually compute the intersection between the columns.
                var hasIntersection = false
                val remainingColumns: MutableList<Int> = mutableListOf()
                val remainingIsFirstLocations: MutableList<Boolean> = mutableListOf()
                joinInfo.filterColumns.forEachIndexed { index, column ->
                    val (column, isFirstLocation) =
                        columnMatchResult(
                            column,
                            joinInfo.filterIsFirstLocations[index],
                            otherJoinInfo.filterColumns[index],
                            otherJoinInfo.filterIsFirstLocations[index],
                            false,
                        )
                    hasIntersection = hasIntersection || column != -1
                    remainingColumns.add(column)
                    remainingIsFirstLocations.add(isFirstLocation)
                }
                // Compute the intersection of non-equality filters.
                val firstNonEqualityFilters: MutableMap<Int, MutableSet<Pair<NonEqualityType, Int>>> = mutableMapOf()
                for (nonEqualityFilter in joinInfo.nonEqualityFilterColumns) {
                    if (nonEqualityFilter.columnIndex !in firstNonEqualityFilters) {
                        firstNonEqualityFilters[nonEqualityFilter.columnIndex] = mutableSetOf()
                    }
                    firstNonEqualityFilters[nonEqualityFilter.columnIndex]!!.add(
                        Pair(nonEqualityFilter.type, nonEqualityFilter.originalJoinBuildIndex),
                    )
                }
                val secondNonEqualityFilters: MutableMap<Int, MutableSet<Pair<NonEqualityType, Int>>> = mutableMapOf()
                for (nonEqualityFilter in otherJoinInfo.nonEqualityFilterColumns) {
                    if (nonEqualityFilter.columnIndex !in secondNonEqualityFilters) {
                        secondNonEqualityFilters[nonEqualityFilter.columnIndex] = mutableSetOf()
                    }
                    secondNonEqualityFilters[nonEqualityFilter.columnIndex]!!.add(
                        Pair(nonEqualityFilter.type, nonEqualityFilter.originalJoinBuildIndex),
                    )
                }
                // Compute the actual set level intersection.
                val newNonEqualityFilters: MutableList<NonEqualityJoinFilterColumnInfo> = mutableListOf()
                for ((columnIndex, types) in firstNonEqualityFilters) {
                    if (columnIndex in secondNonEqualityFilters) {
                        val intersection = types.intersect(secondNonEqualityFilters[columnIndex]!!)
                        for ((type, origBuildIndex) in intersection) {
                            newNonEqualityFilters.add(NonEqualityJoinFilterColumnInfo(type, columnIndex, origBuildIndex))
                        }
                    }
                }
                if (hasIntersection || newNonEqualityFilters.isNotEmpty()) {
                    result.add(joinID, remainingColumns, remainingIsFirstLocations, newNonEqualityFilters)
                }
            }
        }
        return result
    }

    /**
     * Compute the Set Union of this ∪ other. For union purposes
     * we do not consider situations in which the same join key refers to
     * a different column value or isFirstLocation value. This is because
     * all intended uses of this function are to combine set when the children
     * are a potential cache node. As a result, if the values do not match we
     * throw an error.
     *
     * Note: As an invariant this function assumes that for the same join ID
     * all lists are the same length because they logically refer to the same
     * join.
     *
     * @param other The other JoinFilterProgramState to compare against.
     * @return The set union of this ∪ other.
     */
    fun union(other: JoinFilterProgramState): JoinFilterProgramState {
        // We must iterate over both sets. To simplify code since the second
        // iteration can only be disjoint joins, we track the seen join IDs.
        val seenJoinIDs = mutableSetOf<Int>()
        val result = JoinFilterProgramState()
        for ((joinID, joinInfo) in joinStateMap) {
            seenJoinIDs.add(joinID)
            val otherJoinInfo = other.joinStateMap[joinID]
            if (otherJoinInfo == null) {
                result.add(joinID, joinInfo.filterColumns, joinInfo.filterIsFirstLocations, joinInfo.nonEqualityFilterColumns)
            } else {
                // We need to actually compute the union between the columns.
                val remainingColumns: MutableList<Int> = mutableListOf()
                val remainingIsFirstLocations: MutableList<Boolean> = mutableListOf()
                joinInfo.filterColumns.forEachIndexed { index, column ->
                    val otherColumn = otherJoinInfo.filterColumns[index]
                    val (finalColumn, finalIsFirst) =
                        if (column == -1 && otherColumn == -1) {
                            // Missing column
                            Pair(-1, false)
                        } else if (column == -1) {
                            Pair(otherColumn, otherJoinInfo.filterIsFirstLocations[index])
                        } else if (otherColumn == -1) {
                            Pair(column, joinInfo.filterIsFirstLocations[index])
                        } else {
                            if (column != otherColumn ||
                                joinInfo.filterIsFirstLocations[index] != otherJoinInfo.filterIsFirstLocations[index]
                            ) {
                                throw IllegalArgumentException("Cannot union two different columns")
                            }
                            Pair(column, joinInfo.filterIsFirstLocations[index])
                        }
                    remainingColumns.add(finalColumn)
                    remainingIsFirstLocations.add(finalIsFirst)
                }
                // Compute a map of the non-equality filters based on the supported types.
                val nonEqualityFilterColumns: MutableMap<Int, MutableSet<Pair<NonEqualityType, Int>>> = mutableMapOf()
                for (nonEqualityFilter in joinInfo.nonEqualityFilterColumns) {
                    if (nonEqualityFilter.columnIndex !in nonEqualityFilterColumns) {
                        nonEqualityFilterColumns[nonEqualityFilter.columnIndex] = mutableSetOf()
                    }
                    nonEqualityFilterColumns[nonEqualityFilter.columnIndex]!!.add(
                        Pair(nonEqualityFilter.type, nonEqualityFilter.originalJoinBuildIndex),
                    )
                }
                for (nonEqualityFilter in otherJoinInfo.nonEqualityFilterColumns) {
                    if (nonEqualityFilter.columnIndex !in nonEqualityFilterColumns) {
                        nonEqualityFilterColumns[nonEqualityFilter.columnIndex] = mutableSetOf()
                    }
                    nonEqualityFilterColumns[nonEqualityFilter.columnIndex]!!.add(
                        Pair(nonEqualityFilter.type, nonEqualityFilter.originalJoinBuildIndex),
                    )
                }
                val newNonEqualityFilters: MutableList<NonEqualityJoinFilterColumnInfo> = mutableListOf()
                for ((columnIndex, columnInfos) in nonEqualityFilterColumns) {
                    for ((type, origBuildIndex) in columnInfos) {
                        newNonEqualityFilters.add(NonEqualityJoinFilterColumnInfo(type, columnIndex, origBuildIndex))
                    }
                }
                result.add(joinID, remainingColumns, remainingIsFirstLocations, newNonEqualityFilters)
            }
        }
        // Check for any joins only in the other join state.
        for ((joinID, joinInfo) in other.joinStateMap) {
            if (joinID !in seenJoinIDs) {
                result.add(joinID, joinInfo.filterColumns, joinInfo.filterIsFirstLocations, joinInfo.nonEqualityFilterColumns)
            }
        }
        return result
    }

    /**
     * Return if this is equal to the other object. This is a deep equality check.
     *
     * @param other The other object to compare against.
     * @return If the underlying maps of two JoinFilterProgramState
     * objects are equal.
     */
    override fun equals(other: Any?): Boolean {
        if (other !is JoinFilterProgramState) {
            return false
        }
        // Map equality checks the equality of the entries
        // and requires the same size.
        // https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/-abstract-map/equals.html
        return joinStateMap == other.joinStateMap
    }
}
