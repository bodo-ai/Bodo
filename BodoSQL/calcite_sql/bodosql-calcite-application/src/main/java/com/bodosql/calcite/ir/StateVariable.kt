package com.bodosql.calcite.ir

/**
 * Type class for state variables. There is no separate
 * functionality from a regular Variable at this time,
 * but it enables compiler checking.
 */
class StateVariable(name: String) : Variable(name) {
}
