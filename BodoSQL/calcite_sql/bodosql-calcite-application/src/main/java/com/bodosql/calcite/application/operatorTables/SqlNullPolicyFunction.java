package com.bodosql.calcite.application.operatorTables;
/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import static java.util.Objects.requireNonNull;

import java.util.function.Function;
import java.util.function.Supplier;
import org.apache.calcite.plan.Strong;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlFunction;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperatorBinding;
import org.apache.calcite.sql.SqlSyntax;
import org.apache.calcite.sql.type.OperandHandlers;
import org.apache.calcite.sql.type.SqlOperandHandler;
import org.apache.calcite.sql.type.SqlOperandTypeChecker;
import org.apache.calcite.sql.type.SqlOperandTypeInference;
import org.apache.calcite.sql.type.SqlReturnTypeInference;
import org.apache.calcite.sql.validate.SqlMonotonicity;
import org.apache.calcite.sql.validate.SqlValidator;
import org.apache.calcite.sql.validate.SqlValidatorScope;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;

/**
 * Extended implementation of SqlBasicFunction that is extended to support SqlNullPolicyFunction.
 * This is necessary because SqlBasicFunction doesn't contain any interface to update the null
 * policy.
 *
 * <p>Hopefully this can be removed in a future Calcite release.
 */
public class SqlNullPolicyFunction extends SqlFunction {
  private final SqlSyntax syntax;
  private final boolean deterministic;
  private final SqlOperandHandler operandHandler;
  private final int callValidator;
  private final Function<SqlOperatorBinding, SqlMonotonicity> monotonicityInference;

  private final Strong.Policy nullablePolicy;

  // ~ Constructors -----------------------------------------------------------

  /**
   * Creates a new SqlFunction for a call to a built-in function.
   *
   * @param name Name of built-in function
   * @param kind Kind of operator implemented by function
   * @param syntax Syntax
   * @param deterministic Whether the function is deterministic
   * @param returnTypeInference Strategy to use for return type inference
   * @param operandTypeInference Strategy to use for parameter type inference
   * @param operandHandler Strategy to use for handling operands
   * @param operandTypeChecker Strategy to use for parameter type checking
   * @param callValidator Strategy to validate calls
   * @param category Categorization for function
   * @param monotonicityInference Strategy to infer monotonicity of a call
   * @param nullablePolicy Policy used to simplify IS NULL and IS NOT NULL.
   */
  private SqlNullPolicyFunction(
      String name,
      SqlKind kind,
      SqlSyntax syntax,
      boolean deterministic,
      SqlReturnTypeInference returnTypeInference,
      @Nullable SqlOperandTypeInference operandTypeInference,
      SqlOperandHandler operandHandler,
      SqlOperandTypeChecker operandTypeChecker,
      Integer callValidator,
      SqlFunctionCategory category,
      Function<SqlOperatorBinding, SqlMonotonicity> monotonicityInference,
      Strong.Policy nullablePolicy) {
    super(
        name,
        kind,
        requireNonNull(returnTypeInference, "returnTypeInference"),
        operandTypeInference,
        requireNonNull(operandTypeChecker, "operandTypeChecker"),
        category);
    this.syntax = requireNonNull(syntax, "syntax");
    this.deterministic = deterministic;
    this.operandHandler = requireNonNull(operandHandler, "operandHandler");
    this.callValidator = requireNonNull(callValidator, "callValidator");
    this.monotonicityInference = requireNonNull(monotonicityInference, "monotonicityInference");
    this.nullablePolicy = nullablePolicy;
  }

  /** Creates a {@code SqlNullPolicyFunction}. */
  private static SqlNullPolicyFunction create(
      String name,
      SqlReturnTypeInference returnTypeInference,
      SqlOperandTypeChecker operandTypeChecker,
      SqlFunctionCategory category,
      Strong.Policy nullablePolicy) {
    return new SqlNullPolicyFunction(
        name,
        SqlKind.OTHER_FUNCTION,
        SqlSyntax.FUNCTION,
        true,
        returnTypeInference,
        null,
        OperandHandlers.DEFAULT,
        operandTypeChecker,
        0,
        category,
        call -> SqlMonotonicity.NOT_MONOTONIC,
        nullablePolicy);
  }

  /** Create a SqlNullPolicyFunction without a Null Policy. */
  public static SqlNullPolicyFunction createNoPolicy(
      String name,
      SqlReturnTypeInference returnTypeInference,
      SqlOperandTypeChecker operandTypeChecker,
      SqlFunctionCategory category) {
    return create(name, returnTypeInference, operandTypeChecker, category, null);
  }

  /**
   * Create a SqlNullPolicyFunction with the ANY policy. This is used if a is only null if any input
   * is null.
   */
  public static SqlNullPolicyFunction createAnyPolicy(
      String name,
      SqlReturnTypeInference returnTypeInference,
      SqlOperandTypeChecker operandTypeChecker,
      SqlFunctionCategory category) {
    return create(name, returnTypeInference, operandTypeChecker, category, Strong.Policy.ANY);
  }

  // ~ Methods ----------------------------------------------------------------

  @Override
  public SqlReturnTypeInference getReturnTypeInference() {
    return requireNonNull(super.getReturnTypeInference(), "returnTypeInference");
  }

  @Override
  public SqlOperandTypeChecker getOperandTypeChecker() {
    return requireNonNull(super.getOperandTypeChecker(), "operandTypeChecker");
  }

  @Override
  public SqlSyntax getSyntax() {
    return syntax;
  }

  @Override
  public boolean isDeterministic() {
    return deterministic;
  }

  @Override
  public SqlMonotonicity getMonotonicity(SqlOperatorBinding call) {
    return monotonicityInference.apply(call);
  }

  @Override
  public SqlNode rewriteCall(SqlValidator validator, SqlCall call) {
    return operandHandler.rewriteCall(validator, call);
  }

  @Override
  public void validateCall(
      SqlCall call,
      SqlValidator validator,
      SqlValidatorScope scope,
      SqlValidatorScope operandScope) {
    super.validateCall(call, validator, scope, operandScope);
  }

  /** Returns a copy of this function with a given name. */
  public SqlNullPolicyFunction withName(String name) {
    return new SqlNullPolicyFunction(
        name,
        kind,
        syntax,
        deterministic,
        getReturnTypeInference(),
        getOperandTypeInference(),
        operandHandler,
        getOperandTypeChecker(),
        callValidator,
        getFunctionType(),
        monotonicityInference,
        nullablePolicy);
  }

  /** Returns a copy of this function with a given kind. */
  public SqlNullPolicyFunction withKind(SqlKind kind) {
    return new SqlNullPolicyFunction(
        getName(),
        kind,
        syntax,
        deterministic,
        getReturnTypeInference(),
        getOperandTypeInference(),
        operandHandler,
        getOperandTypeChecker(),
        callValidator,
        getFunctionType(),
        monotonicityInference,
        nullablePolicy);
  }

  /** Returns a copy of this function with a given category. */
  public SqlNullPolicyFunction withFunctionType(SqlFunctionCategory category) {
    return new SqlNullPolicyFunction(
        getName(),
        kind,
        syntax,
        deterministic,
        getReturnTypeInference(),
        getOperandTypeInference(),
        operandHandler,
        getOperandTypeChecker(),
        callValidator,
        category,
        monotonicityInference,
        nullablePolicy);
  }

  /** Returns a copy of this function with a given syntax. */
  public SqlNullPolicyFunction withSyntax(SqlSyntax syntax) {
    return new SqlNullPolicyFunction(
        getName(),
        kind,
        syntax,
        deterministic,
        getReturnTypeInference(),
        getOperandTypeInference(),
        operandHandler,
        getOperandTypeChecker(),
        callValidator,
        getFunctionType(),
        monotonicityInference,
        nullablePolicy);
  }

  /** Returns a copy of this function with a given strategy for inferring returned type. */
  public SqlNullPolicyFunction withReturnTypeInference(SqlReturnTypeInference returnTypeInference) {
    return new SqlNullPolicyFunction(
        getName(),
        kind,
        syntax,
        deterministic,
        returnTypeInference,
        getOperandTypeInference(),
        operandHandler,
        getOperandTypeChecker(),
        callValidator,
        getFunctionType(),
        monotonicityInference,
        nullablePolicy);
  }

  /**
   * Returns a copy of this function with a given strategy for inferring the types of its operands.
   */
  public SqlNullPolicyFunction withOperandTypeInference(
      SqlOperandTypeInference operandTypeInference) {
    return new SqlNullPolicyFunction(
        getName(),
        kind,
        syntax,
        deterministic,
        getReturnTypeInference(),
        operandTypeInference,
        operandHandler,
        getOperandTypeChecker(),
        callValidator,
        getFunctionType(),
        monotonicityInference,
        nullablePolicy);
  }

  /** Returns a copy of this function with a given strategy for handling operands. */
  public SqlNullPolicyFunction withOperandHandler(SqlOperandHandler operandHandler) {
    return new SqlNullPolicyFunction(
        getName(),
        kind,
        syntax,
        deterministic,
        getReturnTypeInference(),
        getOperandTypeInference(),
        operandHandler,
        getOperandTypeChecker(),
        callValidator,
        getFunctionType(),
        monotonicityInference,
        nullablePolicy);
  }
  /** Returns a copy of this function with a given determinism. */
  public SqlNullPolicyFunction withDeterministic(boolean deterministic) {
    return new SqlNullPolicyFunction(
        getName(),
        kind,
        syntax,
        deterministic,
        getReturnTypeInference(),
        getOperandTypeInference(),
        operandHandler,
        getOperandTypeChecker(),
        callValidator,
        getFunctionType(),
        monotonicityInference,
        nullablePolicy);
  }

  /**
   * Returns a copy of this function with a given strategy for inferring whether a call is
   * monotonic.
   */
  public SqlNullPolicyFunction withMonotonicityInference(
      Function<SqlOperatorBinding, SqlMonotonicity> monotonicityInference) {
    return new SqlNullPolicyFunction(
        getName(),
        kind,
        syntax,
        deterministic,
        getReturnTypeInference(),
        getOperandTypeInference(),
        operandHandler,
        getOperandTypeChecker(),
        callValidator,
        getFunctionType(),
        monotonicityInference,
        nullablePolicy);
  }

  public SqlFunction withValidation(int callValidator) {
    return new SqlNullPolicyFunction(
        getName(),
        kind,
        syntax,
        deterministic,
        getReturnTypeInference(),
        getOperandTypeInference(),
        operandHandler,
        getOperandTypeChecker(),
        callValidator,
        getFunctionType(),
        monotonicityInference,
        nullablePolicy);
  }

  /**
   * Returns the {@link Strong.Policy} strategy for this operator, or null if there is no particular
   * strategy, in which case this policy will be deducted from the operator's {@link SqlKind}.
   *
   * @see Strong
   */
  @Override
  @Pure
  public @Nullable Supplier<Strong.Policy> getStrongPolicyInference() {
    if (nullablePolicy == null) {
      return null;
    } else {
      return () -> requireNonNull(nullablePolicy, "nullablePolicy");
    }
  }
}
