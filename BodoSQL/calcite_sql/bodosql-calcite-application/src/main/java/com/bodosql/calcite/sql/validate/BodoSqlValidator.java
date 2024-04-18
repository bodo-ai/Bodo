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
package com.bodosql.calcite.sql.validate;

import static java.util.Objects.requireNonNull;
import static org.apache.calcite.util.BodoStatic.BODO_SQL_RESOURCE;
import static org.apache.calcite.util.Static.RESOURCE;

import com.bodosql.calcite.application.operatorTables.SelectOperatorTable;
import com.bodosql.calcite.sql.ddl.SqlSnowflakeUpdate;
import com.bodosql.calcite.sql.func.SqlNamedParam;
import com.bodosql.calcite.table.ColumnDataTypeInfo;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.apache.calcite.adapter.java.JavaTypeFactory;
import org.apache.calcite.prepare.CalciteCatalogReader;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.runtime.Resources;
import org.apache.calcite.sql.JoinConditionType;
import org.apache.calcite.sql.JoinType;
import org.apache.calcite.sql.SqlBasicCall;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlCharStringLiteral;
import org.apache.calcite.sql.SqlDynamicParam;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlInsert;
import org.apache.calcite.sql.SqlJoin;
import org.apache.calcite.sql.SqlLiteral;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlNodeList;
import org.apache.calcite.sql.SqlOperatorTable;
import org.apache.calcite.sql.SqlSelect;
import org.apache.calcite.sql.SqlUpdate;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.type.BodoSqlTypeCoercionRule;
import org.apache.calcite.sql.type.SqlTypeMappingRule;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.type.SqlTypeUtil;
import org.apache.calcite.sql.validate.SqlNameMatcher;
import org.apache.calcite.sql.validate.SqlValidator;
import org.apache.calcite.sql.validate.SqlValidatorException;
import org.apache.calcite.sql.validate.SqlValidatorImpl;
import org.apache.calcite.sql.validate.SqlValidatorScope;
import org.checkerframework.checker.nullness.qual.Nullable;

/** Duplication of the CalciteSqlValidator class from Calcite. */
public class BodoSqlValidator extends SqlValidatorImpl {

  List<RelDataType> runtimeDynamicParamTypes;
  Map<String, RelDataType> runtimeNamedParamTypeMap;

  public BodoSqlValidator(
      SqlOperatorTable opTab,
      CalciteCatalogReader catalogReader,
      JavaTypeFactory typeFactory,
      SqlValidator.Config config,
      List<ColumnDataTypeInfo> dynamicParamTypes,
      Map<String, ColumnDataTypeInfo> namedParamTypeMap) {
    super(opTab, catalogReader, typeFactory, config);
    this.runtimeDynamicParamTypes =
        dynamicParamTypes.stream()
            // Note: Dynamic parameters are currently always nullable within Calcite.
            .map(x -> typeFactory.createTypeWithNullability(x.convertToSqlType(typeFactory), true))
            .collect(Collectors.toList());
    this.runtimeNamedParamTypeMap =
        namedParamTypeMap.entrySet().stream()
            .collect(
                Collectors.toMap(
                    Map.Entry::getKey,
                    x ->
                        typeFactory.createTypeWithNullability(
                            x.getValue().convertToSqlType(typeFactory), true)));
  }

  @Override
  protected RelDataType getLogicalSourceRowType(RelDataType sourceRowType, SqlInsert insert) {
    final RelDataType superType = super.getLogicalSourceRowType(sourceRowType, insert);
    return ((JavaTypeFactory) typeFactory).toSql(superType);
  }

  @Override
  protected RelDataType getLogicalTargetRowType(RelDataType targetRowType, SqlInsert insert) {
    final RelDataType superType = super.getLogicalTargetRowType(targetRowType, insert);
    return ((JavaTypeFactory) typeFactory).toSql(superType);
  }

  /**
   * Creates the SELECT statement that putatively feeds rows into an UPDATE statement to be updated.
   *
   * @param call Call to the UPDATE operator
   * @return select statement
   */
  protected SqlSelect createSourceSelectForUpdate(SqlUpdate call) {
    SqlSelect select = super.createSourceSelectForUpdate(call);
    // Cast to the BodoSQL subclass of Update to extract the FROM field.
    // If it exists, this table is cross joined with the source table.
    if (call instanceof SqlSnowflakeUpdate) {
      SqlNode from = ((SqlSnowflakeUpdate) call).getFrom();
      if (from != null) {
        select.setFrom(
            new SqlJoin(
                SqlParserPos.ZERO,
                select.getFrom(),
                SqlLiteral.createBoolean(false, SqlParserPos.ZERO),
                JoinType.CROSS.symbol(SqlParserPos.ZERO),
                from,
                JoinConditionType.NONE.symbol(SqlParserPos.ZERO),
                null));
      }
    }
    return select;
  }

  @Override
  protected void validateSelect(SqlSelect select, RelDataType targetRowType) {
    // Validate the select.
    super.validateSelect(select, targetRowType);

    // Handle fetch and offset directly. The super call only handles literals and dynamic
    // parameters.
    // It does not handle named parameters.
    handleFetchOffset(select);
  }

  /**
   * Helper function for advancedStarExpansion: takes in the current list of columns & their names,
   * the list of names to remove, and destructively removes the corresponding indices from the
   * argument & names lists.
   *
   * @param node The node that originally caused the star expansion to take place.
   * @param arguments The list of identifiers representing the columns expanded from a *.
   * @param excludingNames The list of column names for each of the arguments.
   * @param excludingNames The list of columns to exclude when expanding the star.
   */
  public void pruneExclusionColumns(
      SqlNode node,
      List<SqlNode> arguments,
      List<String> argumentNames,
      List<SqlIdentifier> excludingNames) {
    SqlNameMatcher matcher = getCatalogReader().nameMatcher();
    // For every column to be excluded, verify that it is part of the columns that were
    // created from expanding the star term, then remove that index from said lists.
    for (SqlIdentifier colToExclude : excludingNames) {
      String nameToExclude = colToExclude.toString();
      int idxToExclude = matcher.indexOf(argumentNames, nameToExclude);
      if (idxToExclude == -1) {
        throw newValidationError(node, RESOURCE.unknownIdentifier(nameToExclude));
      }
      arguments.remove(idxToExclude);
      argumentNames.remove(idxToExclude);
      // Verify that the column to exclude does not appear more than once in the names
      int secondIdx = matcher.indexOf(argumentNames, nameToExclude);
      if (secondIdx != -1) {
        throw newValidationError(node, RESOURCE.columnAmbiguous(nameToExclude));
      }
    }
  }

  /**
   * @param expandedSelectItems List to append any expanded arguments to.
   * @param expansionNode The node that is to be expanded if it is a star or special select star
   *     modifier.
   * @param scope The original scope of the select having its terms expanded.
   * @param excludingNames The list of column names to exclude when expanding the star.
   */
  private void advancedStarExpansion(
      List<SqlNode> expandedSelectItems,
      SqlNode expansionNode,
      SqlValidatorScope scope,
      List<SqlIdentifier> excludingNames) {
    if (expansionNode instanceof SqlCall) {
      SqlNameMatcher matcher = getCatalogReader().nameMatcher();
      SqlCall selectCall = (SqlCall) expansionNode;
      // If the node is a SELECT * EXCLUDING (cols), extract the column names and
      // recursively expand the * but with the columns to exclude noted in the excluding
      // names list.
      if (selectCall.getOperator().getName().equals(SelectOperatorTable.STAR_EXCLUDING.getName())) {
        assert selectCall.operandCount() == 2;
        SqlNode source = selectCall.getOperandList().get(0);
        SqlNodeList excluding = (SqlNodeList) (selectCall.getOperandList().get(1));
        for (SqlNode node : excluding.getList()) {
          excludingNames.add((SqlIdentifier) node);
        }
        // Verify that none of the names appeared more than once:
        for (int i = 0; i < excludingNames.size(); i++) {
          String excludeName = excludingNames.get(i).toString();
          for (int j = i + 1; j < excludingNames.size(); j++) {
            String otherName = excludingNames.get(j).toString();
            if (matcher.matches(excludeName, otherName)) {
              throw newValidationError(expansionNode, RESOURCE.duplicateColumnName(excludeName));
            }
          }
        }
        advancedStarExpansion(expandedSelectItems, source, scope, excludingNames);
        return;
      }
    } else if (expansionNode instanceof SqlIdentifier && ((SqlIdentifier) expansionNode).isStar()) {
      // Base case for when the node is a star. Ignore if there are no modifiers like
      // columns to ignore, as regular star expansion can deal with it.
      if (excludingNames.size() > 0) {
        // Expand the star term
        List<SqlNode> newArguments = new ArrayList<SqlNode>();
        List<String> newColumnNames = new ArrayList<>();
        SqlIdentifier starNode = (SqlIdentifier) expansionNode;
        expandStarNodes(List.of(starNode), scope, newArguments, newColumnNames);

        // Prune the EXCLUDE columns
        pruneExclusionColumns(expansionNode, newArguments, newColumnNames, excludingNames);

        // Add every argument from the star expansion that was not removed during exclusion
        expandedSelectItems.addAll(newArguments);
        return;
      }
    }
    // For everything else, add the term un-modified so regular expansion can deal with it.
    expandedSelectItems.add(expansionNode);
  }

  @Override
  protected RelDataType validateSelectList(
      final SqlNodeList selectItems, SqlSelect select, RelDataType targetRowType) {
    final SqlValidatorScope selectScope = getSelectScope(select);
    final List<SqlNode> expandedSelectItems = new ArrayList<>();

    // Apply the specialized star expansion procedure to each argument, appending any
    // regular or expanded terms to a list as we go along
    for (int i = 0; i < selectItems.size(); i++) {
      SqlNode selectItem = selectItems.get(i);
      advancedStarExpansion(
          expandedSelectItems, selectItem, selectScope, new ArrayList<SqlIdentifier>());
    }
    // Invoke regular select validation on the list of appended terms
    return super.validateSelectList(
        new SqlNodeList(expandedSelectItems, SqlParserPos.ZERO), select, targetRowType);
  }

  private void handleFetchOffset(SqlSelect select) {
    SqlValidatorScope scope = getEmptyScope();

    SqlNode fetchNode = select.getFetch();
    if (fetchNode instanceof SqlCall) {
      SqlCall fetch = (SqlCall) fetchNode;
      RelDataType type = deriveType(scope, fetch);
      if (!SqlTypeUtil.isIntType(type)) {
        throw newValidationError(fetch, RESOURCE.typeNotSupported(type.getFullTypeString()));
      }
    }

    SqlNode offsetNode = select.getOffset();
    if (offsetNode instanceof SqlCall) {
      SqlCall offset = (SqlCall) offsetNode;
      RelDataType type = deriveType(scope, offset);
      if (!SqlTypeUtil.isIntType(type)) {
        throw newValidationError(offset, RESOURCE.typeNotSupported(type.getFullTypeString()));
      }
    }
  }

  /**
   * @brief Checks whether a SqlNode is a function call containing a star operand that should be
   *     expanded to variadic arguments, e.g. HASH(*) or OBJECT_CONSTRUCT_KEEP_NULL(*)
   * @param call A SqlCall object being checked for the property.
   * @return True if the node is a call in the desired format.
   */
  @Override
  protected boolean isStarCall(SqlCall call) {
    String name = call.getOperator().getName();
    // HASH can have a * as a single operand, or one of several operands
    if (name.equals("HASH")) {
      for (SqlNode operand : call.getOperandList()) {
        if (operand instanceof SqlIdentifier && ((SqlIdentifier) operand).isStar()) {
          return true;
        }
      }
    }
    // OBJECT_CONSTRUCT(_KEEP_NULL) only allows a * when it is the only operand
    else if (name.equals("OBJECT_CONSTRUCT") || name.equals("OBJECT_CONSTRUCT_KEEP_NULL")) {
      if (call.getOperandList().size() == 1) {
        SqlNode operand = call.getOperandList().get(0);
        if (operand instanceof SqlIdentifier && ((SqlIdentifier) operand).isStar()) {
          return true;
        }
      }
    }
    return false;
  }

  /**
   * @brief Rewrites a SqlCall with a new argument list after it has had its original arguments
   *     expanded due to the presence of * terms.
   * @param call A SqlCall object being transformed.
   * @param newArgs The expanded arguments to the original SqlCall.
   * @param newColumnNames The names of the expanded arguments.
   * @return A transformed SqlCall object.
   */
  @Override
  protected SqlCall rewriteStarCall(
      SqlCall call, List<SqlNode> newArgs, List<String> newColumnNames) {
    String name = call.getOperator().getName();
    // HASH replaces the * with
    if (name.equals("HASH")) {
      return new SqlBasicCall(call.getOperator(), newArgs, call.getParserPosition());
    }
    if (name.equals("OBJECT_CONSTRUCT") || name.equals("OBJECT_CONSTRUCT_KEEP_NULL")) {
      ArrayList<SqlNode> objectArgs = new ArrayList<>();
      for (int i = 0; i < newArgs.size(); i++) {
        SqlLiteral key =
            SqlCharStringLiteral.createCharString(newColumnNames.get(i), call.getParserPosition());
        objectArgs.add(key);
        objectArgs.add(newArgs.get(i));
      }
      return new SqlBasicCall(call.getOperator(), objectArgs, call.getParserPosition());
    }
    return call;
  }

  /** Returns the type mapping rule. */
  @Override
  public SqlTypeMappingRule getTypeMappingRule() {
    return BodoSqlTypeCoercionRule.instance();
  }

  /**
   * Override inferUnknownTypes to have dynamic parameters check use the runtime type if it is
   * known, and it can be inferred.
   */
  @Override
  protected void inferUnknownTypes(
      RelDataType inferredType, SqlValidatorScope scope, SqlNode node) {
    requireNonNull(inferredType, "inferredType");
    requireNonNull(scope, "scope");
    requireNonNull(node, "node");
    final SqlValidatorScope newScope = scopes.get(node);
    if (newScope != null) {
      scope = newScope;
    }
    if ((node instanceof SqlDynamicParam) && inferredType.equals(unknownType)) {
      if (this.config().typeCoercionEnabled()) {
        if (node instanceof SqlNamedParam) {
          // Attempt to cast the named parameter based on the runtime type.
          RelDataType runtimeType = getDynamicParamType((SqlNamedParam) node);
          if (runtimeType != null) {
            setValidatedNodeType(node, runtimeType);
            return;
          } else {
            throw newValidationError(
                node, BODO_SQL_RESOURCE.namedParamIllegal(((SqlNamedParam) node).getParamName()));
          }
        } else {
          // Attempt to cast the dynamic parameter based on the runtime type.
          RelDataType runtimeType = getDynamicParamType((SqlDynamicParam) node);
          if (runtimeType != null) {
            setValidatedNodeType(node, runtimeType);
            return;
          } else {
            throw newValidationError(node, BODO_SQL_RESOURCE.dynamicParamIllegal());
          }
        }
      } else {
        Resources.ExInst<SqlValidatorException> error =
            node instanceof SqlNamedParam
                ? BODO_SQL_RESOURCE.namedParamIllegal(((SqlNamedParam) node).getParamName())
                : BODO_SQL_RESOURCE.dynamicParamIllegal();
        throw newValidationError(node, error);
      }
    }
    super.inferUnknownTypes(inferredType, scope, node);
  }

  private RelDataType getDynamicParamType(SqlDynamicParam dynamicParam) {
    if (dynamicParam instanceof SqlNamedParam) {
      SqlNamedParam namedParam = (SqlNamedParam) dynamicParam;
      String paramName = namedParam.getParamName();
      if (runtimeNamedParamTypeMap.containsKey(paramName)) {
        return runtimeNamedParamTypeMap.get(paramName);
      } else {
        return null;
      }
    } else {
      int index = dynamicParam.getIndex();
      if (index < runtimeDynamicParamTypes.size()) {
        return runtimeDynamicParamTypes.get(index);
      } else {
        return null;
      }
    }
  }

  @Override
  public @Nullable RelDataType getValidatedNodeTypeIfKnown(SqlNode node) {
    RelDataType type = super.getValidatedNodeTypeIfKnown(node);
    // If we have a named parameter fetch the type.
    if (type == null && node instanceof SqlDynamicParam) {
      type = getDynamicParamType((SqlDynamicParam) node);
      if (type != null) {
        setValidatedNodeType(node, type);
      }
    }
    return type;
  }

  @Override
  public RelDataType deriveType(SqlValidatorScope scope, SqlNode expr) {
    RelDataType type = super.deriveType(scope, expr);
    if (type.getSqlTypeName() == SqlTypeName.UNKNOWN && expr instanceof SqlDynamicParam) {
      RelDataType paramType = getDynamicParamType((SqlDynamicParam) expr);
      if (paramType != null) {
        return paramType;
      }
    }
    return type;
  }
}
