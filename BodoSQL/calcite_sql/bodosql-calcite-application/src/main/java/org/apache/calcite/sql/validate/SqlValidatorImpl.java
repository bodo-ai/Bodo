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
package org.apache.calcite.sql.validate;

import com.bodosql.calcite.sql.util.SqlDeepCopyShuttle;
import org.apache.calcite.linq4j.Ord;
import org.apache.calcite.linq4j.function.Functions;
import org.apache.calcite.plan.RelOptTable;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.prepare.Prepare;
import org.apache.calcite.rel.type.DynamicRecordType;
import org.apache.calcite.rel.type.RelCrossType;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rel.type.RelDataTypeSystem;
import org.apache.calcite.rel.type.RelRecordType;
import org.apache.calcite.rel.type.TimeFrame;
import org.apache.calcite.rel.type.TimeFrameSet;
import org.apache.calcite.rel.type.TimeFrames;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexPatternFieldRef;
import org.apache.calcite.rex.RexVisitor;
import org.apache.calcite.runtime.CalciteContextException;
import org.apache.calcite.runtime.CalciteException;
import org.apache.calcite.runtime.Feature;
import org.apache.calcite.runtime.PairList;
import org.apache.calcite.runtime.Resources;
import org.apache.calcite.schema.ColumnStrategy;
import org.apache.calcite.schema.Schema;
import org.apache.calcite.schema.Table;
import org.apache.calcite.schema.impl.ModifiableViewTable;
import org.apache.calcite.sql.JoinConditionType;
import org.apache.calcite.sql.JoinType;
import org.apache.calcite.sql.SqlAccessEnum;
import org.apache.calcite.sql.SqlAccessType;
import org.apache.calcite.sql.SqlAggFunction;
import org.apache.calcite.sql.SqlAsOperator;
import org.apache.calcite.sql.SqlAsofJoin;
import org.apache.calcite.sql.SqlBasicCall;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlCallBinding;
import org.apache.calcite.sql.SqlCreate;
import org.apache.calcite.sql.SqlDataTypeSpec;
import org.apache.calcite.sql.SqlDelete;
import org.apache.calcite.sql.SqlDeleteUsingItem;
import org.apache.calcite.sql.SqlDynamicParam;
import org.apache.calcite.sql.SqlExplain;
import org.apache.calcite.sql.SqlFunction;
import org.apache.calcite.sql.SqlFunctionCategory;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlInsert;
import org.apache.calcite.sql.SqlIntervalLiteral;
import org.apache.calcite.sql.SqlIntervalQualifier;
import org.apache.calcite.sql.SqlJoin;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlLambda;
import org.apache.calcite.sql.SqlLiteral;
import org.apache.calcite.sql.SqlMatchRecognize;
import org.apache.calcite.sql.SqlMerge;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlNodeList;
import org.apache.calcite.sql.SqlOperator;
import org.apache.calcite.sql.SqlOperatorTable;
import org.apache.calcite.sql.SqlOrderBy;
import org.apache.calcite.sql.SqlOverOperator;
import org.apache.calcite.sql.SqlPivot;
import org.apache.calcite.sql.SqlSampleSpec;
import org.apache.calcite.sql.SqlSelect;
import org.apache.calcite.sql.SqlSelectKeyword;
import org.apache.calcite.sql.SqlSnapshot;
import org.apache.calcite.sql.SqlSyntax;
import org.apache.calcite.sql.SqlTableFunction;
import org.apache.calcite.sql.SqlTableIdentifierWithID;
import org.apache.calcite.sql.SqlUnknownLiteral;
import org.apache.calcite.sql.SqlUnpivot;
import org.apache.calcite.sql.SqlUnresolvedFunction;
import org.apache.calcite.sql.SqlUpdate;
import org.apache.calcite.sql.SqlUtil;
import org.apache.calcite.sql.SqlValuesOperator;
import org.apache.calcite.sql.SqlWindow;
import org.apache.calcite.sql.SqlWindowTableFunction;
import org.apache.calcite.sql.SqlWith;
import org.apache.calcite.sql.SqlWithItem;
import org.apache.calcite.sql.TableCharacteristic;
import org.apache.calcite.sql.ddl.SqlCreateTable;
import org.apache.calcite.sql.fun.SqlAggOperatorTable;
import org.apache.calcite.sql.fun.SqlCase;
import org.apache.calcite.sql.fun.SqlInternalOperators;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.type.AssignableOperandTypeChecker;
import org.apache.calcite.sql.type.OperandTypes;
import org.apache.calcite.sql.type.ReturnTypes;
import org.apache.calcite.sql.type.SqlOperandTypeChecker;
import org.apache.calcite.sql.type.SqlOperandTypeInference;
import org.apache.calcite.sql.type.SqlTypeCoercionRule;
import org.apache.calcite.sql.type.SqlTypeFamily;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.type.SqlTypeUtil;
import org.apache.calcite.sql.util.IdPair;
import org.apache.calcite.sql.util.SqlBasicVisitor;
import org.apache.calcite.sql.util.SqlShuttle;
import org.apache.calcite.sql.util.SqlVisitor;
import org.apache.calcite.sql.validate.implicit.TypeCoercion;
import org.apache.calcite.tools.ValidationException;
import org.apache.calcite.util.BitString;
import org.apache.calcite.util.Bug;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.util.ImmutableIntList;
import org.apache.calcite.util.ImmutableNullableList;
import org.apache.calcite.util.Litmus;
import org.apache.calcite.util.Optionality;
import org.apache.calcite.util.Pair;
import org.apache.calcite.util.Static;
import org.apache.calcite.util.Util;
import org.apache.calcite.util.trace.CalciteTrace;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;

import org.apache.commons.lang3.NotImplementedException;
import org.apiguardian.api.API;
import org.checkerframework.checker.nullness.qual.KeyFor;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.nullness.qual.PolyNull;
import org.checkerframework.dataflow.qual.Pure;
import org.slf4j.Logger;

import java.math.BigDecimal;
import java.util.AbstractList;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Calendar;
import java.util.Collection;
import java.util.Collections;
import java.util.Deque;
import java.util.GregorianCalendar;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

import static org.apache.calcite.linq4j.Nullness.castNonNull;
import static org.apache.calcite.linq4j.Ord.forEach;
import static org.apache.calcite.sql.SqlKind.TABLE_IDENTIFIER_WITH_ID;
import static org.apache.calcite.sql.SqlUtil.stripAs;
import static org.apache.calcite.sql.type.NonNullableAccessors.getCharset;
import static org.apache.calcite.sql.type.NonNullableAccessors.getCollation;
import static org.apache.calcite.sql.validate.SqlNonNullableAccessors.getCondition;
import static org.apache.calcite.sql.validate.SqlNonNullableAccessors.getMatchCondition;
import static org.apache.calcite.sql.validate.SqlNonNullableAccessors.getTable;
import static org.apache.calcite.util.BodoStatic.BODO_SQL_RESOURCE;
import static org.apache.calcite.util.Static.RESOURCE;
import static org.apache.calcite.util.Util.first;


import static java.util.Collections.emptyList;
import static java.util.Objects.requireNonNull;

/**
 * Default implementation of {@link SqlValidator}.
 */
public class SqlValidatorImpl implements SqlValidatorWithHints {
  //~ Static fields/initializers ---------------------------------------------

  public static final Logger TRACER = CalciteTrace.PARSER_LOGGER;

  /**
   * Alias generated for the source table when rewriting UPDATE to MERGE.
   */
  public static final String UPDATE_SRC_ALIAS = "SYS$SRC";

  /**
   * Alias generated for the target table when rewriting UPDATE to MERGE if no
   * alias was specified by the user.
   */
  public static final String UPDATE_TGT_ALIAS = "SYS$TGT";

  /**
   * Alias prefix generated for source columns when rewriting UPDATE to MERGE.
   */
  public static final String UPDATE_ANON_PREFIX = "SYS$ANON";

  //~ Instance fields --------------------------------------------------------

  private final SqlOperatorTable opTab;
  final SqlValidatorCatalogReader catalogReader;

  /**
   * Maps {@link SqlParserPos} strings to the {@link SqlIdentifier} identifier
   * objects at these positions.
   */
  protected final Map<String, IdInfo> idPositions = new HashMap<>();

  /**
   * Maps {@link SqlNode query node} objects to the {@link SqlValidatorScope}
   * scope created from them.
   */
  protected final IdentityHashMap<SqlNode, SqlValidatorScope> scopes =
      new IdentityHashMap<>();

  /**
   * Maps a {@link SqlSelect} and a {@link Clause} to the scope used by that
   * clause.
   */
  private final Map<IdPair<SqlSelect, Clause>, SqlValidatorScope>
      clauseScopes = new HashMap<>();

  /**
   * The name-resolution scope of a LATERAL TABLE clause.
   */
  private @Nullable TableScope tableScope = null;

  /**
   * Maps a {@link SqlNode node} to the
   * {@link SqlValidatorNamespace namespace} which describes what columns they
   * contain.
   */
  protected final IdentityHashMap<SqlNode, SqlValidatorNamespace> namespaces =
      new IdentityHashMap<>();

  /**
   * Set of select expressions used as cursor definitions. In standard SQL,
   * only the top-level SELECT is a cursor; Calcite extends this with
   * cursors as inputs to table functions.
   */
  private final Set<SqlNode> cursorSet = Sets.newIdentityHashSet();

  /**
   * Stack of objects that maintain information about function calls. A stack
   * is needed to handle nested function calls. The function call currently
   * being validated is at the top of the stack.
   */
  protected final Deque<FunctionParamInfo> functionCallStack =
      new ArrayDeque<>();

  private int nextGeneratedId;
  protected final RelDataTypeFactory typeFactory;

  /** The type of dynamic parameters until a type is imposed on them. */
  protected final RelDataType unknownType;
  private final RelDataType booleanType;

  protected final TimeFrameSet timeFrameSet;

  /**
   * Map of derived RelDataType for each node. This is an IdentityHashMap
   * since in some cases (such as null literals) we need to discriminate by
   * instance.
   */
  private final IdentityHashMap<SqlNode, RelDataType> nodeToTypeMap =
      new IdentityHashMap<>();

  /** Provides the data for {@link #getValidatedOperandTypes(SqlCall)}. */
  public final IdentityHashMap<SqlCall, List<RelDataType>> callToOperandTypesMap =
      new IdentityHashMap<>();

  private final AggFinder aggFinder;
  private final AggFinder aggOrOverFinder;
  private final AggFinder aggOrOverOrGroupFinder;
  private final AggFinder groupFinder;
  private final AggFinder overFinder;

  private Config config;

  private final Map<SqlNode, SqlNode> originalExprs = new HashMap<>();

  private @Nullable SqlNode top;

  // TODO jvs 11-Dec-2008:  make this local to performUnconditionalRewrites
  // if it's OK to expand the signature of that method.
  private boolean validatingSqlMerge;

  private boolean inWindow;                        // Allow nested aggregates

  private final SqlValidatorImpl.ValidationErrorFunction validationErrorFunction =
      new SqlValidatorImpl.ValidationErrorFunction();

  // TypeCoercion instance used for implicit type coercion.
  private final TypeCoercion typeCoercion;

  //~ Constructors -----------------------------------------------------------

  /**
   * Creates a validator.
   *
   * @param opTab         Operator table
   * @param catalogReader Catalog reader
   * @param typeFactory   Type factory
   * @param config        Config
   */
  protected SqlValidatorImpl(
      SqlOperatorTable opTab,
      SqlValidatorCatalogReader catalogReader,
      RelDataTypeFactory typeFactory,
      Config config) {
    this.opTab = requireNonNull(opTab, "opTab");
    this.catalogReader = requireNonNull(catalogReader, "catalogReader");
    this.typeFactory = requireNonNull(typeFactory, "typeFactory");
    final RelDataTypeSystem typeSystem = typeFactory.getTypeSystem();
    this.timeFrameSet =
        requireNonNull(typeSystem.deriveTimeFrameSet(TimeFrames.CORE),
            "timeFrameSet");
    this.config = requireNonNull(config, "config");

    // It is assumed that unknown type is nullable by default
    unknownType = typeFactory.createTypeWithNullability(typeFactory.createUnknownType(), true);
    booleanType = typeFactory.createSqlType(SqlTypeName.BOOLEAN);

    final SqlNameMatcher nameMatcher = catalogReader.nameMatcher();
    aggFinder = new AggFinder(opTab, false, true, false, null, nameMatcher);
    aggOrOverFinder =
        new AggFinder(opTab, true, true, false, null, nameMatcher);
    overFinder =
        new AggFinder(opTab, true, false, false, aggOrOverFinder, nameMatcher);
    groupFinder = new AggFinder(opTab, false, false, true, null, nameMatcher);
    aggOrOverOrGroupFinder =
        new AggFinder(opTab, true, true, true, null, nameMatcher);
    @SuppressWarnings("argument.type.incompatible")
    TypeCoercion typeCoercion = config.typeCoercionFactory().create(typeFactory, this);
    this.typeCoercion = typeCoercion;
    if (config.conformance().allowLenientCoercion()) {
      final SqlTypeCoercionRule rules =
          first(config.typeCoercionRules(),
              SqlTypeCoercionRule.instance());

      final ImmutableSet<SqlTypeName> arrayMapping =
          ImmutableSet.<SqlTypeName>builder()
              .addAll(rules.getTypeMapping()
                  .getOrDefault(SqlTypeName.ARRAY, ImmutableSet.of()))
              .add(SqlTypeName.VARCHAR)
              .add(SqlTypeName.CHAR)
              .build();

      Map<SqlTypeName, ImmutableSet<SqlTypeName>> mapping =
          new HashMap<>(rules.getTypeMapping());
      mapping.replace(SqlTypeName.ARRAY, arrayMapping);
      SqlTypeCoercionRule rules2 = SqlTypeCoercionRule.instance(mapping);

      SqlTypeCoercionRule.THREAD_PROVIDERS.set(rules2);
    } else if (config.typeCoercionRules() != null) {
      SqlTypeCoercionRule.THREAD_PROVIDERS.set(config.typeCoercionRules());
    }
  }

  //~ Methods ----------------------------------------------------------------

  public SqlConformance getConformance() {
    return config.conformance();
  }

  @Pure
  @Override public SqlValidatorCatalogReader getCatalogReader() {
    return catalogReader;
  }

  @Pure
  @Override public SqlOperatorTable getOperatorTable() {
    return opTab;
  }

  @Pure
  @Override public RelDataTypeFactory getTypeFactory() {
    return typeFactory;
  }

  @Override public RelDataType getUnknownType() {
    return unknownType;
  }

  @Override public TimeFrameSet getTimeFrameSet() {
    return timeFrameSet;
  }

  @Override public SqlNodeList expandStar(SqlNodeList selectList,
      SqlSelect select, boolean includeSystemVars) {
    final List<SqlNode> list = new ArrayList<>();
    final PairList<String, RelDataType> types = PairList.of();
    for (int i = 0; i < selectList.size(); i++) {
      final SqlNode selectItem = selectList.get(i);
      final RelDataType originalType = getValidatedNodeTypeIfKnown(selectItem);
      expandSelectItem(selectItem, select, first(originalType, unknownType),
          list, catalogReader.nameMatcher().createSet(), types,
          includeSystemVars, i);
    }
    getRawSelectScopeNonNull(select).setExpandedSelectList(list);
    return new SqlNodeList(list, SqlParserPos.ZERO);
  }

  @Override public void declareCursor(SqlSelect select,
      SqlValidatorScope parentScope) {
    cursorSet.add(select);

    // add the cursor to a map that maps the cursor to its select based on
    // the position of the cursor relative to other cursors in that call
    FunctionParamInfo funcParamInfo =
        requireNonNull(functionCallStack.peek(), "functionCall");
    Map<Integer, SqlSelect> cursorMap = funcParamInfo.cursorPosToSelectMap;
    final int cursorCount = cursorMap.size();
    cursorMap.put(cursorCount, select);

    // create a namespace associated with the result of the select
    // that is the argument to the cursor constructor; register it
    // with a scope corresponding to the cursor
    SelectScope cursorScope =
        new SelectScope(parentScope, getEmptyScope(), select);
    clauseScopes.put(IdPair.of(select, Clause.CURSOR), cursorScope);
    final SelectNamespace selectNs = createSelectNamespace(select, select);
    final String alias = SqlValidatorUtil.alias(select, nextGeneratedId++);
    registerNamespace(cursorScope, alias, selectNs, false);
  }

  @Override public void pushFunctionCall() {
    FunctionParamInfo funcInfo = new FunctionParamInfo();
    functionCallStack.push(funcInfo);
  }

  @Override public void popFunctionCall() {
    functionCallStack.pop();
  }

  @Override public @Nullable String getParentCursor(String columnListParamName) {
    FunctionParamInfo funcParamInfo =
        requireNonNull(functionCallStack.peek(), "functionCall");
    Map<String, String> parentCursorMap =
        funcParamInfo.columnListParamToParentCursorMap;
    return parentCursorMap.get(columnListParamName);
  }

  /**
   * If <code>selectItem</code> is "*" or "TABLE.*", expands it and returns
   * true; otherwise writes the unexpanded item.
   *
   * @param selectItem        Select-list item
   * @param select            Containing select clause
   * @param selectItems       List that expanded items are written to
   * @param aliases           Set of aliases
   * @param fields            List of field names and types, in alias order
   * @param includeSystemVars If true include system vars in lists
   * @return Whether the node was expanded
   */
   private boolean expandSelectItem(final SqlNode selectItem, SqlSelect select,
      RelDataType targetType, List<SqlNode> selectItems, Set<String> aliases,
      PairList<String, RelDataType> fields, boolean includeSystemVars,
      Integer selectItemIdx) {
     final SqlValidatorScope selectScope;
     SqlNode expanded;
     if (SqlValidatorUtil.isMeasure(selectItem)) {
       selectScope = getMeasureScope(select);
       expanded = selectItem;
     } else {
       final SelectScope scope = (SelectScope) getWhereScope(select);
       if (expandStar(selectItems, aliases, fields, includeSystemVars, scope,
           selectItem)) {
         return true;
       }

       // Expand the select item: fully-qualify columns, and convert
       // parentheses-free functions such as LOCALTIME into explicit function
       // calls.
       selectScope = getSelectScope(select);
       expanded = expandSelectExpr(selectItem, scope, select, selectItemIdx);
     }
    final String alias =
        SqlValidatorUtil.alias(selectItem, aliases.size());

    // If expansion has altered the natural alias, supply an explicit 'AS'.
    if (expanded != selectItem) {
      String newAlias =
          SqlValidatorUtil.alias(expanded, aliases.size());
      if (!Objects.equals(newAlias, alias)) {
        expanded =
            SqlStdOperatorTable.AS.createCall(
                selectItem.getParserPosition(),
                expanded,
                new SqlIdentifier(alias, SqlParserPos.ZERO));
        deriveTypeImpl(selectScope, expanded);
      }
    }

    selectItems.add(expanded);
    aliases.add(alias);

    inferUnknownTypes(targetType, selectScope, expanded);

    RelDataType type = deriveType(selectScope, expanded);
    // Re-derive SELECT ITEM's data type that may be nullable in
    // AggregatingSelectScope when it appears in advanced grouping elements such
    // as CUBE, ROLLUP, GROUPING SETS. For example, in
    //   SELECT CASE WHEN c = 1 THEN '1' ELSE '23' END AS x
    //   FROM t
    //   GROUP BY CUBE(x)
    if (selectScope instanceof AggregatingSelectScope) {
      type = requireNonNull(selectScope.nullifyType(stripAs(expanded), type));
    }
    setValidatedNodeType(expanded, type);
    fields.add(alias, type);
    return false;
  }


  private static SqlNode expandExprFromJoin(SqlJoin join,
      SqlIdentifier identifier, SelectScope scope, SqlValidatorImpl validator) {
     JoinScope joinScope = (JoinScope) validator.getJoinScope(join);
    // BODO CHANGE:
    // Several changes were made to this function to enable properly expanding
    // USING columns.

    //List of joinChildren that have not yet been traversed when looking for instances of identifier
    final List<ScopeChild> joinChildrenToBeTraversed = new ArrayList<>(requireNonNull(joinScope, "scope").children);
    //List of possible expansions of the input identifier
    final List<SqlNode> qualifiedNode = new ArrayList<>();

    // Recursively visit both the left and right subjoins (if they exist)
    if ((join.getLeft() instanceof SqlJoin)) {
      SqlJoin leftSubJoin = (SqlJoin) join.getLeft();
      //Join scope is a ListScope, so this should always be valid
      JoinScope leftSubJoinScope = (JoinScope) validator.getJoinScope(leftSubJoin);
      SqlNode unExpandedOutput = expandExprFromJoin(leftSubJoin, identifier, scope, validator);
      if (identifier != unExpandedOutput) {
        qualifiedNode.add(unExpandedOutput);
        // Remove the already traversed children from joinChildrenToBeTraversed
        for (int i = 0; i < leftSubJoinScope.getChildren().size(); i++){
          //Left children are always first
          joinChildrenToBeTraversed.remove(0);
        }
      }
    }
    if ((join.getRight() instanceof SqlJoin)) {
      SqlJoin rightSubJoin = (SqlJoin) join.getRight();
      //Join scope is a ListScope, so this should always be valid
      JoinScope rightSubJoinScope = (JoinScope) validator.getJoinScope(rightSubJoin);
      identifier = (SqlIdentifier) expandExprFromJoin(rightSubJoin, identifier, scope, validator);
      SqlNode unExpandedOutput = expandExprFromJoin(rightSubJoin, identifier, scope, validator);
      if (identifier != unExpandedOutput){
        qualifiedNode.add(unExpandedOutput);
        for (int i = 0; i < rightSubJoinScope.getChildren().size(); i++){
          //right children are always last
          joinChildrenToBeTraversed.remove(joinChildrenToBeTraversed.size() - 1);
        }
      }
    }

    if (join.getConditionType() != JoinConditionType.USING) {
      //Confirm that the identifier does not exist on both sides of the table

      for (ScopeChild child : joinChildrenToBeTraversed) {
        if (child.namespace.getRowType().getFieldNames().contains(identifier.getSimple())) {
          qualifiedNode.add(identifier);
        }
      }

      if (qualifiedNode.size() > 1) {
        throw validator.newValidationError(identifier, RESOURCE.columnAmbiguous(identifier.toString()));
      } else if (qualifiedNode.size() == 1) {
        return qualifiedNode.get(0);
      } else {
        //This could be an alias that has yet to be expanded, so just return the input identifier
        // and let the later code throw the error if needed
        return identifier;
      }
    }

    final Map<String, String> fieldAliases = getFieldAliases(scope);

    // USING join case.
    // Iterate through all of the children that we haven't already visited,
    // and find all instances of the identifier.
    for (String name
        : SqlIdentifier.simpleNames((SqlNodeList) getCondition(join))) {
      if (identifier.getSimple().equals(name)) {

        for (ScopeChild child : joinChildrenToBeTraversed) {
          if (child.namespace.getRowType().getFieldNames().contains(name)) {
            final SqlIdentifier exp =
                new SqlIdentifier(
                    ImmutableList.of(child.name, name),
                    identifier.getParserPosition());
            qualifiedNode.add(exp);
          }
        }

        //If we have more than two instances, we have an ambiguous column.
        if (qualifiedNode.size() != 2){
          throw validator.newValidationError(identifier, RESOURCE.columnAmbiguous(identifier.toString()));
        }


        //coalesce is necessary to properly handle outer joins.
        assert qualifiedNode.size() == 2;

        // If there is an alias for the column, no need to wrap the coalesce with an AS operator
        boolean haveAlias = fieldAliases.containsKey(name);

        final SqlCall coalesceCall =
            SqlStdOperatorTable.COALESCE.createCall(SqlParserPos.ZERO, qualifiedNode.get(0),
                qualifiedNode.get(1));

        if (haveAlias) {
          return coalesceCall;
        } else {
          return SqlStdOperatorTable.AS.createCall(SqlParserPos.ZERO, coalesceCall,
              new SqlIdentifier(name, SqlParserPos.ZERO));
        }
      }
    }

    // Only need to try to expand the expr from the left input of join
    // since it is always left-deep join.
    final SqlNode node = join.getLeft();
    if (node instanceof SqlJoin) {
      return expandExprFromJoin((SqlJoin) node, identifier, scope, validator);
    } else {
      return identifier;
    }
  }
  private static Map<String, String> getFieldAliases(final SelectScope scope) {
    final ImmutableMap.Builder<String, String> fieldAliases = new ImmutableMap.Builder<>();

    for (SqlNode selectItem : scope.getNode().getSelectList()) {
      if (selectItem instanceof SqlCall) {
        final SqlCall call = (SqlCall) selectItem;
        if (!(call.getOperator() instanceof SqlAsOperator)
            || !(call.operand(0) instanceof SqlIdentifier)) {
          continue;
        }

        final SqlIdentifier fieldIdentifier = call.operand(0);
        // Bodo Change: Can't support complex expressions in field aliases
        if (!fieldIdentifier.isSimple()) {
          continue;
        }
        fieldAliases.put(fieldIdentifier.getSimple(),
            ((SqlIdentifier) call.operand(1)).getSimple());
      }
    }

    return fieldAliases.build();
  }

  /** Returns the set of field names in the join condition specified by USING
   * or implicitly by NATURAL, de-duplicated and in order. */
  public @Nullable List<String> usingNames(SqlJoin join) {
    switch (join.getConditionType()) {
    case USING:
      SqlNodeList condition = (SqlNodeList) getCondition(join);
      List<String> simpleNames = SqlIdentifier.simpleNames(condition);
      return catalogReader.nameMatcher().distinctCopy(simpleNames);
    case NONE:
      if (join.isNatural()) {
        return deriveNaturalJoinColumnList(join);
      }
      return null;
    default:
      return null;
    }
  }

  private List<String> deriveNaturalJoinColumnList(SqlJoin join) {
    return SqlValidatorUtil.deriveNaturalJoinColumnList(
        catalogReader.nameMatcher(),
        getNamespaceOrThrow(join.getLeft()).getRowType(),
        getNamespaceOrThrow(join.getRight()).getRowType());
  }

  private static SqlNode expandCommonColumn(SqlSelect sqlSelect,
      SqlNode selectItem, SelectScope scope, SqlValidatorImpl validator) {
    if (!(selectItem instanceof SqlIdentifier)) {
      return selectItem;
    }

    final SqlNode from = sqlSelect.getFrom();
    if (!(from instanceof SqlJoin)) {
      return selectItem;
    }

    final SqlIdentifier identifier = (SqlIdentifier) selectItem;
    if (!identifier.isSimple()) {
      if (!validator.config().conformance().allowQualifyingCommonColumn()) {
        validateQualifiedCommonColumn((SqlJoin) from, identifier, scope, validator);
      }
      return selectItem;
    }

    assert validator.getJoinScope(from) instanceof JoinScope: "Error in expandCommonColumn: scope is not a JoinScope";
    return expandExprFromJoin((SqlJoin) from, identifier, scope, validator);
  }

  private static void validateQualifiedCommonColumn(SqlJoin join,
      SqlIdentifier identifier, SelectScope scope, SqlValidatorImpl validator) {
    List<String> names = validator.usingNames(join);
    if (names == null) {
      // Not USING or NATURAL.
      return;
    }

    // First we should make sure that the first component is the table name.
    // Then check whether the qualified identifier contains common column.
    for (ScopeChild child : scope.children) {
      if (Objects.equals(child.name, identifier.getComponent(0).toString())) {
        if (names.contains(identifier.getComponent(1).toString())) {
          throw validator.newValidationError(identifier,
              RESOURCE.disallowsQualifyingCommonColumn(identifier.toString()));
        }
      }
    }

    // Only need to try to validate the expr from the left input of join
    // since it is always left-deep join.
    final SqlNode node = join.getLeft();
    if (node instanceof SqlJoin) {
      validateQualifiedCommonColumn((SqlJoin) node, identifier, scope, validator);
    }
  }

  private boolean expandStar(List<SqlNode> selectItems, Set<String> aliases,
      PairList<String, RelDataType> fields, boolean includeSystemVars,
      SelectScope scope, SqlNode node) {
    if (!(node instanceof SqlIdentifier)) {
      return false;
    }
    final SqlIdentifier identifier = (SqlIdentifier) node;
    if (!identifier.isStar()) {
      return false;
    }
    final int originalSize = selectItems.size();
    final SqlParserPos startPosition = identifier.getParserPosition();
    switch (identifier.names.size()) {
    case 1:
      SqlNode from = scope.getNode().getFrom();
      if (from == null) {
        throw newValidationError(identifier, RESOURCE.selectStarRequiresFrom());
      }

      boolean hasDynamicStruct = false;
      for (ScopeChild child : scope.children) {
        final int before = fields.size();
        if (child.namespace.getRowType().isDynamicStruct()) {
          hasDynamicStruct = true;
          // don't expand star if the underneath table is dynamic.
          // Treat this star as a special field in validation/conversion and
          // wait until execution time to expand this star.
          final SqlNode exp =
              new SqlIdentifier(
                  ImmutableList.of(child.name,
                      DynamicRecordType.DYNAMIC_STAR_PREFIX),
                  startPosition);
          addToSelectList(
               selectItems,
               aliases,
               fields,
               exp,
               scope,
               includeSystemVars);
        } else {
          final SqlNode from2 = SqlNonNullableAccessors.getNode(child);
          final SqlValidatorNamespace fromNs = getNamespaceOrThrow(from2, scope);
          final RelDataType rowType = fromNs.getRowType();
          for (RelDataTypeField field : rowType.getFieldList()) {
            String columnName = field.getName();

            // TODO: do real implicit collation here
            final SqlIdentifier exp =
                new SqlIdentifier(
                    ImmutableList.of(child.name, columnName),
                    startPosition);
            // Don't add expanded rolled up columns
            if (!isRolledUpColumn(exp, scope)) {
              addOrExpandField(
                      selectItems,
                      aliases,
                      fields,
                      includeSystemVars,
                      scope,
                      exp,
                      field);
            }
          }
        }
        if (child.nullable) {
          for (int i = before; i < fields.size(); i++) {
            final Map.Entry<String, RelDataType> entry = fields.get(i);
            final RelDataType type = entry.getValue();
            if (!type.isNullable()) {
              fields.set(i,
                  entry.getKey(),
                  typeFactory.createTypeWithNullability(type, true));
            }
          }
        }
      }
      // If NATURAL JOIN or USING is present, move key fields to the front of
      // the list, per standard SQL. Disabled if there are dynamic fields.
      if (!hasDynamicStruct || Bug.CALCITE_2400_FIXED) {
        // If some fields before star identifier,
        // we should move offset.
        int offset = Math.min(calculatePermuteOffset(selectItems), originalSize);
        new Permute(from, offset).permute(selectItems, fields);
      }
      return true;

    default:
      final SqlIdentifier prefixId = identifier.skipLast(1);
      final SqlValidatorScope.ResolvedImpl resolved =
          new SqlValidatorScope.ResolvedImpl();
      final SqlNameMatcher nameMatcher =
          scope.validator.catalogReader.nameMatcher();
      scope.resolve(prefixId.names, nameMatcher, true, resolved);
      if (resolved.count() == 0) {
        // e.g. "select s.t.* from e"
        // or "select r.* from e"
        throw newValidationError(prefixId,
            RESOURCE.unknownIdentifier(prefixId.toString()));
      }
      final RelDataType rowType = resolved.only().rowType();
      if (rowType.isDynamicStruct()) {
        // don't expand star if the underneath table is dynamic.
        addToSelectList(
            selectItems,
            aliases,
            fields,
            prefixId.plus(DynamicRecordType.DYNAMIC_STAR_PREFIX, startPosition),
            scope,
            includeSystemVars);
      } else if (rowType.isStruct()) {
        for (RelDataTypeField field : rowType.getFieldList()) {
          String columnName = field.getName();

          // TODO: do real implicit collation here
          addOrExpandField(
              selectItems,
              aliases,
              fields,
              includeSystemVars,
              scope,
              prefixId.plus(columnName, startPosition),
              field);
        }
      } else {
        throw newValidationError(prefixId, RESOURCE.starRequiresRecordType());
      }
      return true;
    }
  }

  /**
   * @brief Checks whether a SqlNode is a function call containing a star
   * operand that should be expanded to variadic arguments, e.g. HASH(*).
   * @param call A SqlCall object being checked for the property.
   * @return True if the node is a call in the desired format.
   */
  protected boolean isStarCall(SqlCall call) { return false; }

  /**
   * @brief Rewrites a SqlCall with a new argument list after it has
   * had its original arguments expanded due to the presence of * terms.
   * @param call A SqlCall object being transformed.
   * @param newArgs The expanded arguments to the original SqlCall.
   * @param newColumnNames The names of the expanded arguments.
   * @return A transformed SqlCall object.
   */
  protected SqlCall rewriteStarCall(SqlCall call, List<SqlNode> newArgs, List<String> newColumnNames) {
    throw new NotImplementedException("Need to implement rewriteStarCall to use star expansion");
  }

  /**
   * Expands the a "*" term into the columns of all tables in scope, e.g. HASH(*) becomes
   * HASH(T1.A, T1.B, T2.A, T2.E, T2.I, T3.X, T3.Y)
   * @param outList The list where new operands are appended to.
   * @param outNames The list where new names are appended to.
   * @param scope The scope containing all tables that a "*" term could refer to
   * @param starNode The node being expanded
   */
  protected void expandStarTermWithoutTable(List<SqlNode> outList, List<String> outNames, ListScope scope, SqlIdentifier starNode) {
    final SqlParserPos startPosition = starNode.getParserPosition();
    for (int c = 0; c < scope.children.size(); c++) {
      final SqlValidatorNamespace fromNs = scope.getChildren().get(c);
      final String fromName = scope.getChildNames().get(c);
      final RelDataType rowType = fromNs.getRowType();
      for (RelDataTypeField field : rowType.getFieldList()) {
        String columnName = field.getName();
        final SqlIdentifier exp =
                new SqlIdentifier(ImmutableList.of(fromName, columnName), startPosition);
        outNames.add(columnName);
        outList.add(exp);
      }
    }
  }

  /**
   * Expands the a "t.*" term into the columns of a specified table, e.g. HASH(T1.*) becomes
   * HASH(T1.A, T1.B)
   * @param outList The list where new operands are appended to.
   * @param outNames The list where new names are appended to.
   * @param scope The scope containing all tables that a "*" term could refer to
   * @param starNode The node being expanded
   */
  protected void expandStarTermWithTable(List<SqlNode> outList, List<String> outNames, ListScope scope, SqlIdentifier starNode) {
    final SqlParserPos startPosition = starNode.getParserPosition();
    final SqlIdentifier prefixId = starNode.skipLast(1);
    final SqlValidatorScope.ResolvedImpl resolved = new SqlValidatorScope.ResolvedImpl();
    final SqlNameMatcher nameMatcher = scope.getValidator().getCatalogReader().nameMatcher();
    scope.resolve(prefixId.names, nameMatcher, true, resolved);
    if (resolved.count() == 0) {
      // e.g. "select s.t.* from e" or "select r.* from e"
      throw newValidationError(prefixId, RESOURCE.unknownIdentifier(prefixId.toString()));
    }
    final RelDataType rowType = resolved.only().rowType();
    if (rowType.isDynamicStruct()) {
      // TODO: add column names in this case
      outList.add(prefixId.plus(DynamicRecordType.DYNAMIC_STAR_PREFIX, startPosition));
    } else if (rowType.isStruct()) {
      for (RelDataTypeField field : rowType.getFieldList()) {
        String columnName = field.getName();
        outNames.add(columnName);
        outList.add(prefixId.plus(columnName, startPosition));
      }
    } else {
      throw newValidationError(prefixId, RESOURCE.starRequiresRecordType());
    }
  }

  /**
   * @brief Expands a list of nodes so any "*" or "t.*" which are syntactic sugar
   * to refer to multiple columns are replaced with all of said columns.
   * @param nodesToExpand The sequence of SqlNodes that may or may not be star terms.
   * @param scope The scope that is used to look up what column names the * refers to.
   * @param newNodes Where all the expanded SqlNodes are stored.
   * @param newNames Where all the expanded column names are stored.
   */
  protected void expandStarNodes(List<SqlNode> nodesToExpand, SqlValidatorScope scope, List<SqlNode> newNodes, List<String> newNames) {
    while (!(scope instanceof ListScope)) {
      if (scope instanceof DelegatingScope && ((DelegatingScope)scope).parent != null) {
        scope = ((DelegatingScope)scope).parent;
      } else {
        throw new RuntimeException("Error: Call to function with * arguments must happen inside of a scope whose operand scope is a ListScope or whose ancestry contains a ListScope.");
      }
    }
    ListScope trueScope = (ListScope) scope;
    for (SqlNode operand : nodesToExpand) {
      if ((operand instanceof SqlIdentifier) && ((SqlIdentifier) operand).isStar()) {
        SqlIdentifier starNode = (SqlIdentifier)operand;
        switch (starNode.names.size()) {
          case 1: {
            expandStarTermWithoutTable(newNodes, newNames, trueScope, starNode);
            break;
          }
          default: {
            expandStarTermWithTable(newNodes, newNames, trueScope, starNode);
            break;
          }
        }
      } else {
        newNodes.add(operand);
      }
    }
  }


  /**
   * @brief Expands the operands to a SqlCall that can use "*" or "t.*"
   * as syntactic sugar to refer to multiple columns.
   * @param call The call that is being expanded.
   * @param scope The scope of the function call.
   * @return The call transformed to have any "*" or "t.*" terms expanded.
   */
  private SqlCall starExpansion(SqlCall call, SqlValidatorScope scope) {
    List<SqlNode> newArguments = new ArrayList<SqlNode>();
    List<String> newColumnNames = new ArrayList<>();
    expandStarNodes(call.getOperandList(), scope, newArguments, newColumnNames);
    return rewriteStarCall(call, newArguments, newColumnNames);
  }

  private static int calculatePermuteOffset(List<SqlNode> selectItems) {
    for (int i = 0; i < selectItems.size(); i++) {
      SqlNode selectItem = selectItems.get(i);
      SqlNode col = stripAs(selectItem);
      if (col.getKind() == SqlKind.IDENTIFIER
          && selectItem.getKind() != SqlKind.AS) {
        return i;
      }
    }
    return 0;
  }

  private SqlNode maybeCast(SqlNode node, RelDataType currentType,
      RelDataType desiredType) {
    return SqlTypeUtil.equalSansNullability(typeFactory, currentType, desiredType)
        ? node
        : SqlStdOperatorTable.CAST.createCall(SqlParserPos.ZERO,
            node, SqlTypeUtil.convertTypeToSpec(desiredType));
  }

  private boolean addOrExpandField(List<SqlNode> selectItems, Set<String> aliases,
      PairList<String, RelDataType> fields, boolean includeSystemVars,
      SelectScope scope, SqlIdentifier id, RelDataTypeField field) {
    switch (field.getType().getStructKind()) {
    case PEEK_FIELDS:
    case PEEK_FIELDS_DEFAULT:
      final SqlNode starExp = id.plusStar();
      expandStar(
          selectItems,
          aliases,
          fields,
          includeSystemVars,
          scope,
          starExp);
      return true;

    default:
      addToSelectList(
          selectItems,
          aliases,
          fields,
          id,
          scope,
          includeSystemVars);
    }

    return false;
  }

  @Override public SqlNode validate(SqlNode topNode) {
    SqlValidatorScope scope = new EmptyScope(this);
    scope = new CatalogScope(scope, ImmutableList.of("CATALOG"));
    final SqlNode topNode2 = validateScopedExpression(topNode, scope);
    final RelDataType type = getValidatedNodeType(topNode2);
    Util.discard(type);
    return topNode2;
  }

  @Override public List<SqlMoniker> lookupHints(SqlNode topNode, SqlParserPos pos) {
    SqlValidatorScope scope = new EmptyScope(this);
    SqlNode outermostNode = performUnconditionalRewrites(topNode, false);
    cursorSet.add(outermostNode);
    if (outermostNode.isA(SqlKind.TOP_LEVEL)) {
      registerQuery(
          scope,
          null,
          outermostNode,
          outermostNode,
          null,
          false);
    }
    final SqlValidatorNamespace ns = getNamespace(outermostNode);
    if (ns == null) {
      throw new AssertionError("Not a query: " + outermostNode);
    }
    Collection<SqlMoniker> hintList = Sets.newTreeSet(SqlMoniker.COMPARATOR);
    lookupSelectHints(ns, pos, hintList);
    return ImmutableList.copyOf(hintList);
  }

  @Override public @Nullable SqlMoniker lookupQualifiedName(SqlNode topNode, SqlParserPos pos) {
    final String posString = pos.toString();
    IdInfo info = idPositions.get(posString);
    if (info != null) {
      final SqlQualified qualified = info.scope.fullyQualify(info.id);
      return new SqlIdentifierMoniker(qualified.identifier);
    } else {
      return null;
    }
  }

  /**
   * Looks up completion hints for a syntactically correct select SQL that has
   * been parsed into an expression tree.
   *
   * @param select   the Select node of the parsed expression tree
   * @param pos      indicates the position in the sql statement we want to get
   *                 completion hints for
   * @param hintList list of {@link SqlMoniker} (sql identifiers) that can
   *                 fill in at the indicated position
   */
  void lookupSelectHints(
      SqlSelect select,
      SqlParserPos pos,
      Collection<SqlMoniker> hintList) {
    IdInfo info = idPositions.get(pos.toString());
    if (info == null) {
      SqlNode fromNode = select.getFrom();
      final SqlValidatorScope fromScope = getFromScope(select);
      lookupFromHints(fromNode, fromScope, pos, hintList);
    } else {
      lookupNameCompletionHints(info.scope, info.id.names,
          info.id.getParserPosition(), hintList);
    }
  }

  private void lookupSelectHints(
      SqlValidatorNamespace ns,
      SqlParserPos pos,
      Collection<SqlMoniker> hintList) {
    final SqlNode node = ns.getNode();
    if (node instanceof SqlSelect) {
      lookupSelectHints((SqlSelect) node, pos, hintList);
    }
  }

  private void lookupFromHints(
      @Nullable SqlNode node,
      SqlValidatorScope scope,
      SqlParserPos pos,
      Collection<SqlMoniker> hintList) {
    if (node == null) {
      // This can happen in cases like "select * _suggest_", so from clause is absent
      return;
    }
    final SqlValidatorNamespace ns = getNamespaceOrThrow(node);
    if (ns.isWrapperFor(IdentifierNamespace.class)) {
      IdentifierNamespace idNs = ns.unwrap(IdentifierNamespace.class);
      final SqlIdentifier id = idNs.getId();
      for (int i = 0; i < id.names.size(); i++) {
        if (pos.toString().equals(
            id.getComponent(i).getParserPosition().toString())) {
          final List<SqlMoniker> objNames = new ArrayList<>();
          SqlValidatorUtil.getSchemaObjectMonikers(
              getCatalogReader(),
              id.names.subList(0, i + 1),
              objNames);
          for (SqlMoniker objName : objNames) {
            if (objName.getType() != SqlMonikerType.FUNCTION) {
              hintList.add(objName);
            }
          }
          return;
        }
      }
    }
    switch (node.getKind()) {
    case JOIN:
      lookupJoinHints((SqlJoin) node, scope, pos, hintList);
      break;
    default:
      lookupSelectHints(ns, pos, hintList);
      break;
    }
  }

  private void lookupJoinHints(
      SqlJoin join,
      SqlValidatorScope scope,
      SqlParserPos pos,
      Collection<SqlMoniker> hintList) {
    SqlNode left = join.getLeft();
    SqlNode right = join.getRight();
    SqlNode condition = join.getCondition();
    lookupFromHints(left, scope, pos, hintList);
    if (!hintList.isEmpty()) {
      return;
    }
    lookupFromHints(right, scope, pos, hintList);
    if (!hintList.isEmpty()) {
      return;
    }
    final JoinConditionType conditionType = join.getConditionType();
    switch (conditionType) {
    case ON:
      requireNonNull(condition, () -> "join.getCondition() for " + join)
          .findValidOptions(this,
              getScopeOrThrow(join),
              pos, hintList);
      return;
    default:

      // No suggestions.
      // Not supporting hints for other types such as 'Using' yet.
    }
  }

  /**
   * Populates a list of all the valid alternatives for an identifier.
   *
   * @param scope    Validation scope
   * @param names    Components of the identifier
   * @param pos      position
   * @param hintList a list of valid options
   */
  public final void lookupNameCompletionHints(
      SqlValidatorScope scope,
      List<String> names,
      SqlParserPos pos,
      Collection<SqlMoniker> hintList) {
    // Remove the last part of name - it is a dummy
    List<String> subNames = Util.skipLast(names);

    if (subNames.size() > 0) {
      // If there's a prefix, resolve it to a namespace.
      SqlValidatorNamespace ns = null;
      for (String name : subNames) {
        if (ns == null) {
          final SqlValidatorScope.ResolvedImpl resolved =
              new SqlValidatorScope.ResolvedImpl();
          final SqlNameMatcher nameMatcher = catalogReader.nameMatcher();
          scope.resolve(ImmutableList.of(name), nameMatcher, false, resolved);
          if (resolved.count() == 1) {
            ns = resolved.only().namespace;
          }
        } else {
          ns = ns.lookupChild(name);
        }
        if (ns == null) {
          break;
        }
      }
      if (ns != null) {
        RelDataType rowType = ns.getRowType();
        if (rowType.isStruct()) {
          for (RelDataTypeField field : rowType.getFieldList()) {
            hintList.add(
                new SqlMonikerImpl(
                    field.getName(),
                    SqlMonikerType.COLUMN));
          }
        }
      }

      // builtin function names are valid completion hints when the
      // identifier has only 1 name part
      findAllValidFunctionNames(names, this, hintList, pos);
    } else {
      // No prefix; use the children of the current scope (that is,
      // the aliases in the FROM clause)
      scope.findAliases(hintList);

      // If there's only one alias, add all child columns
      SelectScope selectScope =
          SqlValidatorUtil.getEnclosingSelectScope(scope);
      if ((selectScope != null)
          && (selectScope.getChildren().size() == 1)) {
        RelDataType rowType =
            selectScope.getChildren().get(0).getRowType();
        for (RelDataTypeField field : rowType.getFieldList()) {
          hintList.add(
              new SqlMonikerImpl(
                  field.getName(),
                  SqlMonikerType.COLUMN));
        }
      }
    }

    findAllValidUdfNames(names, this, hintList);
  }

  private static void findAllValidUdfNames(
      List<String> names,
      SqlValidator validator,
      Collection<SqlMoniker> result) {
    final List<SqlMoniker> objNames = new ArrayList<>();
    SqlValidatorUtil.getSchemaObjectMonikers(
        validator.getCatalogReader(),
        names,
        objNames);
    for (SqlMoniker objName : objNames) {
      if (objName.getType() == SqlMonikerType.FUNCTION) {
        result.add(objName);
      }
    }
  }

  private static void findAllValidFunctionNames(
      List<String> names,
      SqlValidator validator,
      Collection<SqlMoniker> result,
      SqlParserPos pos) {
    // a function name can only be 1 part
    if (names.size() > 1) {
      return;
    }
    for (SqlOperator op : validator.getOperatorTable().getOperatorList()) {
      SqlIdentifier curOpId =
          new SqlIdentifier(
              op.getName(),
              pos);

      final SqlCall call = validator.makeNullaryCall(curOpId);
      if (call != null) {
        result.add(
            new SqlMonikerImpl(
                op.getName(),
                SqlMonikerType.FUNCTION));
      } else {
        if ((op.getSyntax() == SqlSyntax.FUNCTION)
            || (op.getSyntax() == SqlSyntax.PREFIX)) {
          if (op.getOperandTypeChecker() != null) {
            String sig = op.getAllowedSignatures();
            sig = sig.replace("'", "");
            result.add(
                new SqlMonikerImpl(
                    sig,
                    SqlMonikerType.FUNCTION));
            continue;
          }
          result.add(
              new SqlMonikerImpl(
                  op.getName(),
                  SqlMonikerType.FUNCTION));
        }
      }
    }
  }

  @Override public SqlNode validateParameterizedExpression(
      SqlNode topNode,
      final Map<String, RelDataType> nameToTypeMap) {
    SqlValidatorScope scope = new ParameterScope(this, nameToTypeMap);
    return validateScopedExpression(topNode, scope);
  }

  private SqlNode validateScopedExpression(
      SqlNode topNode,
      SqlValidatorScope scope) {
    SqlNode outermostNode = performUnconditionalRewrites(topNode, false);
    cursorSet.add(outermostNode);
    top = outermostNode;
    TRACER.trace("After unconditional rewrite: {}", outermostNode);
    if (outermostNode.isA(SqlKind.TOP_LEVEL)) {
      registerQuery(scope, null, outermostNode, outermostNode, null, false);
    }
    outermostNode.validate(this, scope);
    if (!outermostNode.isA(SqlKind.TOP_LEVEL)) {
      // force type derivation so that we can provide it to the
      // caller later without needing the scope
      deriveType(scope, outermostNode);
    }
    TRACER.trace("After validation: {}", outermostNode);
    return outermostNode;
  }

  @Override public void validateQuery(SqlNode node, SqlValidatorScope scope,
      RelDataType targetRowType) {
    final SqlValidatorNamespace ns = getNamespaceOrThrow(node, scope);
    if (node.getKind() == SqlKind.TABLESAMPLE) {
      List<SqlNode> operands = ((SqlCall) node).getOperandList();
      SqlSampleSpec sampleSpec = SqlLiteral.sampleValue(operands.get(1));
      if (sampleSpec instanceof SqlSampleSpec.SqlTableSampleSpec) {
        // The sampling percentage must be between 0 (0%) and 1 (100%).
        BigDecimal samplePercentage =
            ((SqlSampleSpec.SqlTableSampleSpec) sampleSpec).sampleRate;
        // Check the samplePercentage whether is between 0 and 1
        if (samplePercentage.compareTo(BigDecimal.ZERO) < 0
            || samplePercentage.compareTo(BigDecimal.ONE) > 0) {
          throw SqlUtil.newContextException(node.getParserPosition(),
              RESOURCE.invalidSampleSize());
        }
        validateFeature(RESOURCE.sQLFeature_T613(), node.getParserPosition());
      } else if (sampleSpec
          instanceof SqlSampleSpec.SqlSubstitutionSampleSpec) {
        validateFeature(RESOURCE.sQLFeatureExt_T613_Substitution(),
            node.getParserPosition());
      }
    }

    validateNamespace(ns, targetRowType);
    switch (node.getKind()) {
    case EXTEND:
      // Until we have a dedicated namespace for EXTEND
      deriveType(requireNonNull(scope, "scope"), node);
      break;
    default:
      break;
    }
    if (node == top && !config.embeddedQuery()) {
      validateModality(node);
    }
    validateAccess(
        node,
        ns.getTable(),
        SqlAccessEnum.SELECT);

    validateSnapshot(node, scope, ns);
  }

  /**
   * Validates a namespace.
   *
   * @param namespace Namespace
   * @param targetRowType Desired row type, must not be null, may be the data
   *                      type 'unknown'.
   */
  protected void validateNamespace(final SqlValidatorNamespace namespace,
      RelDataType targetRowType) {
    namespace.validate(targetRowType);
    final SqlNode node = namespace.getNode();
    if (node != null) {
      RelDataType type = namespace.getType();

      if (node == top) {
        // A top-level namespace must not return any must-filter fields.
        // A non-top-level namespace (e.g. a subquery) may return must-filter
        // fields; these are neutralized if the consuming query filters on them.
        final ImmutableBitSet mustFilterFields =
            namespace.getMustFilterFields();
        if (!mustFilterFields.isEmpty()) {
          // Set of field names, sorted alphabetically for determinism.
          Set<String> fieldNameSet =
              StreamSupport.stream(mustFilterFields.spliterator(), false)
                  .map(namespace.getRowType().getFieldNames()::get)
                  .collect(Collectors.toCollection(TreeSet::new));
          throw newValidationError(node,
              RESOURCE.mustFilterFieldsMissing(fieldNameSet.toString()));
        }

        if (!config.embeddedQuery()) {
          type = SqlTypeUtil.fromMeasure(typeFactory, type);
        }
      }
      setValidatedNodeType(node, type);
    }
  }

  @Override public SqlValidatorScope getEmptyScope() {
    return new EmptyScope(this);
  }

  private SqlValidatorScope getScope(SqlSelect select, Clause clause) {
    return requireNonNull(
        clauseScopes.get(IdPair.of(select, clause)),
        () -> "no " + clause + " scope for " + select);
  }

  public SqlValidatorScope getCursorScope(SqlSelect select) {
    return getScope(select, Clause.CURSOR);
  }

  @Override public SqlValidatorScope getWhereScope(SqlSelect select) {
    return getScope(select, Clause.WHERE);
  }

  @Override public SqlValidatorScope getSelectScope(SqlSelect select) {
    return getScope(select, Clause.SELECT);
  }

  @Override public SqlValidatorScope getMeasureScope(SqlSelect select) {
    return getScope(select, Clause.MEASURE);
  }

  @Override public @Nullable SelectScope getRawSelectScope(SqlSelect select) {
    SqlValidatorScope scope = clauseScopes.get(IdPair.of(select, Clause.SELECT));
    if (scope instanceof AggregatingSelectScope) {
      scope = ((AggregatingSelectScope) scope).getParent();
    }
    return (SelectScope) scope;
  }

  private SelectScope getRawSelectScopeNonNull(SqlSelect select) {
    return requireNonNull(getRawSelectScope(select),
        () -> "getRawSelectScope for " + select);
  }

  @Override public SqlValidatorScope getHavingScope(SqlSelect select) {
    // Yes, it's the same as getSelectScope
    return getScope(select, Clause.SELECT);
  }

  @Override public SqlValidatorScope getGroupScope(SqlSelect select) {
    // Yes, it's the same as getWhereScope
    return getScope(select, Clause.WHERE);
  }

  @Override public SqlValidatorScope getFromScope(SqlSelect select) {
    return requireNonNull(scopes.get(select),
        () -> "no scope for " + select);
  }


  @Override public SqlValidatorScope getCreateTableScope(SqlCreateTable createTable) {
    return requireNonNull(scopes.get(createTable));
  }

  @Override public SqlValidatorScope getOrderScope(SqlSelect select) {
    return getScope(select, Clause.ORDER);
  }

  @Override public SqlValidatorScope getMatchRecognizeScope(SqlMatchRecognize node) {
    return getScopeOrThrow(node);
  }

  @Override public SqlValidatorScope getLambdaScope(SqlLambda node) {
    return getScopeOrThrow(node);
  }

  @Override public SqlValidatorScope getJoinScope(SqlNode node) {
    return requireNonNull(scopes.get(stripAs(node)),
        () -> "scope for " + node);
  }

  @Override public SqlValidatorScope getOverScope(SqlNode node) {
    return getScopeOrThrow(node);
  }

  @Override public SqlValidatorScope getWithScope(SqlNode withItem) {
    assert withItem.getKind() == SqlKind.WITH_ITEM;
    return getScopeOrThrow(withItem);
  }

  private SqlValidatorScope getScopeOrThrow(SqlNode node) {
    return requireNonNull(scopes.get(node), () -> "scope for " + node);
  }

  private @Nullable SqlValidatorNamespace getNamespace(SqlNode node,
      SqlValidatorScope scope) {
    if (node instanceof SqlIdentifier && scope instanceof DelegatingScope) {
      final SqlIdentifier id = (SqlIdentifier) node;
      final SqlValidatorScope idScope =
          ((DelegatingScope) scope).getParent();
      return getNamespace(id, idScope);
    } else if (node instanceof SqlCall) {
      // Handle extended identifiers.
      final SqlCall call = (SqlCall) node;
      switch (call.getOperator().getKind()) {
      case TABLE_REF:
        return getNamespace(call.operand(0), scope);
      case EXTEND:
        final SqlNode operand0 = call.getOperandList().get(0);
        final SqlIdentifier identifier = operand0.getKind() == SqlKind.TABLE_REF
            ? ((SqlCall) operand0).operand(0)
            : (SqlIdentifier) operand0;
        final DelegatingScope idScope = (DelegatingScope) scope;
        return getNamespace(identifier, idScope);
      case AS:
        final SqlNode nested = call.getOperandList().get(0);
        switch (nested.getKind()) {
        case TABLE_REF:
        case EXTEND:
          return getNamespace(nested, scope);
        default:
          break;
        }
        break;
      default:
        break;
      }
    }
    return getNamespace(node);
  }

  private @Nullable SqlValidatorNamespace getNamespace(SqlIdentifier id,
      @Nullable DelegatingScope scope) {
    if (id.isSimple()) {
      final SqlNameMatcher nameMatcher = catalogReader.nameMatcher();
      final SqlValidatorScope.ResolvedImpl resolved =
          new SqlValidatorScope.ResolvedImpl();
      requireNonNull(scope, () -> "scope needed to lookup " + id)
          .resolve(id.names, nameMatcher, false, resolved);
      if (resolved.count() == 1) {
        return resolved.only().namespace;
      }
    }
    return getNamespace(id);
  }

  @Override public @Nullable SqlValidatorNamespace getNamespace(SqlNode node) {
    switch (node.getKind()) {
    case AS:

      // AS has a namespace if it has a column list 'AS t (c1, c2, ...)'
      final SqlValidatorNamespace ns = namespaces.get(node);
      if (ns != null) {
        return ns;
      }
      // fall through
    case TABLE_REF:
    case TABLE_REF_WITH_ID:
    case SNAPSHOT:
    case OVER:
    case COLLECTION_TABLE:
    case ORDER_BY:
    case TABLESAMPLE:
      return getNamespace(((SqlCall) node).operand(0));
    default:
      return namespaces.get(node);
    }
  }

  /**
   * Namespace for the given node.
   *
   * @param node node to compute the namespace for
   * @return namespace for the given node, never null
   * @see #getNamespace(SqlNode)
   */
  @API(since = "1.27", status = API.Status.INTERNAL)
  SqlValidatorNamespace getNamespaceOrThrow(SqlNode node) {
    return requireNonNull(
        getNamespace(node),
        () -> "namespace for " + node);
  }

  /**
   * Namespace for the given node.
   *
   * @param node node to compute the namespace for
   * @param scope namespace scope
   * @return namespace for the given node, never null
   * @see #getNamespace(SqlNode)
   */
  @API(since = "1.27", status = API.Status.INTERNAL)
  SqlValidatorNamespace getNamespaceOrThrow(SqlNode node,
      SqlValidatorScope scope) {
    return requireNonNull(
        getNamespace(node, scope),
        () -> "namespace for " + node + ", scope " + scope);
  }

  /**
   * Namespace for the given node.
   *
   * @param id identifier to resolve
   * @param scope namespace scope
   * @return namespace for the given node, never null
   * @see #getNamespace(SqlIdentifier, DelegatingScope)
   */
  @API(since = "1.26", status = API.Status.INTERNAL)
  SqlValidatorNamespace getNamespaceOrThrow(SqlIdentifier id,
      @Nullable DelegatingScope scope) {
    return requireNonNull(
        getNamespace(id, scope),
        () -> "namespace for " + id + ", scope " + scope);
  }

  private void handleOffsetFetch(@Nullable SqlNode offset, @Nullable SqlNode fetch) {
    if (offset instanceof SqlDynamicParam) {
      setValidatedNodeType(offset,
          typeFactory.createSqlType(SqlTypeName.INTEGER));
    }
    if (fetch instanceof SqlDynamicParam) {
      setValidatedNodeType(fetch,
          typeFactory.createSqlType(SqlTypeName.INTEGER));
    }
  }

  /**
   * Performs expression rewrites which are always used unconditionally. These
   * rewrites massage the expression tree into a standard form so that the
   * rest of the validation logic can be simpler.
   *
   * <p>Returns null if and only if the original expression is null.
   *
   * @param node      expression to be rewritten
   * @param underFrom whether node appears directly under a FROM clause
   * @return rewritten expression, or null if the original expression is null
   */
  protected @PolyNull SqlNode performUnconditionalRewrites(
      @PolyNull SqlNode node,
      boolean underFrom) {
    if (node == null) {
      return null;
    }

    // first transform operands and invoke generic call rewrite
    if (node instanceof SqlCall) {
      if (node instanceof SqlMerge) {
        validatingSqlMerge = true;
      }
      SqlCall call = (SqlCall) node;
      final SqlKind kind = call.getKind();
      final List<SqlNode> operands = call.getOperandList();
      for (int i = 0; i < operands.size(); i++) {
        SqlNode operand = operands.get(i);
        boolean childUnderFrom;
        if (kind == SqlKind.SELECT) {
          childUnderFrom = i == SqlSelect.FROM_OPERAND;
        } else if (kind == SqlKind.AS && (i == 0)) {
          // for an aliased expression, it is under FROM if
          // the AS expression is under FROM
          childUnderFrom = underFrom;
        } else {
          childUnderFrom = false;
        }

        SqlNode newOperand =
            performUnconditionalRewrites(operand, childUnderFrom);
        if (newOperand != null && newOperand != operand) {
          call.setOperand(i, newOperand);
        }
      }

      if (call.getOperator() instanceof SqlUnresolvedFunction) {
        assert call instanceof SqlBasicCall;
        final SqlUnresolvedFunction function =
            (SqlUnresolvedFunction) call.getOperator();
        // This function hasn't been resolved yet.  Perform
        // a half-hearted resolution now in case it's a
        // builtin function requiring special casing.  If it's
        // not, we'll handle it later during overload resolution.
        final List<SqlOperator> overloads = new ArrayList<>();
        opTab.lookupOperatorOverloads(function.getNameAsId(),
            function.getFunctionType(), SqlSyntax.FUNCTION, overloads,
            catalogReader.nameMatcher());
        // Bodo Change: HardCode on LISTAGG to avoid copying the operator table.
        // We don't seem to be able to overload this like regular functions.
        String funcName = function.getName().toUpperCase(Locale.ROOT);
        if (overloads.size() == 1 || (overloads.size() == 2 && (funcName.equals(SqlAggOperatorTable.LISTAGG.getName())))) {
          ((SqlBasicCall) call).setOperator(overloads.get(0));
        }
      }
      if (config.callRewrite()) {
        node = call.getOperator().rewriteCall(this, call);
      }
    } else if (node instanceof SqlNodeList) {
      final SqlNodeList list = (SqlNodeList) node;
      for (int i = 0; i < list.size(); i++) {
        SqlNode operand = list.get(i);
        SqlNode newOperand =
            performUnconditionalRewrites(
                operand,
                false);
        if (newOperand != null) {
          list.set(i, newOperand);
        }
      }
    }

    // now transform node itself
    final SqlKind kind = node.getKind();
    switch (kind) {
    case VALUES:
      // Do not rewrite VALUES clauses.
      // At some point we used to rewrite VALUES(...) clauses
      // to (SELECT * FROM VALUES(...)) but this was problematic
      // in various cases such as FROM (VALUES(...)) [ AS alias ]
      // where the rewrite was invoked over and over making the
      // expression grow indefinitely.
      return node;
    case ORDER_BY: {
      SqlOrderBy orderBy = (SqlOrderBy) node;
      handleOffsetFetch(orderBy.offset, orderBy.fetch);
      if (orderBy.query instanceof SqlSelect) {
        SqlSelect select = (SqlSelect) orderBy.query;

        // Don't clobber existing ORDER BY.  It may be needed for
        // an order-sensitive function like RANK.
        if (select.getOrderList() == null) {
          // push ORDER BY into existing select
          select.setOrderBy(orderBy.orderList);
          select.setOffset(orderBy.offset);
          select.setFetch(orderBy.fetch);
          return select;
        }
      }
      if (orderBy.query instanceof SqlWith
          && ((SqlWith) orderBy.query).body instanceof SqlSelect) {
        SqlWith with = (SqlWith) orderBy.query;
        SqlSelect select = (SqlSelect) with.body;

        // Don't clobber existing ORDER BY.  It may be needed for
        // an order-sensitive function like RANK.
        if (select.getOrderList() == null) {
          // push ORDER BY into existing select
          select.setOrderBy(orderBy.orderList);
          select.setOffset(orderBy.offset);
          select.setFetch(orderBy.fetch);
          return with;
        }
      }
      final SqlNodeList selectList = new SqlNodeList(SqlParserPos.ZERO);
      selectList.add(SqlIdentifier.star(SqlParserPos.ZERO));
      final SqlNodeList orderList;
      SqlSelect innerSelect = getInnerSelect(node);
      if (innerSelect != null && isAggregate(innerSelect)) {
        orderList = SqlNode.clone(orderBy.orderList);
        // We assume that ORDER BY item does not have ASC etc.
        // We assume that ORDER BY item is present in SELECT list.
        for (int i = 0; i < orderList.size(); i++) {
          SqlNode sqlNode = orderList.get(i);
          SqlNodeList selectList2 = SqlNonNullableAccessors.getSelectList(innerSelect);
          for (Ord<SqlNode> sel : Ord.zip(selectList2)) {
            if (stripAs(sel.e).equalsDeep(sqlNode, Litmus.IGNORE)) {
              orderList.set(i,
                  SqlLiteral.createExactNumeric(Integer.toString(sel.i + 1),
                      SqlParserPos.ZERO));
            }
          }
        }
      } else {
        orderList = orderBy.orderList;
      }
      return new SqlSelect(SqlParserPos.ZERO, null, selectList, orderBy.query,
          null, null, null, null, null, orderList, orderBy.offset,
          orderBy.fetch, null);
    }

    case EXPLICIT_TABLE: {
      // (TABLE t) is equivalent to (SELECT * FROM t)
      SqlCall call = (SqlCall) node;
      final SqlNodeList selectList = new SqlNodeList(SqlParserPos.ZERO);
      selectList.add(SqlIdentifier.star(SqlParserPos.ZERO));
      return new SqlSelect(SqlParserPos.ZERO, null, selectList, call.operand(0),
          null, null, null, null, null, null, null, null, null);
    }

    case DELETE: {
      SqlDelete call = (SqlDelete) node;
      if (call.getUsing() != null) {
        // If we have a USING clause, we rewrite the delete as a merge operation
        node = rewriteDeleteToMerge(call);
      } else {
        // Otherwise, we leave it as is, and just generate the source select
        SqlSelect select = createSourceSelectForDelete(call);
        call.setSourceSelect(select);
      }
      break;
    }

    case UPDATE: {
      SqlUpdate call = (SqlUpdate) node;
      SqlSelect select = createSourceSelectForUpdate(call);
      call.setSourceSelect(select);

      // See if we're supposed to rewrite UPDATE to MERGE
      // (unless this is the UPDATE clause of a MERGE,
      // in which case leave it alone).
      if (!validatingSqlMerge) {
        SqlNode selfJoinSrcExpr =
            getSelfJoinExprForUpdate(
                call.getTargetTable(),
                UPDATE_SRC_ALIAS);
        if (selfJoinSrcExpr != null) {
          node = rewriteUpdateToMerge(call, selfJoinSrcExpr);
        }
      }
      break;
    }
    case INSERT: {
      SqlInsert call = (SqlInsert) node;
      call.setSourceSelect(createSourceSelectForInsert(call));
      break;
    }
    case MERGE: {
      SqlMerge call = (SqlMerge) node;
      rewriteMerge(call);
      break;
    }
    default:
      break;
    }
    return node;
  }

  private static @Nullable SqlSelect getInnerSelect(SqlNode node) {
    for (;;) {
      if (node instanceof SqlSelect) {
        return (SqlSelect) node;
      } else if (node instanceof SqlOrderBy) {
        node = ((SqlOrderBy) node).query;
      } else if (node instanceof SqlWith) {
        node = ((SqlWith) node).body;
      } else {
        return null;
      }
    }
  }

  /**
   * Helper function for rewriteMerge. Adds the node to the sqlNodeList if it's non-null
   * otherwise, adds a literal True. This is used for adding the condition nodes for each
   * matched/not matched clause, as those conditions default to True if not specified.
   *
   * @param l The list to append to.
   * @param node The condition to add
   * @param pos the parser position for the newly created literal (if the input node is null)
   */
  private static void addOrDefaultTrue(SqlNodeList l, @Nullable SqlNode node, SqlParserPos pos) {
    if (node != null) {
      l.add(node);
    } else {
      l.add(SqlLiteral.createBoolean(true, pos));
    }
  }
  private static void rewriteMerge(SqlMerge call) {

    /**
     * Rewrite merge is responsible for setting the source select field of the SqlMerge.
     * Due to our changes, the values present in the source select should be as follows:
     * (All cols from source)
     * (All cols from dest)
     * (bool flag for checking if the row is a "match" or a "not match")
     * ((Match Condition) (Update Values (If the match condition is Update)))*
     * ((Insert Condition) (Insert Values))*
     *
     * Note that the Match/Insert Conditions are simply the boolean secondary conditions (WHEN
     * (NOT) MATCHED AND __COND__), we handle the checks for it a given row was matched/not matched
     * in sql to rel conversion.
     *
     * The source select does NOT enforce that the source/dest cols have the same name, only that
     * they evaluate to same expression. In fact, the source select should have no naming/aliasing
     * whatever (it should all be EXPR#X), but this isn't relied upon or enforced in any way.
     *
     * This is needed so that we can use convertSelect to handle all the subqueries that may
     * be present in the various conditions/clauses.
     */

    SqlNode origTargetTable = call.getTargetTable();

    // NOTE, we add a projection onto the dest/target table, which adds a literal TRUE
    // This is used for checking if a merge has occurred later on. For all rows which did not match
    // in the join, this column will be set to NA. For all rows which did match, it will be set
    // to True. This check is implemented during the conversion from SQL node to RelNode.
    SqlNodeList targetTableSelectList = new SqlNodeList(SqlParserPos.ZERO);
    targetTableSelectList.add(SqlIdentifier.star(SqlParserPos.ZERO));
    targetTableSelectList.add(SqlLiteral.createBoolean(true, SqlParserPos.ZERO));

    SqlNode targetTable = new SqlSelect(SqlParserPos.ZERO, null, targetTableSelectList,
        origTargetTable, null, null, null, null, null,
        null, null, null, null);
    if (call.getAlias() != null) {
      targetTable =
          SqlValidatorUtil.addAlias(
              targetTable,
              call.getAlias().getSimple());
    } else {
      // Due to the manner in which calcite handles subqueries, we need to explicitly add an
      // alias here in order to handle fully qualified table names.
      //
      // For example this will succseed:
      //
      // SELECT *, True from dept inner join (SELECT *, True from emp) as emp on emp.ename = 1
      //                                                               ^^^^^^
      // But this will not:
      //
      // SELECT *, True from dept inner join (SELECT *, True from emp) on emp.ename = 1
      //
      // Therefore, we need add this explicit alias because we expect a query like:
      // ... using table T1
      // WHEN MATCHED AND T1.foo > 1
      //
      // to be able to properly handle the fully qualified reference to T1.foo
      //
      targetTable =
          SqlValidatorUtil.addAlias(
              targetTable,
              ((SqlIdentifier) origTargetTable).getSimple()
      );
    }

    // Provided there is an insert sub statement, the source select for
    // the merge is a left outer join between the source in the USING
    // clause and the target table; otherwise, the join is just an
    // inner join. Need to clone the source table reference in order
    // for validation to work
    SqlNode sourceTableRef = call.getSourceTableRef();
    SqlNodeList insertCallList = call.getNotMatchedCallList();
    JoinType joinType = (insertCallList.size() == 0) ? JoinType.INNER : JoinType.LEFT;
    // In this case, it's ok to keep the original pos, but we need to do a deep copy so that
    // all of the sub nodes are different java objects, otherwise we get issues later during
    // validation (scopes, clauseScopes, and namespaces fields for the validator can conflict)
    final SqlDeepCopyShuttle shuttle = new SqlDeepCopyShuttle();
    final SqlNode leftJoinTerm = shuttle.visitNode(sourceTableRef);
    SqlNode outerJoin =
        new SqlJoin(SqlParserPos.ZERO,
            leftJoinTerm,
            SqlLiteral.createBoolean(false, SqlParserPos.ZERO),
            joinType.symbol(SqlParserPos.ZERO),
            targetTable,
            JoinConditionType.ON.symbol(SqlParserPos.ZERO),
            call.getCondition());

    // Originally, we used the source select of the update statement if it existed.
    // Now, we always just use select *
    SqlNodeList topmostSelectList = new SqlNodeList(SqlParserPos.ZERO);
    topmostSelectList.add(SqlIdentifier.star(SqlParserPos.ZERO));

    // Add all the conditions and values of the matched clauses into the select.
    // This project is needed to ensure
    // that nested sub queries are properly evaluated, and any needed joins needed to get the
    // required subquery values are properly created when expanding the select list
    SqlNodeList matchedCallList = call.getMatchedCallList();

    for (int i = 0; i < matchedCallList.size(); i++) {
      SqlNode curMatchCall = matchedCallList.get(i);
      if (curMatchCall instanceof SqlUpdate) {
        SqlUpdate curUpdateCall = (SqlUpdate) curMatchCall;
        // Note: we're only adding the secondary condition (... and cond). Checks for matched/not
        // matched rows are added during sql to rel conversion
        addOrDefaultTrue(topmostSelectList, curUpdateCall.getCondition(),
            curUpdateCall.getParserPosition());
        for (int j = 0; j < curUpdateCall.getSourceExpressionList().size(); j++) {
          SqlNode updateValue = curUpdateCall.getSourceExpressionList().get(j);
          topmostSelectList.add(updateValue);
        }
      } else {
        SqlDelete curDeleteCall = (SqlDelete) curMatchCall;
        // Note: we're only adding the secondary condition (... and cond). Checks for matched/not
        // matched rows are added during sql to rel conversion
        addOrDefaultTrue(topmostSelectList, curDeleteCall.getCondition(),
            curDeleteCall.getParserPosition());
      }
    }

    // Add all the conditions and values of the Not matched clauses into the select.
    for (int i = 0; i < insertCallList.size(); i++) {
      SqlInsert curInsertCall = (SqlInsert) insertCallList.get(i);
      // Note: we're only adding the secondary condition (... and cond). Checks for matched/not
      // matched rows are added during sql to rel conversion
      addOrDefaultTrue(topmostSelectList, curInsertCall.getCondition(),
          curInsertCall.getParserPosition());
      // From what I've seen, in merge, the source is always an VALUES statement
      assert curInsertCall.getSource() instanceof SqlBasicCall;
      SqlBasicCall insertSource = (SqlBasicCall) curInsertCall.getSource();
      assert insertSource.getOperator() instanceof SqlValuesOperator;

      List<SqlNode> curInsertValues = ((SqlBasicCall) insertSource.getOperandList().get(0))
          .getOperandList();

      for (int j = 0; j < curInsertValues.size(); j++) {
        // TODO (Nick):
        // If we are inside an insert then any column references must refer to the source table.
        // Since we are producing a top-level select it is important to append this information.
        SqlNode insertVal = curInsertValues.get(j);
        topmostSelectList.add(insertVal);
      }
    }

    // Finally, create the select statement itself.
    SqlSelect select =
        new SqlSelect(SqlParserPos.ZERO, null, topmostSelectList, outerJoin, null,
            null, null, null, null, null, null, null, null);
    call.setSourceSelect(select);

    // Source for the insert call is a select of the source table
    // reference with the select list being the value expressions;
    // note that the values clause has already been converted to a
    // select on the values row constructor; so we need to extract
    // that via the from clause on the select
    for  (int i = 0; i < insertCallList.size(); i++) {
      SqlInsert insertCall = (SqlInsert) insertCallList.get(i);
      SqlCall valuesCall = (SqlCall) insertCall.getSource();
      SqlCall rowCall = valuesCall.operand(0);
      SqlNodeList selectList =
          new SqlNodeList(
              rowCall.getOperandList(),
              SqlParserPos.ZERO);
      final SqlNode insertSource = shuttle.visitNode(sourceTableRef);
      select =
          new SqlSelect(SqlParserPos.ZERO, null, selectList, insertSource,
              insertCall.getCondition(), null, null, null, null, null,
              null, null, null);

      insertCall.setSource(select);
    }

  }



  /**
   * Used in UnconditonalRewrites. In the case that we have a DELETE with a USING clause.
   *
   * DELETE FROM target using T1, T2, ... where (cond) is equivalent to
   *
   * MERGE INTO target using (T1 full outer join T2...) on (cond) WHEN MATCHED THEN DELETE
   *
   * We choose to do this rewrite (similar to rewriteUpdateToMerge), in order to simplify validation
   * and sqlToRel code generation.
   *
   * @param originalDeleteCall The DELETE call to transform. Must have at least one table/subquery
   *                           in the "USING" clause.
   * @return A new SqlMerge, which is equivalent to the original SqlDelete call.
   */
  private SqlMerge rewriteDeleteToMerge(
      SqlDelete originalDeleteCall
  ) {
    //This should already be enforced in the one location we call this helper, but just to be safe
    SqlNodeList usingClauses = requireNonNull(originalDeleteCall.getUsing(),
        "rewriteDeleteToMerge called on a delete with no 'USING' clause");

    //This should be required by parsing, but to be safe:
    assert usingClauses.size() >= 1
        :
        "rewriteDeleteToMerge called on a delete with no 'USING' clause";
    SqlNode targetTable = originalDeleteCall.getTargetTable();
    SqlNode condition = originalDeleteCall.getCondition();
    if (condition == null) {
      condition = SqlLiteral.createBoolean(true, SqlParserPos.ZERO);
    }
    SqlIdentifier alias = originalDeleteCall.getAlias();

    SqlNodeList matchedCallList = SqlNodeList.EMPTY.clone(originalDeleteCall.getParserPosition());

    SqlDelete matchedDeleteExpression = new
        SqlDelete(originalDeleteCall.getParserPosition(),
        targetTable, null, null, alias);

    // TODO(keaton)
    // There's a wierd issue here. Essentially, in validation, it will infer
    // the row type of EMP as the row type from the merge in one location, but
    // as the emp from the original table in another location, which causes a number conflicts.
    //
    // I spent about a day trying to resolve this, but all the fixes I tried ended up breaking
    // existing merge into paths, and we don't even use the created Delete List anywhere
    // in the merge path, I'm just going to set this to something arbitrary, and leave this as
    // technical debt to figure out later.
    SqlSelect select = createSourceSelectForDelete(matchedDeleteExpression);
    select.getSelectList().remove(0);
    select.getSelectList().add(SqlLiteral.createBoolean(true, SqlParserPos.ZERO));
    matchedDeleteExpression.setSourceSelect(select);

    matchedCallList.add(matchedDeleteExpression);
    SqlNodeList unMatchedCallList = SqlNodeList.EMPTY.clone(originalDeleteCall.getParserPosition());

    //Set source to be the join of all the tables in Using.
    //We're relying on the optimizer to push filters from the ON clause
    //into the source table when appropriate.
    SqlNode source = ((SqlDeleteUsingItem) usingClauses.get(0)).getSqlDeleteItemAsJoinExpression();
    for (int i = 1; i < usingClauses.size(); i++) {
      SqlNode newExpr = ((SqlDeleteUsingItem) usingClauses.get(i))
          .getSqlDeleteItemAsJoinExpression();
      source = new SqlJoin(SqlParserPos.ZERO,
          source,
          SqlLiteral.createBoolean(false, SqlParserPos.ZERO),
          JoinType.FULL.symbol(SqlParserPos.ZERO),
          newExpr,
          JoinConditionType.ON.symbol(SqlParserPos.ZERO),
          SqlLiteral.createBoolean(true, SqlParserPos.ZERO));
    }

    SqlMerge mergeCall =
        new SqlMerge(originalDeleteCall.getParserPosition(), targetTable, condition, source,
            matchedCallList, unMatchedCallList, null,
            alias);

    //Run rewrite merge, so that it can apply whatever changes it needs to.
    rewriteMerge(mergeCall);
    return mergeCall;
  }

  private SqlNode rewriteUpdateToMerge(
      SqlUpdate updateCall,
      SqlNode selfJoinSrcExpr) {
    // Make sure target has an alias.
    SqlIdentifier updateAlias = updateCall.getAlias();
    if (updateAlias == null) {
      updateAlias = new SqlIdentifier(UPDATE_TGT_ALIAS, SqlParserPos.ZERO);
      updateCall.setAlias(updateAlias);
    }
    SqlNode selfJoinTgtExpr =
        getSelfJoinExprForUpdate(
            updateCall.getTargetTable(),
            updateAlias.getSimple());
    assert selfJoinTgtExpr != null;

    // Create join condition between source and target exprs,
    // creating a conjunction with the user-level WHERE
    // clause if one was supplied
    SqlNode condition = updateCall.getCondition();
    SqlNode selfJoinCond =
        SqlStdOperatorTable.EQUALS.createCall(
            SqlParserPos.ZERO,
            selfJoinSrcExpr,
            selfJoinTgtExpr);
    if (condition == null) {
      condition = selfJoinCond;
    } else {
      condition =
          SqlStdOperatorTable.AND.createCall(
              SqlParserPos.ZERO,
              selfJoinCond,
              condition);
    }
    SqlNode target =
        updateCall.getTargetTable().clone(SqlParserPos.ZERO);

    // For the source, we need to anonymize the fields, so
    // that for a statement like UPDATE T SET I = I + 1,
    // there's no ambiguity for the "I" in "I + 1";
    // this is OK because the source and target have
    // identical values due to the self-join.
    // Note that we anonymize the source rather than the
    // target because downstream, the optimizer rules
    // don't want to see any projection on top of the target.
    TableIdentifierWithIDNamespace ns =
        new TableIdentifierWithIDNamespace(this, target, null,
            castNonNull(null));
    RelDataType rowType = ns.getRowType();
    SqlNode source = updateCall.getTargetTable().clone(SqlParserPos.ZERO);
    final SqlNodeList selectList = new SqlNodeList(SqlParserPos.ZERO);
    int i = 1;
    for (RelDataTypeField field : rowType.getFieldList()) {
      SqlIdentifier col =
          new SqlIdentifier(
              field.getName(),
              SqlParserPos.ZERO);
      selectList.add(
          SqlValidatorUtil.addAlias(col, UPDATE_ANON_PREFIX + i));
      ++i;
    }
    source =
        new SqlSelect(SqlParserPos.ZERO, null, selectList, source, null, null,
            null, null, null, null, null, null, null);
    source = SqlValidatorUtil.addAlias(source, UPDATE_SRC_ALIAS);
    SqlMerge mergeCall =
        new SqlMerge(updateCall.getParserPosition(), target, condition, source,
            SqlNodeList.of(updateCall), SqlNodeList.EMPTY, null,
            updateCall.getAlias());
    rewriteMerge(mergeCall);
    return mergeCall;
  }

  /**
   * Allows a subclass to provide information about how to convert an UPDATE
   * into a MERGE via self-join. If this method returns null, then no such
   * conversion takes place. Otherwise, this method should return a suitable
   * unique identifier expression for the given table.
   *
   * @param table identifier for table being updated
   * @param alias alias to use for qualifying columns in expression, or null
   *              for unqualified references; if this is equal to
   *              {@value #UPDATE_SRC_ALIAS}, then column references have been
   *              anonymized to "SYS$ANONx", where x is the 1-based column
   *              number.
   * @return expression for unique identifier, or null to prevent conversion
   */
  protected @Nullable SqlNode getSelfJoinExprForUpdate(
      SqlNode table,
      String alias) {
    return null;
  }

  /**
   * Creates the SELECT statement that putatively feeds rows into an UPDATE
   * statement to be updated.
   *
   * @param call Call to the UPDATE operator
   * @return select statement
   */
  protected SqlSelect createSourceSelectForUpdate(SqlUpdate call) {
    final SqlNodeList selectList = new SqlNodeList(SqlParserPos.ZERO);
    selectList.add(SqlIdentifier.star(SqlParserPos.ZERO));
    int ordinal = 0;
    for (SqlNode exp : call.getSourceExpressionList()) {
      // Force unique aliases to avoid a duplicate for Y with
      // SET X=Y
      String alias = SqlUtil.deriveAliasFromOrdinal(ordinal);
      selectList.add(SqlValidatorUtil.addAlias(exp, alias));
      ++ordinal;
    }
    SqlNode sourceTable = call.getTargetTable();
    SqlIdentifier alias = call.getAlias();
    if (alias != null) {
      sourceTable =
          SqlValidatorUtil.addAlias(
              sourceTable,
              alias.getSimple());
    }
    return new SqlSelect(SqlParserPos.ZERO, null, selectList, sourceTable,
        call.getCondition(), null, null, null, null, null, null, null, null);
  }


  /**
   * Creates the SELECT statement that feeds rows into an INSERT
   * statement to be updated. Currently, this is only used in registerQuery, for the purpose of
   * verifying the type of the condition.
   *
   * @param call Call to the INSERT operator
   * @return select statement
   */
  protected @Nullable SqlSelect createSourceSelectForInsert(SqlInsert call) {
    final SqlNodeList selectList = new SqlNodeList(SqlParserPos.ZERO);

    // We only require the source select if the insert is part of a merge into clause.
    // the merge into clause requires the syntax to be VALUES (x,y,z...) as opposed to any
    // other type of query.
    if (!(call.getSource() instanceof SqlBasicCall)) {
      return null;
    }
    SqlBasicCall insertSource = (SqlBasicCall) call.getSource();
    if (!(insertSource.getOperator() instanceof SqlValuesOperator)) {
      return null;
    }
    List<SqlNode> curInsertValues = ((SqlBasicCall) insertSource.getOperandList().get(0))
        .getOperandList();

    int ordinal = 0;
    for (SqlNode exp : curInsertValues) {
      // Force unique aliases to avoid a duplicate for Y with
      // SET X=Y
      String alias = SqlUtil.deriveAliasFromOrdinal(ordinal);
      selectList.add(SqlValidatorUtil.addAlias(exp, alias));
      ++ordinal;
    }
    SqlNode sourceTable = call.getTargetTable();

    return new SqlSelect(SqlParserPos.ZERO, null, selectList, sourceTable,
        call.getCondition(), null, null, null, null,
        null, null, null, null);

  }

  /**
   * Creates the SELECT statement that putatively feeds rows into a DELETE
   * statement to be deleted.
   *
   * @param call Call to the DELETE operator
   * @return select statement
   */
  protected SqlSelect createSourceSelectForDelete(SqlDelete call) {
    final SqlNodeList selectList = new SqlNodeList(SqlParserPos.ZERO);
    selectList.add(SqlIdentifier.star(SqlParserPos.ZERO));
    SqlNode sourceTable = call.getTargetTable();
    SqlIdentifier alias = call.getAlias();
    if (alias != null) {
      sourceTable =
          SqlValidatorUtil.addAlias(
              sourceTable,
              alias.getSimple());
    }
    return new SqlSelect(SqlParserPos.ZERO, null, selectList, sourceTable,
        call.getCondition(), null, null, null, null, null, null, null, null);
  }

  /**
   * Returns null if there is no common type. E.g. if the rows have a
   * different number of columns.
   */
  @Nullable RelDataType getTableConstructorRowType(
      SqlCall values,
      SqlValidatorScope scope) {
    final List<SqlNode> rows = values.getOperandList();
    assert !rows.isEmpty();
    final List<RelDataType> rowTypes = new ArrayList<>();
    for (final SqlNode row : rows) {
      assert row.getKind() == SqlKind.ROW;
      SqlCall rowConstructor = (SqlCall) row;

      // REVIEW jvs 10-Sept-2003: Once we support single-row queries as
      // rows, need to infer aliases from there.
      final List<String> aliasList = new ArrayList<>();
      final List<RelDataType> typeList = new ArrayList<>();
      for (Ord<SqlNode> column : Ord.zip(rowConstructor.getOperandList())) {
        final String alias = SqlValidatorUtil.alias(column.e, column.i);
        aliasList.add(alias);
        final RelDataType type = deriveType(scope, column.e);
        typeList.add(type);
      }
      rowTypes.add(typeFactory.createStructType(typeList, aliasList));
    }
    if (rows.size() == 1) {
      // TODO jvs 10-Oct-2005:  get rid of this workaround once
      // leastRestrictive can handle all cases
      return rowTypes.get(0);
    }
    return typeFactory.leastRestrictive(rowTypes);
  }

  @Override public RelDataType getValidatedNodeType(SqlNode node) {
    RelDataType type = getValidatedNodeTypeIfKnown(node);
    if (type == null) {
      if (node.getKind() == SqlKind.IDENTIFIER) {
        throw newValidationError(node, RESOURCE.unknownIdentifier(node.toString()));
      }
      throw Util.needToImplement(node);
    } else {
      return type;
    }
  }

  @Override public @Nullable RelDataType getValidatedNodeTypeIfKnown(SqlNode node) {
    final RelDataType type = nodeToTypeMap.get(node);
    if (type != null) {
      return type;
    }
    final SqlValidatorNamespace ns = getNamespace(node);
    if (ns != null) {
      return ns.getType();
    }
    final SqlNode original = originalExprs.get(node);
    if (original != null && original != node) {
      return getValidatedNodeTypeIfKnown(original);
    }
    if (node instanceof SqlIdentifier) {
      return getCatalogReader().getNamedType((SqlIdentifier) node);
    }
    return null;
  }

  @Override public @Nullable List<RelDataType> getValidatedOperandTypes(SqlCall call) {
    return callToOperandTypesMap.get(call);
  }

  /**
   * Saves the type of a {@link SqlNode}, now that it has been validated.
   *
   * <p>Unlike the base class method, this method is not deprecated.
   * It is available from within Calcite, but is not part of the public API.
   *
   * @param node A SQL parse tree node, never null
   * @param type Its type; must not be null
   */
  @Override public final void setValidatedNodeType(SqlNode node, RelDataType type) {
    requireNonNull(type, "type");
    requireNonNull(node, "node");
    if (type.equals(unknownType)) {
      // don't set anything until we know what it is, and don't overwrite
      // a known type with the unknown type
      return;
    }
    nodeToTypeMap.put(node, type);
  }

  @Override public void removeValidatedNodeType(SqlNode node) {
    nodeToTypeMap.remove(node);
  }

  @Override public @Nullable SqlCall makeNullaryCall(SqlIdentifier id) {
    if (id.names.size() == 1 && !id.isComponentQuoted(0)) {
      final List<SqlOperator> list = new ArrayList<>();
      opTab.lookupOperatorOverloads(id, null, SqlSyntax.FUNCTION, list,
          catalogReader.nameMatcher());
      for (SqlOperator operator : list) {
        if (operator.getSyntax() == SqlSyntax.FUNCTION_ID) {
          // Even though this looks like an identifier, it is a
          // actually a call to a function. Construct a fake
          // call to this function, so we can use the regular
          // operator validation.
          return new SqlBasicCall(operator, ImmutableList.of(),
              id.getParserPosition(), null).withExpanded(true);
        }
      }
    }
    return null;
  }

  @Override public void requireNonCall(SqlTableIdentifierWithID id) throws ValidationException {
    ImmutableList<String> names = id.getNames();
    if (names.size() == 1 && !id.isComponentQuoted(0)) {
      final List<SqlOperator> list = new ArrayList<>();
      SqlIdentifier regularId = id.convertToSQLIdentifier();
      opTab.lookupOperatorOverloads(regularId, null, SqlSyntax.FUNCTION, list,
          catalogReader.nameMatcher());
      for (SqlOperator operator : list) {
        if (operator.getSyntax() == SqlSyntax.FUNCTION_ID) {
          // Even though this looks like an identifier, it is a
          // actually a call to a function. Here we raise an exception.
          throw new ValidationException(
              "Target Table for SQL Function must be a Table, but a function call was found");
        }
      }

    }
  }

  @Override public RelDataType deriveType(
      SqlValidatorScope scope,
      SqlNode expr) {
    requireNonNull(scope, "scope");
    requireNonNull(expr, "expr");

    // if we already know the type, no need to re-derive
    RelDataType type = nodeToTypeMap.get(expr);
    if (type != null) {
      return type;
    }
    final SqlValidatorNamespace ns = getNamespace(expr);
    if (ns != null) {
      return ns.getType();
    }
    type = deriveTypeImpl(scope, expr);
    requireNonNull(type, "SqlValidator.deriveTypeInternal returned null");
    setValidatedNodeType(expr, type);
    return type;
  }

  /**
   * Derives the type of a node, never null.
   */
  RelDataType deriveTypeImpl(
      SqlValidatorScope scope,
      SqlNode operand) {
    DeriveTypeVisitor v = new DeriveTypeVisitor(scope);
    final RelDataType type = operand.accept(v);
    return requireNonNull(scope.nullifyType(operand, type));
  }

  @Override public RelDataType deriveConstructorType(
      SqlValidatorScope scope,
      SqlCall call,
      SqlFunction unresolvedConstructor,
      @Nullable SqlFunction resolvedConstructor,
      List<RelDataType> argTypes) {
    SqlIdentifier sqlIdentifier = unresolvedConstructor.getSqlIdentifier();
    requireNonNull(sqlIdentifier, "sqlIdentifier");
    RelDataType type = catalogReader.getNamedType(sqlIdentifier);
    if (type == null) {
      // TODO jvs 12-Feb-2005:  proper type name formatting
      throw newValidationError(sqlIdentifier,
          RESOURCE.unknownDatatypeName(sqlIdentifier.toString()));
    }

    if (resolvedConstructor == null) {
      if (call.operandCount() > 0) {
        // This is not a default constructor invocation, and
        // no user-defined constructor could be found
        throw handleUnresolvedFunction(call, unresolvedConstructor, argTypes,
            null);
      }
    } else {
      SqlCall testCall =
          resolvedConstructor.createCall(
              call.getParserPosition(),
              call.getOperandList());
      RelDataType returnType =
          resolvedConstructor.validateOperands(
              this,
              scope,
              testCall);
      assert type == returnType;
    }

    if (config.identifierExpansion()) {
      if (resolvedConstructor != null) {
        ((SqlBasicCall) call).setOperator(resolvedConstructor);
      } else {
        // fake a fully-qualified call to the default constructor
        ((SqlBasicCall) call).setOperator(
            new SqlFunction(
                requireNonNull(type.getSqlIdentifier(), () -> "sqlIdentifier of " + type),
                ReturnTypes.explicit(type),
                null,
                null,
                null,
                SqlFunctionCategory.USER_DEFINED_CONSTRUCTOR));
      }
    }
    return type;
  }

  @Override public CalciteException handleUnresolvedFunction(SqlCall call,
      SqlOperator unresolvedFunction, List<RelDataType> argTypes,
      @Nullable List<String> argNames) {
    // For builtins, we can give a better error message
    final List<SqlOperator> overloads = new ArrayList<>();
    opTab.lookupOperatorOverloads(unresolvedFunction.getNameAsId(), null,
        SqlSyntax.FUNCTION, overloads, catalogReader.nameMatcher());
    if (overloads.size() == 1) {
      SqlFunction fun = (SqlFunction) overloads.get(0);
      if ((fun.getSqlIdentifier() == null)
          && (fun.getSyntax() != SqlSyntax.FUNCTION_ID)) {
        final int expectedArgCount =
            fun.getOperandCountRange().getMin();
        throw newValidationError(call,
            RESOURCE.invalidArgCount(call.getOperator().getName(),
                expectedArgCount));
      }
    }

    final String signature;
    if (unresolvedFunction instanceof SqlFunction) {
      final SqlOperandTypeChecker typeChecking =
          new AssignableOperandTypeChecker(argTypes, argNames);
      signature =
          typeChecking.getAllowedSignatures(unresolvedFunction,
              unresolvedFunction.getName());
    } else {
      signature = unresolvedFunction.getName();
    }
    throw newValidationError(call,
        RESOURCE.validatorUnknownFunction(signature));
  }

  protected void inferUnknownTypes(
      RelDataType inferredType,
      SqlValidatorScope scope,
      SqlNode node) {
    requireNonNull(inferredType, "inferredType");
    requireNonNull(scope, "scope");
    requireNonNull(node, "node");
    final SqlValidatorScope newScope = scopes.get(node);
    if (newScope != null) {
      scope = newScope;
    }
    boolean isNullLiteral = SqlUtil.isNullLiteral(node, false);
    if ((node instanceof SqlDynamicParam) || isNullLiteral) {
      if (inferredType.equals(unknownType)) {
        if (isNullLiteral) {
          if (config.typeCoercionEnabled()) {
            // derive type of null literal
            deriveType(scope, node);
            return;
          } else {
            throw newValidationError(node, RESOURCE.nullIllegal());
          }
        } else {
          throw newValidationError(node, RESOURCE.dynamicParamIllegal());
        }
      }

      // REVIEW:  should dynamic parameter types always be nullable?
      RelDataType newInferredType =
          typeFactory.createTypeWithNullability(inferredType, true);
      if (SqlTypeUtil.inCharFamily(inferredType)) {
        newInferredType =
            typeFactory.createTypeWithCharsetAndCollation(
                newInferredType,
                getCharset(inferredType),
                getCollation(inferredType));
      }
      setValidatedNodeType(node, newInferredType);
    } else if (node instanceof SqlNodeList) {
      SqlNodeList nodeList = (SqlNodeList) node;
      if (inferredType.isStruct()) {
        if (inferredType.getFieldCount() != nodeList.size()) {
          // this can happen when we're validating an INSERT
          // where the source and target degrees are different;
          // bust out, and the error will be detected higher up
          return;
        }
      }
      int i = 0;
      for (SqlNode child : nodeList) {
        RelDataType type;
        if (inferredType.isStruct()) {
          type = inferredType.getFieldList().get(i).getType();
          ++i;
        } else {
          type = inferredType;
        }
        inferUnknownTypes(type, scope, child);
      }
    } else if (node instanceof SqlCase) {
      final SqlCase caseCall = (SqlCase) node;

      final RelDataType whenType =
          caseCall.getValueOperand() == null ? booleanType : unknownType;
      for (SqlNode sqlNode : caseCall.getWhenOperands()) {
        inferUnknownTypes(whenType, scope, sqlNode);
      }
      RelDataType returnType = deriveType(scope, node);
      for (SqlNode sqlNode : caseCall.getThenOperands()) {
        inferUnknownTypes(returnType, scope, sqlNode);
      }

      SqlNode elseOperand =
          requireNonNull(caseCall.getElseOperand(),
              () -> "elseOperand for " + caseCall);
      if (!SqlUtil.isNullLiteral(elseOperand, false)) {
        inferUnknownTypes(
            returnType,
            scope,
            elseOperand);
      } else {
        setValidatedNodeType(elseOperand, returnType);
      }
    } else if (node.getKind()  == SqlKind.AS) {
      // For AS operator, only infer the operand not the alias
      inferUnknownTypes(inferredType, scope, ((SqlCall) node).operand(0));
    } else if (node.getKind() == SqlKind.MEASURE) {
      // For MEASURE operator, use the measure scope (which has additional
      // aliases available)
      if (scope instanceof SelectScope) {
        scope = getMeasureScope(((SelectScope) scope).getNode());
      }
      inferUnknownTypes(inferredType, scope, ((SqlCall) node).operand(0));
    } else if (node instanceof SqlCall) {
      final SqlCall call = (SqlCall) node;
      final SqlOperandTypeInference operandTypeInference =
          call.getOperator().getOperandTypeInference();
      final SqlCallBinding callBinding = new SqlCallBinding(this, scope, call);
      final List<SqlNode> operands = callBinding.operands();
      final RelDataType[] operandTypes = new RelDataType[operands.size()];
      Arrays.fill(operandTypes, unknownType);
      // TODO:  eventually should assert(operandTypeInference != null)
      // instead; for now just eat it
      if (operandTypeInference != null) {
        operandTypeInference.inferOperandTypes(
            callBinding,
            inferredType,
            operandTypes);
      }
      for (int i = 0; i < operands.size(); ++i) {
        final SqlNode operand = operands.get(i);
        if (operand != null) {
          inferUnknownTypes(operandTypes[i], scope, operand);
        }
      }
    }
  }

  /**
   * Adds an expression to a select list, ensuring that its alias does not
   * clash with any existing expressions on the list.
   */
  protected void addToSelectList(
      List<SqlNode> list,
      Set<String> aliases,
      List<Map.Entry<String, RelDataType>> fieldList,
      SqlNode exp,
      SelectScope scope,
      final boolean includeSystemVars) {
    final @Nullable String alias = SqlValidatorUtil.alias(exp);
    String uniqueAlias =
        SqlValidatorUtil.uniquify(
            alias, aliases, SqlValidatorUtil.EXPR_SUGGESTER);
    if (!Objects.equals(alias, uniqueAlias)) {
      exp = SqlValidatorUtil.addAlias(exp, uniqueAlias);
    }
    ((PairList<String, RelDataType>) fieldList)
        .add(uniqueAlias, deriveType(scope, exp));
    list.add(exp);
  }

  @Override public @Nullable String deriveAlias(
      SqlNode node,
      int ordinal) {
    return ordinal < 0 ? SqlValidatorUtil.alias(node)
        : SqlValidatorUtil.alias(node, ordinal);
  }

  protected boolean shouldAllowIntermediateOrderBy() {
    return true;
  }

  private void registerMatchRecognize(
      SqlValidatorScope parentScope,
      SqlValidatorScope usingScope,
      SqlMatchRecognize call,
      SqlNode enclosingNode,
      @Nullable String alias,
      boolean forceNullable) {

    final MatchRecognizeNamespace matchRecognizeNamespace =
        createMatchRecognizeNameSpace(call, enclosingNode);
    registerNamespace(usingScope, alias, matchRecognizeNamespace, forceNullable);

    final MatchRecognizeScope matchRecognizeScope =
        new MatchRecognizeScope(parentScope, call);
    scopes.put(call, matchRecognizeScope);

    // parse input query
    SqlNode expr = call.getTableRef();
    SqlNode newExpr =
        registerFrom(usingScope, matchRecognizeScope, true, expr,
            expr, null, null, forceNullable, false);
    if (expr != newExpr) {
      call.setOperand(0, newExpr);
    }
  }

  protected MatchRecognizeNamespace createMatchRecognizeNameSpace(
      SqlMatchRecognize call,
      SqlNode enclosingNode) {
    return new MatchRecognizeNamespace(this, call, enclosingNode);
  }

  private void registerPivot(
      SqlValidatorScope parentScope,
      SqlValidatorScope usingScope,
      SqlPivot pivot,
      SqlNode enclosingNode,
      @Nullable String alias,
      boolean forceNullable) {
    final PivotNamespace namespace =
        createPivotNameSpace(pivot, enclosingNode);
    registerNamespace(usingScope, alias, namespace, forceNullable);

    final SqlValidatorScope scope =
        new PivotScope(parentScope, pivot);
    scopes.put(pivot, scope);

    // parse input query
    SqlNode expr = pivot.query;
    SqlNode newExpr =
        registerFrom(parentScope, scope, true, expr,
            expr, null, null, forceNullable, false);
    if (expr != newExpr) {
      pivot.setOperand(0, newExpr);
    }
  }

  protected PivotNamespace createPivotNameSpace(SqlPivot call,
      SqlNode enclosingNode) {
    return new PivotNamespace(this, call, enclosingNode);
  }

  private void registerUnpivot(
      SqlValidatorScope parentScope,
      SqlValidatorScope usingScope,
      SqlUnpivot call,
      SqlNode enclosingNode,
      @Nullable String alias,
      boolean forceNullable) {
    final UnpivotNamespace namespace =
        createUnpivotNameSpace(call, enclosingNode);
    registerNamespace(usingScope, alias, namespace, forceNullable);

    final SqlValidatorScope scope =
        new UnpivotScope(parentScope, call);
    scopes.put(call, scope);

    // parse input query
    SqlNode expr = call.query;
    SqlNode newExpr =
        registerFrom(parentScope, scope, true, expr,
            expr, null, null, forceNullable, false);
    if (expr != newExpr) {
      call.setOperand(0, newExpr);
    }
  }

  protected UnpivotNamespace createUnpivotNameSpace(SqlUnpivot call,
      SqlNode enclosingNode) {
    return new UnpivotNamespace(this, call, enclosingNode);
  }

  /**
   * Registers a new namespace, and adds it as a child of its parent scope.
   * Derived class can override this method to tinker with namespaces as they
   * are created.
   *
   * @param usingScope    Parent scope (which will want to look for things in
   *                      this namespace)
   * @param alias         Alias by which parent will refer to this namespace
   * @param ns            Namespace
   * @param forceNullable Whether to force the type of namespace to be nullable
   */
  protected void registerNamespace(
      @Nullable SqlValidatorScope usingScope,
      @Nullable String alias,
      SqlValidatorNamespace ns,
      boolean forceNullable) {
    SqlValidatorNamespace namespace =
        namespaces.get(requireNonNull(ns.getNode(), () -> "ns.getNode() for " + ns));
    if (namespace == null) {
      namespaces.put(requireNonNull(ns.getNode()), ns);
      namespace = ns;
    }

    if (usingScope != null) {
      if (alias == null) {
        throw new IllegalArgumentException("Registering namespace " + ns
            + ", into scope " + usingScope + ", so alias must not be null");
      }
      usingScope.addChild(namespace, alias, forceNullable);
    }
  }

  /**
   * Registers scopes and namespaces implied a relational expression in the
   * FROM clause.
   *
   * <p>{@code parentScope0} and {@code usingScope} are often the same. They
   * differ when the namespace are not visible within the parent. (Example
   * needed.)
   *
   * <p>Likewise, {@code enclosingNode} and {@code node} are often the same.
   * {@code enclosingNode} is the topmost node within the FROM clause, from
   * which any decorations like an alias (<code>AS alias</code>) or a table
   * sample clause are stripped away to get {@code node}. Both are recorded in
   * the namespace.
   *
   * @param parentScope0  Parent scope that this scope turns to in order to
   *                      resolve objects
   * @param usingScope    Scope whose child list this scope should add itself to
   * @param register      Whether to register this scope as a child of
   *                      {@code usingScope}
   * @param node          Node which namespace is based on
   * @param enclosingNode Outermost node for namespace, including decorations
   *                      such as alias and sample clause
   * @param alias         Alias
   * @param extendList    Definitions of extended columns
   * @param forceNullable Whether to force the type of namespace to be
   *                      nullable because it is in an outer join
   * @param lateral       Whether LATERAL is specified, so that items to the
   *                      left of this in the JOIN tree are visible in the
   *                      scope
   * @return registered node, usually the same as {@code node}
   */
  // CHECKSTYLE: OFF
  // CheckStyle thinks this method is too long
  private SqlNode registerFrom(
      SqlValidatorScope parentScope0,
      SqlValidatorScope usingScope,
      boolean register,
      final SqlNode node,
      SqlNode enclosingNode,
      @Nullable String alias,
      @Nullable SqlNodeList extendList,
      boolean forceNullable,
      final boolean lateral) {
    final SqlKind kind = node.getKind();

    SqlNode expr;
    SqlNode newExpr;

    // Add an alias if necessary.
    SqlNode newNode = node;
    if (alias == null) {
      switch (kind) {
      case IDENTIFIER:
      case TABLE_IDENTIFIER_WITH_ID:
      case OVER:
        alias = SqlValidatorUtil.alias(node);
        if (alias == null) {
          alias = SqlValidatorUtil.alias(node, nextGeneratedId++);
        }
        if (config.identifierExpansion()) {
          newNode = SqlValidatorUtil.addAlias(node, alias);
        }
        break;

      case SELECT:
      case UNION:
      case INTERSECT:
      case EXCEPT:
      case VALUES:
      case UNNEST:
      case OTHER_FUNCTION:
      case COLLECTION_TABLE:
      case PIVOT:
      case UNPIVOT:
      case MATCH_RECOGNIZE:
      case WITH:
        // give this anonymous construct a name since later
        // query processing stages rely on it
        alias = SqlValidatorUtil.alias(node, nextGeneratedId++);
        if (config.identifierExpansion()) {
          // Since we're expanding identifiers, we should make the
          // aliases explicit too, otherwise the expanded query
          // will not be consistent if we convert back to SQL, e.g.
          // "select EXPR$1.EXPR$2 from values (1)".
          newNode = SqlValidatorUtil.addAlias(node, alias);
        }
        break;
      default:
        break;
      }
    }

    final SqlValidatorScope parentScope;
    if (lateral) {
      SqlValidatorScope s = usingScope;
      while (s instanceof JoinScope) {
        s = ((JoinScope) s).getUsingScope();
      }
      final SqlNode node2 = s != null ? s.getNode() : node;
      final TableScope tableScope = new TableScope(parentScope0, node2);
      if (usingScope instanceof ListScope) {
        for (ScopeChild child : ((ListScope) usingScope).children) {
          tableScope.addChild(child.namespace, child.name, child.nullable);
        }
      }
      parentScope = tableScope;
    } else {
      parentScope = parentScope0;
    }

    SqlCall call;
    SqlNode operand;
    SqlNode newOperand;
    switch (kind) {
    case AS:
      call = (SqlCall) node;
      if (alias == null) {
        alias = String.valueOf(call.operand(1));
      }
      expr = call.operand(0);
      final boolean needAliasNamespace = call.operandCount() > 2
          || expr.getKind() == SqlKind.VALUES || expr.getKind() == SqlKind.UNNEST
          || expr.getKind() == SqlKind.COLLECTION_TABLE;
      newExpr =
          registerFrom(
              parentScope,
              usingScope,
              !needAliasNamespace,
              expr,
              enclosingNode,
              alias,
              extendList,
              forceNullable,
              lateral);
      if (newExpr != expr) {
        call.setOperand(0, newExpr);
      }
      // If alias has a column list, introduce a namespace to translate
      // column names. We skipped registering it just now.
      if (needAliasNamespace) {
        registerNamespace(
            usingScope,
            alias,
            new AliasNamespace(this, call, enclosingNode),
            forceNullable);
      }
      return node;
    case MATCH_RECOGNIZE:
      registerMatchRecognize(parentScope, usingScope,
          (SqlMatchRecognize) node, enclosingNode, alias, forceNullable);
      return node;
    case PIVOT:
      // Bodo Change: Honor the register parameter
      registerPivot(parentScope, register ? usingScope : null, (SqlPivot) node, enclosingNode,
          alias, forceNullable);
      return node;
    case UNPIVOT:
      // Bodo Change: Honor the register parameter
      registerUnpivot(parentScope, register ? usingScope : null, (SqlUnpivot) node, enclosingNode,
          alias, forceNullable);
      return node;
    case TABLESAMPLE:
      call = (SqlCall) node;
      expr = call.operand(0);
      newExpr =
          registerFrom(
              parentScope,
              usingScope,
              true,
              expr,
              enclosingNode,
              alias,
              extendList,
              forceNullable,
              lateral);
      if (newExpr != expr) {
        call.setOperand(0, newExpr);
      }
      return node;
    case JOIN:
      final SqlJoin join = (SqlJoin) node;
      final JoinScope joinScope =
          new JoinScope(parentScope, usingScope, join);
      scopes.put(join, joinScope);
      final SqlNode left = join.getLeft();
      final SqlNode right = join.getRight();
      boolean forceLeftNullable = forceNullable;
      boolean forceRightNullable = forceNullable;
      switch (join.getJoinType()) {
      case LEFT:
      case LEFT_ASOF:
        forceRightNullable = true;
        break;
      case RIGHT:
        forceLeftNullable = true;
        break;
      case FULL:
        forceLeftNullable = true;
        forceRightNullable = true;
        break;
      default:
        break;
      }
      final SqlNode newLeft =
          registerFrom(
              parentScope,
              joinScope,
              true,
              left,
              left,
              null,
              null,
              forceLeftNullable,
              lateral);
      if (newLeft != left) {
        join.setLeft(newLeft);
      }
      final SqlNode newRight =
          registerFrom(
              parentScope,
              joinScope,
              true,
              right,
              right,
              null,
              null,
              forceRightNullable,
              lateral);
      if (newRight != right) {
        join.setRight(newRight);
      }
      scopes.putIfAbsent(stripAs(join.getRight()), parentScope);
      scopes.putIfAbsent(stripAs(join.getLeft()), parentScope);
      registerSubQueries(joinScope, join.getCondition());
      final JoinNamespace joinNamespace = new JoinNamespace(this, join);
      registerNamespace(null, null, joinNamespace, forceNullable);
      return join;

    case IDENTIFIER:
    case TABLE_IDENTIFIER_WITH_ID:
      final SqlValidatorNamespace newNs;
      if (node.getKind() == TABLE_IDENTIFIER_WITH_ID) {
        newNs = new TableIdentifierWithIDNamespace(
            this, (SqlTableIdentifierWithID) node, extendList, enclosingNode, parentScope
        );
      } else {
        newNs = new IdentifierNamespace(
            this, (SqlIdentifier) node, extendList, enclosingNode, parentScope
        );
      }
      registerNamespace(register ? usingScope : null, alias, newNs,
          forceNullable);
      if (tableScope == null) {
        tableScope = new TableScope(parentScope, node);
      }
      tableScope.addChild(newNs, requireNonNull(alias, "alias"), forceNullable);
      if (extendList != null && !extendList.isEmpty()) {
        return enclosingNode;
      }
      return newNode;

    case LATERAL:
      return registerFrom(
          parentScope,
          usingScope,
          register,
          ((SqlCall) node).operand(0),
          enclosingNode,
          alias,
          extendList,
          forceNullable,
          true);

    case COLLECTION_TABLE:
      call = (SqlCall) node;
      operand = call.operand(0);
      newOperand =
          registerFrom(
              parentScope,
              usingScope,
              register,
              operand,
              enclosingNode,
              alias,
              extendList,
              forceNullable, lateral);
      if (newOperand != operand) {
        call.setOperand(0, newOperand);
      }
      // If the operator is SqlWindowTableFunction, restricts the scope as
      // its first operand's (the table) scope.
      if (operand instanceof SqlBasicCall) {
        final SqlBasicCall call1 = (SqlBasicCall) operand;
        final SqlOperator op = call1.getOperator();
        if (op instanceof SqlWindowTableFunction
            && call1.operand(0).getKind() == SqlKind.SELECT) {
          scopes.put(node, getSelectScope(call1.operand(0)));
          return newNode;
        }
      }
      // Put the usingScope which can be a JoinScope
      // or a SelectScope, in order to see the left items
      // of the JOIN tree.
      scopes.put(node, usingScope);
      return newNode;

    case UNNEST:
      if (!lateral) {
        return registerFrom(parentScope, usingScope, register, node,
            enclosingNode, alias, extendList, forceNullable, true);
      }
    // fall through
    case SELECT:
    case UNION:
    case INTERSECT:
    case EXCEPT:
    case VALUES:
    case WITH:
    case OTHER_FUNCTION:
      if (alias == null) {
        alias = SqlValidatorUtil.alias(node, nextGeneratedId++);
      }
      registerQuery(
          parentScope,
          register ? usingScope : null,
          node,
          enclosingNode,
          alias,
          forceNullable);
      return newNode;

    case OVER:
      if (!shouldAllowOverRelation()) {
        throw Util.unexpected(kind);
      }
      call = (SqlCall) node;
      final OverScope overScope = new OverScope(usingScope, call);
      scopes.put(call, overScope);
      operand = call.operand(0);
      newOperand =
          registerFrom(
              parentScope,
              overScope,
              true,
              operand,
              enclosingNode,
              alias,
              extendList,
              forceNullable,
              lateral);
      if (newOperand != operand) {
        call.setOperand(0, newOperand);
      }

      for (ScopeChild child : overScope.children) {
        registerNamespace(register ? usingScope : null, child.name,
            child.namespace, forceNullable);
      }

      return newNode;

    case TABLE_REF:
    case TABLE_REF_WITH_ID:
      call = (SqlCall) node;
      registerFrom(parentScope,
          usingScope,
          register,
          call.operand(0),
          enclosingNode,
          alias,
          extendList,
          forceNullable,
          lateral);
      if (extendList != null && !extendList.isEmpty()) {
        return enclosingNode;
      }
      return newNode;

    case EXTEND:
      final SqlCall extend = (SqlCall) node;
      return registerFrom(parentScope,
          usingScope,
          true,
          extend.getOperandList().get(0),
          extend,
          alias,
          (SqlNodeList) extend.getOperandList().get(1),
          forceNullable,
          lateral);

    case SNAPSHOT:
      call = (SqlCall) node;
      operand = call.operand(0);
      newOperand =
          registerFrom(parentScope,
              usingScope,
              register,
              operand,
              enclosingNode,
              alias,
              extendList,
              forceNullable,
              lateral);
      if (newOperand != operand) {
        call.setOperand(0, newOperand);
      }
      // Put the usingScope which can be a JoinScope
      // or a SelectScope, in order to see the left items
      // of the JOIN tree.
      scopes.put(node, usingScope);
      return newNode;

    default:
      throw Util.unexpected(kind);
    }
  }
  // CHECKSTYLE: ON

  protected boolean shouldAllowOverRelation() {
    return false;
  }

  /**
   * Creates a namespace for a <code>SELECT</code> node. Derived class may
   * override this factory method.
   *
   * @param select        Select node
   * @param enclosingNode Enclosing node
   * @return Select namespace
   */
  protected SelectNamespace createSelectNamespace(
      SqlSelect select,
      SqlNode enclosingNode) {
    return new SelectNamespace(this, select, enclosingNode);
  }

  /**
   * Creates a namespace for a set operation (<code>UNION</code>, <code>
   * INTERSECT</code>, or <code>EXCEPT</code>). Derived class may override
   * this factory method.
   *
   * @param call          Call to set operation
   * @param enclosingNode Enclosing node
   * @return Set operation namespace
   */
  protected SetopNamespace createSetopNamespace(
      SqlCall call,
      SqlNode enclosingNode) {
    return new SetopNamespace(this, call, enclosingNode);
  }

  /**
   * Helper fn to avoid calcite function line limits.
   */
  private void registerCreateTableQuery(SqlValidatorScope parentScope,
      @Nullable SqlValidatorScope usingScope,
      SqlCreateTable createTable,
      SqlNode enclosingNode,
      boolean forceNullable) {
    // TODO: I'll likely need some sort of call to
    // validateFeature()
    // to confirm that the sql dialect we're validating for even supports CREATE_TABLE.
    // This will be done as followup: https://bodo.atlassian.net/browse/BE-4429

    final SqlNode queryNode = createTable.getQuery();

    // NOTE: query can be null, in the case that we're just doing a table definition with no data.
    // For now, only supporting the case where we have a query, or a table identifier
    if (queryNode != null) {
      if (!(queryNode instanceof SqlIdentifier)) {
        //In the case that the query is a select statement, we need to register it and
        // it's sub queries.
        registerQuery(
            parentScope,
            usingScope,
            queryNode,
            enclosingNode,
            null,
            false
        );
      } else {
        // Modified version of the code for registering the namespace of an
        // identifier in registerFrom.
        final SqlValidatorNamespace newNs = new IdentifierNamespace(
            this, (SqlIdentifier) queryNode, null, enclosingNode, parentScope
        );
        registerNamespace(usingScope, null, newNs, forceNullable);
      }

      DdlNamespace createTableNs = new DdlNamespace(
          this,
          createTable,
          enclosingNode,
          parentScope,
          queryNode);
      registerNamespace(usingScope, null, createTableNs, forceNullable);

    } else {
      throw newValidationError(createTable, BODO_SQL_RESOURCE.createTableRequiresAsQuery());
    }
    // Store the scope of the create table node itself.
    // This will be used later during validation to resolve table aliases and/or
    // the associated sub query.
    // Since the create table node is always the topmost node, the usingScope
    // should always be null, but better to be overly defensive.
    scopes.put(createTable, (usingScope != null) ? usingScope : parentScope);
  }

  /**
   * Registers a query in a parent scope.
   *
   * @param parentScope Parent scope which this scope turns to in order to
   *                    resolve objects
   * @param usingScope  Scope whose child list this scope should add itself to
   * @param node        Query node
   * @param alias       Name of this query within its parent. Must be specified
   *                    if usingScope != null
   */
  protected void registerQuery(
      SqlValidatorScope parentScope,
      @Nullable SqlValidatorScope usingScope,
      SqlNode node,
      SqlNode enclosingNode,
      @Nullable String alias,
      boolean forceNullable) {
    checkArgument(usingScope == null || alias != null);
    registerQuery(
        parentScope,
        usingScope,
        node,
        enclosingNode,
        alias,
        forceNullable,
        true);
  }

  /**
   * Registers a query in a parent scope.
   *
   * @param parentScope Parent scope which this scope turns to in order to
   *                    resolve objects
   * @param usingScope  Scope whose child list this scope should add itself to
   * @param node        Query node
   * @param alias       Name of this query within its parent. Must be specified
   *                    if usingScope != null
   * @param checkUpdate if true, validate that the update feature is supported
   *                    if validating the update statement
   */
  protected void registerQuery(
      SqlValidatorScope parentScope,
      @Nullable SqlValidatorScope usingScope,
      SqlNode node,
      SqlNode enclosingNode,
      @Nullable String alias,
      boolean forceNullable,
      boolean checkUpdate) {
    requireNonNull(node, "node");
    requireNonNull(enclosingNode, "enclosingNode");
    checkArgument(usingScope == null || alias != null);

    SqlCall call;
    List<SqlNode> operands;
    switch (node.getKind()) {
    case SELECT:

      final SqlSelect select = (SqlSelect) node;
      final SqlValidatorNamespace registeredSelectNs = getNamespace(select);
      if (registeredSelectNs == null) {
        final SelectNamespace selectNs = createSelectNamespace(select, enclosingNode);
        registerNamespace(usingScope, alias, selectNs, forceNullable);
      }

      final SqlValidatorScope windowParentScope =
          first(usingScope, parentScope);
      SelectScope selectScope =
          new SelectScope(parentScope, windowParentScope, select);
      scopes.put(select, selectScope);

      // Start by registering the WHERE clause
      // The where clause should be able to see all the aliases in the select list
      clauseScopes.put(IdPair.of(select, Clause.WHERE), selectScope);
      registerOperandSubQueries(
          selectScope,
          select,
          SqlSelect.WHERE_OPERAND);

      // Register subqueries in the QUALIFY clause
      registerOperandSubQueries(
          selectScope,
          select,
          SqlSelect.QUALIFY_OPERAND);

      // Register FROM with the inherited scope 'parentScope', not
      // 'selectScope', otherwise tables in the FROM clause would be
      // able to see each other.
      final SqlNode from = select.getFrom();
      if (from != null) {
        final SqlNode newFrom =
            registerFrom(
                parentScope,
                selectScope,
                true,
                from,
                from,
                null,
                null,
                false,
                false);
        if (newFrom != from) {
          select.setFrom(newFrom);
        }
      }

      // If this is an aggregate query, the SELECT list and HAVING
      // clause use a different scope (AggregatingSelectScope), which ensures that
      // you can only reference
      // columns which are in the GROUP BY clause.
      final SqlValidatorScope selectScope2 =
          isAggregate(select)
              ? new AggregatingSelectScope(selectScope, select, false)
              : selectScope;
      clauseScopes.put(IdPair.of(select, Clause.SELECT), selectScope2);
      clauseScopes.put(IdPair.of(select, Clause.MEASURE),
          new MeasureScope(selectScope, select));
      if (select.getGroup() != null) {
        GroupByScope groupByScope =
            new GroupByScope(selectScope, select.getGroup(), select);
        clauseScopes.put(IdPair.of(select, Clause.GROUP_BY), groupByScope);
        registerSubQueries(groupByScope, select.getGroup());
      }
      registerOperandSubQueries(
          selectScope2,
          select,
          SqlSelect.HAVING_OPERAND);
      registerSubQueries(selectScope2,
          SqlNonNullableAccessors.getSelectList(select));
      final SqlNodeList orderList = select.getOrderList();
      if (orderList != null) {
        // If the query is 'SELECT DISTINCT', restrict the columns
        // available to the ORDER BY clause.
        final SqlValidatorScope selectScope3 =
            select.isDistinct()
                ? new AggregatingSelectScope(selectScope, select, true)
                : selectScope2;
        OrderByScope orderScope =
            new OrderByScope(selectScope3, orderList, select);
        clauseScopes.put(IdPair.of(select, Clause.ORDER), orderScope);
        registerSubQueries(orderScope, orderList);

        if (!isAggregate(select)) {
          // Since this is not an aggregate query,
          // there cannot be any aggregates in the ORDER BY clause.
          SqlNode agg = aggFinder.findAgg(orderList);
          if (agg != null) {
            throw newValidationError(agg, RESOURCE.aggregateIllegalInOrderBy());
          }
        }
      }
      break;

    case INTERSECT:
      validateFeature(RESOURCE.sQLFeature_F302(), node.getParserPosition());
      registerSetop(
          parentScope,
          usingScope,
          node,
          node,
          alias,
          forceNullable);
      break;

    case EXCEPT:
      validateFeature(RESOURCE.sQLFeature_E071_03(), node.getParserPosition());
      registerSetop(
          parentScope,
          usingScope,
          node,
          node,
          alias,
          forceNullable);
      break;

    case UNION:
      registerSetop(
          parentScope,
          usingScope,
          node,
          enclosingNode,
          alias,
          forceNullable);
      break;

    case LAMBDA:
      call = (SqlCall) node;
      SqlLambdaScope lambdaScope =
          new SqlLambdaScope(parentScope, (SqlLambda) call);
      scopes.put(call, lambdaScope);
      final LambdaNamespace lambdaNamespace =
          new LambdaNamespace(this, (SqlLambda) call, node);
      registerNamespace(
          usingScope,
          alias,
          lambdaNamespace,
          forceNullable);
      operands = call.getOperandList();
      for (int i = 0; i < operands.size(); i++) {
        registerOperandSubQueries(parentScope, call, i);
      }
      break;

    case WITH:
      registerWith(parentScope, usingScope, (SqlWith) node, enclosingNode,
          alias, forceNullable, checkUpdate);
      break;

    case VALUES:
      call = (SqlCall) node;
      scopes.put(call, parentScope);
      final TableConstructorNamespace tableConstructorNamespace =
          new TableConstructorNamespace(
              this,
              call,
              parentScope,
              enclosingNode);
      registerNamespace(
          usingScope,
          alias,
          tableConstructorNamespace,
          forceNullable);
      operands = call.getOperandList();
      for (int i = 0; i < operands.size(); ++i) {
        assert operands.get(i).getKind() == SqlKind.ROW;

        // FIXME jvs 9-Feb-2005:  Correlation should
        // be illegal in these sub-queries.  Same goes for
        // any non-lateral SELECT in the FROM list.
        registerOperandSubQueries(parentScope, call, i);
      }
      break;

    case INSERT:
      SqlInsert insertCall = (SqlInsert) node;
      InsertNamespace insertNs =
          new InsertNamespace(
              this,
              insertCall,
              enclosingNode,
              parentScope);
      registerNamespace(usingScope, null, insertNs, forceNullable);
      registerQuery(
          parentScope,
          usingScope,
          insertCall.getSource(),
          enclosingNode,
          null,
          false);

      SqlSelect insertSourceSelect = insertCall.getSourceSelect();

      if (insertSourceSelect != null) {
        // registering the source select is done to verify if the condition is boolean, which
        // is only needed in MERGE with a NOT MATCHED clause.
        // the insertSourceSelect is only set if the inserted values are a VALUES expression, which
        // is the only supported expression for an INSERT sqlnode which originates form a MERGE
        // INTO clause.
        // The sourceSelect is not used after this point for Insert
        registerQuery(
            parentScope,
            usingScope,
            insertSourceSelect,
            enclosingNode,
            null,
            false);
      }
      break;

    case DELETE:
      SqlDelete deleteCall = (SqlDelete) node;
      DeleteNamespace deleteNs =
          new DeleteNamespace(
              this,
              deleteCall,
              enclosingNode,
              parentScope);
      registerNamespace(usingScope, null, deleteNs, forceNullable);
      registerQuery(
          parentScope,
          usingScope,
          SqlNonNullableAccessors.getSourceSelect(deleteCall),
          enclosingNode,
          null,
          false);
      break;

    case UPDATE:
      if (checkUpdate) {
        validateFeature(RESOURCE.sQLFeature_E101_03(),
            node.getParserPosition());
      }
      SqlUpdate updateCall = (SqlUpdate) node;
      UpdateNamespace updateNs =
          new UpdateNamespace(
              this,
              updateCall,
              enclosingNode,
              parentScope);
      registerNamespace(usingScope, null, updateNs, forceNullable);
      registerQuery(
          parentScope,
          usingScope,
          SqlNonNullableAccessors.getSourceSelect(updateCall),
          enclosingNode,
          null,
          false);
      break;

    case CREATE_TABLE:
      registerCreateTableQuery(parentScope, usingScope,
          (SqlCreateTable) node, enclosingNode, forceNullable);
      break;

    case MERGE:
      validateFeature(RESOURCE.sQLFeature_F312(), node.getParserPosition());
      SqlMerge mergeCall = (SqlMerge) node;
      MergeNamespace mergeNs =
          new MergeNamespace(
              this,
              mergeCall,
              enclosingNode,
              parentScope);
      registerNamespace(usingScope, null, mergeNs, forceNullable);
      registerQuery(
          parentScope,
          usingScope,
          SqlNonNullableAccessors.getSourceSelect(mergeCall),
          enclosingNode,
          null,
          false);

      // update call can reference either the source table reference
      // or the target table, so set its parent scope to the merge's
      // source select; when validating the update, skip the feature
      // validation check
      for (int i = 0; i < mergeCall.getMatchedCallList().size(); i++) {
        SqlNode mergeMatchCall = mergeCall.getMatchedCallList().get(i);
        requireNonNull(mergeMatchCall, "mergeMatchCall");
        registerQuery(
            getScope(SqlNonNullableAccessors.getSourceSelect(mergeCall), Clause.WHERE),
            null,
            mergeMatchCall,
            enclosingNode,
            null,
            false,
            false);
      }
      for (int i = 0; i < mergeCall.getNotMatchedCallList().size(); i++) {
        SqlInsert mergeInsertCall = (SqlInsert) mergeCall.getNotMatchedCallList().get(i);
        requireNonNull(mergeInsertCall, "mergeInsertCall");
        registerQuery(
            parentScope,
            null,
            mergeInsertCall,
            enclosingNode,
            null,
            false);
      }
      break;

    case UNNEST:
      call = (SqlCall) node;
      final UnnestNamespace unnestNs =
          new UnnestNamespace(this, call, parentScope, enclosingNode);
      registerNamespace(
          usingScope,
          alias,
          unnestNs,
          forceNullable);
      registerOperandSubQueries(parentScope, call, 0);
      scopes.put(node, parentScope);
      break;
    case OTHER_FUNCTION:
      call = (SqlCall) node;
      ProcedureNamespace procNs =
          new ProcedureNamespace(
              this,
              parentScope,
              call,
              enclosingNode);
      registerNamespace(
          usingScope,
          alias,
          procNs,
          forceNullable);
      registerSubQueries(parentScope, call);
      break;

    case MULTISET_QUERY_CONSTRUCTOR:
    case MULTISET_VALUE_CONSTRUCTOR:
      validateFeature(RESOURCE.sQLFeature_S271(), node.getParserPosition());
      call = (SqlCall) node;
      CollectScope cs = new CollectScope(parentScope, usingScope, call);
      final CollectNamespace tableConstructorNs =
          new CollectNamespace(call, cs, enclosingNode);
      final String alias2 = SqlValidatorUtil.alias(node, nextGeneratedId++);
      registerNamespace(
          usingScope,
          alias2,
          tableConstructorNs,
          forceNullable);
      operands = call.getOperandList();
      for (int i = 0; i < operands.size(); i++) {
        registerOperandSubQueries(parentScope, call, i);
      }
      break;

    default:
      throw Util.unexpected(node.getKind());
    }
  }

  private void registerSetop(
      SqlValidatorScope parentScope,
      @Nullable SqlValidatorScope usingScope,
      SqlNode node,
      SqlNode enclosingNode,
      @Nullable String alias,
      boolean forceNullable) {
    SqlCall call = (SqlCall) node;
    final SetopNamespace setopNamespace =
        createSetopNamespace(call, enclosingNode);
    registerNamespace(usingScope, alias, setopNamespace, forceNullable);

    // A setop is in the same scope as its parent.
    scopes.put(call, parentScope);
    @NonNull SqlValidatorScope recursiveScope = parentScope;
    if (enclosingNode.getKind() == SqlKind.WITH_ITEM) {
      if (node.getKind() != SqlKind.UNION) {
        throw newValidationError(node, RESOURCE.recursiveWithMustHaveUnionSetOp());
      } else if (call.getOperandList().size() > 2) {
        throw newValidationError(node, RESOURCE.recursiveWithMustHaveTwoChildUnionSetOp());
      }
      final WithScope scope = (WithScope) scopes.get(enclosingNode);
      // recursive scope is only set for the recursive queries.
      recursiveScope = scope != null && scope.recursiveScope != null
          ? requireNonNull(scope.recursiveScope) : parentScope;
    }
    for (int i = 0; i < call.getOperandList().size(); i++) {
      SqlNode operand = call.getOperandList().get(i);
      @NonNull SqlValidatorScope scope = i == 0 ? parentScope : recursiveScope;
      registerQuery(
          scope,
          null,
          operand,
          operand,
          null,
          false);
    }
  }

  private void registerWith(
      SqlValidatorScope parentScope,
      @Nullable SqlValidatorScope usingScope,
      SqlWith with,
      SqlNode enclosingNode,
      @Nullable String alias,
      boolean forceNullable,
      boolean checkUpdate) {
    final WithNamespace withNamespace =
        new WithNamespace(this, with, enclosingNode);
    registerNamespace(usingScope, alias, withNamespace, forceNullable);
    scopes.put(with, parentScope);

    SqlValidatorScope scope = parentScope;
    for (SqlNode withItem_ : with.withList) {
      final SqlWithItem withItem = (SqlWithItem) withItem_;

      final boolean isRecursiveWith = withItem.recursive.booleanValue();
      final SqlValidatorScope withScope =
          new WithScope(scope, withItem,
              isRecursiveWith ? new WithRecursiveScope(scope, withItem) : null);
      scopes.put(withItem, withScope);

      registerQuery(scope, null, withItem.query,
          withItem.recursive.booleanValue() ? withItem : with, withItem.name.getSimple(),
          forceNullable);
      registerNamespace(null, alias,
          new WithItemNamespace(this, withItem, enclosingNode),
          false);
      scope = withScope;
    }

    registerQuery(scope, null, with.body, enclosingNode, alias, forceNullable,
        checkUpdate);
  }

  @Override public boolean isAggregate(SqlSelect select) {
    if (getAggregate(select) != null) {
      return true;
    }
    // Also when nested window aggregates are present
    for (SqlCall call : overFinder.findAll(SqlNonNullableAccessors.getSelectList(select))) {
      assert call.getKind() == SqlKind.OVER;
      if (isNestedAggregateWindow(call.operand(0))) {
        return true;
      }
      if (isOverAggregateWindow(call.operand(1))) {
        return true;
      }
    }
    return false;
  }

  protected boolean isNestedAggregateWindow(SqlNode node) {
    AggFinder nestedAggFinder =
        new AggFinder(opTab, false, false, false, aggFinder,
            catalogReader.nameMatcher());
    return nestedAggFinder.findAgg(node) != null;
  }

  protected boolean isOverAggregateWindow(SqlNode node) {
    return aggFinder.findAgg(node) != null;
  }

  /** Returns the parse tree node (GROUP BY, HAVING, or an aggregate function
   * call) that causes {@code select} to be an aggregate query, or null if it
   * is not an aggregate query.
   *
   * <p>The node is useful context for error messages,
   * but you cannot assume that the node is the only aggregate function. */
  protected @Nullable SqlNode getAggregate(SqlSelect select) {
    SqlNode node = select.getGroup();
    if (node != null) {
      return node;
    }

    // Bodo change: we allow having in non-aggregate clauses. (where it is equivalent to a
    // WHERE clause).
    // Note that we still require the having to behave as normal in the
    // case that we encounter an aggregate in the having clause itself
    node = select.getHaving();
    if (node != null && aggFinder.findAgg(node) != null) {
      return node;
    }

    return getAgg(select);
  }

  /** If there is at least one call to an aggregate function, returns the
   * first. */
  private @Nullable SqlNode getAgg(SqlSelect select) {
    final SelectScope selectScope = getRawSelectScope(select);
    if (selectScope != null) {
      final List<SqlNode> selectList = selectScope.getExpandedSelectList();
      if (selectList != null) {
        return aggFinder.findAgg(selectList);
      }
    }
    return aggFinder.findAgg(SqlNonNullableAccessors.getSelectList(select));
  }

  @Deprecated
  @Override public boolean isAggregate(SqlNode selectNode) {
    return aggFinder.findAgg(selectNode) != null;
  }

  private void validateNodeFeature(SqlNode node) {
    switch (node.getKind()) {
    case MULTISET_VALUE_CONSTRUCTOR:
      validateFeature(RESOURCE.sQLFeature_S271(), node.getParserPosition());
      break;
    default:
      break;
    }
  }

  private void registerSubQueries(
      SqlValidatorScope parentScope,
      @Nullable SqlNode node) {
    if (node == null) {
      return;
    }
    if (node.getKind().belongsTo(SqlKind.QUERY)
        || node.getKind() == SqlKind.LAMBDA
        || node.getKind() == SqlKind.MULTISET_QUERY_CONSTRUCTOR
        || node.getKind() == SqlKind.MULTISET_VALUE_CONSTRUCTOR) {
      registerQuery(parentScope, null, node, node, null, false);
    } else if (node instanceof SqlCall) {
      validateNodeFeature(node);
      SqlCall call = (SqlCall) node;
      for (int i = 0; i < call.operandCount(); i++) {
        registerOperandSubQueries(parentScope, call, i);
      }
    } else if (node instanceof SqlNodeList) {
      SqlNodeList list = (SqlNodeList) node;
      for (int i = 0, count = list.size(); i < count; i++) {
        SqlNode listNode = list.get(i);
        if (listNode.getKind().belongsTo(SqlKind.QUERY)) {
          listNode =
              SqlStdOperatorTable.SCALAR_QUERY.createCall(
                  listNode.getParserPosition(),
                  listNode);
          list.set(i, listNode);
        }
        registerSubQueries(parentScope, listNode);
      }
    } else {
      // atomic node -- can be ignored
    }
  }

  /**
   * Registers any sub-queries inside a given call operand, and converts the
   * operand to a scalar sub-query if the operator requires it.
   *
   * @param parentScope    Parent scope
   * @param call           Call
   * @param operandOrdinal Ordinal of operand within call
   * @see SqlOperator#argumentMustBeScalar(int)
   */
  private void registerOperandSubQueries(
      SqlValidatorScope parentScope,
      SqlCall call,
      int operandOrdinal) {
    SqlNode operand = call.operand(operandOrdinal);
    if (operand == null) {
      return;
    }
    if (operand.getKind().belongsTo(SqlKind.QUERY)
        && call.getOperator().argumentMustBeScalar(operandOrdinal)) {
      operand =
          SqlStdOperatorTable.SCALAR_QUERY.createCall(
              operand.getParserPosition(),
              operand);
      call.setOperand(operandOrdinal, operand);
    }
    registerSubQueries(parentScope, operand);
  }

  @Override public void validateIdentifier(SqlIdentifier id, SqlValidatorScope scope) {
    final SqlQualified fqId = scope.fullyQualify(id);
    if (this.config.columnReferenceExpansion()) {
      // NOTE jvs 9-Apr-2007: this doesn't cover ORDER BY, which has its
      // own ideas about qualification.
      id.assignNamesFrom(fqId.identifier);
    } else {
      Util.discard(fqId);
    }
  }

  /**
   * Resolves a SqlTableIdentifierWithID to a fully-qualified name.
   *
   * @param id    SqlTableIdentifierWithID
   * @param scope Naming scope
   */
  @Override public void validateTableIdentifierWithID(
      SqlTableIdentifierWithID id, SqlValidatorScope scope) {
    final SqlTableIdentifierWithIDQualified fqId = scope.fullyQualify(id);
    if (this.config.columnReferenceExpansion()) {
      // NOTE jvs 9-Apr-2007: this doesn't cover ORDER BY, which has its
      // own ideas about qualification.
      id.assignNamesFrom(fqId.identifier);
    } else {
      Util.discard(fqId);
    }
  }

  @SuppressWarnings("deprecation") // [CALCITE-6598]
  @Override public void validateLiteral(SqlLiteral literal) {
    switch (literal.getTypeName()) {
    case DECIMAL:
      // Accept any decimal value that does not exceed the max
      // precision and scale of the type system.
      final RelDataTypeSystem typeSystem = getTypeFactory().getTypeSystem();
      final BigDecimal bd = literal.getValueAs(BigDecimal.class);
      final BigDecimal noTrailingZeros = bd.stripTrailingZeros();
      // If we don't strip trailing zeros we may reject values such as 1.000....0.

      final int maxPrecision = typeSystem.getMaxNumericPrecision();
      if (noTrailingZeros.precision() > maxPrecision) {
        throw newValidationError(literal,
            RESOURCE.numberLiteralOutOfRange(bd.toString()));
      }

      final int maxScale = typeSystem.getMaxNumericScale();
      if (noTrailingZeros.scale() > maxScale) {
        throw newValidationError(literal,
            RESOURCE.numberLiteralOutOfRange(bd.toString()));
      }
      break;

    case DOUBLE:
    case FLOAT:
    case REAL:
      validateLiteralAsDouble(literal);
      break;

    case BINARY:
      final BitString bitString = literal.getValueAs(BitString.class);
      if ((bitString.getBitCount() % 8) != 0) {
        throw newValidationError(literal, RESOURCE.binaryLiteralOdd());
      }
      break;

    case DATE:
    case TIME:
    case TIMESTAMP:
      Calendar calendar = literal.getValueAs(Calendar.class);
      final int year = calendar.get(Calendar.YEAR);
      final int era = calendar.get(Calendar.ERA);
      if (year < 1 || era == GregorianCalendar.BC || year > 9999) {
        throw newValidationError(literal,
            RESOURCE.dateLiteralOutOfRange(literal.toString()));
      }
      break;

    case INTERVAL_YEAR:
    case INTERVAL_YEAR_MONTH:
    case INTERVAL_MONTH:
    case INTERVAL_DAY:
    case INTERVAL_DAY_HOUR:
    case INTERVAL_DAY_MINUTE:
    case INTERVAL_DAY_SECOND:
    case INTERVAL_HOUR:
    case INTERVAL_HOUR_MINUTE:
    case INTERVAL_HOUR_SECOND:
    case INTERVAL_MINUTE:
    case INTERVAL_MINUTE_SECOND:
    case INTERVAL_SECOND:
      if (literal instanceof SqlIntervalLiteral) {
        SqlIntervalLiteral.IntervalValue interval =
            literal.getValueAs(SqlIntervalLiteral.IntervalValue.class);
        SqlIntervalQualifier intervalQualifier =
            interval.getIntervalQualifier();

        // ensure qualifier is good before attempting to validate literal
        validateIntervalQualifier(intervalQualifier);
        String intervalStr = interval.getIntervalLiteral();
        // throws CalciteContextException if string is invalid
        int[] values =
            intervalQualifier.evaluateIntervalLiteral(intervalStr,
                literal.getParserPosition(), typeFactory.getTypeSystem());
        Util.discard(values);
      }
      break;
    default:
      // default is to do nothing
    }
  }

  private void validateLiteralAsDouble(SqlLiteral literal) {
    BigDecimal bd = literal.getValueAs(BigDecimal.class);
    double d = bd.doubleValue();
    if (Double.isInfinite(d) || Double.isNaN(d)) {
      // overflow
      throw newValidationError(literal,
          RESOURCE.numberLiteralOutOfRange(Util.toScientificNotation(bd)));
    }

    // REVIEW jvs 4-Aug-2004:  what about underflow?
  }

  @Override public void validateIntervalQualifier(SqlIntervalQualifier qualifier) {
    requireNonNull(qualifier, "qualifier");
    boolean startPrecisionOutOfRange = false;
    boolean fractionalSecondPrecisionOutOfRange = false;
    final RelDataTypeSystem typeSystem = typeFactory.getTypeSystem();

    final int startPrecision = qualifier.getStartPrecision(typeSystem);
    final int fracPrecision =
        qualifier.getFractionalSecondPrecision(typeSystem);
    final int maxPrecision = typeSystem.getMaxPrecision(qualifier.typeName());
    final int minPrecision = typeSystem.getMinPrecision(qualifier.typeName());
    final int minScale = typeSystem.getMinScale(qualifier.typeName());
    final int maxScale = typeSystem.getMaxScale(qualifier.typeName());
    if (startPrecision < minPrecision || startPrecision > maxPrecision) {
      startPrecisionOutOfRange = true;
    } else {
      if (fracPrecision < minScale || fracPrecision > maxScale) {
        fractionalSecondPrecisionOutOfRange = true;
      }
    }

    if (startPrecisionOutOfRange) {
      throw newValidationError(qualifier,
          RESOURCE.intervalStartPrecisionOutOfRange(startPrecision,
              "INTERVAL " + qualifier));
    } else if (fractionalSecondPrecisionOutOfRange) {
      throw newValidationError(qualifier,
          RESOURCE.intervalFractionalSecondPrecisionOutOfRange(
              fracPrecision,
              "INTERVAL " + qualifier));
    }
  }

  @Override public TimeFrame validateTimeFrame(SqlIntervalQualifier qualifier) {
    if (qualifier.timeFrameName == null) {
      final TimeFrame timeFrame = timeFrameSet.get(qualifier.getUnit());
      return requireNonNull(timeFrame,
          () -> "time frame for " + qualifier.getUnit());
    }
    final @Nullable TimeFrame timeFrame =
        timeFrameSet.getOpt(qualifier.timeFrameName);
    if (timeFrame != null) {
      return timeFrame;
    }
    throw newValidationError(qualifier,
        RESOURCE.invalidTimeFrame(qualifier.timeFrameName));
  }

  /**
   * Validates the FROM clause of a query, or (recursively) a child node of
   * the FROM clause: AS, OVER, JOIN, VALUES, or sub-query.
   *
   * @param node          Node in FROM clause, typically a table or derived
   *                      table
   * @param targetRowType Desired row type of this expression, or
   *                      {@link #unknownType} if not fussy. Must not be null.
   * @param scope         Scope
   */
  protected void validateFrom(
      SqlNode node,
      RelDataType targetRowType,
      SqlValidatorScope scope) {
    requireNonNull(scope, "scope");
    requireNonNull(targetRowType, "targetRowType");
    switch (node.getKind()) {
    case AS:
    case TABLE_REF:
      validateFrom(
          ((SqlCall) node).operand(0),
          targetRowType,
          scope);
      break;
    case VALUES:
      validateValues((SqlCall) node, targetRowType, scope);
      break;
    case JOIN:
      validateJoin((SqlJoin) node, scope);
      break;
    case OVER:
      validateOver((SqlCall) node, scope);
      break;
    case UNNEST:
      validateUnnest((SqlCall) node, scope, targetRowType);
      break;
    case COLLECTION_TABLE:
      validateTableFunction((SqlCall) node, scope, targetRowType);
      break;
    default:
      validateQuery(node, scope, targetRowType);
      break;
    }

    // Validate the namespace representation of the node, just in case the
    // validation did not occur implicitly.
    getNamespaceOrThrow(node, scope).validate(targetRowType);
  }

  protected void validateTableFunction(SqlCall node, SqlValidatorScope scope,
      RelDataType targetRowType) {
    // Dig out real call; TABLE() wrapper is just syntactic.
    SqlCall call = node.operand(0);
    if (call.getOperator() instanceof SqlTableFunction) {
      SqlTableFunction tableFunction = (SqlTableFunction) call.getOperator();
      boolean visitedRowSemanticsTable = false;
      for (int idx = 0; idx < call.operandCount(); idx++) {
        TableCharacteristic tableCharacteristic = tableFunction.tableCharacteristic(idx);
        if (tableCharacteristic != null) {
          // Skip validate if current input table has set semantics
          if (tableCharacteristic.semantics == TableCharacteristic.Semantics.SET) {
            continue;
          }
          // A table function at most has one input table with row semantics
          if (visitedRowSemanticsTable) {
            throw newValidationError(
                call,
                RESOURCE.multipleRowSemanticsTables(call.getOperator().getName()));
          }
          visitedRowSemanticsTable = true;
        }
        // If table function defines the parameter is not table parameter, or is an input table
        // parameter with row semantics, then it should not be with PARTITION BY OR ORDER BY.
        SqlNode currentNode = call.operand(idx);
        if (currentNode instanceof SqlCall) {
          SqlOperator op = ((SqlCall) currentNode).getOperator();
          if (op == SqlStdOperatorTable.ARGUMENT_ASSIGNMENT) {
            // Dig out the underlying operand
            SqlNode realNode = ((SqlBasicCall) currentNode).operand(0);
            if (realNode instanceof SqlCall) {
              currentNode = realNode;
              op = ((SqlCall) realNode).getOperator();
            }
          }
          if (op == SqlStdOperatorTable.SET_SEMANTICS_TABLE) {
            throwInvalidRowSemanticsTable(call, idx, (SqlCall) currentNode);
          }
        }
      }
    }
    validateQuery(node, scope, targetRowType);
  }

  private void throwInvalidRowSemanticsTable(SqlCall call, int idx, SqlCall table) {
    SqlNodeList partitionList = table.operand(1);
    if (!partitionList.isEmpty()) {
      throw newValidationError(call,
          RESOURCE.invalidPartitionKeys(
              idx, call.getOperator().getName()));
    }
    SqlNodeList orderList = table.operand(2);
    if (!orderList.isEmpty()) {
      throw newValidationError(call,
          RESOURCE.invalidOrderBy(
              idx, call.getOperator().getName()));
    }
  }

  protected void validateOver(SqlCall call, SqlValidatorScope scope) {
    throw new AssertionError("OVER unexpected in this context");
  }

  protected void validateUnnest(SqlCall call, SqlValidatorScope scope,
      RelDataType targetRowType) {
    for (int i = 0; i < call.operandCount(); i++) {
      SqlNode expandedItem = expand(call.operand(i), scope);
      call.setOperand(i, expandedItem);
    }
    validateQuery(call, scope, targetRowType);
  }

  private void checkRollUpInUsing(SqlIdentifier identifier,
      SqlNode leftOrRight, SqlValidatorScope scope) {
    SqlValidatorNamespace namespace = getNamespace(leftOrRight, scope);
    if (namespace != null) {
      SqlValidatorTable sqlValidatorTable = namespace.getTable();
      if (sqlValidatorTable != null) {
        Table table = sqlValidatorTable.table();
        String column = Util.last(identifier.names);

        if (table.isRolledUp(column)) {
          throw newValidationError(identifier,
              RESOURCE.rolledUpNotAllowed(column, "USING"));
        }
      }
    }
  }

  private void checkIfTableFunctionIsPartOfCondition(SqlNode node) {
    if (node == null) {
      return;
    }
    if (node instanceof SqlBasicCall) {
      SqlBasicCall call = (SqlBasicCall) node;
      List<SqlNode> conditionOperands = call.getOperandList();
      for (SqlNode operand : conditionOperands) {
        if (operand instanceof SqlBasicCall && ((SqlBasicCall) operand).getOperator() instanceof SqlTableFunction) {
          SqlBasicCall callOperand = (SqlBasicCall) operand;
          throw RESOURCE.cannotCallTableFunctionHere(callOperand.getOperator().getName()).ex();
        }
        checkIfTableFunctionIsPartOfCondition(operand);
      }
    }
  }

  protected void validateJoin(SqlJoin join, SqlValidatorScope scope) {
    // Bodo Change: Verify a table function is not part of the join condition.
    checkIfTableFunctionIsPartOfCondition(join.getCondition());

    final SqlNode left = join.getLeft();
    final SqlNode right = join.getRight();
    final boolean natural = join.isNatural();
    final JoinType joinType = join.getJoinType();
    final JoinConditionType conditionType = join.getConditionType();
    final SqlValidatorScope joinScope = getScopeOrThrow(join); // getJoinScope?
    validateFrom(left, unknownType, joinScope);
    validateFrom(right, unknownType, joinScope);

    // Validate condition.
    switch (conditionType) {
    case NONE:
      checkArgument(join.getCondition() == null);
      break;
    case ON:
      final SqlNode condition;
      if (scope.getNode() instanceof SqlSelect) {
        // Note: SELECT is used for expansion, not scoping.
        condition = expandWithAlias(getCondition(join), joinScope, (SqlSelect) scope.getNode(),
            Clause.SELECT);
      } else {
        condition = expand(getCondition(join), joinScope);
      }
      join.setOperand(5, condition);
      validateWhereOrOnOrNonAggregateHaving(joinScope, condition, "ON");
      checkRollUp(null, join, condition, joinScope, "ON");
      break;
    case USING:
      @SuppressWarnings({"rawtypes", "unchecked"}) List<SqlIdentifier> list =
          (List) getCondition(join);

      // Parser ensures that using clause is not empty.
      checkArgument(!list.isEmpty(), "Empty USING clause");
      for (SqlIdentifier id : list) {
        validateCommonJoinColumn(id, left, right, scope, natural);
      }
      break;
    default:
      throw Util.unexpected(conditionType);
    }

    // Validate NATURAL.
    if (natural) {
      if (join.getCondition() != null) {
        throw newValidationError(getCondition(join),
            RESOURCE.naturalDisallowsOnOrUsing());
      }

      // Join on fields that occur on each side.
      // Check compatibility of the chosen columns.
      for (String name : deriveNaturalJoinColumnList(join)) {
        final SqlIdentifier id =
            new SqlIdentifier(name, join.isNaturalNode().getParserPosition());
        validateCommonJoinColumn(id, left, right, scope, natural);
      }
    }

    // Which join types require/allow a ON/USING condition, or allow
    // a NATURAL keyword?
    switch (joinType) {
    case LEFT_ANTI_JOIN:
    case LEFT_SEMI_JOIN:
      if (!this.config.conformance().isLiberal()) {
        throw newValidationError(join.getJoinTypeNode(),
            RESOURCE.dialectDoesNotSupportFeature(joinType.name()));
      }
      // fall through
    case INNER:
    case LEFT:
    case RIGHT:
    case FULL:
      if ((join.getCondition() == null) && !natural) {
        throw newValidationError(join, RESOURCE.joinRequiresCondition());
      }
      break;
    case COMMA:
    case CROSS:
      if (join.getCondition() != null) {
        throw newValidationError(join.getConditionTypeNode(),
            RESOURCE.crossJoinDisallowsCondition());
      }
      if (natural) {
        throw newValidationError(join.getConditionTypeNode(),
            RESOURCE.crossJoinDisallowsCondition());
      }
      break;
    case LEFT_ASOF:
    case ASOF: {
      // In addition to the standard join checks, the ASOF join requires the
      // ON conditions to be a conjunction of simple equalities from both relations.
      SqlAsofJoin asof = (SqlAsofJoin) join;
      SqlNode matchCondition = getMatchCondition(asof);
      matchCondition = expand(matchCondition, joinScope);
      join.setOperand(6, matchCondition);
      validateWhereOrOnOrNonAggregateHaving(joinScope, matchCondition, "MATCH_CONDITION");
      SqlNode condition = join.getCondition();
      if (condition == null) {
        throw newValidationError(join, RESOURCE.joinRequiresCondition());
      }
      ConjunctionOfEqualities conj = new ConjunctionOfEqualities();
      condition.accept(conj);
      if (conj.illegal) {
        throw newValidationError(condition, RESOURCE.asofConditionMustBeComparison());
      }

      CompareFromBothSides validateCompare =
          new CompareFromBothSides(joinScope,
              catalogReader, RESOURCE.asofConditionMustBeComparison());
      condition.accept(validateCompare);

      // It also requires the MATCH condition to be a comparison.
      if (!(matchCondition instanceof SqlCall)) {
        throw newValidationError(matchCondition, RESOURCE.asofMatchMustBeComparison());
      }
      SqlCall matchCall = (SqlCall) matchCondition;
      SqlOperator operator = matchCall.getOperator();
      if (!SqlKind.ORDER_COMPARISON.contains(operator.kind)) {
        throw newValidationError(matchCondition, RESOURCE.asofMatchMustBeComparison());
      }

      // Change the exception in validateCompare when we validate the match condition
      validateCompare =
          new CompareFromBothSides(joinScope,
              catalogReader, RESOURCE.asofMatchMustBeComparison());
      matchCondition.accept(validateCompare);
      break;
    }
    default:
      throw Util.unexpected(joinType);
    }
  }

  /**
   * Shuttle which determines whether all SqlCalls that are
   * comparisons are comparing columns from both namespaces.
   * The shuttle will throw an exception if that happens.
   * If it returns all SqlCalls have the expected shape.
   */
  private class CompareFromBothSides extends SqlShuttle {
    final SqlValidatorScope scope;
    final SqlValidatorCatalogReader catalogReader;
    final Resources.ExInst<SqlValidatorException> exception;

    private CompareFromBothSides(
        SqlValidatorScope scope,
        SqlValidatorCatalogReader catalogReader,
        Resources.ExInst<SqlValidatorException> exception) {
      this.scope = scope;
      this.catalogReader = catalogReader;
      this.exception = exception;
    }

    @Override public @Nullable SqlNode visit(final SqlCall call) {
      SqlKind kind = call.getKind();
      if (SqlKind.COMPARISON.contains(kind)) {
        assert call.getOperandList().size() == 2;

        boolean leftFound = false;
        boolean rightFound = false;
        // The two sides of the comparison must be from different tables
        for (SqlNode operand : call.getOperandList()) {
          if (!(operand instanceof SqlIdentifier)) {
            throw newValidationError(call, this.exception);
          }
          // We know that all identifiers have been expanded by the caller,
          // so they have the shape namespace.field
          SqlIdentifier id = (SqlIdentifier) operand;
          final SqlNameMatcher nameMatcher = catalogReader.nameMatcher();
          final SqlValidatorScope.ResolvedImpl resolved = new SqlValidatorScope.ResolvedImpl();
          // Lookup just the first component of the name
          scope.resolve(id.names.subList(0, id.names.size() - 1), nameMatcher, false, resolved);
          SqlValidatorScope.Resolve resolve = resolved.only();
          int index = resolve.path.steps().get(0).i;
          if (index == 0) {
            leftFound = true;
          }
          if (index == 1) {
            rightFound = true;
          }

          if (!leftFound && !rightFound) {
            throw newValidationError(call, this.exception);
          }
        }
        if (!leftFound || !rightFound) {
          // The comparison does not look at both tables
          throw newValidationError(call, this.exception);
        }
      }
      return super.visit(call);
    }
  }

  /**
   * Shuttle which determines whether an expression is a simple conjunction
   * of equalities. */
  private static class ConjunctionOfEqualities extends SqlShuttle {
    boolean illegal = false;

    // Check an AND node.  Children can be AND nodes or EQUAL nodes.
    void checkAnd(SqlCall call) {
      // This doesn't seem to use the visitor pattern,
      // because we recurse explicitly on the tree structure.
      // The visitor is useful to make sure no other kinds of operations
      // appear in the expression tree.
      List<SqlNode> operands = call.getOperandList();
      for (SqlNode operand : operands) {
        if (operand.getKind() == SqlKind.AND) {
          this.checkAnd((SqlCall) operand);
          return;
        }
        if (operand.getKind() != SqlKind.EQUALS) {
          illegal = true;
        }
      }
    }

    @Override public @Nullable SqlNode visit(final org.apache.calcite.sql.SqlCall call) {
      SqlKind kind = call.getKind();
      if (kind != SqlKind.AND && kind != SqlKind.EQUALS) {
        illegal = true;
      }
      if (kind == SqlKind.AND) {
        this.checkAnd(call);
      }
      return super.visit(call);
    }
  }



  /**
   * Throws an error if there is an aggregate or windowed aggregate in the
   * given clause.
   *
   * @param aggFinder Finder for the particular kind(s) of aggregate function
   * @param node      Parse tree
   * @param clause    Name of clause: "WHERE", "GROUP BY", "ON"
   */
  private void validateNoAggs(AggFinder aggFinder, SqlNode node,
      String clause) {
    final SqlCall agg = aggFinder.findAgg(node);
    if (agg == null) {
      return;
    }
    final SqlOperator op = agg.getOperator();
    if (op == SqlStdOperatorTable.OVER) {
      throw newValidationError(agg,
          RESOURCE.windowedAggregateIllegalInClause(clause));
    } else if (op.isGroup() || op.isGroupAuxiliary()) {
      throw newValidationError(agg,
          RESOURCE.groupFunctionMustAppearInGroupByClause(op.getName()));
    } else {
      throw newValidationError(agg,
          RESOURCE.aggregateIllegalInClause(clause));
    }
  }

  /** Validates a column in a USING clause, or an inferred join key in a NATURAL join. */
  private void validateCommonJoinColumn(SqlIdentifier id, SqlNode left,
      SqlNode right, SqlValidatorScope scope, boolean natural) {
    if (id.names.size() != 1) {
      throw newValidationError(id, RESOURCE.columnNotFound(id.toString()));
    }

    final RelDataType leftColType = natural
        ? checkAndDeriveDataType(id, left)
        : validateCommonInputJoinColumn(id, left, scope, natural);
    final RelDataType rightColType = validateCommonInputJoinColumn(id, right, scope, natural);
    if (!SqlTypeUtil.isComparable(leftColType, rightColType)) {
      throw newValidationError(id,
          RESOURCE.naturalOrUsingColumnNotCompatible(id.getSimple(),
              leftColType.toString(), rightColType.toString()));
    }
  }

  private RelDataType checkAndDeriveDataType(SqlIdentifier id, SqlNode node) {
    checkArgument(id.names.size() == 1);
    String name = id.names.get(0);
    SqlNameMatcher nameMatcher = getCatalogReader().nameMatcher();
    RelDataType rowType = getNamespaceOrThrow(node).getRowType();
    final RelDataTypeField field =
        requireNonNull(nameMatcher.field(rowType, name),
            () -> "unable to find left field " + name + " in " + rowType);
    return field.getType();
  }

  /** Validates a column in a USING clause, or an inferred join key in a
   * NATURAL join, in the left or right input to the join. */
  private RelDataType validateCommonInputJoinColumn(SqlIdentifier id,
      SqlNode leftOrRight, SqlValidatorScope scope, boolean natural) {
    checkArgument(id.names.size() == 1);
    final String name = id.names.get(0);
    final SqlValidatorNamespace namespace = getNamespaceOrThrow(leftOrRight);
    final RelDataType rowType = namespace.getRowType();
    final SqlNameMatcher nameMatcher = catalogReader.nameMatcher();
    final RelDataTypeField field = nameMatcher.field(rowType, name);
    if (field == null) {
      throw newValidationError(id, RESOURCE.columnNotFound(name));
    }
    Collection<RelDataType> rowTypes;
    if (!natural && rowType instanceof RelCrossType) {
      final RelCrossType crossType = (RelCrossType) rowType;
      rowTypes = new ArrayList<>(crossType.getTypes());
    } else {
      rowTypes = Collections.singleton(rowType);
    }
    for (RelDataType rowType0 : rowTypes) {
      if (nameMatcher.frequency(rowType0.getFieldNames(), name) > 1) {
        throw newValidationError(id, RESOURCE.columnInUsingNotUnique(name));
      }
    }
    checkRollUpInUsing(id, leftOrRight, scope);
    return field.getType();
  }

  /**
   * Validates a SELECT statement.
   *
   * @param select        Select statement
   * @param targetRowType Desired row type, must not be null, may be the data
   *                      type 'unknown'.
   */
  protected void validateSelect(
      SqlSelect select,
      RelDataType targetRowType) {
    requireNonNull(targetRowType, "targetRowType");

    // Namespace is either a select namespace or a wrapper around one.
    final SelectNamespace ns =
        getNamespaceOrThrow(select).unwrap(SelectNamespace.class);

    // Its rowtype is null, meaning it hasn't been validated yet.
    // This is important, because we need to take the targetRowType into
    // account.
    assert ns.rowType == null;

    SqlNode distinctNode = select.getModifierNode(SqlSelectKeyword.DISTINCT);
    if (distinctNode != null) {
      validateFeature(RESOURCE.sQLFeature_E051_01(),
          distinctNode
              .getParserPosition());
    }

    final SqlNodeList selectItems = SqlNonNullableAccessors.getSelectList(select);
    RelDataType fromType = unknownType;
    if (selectItems.size() == 1) {
      final SqlNode selectItem = selectItems.get(0);
      if (selectItem instanceof SqlIdentifier) {
        SqlIdentifier id = (SqlIdentifier) selectItem;
        if (id.isStar() && (id.names.size() == 1)) {
          // Special case: for INSERT ... VALUES(?,?), the SQL
          // standard says we're supposed to propagate the target
          // types down.  So iff the select list is an unqualified
          // star (as it will be after an INSERT ... VALUES has been
          // expanded), then propagate.
          fromType = targetRowType;
        }
      }
    }

    // Make sure that items in FROM clause have distinct aliases.
    final SelectScope fromScope = (SelectScope) getFromScope(select);
    List<@Nullable String> names = fromScope.getChildNames();
    if (!catalogReader.nameMatcher().isCaseSensitive()) {
      //noinspection RedundantTypeArguments
      names = names.stream()
          .<@Nullable String>map(s -> s == null ? null : s.toUpperCase(Locale.ROOT))
          .collect(Collectors.toList());
    }
    final int duplicateAliasOrdinal = Util.firstDuplicate(names);
    if (duplicateAliasOrdinal >= 0) {
      final ScopeChild child =
          fromScope.children.get(duplicateAliasOrdinal);
      throw newValidationError(
          requireNonNull(
              child.namespace.getEnclosingNode(),
              () -> "enclosingNode of namespace of " + child.name),
          RESOURCE.fromAliasDuplicate(child.name));
    }

    final SqlNode from = select.getFrom();
    if (from == null) {
      if (this.config.conformance().isFromRequired()) {
        throw newValidationError(select, RESOURCE.selectMissingFrom());
      }
    } else {
      validateFrom(from, fromType, fromScope);
    }


    validateWhereClause(select);
    validateGroupClause(select);
    validateHavingClause(select);
    validateWindowClause(select);
    validateQualifyClause(select);
    handleOffsetFetch(select.getOffset(), select.getFetch());

    // Validate the SELECT clause late, because a select item might
    // depend on the GROUP BY list, or the window function might reference
    // window name in the WINDOW clause etc.
    final RelDataType rowType =
        validateSelectList(selectItems, select, targetRowType);
    ns.setType(rowType);

    // Deduce which columns must be filtered.
    ns.mustFilterFields = ImmutableBitSet.of();
    if (from != null) {
      final Set<SqlQualified> qualifieds = new LinkedHashSet<>();
      for (ScopeChild child : fromScope.children) {
        final List<String> fieldNames =
            child.namespace.getRowType().getFieldNames();
        child.namespace.getMustFilterFields()
            .forEachInt(i ->
                qualifieds.add(
                    SqlQualified.create(fromScope, 1, child.namespace,
                        new SqlIdentifier(
                            ImmutableList.of(child.name, fieldNames.get(i)),
                            SqlParserPos.ZERO))));
      }
      if (!qualifieds.isEmpty()) {
        if (select.getWhere() != null) {
          forEachQualified(select.getWhere(), getWhereScope(select),
              qualifieds::remove);
        }
        if (select.getHaving() != null) {
          forEachQualified(select.getHaving(), getHavingScope(select),
              qualifieds::remove);
        }

        // Each of the must-filter fields identified must be returned as a
        // SELECT item, which is then flagged as must-filter.
        final BitSet mustFilterFields = new BitSet();
        final List<SqlNode> expandedSelectItems =
            requireNonNull(fromScope.getExpandedSelectList(),
                "expandedSelectList");
        forEach(expandedSelectItems, (selectItem, i) -> {
          selectItem = stripAs(selectItem);
          if (selectItem instanceof SqlIdentifier) {
            SqlQualified qualified =
                fromScope.fullyQualify((SqlIdentifier) selectItem);
            if (qualifieds.remove(qualified)) {
              // SELECT item #i referenced a must-filter column that was not
              // filtered in the WHERE or HAVING. It becomes a must-filter
              // column for our consumer.
              mustFilterFields.set(i);
            }
          }
        });

        // If there are must-filter fields that are not in the SELECT clause,
        // this is an error.
        if (!qualifieds.isEmpty()) {
          throw newValidationError(select,
              RESOURCE.mustFilterFieldsMissing(
                  qualifieds.stream()
                      .map(q -> q.suffix().get(0))
                      .collect(Collectors.toCollection(TreeSet::new))
                      .toString()));
        }
        ns.mustFilterFields = ImmutableBitSet.fromBitSet(mustFilterFields);
      }
    }

    // Validate ORDER BY after we have set ns.rowType because in some
    // dialects you can refer to columns of the select list, e.g.
    // "SELECT empno AS x FROM emp ORDER BY x"
    validateOrderList(select);

    if (shouldCheckForRollUp(from)) {
      checkRollUpInSelectList(select);
      checkRollUp(null, select, select.getWhere(), getWhereScope(select));
      checkRollUp(null, select, select.getHaving(), getHavingScope(select));
      checkRollUpInWindowDecl(select);
      checkRollUpInGroupBy(select);
      checkRollUpInOrderBy(select);
    }
  }

  /** For each identifier in an expression, resolves it to a qualified name
   * and calls the provided action. */
  private static void forEachQualified(SqlNode node, SqlValidatorScope scope,
      Consumer<SqlQualified> consumer) {
    node.accept(new SqlBasicVisitor<Void>() {
      @Override public Void visit(SqlIdentifier id) {
        final SqlQualified qualified = scope.fullyQualify(id);
        consumer.accept(qualified);
        return null;
      }
    });
  }

  private void checkRollUpInSelectList(SqlSelect select) {
    SqlValidatorScope scope = getSelectScope(select);
    for (SqlNode item : SqlNonNullableAccessors.getSelectList(select)) {
      if (SqlValidatorUtil.isMeasure(item)) {
        continue;
      }
      checkRollUp(null, select, item, scope);
    }
  }

  private void checkRollUpInGroupBy(SqlSelect select) {
    SqlNodeList group = select.getGroup();
    if (group != null) {
      for (SqlNode node : group) {
        checkRollUp(null, select, node, getGroupScope(select), "GROUP BY");
      }
    }
  }

  private void checkRollUpInOrderBy(SqlSelect select) {
    SqlNodeList orderList = select.getOrderList();
    if (orderList != null) {
      for (SqlNode node : orderList) {
        checkRollUp(null, select, node, getOrderScope(select), "ORDER BY");
      }
    }
  }

  private void checkRollUpInWindow(@Nullable SqlWindow window, SqlValidatorScope scope) {
    if (window != null) {
      for (SqlNode node : window.getPartitionList()) {
        checkRollUp(null, window, node, scope, "PARTITION BY");
      }

      for (SqlNode node : window.getOrderList()) {
        checkRollUp(null, window, node, scope, "ORDER BY");
      }
    }
  }

  private void checkRollUpInWindowDecl(SqlSelect select) {
    for (SqlNode decl : select.getWindowList()) {
      checkRollUpInWindow((SqlWindow) decl, getSelectScope(select));
    }
  }

  /**
   * If the {@code node} is a DOT call, returns its first operand. Recurse, if
   * the first operand is another DOT call.
   *
   * <p>In other words, it converts {@code a DOT b DOT c} to {@code a}.
   *
   * @param node The node to strip DOT
   * @return the DOT's first operand
   */
  private static SqlNode stripDot(SqlNode node) {
    SqlNode res = node;
    while (res.getKind() == SqlKind.DOT) {
      res = requireNonNull(((SqlCall) res).operand(0), "operand");
    }
    return res;
  }

  private void checkRollUp(@Nullable SqlNode grandParent, @Nullable SqlNode parent,
      @Nullable SqlNode current, SqlValidatorScope scope, @Nullable String contextClause) {
    current = stripAs(current);
    if (current instanceof SqlCall && !(current instanceof SqlSelect)) {
      // Validate OVER separately
      checkRollUpInWindow(getWindowInOver(current), scope);
      current = stripOver(current);

      SqlNode stripDot = stripDot(current);
      if (stripDot != current) {
        // we stripped the field access. Recurse to this method, the DOT's operand
        // can be another SqlCall, or an SqlIdentifier.
        checkRollUp(grandParent, parent, stripDot, scope, contextClause);
      } else if (stripDot.getKind() == SqlKind.CONVERT
          || stripDot.getKind() == SqlKind.TRANSLATE) {
        // only need to check operand[0] for CONVERT or TRANSLATE
        SqlNode child = ((SqlCall) stripDot).getOperandList().get(0);
        checkRollUp(parent, current, child, scope, contextClause);
      } else if (stripDot.getKind() == SqlKind.LAMBDA) {
        // do not need to check lambda
      } else {
        List<? extends @Nullable SqlNode> children =
            ((SqlCall) stripDot).getOperandList();
        for (SqlNode child : children) {
          checkRollUp(parent, current, child, scope, contextClause);
        }
      }
    } else if (current instanceof SqlIdentifier) {
      SqlIdentifier id = (SqlIdentifier) current;
      if (!id.isStar() && isRolledUpColumn(id, scope)) {
        if (!isAggregation(requireNonNull(parent, "parent").getKind())
            || !isRolledUpColumnAllowedInAgg(id, scope, (SqlCall) parent, grandParent)) {
          String context = contextClause != null ? contextClause : parent.getKind().toString();
          throw newValidationError(id,
              RESOURCE.rolledUpNotAllowed(SqlValidatorUtil.alias(id, 0),
                  context));
        }
      }
    }
  }

  private void checkRollUp(@Nullable SqlNode grandParent, SqlNode parent,
      @Nullable SqlNode current, SqlValidatorScope scope) {
    checkRollUp(grandParent, parent, current, scope, null);
  }

  private static @Nullable SqlWindow getWindowInOver(SqlNode over) {
    if (over.getKind() == SqlKind.OVER) {
      SqlNode window = ((SqlCall) over).getOperandList().get(1);
      if (window instanceof SqlWindow) {
        return (SqlWindow) window;
      }
      // SqlIdentifier, gets validated elsewhere
      return null;
    }
    return null;
  }

  private static SqlNode stripOver(SqlNode node) {
    switch (node.getKind()) {
    case OVER:
      return ((SqlCall) node).getOperandList().get(0);
    default:
      return node;
    }
  }

  private @Nullable Pair<String, String> findTableColumnPair(SqlIdentifier identifier,
      SqlValidatorScope scope) {
    final SqlCall call = makeNullaryCall(identifier);
    if (call != null) {
      return null;
    }
    SqlQualified qualified = scope.fullyQualify(identifier);
    List<String> names = qualified.identifier.names;

    if (names.size() < 2) {
      return null;
    }

    return new Pair<>(names.get(names.size() - 2), Util.last(names));
  }

  // Returns true iff the given column is valid inside the given aggCall.
  private boolean isRolledUpColumnAllowedInAgg(SqlIdentifier identifier, SqlValidatorScope scope,
      SqlCall aggCall, @Nullable SqlNode parent) {
    Pair<String, String> pair = findTableColumnPair(identifier, scope);

    if (pair == null) {
      return true;
    }

    String columnName = pair.right;

    Table table = resolveTable(identifier, scope);
    if (table != null) {
      return table.rolledUpColumnValidInsideAgg(columnName, aggCall, parent,
          catalogReader.getConfig());
    }
    return true;
  }

  private static @Nullable Table resolveTable(SqlIdentifier identifier,
      SqlValidatorScope scope) {
    SqlQualified fullyQualified = scope.fullyQualify(identifier);
    if (fullyQualified.namespace == null) {
      throw new IllegalArgumentException("namespace must not be null in "
          + fullyQualified);
    }
    SqlValidatorTable sqlValidatorTable =
        fullyQualified.namespace.getTable();
    if (sqlValidatorTable != null) {
      return sqlValidatorTable.table();
    }
    return null;
  }


  // Returns true iff the given column is actually rolled up.
  private boolean isRolledUpColumn(SqlIdentifier identifier, SqlValidatorScope scope) {
    Pair<String, String> pair = findTableColumnPair(identifier, scope);

    if (pair == null) {
      return false;
    }

    String columnName = pair.right;

    Table table = resolveTable(identifier, scope);
    if (table != null) {
      return table.isRolledUp(columnName);
    }
    return false;
  }

  private static boolean shouldCheckForRollUp(@Nullable SqlNode from) {
    if (from != null) {
      SqlKind kind = stripAs(from).getKind();
      return kind != SqlKind.VALUES && kind != SqlKind.SELECT;
    }
    return false;
  }

  /** Validates that a query can deliver the modality it promises. Only called
   * on the top-most SELECT or set operator in the tree. */
  private void validateModality(SqlNode query) {
    final SqlModality modality = deduceModality(query);
    if (query instanceof SqlSelect) {
      final SqlSelect select = (SqlSelect) query;
      validateModality(select, modality, true);
    } else if (query.getKind() == SqlKind.VALUES) {
      switch (modality) {
      case STREAM:
        throw newValidationError(query, Static.RESOURCE.cannotStreamValues());
      default:
        break;
      }
    } else {
      assert query.isA(SqlKind.SET_QUERY);
      final SqlCall call = (SqlCall) query;
      for (SqlNode operand : call.getOperandList()) {
        if (deduceModality(operand) != modality) {
          throw newValidationError(operand,
              Static.RESOURCE.streamSetOpInconsistentInputs());
        }
        validateModality(operand);
      }
    }
  }

  /** Return the intended modality of a SELECT or set-op. */
  private static SqlModality deduceModality(SqlNode query) {
    if (query instanceof SqlSelect) {
      SqlSelect select = (SqlSelect) query;
      return select.getModifierNode(SqlSelectKeyword.STREAM) != null
          ? SqlModality.STREAM
          : SqlModality.RELATION;
    } else if (query.getKind() == SqlKind.VALUES) {
      return SqlModality.RELATION;
    } else {
      assert query.isA(SqlKind.SET_QUERY);
      final SqlCall call = (SqlCall) query;
      return deduceModality(call.getOperandList().get(0));
    }
  }

  @Override public boolean validateModality(SqlSelect select, SqlModality modality,
      boolean fail) {
    final SelectScope scope = getRawSelectScopeNonNull(select);

    switch (modality) {
    case STREAM:
      if (scope.children.size() == 1) {
        for (ScopeChild child : scope.children) {
          if (!child.namespace.supportsModality(modality)) {
            if (fail) {
              SqlNode node = SqlNonNullableAccessors.getNode(child);
              throw newValidationError(node,
                  Static.RESOURCE.cannotConvertToStream(child.name));
            } else {
              return false;
            }
          }
        }
      } else {
        int supportsModalityCount = 0;
        for (ScopeChild child : scope.children) {
          if (child.namespace.supportsModality(modality)) {
            ++supportsModalityCount;
          }
        }

        if (supportsModalityCount == 0) {
          if (fail) {
            String inputs = String.join(", ", scope.getChildNames());
            throw newValidationError(select,
                Static.RESOURCE.cannotStreamResultsForNonStreamingInputs(inputs));
          } else {
            return false;
          }
        }
      }
      break;
    default:
      for (ScopeChild child : scope.children) {
        if (!child.namespace.supportsModality(modality)) {
          if (fail) {
            SqlNode node = SqlNonNullableAccessors.getNode(child);
            throw newValidationError(node,
                Static.RESOURCE.cannotConvertToRelation(child.name));
          } else {
            return false;
          }
        }
      }
    }

    // Make sure that aggregation is possible.
    final SqlNode aggregateNode = getAggregate(select);
    if (aggregateNode != null) {
      switch (modality) {
      case STREAM:
        SqlNodeList groupList = select.getGroup();
        if (groupList == null
            || !SqlValidatorUtil.containsMonotonic(scope, groupList)) {
          if (fail) {
            throw newValidationError(aggregateNode,
                Static.RESOURCE.streamMustGroupByMonotonic());
          } else {
            return false;
          }
        }
        break;
      default:
        break;
      }
    }

    // Make sure that ORDER BY is possible.
    final SqlNodeList orderList  = select.getOrderList();
    if (orderList != null && !orderList.isEmpty()) {
      switch (modality) {
      case STREAM:
        if (!hasSortedPrefix(scope, orderList)) {
          if (fail) {
            throw newValidationError(orderList.get(0),
                Static.RESOURCE.streamMustOrderByMonotonic());
          } else {
            return false;
          }
        }
        break;
      default:
        break;
      }
    }
    return true;
  }

  /** Returns whether the prefix is sorted. */
  private static boolean hasSortedPrefix(SelectScope scope, SqlNodeList orderList) {
    return isSortCompatible(scope, orderList.get(0), false);
  }

  private static boolean isSortCompatible(SelectScope scope, SqlNode node,
      boolean descending) {
    switch (node.getKind()) {
    case DESCENDING:
      return isSortCompatible(scope, ((SqlCall) node).getOperandList().get(0),
          true);
    default:
      break;
    }
    final SqlMonotonicity monotonicity = scope.getMonotonicity(node);
    switch (monotonicity) {
    case INCREASING:
    case STRICTLY_INCREASING:
      return !descending;
    case DECREASING:
    case STRICTLY_DECREASING:
      return descending;
    default:
      return false;
    }
  }

  @SuppressWarnings({"unchecked", "rawtypes"})
  protected void validateWindowClause(SqlSelect select) {
    final SqlNodeList windowList = select.getWindowList();
    if (windowList.isEmpty()) {
      return;
    }

    final SelectScope windowScope = (SelectScope) getFromScope(select);

    // 1. ensure window names are simple
    // 2. ensure they are unique within this scope
    for (SqlWindow window : (List<SqlWindow>) (List) windowList) {
      SqlIdentifier declName =
          requireNonNull(window.getDeclName(),
              () -> "window.getDeclName() for " + window);
      if (!declName.isSimple()) {
        throw newValidationError(declName, RESOURCE.windowNameMustBeSimple());
      }

      if (windowScope.existingWindowName(declName.toString())) {
        throw newValidationError(declName, RESOURCE.duplicateWindowName());
      } else {
        windowScope.addWindowName(declName.toString());
      }
    }

    // 7.10 rule 2
    // Check for pairs of windows which are equivalent.
    for (int i = 0; i < windowList.size(); i++) {
      SqlNode window1 = windowList.get(i);
      for (int j = i + 1; j < windowList.size(); j++) {
        SqlNode window2 = windowList.get(j);
        if (window1.equalsDeep(window2, Litmus.IGNORE)) {
          throw newValidationError(window2, RESOURCE.dupWindowSpec());
        }
      }
    }

    for (SqlWindow window : (List<SqlWindow>) (List) windowList) {
      final SqlNodeList expandedOrderList =
          (SqlNodeList) expand(window.getOrderList(), windowScope);
      window.setOrderList(expandedOrderList);
      expandedOrderList.validate(this, windowScope);

      final SqlNodeList expandedPartitionList =
          (SqlNodeList) expand(window.getPartitionList(), windowScope);
      window.setPartitionList(expandedPartitionList);
      expandedPartitionList.validate(this, windowScope);
    }

    // Hand off to validate window spec components
    windowList.validate(this, windowScope);
  }

  protected void validateQualifyClause(SqlSelect select) {
    SqlNode qualifyNode = select.getQualify();
    if (qualifyNode == null) {
      return;
    }

    SqlValidatorScope qualifyScope = getSelectScope(select);

    qualifyNode = extendedExpand(qualifyNode, qualifyScope, select, Clause.QUALIFY);
    select.setQualify(qualifyNode);

    // Bodo Change: Verify a table function is not part of the qualify condition.
    checkIfTableFunctionIsPartOfCondition(qualifyNode);

    // Bodo Change: Need to validate the qualify expression before inferring unknown types,
    // so that any aliases can be resolved before qualification occurs
    qualifyNode.validate(this, qualifyScope);

    // Bodo Change: We also need to check that the expression is valid within its scope.
    // This is only important if we have an aggregating scope.
    qualifyScope.validateExpr(qualifyNode);

    inferUnknownTypes(
        booleanType,
        qualifyScope,
        qualifyNode);

    final RelDataType type = deriveType(qualifyScope, qualifyNode);
    if (!SqlTypeUtil.inBooleanFamily(type)) {
      throw newValidationError(qualifyNode, RESOURCE.condMustBeBoolean("QUALIFY"));
    }

    boolean qualifyContainsWindowFunction = overFinder.findAgg(qualifyNode) != null;
    if (!qualifyContainsWindowFunction) {
      throw newValidationError(qualifyNode,
          RESOURCE.qualifyExpressionMustContainWindowFunction(qualifyNode.toString()));
    }
  }

  @Override public void validateWith(SqlWith with, SqlValidatorScope scope) {
    final SqlValidatorNamespace namespace = getNamespaceOrThrow(with);
    validateNamespace(namespace, unknownType);
  }

  @Override public void validateWithItem(SqlWithItem withItem) {
    SqlNodeList columnList = withItem.columnList;
    if (columnList != null) {
      final RelDataType rowType = getValidatedNodeType(withItem.query);
      final int fieldCount = rowType.getFieldCount();
      if (columnList.size() != fieldCount) {
        throw newValidationError(columnList,
            RESOURCE.columnCountMismatch());
      }
      SqlValidatorUtil.checkIdentifierListForDuplicates(
          columnList, validationErrorFunction);
    } else {
      // Luckily, field names have not been make unique yet.
      final List<String> fieldNames =
          getValidatedNodeType(withItem.query).getFieldNames();
      final int i = Util.firstDuplicate(fieldNames);
      if (i >= 0) {
        throw newValidationError(withItem.query,
            RESOURCE.duplicateColumnAndNoColumnList(fieldNames.get(i)));
      }
    }
  }

  @Override public void validateSequenceValue(SqlValidatorScope scope, SqlIdentifier id) {
    // Resolve identifier as a table.
    final SqlValidatorScope.ResolvedImpl resolved =
        new SqlValidatorScope.ResolvedImpl();
    scope.resolveTable(id.names, catalogReader.nameMatcher(),
        SqlValidatorScope.Path.EMPTY, resolved);
    if (resolved.count() != 1) {
      throw newValidationError(id, RESOURCE.tableNameNotFound(id.toString()));
    }
    // We've found a table. But is it a sequence?
    final SqlValidatorNamespace ns = resolved.only().namespace;
    if (ns instanceof TableNamespace) {
      final Table table = getTable(ns).table();
      switch (table.getJdbcTableType()) {
      case SEQUENCE:
      case TEMPORARY_SEQUENCE:
        return;
      default:
        break;
      }
    }
    throw newValidationError(id, RESOURCE.notASequence(id.toString()));
  }

  @Override public TypeCoercion getTypeCoercion() {
    assert config.typeCoercionEnabled();
    return this.typeCoercion;
  }

  @Override public Config config() {
    return requireNonNull(this.config, "config");
  }

  @Override public SqlValidator transform(UnaryOperator<Config> transform) {
    this.config = requireNonNull(transform.apply(this.config), "config");
    return this;
  }

  /**
   * Validates the ORDER BY clause of a SELECT statement.
   *
   * @param select Select statement
   */
  protected void validateOrderList(SqlSelect select) {
    // Bodo Change: Remove ORDER BYs following a non-window aggregation with no GROUP BY
    // This is because the ORDER BY is meaningless since there
    // is only one row in the result. In some queries we've observed, such ORDER BY clauses reference columns that are not selected, making the query fail to validate.
    @Nullable SqlCall agg = aggOrOverFinder.findAgg(select.getSelectList());
    boolean selectIsNonWindowAgg = agg != null && !(agg.getOperator() instanceof SqlOverOperator);
    if (selectIsNonWindowAgg && select.hasOrderBy() && select.getGroup() == null) {
        select.setOrderBy(null);
    }
    // ORDER BY is validated in a scope where aliases in the SELECT clause
    // are visible. For example, "SELECT empno AS x FROM emp ORDER BY x"
    // is valid.
    SqlNodeList orderList = select.getOrderList();
    if (orderList == null) {
      return;
    }
    if (!shouldAllowIntermediateOrderBy()) {
      if (!cursorSet.contains(select)) {
        throw newValidationError(select, RESOURCE.invalidOrderByPos());
      }
    }
    final SqlValidatorScope orderScope = getOrderScope(select);
    requireNonNull(orderScope, "orderScope");

    List<SqlNode> expandList = new ArrayList<>();
    for (SqlNode orderItem : orderList) {
      SqlNode expandedOrderItem = expand(orderItem, orderScope);
      expandList.add(expandedOrderItem);
    }

    SqlNodeList expandedOrderList =
        new SqlNodeList(expandList, orderList.getParserPosition());
    select.setOrderBy(expandedOrderList);

    for (SqlNode orderItem : expandedOrderList) {
      validateOrderItem(select, orderItem);
    }
  }

  /**
   * Validates an item in the GROUP BY clause of a SELECT statement.
   *
   * @param select Select statement
   * @param groupByItem GROUP BY clause item
   */
  private void validateGroupByItem(SqlSelect select, SqlNode groupByItem) {
    final SqlValidatorScope groupByScope = getGroupScope(select);
    validateGroupByExpr(groupByItem, groupByScope);
    groupByScope.validateExpr(groupByItem);
  }

  private void validateGroupByExpr(SqlNode groupByItem,
      SqlValidatorScope groupByScope) {
    switch (groupByItem.getKind()) {
    case GROUP_BY_DISTINCT:
      SqlCall call = (SqlCall) groupByItem;
      for (SqlNode operand : call.getOperandList()) {
        validateGroupByExpr(operand, groupByScope);
      }
      break;
    case GROUPING_SETS:
    case ROLLUP:
    case CUBE:
      call = (SqlCall) groupByItem;
      for (SqlNode operand : call.getOperandList()) {
        validateExpr(operand, groupByScope);
      }
      break;
    default:
      validateExpr(groupByItem, groupByScope);
    }
  }

  /**
   * Validates an item in the ORDER BY clause of a SELECT statement.
   *
   * @param select Select statement
   * @param orderItem ORDER BY clause item
   */
  private void validateOrderItem(SqlSelect select, SqlNode orderItem) {
    switch (orderItem.getKind()) {
    case DESCENDING:
      validateFeature(RESOURCE.sQLConformance_OrderByDesc(),
          orderItem.getParserPosition());
      validateOrderItem(select,
          ((SqlCall) orderItem).operand(0));
      return;
    default:
      break;
    }

    final SqlValidatorScope orderScope = getOrderScope(select);
    validateExpr(orderItem, orderScope);
  }

  @Override public SqlNode expandOrderExpr(SqlSelect select, SqlNode orderExpr) {
    final SqlNode orderExpr2 =
        new OrderExpressionExpander(select, orderExpr).go();
    if (orderExpr2 == orderExpr) {
      return orderExpr2;
    }

    final SqlValidatorScope scope = getOrderScope(select);
    inferUnknownTypes(unknownType, scope, orderExpr2);
    final RelDataType type = deriveType(scope, orderExpr2);
    setValidatedNodeType(orderExpr2, type);
    if (!type.isMeasure()) {
      return orderExpr2;
    }

    final SqlNode orderExpr3 = measureToValue(orderExpr2);
    final RelDataType type3 = deriveType(scope, orderExpr3);
    setValidatedNodeType(orderExpr3, type3);
    return orderExpr3;
  }

  private static SqlNode measureToValue(SqlNode e) {
    if (e.getKind() == SqlKind.V2M) {
      return ((SqlCall) e).operand(0);
    }
    return SqlInternalOperators.M2V.createCall(e.getParserPosition(), e);
  }

  /**
   * Validates the GROUP BY clause of a SELECT statement. This method is
   * called even if no GROUP BY clause is present.
   */
  protected void validateGroupClause(SqlSelect select) {
    SqlNodeList groupList = select.getGroup();
    if (groupList == null) {
      return;
    }
    final String clause = "GROUP BY";
    validateNoAggs(aggOrOverFinder, groupList, clause);
    final SqlValidatorScope groupScope = getGroupScope(select);

    // expand the expression in group list.
    List<SqlNode> expandedList = new ArrayList<>();
    for (SqlNode groupItem : groupList) {
      SqlNode expandedItem =
          extendedExpand(groupItem, groupScope, select, Clause.GROUP_BY);
      expandedList.add(expandedItem);
    }
    groupList = new SqlNodeList(expandedList, groupList.getParserPosition());
    select.setGroupBy(groupList);
    inferUnknownTypes(unknownType, groupScope, groupList);
    for (SqlNode groupItem : expandedList) {
      validateGroupByItem(select, groupItem);
    }

    // Nodes in the GROUP BY clause are expressions except if they are calls
    // to the GROUPING SETS, ROLLUP or CUBE operators; this operators are not
    // expressions, because they do not have a type.
    for (SqlNode node : groupList) {
      switch (node.getKind()) {
      case GROUP_BY_DISTINCT:
      case GROUPING_SETS:
      case ROLLUP:
      case CUBE:
        node.validate(this, groupScope);
        break;
      default:
        node.validateExpr(this, groupScope);
      }
    }

    // Derive the type of each GROUP BY item. We don't need the type, but
    // it resolves functions, and that is necessary for deducing
    // monotonicity.
    final SqlValidatorScope selectScope = getSelectScope(select);
    AggregatingSelectScope aggregatingScope = null;
    if (selectScope instanceof AggregatingSelectScope) {
      aggregatingScope = (AggregatingSelectScope) selectScope;
    }
    for (SqlNode groupItem : groupList) {
      if (groupItem instanceof SqlNodeList
          && ((SqlNodeList) groupItem).isEmpty()) {
        continue;
      }
      validateGroupItem(groupScope, aggregatingScope, groupItem);
    }

    SqlNode agg = aggFinder.findAgg(groupList);
    if (agg != null) {
      throw newValidationError(agg, RESOURCE.aggregateIllegalInClause(clause));
    }
  }

  private void validateGroupItem(SqlValidatorScope groupScope,
      @Nullable AggregatingSelectScope aggregatingScope,
      SqlNode groupItem) {
    switch (groupItem.getKind()) {
    case GROUP_BY_DISTINCT:
      for (SqlNode sqlNode : ((SqlCall) groupItem).getOperandList()) {
        validateGroupItem(groupScope, aggregatingScope, sqlNode);
      }
      break;
    case GROUPING_SETS:
    case ROLLUP:
    case CUBE:
      validateGroupingSets(groupScope, aggregatingScope, (SqlCall) groupItem);
      break;
    default:
      if (groupItem instanceof SqlNodeList) {
        break;
      }
      final RelDataType type = deriveType(groupScope, groupItem);
      setValidatedNodeType(groupItem, type);
    }
  }

  private void validateGroupingSets(SqlValidatorScope groupScope,
      @Nullable AggregatingSelectScope aggregatingScope, SqlCall groupItem) {
    for (SqlNode node : groupItem.getOperandList()) {
      validateGroupItem(groupScope, aggregatingScope, node);
    }
  }

  protected void validateWhereClause(SqlSelect select) {
    // validate WHERE clause
    final SqlNode where = select.getWhere();
    if (where == null) {
      return;
    }
    // Bodo Change: Verify a table function is not part of the where condition.
    checkIfTableFunctionIsPartOfCondition(where);
    final SqlValidatorScope whereScope = getWhereScope(select);
    final SqlNode expandedWhere = expandWithAlias(where, whereScope, select, Clause.WHERE);
    select.setWhere(expandedWhere);
    validateWhereOrOnOrNonAggregateHaving(whereScope, expandedWhere, "WHERE");
  }

  protected void validateWhereOrOnOrNonAggregateHaving(
      SqlValidatorScope scope,
      SqlNode condition,
      String clause) {
    validateNoAggs(aggOrOverOrGroupFinder, condition, clause);
    inferUnknownTypes(
        booleanType,
        scope,
        condition);
    condition.validate(this, scope);

    final RelDataType type = deriveType(scope, condition);
    if (!isReturnBooleanType(type)) {
      throw newValidationError(condition, RESOURCE.condMustBeBoolean(clause));
    }
  }

  private static boolean isReturnBooleanType(RelDataType relDataType) {
    if (relDataType instanceof RelRecordType) {
      RelRecordType recordType = (RelRecordType) relDataType;
      checkState(recordType.getFieldList().size() == 1,
          "sub-query as condition must return only one column");
      RelDataTypeField recordField = recordType.getFieldList().get(0);
      return SqlTypeUtil.inBooleanFamily(recordField.getType());
    }
    return SqlTypeUtil.inBooleanFamily(relDataType);
  }

  protected void validateHavingClause(SqlSelect select) {
    SqlNode having = select.getHaving();
    if (having == null) {
      return;
    }
    SqlNode originalHaving = having;

    // Bodo Change: Verify a table function is not part of the having condition.
    checkIfTableFunctionIsPartOfCondition(having);

    // If we have an HAVING clause, the select scope can either be an aggregating scope,
    // or a non-aggregate scope, with both having different validation paths.
    if (isAggregate(select)) {
      validateAggregateHavingClause(select, having);
    } else {
      validateNonAggregateHavingClause(select, having);
    }

  }

  protected void validateAggregateHavingClause(SqlSelect select, SqlNode having) {
    // In the case that we're handling an aggregate select,
    // HAVING is validated in the scope after groups have been created.
    // For example, in "SELECT empno FROM emp WHERE empno = 10 GROUP BY
    // deptno HAVING empno = 10", the reference to 'empno' in the HAVING
    // clause is illegal.

    SqlNode originalHaving = having;
    final AggregatingScope havingScope =
        (AggregatingScope) getSelectScope(select);
    if (config.conformance().isHavingAlias()) {
      SqlNode newExpr = extendedExpand(having, havingScope, select, Clause.HAVING);
      if (having != newExpr) {
        having = newExpr;
        select.setHaving(newExpr);
      }
    }
    if (SqlUtil.containsCall(having, call -> call.getOperator() instanceof SqlOverOperator)) {
      throw newValidationError(originalHaving, RESOURCE.windowInHavingNotAllowed());
    }
    havingScope.checkAggregateExpr(having, true);
    // Need to validate the having expression before inferring unknown types, so that any aliases
    // can be resolved before qualification occurs
    // TODO: push this change into calcite
    having.validate(this, havingScope);
    // having must be a boolean expression
    inferUnknownTypes(booleanType, havingScope, having);
    final RelDataType type = deriveType(havingScope, having);
    if (!SqlTypeUtil.inBooleanFamily(type)) {
      throw newValidationError(having, RESOURCE.havingMustBeBoolean());
    }
  }

  protected void validateNonAggregateHavingClause(SqlSelect select, SqlNode having) {
    // In the case that we're handling a non-aggregate select,
    // HAVING is semantically equivalent to a WHERE expression

    SqlNode originalHaving = having;
    final SqlValidatorScope havingScope = getSelectScope(select);
    SqlNode expandedHaving = expandWithAlias(having, havingScope, select, Clause.WHERE);
    validateWhereOrOnOrNonAggregateHaving(havingScope, expandedHaving, "HAVING");
    select.setHaving(expandedHaving);
    if (SqlUtil.containsCall(having, call -> call.getOperator() instanceof SqlOverOperator)) {
      throw newValidationError(originalHaving, RESOURCE.windowInHavingNotAllowed());
    }
  }

  protected RelDataType validateSelectList(final SqlNodeList selectItems,
      SqlSelect select, RelDataType targetRowType) {
    // First pass, ensure that aliases are unique. "*" and "TABLE.*" items
    // are ignored.

    // Validate SELECT list. Expand terms of the form "*" or "TABLE.*".
    final SqlValidatorScope selectScope = getSelectScope(select);
    final List<SqlNode> expandedSelectItems = new ArrayList<>();
    final Set<String> aliases = new HashSet<>();
    final PairList<String, RelDataType> fieldList = PairList.of();

    for (int i = 0; i < selectItems.size(); i++) {
      SqlNode selectItem = selectItems.get(i);
      if (selectItem instanceof SqlSelect) {
        handleScalarSubQuery(select, (SqlSelect) selectItem,
            expandedSelectItems, aliases, fieldList);
      } else {
        // Use the field list size to record the field index
        // because the select item may be a STAR(*), which could have been expanded.
        final int fieldIdx = fieldList.size();
        final RelDataType fieldType =
            targetRowType.isStruct()
                && targetRowType.getFieldCount() > fieldIdx
                ? targetRowType.getFieldList().get(fieldIdx).getType()
                : unknownType;
        expandSelectItem(selectItem, select, fieldType, expandedSelectItems,
            aliases, fieldList, false, i);
      }
    }

    // Create the new select list with expanded items.  Pass through
    // the original parser position so that any overall failures can
    // still reference the original input text.
    SqlNodeList newSelectList =
        new SqlNodeList(expandedSelectItems, selectItems.getParserPosition());
    if (config.identifierExpansion()) {
      select.setSelectList(newSelectList);
    }
    getRawSelectScopeNonNull(select).setExpandedSelectList(expandedSelectItems);

    // TODO: when SELECT appears as a value sub-query, should be using
    // something other than unknownType for targetRowType
    inferUnknownTypes(targetRowType, selectScope, newSelectList);

    final boolean aggregate = isAggregate(select) || select.isDistinct();
    for (SqlNode selectItem : expandedSelectItems) {
      if (SqlValidatorUtil.isMeasure(selectItem) && aggregate) {
        throw newValidationError(selectItem,
            RESOURCE.measureInAggregateQuery());
      }
      validateNoAggs(groupFinder, selectItem, "SELECT");
      validateExpr(selectItem, selectScope);
    }

    return typeFactory.createStructType(fieldList);
  }

  /**
   * Validates an expression.
   *
   * @param expr  Expression
   * @param scope Scope in which expression occurs
   */
  private void validateExpr(SqlNode expr, SqlValidatorScope scope) {
    if (expr instanceof SqlCall) {
      final SqlOperator op = ((SqlCall) expr).getOperator();
      if (op.isAggregator() && op.requiresOver()) {
        throw newValidationError(expr,
            RESOURCE.absentOverClause());
      }
      if (op instanceof SqlTableFunction) {
        throw RESOURCE.cannotCallTableFunctionHere(op.getName()).ex();
      }
    }

    // Unless 'naked measures' are enabled, a non-aggregate query cannot
    // reference measure columns. (An aggregate query can use them as
    // argument to the AGGREGATE function.)
    if (!config.nakedMeasuresInNonAggregateQuery()
        && !(scope instanceof AggregatingScope)
        && scope.isMeasureRef(expr)) {
      throw newValidationError(expr,
          RESOURCE.measureMustBeInAggregateQuery());
    }

    if (SqlValidatorUtil.isMeasure(expr) && scope instanceof SelectScope) {
      scope = getMeasureScope(((SelectScope) scope).getNode());
    }

    // Call on the expression to validate itself.
    expr.validateExpr(this, scope);

    // Perform any validation specific to the scope. For example, an
    // aggregating scope requires that expressions are valid aggregations.
    scope.validateExpr(expr);
  }

  /**
   * Processes SubQuery found in Select list. Checks that is actually Scalar
   * sub-query and makes proper entries in each of the 3 lists used to create
   * the final rowType entry.
   *
   * @param parentSelect        base SqlSelect item
   * @param selectItem          child SqlSelect from select list
   * @param expandedSelectItems Select items after processing
   * @param aliasList           built from user or system values
   * @param fieldList           Built up entries for each select list entry
   */
   private void handleScalarSubQuery(SqlSelect parentSelect,
      SqlSelect selectItem, List<SqlNode> expandedSelectItems,
      Set<String> aliasList, PairList<String, RelDataType> fieldList) {
    // A scalar sub-query only has one output column.
    if (1 != SqlNonNullableAccessors.getSelectList(selectItem).size()) {
      throw newValidationError(selectItem,
          RESOURCE.onlyScalarSubQueryAllowed());
    }

    // No expansion in this routine just append to list.
    expandedSelectItems.add(selectItem);

    // Get or generate alias and add to list.
    final String alias =
        SqlValidatorUtil.alias(selectItem, aliasList.size());
    aliasList.add(alias);

    final SelectScope scope = (SelectScope) getWhereScope(parentSelect);
    final RelDataType type = deriveType(scope, selectItem);
    setValidatedNodeType(selectItem, type);

    // We do not want to pass on the RelRecordType returned
    // by the sub-query. Just the type of the single expression
    // in the sub-query select list.
    assert type instanceof RelRecordType;
    RelRecordType rec = (RelRecordType) type;

    RelDataType nodeType = rec.getFieldList().get(0).getType();
    nodeType = typeFactory.createTypeWithNullability(nodeType, true);
    fieldList.add(alias, nodeType);
  }

  /**
   * Derives a row-type for INSERT and UPDATE operations.
   *
   * @param table            Target table for INSERT/UPDATE
   * @param targetColumnList List of target columns, or null if not specified
   * @param append           Whether to append fields to those in <code>
   *                         baseRowType</code>
   * @param targetTableAlias Target table alias, or null if not specified
   * @return Rowtype
   */
  protected RelDataType createTargetRowType(
      SqlValidatorTable table,
      @Nullable SqlNodeList targetColumnList,
      boolean append,
      @Nullable SqlIdentifier targetTableAlias) {
    RelDataType baseRowType = table.getRowType();
    if (targetColumnList == null) {
      return baseRowType;
    }
    List<RelDataTypeField> targetFields = baseRowType.getFieldList();
    final PairList<String, RelDataType> fields = PairList.of();
    if (append) {
      for (RelDataTypeField targetField : targetFields) {
        fields.add(SqlUtil.deriveAliasFromOrdinal(fields.size()),
            targetField.getType());
      }
    }
    final Set<Integer> assignedFields = new HashSet<>();
    final RelOptTable relOptTable = table instanceof RelOptTable
        ? ((RelOptTable) table) : null;
    for (SqlNode node : targetColumnList) {
      SqlIdentifier id = (SqlIdentifier) node;
      if (!id.isSimple() && targetTableAlias != null) {
        // checks that target column identifiers are prefixed with the target
        // table alias
        SqlIdentifier prefixId = id.skipLast(1);
        if (!prefixId.toString().equals(targetTableAlias.toString())) {
          throw newValidationError(prefixId,
              RESOURCE.unknownIdentifier(prefixId.toString()));
        }
      }
      RelDataTypeField targetField =
          SqlValidatorUtil.getTargetField(
              baseRowType, typeFactory, id, catalogReader, relOptTable);
      if (targetField == null) {
        throw newValidationError(id,
            RESOURCE.unknownTargetColumn(id.toString()));
      }
      if (!assignedFields.add(targetField.getIndex())) {
        throw newValidationError(id,
            RESOURCE.duplicateTargetColumn(targetField.getName()));
      }
      fields.add(targetField);
    }
    return typeFactory.createStructType(fields);
  }

  @Override public void validateInsert(SqlInsert insert) {
    final SqlValidatorNamespace targetNamespace = getNamespaceOrThrow(insert);
    validateNamespace(targetNamespace, unknownType);
    final RelOptTable relOptTable =
        SqlValidatorUtil.getRelOptTable(targetNamespace,
            catalogReader.unwrap(Prepare.CatalogReader.class), null, null);
    final SqlValidatorTable table = relOptTable == null
        ? getTable(targetNamespace)
        : relOptTable.unwrapOrThrow(SqlValidatorTable.class);

    final SqlNode source = insert.getSource();
    final SqlValidatorScope scope = scopes.get(source);

    // INSERT has an optional column name list.  If present then
    // reduce the rowtype to the columns specified.  If not present
    // then the entire target rowtype is used.
    final RelDataType targetRowType =
        createTargetRowType(
            table,
            insert.getTargetColumnList(),
            false,
            null
            );

    if (source instanceof SqlSelect) {
      final SqlSelect sqlSelect = (SqlSelect) source;
      validateSelect(sqlSelect, targetRowType);
    } else {
      requireNonNull(scope, "scope");
      validateQuery(source, scope, targetRowType);
    }

    // REVIEW jvs 4-Dec-2008: In FRG-365, this namespace row type is
    // discarding the type inferred by inferUnknownTypes (which was invoked
    // from validateSelect above).  It would be better if that information
    // were used here so that we never saw any untyped nulls during
    // checkTypeAssignment.
    final RelDataType sourceRowType = getNamespaceOrThrow(source).getRowType();
    final RelDataType logicalTargetRowType =
        getLogicalTargetRowType(targetRowType, insert);
    setValidatedNodeType(insert, logicalTargetRowType);
    final RelDataType logicalSourceRowType =
        getLogicalSourceRowType(sourceRowType, insert);

    final List<ColumnStrategy> strategies =
        table.unwrapOrThrow(RelOptTable.class).getColumnStrategies();

    final RelDataType realTargetRowType =
        typeFactory.createStructType(
            logicalTargetRowType.getFieldList()
                .stream()
                .filter(f -> strategies.get(f.getIndex()).canInsertInto())
                .collect(Collectors.toList()));

    final RelDataType targetRowTypeToValidate =
        logicalSourceRowType.getFieldCount() == logicalTargetRowType.getFieldCount()
        ? logicalTargetRowType
        : realTargetRowType;

    checkFieldCount(insert.getTargetTable(), table, strategies,
        targetRowTypeToValidate, realTargetRowType,
        source, logicalSourceRowType, logicalTargetRowType);

    checkTypeAssignment(scopes.get(source),
        table,
        logicalSourceRowType,
        targetRowTypeToValidate,
        insert);

    checkConstraint(table, source, logicalTargetRowType);

    validateAccess(insert.getTargetTable(), table, SqlAccessEnum.INSERT);

    // Refresh the insert row type to keep sync with source.
    setValidatedNodeType(insert, targetRowTypeToValidate);
  }

  /**
   * Validates insert values against the constraint of a modifiable view.
   *
   * @param validatorTable Table that may wrap a ModifiableViewTable
   * @param source        The values being inserted
   * @param targetRowType The target type for the view
   */
  private void checkConstraint(
      SqlValidatorTable validatorTable,
      SqlNode source,
      RelDataType targetRowType) {
    final ModifiableViewTable modifiableViewTable =
        validatorTable.unwrap(ModifiableViewTable.class);
    if (modifiableViewTable != null && source instanceof SqlCall) {
      final Table table = modifiableViewTable.getTable();
      final RelDataType tableRowType = table.getRowType(typeFactory);
      final List<RelDataTypeField> tableFields = tableRowType.getFieldList();

      // Get the mapping from column indexes of the underlying table
      // to the target columns and view constraints.
      final Map<Integer, RelDataTypeField> tableIndexToTargetField =
          SqlValidatorUtil.getIndexToFieldMap(tableFields, targetRowType);
      final Map<Integer, RexNode> projectMap =
          RelOptUtil.getColumnConstraints(modifiableViewTable, targetRowType, typeFactory);

      // Determine columns (indexed to the underlying table) that need
      // to be validated against the view constraint.
      @SuppressWarnings("RedundantCast")
      final ImmutableBitSet targetColumns =
          ImmutableBitSet.of((Iterable<Integer>) tableIndexToTargetField.keySet());
      @SuppressWarnings("RedundantCast")
      final ImmutableBitSet constrainedColumns =
          ImmutableBitSet.of((Iterable<Integer>) projectMap.keySet());
      @SuppressWarnings("assignment.type.incompatible")
      List<@KeyFor({"tableIndexToTargetField", "projectMap"}) Integer> constrainedTargetColumns =
          targetColumns.intersect(constrainedColumns).asList();

      // Validate insert values against the view constraint.
      final List<SqlNode> values = ((SqlCall) source).getOperandList();
      for (final int colIndex : constrainedTargetColumns) {
        final String colName = tableFields.get(colIndex).getName();
        final RelDataTypeField targetField =
            requireNonNull(tableIndexToTargetField.get(colIndex));
        for (SqlNode row : values) {
          final SqlCall call = (SqlCall) row;
          final SqlNode sourceValue = call.operand(targetField.getIndex());
          final ValidationError validationError =
              new ValidationError(sourceValue,
                  RESOURCE.viewConstraintNotSatisfied(colName,
                      Util.last(validatorTable.getQualifiedName())));
          RelOptUtil.validateValueAgainstConstraint(sourceValue,
              projectMap.get(colIndex), validationError);
        }
      }
    }
  }

  /**
   * Validates updates against the constraint of a modifiable view.
   *
   * @param validatorTable A {@link SqlValidatorTable} that may wrap a
   *                       ModifiableViewTable
   * @param update         The UPDATE parse tree node
   * @param targetRowType  The target type
   */
  private void checkConstraint(
      SqlValidatorTable validatorTable,
      SqlUpdate update,
      RelDataType targetRowType) {
    final ModifiableViewTable modifiableViewTable =
        validatorTable.unwrap(ModifiableViewTable.class);
    if (modifiableViewTable != null) {
      final Table table = modifiableViewTable.getTable();
      final RelDataType tableRowType = table.getRowType(typeFactory);

      final Map<Integer, RexNode> projectMap =
          RelOptUtil.getColumnConstraints(modifiableViewTable, targetRowType,
              typeFactory);
      final Map<String, Integer> nameToIndex =
          SqlValidatorUtil.mapNameToIndex(tableRowType.getFieldList());

      // Validate update values against the view constraint.
      final List<String> targetNames =
          SqlIdentifier.simpleNames(update.getTargetColumnList());
      final List<SqlNode> sources = update.getSourceExpressionList();
      Pair.forEach(targetNames, sources, (columnName, expr) -> {
        final Integer columnIndex = nameToIndex.get(columnName);
        if (projectMap.containsKey(columnIndex)) {
          final RexNode columnConstraint = projectMap.get(columnIndex);
          final ValidationError validationError =
              new ValidationError(expr,
                  RESOURCE.viewConstraintNotSatisfied(columnName,
                      Util.last(validatorTable.getQualifiedName())));
          RelOptUtil.validateValueAgainstConstraint(expr,
              columnConstraint, validationError);
        }
      });
    }
  }

  /**
   * Check the field count of sql insert source and target node row type.
   *
   * @param node                    target table sql identifier
   * @param table                   target table
   * @param strategies              column strategies of target table
   * @param targetRowTypeToValidate row type to validate mainly for column strategies
   * @param realTargetRowType       target table row type exclusive virtual columns
   * @param source                  source node
   * @param logicalSourceRowType    source node row type
   * @param logicalTargetRowType    logical target row type, contains only target columns if
   *                                they are specified or if the sql dialect allows subset insert,
   *                                make a subset of fields(start from the left first field) whose
   *                                length is equals with the source row type fields number
   */
  private void checkFieldCount(SqlNode node, SqlValidatorTable table,
      List<ColumnStrategy> strategies, RelDataType targetRowTypeToValidate,
      RelDataType realTargetRowType, SqlNode source,
      RelDataType logicalSourceRowType, RelDataType logicalTargetRowType) {
    final int sourceFieldCount = logicalSourceRowType.getFieldCount();
    final int targetFieldCount = logicalTargetRowType.getFieldCount();
    final int targetRealFieldCount = realTargetRowType.getFieldCount();
    if (sourceFieldCount != targetFieldCount
        && sourceFieldCount != targetRealFieldCount) {
      // Allows the source row fields count to be equal with either
      // the logical or the real(excludes columns that can not insert into)
      // target row fields count.
      throw newValidationError(node,
          RESOURCE.unmatchInsertColumn(targetFieldCount, sourceFieldCount));
    }
    // Ensure that non-nullable fields are targeted.
    for (final RelDataTypeField field : table.getRowType().getFieldList()) {
      final RelDataTypeField targetField =
          targetRowTypeToValidate.getField(field.getName(), true, false);
      switch (strategies.get(field.getIndex())) {
      case NOT_NULLABLE:
        assert !field.getType().isNullable();
        if (targetField == null) {
          throw newValidationError(node,
              RESOURCE.columnNotNullable(field.getName()));
        }
        break;
      case NULLABLE:
        assert field.getType().isNullable();
        break;
      case VIRTUAL:
      case STORED:
        if (targetField != null
            && !isValuesWithDefault(source, targetField.getIndex())) {
          throw newValidationError(node,
              RESOURCE.insertIntoAlwaysGenerated(field.getName()));
        }
        break;
      default:
        break;
      }
    }
  }

  /** Returns whether a query uses {@code DEFAULT} to populate a given
   * column. */
  private static boolean isValuesWithDefault(SqlNode source, int column) {
    switch (source.getKind()) {
    case VALUES:
      for (SqlNode operand : ((SqlCall) source).getOperandList()) {
        if (!isRowWithDefault(operand, column)) {
          return false;
        }
      }
      return true;
    default:
      break;
    }
    return false;
  }

  private static boolean isRowWithDefault(SqlNode operand, int column) {
    switch (operand.getKind()) {
    case ROW:
      final SqlCall row = (SqlCall) operand;
      return row.getOperandList().size() >= column
          && row.getOperandList().get(column).getKind() == SqlKind.DEFAULT;
    default:
      break;
    }
    return false;
  }

  protected RelDataType getLogicalTargetRowType(
      RelDataType targetRowType,
      SqlInsert insert) {
    if (insert.getTargetColumnList() == null
        && this.config.conformance().isInsertSubsetColumnsAllowed()) {
      // Target an implicit subset of columns.
      final SqlNode source = insert.getSource();
      final RelDataType sourceRowType = getNamespaceOrThrow(source).getRowType();
      final RelDataType logicalSourceRowType =
          getLogicalSourceRowType(sourceRowType, insert);
      final RelDataType implicitTargetRowType =
          typeFactory.createStructType(
              targetRowType.getFieldList()
                  .subList(0, logicalSourceRowType.getFieldCount()));
      final SqlValidatorNamespace targetNamespace = getNamespaceOrThrow(insert);
      validateNamespace(targetNamespace, implicitTargetRowType);
      return implicitTargetRowType;
    } else {
      // Either the set of columns are explicitly targeted, or target the full
      // set of columns.
      return targetRowType;
    }
  }

  protected RelDataType getLogicalSourceRowType(
      RelDataType sourceRowType,
      SqlInsert insert) {
    return sourceRowType;
  }

  /**
   * Checks the type assignment of an INSERT or UPDATE query.
   *
   * <p>Skip the virtual columns(can not insert into) type assignment
   * check if the source fields count equals with
   * the real target table fields count, see how #checkFieldCount was used.
   *
   * @param sourceScope   Scope of query source which is used to infer node type
   * @param table         Target table
   * @param sourceRowType Source row type
   * @param targetRowType Target row type, it should either contain all the virtual columns
   *                      (can not insert into) or exclude all the virtual columns
   * @param query The query
   */
  protected void checkTypeAssignment(
      @Nullable SqlValidatorScope sourceScope,
      SqlValidatorTable table,
      RelDataType sourceRowType,
      RelDataType targetRowType,
      final SqlNode query) {
    // NOTE jvs 23-Feb-2006: subclasses may allow for extra targets
    // representing system-maintained columns, so stop after all sources
    // matched
    boolean isUpdateModifiableViewTable = false;
    if (query instanceof SqlUpdate) {
      final SqlNodeList targetColumnList =
          requireNonNull(((SqlUpdate) query).getTargetColumnList());
      final int targetColumnCount = targetColumnList.size();
      targetRowType =
          SqlTypeUtil.extractLastNFields(typeFactory, targetRowType,
              targetColumnCount);
      sourceRowType =
          SqlTypeUtil.extractLastNFields(typeFactory, sourceRowType,
              targetColumnCount);
      isUpdateModifiableViewTable =
          table.unwrap(ModifiableViewTable.class) != null;
    }
    if (SqlTypeUtil.equalAsStructSansNullability(typeFactory,
        sourceRowType, targetRowType, null)) {
      // Returns early if source and target row type equals sans nullability.
      return;
    }
    if (config.typeCoercionEnabled() && !isUpdateModifiableViewTable) {
      // Try type coercion first if implicit type coercion is allowed.
      boolean coerced =
          typeCoercion.querySourceCoercion(sourceScope, sourceRowType,
              targetRowType, query);
      if (coerced) {
        return;
      }
    }

    // Fall back to default behavior: compare the type families.
    List<RelDataTypeField> sourceFields = sourceRowType.getFieldList();
    List<RelDataTypeField> targetFields = targetRowType.getFieldList();
    final int sourceCount = sourceFields.size();
    for (int i = 0; i < sourceCount; ++i) {
      RelDataType sourceType = sourceFields.get(i).getType();
      RelDataType targetType = targetFields.get(i).getType();
      if (!SqlTypeUtil.canAssignFrom(targetType, sourceType)) {
        SqlNode node = getNthExpr(query, i, sourceCount);
        if (node instanceof SqlDynamicParam) {
          continue;
        }
        String targetTypeString;
        String sourceTypeString;
        if (SqlTypeUtil.areCharacterSetsMismatched(
            sourceType,
            targetType)) {
          sourceTypeString = sourceType.getFullTypeString();
          targetTypeString = targetType.getFullTypeString();
        } else {
          sourceTypeString = sourceType.toString();
          targetTypeString = targetType.toString();
        }
        throw newValidationError(node,
            RESOURCE.typeNotAssignable(
                targetFields.get(i).getName(), targetTypeString,
                sourceFields.get(i).getName(), sourceTypeString));
      }
    }
  }

  /**
   * Locates the n'th expression in an INSERT or UPDATE query.
   *
   * @param query       Query
   * @param ordinal     Ordinal of expression
   * @param sourceCount Number of expressions
   * @return Ordinal'th expression, never null
   */
  private static SqlNode getNthExpr(SqlNode query, int ordinal, int sourceCount) {
    if (query instanceof SqlInsert) {
      SqlInsert insert = (SqlInsert) query;
      if (insert.getTargetColumnList() != null) {
        return insert.getTargetColumnList().get(ordinal);
      } else {
        return getNthExpr(
            insert.getSource(),
            ordinal,
            sourceCount);
      }
    } else if (query instanceof SqlUpdate) {
      SqlUpdate update = (SqlUpdate) query;
      if (update.getSourceExpressionList() != null) {
        return update.getSourceExpressionList().get(ordinal);
      } else {
        return getNthExpr(SqlNonNullableAccessors.getSourceSelect(update),
            ordinal, sourceCount);
      }
    } else if (query instanceof SqlSelect) {
      SqlSelect select = (SqlSelect) query;
      SqlNodeList selectList = SqlNonNullableAccessors.getSelectList(select);
      if (selectList.size() == sourceCount) {
        return selectList.get(ordinal);
      } else {
        return query; // give up
      }
    } else {
      return query; // give up
    }
  }

  @Override public void validateDelete(SqlDelete call) {
    final SqlSelect sqlSelect = SqlNonNullableAccessors.getSourceSelect(call);
    validateSelect(sqlSelect, unknownType);

    final SqlValidatorNamespace targetNamespace = getNamespaceOrThrow(call);
    validateNamespace(targetNamespace, unknownType);
    final SqlValidatorTable table = targetNamespace.getTable();

    validateAccess(call.getTargetTable(), table, SqlAccessEnum.DELETE);
  }

  @Override public void validateUpdate(SqlUpdate call) {
    final SqlValidatorNamespace targetNamespace = getNamespaceOrThrow(call);
    validateNamespace(targetNamespace, unknownType);
    final RelOptTable relOptTable =
        SqlValidatorUtil.getRelOptTable(targetNamespace,
            castNonNull(catalogReader.unwrap(Prepare.CatalogReader.class)),
            null, null);
    final SqlValidatorTable table = relOptTable == null
        ? getTable(targetNamespace)
        : relOptTable.unwrapOrThrow(SqlValidatorTable.class);

    final RelDataType targetRowType =
        createTargetRowType(table, call.getTargetColumnList(), true,
            call.getAlias());

    final SqlSelect select = SqlNonNullableAccessors.getSourceSelect(call);
    validateSelect(select, targetRowType);

    final RelDataType sourceRowType = getValidatedNodeType(select);
    checkTypeAssignment(scopes.get(select), table, sourceRowType, targetRowType,
        call);

    checkConstraint(table, call, targetRowType);

    validateAccess(call.getTargetTable(), table, SqlAccessEnum.UPDATE);
  }

  @Override public void validateCreateTable(SqlCreateTable createTable) {

    // Issue: If both "IF NOT EXISTS" and "OR REPLACE", are specified,
    // Snowflake throws the error: "IF NOT EXISTS and OR REPLACE are incompatible."
    // I'm not sure if this is true in other dialects, but I'm going to assume
    // it is
    // If it isn't it will be handled as a followup issue:
    // https://bodo.atlassian.net/browse/BE-4429

    if (createTable.ifNotExists && createTable.getReplace()) {
      throw newValidationError(createTable, BODO_SQL_RESOURCE.createTableInvalidSyntax());
    }


    final SqlNode queryNode = createTable.getQuery();
    if (queryNode == null) {
      throw newValidationError(createTable, BODO_SQL_RESOURCE.createTableRequiresAsQuery());
    }

    // Note, this can either a row expression or a query expression with an optional ORDER BY
    // We're not currently handling the row expression case.

    SqlValidatorScope createTableScope = this.getCreateTableScope(createTable);
    // In order to be sufficiently general to the input of Create table,
    // We have to validate this expression in the overall scope of the create table node,
    // Since the associated query node
    // doesn't necessarily have to be a select (notably, validaSelect doesn't work if the query has
    // any 'with' clauses)
    // Note that we also can't use validateScopedExpression, as this can rewrite the sqlNode
    // via a call to performUnconditionalRewrites, which should have already been done by the time
    // that we reach this points, and calling performUnconditionalRewrites twice is likely invalid.

    if (!(queryNode instanceof SqlIdentifier)) {
      queryNode.validate(this, createTableScope);
    } else {
      // Validate the namespace representation of the node,
      // This is needed, as attempting to validate this identifier in the
      // createTableScope will result in it being treated as a column identifier
      // instead of a table.
      requireNonNull(getNamespace(queryNode)).validate(unknownType);

      final SqlValidatorScope.ResolvedImpl resolved = new SqlValidatorScope.ResolvedImpl();
      createTableScope.resolveTable(((SqlIdentifier) queryNode).names, catalogReader.nameMatcher(),
          SqlValidatorScope.Path.EMPTY, resolved);
      if (resolved.count() != 1) {
        throw newValidationError(queryNode, RESOURCE.tableNameNotFound(queryNode.toString()));
      }
    }


    final SqlIdentifier tableNameNode = createTable.getName();
    final List<String> names = tableNameNode.names;

    //Create an empty resolve to accumulate the results
    final SqlValidatorScope.ResolvedImpl resolved = new SqlValidatorScope.ResolvedImpl();

    createTableScope.resolveSchema(
        Util.skipLast(names), //Skip the last name element (the table name)
        this.catalogReader.nameMatcher(),
        SqlValidatorScope.Path.EMPTY,
        resolved
    );


    SqlValidatorScope.Resolve bestResolve;
    if (resolved.count() != 1) {
      //If we have multiple resolutions, they're all invalid,
      //but we want to pick the closest resolution to throw the best error.
      bestResolve = resolved.resolves.get(0);
      for (int i = 1; i < resolved.count(); i++) {
        SqlValidatorScope.Resolve curResolve = resolved.resolves.get(i);
        if (curResolve.remainingNames.size() < bestResolve.remainingNames.size()) {
          bestResolve = curResolve;
        }
      }
    } else {
      bestResolve = resolved.only();
    }

    Schema resolvedSchema = ((SchemaNamespace) bestResolve.namespace).getSchema();
    if (!bestResolve.remainingNames.isEmpty()) {
      throw new RuntimeException(
          "Unable to find schema "
              + String.join(".", bestResolve.remainingNames)
              + " in " + bestResolve.path);
    }

    if (!resolvedSchema.isMutable()) {
      throw new RuntimeException("Error: Schema " + bestResolve.path + " is not mutable.");
    }

    createTable.setValidationInformation(Util.last(names),
        resolvedSchema, bestResolve.path.stepNames());

  }

  @Override public void validateMerge(SqlMerge call) {

    SqlSelect sqlSelect = SqlNonNullableAccessors.getSourceSelect(call);
    // REVIEW zfong 5/25/06 - Does an actual type have to be passed into
    // validateSelect()?

    // REVIEW jvs 6-June-2006:  In general, passing unknownType like
    // this means we won't be able to correctly infer the types
    // for dynamic parameter markers (SET x = ?).  But
    // maybe validateUpdate and validateInsert below will do
    // the job?

    // REVIEW ksecretan 15-July-2011: They didn't get a chance to
    // since validateSelect() would bail.
    // Let's use the update/insert targetRowType when available.
    SqlValidatorNamespace targetNamespace = getNamespaceOrThrow(call.getTargetTable());
    validateNamespace(targetNamespace, unknownType);

    SqlValidatorTable table = targetNamespace.getTable();
    validateAccess(call.getTargetTable(), table, SqlAccessEnum.UPDATE);

    // Bodo Change: The target row type for merge into should always match
    // the original type of the target table because the only valid options
    // are to:
    //
    // 1. Insert rows
    // 2. Update rows
    // 3. Delete rows
    //
    // None of these options can result in deleting a column or modifying
    // its type.
    //
    RelDataType targetRowType = table.getRowType();

    validateSelect(sqlSelect, targetRowType);

    // Now verify each of the individual clauses, and ensure that we don't encounter
    // a conditional match clause after an unconditional match clause
    boolean seenUnconditionalCondition = false;
    for (int i = 0; i < call.getMatchedCallList().size(); i++) {
      SqlNode matchCallAfterValidate = call.getMatchedCallList().get(i);
      if (seenUnconditionalCondition) {
        throw newValidationError(call.getMatchedCallList(),
            BODO_SQL_RESOURCE.mergeClauseUnconditionalPrecedesConditional());
      }

      SqlNode cond;
      if (matchCallAfterValidate instanceof SqlUpdate) {
        validateUpdate((SqlUpdate) matchCallAfterValidate);
        cond = ((SqlUpdate) matchCallAfterValidate).getCondition();
      } else {
        validateDelete((SqlDelete) matchCallAfterValidate);
        cond = ((SqlDelete) matchCallAfterValidate).getCondition();
      }

      // Verify that we don't encounter a conditional match clause after an unconditional
      // match clause. Note booleanValue is safe, as we've validated the Insert, which requires
      // the condition to be boolean
      if (cond == null || (cond instanceof SqlLiteral && ((SqlLiteral) cond).booleanValue())) {
        seenUnconditionalCondition = true;
      }
    }

    seenUnconditionalCondition = false;
    for (int i = 0; i < call.getNotMatchedCallList().size(); i++) {
      SqlInsert insertCallAfterValidate = (SqlInsert) call.getNotMatchedCallList().get(i);
      if (seenUnconditionalCondition) {
        throw newValidationError(call.getNotMatchedCallList(),
                BODO_SQL_RESOURCE.mergeClauseUnconditionalPrecedesConditional());
      }
      validateInsert(insertCallAfterValidate);
      // Throw if select list contains NULL literal and target is NOT NULL
      if (insertCallAfterValidate.getSource() instanceof SqlSelect) {
        final SqlSelect sourceSelect = (SqlSelect) insertCallAfterValidate.getSource();
        final SqlNodeList sourceSelectList = sourceSelect.getSelectList();
        // Bodo Change: Update to match Calcite's code for deriving the row
        // type to check for each individual insert. This seems like it shouldn't
        // be necessary, but may help future proof the code.
        //
        // TODO: Extend this check to update as well, athough that is missing from Calcite.
        final RelDataType prunedTargetRowType = createTargetRowType(table, insertCallAfterValidate.getTargetColumnList(), false,
            call.getAlias());
        for (int j = 0; j < sourceSelectList.size(); j++) {
          final RelDataTypeField targetField = prunedTargetRowType.getFieldList().get(j);
          final SqlNode selectItem = sourceSelect.getSelectList().get(j);
          if (!targetField.getType().isNullable() && SqlUtil.isNullLiteral(selectItem, true)) {
            throw newValidationError(selectItem,
                RESOURCE.columnNotNullable(targetField.getName()));
          }
        }
      }

      SqlNode cond = insertCallAfterValidate.getCondition();

      // Verify that we don't encounter a conditional match clause after an unconditional
      // match clause. Note booleanValue is safe, as we've validated the Insert, which requires
      // the condition to be boolean
      if (cond == null || (cond instanceof SqlLiteral && ((SqlLiteral) cond).booleanValue())) {
        seenUnconditionalCondition = true;
      }
    }
  }

  /**
   * Validates access to a table.
   *
   * @param table          Table
   * @param requiredAccess Access requested on table
   */
  protected void validateAccess(
      SqlNode node,
      @Nullable SqlValidatorTable table,
      SqlAccessEnum requiredAccess) {
    if (table != null) {
      SqlAccessType access = table.getAllowedAccess();
      if (!access.allowsAccess(requiredAccess)) {
        throw newValidationError(node,
            RESOURCE.accessNotAllowed(requiredAccess.name(),
                table.getQualifiedName().toString()));
      }
    }
  }

  /**
   * Validates snapshot to a table.
   *
   * @param node  The node to validate
   * @param scope Validator scope to derive type
   * @param ns    The namespace to lookup table
   */
  private void validateSnapshot(
      SqlNode node,
      @Nullable SqlValidatorScope scope,
      SqlValidatorNamespace ns) {
    if (node.getKind() == SqlKind.SNAPSHOT) {
      SqlSnapshot snapshot = (SqlSnapshot) node;
      SqlNode period = snapshot.getPeriod();
      RelDataType dataType = deriveType(requireNonNull(scope, "scope"), period);
      if (!SqlTypeUtil.isTimestamp(dataType)) {
        throw newValidationError(period,
            Static.RESOURCE.illegalExpressionForTemporal(dataType.getSqlTypeName().getName()));
      }
      SqlValidatorTable table = getTable(ns);
      if (!table.isTemporal()) {
        List<String> qualifiedName = table.getQualifiedName();
        String tableName = qualifiedName.get(qualifiedName.size() - 1);
        throw newValidationError(snapshot.getTableRef(),
            Static.RESOURCE.notTemporalTable(tableName));
      }
    }
  }

  /**
   * Validates a VALUES clause.
   *
   * @param node          Values clause
   * @param targetRowType Row type which expression must conform to
   * @param scope         Scope within which clause occurs
   */
  protected void validateValues(
      SqlCall node,
      RelDataType targetRowType,
      final SqlValidatorScope scope) {
    assert node.getKind() == SqlKind.VALUES;

    final List<SqlNode> operands = node.getOperandList();
    for (SqlNode operand : operands) {
      if (!(operand.getKind() == SqlKind.ROW)) {
        throw Util.needToImplement(
            "Values function where operands are scalars");
      }

      SqlCall rowConstructor = (SqlCall) operand;
      if (this.config.conformance().isInsertSubsetColumnsAllowed()
          && targetRowType.isStruct()
          && rowConstructor.operandCount() < targetRowType.getFieldCount()) {
        targetRowType =
            typeFactory.createStructType(
                targetRowType.getFieldList()
                    .subList(0, rowConstructor.operandCount()));
      } else if (targetRowType.isStruct()
          && rowConstructor.operandCount() != targetRowType.getFieldCount()) {
        return;
      }

      inferUnknownTypes(
          targetRowType,
          scope,
          rowConstructor);

      if (targetRowType.isStruct()) {
        for (Pair<SqlNode, RelDataTypeField> pair
            : Pair.zip(rowConstructor.getOperandList(),
                targetRowType.getFieldList())) {
          if (!pair.right.getType().isNullable()
              && SqlUtil.isNullLiteral(pair.left, false)) {
            throw newValidationError(node,
                RESOURCE.columnNotNullable(pair.right.getName()));
          }
        }
      }
    }

    for (SqlNode operand : operands) {
      operand.validate(this, scope);
    }

    // validate that all row types have the same number of columns
    //  and that expressions in each column are compatible.
    // A values expression is turned into something that looks like
    // ROW(type00, type01,...), ROW(type11,...),...
    final int rowCount = operands.size();
    if (rowCount >= 2) {
      SqlCall firstRow = (SqlCall) operands.get(0);
      final int columnCount = firstRow.operandCount();

      // 1. check that all rows have the same cols length
      for (SqlNode operand : operands) {
        SqlCall thisRow = (SqlCall) operand;
        if (columnCount != thisRow.operandCount()) {
          throw newValidationError(node,
              RESOURCE.incompatibleValueType(
                  SqlStdOperatorTable.VALUES.getName()));
        }
      }

      // 2. check if types at i:th position in each row are compatible
      for (int col = 0; col < columnCount; col++) {
        final int c = col;
        final RelDataType type =
            typeFactory.leastRestrictive(
                new AbstractList<RelDataType>() {
                  @Override public RelDataType get(int row) {
                    SqlCall thisRow = (SqlCall) operands.get(row);
                    return deriveType(scope, thisRow.operand(c));
                  }

                  @Override public int size() {
                    return rowCount;
                  }
                });

        if (null == type) {
          throw newValidationError(node,
              RESOURCE.incompatibleValueType(
                  SqlStdOperatorTable.VALUES.getName()));
        }
      }
    }
  }

  @Override public void validateDataType(SqlDataTypeSpec dataType) {
  }

  @Override public void validateDynamicParam(SqlDynamicParam dynamicParam) {
  }

  /**
   * Throws a validator exception with access to the validator context.
   * The exception is determined when an instance is created.
   */
  private class ValidationError implements Supplier<CalciteContextException> {
    private final SqlNode sqlNode;
    private final Resources.ExInst<SqlValidatorException> validatorException;

    ValidationError(SqlNode sqlNode,
        Resources.ExInst<SqlValidatorException> validatorException) {
      this.sqlNode = sqlNode;
      this.validatorException = validatorException;
    }

    @Override public CalciteContextException get() {
      return newValidationError(sqlNode, validatorException);
    }
  }

  /**
   * Throws a validator exception with access to the validator context.
   * The exception is determined when the function is applied.
   */
  class ValidationErrorFunction
      implements BiFunction<SqlNode, Resources.ExInst<SqlValidatorException>,
            CalciteContextException> {
    @Override public CalciteContextException apply(
        SqlNode v0, Resources.ExInst<SqlValidatorException> v1) {
      return newValidationError(v0, v1);
    }
  }

  public ValidationErrorFunction getValidationErrorFunction() {
    return validationErrorFunction;
  }

  @Override public CalciteContextException newValidationError(SqlNode node,
      Resources.ExInst<SqlValidatorException> e) {
    requireNonNull(node, "node");
    final SqlParserPos pos = node.getParserPosition();
    return SqlUtil.newContextException(pos, e);
  }

  protected SqlWindow getWindowByName(
      SqlIdentifier id,
      SqlValidatorScope scope) {
    SqlWindow window = null;
    if (id.isSimple()) {
      final String name = id.getSimple();
      window = scope.lookupWindow(name);
    }
    if (window == null) {
      throw newValidationError(id, RESOURCE.windowNotFound(id.toString()));
    }
    return window;
  }

  @Override public SqlWindow resolveWindow(
      SqlNode windowOrRef,
      SqlValidatorScope scope) {
    SqlWindow window;
    if (windowOrRef instanceof SqlIdentifier) {
      window = getWindowByName((SqlIdentifier) windowOrRef, scope);
    } else {
      window = (SqlWindow) windowOrRef;
    }
    while (true) {
      final SqlIdentifier refId = window.getRefName();
      if (refId == null) {
        break;
      }
      final String refName = refId.getSimple();
      SqlWindow refWindow = scope.lookupWindow(refName);
      if (refWindow == null) {
        throw newValidationError(refId, RESOURCE.windowNotFound(refName));
      }
      window = window.overlay(refWindow, this);
    }

    return window;
  }

  public SqlNode getOriginal(SqlNode expr) {
    SqlNode original = originalExprs.get(expr);
    if (original == null) {
      original = expr;
    }
    return original;
  }

  public void setOriginal(SqlNode expr, SqlNode original) {
    // Don't overwrite the original original.
    originalExprs.putIfAbsent(expr, original);
  }

  @Nullable SqlValidatorNamespace lookupFieldNamespace(RelDataType rowType, String name) {
    final SqlNameMatcher nameMatcher = catalogReader.nameMatcher();
    final RelDataTypeField field = nameMatcher.field(rowType, name);
    if (field == null) {
      return null;
    }
    return new FieldNamespace(this, field.getType());
  }

  @Override public void validateWindow(
      SqlNode windowOrId,
      SqlValidatorScope scope,
      @Nullable SqlCall call) {
    // Enable nested aggregates with window aggregates (OVER operator)
    inWindow = true;

    final SqlWindow targetWindow;
    switch (windowOrId.getKind()) {
    case IDENTIFIER:
      // Just verify the window exists in this query.  It will validate
      // when the definition is processed
      targetWindow = getWindowByName((SqlIdentifier) windowOrId, scope);
      break;
    case WINDOW:
      targetWindow = (SqlWindow) windowOrId;
      break;
    default:
      throw Util.unexpected(windowOrId.getKind());
    }

    requireNonNull(call, () -> "call is null when validating windowOrId " + windowOrId);
    assert targetWindow.getWindowCall() == null;
    targetWindow.setWindowCall(call);
    targetWindow.validate(this, scope);
    targetWindow.setWindowCall(null);
    call.validate(this, scope);

    validateAggregateParams(call, null, null, null, scope);

    // Disable nested aggregates post validation
    inWindow = false;
  }

  @Override public void validateLambda(SqlLambda lambdaExpr) {
    final SqlLambdaScope scope = (SqlLambdaScope) scopes.get(lambdaExpr);
    requireNonNull(scope, "scope");
    final LambdaNamespace ns =
        getNamespaceOrThrow(lambdaExpr).unwrap(LambdaNamespace.class);

    deriveType(scope, lambdaExpr.getExpression());
    RelDataType type = deriveTypeImpl(scope, lambdaExpr);
    setValidatedNodeType(lambdaExpr, type);
    ns.setType(type);
  }

  @Override public void validateMatchRecognize(SqlCall call) {
    final SqlMatchRecognize matchRecognize = (SqlMatchRecognize) call;
    final MatchRecognizeScope scope =
        (MatchRecognizeScope) getMatchRecognizeScope(matchRecognize);

    final MatchRecognizeNamespace ns =
        getNamespaceOrThrow(call).unwrap(MatchRecognizeNamespace.class);
    assert ns.rowType == null;

    // rows per match
    final SqlLiteral rowsPerMatch = matchRecognize.getRowsPerMatch();
    final boolean allRows = rowsPerMatch != null
        && rowsPerMatch.getValue()
        == SqlMatchRecognize.RowsPerMatchOption.ALL_ROWS;

    final RelDataTypeFactory.Builder typeBuilder = typeFactory.builder();

    // parse PARTITION BY column
    for (SqlNode node : matchRecognize.getPartitionList()) {
      SqlIdentifier identifier = (SqlIdentifier) node;
      identifier.validate(this, scope);
      RelDataType type = deriveType(scope, identifier);
      String name = identifier.names.get(1);
      typeBuilder.add(name, type);
    }

    // parse ORDER BY column
    for (SqlNode node : matchRecognize.getOrderList()) {
      node.validate(this, scope);
      SqlIdentifier identifier;
      if (node instanceof SqlBasicCall) {
        identifier = ((SqlBasicCall) node).operand(0);
      } else {
        identifier =
            requireNonNull((SqlIdentifier) node,
                () -> "order by field is null. All fields: "
                    + matchRecognize.getOrderList());
      }

      if (allRows) {
        RelDataType type = deriveType(scope, identifier);
        String name = identifier.names.get(1);
        if (!typeBuilder.nameExists(name)) {
          typeBuilder.add(name, type);
        }
      }
    }

    if (allRows) {
      final SqlValidatorNamespace sqlNs =
          getNamespaceOrThrow(matchRecognize.getTableRef());
      final RelDataType inputDataType = sqlNs.getRowType();
      for (RelDataTypeField fs : inputDataType.getFieldList()) {
        if (!typeBuilder.nameExists(fs.getName())) {
          typeBuilder.add(fs);
        }
      }
    }

    // retrieve pattern variables used in pattern and subset
    SqlNode pattern = matchRecognize.getPattern();
    PatternVarVisitor visitor = new PatternVarVisitor(scope);
    pattern.accept(visitor);

    SqlLiteral interval = matchRecognize.getInterval();
    if (interval != null) {
      interval.validate(this, scope);
      if (((SqlIntervalLiteral) interval).signum() < 0) {
        String intervalValue = interval.toValue();
        throw newValidationError(interval,
            RESOURCE.intervalMustBeNonNegative(
                intervalValue != null ? intervalValue : interval.toString()));
      }
      if (matchRecognize.getOrderList().isEmpty()) {
        throw newValidationError(interval,
          RESOURCE.cannotUseWithinWithoutOrderBy());
      }

      SqlNode firstOrderByColumn = matchRecognize.getOrderList().get(0);
      SqlIdentifier identifier;
      if (firstOrderByColumn instanceof SqlBasicCall) {
        identifier = ((SqlBasicCall) firstOrderByColumn).operand(0);
      } else {
        identifier =
            (SqlIdentifier) requireNonNull(firstOrderByColumn,
                "firstOrderByColumn");      }
      RelDataType firstOrderByColumnType = deriveType(scope, identifier);
      if (!SqlTypeUtil.isTimestamp(firstOrderByColumnType)) {
        throw newValidationError(interval,
          RESOURCE.firstColumnOfOrderByMustBeTimestamp());
      }

      SqlNode expand = expand(interval, scope);
      RelDataType type = deriveType(scope, expand);
      setValidatedNodeType(interval, type);
    }

    validateDefinitions(matchRecognize, scope);

    SqlNodeList subsets = matchRecognize.getSubsetList();
    if (!subsets.isEmpty()) {
      for (SqlNode node : subsets) {
        List<SqlNode> operands = ((SqlCall) node).getOperandList();
        String leftString = ((SqlIdentifier) operands.get(0)).getSimple();
        if (scope.getPatternVars().contains(leftString)) {
          throw newValidationError(operands.get(0),
              RESOURCE.patternVarAlreadyDefined(leftString));
        }
        scope.addPatternVar(leftString);
        for (SqlNode right : (SqlNodeList) operands.get(1)) {
          SqlIdentifier id = (SqlIdentifier) right;
          if (!scope.getPatternVars().contains(id.getSimple())) {
            throw newValidationError(id,
                RESOURCE.unknownPattern(id.getSimple()));
          }
          scope.addPatternVar(id.getSimple());
        }
      }
    }

    // validate AFTER ... SKIP TO
    final SqlNode skipTo = matchRecognize.getAfter();
    if (skipTo instanceof SqlCall) {
      final SqlCall skipToCall = (SqlCall) skipTo;
      final SqlIdentifier id = skipToCall.operand(0);
      if (!scope.getPatternVars().contains(id.getSimple())) {
        throw newValidationError(id,
            RESOURCE.unknownPattern(id.getSimple()));
      }
    }

    PairList<String, RelDataType> measureColumns =
        validateMeasure(matchRecognize, scope, allRows);
    measureColumns.forEach((name, type) -> {
      if (!typeBuilder.nameExists(name)) {
        typeBuilder.add(name, type);
      }
    });

    final RelDataType rowType;
    if (matchRecognize.getMeasureList().isEmpty()) {
      rowType = getNamespaceOrThrow(matchRecognize.getTableRef()).getRowType();
    } else {
      rowType = typeBuilder.build();
    }
    ns.setType(rowType);
  }

  private PairList<String, RelDataType> validateMeasure(SqlMatchRecognize mr,
      MatchRecognizeScope scope, boolean allRows) {
    final List<String> aliases = new ArrayList<>();
    final List<SqlNode> sqlNodes = new ArrayList<>();
    final SqlNodeList measures = mr.getMeasureList();
    final PairList<String, RelDataType> fields = PairList.of();

    for (SqlNode measure : measures) {
      assert measure instanceof SqlCall;
      final String alias = SqlValidatorUtil.alias(measure, aliases.size());
      aliases.add(alias);
      SqlNode expand = expand(measure, scope);
      expand = navigationInMeasure(expand, allRows);
      setOriginal(expand, measure);
      inferUnknownTypes(unknownType, scope, expand);
      final RelDataType type = deriveType(scope, expand);
      setValidatedNodeType(measure, type);

      fields.add(alias, type);
      sqlNodes.add(
          SqlStdOperatorTable.AS.createCall(SqlParserPos.ZERO, expand,
              new SqlIdentifier(alias, SqlParserPos.ZERO)));
    }
    SqlNodeList list = new SqlNodeList(sqlNodes, measures.getParserPosition());
    inferUnknownTypes(unknownType, scope, list);
    for (SqlNode node : list) {
      validateExpr(node, scope);
    }
    mr.setOperand(SqlMatchRecognize.OPERAND_MEASURES, list);
    return fields;
  }

  private SqlNode navigationInMeasure(SqlNode node, boolean allRows) {
    final Set<String> prefix = node.accept(new PatternValidator(true));
    Util.discard(prefix);
    final List<SqlNode> ops = ((SqlCall) node).getOperandList();

    final SqlOperator defaultOp =
        allRows ? SqlStdOperatorTable.RUNNING : SqlStdOperatorTable.FINAL;
    final SqlNode op0 = ops.get(0);
    if (!isRunningOrFinal(op0.getKind())
        || !allRows && op0.getKind() == SqlKind.RUNNING) {
      SqlNode newNode = defaultOp.createCall(SqlParserPos.ZERO, op0);
      node = SqlStdOperatorTable.AS.createCall(SqlParserPos.ZERO, newNode, ops.get(1));
    }

    node = new NavigationExpander().go(node);
    return node;
  }

  private void validateDefinitions(SqlMatchRecognize mr,
      MatchRecognizeScope scope) {
    final Set<String> aliases = catalogReader.nameMatcher().createSet();
    for (SqlNode item : mr.getPatternDefList()) {
      final String alias = alias(item);
      if (!aliases.add(alias)) {
        throw newValidationError(item,
            Static.RESOURCE.patternVarAlreadyDefined(alias));
      }
      scope.addPatternVar(alias);
    }

    final List<SqlNode> sqlNodes = new ArrayList<>();
    for (SqlNode item : mr.getPatternDefList()) {
      final String alias = alias(item);
      SqlNode expand = expand(item, scope);
      expand = navigationInDefine(expand, alias);
      setOriginal(expand, item);

      inferUnknownTypes(booleanType, scope, expand);
      expand.validate(this, scope);

      // Some extra work need required here.
      // In PREV, NEXT, FINAL and LAST, only one pattern variable is allowed.
      sqlNodes.add(
          SqlStdOperatorTable.AS.createCall(SqlParserPos.ZERO, expand,
              new SqlIdentifier(alias, SqlParserPos.ZERO)));

      final RelDataType type = deriveType(scope, expand);
      if (!SqlTypeUtil.inBooleanFamily(type)) {
        throw newValidationError(expand, RESOURCE.condMustBeBoolean("DEFINE"));
      }
      setValidatedNodeType(item, type);
    }

    SqlNodeList list =
        new SqlNodeList(sqlNodes, mr.getPatternDefList().getParserPosition());
    inferUnknownTypes(unknownType, scope, list);
    for (SqlNode node : list) {
      validateExpr(node, scope);
    }
    mr.setOperand(SqlMatchRecognize.OPERAND_PATTERN_DEFINES, list);
  }

  /** Returns the alias of a "expr AS alias" expression. */
  private static String alias(SqlNode item) {
    assert item instanceof SqlCall;
    assert item.getKind() == SqlKind.AS;
    final SqlIdentifier identifier = ((SqlCall) item).operand(1);
    return identifier.getSimple();
  }

  public void validatePivot(SqlPivot pivot) {
    final PivotScope scope = (PivotScope) getJoinScope(pivot);

    final PivotNamespace ns =
        getNamespaceOrThrow(pivot).unwrap(PivotNamespace.class);
    assert ns.rowType == null;

    // Given
    //   query PIVOT (agg1 AS a, agg2 AS b, ...
    //   FOR (axis1, ..., axisN)
    //   IN ((v11, ..., v1N) AS label1,
    //       (v21, ..., v2N) AS label2, ...))
    // the type is
    //   k1, ... kN, a_label1, b_label1, ..., a_label2, b_label2, ...
    // where k1, ... kN are columns that are not referenced as an argument to
    // an aggregate or as an axis.

    // Aggregates, e.g. "PIVOT (sum(x) AS sum_x, count(*) AS c)"
    final PairList<@Nullable String, RelDataType> aggNames = PairList.of();
    pivot.forEachAgg((alias, call) -> {
      call.validate(this, scope);
      final RelDataType type = deriveType(scope, call);
      aggNames.add(alias, type);
      if (!(call instanceof SqlCall)
          || !(((SqlCall) call).getOperator() instanceof SqlAggFunction)) {
        throw newValidationError(call, RESOURCE.pivotAggMalformed());
      }
    });

    // Axes, e.g. "FOR (JOB, DEPTNO)"
    final List<RelDataType> axisTypes = new ArrayList<>();
    final List<SqlIdentifier> axisIdentifiers = new ArrayList<>();
    for (SqlNode axis : pivot.axisList) {
      SqlIdentifier identifier = (SqlIdentifier) axis;
      identifier.validate(this, scope);
      final RelDataType type = deriveType(scope, identifier);
      axisTypes.add(type);
      axisIdentifiers.add(identifier);
    }

    // Columns that have been seen as arguments to aggregates or as axes
    // do not appear in the output.
    final Set<String> origColumnNames = pivot.usedColumnNames();
    // Bodo Change: Generate a name matcher for comparing names.
    final Set<String> columnNames = catalogReader.nameMatcher().createSet();
    columnNames.addAll(origColumnNames);
    final RelDataTypeFactory.Builder typeBuilder = typeFactory.builder();
    scope.getChild().getRowType().getFieldList().forEach(field -> {
      if (!columnNames.contains(field.getName())) {
        typeBuilder.add(field);
      }
    });

    // Values, e.g. "IN (('CLERK', 10) AS c10, ('MANAGER, 20) AS m20)"
    pivot.forEachNameValues((alias, nodeList) -> {
      if (nodeList.size() != axisTypes.size()) {
        throw newValidationError(nodeList,
            RESOURCE.pivotValueArityMismatch(nodeList.size(),
                axisTypes.size()));
      }
      final SqlOperandTypeChecker typeChecker =
          OperandTypes.COMPARABLE_UNORDERED_COMPARABLE_UNORDERED;
      Pair.forEach(axisIdentifiers, nodeList, (identifier, subNode) -> {
        subNode.validate(this, scope);
        typeChecker.checkOperandTypes(
            new SqlCallBinding(this, scope,
                SqlStdOperatorTable.EQUALS.createCall(
                    subNode.getParserPosition(), identifier, subNode)),
            true);
      });
      aggNames.forEach((aggAlias, aggType) ->
          typeBuilder.add(aggAlias == null ? alias : alias + "_" + aggAlias,
              aggType));
    });

    final RelDataType rowType = typeBuilder.build();
    ns.setType(rowType);
  }

  public void validateUnpivot(SqlUnpivot unpivot) {
    final UnpivotScope scope = (UnpivotScope) getJoinScope(unpivot);

    final UnpivotNamespace ns =
        getNamespaceOrThrow(unpivot).unwrap(UnpivotNamespace.class);
    assert ns.rowType == null;

    // Given
    //   query UNPIVOT ((measure1, ..., measureM)
    //   FOR (axis1, ..., axisN)
    //   IN ((c11, ..., c1M) AS (value11, ..., value1N),
    //       (c21, ..., c2M) AS (value21, ..., value2N), ...)
    // the type is
    //   k1, ... kN, axis1, ..., axisN, measure1, ..., measureM
    // where k1, ... kN are columns that are not referenced as an argument to
    // an aggregate or as an axis.

    // First, And make sure that each
    final int measureCount = unpivot.measureList.size();
    final int axisCount = unpivot.axisList.size();
    unpivot.forEachNameValues((nodeList, valueList) -> {
      // Make sure that each (ci1, ... ciM) list has the same arity as
      // (measure1, ..., measureM).
      if (nodeList.size() != measureCount) {
        throw newValidationError(nodeList,
            RESOURCE.unpivotValueArityMismatch(nodeList.size(),
                measureCount));
      }

      // Make sure that each (vi1, ... viN) list has the same arity as
      // (axis1, ..., axisN).
      if (valueList != null && valueList.size() != axisCount) {
        throw newValidationError(valueList,
            RESOURCE.unpivotValueArityMismatch(valueList.size(),
                axisCount));
      }

      // Make sure that each IN expression is a valid column from the input.
      nodeList.forEach(node -> deriveType(scope, node));
    });

    // What columns from the input are not referenced by a column in the IN
    // list?
    final SqlValidatorNamespace inputNs =
        requireNonNull(getNamespace(unpivot.query));
    final Set<String> unusedColumnNames =
        catalogReader.nameMatcher().createSet();
    unusedColumnNames.addAll(inputNs.getRowType().getFieldNames());
    unusedColumnNames.removeAll(unpivot.usedColumnNames());

    // What columns will be present in the output row type?
    final Set<String> columnNames = catalogReader.nameMatcher().createSet();
    columnNames.addAll(unusedColumnNames);

    // Gather the name and type of each measure.
    final PairList<String, RelDataType> measureNameTypes = PairList.of();
    forEach(unpivot.measureList, (measure, i) -> {
      final String measureName = ((SqlIdentifier) measure).getSimple();
      final List<RelDataType> types = new ArrayList<>();
      final List<SqlNode> nodes = new ArrayList<>();
      unpivot.forEachNameValues((nodeList, valueList) -> {
        final SqlNode alias = nodeList.get(i);
        nodes.add(alias);
        types.add(deriveType(scope, alias));
      });
      final RelDataType type0 = typeFactory.leastRestrictive(types);
      if (type0 == null) {
        throw newValidationError(nodes.get(0),
            RESOURCE.unpivotCannotDeriveMeasureType(measureName));
      }
      final RelDataType type =
          typeFactory.createTypeWithNullability(type0,
              unpivot.includeNulls || unpivot.measureList.size() > 1);
      setValidatedNodeType(measure, type);
      if (!columnNames.add(measureName)) {
        throw newValidationError(measure,
            RESOURCE.unpivotDuplicate(measureName));
      }
      measureNameTypes.add(measureName, type);
    });

    // Gather the name and type of each axis.
    // Consider
    //   FOR (job, deptno)
    //   IN (a AS ('CLERK', 10),
    //       b AS ('ANALYST', 20))
    // There are two axes, (job, deptno), and so each value list ('CLERK', 10),
    // ('ANALYST', 20) must have arity two.
    //
    // The type of 'job' is derived as the least restrictive type of the values
    // ('CLERK', 'ANALYST'), namely VARCHAR(7). The derived type of 'deptno' is
    // the type of values (10, 20), namely INTEGER.
    final PairList<String, RelDataType> axisNameTypes = PairList.of();
    forEach(unpivot.axisList, (axis, i) -> {
      final String axisName = ((SqlIdentifier) axis).getSimple();
      final List<RelDataType> types = new ArrayList<>();
      unpivot.forEachNameValues((aliasList, valueList) ->
          types.add(
              valueList == null
                  ? typeFactory.createSqlType(SqlTypeName.VARCHAR,
                        SqlUnpivot.aliasValue(aliasList).length())
                  : deriveType(scope, valueList.get(i))));
      final RelDataType type = typeFactory.leastRestrictive(types);
      if (type == null) {
        throw newValidationError(axis,
            RESOURCE.unpivotCannotDeriveAxisType(axisName));
      }
      setValidatedNodeType(axis, type);
      if (!columnNames.add(axisName)) {
        throw newValidationError(axis, RESOURCE.unpivotDuplicate(axisName));
      }
      axisNameTypes.add(axisName, type);
    });

    // Columns that have been seen as arguments to aggregates or as axes
    // do not appear in the output.
    final RelDataTypeFactory.Builder typeBuilder = typeFactory.builder();
    scope.getChild().getRowType().getFieldList().forEach(field -> {
      if (unusedColumnNames.contains(field.getName())) {
        typeBuilder.add(field);
      }
    });
    typeBuilder.addAll(axisNameTypes);
    typeBuilder.addAll(measureNameTypes);

    final RelDataType rowType = typeBuilder.build();
    ns.setType(rowType);
  }

  /** Checks that all pattern variables within a function are the same,
   * and canonizes expressions such as {@code PREV(B.price)} to
   * {@code LAST(B.price, 0)}. */
  private SqlNode navigationInDefine(SqlNode node, String alpha) {
    Set<String> prefix = node.accept(new PatternValidator(false));
    Util.discard(prefix);
    node = new NavigationExpander().go(node);
    node = new NavigationReplacer(alpha).go(node);
    return node;
  }

  @Override public void validateAggregateParams(SqlCall aggCall,
      @Nullable SqlNode filter, @Nullable SqlNodeList distinctList,
      @Nullable SqlNodeList orderList, SqlValidatorScope scope) {
    // For "agg(expr)", expr cannot itself contain aggregate function
    // invocations.  For example, "SUM(2 * MAX(x))" is illegal; when
    // we see it, we'll report the error for the SUM (not the MAX).
    // For more than one level of nesting, the error which results
    // depends on the traversal order for validation.
    //
    // For a windowed aggregate "agg(expr)", expr can contain an aggregate
    // function. For example,
    //   SELECT AVG(2 * MAX(x)) OVER (PARTITION BY y)
    //   FROM t
    //   GROUP BY y
    // is legal. Only one level of nesting is allowed since non-windowed
    // aggregates cannot nest aggregates.

    // Store nesting level of each aggregate. If an aggregate is found at an invalid
    // nesting level, throw an assert.
    final AggFinder a;
    if (inWindow) {
      a = overFinder;
    } else {
      a = aggOrOverFinder;
    }

    for (SqlNode param : aggCall.getOperandList()) {
      if (a.findAgg(param) != null) {
        throw newValidationError(aggCall, RESOURCE.nestedAggIllegal());
      }
    }
    if (filter != null) {
      if (a.findAgg(filter) != null) {
        throw newValidationError(filter, RESOURCE.aggregateInFilterIllegal());
      }
    }
    if (distinctList != null) {
      for (SqlNode param : distinctList) {
        if (a.findAgg(param) != null) {
          throw newValidationError(aggCall,
                  RESOURCE.aggregateInWithinDistinctIllegal());
        }
      }
    }
    if (orderList != null) {
      for (SqlNode param : orderList) {
        if (a.findAgg(param) != null) {
          throw newValidationError(aggCall,
                  RESOURCE.aggregateInWithinGroupIllegal());
        }
      }
    }

    final SqlAggFunction op = (SqlAggFunction) aggCall.getOperator();
    switch (op.requiresGroupOrder()) {
      case MANDATORY:
        if (orderList == null || orderList.isEmpty()) {
          throw newValidationError(aggCall,
                  RESOURCE.aggregateMissingWithinGroupClause(op.getName()));
        }
        break;
      case OPTIONAL:
        break;
      case IGNORED:
        // rewrite the order list to empty
        if (orderList != null) {
          orderList.clear();
        }
        break;
      case FORBIDDEN:
        if (orderList != null && !orderList.isEmpty()) {
          throw newValidationError(aggCall,
                  RESOURCE.withinGroupClauseIllegalInAggregate(op.getName()));
        }
        break;
      default:
        throw new AssertionError(op);
    }

    // Because there are two forms of the PERCENTILE_CONT/PERCENTILE_DISC functions,
    // they are distinguished by their operand count and then validated accordingly.
    // For example, the standard single operand form requires group order while the
    // 2-operand form allows for null treatment and requires an OVER() clause.
    if (op.isPercentile()) {
      switch (aggCall.operandCount()) {
        case 1:
          assert op.requiresGroupOrder() == Optionality.MANDATORY;
          assert orderList != null;
          // Validate that percentile function have a single ORDER BY expression
          if (orderList.size() != 1) {
            throw newValidationError(orderList,
                    RESOURCE.orderByRequiresOneKey(op.getName()));
          }
          // Validate that the ORDER BY field is of NUMERIC type
          SqlNode node = requireNonNull(orderList.get(0));
          final RelDataType type = deriveType(scope, node);
          final @Nullable SqlTypeFamily family = type.getSqlTypeName().getFamily();
          if (family == null
                  || family.allowableDifferenceTypes().isEmpty()) {
            throw newValidationError(orderList,
                    RESOURCE.unsupportedTypeInOrderBy(
                            type.getSqlTypeName().getName(),
                            op.getName()));
          }
          break;
        case 2:
          assert op.allowsNullTreatment();
          assert op.requiresOver();
          assert op.requiresGroupOrder() == Optionality.FORBIDDEN;
          break;
        default:
          throw newValidationError(aggCall, RESOURCE.percentileFunctionsArgumentLimit());
      }
    }
  }

  @Override public void validateCall(
      SqlCall call,
      SqlValidatorScope scope) {
    final SqlOperator operator = call.getOperator();
    if ((call.operandCount() == 0)
        && (operator.getSyntax() == SqlSyntax.FUNCTION_ID)
        && !call.isExpanded()
        && !this.config.conformance().allowNiladicParentheses()) {
      // For example, "LOCALTIME()" is illegal. (It should be
      // "LOCALTIME", which would have been handled as a
      // SqlIdentifier.)
      throw handleUnresolvedFunction(call, operator,
          ImmutableList.of(), null);
    }

    SqlValidatorScope operandScope = scope.getOperandScope(call);

    if (operator instanceof SqlFunction
        && ((SqlFunction) operator).getFunctionType()
            == SqlFunctionCategory.MATCH_RECOGNIZE
        && !(operandScope instanceof MatchRecognizeScope)) {
      throw newValidationError(call,
          Static.RESOURCE.functionMatchRecognizeOnly(call.toString()));
    }
    // Delegate validation to the operator.
    operator.validateCall(call, this, scope, operandScope);
  }

  /**
   * Validates that a particular feature is enabled. By default, all features
   * are enabled; subclasses may override this method to be more
   * discriminating.
   *
   * @param feature feature being used, represented as a resource instance
   * @param context parser position context for error reporting, or null if
   */
  protected void validateFeature(
      Feature feature,
      SqlParserPos context) {
    // By default, do nothing except to verify that the resource
    // represents a real feature definition.
    assert feature.getProperties().get("FeatureDefinition") != null;
  }

  @Override public SqlLiteral resolveLiteral(SqlLiteral literal) {
    switch (literal.getTypeName()) {
    case UNKNOWN:
      final SqlUnknownLiteral unknownLiteral = (SqlUnknownLiteral) literal;
      final SqlIdentifier identifier =
          new SqlIdentifier(unknownLiteral.tag, SqlParserPos.ZERO);
      final @Nullable RelDataType type = catalogReader.getNamedType(identifier);
      final SqlTypeName typeName;
      if (type != null) {
        typeName = type.getSqlTypeName();
      } else {
        typeName = SqlTypeName.lookup(unknownLiteral.tag);
      }
      return unknownLiteral.resolve(typeName);

    default:
      return literal;
    }
  }

  public SqlNode expandSelectExpr(SqlNode expr,
      SelectScope scope, SqlSelect select, Integer selectItemIdx) {
    final Expander expander = new SelectExpander(this, scope, select, selectItemIdx);
    final SqlNode newExpr = expander.go(expr);
    if (expr != newExpr) {
      setOriginal(newExpr, expr);
    }
    return newExpr;
  }

  @Override public SqlNode expand(SqlNode expr, SqlValidatorScope scope) {
    final Expander expander = new Expander(this, scope);
    SqlNode newExpr = expander.go(expr);
    if (expr != newExpr) {
      setOriginal(newExpr, expr);
    }
    return newExpr;
  }

  /** Expands an expression in a GROUP BY, HAVING or QUALIFY clause. */
  private SqlNode extendedExpand(SqlNode expr,
      SqlValidatorScope scope, SqlSelect select, Clause clause) {
    final Expander expander =
        new ExtendedExpander(this, scope, select, expr, clause);
    SqlNode newExpr = expander.go(expr);
    if (expr != newExpr) {
      setOriginal(newExpr, expr);
    }
    return newExpr;
  }

  public SqlNode extendedExpandGroupBy(SqlNode expr,
      SqlValidatorScope scope, SqlSelect select) {
    return extendedExpand(expr, scope, select, Clause.GROUP_BY);
  }

  /**
   * Expands the given expression, expanding any identifiers that correspond to aliases
   * in the select list. This is used to allow for propagating aliases from the select list
   * to later expressions in the select list, or into the select clauses. IE:
   * SELECT (A + B * C - D) as x, x + 1 FROM table1 WHERE x &gt; 1 JOIN table2 ON x = table2.foo
   *
   * @param expr The expression to be expanded.
   * @param scope The scope of the current expression,
   *              generally equivalent to the scope of the select.
   * @param select The select query from which to locate aliases.
   * @return returns the expanded expression
   */
  public SqlNode expandWithAlias(SqlNode expr,
      SqlValidatorScope scope, SqlSelect select, Clause clause) {
    final Expander expander = new ExtendedExpander(this, scope, select, select, clause);
    SqlNode newExpr = expr.accept(expander);
    requireNonNull(newExpr, "newExpr");
    if (expr != newExpr) {
      setOriginal(newExpr, expr);
    }
    return newExpr;
  }

  @Override public boolean isSystemField(RelDataTypeField field) {
    return false;
  }

  @Override public List<@Nullable List<String>> getFieldOrigins(SqlNode sqlQuery) {
    if (sqlQuery instanceof SqlExplain) {
      return emptyList();
    }
    final RelDataType rowType = getValidatedNodeType(sqlQuery);
    final int fieldCount = rowType.getFieldCount();
    if (!sqlQuery.isA(SqlKind.QUERY)) {
      return Collections.nCopies(fieldCount, null);
    }
    final List<@Nullable List<String>> list = new ArrayList<>();
    for (int i = 0; i < fieldCount; i++) {
      list.add(getFieldOrigin(sqlQuery, i));
    }
    return ImmutableNullableList.copyOf(list);
  }

  private @Nullable List<String> getFieldOrigin(SqlNode sqlQuery, int i) {
    if (sqlQuery instanceof SqlSelect) {
      SqlSelect sqlSelect = (SqlSelect) sqlQuery;
      final SelectScope scope = getRawSelectScopeNonNull(sqlSelect);
      final List<SqlNode> selectList =
          requireNonNull(scope.getExpandedSelectList(),
              () -> "expandedSelectList for " + scope);
      final SqlNode selectItem = stripAs(selectList.get(i));
      if (selectItem instanceof SqlIdentifier) {
        final SqlQualified qualified =
            scope.fullyQualify((SqlIdentifier) selectItem);
        SqlValidatorNamespace namespace =
            requireNonNull(qualified.namespace,
                () -> "namespace for " + qualified);
        if (namespace.isWrapperFor(AliasNamespace.class)) {
          AliasNamespace aliasNs = namespace.unwrap(AliasNamespace.class);
          SqlNode aliased = requireNonNull(aliasNs.getNode(), () ->
              "sqlNode for aliasNs " + aliasNs);
          namespace = getNamespaceOrThrow(stripAs(aliased));
        }
        final SqlValidatorTable table = namespace.getTable();
        if (table == null) {
          return null;
        }
        final List<String> origin =
            new ArrayList<>(table.getQualifiedName());
        for (String name : qualified.suffix()) {
          if (namespace.isWrapperFor(UnnestNamespace.class)) {
            // If identifier is drawn from a repeated subrecord via unnest, add name of array field
            UnnestNamespace unnestNamespace = namespace.unwrap(UnnestNamespace.class);
            final SqlQualified columnUnnestedFrom = unnestNamespace.getColumnUnnestedFrom(name);
            if (columnUnnestedFrom != null) {
              origin.addAll(columnUnnestedFrom.suffix());
            }
          }
          namespace = namespace.lookupChild(name);

          if (namespace == null) {
            return null;
          }
          origin.add(name);
        }
        return origin;
      }
      return null;
    } else if (sqlQuery instanceof SqlOrderBy) {
      return getFieldOrigin(((SqlOrderBy) sqlQuery).query, i);
    } else {
      return null;
    }
  }

  @Override public RelDataType getParameterRowType(SqlNode sqlQuery) {
    // NOTE: We assume that bind variables occur in depth-first tree
    // traversal in the same order that they occurred in the SQL text.
    final List<RelDataType> types = new ArrayList<>();
    // NOTE: but parameters on fetch/offset would be counted twice
    // as they are counted in the SqlOrderBy call and the inner SqlSelect call
    final Set<SqlNode> alreadyVisited = new HashSet<>();
    sqlQuery.accept(
        new SqlShuttle() {

          @Override public SqlNode visit(SqlDynamicParam param) {
            if (alreadyVisited.add(param)) {
              RelDataType type = getValidatedNodeType(param);
              types.add(type);
            }
            return param;
          }
        });
    // TODO: Figure out if we need to make changes for Named Parameters
    return typeFactory.createStructType(
        types,
        new AbstractList<String>() {
          @Override public String get(int index) {
            return "?" + index;
          }

          @Override public int size() {
            return types.size();
          }
        });
  }

  private static boolean isPhysicalNavigation(SqlKind kind) {
    return kind == SqlKind.PREV || kind == SqlKind.NEXT;
  }

  private static boolean isLogicalNavigation(SqlKind kind) {
    return kind == SqlKind.FIRST || kind == SqlKind.LAST;
  }

  private static boolean isAggregation(SqlKind kind) {
    return kind == SqlKind.SUM || kind == SqlKind.SUM0
        || kind == SqlKind.AVG || kind == SqlKind.COUNT
        || kind == SqlKind.MAX || kind == SqlKind.MIN;
  }

  private static boolean isRunningOrFinal(SqlKind kind) {
    return kind == SqlKind.RUNNING || kind == SqlKind.FINAL;
  }

  private static boolean isSingleVarRequired(SqlKind kind) {
    return isPhysicalNavigation(kind)
        || isLogicalNavigation(kind)
        || isAggregation(kind);
  }

  //~ Inner Classes ----------------------------------------------------------

  /**
   * Common base class for DML statement namespaces. To handle
   * both TableIdentifiers and Regular Identifiers we perform all
   * operations on a stored namespace object.
   */
  public static class DmlNamespace implements SqlValidatorNamespace {


    private final SqlValidatorNamespace ns;

    protected DmlNamespace(SqlValidatorImpl validator, SqlNode id,
        SqlNode enclosingNode, SqlValidatorScope parentScope) {
      switch (id.getKind()) {
      case TABLE_IDENTIFIER_WITH_ID:
      case TABLE_REF_WITH_ID:
        ns = new TableIdentifierWithIDNamespace(validator, id, enclosingNode, parentScope);
        break;
      default:
        ns = new IdentifierNamespace(validator, id, enclosingNode, parentScope);
      }
    }

    public @Nullable SqlNodeList getExtendList() {
      if (ns instanceof TableIdentifierWithIDNamespace) {
        return ((TableIdentifierWithIDNamespace) ns).extendList;
      } else {
        return ((IdentifierNamespace) ns).extendList;
      }
    }

    /**
     * Returns the validator.
     *
     * @return validator
     */
    @Override public SqlValidator getValidator() {
      return ns.getValidator();
    }

    /**
     * Returns the underlying table, or null if there is none.
     */
    @Override public @Nullable SqlValidatorTable getTable() {
      return ns.getTable();
    }

    /**
     * Returns the row type of this namespace, which comprises a list of names
     * and types of the output columns. If the scope's type has not yet been
     * derived, derives it.
     *
     * @return Row type of this namespace, never null, always a struct
     */
    @Override public RelDataType getRowType() {
      return ns.getRowType();
    }

    /**
     * Returns the type of this namespace.
     *
     * @return Row type converted to struct
     */
    @Override public RelDataType getType() {
      return ns.getType();
    }

    /**
     * Sets the type of this namespace.
     *
     * <p>Allows the type for the namespace to be explicitly set, but usually is
     * called during {@link #validate(RelDataType)}.</p>
     *
     * <p>Implicitly also sets the row type. If the type is not a struct, then
     * the row type is the type wrapped as a struct with a single column,
     * otherwise the type and row type are the same.</p>
     *
     * @param type Type to set.
     */
    @Override public void setType(RelDataType type) {
      ns.setType(type);
    }

    /**
     * Returns the row type of this namespace, sans any system columns.
     *
     * @return Row type sans system columns
     */
    @Override public RelDataType getRowTypeSansSystemColumns() {
      return ns.getRowTypeSansSystemColumns();
    }

    /**
     * Validates this namespace.
     *
     * <p>If the scope has already been validated, does nothing.</p>
     *
     * <p>Please call {@link SqlValidatorImpl#validateNamespace} rather than
     * calling this method directly.</p>
     *
     * @param targetRowType Desired row type, must not be null, may be the data
     *                      type 'unknown'.
     */
    @Override public void validate(RelDataType targetRowType) {
      ns.validate(targetRowType);
    }

    /**
     * Returns the parse tree node at the root of this namespace.
     *
     * @return parse tree node; null for {@link TableNamespace}
     */
    @Override public @Nullable SqlNode getNode() {
      return ns.getNode();
    }

    /**
     * Returns the parse tree node that at is at the root of this namespace and
     * includes all decorations. If there are no decorations, returns the same
     * as {@link #getNode()}.
     */
    @Override public @Nullable SqlNode getEnclosingNode() {
      return ns.getEnclosingNode();
    }

    /**
     * Looks up a child namespace of a given name.
     *
     * <p>For example, in the query <code>select e.name from emps as e</code>,
     * <code>e</code> is an {@link IdentifierNamespace} which has a child <code>
     * name</code> which is a {@link FieldNamespace}.
     *
     * @param name Name of namespace
     * @return Namespace
     */
    @Override public @Nullable SqlValidatorNamespace lookupChild(String name) {
      return ns.lookupChild(name);
    }

    /**
     * Returns whether this namespace has a field of a given name.
     *
     * @param name Field name
     * @return Whether field exists
     */
    @Override public boolean fieldExists(String name) {
      return ns.fieldExists(name);
    }

    /**
     * Returns a field of a given name, or null.
     *
     * @param name Field name
     * @return Field, or null
     */
    @Override
    public @Nullable RelDataTypeField field(String name) {
      return ns.field(name);
    }

    /**
     * Returns a list of expressions which are monotonic in this namespace. For
     * example, if the namespace represents a relation ordered by a column
     * called "TIMESTAMP", then the list would contain a
     * {@link SqlIdentifier} called "TIMESTAMP".
     */
    @Override public List<Pair<SqlNode, SqlMonotonicity>> getMonotonicExprs() {
      return ns.getMonotonicExprs();
    }

    /**
     * Returns whether and how a given column is sorted.
     *
     * @param columnName Name of column to check Monotonicity
     */
    @Override public SqlMonotonicity getMonotonicity(String columnName) {
      return ns.getMonotonicity(columnName);
    }

    @Deprecated
    @Override public void makeNullable() {
      ns.makeNullable();
    }

    /**
     * Returns this namespace, or a wrapped namespace, cast to a particular
     * class.
     *
     * @param clazz Desired type
     * @return This namespace cast to desired type
     * @throws ClassCastException if no such interface is available
     */
    @Override public <T extends Object> T unwrap(Class<T> clazz) {
      return clazz.cast(this);
    }

    /**
     * Returns whether this namespace implements a given interface, or wraps a
     * class which does.
     *
     * @param clazz Interface
     * @return Whether namespace implements given interface
     */
    @Override public boolean isWrapperFor(Class<?> clazz) {
      return clazz.isInstance(this);
    }

    /**
     * If this namespace resolves to another namespace, returns that namespace,
     * following links to the end of the chain.
     *
     * <p>A {@code WITH}) clause defines table names that resolve to queries
     * (the body of the with-item). An {@link IdentifierNamespace} typically
     * resolves to a {@link TableNamespace}.</p>
     *
     * <p>You must not call this method before {@link #validate(RelDataType)} has
     * completed.</p>
     */
    @Override public SqlValidatorNamespace resolve() {
      return ns.resolve();
    }

    /**
     * Returns whether this namespace is capable of giving results of the desired
     * modality. {@code true} means streaming, {@code false} means relational.
     *
     * @param modality Modality
     */
    @Override public boolean supportsModality(SqlModality modality) {
      return ns.supportsModality(modality);
    }
  }

  /**
   * Namespace for an INSERT statement.
   */
  private static class InsertNamespace extends DmlNamespace {
    private final SqlInsert node;

    InsertNamespace(SqlValidatorImpl validator, SqlInsert node,
        SqlNode enclosingNode, SqlValidatorScope parentScope) {
      super(validator, node.getTargetTable(), enclosingNode, parentScope);
      this.node = requireNonNull(node, "node");
    }

    @Override public @Nullable SqlNode getNode() {
      return node;
    }
  }

  /**
   * Namespace for an UPDATE statement.
   */
  private static class UpdateNamespace extends DmlNamespace {
    private final SqlUpdate node;

    UpdateNamespace(SqlValidatorImpl validator, SqlUpdate node,
        SqlNode enclosingNode, SqlValidatorScope parentScope) {
      super(validator, node.getTargetTable(), enclosingNode, parentScope);
      this.node = requireNonNull(node, "node");
    }

    @Override public @Nullable SqlNode getNode() {
      return node;
    }
  }

  /**
   * Namespace for a DELETE statement.
   */
  private static class DeleteNamespace extends DmlNamespace {
    private final SqlDelete node;

    DeleteNamespace(SqlValidatorImpl validator, SqlDelete node,
        SqlNode enclosingNode, SqlValidatorScope parentScope) {
      super(validator, node.getTargetTable(), enclosingNode, parentScope);
      this.node = requireNonNull(node, "node");
    }

    @Override public @Nullable SqlNode getNode() {
      return node;
    }
  }


  /**
   * Namespace for DDL statements (Data Definition Language, such as create [Or replace] Table).
   * Currently, defers everything to the child query/table's namespace. This will likely need to be
   * extended in the future in order to handle the case where the output column type are
   * explicitly defined in the query.
   *
   * Note: this does not extend DmlNamespace because DmlNamespace
   * requires there to be an existing target table, which is not necessarily true for DDL
   * statements.
   */
  public static class DdlNamespace implements SqlValidatorNamespace {
    private final SqlCreate node;
    private final SqlValidatorNamespace childNamespace;

    DdlNamespace(
        SqlValidatorImpl validator,
        SqlCreate node,
        SqlNode enclosingNode, SqlValidatorScope parentScope, SqlNode childNode) {
      requireNonNull(childNode, "childNode");
      requireNonNull(node, "node");
      this.node = node;

      if (childNode instanceof SqlIdentifier) {
        SqlIdentifier id = (SqlIdentifier) childNode;
        switch (id.getKind()) {
        case TABLE_IDENTIFIER_WITH_ID:
        case TABLE_REF_WITH_ID:
          childNamespace = new TableIdentifierWithIDNamespace(validator, id, enclosingNode,
              parentScope);
          break;
        default:
          childNamespace = new IdentifierNamespace(validator, id, enclosingNode, parentScope);
        }
      } else {
        SqlValidatorNamespace childNs = validator.getNamespaceOrThrow(childNode);
        childNamespace = childNs;
      }
    }


    /**
     * Returns the validator.
     *
     * @return validator
     */
    @Override public SqlValidator getValidator() {
      return childNamespace.getValidator();
    }

    /**
     * Returns the underlying table, or null if there is none.
     */
    @Override public @Nullable SqlValidatorTable getTable() {
      return childNamespace.getTable();
    }

    /**
     * Returns the row type of this namespace, which comprises a list of names
     * and types of the output columns. If the scope's type has not yet been
     * derived, derives it.
     *
     * @return Row type of this namespace, never null, always a struct
     */
    @Override public RelDataType getRowType() {
      return childNamespace.getRowType();
    }

    /**
     * Returns the type of this namespace.
     *
     * @return Row type converted to struct
     */
    @Override public RelDataType getType() {
      return childNamespace.getType();
    }

    /**
     * Sets the type of this namespace.
     *
     * <p>Allows the type for the namespace to be explicitly set, but usually is
     * called during {@link #validate(RelDataType)}.</p>
     *
     * <p>Implicitly also sets the row type. If the type is not a struct, then
     * the row type is the type wrapped as a struct with a single column,
     * otherwise the type and row type are the same.</p>
     *
     * @param type the type to set
     */
    @Override public void setType(final RelDataType type) {
      childNamespace.setType(type);
    }

    /**
     * Returns the row type of this namespace, sans any system columns.
     *
     * @return Row type sans system columns
     */
    @Override public RelDataType getRowTypeSansSystemColumns() {
      return childNamespace.getRowTypeSansSystemColumns();
    }

    /**
     * Validates this namespace.
     *
     * <p>If the scope has already been validated, does nothing.</p>
     *
     * <p>Please call {@link SqlValidatorImpl#validateNamespace} rather than
     * calling this method directly.</p>
     *
     * @param targetRowType Desired row type, must not be null, may be the data
     *                      type 'unknown'.
     */
    @Override public void validate(final RelDataType targetRowType) {
      childNamespace.validate(targetRowType);
    }

    @Override public @Nullable SqlNode getNode() {
      return node;
    }

    /**
     * Returns the parse tree node that at is at the root of this namespace and
     * includes all decorations. If there are no decorations, returns the same
     * as {@link #getNode()}.
     */
    @Override public @Nullable SqlNode getEnclosingNode() {
      return node;
    }

    /**
     * Looks up a child namespace of a given name.
     *
     * <p>For example, in the query <code>select e.name from emps as e</code>,
     * <code>e</code> is an {@link IdentifierNamespace} which has a child <code>
     * name</code> which is a {@link FieldNamespace}.
     *
     * @param name Name of namespace
     * @return Namespace
     */
    @Override public @Nullable SqlValidatorNamespace lookupChild(final String name) {
      return this.childNamespace.lookupChild(name);
    }

    /**
     * Returns whether this namespace has a field of a given name.
     *
     * @param name Field name
     * @return Whether field exists
     */
    @Override public boolean fieldExists(final String name) {
      return this.childNamespace.fieldExists(name);
    }

    /**
     * Returns a field of a given name, or null.
     *
     * @param name Field name
     * @return Field, or null
     */
    @Override
    public @Nullable RelDataTypeField field(String name) {
      return this.childNamespace.field(name);
    }

    /**
     * Returns a list of expressions which are monotonic in this namespace. For
     * example, if the namespace represents a relation ordered by a column
     * called "TIMESTAMP", then the list would contain a
     * {@link SqlIdentifier} called "TIMESTAMP".
     */
    @Override public List<Pair<SqlNode, SqlMonotonicity>> getMonotonicExprs() {
      return this.childNamespace.getMonotonicExprs();
    }

    /**
     * Returns whether and how a given column is sorted.
     *
     * @param columnName the column to check
     */
    @Override public SqlMonotonicity getMonotonicity(final String columnName) {
      return this.childNamespace.getMonotonicity(columnName);
    }

    @Override @Deprecated
    public void makeNullable() {
      this.childNamespace.makeNullable();
    }

    /**
     * Returns this namespace, or a wrapped namespace, cast to a particular
     * class.
     *
     * @param clazz Desired type
     * @return This namespace cast to desired type
     * @throws ClassCastException if no such interface is available
     */
    @Override public <T extends Object> T unwrap(final Class<T> clazz) {
      return clazz.cast(this);
    }

    /**
     * Returns whether this namespace implements a given interface, or wraps a
     * class which does.
     *
     * @param clazz Interface
     * @return Whether namespace implements given interface
     */
    @Override public boolean isWrapperFor(final Class<?> clazz) {
      return clazz.isInstance(this);
    }

    /**
     * If this namespace resolves to another namespace, returns that namespace,
     * following links to the end of the chain.
     *
     * <p>A {@code WITH}) clause defines table names that resolve to queries
     * (the body of the with-item). An {@link IdentifierNamespace} typically
     * resolves to a {@link TableNamespace}.</p>
     *
     * <p>You must not call this method before {@link #validate(RelDataType)} has
     * completed.</p>
     */
    @Override public SqlValidatorNamespace resolve() {
      return this;
    }

    /**
     * Returns whether this namespace is capable of giving results of the desired
     * modality. {@code true} means streaming, {@code false} means relational.
     *
     * @param modality Modality
     */
    @Override public boolean supportsModality(final SqlModality modality) {
      return childNamespace.supportsModality(modality);
    }
  }

  /**
   * Namespace for a MERGE statement.
   */
  private static class MergeNamespace extends DmlNamespace {
    private final SqlMerge node;

    MergeNamespace(SqlValidatorImpl validator, SqlMerge node,
        SqlNode enclosingNode, SqlValidatorScope parentScope) {
      super(validator, node.getTargetTable(), enclosingNode, parentScope);
      this.node = requireNonNull(node, "node");
    }

    @Override public @Nullable SqlNode getNode() {
      return node;
    }
  }

  /** Visitor that retrieves pattern variables defined. */
  private static class PatternVarVisitor implements SqlVisitor<Void> {
    private final MatchRecognizeScope scope;
    PatternVarVisitor(MatchRecognizeScope scope) {
      this.scope = scope;
    }

    @Override public Void visit(SqlLiteral literal) {
      return null;
    }

    @Override public Void visit(SqlCall call) {
      for (int i = 0; i < call.getOperandList().size(); i++) {
        call.getOperandList().get(i).accept(this);
      }
      return null;
    }

    @Override public Void visit(SqlNodeList nodeList) {
      throw Util.needToImplement(nodeList);
    }

    @Override public Void visit(SqlIdentifier id) {
      checkArgument(id.isSimple());
      scope.addPatternVar(id.getSimple());
      return null;
    }

    @Override public Void visit(SqlTableIdentifierWithID id) {
      checkArgument(id.isSimple());
      scope.addPatternVar(id.getSimple());
      return null;
    }


    @Override public Void visit(SqlDataTypeSpec type) {
      throw Util.needToImplement(type);
    }

    @Override public Void visit(SqlDynamicParam param) {
      throw Util.needToImplement(param);
    }

    @Override public Void visit(SqlIntervalQualifier intervalQualifier) {
      throw Util.needToImplement(intervalQualifier);
    }
  }

  /**
   * Visitor which derives the type of a given {@link SqlNode}.
   *
   * <p>Each method must return the derived type. This visitor is basically a
   * single-use dispatcher; the visit is never recursive.
   */
  private class DeriveTypeVisitor implements SqlVisitor<RelDataType> {
    private final SqlValidatorScope scope;

    DeriveTypeVisitor(SqlValidatorScope scope) {
      this.scope = scope;
    }

    @Override public RelDataType visit(SqlLiteral literal) {
      return resolveLiteral(literal).createSqlType(typeFactory);
    }

    @Override public RelDataType visit(SqlCall call) {
      final SqlOperator operator = call.getOperator();
      return operator.deriveType(SqlValidatorImpl.this, scope, call);
    }

    @Override public RelDataType visit(SqlNodeList nodeList) {
      // Operand is of a type that we can't derive a type for. If the
      // operand is of a peculiar type, such as a SqlNodeList, then you
      // should override the operator's validateCall() method so that it
      // doesn't try to validate that operand as an expression.
      throw Util.needToImplement(nodeList);
    }

    @Override public RelDataType visit(SqlIdentifier id) {
      // First check for builtin functions which don't have parentheses,
      // like "LOCALTIME".
      final SqlCall call = makeNullaryCall(id);
      if (call != null) {
        return call.getOperator().validateOperands(
            SqlValidatorImpl.this,
            scope,
            call);
      }

      RelDataType type = null;
      if (!(scope instanceof EmptyScope)) {
        id = scope.fullyQualify(id).identifier;
      }

      // Resolve the longest prefix of id that we can
      int i;
      for (i = id.names.size() - 1; i > 0; i--) {
        // REVIEW jvs 9-June-2005: The name resolution rules used
        // here are supposed to match SQL:2003 Part 2 Section 6.6
        // (identifier chain), but we don't currently have enough
        // information to get everything right.  In particular,
        // routine parameters are currently looked up via resolve;
        // we could do a better job if they were looked up via
        // resolveColumn.

        final SqlNameMatcher nameMatcher = catalogReader.nameMatcher();
        final SqlValidatorScope.ResolvedImpl resolved =
            new SqlValidatorScope.ResolvedImpl();
        scope.resolve(id.names.subList(0, i), nameMatcher, false, resolved);
        if (resolved.count() == 1) {
          // There's a namespace with the name we seek.
          final SqlValidatorScope.Resolve resolve = resolved.only();
          type = resolve.rowType();
          for (SqlValidatorScope.Step p : Util.skip(resolve.path.steps())) {
            type = type.getFieldList().get(p.i).getType();
          }
          break;
        }
      }

      // Give precedence to namespace found, unless there
      // are no more identifier components.
      if (type == null || id.names.size() == 1) {
        // See if there's a column with the name we seek in
        // precisely one of the namespaces in this scope.
        RelDataType colType = scope.resolveColumn(id.names.get(0), id);
        if (colType != null) {
          type = colType;
        }
        ++i;
      }

      if (type == null) {
        final SqlIdentifier last = id.getComponent(i - 1, i);
        throw newValidationError(last,
            RESOURCE.unknownIdentifier(last.toString()));
      }

      // Resolve rest of identifier
      for (; i < id.names.size(); i++) {
        String name = id.names.get(i);
        final RelDataTypeField field;
        if (name.isEmpty()) {
          // The wildcard "*" is represented as an empty name. It never
          // resolves to a field.
          name = "*";
          field = null;
        } else {
          final SqlNameMatcher nameMatcher = catalogReader.nameMatcher();
          field = nameMatcher.field(type, name);
        }
        if (field == null) {
          throw newValidationError(id.getComponent(i),
              RESOURCE.unknownField(name));
        }
        type = field.getType();
      }
      type =
          SqlTypeUtil.addCharsetAndCollation(
              type,
              getTypeFactory());
      return type;
    }

    @Override public RelDataType visit(SqlTableIdentifierWithID id) {
      try {
        requireNonCall(id);
      } catch (ValidationException e) {
        throw new RuntimeException(e);
      }

      RelDataType type = null;
      if (!(scope instanceof EmptyScope)) {
        id = scope.fullyQualify(id).identifier;
      }

      // Resolve the longest prefix of id that we can
      int i;
      for (i = id.names.size() - 1; i > 0; i--) {
        // REVIEW jvs 9-June-2005: The name resolution rules used
        // here are supposed to match SQL:2003 Part 2 Section 6.6
        // (identifier chain), but we don't currently have enough
        // information to get everything right.  In particular,
        // routine parameters are currently looked up via resolve;
        // we could do a better job if they were looked up via
        // resolveColumn.

        final SqlNameMatcher nameMatcher = catalogReader.nameMatcher();
        final SqlValidatorScope.ResolvedImpl resolved =
            new SqlValidatorScope.ResolvedImpl();
        scope.resolve(id.names.subList(0, i), nameMatcher, false, resolved);
        if (resolved.count() == 1) {
          // There's a namespace with the name we seek.
          final SqlValidatorScope.Resolve resolve = resolved.only();
          type = resolve.rowType();
          for (SqlValidatorScope.Step p : Util.skip(resolve.path.steps())) {
            type = type.getFieldList().get(p.i).getType();
          }
          break;
        }
      }

      // Give precedence to namespace found, unless there
      // are no more identifier components.
      if (type == null || id.names.size() == 1) {
        // See if there's a column with the name we seek in
        // precisely one of the namespaces in this scope.
        RelDataType colType = scope.resolveColumn(id.names.get(0), id);
        if (colType != null) {
          type = colType;
        }
        ++i;
      }

      if (type == null) {
        final SqlTableIdentifierWithID last = id.getComponent(i - 1, i);
        throw newValidationError(last,
            RESOURCE.unknownIdentifier(last.toString()));
      }

      // Resolve rest of identifier
      for (; i < id.names.size(); i++) {
        String name = id.names.get(i);
        final RelDataTypeField field;
        final SqlNameMatcher nameMatcher = catalogReader.nameMatcher();
        field = nameMatcher.field(type, name);
        if (field == null) {
          throw newValidationError(id.getComponent(i),
              RESOURCE.unknownField(name));
        }
        type = field.getType();
      }
      type =
          SqlTypeUtil.addCharsetAndCollation(
              type,
              getTypeFactory());
      return type;
    }

    @Override public RelDataType visit(SqlDataTypeSpec dataType) {
      // Q. How can a data type have a type?
      // A. When it appears in an expression. (Say as the 2nd arg to the
      //    CAST operator.)
      validateDataType(dataType);
      return dataType.deriveType(SqlValidatorImpl.this);
    }

    @Override public RelDataType visit(SqlDynamicParam param) {
      return unknownType;
    }

    @Override public RelDataType visit(SqlIntervalQualifier intervalQualifier) {
      return typeFactory.createSqlIntervalType(intervalQualifier);
    }
  }

  /**
   * Converts an expression into canonical form by fully-qualifying any
   * identifiers.
   */
  private static class Expander extends SqlScopedShuttle {
    protected final SqlValidatorImpl validator;

    Expander(SqlValidatorImpl validator, SqlValidatorScope scope) {
      super(scope);
      this.validator = validator;
    }

    public SqlNode go(SqlNode root) {
      return requireNonNull(root.accept(this),
          () -> this + " returned null for " + root);
    }

    @Override public @Nullable SqlNode visit(SqlIdentifier id) {
      // First check for builtin functions which don't have
      // parentheses, like "LOCALTIME".
      final SqlCall call = validator.makeNullaryCall(id);
      if (call != null) {
        return call.accept(this);
      }
      final SqlIdentifier fqId = getScope().fullyQualify(id).identifier;
      SqlNode expandedExpr = expandDynamicStar(id, fqId);
      validator.setOriginal(expandedExpr, id);
      return expandedExpr;
    }

    @Override public @Nullable SqlNode visit(SqlTableIdentifierWithID id) {
      try {
        validator.requireNonCall(id);
      } catch (ValidationException e) {
        throw new RuntimeException(e);
      }
      final SqlTableIdentifierWithID fqId = getScope().fullyQualify(id).identifier;
      validator.setOriginal(fqId, id);
      return fqId;
    }

    @Override public @Nullable SqlNode visit(SqlLiteral literal) {
      return validator.resolveLiteral(literal);
    }

    @Override protected SqlNode visitScoped(SqlCall call) {
      switch (call.getKind()) {
      case SCALAR_QUERY:
      case CURRENT_VALUE:
      case NEXT_VALUE:
      case WITH:
      case LAMBDA:
        return call;
      default:
        break;
      }
      if (validator.isStarCall(call)) {
        call = validator.starExpansion(call, getScope());
      }
      // Only visits arguments which are expressions. We don't want to
      // qualify non-expressions such as 'x' in 'empno * 5 AS x'.
      CallCopyingArgHandler argHandler =
          new CallCopyingArgHandler(call, false);
      call.getOperator().acceptCall(this, call, true, argHandler);
      final SqlNode result = argHandler.result();
      validator.setOriginal(result, call);
      return result;
    }

    protected SqlNode expandDynamicStar(SqlIdentifier id, SqlIdentifier fqId) {
      if (DynamicRecordType.isDynamicStarColName(Util.last(fqId.names))
          && !DynamicRecordType.isDynamicStarColName(Util.last(id.names))) {
        // Convert a column ref into ITEM(*, 'col_name')
        // for a dynamic star field in dynTable's rowType.
        return new SqlBasicCall(
            SqlStdOperatorTable.ITEM,
            ImmutableList.of(fqId,
                SqlLiteral.createCharString(Util.last(id.names),
                    id.getParserPosition())),
            id.getParserPosition());
      }
      return fqId;
    }
  }

  /**
   * Helper function called in ExtendedAliasExpander (used to expand identifiers in the Select list,
   * and associated aliases).
   * return information on the aliases to "name", in the select list (see return for more info)
   *
   *
   * @param name The alias to search for in the select list
   * @param nameMatcher The nameMatcher used to check if the aliases match.
   * @param select The select to search.
   * @param maxIdx The maximum idx into the select list to search.
   * @return A nested pair containing the number of occurrences of the alias, the last expression
   * in the select list to which the alias evaluates (if there are multiple)
   * (null if no such expressions exist), and the index
   * into the select list that said expression occurs at (if there are multiple)
   * (-1 if no such expression exists).
   *
   * It is left to the caller function to properly handle the case that there are
   * no aliases, or in the case that there are multiple aliases
   *
   * The information is arranged (numOccurrences, (expr, expr_idx))
   */
  protected static Pair<Integer, Pair<@Nullable SqlNode, Integer>> findAliasesInSelect(
      String name, SqlNameMatcher nameMatcher, SqlSelect select, Integer maxIdx) {
    SqlNode expr = null;
    int numOccurrencesOfAlias = 0;
    int idxOfOccurrence = -1;

    for (int i = 0; i < maxIdx; i++) {
      SqlNode s = select.getSelectList().get(i);
      final @Nullable String alias = SqlValidatorUtil.alias(s);

      if (alias != null && nameMatcher.matches(alias, name)) {
        // Note: In the event that there are multiple matches,
        // we return the final matching expr, so we may clobber
        // idxOfOccurrence and expr. For example:
        // Select expr1 as x, expr2 as x from ...
        // should return <2, <expr2, 1>>
        // As noted in the docstring, we leave it to the calling function to throw the error
        expr = s;
        idxOfOccurrence = i;
        numOccurrencesOfAlias++;
      }
    }
    return new Pair<>(numOccurrencesOfAlias, new Pair<>(expr, idxOfOccurrence));
  }


  /**
   * Shuttle which walks over an expression in the ORDER BY clause, replacing
   * usages of aliases with the underlying expression.
   */
  class OrderExpressionExpander extends SqlScopedShuttle {
    private final List<String> aliasList;
    private final SqlSelect select;
    private final SqlNode root;

    OrderExpressionExpander(SqlSelect select, SqlNode root) {
      super(getOrderScope(select));
      this.select = select;
      this.root = root;
      this.aliasList = getNamespaceOrThrow(select).getRowType().getFieldNames();
    }

    public SqlNode go() {
      return requireNonNull(root.accept(this),
          () -> "OrderExpressionExpander returned null for " + root);
    }

    @Override public @Nullable SqlNode visit(SqlLiteral literal) {
      // Ordinal markers, e.g. 'select a, b from t order by 2'.
      // Only recognize them if they are the whole expression,
      // and if the dialect permits.
      if (literal == root && config.conformance().isSortByOrdinal()) {
        switch (literal.getTypeName()) {
        case DECIMAL:
        case DOUBLE:
          final int intValue = literal.intValue(false);
          if (intValue >= 0) {
            if (intValue < 1 || intValue > aliasList.size()) {
              throw newValidationError(
                  literal, RESOURCE.orderByOrdinalOutOfRange());
            }

            // SQL ordinals are 1-based, but Sort's are 0-based
            int ordinal = intValue - 1;
            return nthSelectItem(ordinal, literal.getParserPosition());
          }
          break;
        default:
          break;
        }
      }

      return super.visit(literal);
    }

    /**
     * Returns the <code>ordinal</code>th item in the select list.
     */
    private SqlNode nthSelectItem(int ordinal, final SqlParserPos pos) {
      // TODO: Don't expand the list every time. Maybe keep an expanded
      // version of each expression -- select lists and identifiers -- in
      // the validator.

      SqlNodeList expandedSelectList =
          expandStar(
              SqlNonNullableAccessors.getSelectList(select),
              select,
              false);
      SqlNode expr = expandedSelectList.get(ordinal);
      expr = stripAs(expr);
      if (expr instanceof SqlIdentifier) {
        expr = getScope().fullyQualify((SqlIdentifier) expr).identifier;
      }

      // Create a copy of the expression with the position of the order
      // item.
      return expr.clone(pos);
    }

    @Override public SqlNode visit(SqlIdentifier id) {
      // Aliases, e.g. 'select a as x, b from t order by x'.
      if (id.isSimple()
          && config.conformance().isSortByAlias()) {
        String alias = id.getSimple();
        final SqlValidatorNamespace selectNs = getNamespaceOrThrow(select);
        final RelDataType rowType =
            selectNs.getRowTypeSansSystemColumns();
        final SqlNameMatcher nameMatcher = catalogReader.nameMatcher();
        RelDataTypeField field = nameMatcher.field(rowType, alias);
        if (field != null) {
          return nthSelectItem(
              field.getIndex(),
              id.getParserPosition());
        }
      }

      // No match. Return identifier unchanged.
      return getScope().fullyQualify(id).identifier;
    }

    @Override protected @Nullable SqlNode visitScoped(SqlCall call) {
      // Don't attempt to expand sub-queries. We haven't implemented
      // these yet.
      if (call instanceof SqlSelect) {
        return call;
      }
      return super.visitScoped(call);
    }
  }

  /**
   * Converts an expression into canonical form by fully-qualifying any
   * identifiers. For common columns in USING, it will be converted to
   * COALESCE(A.col, B.col) AS col.
   */
  static class SelectExpander extends ExtendedExpander {

    SelectExpander(SqlValidatorImpl validator, SelectScope scope,
        SqlSelect select, Integer selectItemIdx) {
      super(validator, scope, select, select, Clause.SELECT,
          selectItemIdx);
    }

    @Override public @Nullable SqlNode visit(SqlIdentifier id) {
      final SqlNode node =
          expandCommonColumn(select, id, (SelectScope) getScope(), validator);
      if (node != id) {
        return node;
      } else {
        return super.visit(id);
      }
    }
  }

  /**
   * For many SQL dialects, it is valid to use aliases from the select statement
   * in later expression in the select list, or in any number of different clauses.
   * This shuttle which walks over a given expression, replacing
   * usages of aliases or ordinals with the underlying expression.
   */
  static class ExtendedExpander extends Expander {
    final SqlSelect select;
    final SqlNode root;
    // Bodo Change: Remove final because we reassign.
    Clause clause;
    // This argument instructs the expander to only check elements in the select list
    // up to but not including this index.
    Integer maxNumCols;

    ExtendedExpander(SqlValidatorImpl validator, SqlValidatorScope scope,
        SqlSelect select, SqlNode root, Clause clause) {
      super(validator, scope);
      this.select = select;
      this.root = root;
      this.clause = clause;
      this.maxNumCols = select.getSelectList().size();
    }

    /**
     * Creates an ExtendedExpander.
     *
     * @param validator The validator class.
     * @param scope The scope of the current expression.
     * @param select The select expression from which to derive aliases.
     * @param root The root expression. Generally equivalent to the select for most applications.
     * @param clause Enum which indicates the type of expression to be expanded.
     * @param maxNumCols This argument instructs the expander to only check elements in the
     *                   select list up to but not including this index when searching for
     *                   aliases.
     */
    ExtendedExpander(SqlValidatorImpl validator, SqlValidatorScope scope,
        SqlSelect select, SqlNode root,
        Clause clause, Integer maxNumCols) {
      super(validator, scope);
      this.select = select;
      this.root = root;
      this.clause = clause;
      this.maxNumCols = maxNumCols;
    }

    @Override public @Nullable SqlNode visit(SqlIdentifier id) {

      if (!id.isSimple()) {
        return super.visit(id);
      }

      //Handle expanding columns in the USING clause if needed
      final boolean replaceAliases = clause.shouldReplaceAliases(validator.config);
      final SelectScope scope = validator.getRawSelectScopeNonNull(select);
      final SqlNode node = expandCommonColumn(select, id, scope, validator);
      if (!replaceAliases) {
        if (node != id) {
          return node;
        }
        return super.visit(id);
      } else {
        if (node != id) {
          return node.accept(this);
        }
      }

      try {
        // If we can resolve this in the regular scope, we should
        // since we prioritize all id's that could reference columns in the "from/on" clauses
        // before we look for aliases within the select list itself.
        SqlNode sqlNode = super.visit(id);
        return sqlNode;
      } catch (Exception e) {

        String name = id.getSimple();

        final SqlNameMatcher nameMatcher =
            validator.catalogReader.nameMatcher();
        Pair<Integer, Pair<@Nullable SqlNode, Integer>> occurrenceNumExprAndExprIdx =
            findAliasesInSelect(
                name, nameMatcher,
                this.select, this.maxNumCols);

        int n = occurrenceNumExprAndExprIdx.left;
        if (n == 0) {
            // If we can't find any aliases in the select list, just throw
            // the original error.
            throw e;
        } else if (n > 1) {
            // More than one column has this alias.
            throw validator.newValidationError(id,
                RESOURCE.columnAmbiguous(name));
        }
        Iterable<SqlCall> allAggList = validator.aggFinder.findAll(ImmutableList.of(root));
        for (SqlCall agg : allAggList) {
          if (clause == Clause.HAVING && containsIdentifier(agg, id)) {
            return super.visit(id);
          }
        }

        SqlNode expr = occurrenceNumExprAndExprIdx.right.left;
        // Returned expr is only null if the number of occurrences of the alias is 0
        requireNonNull(expr, "expr");
        expr = stripAs(expr);

        int idxOfOccurence = occurrenceNumExprAndExprIdx.right.right;

        // set the maxIdx to the idx where we found the alias occurrence
        // and attempt to recursively expand the alias.
        // This is needed to resolve recursive aliasing in the select list,
        // IE: SELECT A as x, x as y, y as z ...

        // NOTE: I don't believe it is necessary to restore the original maxIdx/expression type,
        // since we always go from expanding a clause --> the select list, and maxNumCols is
        // never decreases when resolving nested aliases.
        //
        // However, restoring the original values doesn't hurt,
        // in case we re-use this infrastructure at some point in the future
        int origMaxIdx = this.maxNumCols;
        Clause origClause = this.clause;
        this.maxNumCols = idxOfOccurence;
        this.clause = Clause.SELECT;
        SqlNode expandedExpr = expr.accept(this);
        this.maxNumCols = origMaxIdx;
        this.clause = origClause;

        requireNonNull(expandedExpr, "expandedExpr");

        if (expandedExpr instanceof SqlIdentifier) {
            SqlIdentifier sid = (SqlIdentifier) expandedExpr;
            final SqlIdentifier fqId = getScope().fullyQualify(sid).identifier;
            expandedExpr = expandDynamicStar(sid, fqId);
        }

        validator.setOriginal(expandedExpr, id);

        return expandedExpr;
      }
    }

    @Override public @Nullable SqlNode visit(SqlLiteral literal) {
      if (clause != Clause.GROUP_BY
          || !validator.config().conformance().isGroupByOrdinal()) {
        return super.visit(literal);
      }
      boolean isOrdinalLiteral = literal == root;
      switch (root.getKind()) {
      case GROUPING_SETS:
      case ROLLUP:
      case CUBE:
        if (root instanceof SqlBasicCall) {
          List<SqlNode> operandList = ((SqlBasicCall) root).getOperandList();
          for (SqlNode node : operandList) {
            if (node.equals(literal)) {
              isOrdinalLiteral = true;
              break;
            }
          }
        }
        break;
      default:
        break;
      }
      if (isOrdinalLiteral) {
        switch (literal.getTypeName()) {
        case DECIMAL:
        case DOUBLE:
          final int intValue = literal.intValue(false);
          if (intValue >= 0) {
            if (intValue < 1 || intValue > SqlNonNullableAccessors.getSelectList(select).size()) {
              throw validator.newValidationError(literal,
                  RESOURCE.orderByOrdinalOutOfRange());
            }

            // SQL ordinals are 1-based, but Sort's are 0-based
            int ordinal = intValue - 1;
            SqlNode expr = stripAs(SqlNonNullableAccessors.getSelectList(select)
                .get(ordinal));
            if (!(expr instanceof SqlLiteral)) {
              expr = expr.accept(this);
            }
            return expr;
          }
          break;
        default:
          break;
        }
      }

      return super.visit(literal);
    }

    /**
     * Returns whether a given node contains a {@link SqlIdentifier}.
     *
     * @param sqlNode a SqlNode
     * @param target a SqlIdentifier
     */
    private boolean containsIdentifier(SqlNode sqlNode, SqlIdentifier target) {
      try {
        SqlVisitor<Void> visitor =
            new SqlBasicVisitor<Void>() {
              @Override public Void visit(SqlIdentifier identifier) {
                if (identifier.equalsDeep(target, Litmus.IGNORE)) {
                  throw new Util.FoundOne(target);
                }
                return super.visit(identifier);
              }
            };
        sqlNode.accept(visitor);
        return false;
      } catch (Util.FoundOne e) {
        Util.swallow(e, null);
        return true;
      }
    }
  }

  /** Information about an identifier in a particular scope. */
  protected static class IdInfo {
    public final SqlValidatorScope scope;
    public final SqlIdentifier id;

    public IdInfo(SqlValidatorScope scope, SqlIdentifier id) {
      this.scope = scope;
      this.id = id;
    }
  }

  /**
   * Utility object used to maintain information about the parameters in a
   * function call.
   */
  protected static class FunctionParamInfo {
    /**
     * Maps a cursor (based on its position relative to other cursor
     * parameters within a function call) to the SELECT associated with the
     * cursor.
     */
    public final Map<Integer, SqlSelect> cursorPosToSelectMap;

    /**
     * Maps a column list parameter to the parent cursor parameter it
     * references. The parameters are id'd by their names.
     */
    public final Map<String, String> columnListParamToParentCursorMap;

    public FunctionParamInfo() {
      cursorPosToSelectMap = new HashMap<>();
      columnListParamToParentCursorMap = new HashMap<>();
    }
  }

  /**
   * Modify the nodes in navigation function
   * such as FIRST, LAST, PREV AND NEXT.
   */
  private static class NavigationModifier extends SqlShuttle {
    public SqlNode go(SqlNode node) {
      return requireNonNull(node.accept(this),
          () -> "NavigationModifier returned for " + node);
    }
  }

  /**
   * Shuttle that expands navigation expressions in a MATCH_RECOGNIZE clause.
   *
   * <p>Examples:
   *
   * <ul>
   * <li>{@code PREV(A.price + A.amount)} &rarr;
   * {@code PREV(A.price) + PREV(A.amount)}
   *
   * <li>{@code FIRST(A.price * 2)} &rarr; {@code FIRST(A.PRICE) * 2}
   * </ul>
   */
  private static class NavigationExpander extends NavigationModifier {
    final @Nullable SqlOperator op;
    final @Nullable SqlNode offset;

    NavigationExpander() {
      this(null, null);
    }

    NavigationExpander(@Nullable SqlOperator operator, @Nullable SqlNode offset) {
      this.offset = offset;
      this.op = operator;
    }

    @Override public @Nullable SqlNode visit(SqlCall call) {
      SqlKind kind = call.getKind();
      List<SqlNode> operands = call.getOperandList();
      List<@Nullable SqlNode> newOperands = new ArrayList<>();

      if (call.getFunctionQuantifier() != null
          && call.getFunctionQuantifier().getValue() == SqlSelectKeyword.DISTINCT) {
        final SqlParserPos pos = call.getParserPosition();
        throw SqlUtil.newContextException(pos,
            Static.RESOURCE.functionQuantifierNotAllowed(call.toString()));
      }

      if (isLogicalNavigation(kind) || isPhysicalNavigation(kind)) {
        SqlNode inner = operands.get(0);
        SqlNode offset = operands.get(1);

        // merge two straight prev/next, update offset
        if (isPhysicalNavigation(kind)) {
          SqlKind innerKind = inner.getKind();
          if (isPhysicalNavigation(innerKind)) {
            List<SqlNode> innerOperands = ((SqlCall) inner).getOperandList();
            SqlNode innerOffset = innerOperands.get(1);
            SqlOperator newOperator = innerKind == kind
                ? SqlStdOperatorTable.PLUS : SqlStdOperatorTable.MINUS;
            offset =
              newOperator.createCall(SqlParserPos.ZERO, offset, innerOffset);
            inner =
              call.getOperator().createCall(SqlParserPos.ZERO,
                  innerOperands.get(0), offset);
          }
        }
        SqlNode newInnerNode =
            inner.accept(new NavigationExpander(call.getOperator(), offset));
        if (op != null) {
          newInnerNode =
              op.createCall(SqlParserPos.ZERO, newInnerNode, this.offset);
        }
        return newInnerNode;
      }

      if (operands.size() > 0) {
        for (SqlNode node : operands) {
          if (node != null) {
            SqlNode newNode = node.accept(new NavigationExpander());
            if (op != null) {
              newNode = op.createCall(SqlParserPos.ZERO, newNode, offset);
            }
            newOperands.add(newNode);
          } else {
            newOperands.add(null);
          }
        }
        return call.getOperator().createCall(SqlParserPos.ZERO, newOperands);
      } else {
        if (op == null) {
          return call;
        } else {
          return op.createCall(SqlParserPos.ZERO, call, offset);
        }
      }
    }

    @Override public SqlNode visit(SqlIdentifier id) {
      if (op == null) {
        return id;
      } else {
        return op.createCall(SqlParserPos.ZERO, id, offset);
      }
    }
  }

  /**
   * Shuttle that replaces {@code A as A.price > PREV(B.price)} with
   * {@code PREV(A.price, 0) > LAST(B.price, 0)}.
   *
   * <p>Replacing {@code A.price} with {@code PREV(A.price, 0)} makes the
   * implementation of
   * {@link RexVisitor#visitPatternFieldRef(RexPatternFieldRef)} more unified.
   * Otherwise, it's difficult to implement this method. If it returns the
   * specified field, then the navigation such as {@code PREV(A.price, 1)}
   * becomes impossible; if not, then comparisons such as
   * {@code A.price > PREV(A.price, 1)} become meaningless.
   */
  private static class NavigationReplacer extends NavigationModifier {
    private final String alpha;

    NavigationReplacer(String alpha) {
      this.alpha = alpha;
    }

    @Override public @Nullable SqlNode visit(SqlCall call) {
      SqlKind kind = call.getKind();
      if (isLogicalNavigation(kind)
          || isAggregation(kind)
          || isRunningOrFinal(kind)) {
        return call;
      }

      switch (kind) {
      case PREV:
        final List<SqlNode> operands = call.getOperandList();
        if (operands.get(0) instanceof SqlIdentifier) {
          String name = ((SqlIdentifier) operands.get(0)).names.get(0);
          return name.equals(alpha) ? call
              : SqlStdOperatorTable.LAST.createCall(SqlParserPos.ZERO, operands);
        }
        break;
      default:
        break;
      }
      return super.visit(call);
    }

    @Override public SqlNode visit(SqlIdentifier id) {
      if (id.isSimple()) {
        return id;
      }
      SqlOperator operator = id.names.get(0).equals(alpha)
          ? SqlStdOperatorTable.PREV : SqlStdOperatorTable.LAST;

      return operator.createCall(SqlParserPos.ZERO, id,
        SqlLiteral.createExactNumeric("0", SqlParserPos.ZERO));
    }
  }

  /** Validates that within one navigation function, the pattern var is the
   * same. */
  private class PatternValidator extends SqlBasicVisitor<@Nullable Set<String>> {
    private final boolean isMeasure;
    int firstLastCount;
    int prevNextCount;
    int aggregateCount;

    PatternValidator(boolean isMeasure) {
      this(isMeasure, 0, 0, 0);
    }

    PatternValidator(boolean isMeasure, int firstLastCount, int prevNextCount,
        int aggregateCount) {
      this.isMeasure = isMeasure;
      this.firstLastCount = firstLastCount;
      this.prevNextCount = prevNextCount;
      this.aggregateCount = aggregateCount;
    }

    @Override public Set<String> visit(SqlCall call) {
      boolean isSingle = false;
      Set<String> vars = new HashSet<>();
      SqlKind kind = call.getKind();
      List<SqlNode> operands = call.getOperandList();

      if (isSingleVarRequired(kind)) {
        isSingle = true;
        if (isPhysicalNavigation(kind)) {
          if (isMeasure) {
            throw newValidationError(call,
                Static.RESOURCE.patternPrevFunctionInMeasure(call.toString()));
          }
          if (firstLastCount != 0) {
            throw newValidationError(call,
                Static.RESOURCE.patternPrevFunctionOrder(call.toString()));
          }
          prevNextCount++;
        } else if (isLogicalNavigation(kind)) {
          if (firstLastCount != 0) {
            throw newValidationError(call,
                Static.RESOURCE.patternPrevFunctionOrder(call.toString()));
          }
          firstLastCount++;
        } else if (isAggregation(kind)) {
          // cannot apply aggregation in PREV/NEXT, FIRST/LAST
          if (firstLastCount != 0 || prevNextCount != 0) {
            throw newValidationError(call,
                Static.RESOURCE.patternAggregationInNavigation(call.toString()));
          }
          if (kind == SqlKind.COUNT && call.getOperandList().size() > 1) {
            throw newValidationError(call,
                Static.RESOURCE.patternCountFunctionArg());
          }
          aggregateCount++;
        }
      }

      if (isRunningOrFinal(kind) && !isMeasure) {
        throw newValidationError(call,
            Static.RESOURCE.patternRunningFunctionInDefine(call.toString()));
      }

      for (SqlNode node : operands) {
        if (node != null) {
          vars.addAll(
              requireNonNull(
                  node.accept(
                      new PatternValidator(isMeasure, firstLastCount, prevNextCount,
                          aggregateCount)),
                  () -> "node.accept(PatternValidator) for node " + node));
        }
      }

      if (isSingle) {
        switch (kind) {
        case COUNT:
          if (vars.size() > 1) {
            throw newValidationError(call,
                Static.RESOURCE.patternCountFunctionArg());
          }
          break;
        default:
          if (operands.isEmpty()
              || !(operands.get(0) instanceof SqlCall)
              || ((SqlCall) operands.get(0)).getOperator() != SqlStdOperatorTable.CLASSIFIER) {
            if (vars.isEmpty()) {
              throw newValidationError(call,
                  Static.RESOURCE.patternFunctionNullCheck(call.toString()));
            }
            if (vars.size() != 1) {
              throw newValidationError(call,
                  Static.RESOURCE.patternFunctionVariableCheck(call.toString()));
            }
          }
          break;
        }
      }
      return vars;
    }

    @Override public Set<String> visit(SqlIdentifier identifier) {
      boolean check = prevNextCount > 0 || firstLastCount > 0 || aggregateCount > 0;
      Set<String> vars = new HashSet<>();
      if (identifier.names.size() > 1 && check) {
        vars.add(identifier.names.get(0));
      }
      return vars;
    }

    @Override public Set<String> visit(SqlLiteral literal) {
      return ImmutableSet.of();
    }

    @Override public Set<String> visit(SqlIntervalQualifier qualifier) {
      return ImmutableSet.of();
    }

    @Override public Set<String> visit(SqlDataTypeSpec type) {
      return ImmutableSet.of();
    }

    @Override public Set<String> visit(SqlDynamicParam param) {
      return ImmutableSet.of();
    }
  }

  /** Permutation of fields in NATURAL JOIN or USING. */
  private class Permute {
    final List<ImmutableIntList> sources;
    final RelDataType rowType;
    final boolean trivial;
    final int offset;

    Permute(SqlNode from, int offset) {
      this.offset = offset;
      switch (from.getKind()) {
      case JOIN:
        final SqlJoin join = (SqlJoin) from;
        final Permute left = new Permute(join.getLeft(), offset);
        final int fieldCount =
            getValidatedNodeType(join.getLeft()).getFieldList().size();
        final Permute right =
            new Permute(join.getRight(), offset + fieldCount);
        final List<String> names = usingNames(join);
        final List<ImmutableIntList> sources = new ArrayList<>();
        final Set<ImmutableIntList> sourceSet = new HashSet<>();
        final RelDataTypeFactory.Builder b = typeFactory.builder();
        if (names != null) {
          for (String name : names) {
            final RelDataTypeField f = left.field(name);
            final ImmutableIntList source = left.sources.get(f.getIndex());
            sourceSet.add(source);
            final RelDataTypeField f2 = right.field(name);
            final ImmutableIntList source2 = right.sources.get(f2.getIndex());
            sourceSet.add(source2);
            sources.add(source.appendAll(source2));
            final boolean nullable =
                (f.getType().isNullable()
                    || join.getJoinType().generatesNullsOnLeft())
                && (f2.getType().isNullable()
                    || join.getJoinType().generatesNullsOnRight());
            b.add(f).nullable(nullable);
          }
        }
        for (RelDataTypeField f : left.rowType.getFieldList()) {
          final ImmutableIntList source = left.sources.get(f.getIndex());
          if (sourceSet.add(source)) {
            sources.add(source);
            b.add(f);
          }
        }
        for (RelDataTypeField f : right.rowType.getFieldList()) {
          final ImmutableIntList source = right.sources.get(f.getIndex());
          if (sourceSet.add(source)) {
            sources.add(source);
            b.add(f);
          }
        }
        rowType = b.build();
        this.sources = ImmutableList.copyOf(sources);
        this.trivial = left.trivial
            && right.trivial
            && (names == null || names.isEmpty());
        break;

      default:
        rowType = getValidatedNodeType(from);
        this.sources =
            Functions.generate(rowType.getFieldCount(),
                i -> ImmutableIntList.of(offset + i));
        this.trivial = true;
      }
    }

    private RelDataTypeField field(String name) {
      RelDataTypeField field = catalogReader.nameMatcher().field(rowType, name);
      if (field == null) {
        throw new AssertionError("field " + name + " was not found in "
            + rowType);
      }
      return field;
    }

    /** Moves fields according to the permutation. */
    void permute(List<SqlNode> selectItems,
        PairList<String, RelDataType> fields) {
      if (trivial) {
        return;
      }

      final List<SqlNode> oldSelectItems = ImmutableList.copyOf(selectItems);
      selectItems.clear();
      selectItems.addAll(oldSelectItems.subList(0, offset));
      final PairList<String, RelDataType> oldFields = fields.immutable();
      fields.clear();
      fields.addAll(oldFields.subList(0, offset));
      for (ImmutableIntList source : sources) {
        final int p0 = source.get(0);
        Map.Entry<String, RelDataType> field = oldFields.get(p0);
        final String name = field.getKey();
        RelDataType type = field.getValue();
        SqlNode selectItem = oldSelectItems.get(p0);
        for (int p1 : Util.skip(source)) {
          final Map.Entry<String, RelDataType> field1 = oldFields.get(p1);
          final SqlNode selectItem1 = oldSelectItems.get(p1);
          final RelDataType type1 = field1.getValue();
          // output is nullable only if both inputs are
          final boolean nullable = type.isNullable() && type1.isNullable();
          RelDataType currentType = type;
          final RelDataType type2 =
              requireNonNull(
                  SqlTypeUtil.leastRestrictiveForComparison(typeFactory, type,
                      type1),
                  () -> "leastRestrictiveForComparison for types " + currentType
                      + " and " + type1);
          selectItem =
              SqlStdOperatorTable.AS.createCall(SqlParserPos.ZERO,
                  SqlStdOperatorTable.COALESCE.createCall(SqlParserPos.ZERO,
                      maybeCast(selectItem, type, type2),
                      maybeCast(selectItem1, type1, type2)),
                  new SqlIdentifier(name, SqlParserPos.ZERO));
          type = typeFactory.createTypeWithNullability(type2, nullable);
        }
        fields.add(name, type);
        selectItems.add(selectItem);
      }
    }
  }

  //~ Enums ------------------------------------------------------------------

  /**
   * Validation status.
   */
  public enum Status {
    /**
     * Validation has not started for this scope.
     */
    UNVALIDATED,

    /**
     * Validation is in progress for this scope.
     */
    IN_PROGRESS,

    /**
     * Validation has completed (perhaps unsuccessfully).
     */
    VALID
  }

  /** Allows {@link #clauseScopes} to have multiple values per SELECT. */
  private enum Clause {
    WHERE,
    GROUP_BY,
    SELECT,
    MEASURE,
    ORDER,
    CURSOR,
    HAVING,
    QUALIFY;

    /**
     * Determines if the extender should replace aliases with expanded values.
     * For example:
     *
     * <blockquote><pre>{@code
     * SELECT a + a as twoA
     * GROUP BY twoA
     * }</pre></blockquote>
     *
     * <p>turns into
     *
     * <blockquote><pre>{@code
     * SELECT a + a as twoA
     * GROUP BY a + a
     * }</pre></blockquote>
     *
     * <p>This is determined both by the clause and the config.
     *
     * @param config The configuration
     * @return Whether we should replace the alias with its expanded value
     */
    boolean shouldReplaceAliases(Config config) {
      switch (this) {
      case GROUP_BY:
        return config.conformance().isGroupByAlias();

      case HAVING:
        return config.conformance().isHavingAlias();

      case QUALIFY:
      // Bodo Change: We always want to expand Select or WHERE
      case SELECT:
      case WHERE:
        return true;

      default:
        throw Util.unexpected(this);
      }
    }
  }
}
