# Calcite 1.38 → 1.42 Upgrade Plan

This document tracks the incremental upgrade of our Calcite fork from 1.38.0 to 1.42.0. It is the source of truth for the upgrade effort — each PR should start by re-reading this doc for fresh context.

## Scope summary

Current state (`BodoSQL/calcite_sql/bodosql-calcite-application/pom.xml:27-28`):
- `calcite-core` / `calcite-testkit` **1.38.0**
- `avatica` **1.23.0**

Fork inventory:
- **99** Java + 5 Kotlin `.kt` files under `src/main/java/org/apache/calcite/` (104 total)
- **5** test files under `src/test/java/org/apache/calcite/` (overridden Calcite tests)
- **1** FreeMarker template: `src/main/codegen/templates/Parser.jj` (10,105 lines)
- **339** `<exclude>` entries in the shade plugin (`pom.xml:386+`)
- **153** `// Bodo Change:` markers across 53 main + 18 test/rules files
- **Problematic files** (per the upgrade process doc): `Parser.jj`, `SqlValidatorImpl.java` (9,480 lines), `SqlToRelConverter.java` (7,827 lines), `RexSimplify.java` (3,268 lines)

## Diff magnitude per release

| Release | Files | Notes |
|---|---|---|
| 1.38 → 1.39 | ~100+ | Adds VARIANT/UUID, checked arithmetic, DPhyp join ordering. **Avatica 1.25 → 1.26**. `Parser.jj`, `SqlValidatorImpl`, `SqlToRelConverter`, `RexSimplify`, `StandardConvertletTable`, `SqlStdOperatorTable`, `RelDecorrelator`, `VolcanoPlanner` all modified. |
| 1.39 → 1.40 | ~100 | New set-op→join rules, `ExpandDisjunctionHelper` in `RexUtil`, lateral column alias support in `SqlValidatorImpl`/`Parser.jj`. Avatica unchanged. `SqlToRelConverter`/`RexSimplify`/`StandardConvertletTable`/`SqlStdOperatorTable` **not touched**. |
| 1.40 → 1.41 | ~293 | **Largest jump.** Unsigned types, `Combine` rel node, functional-dependency metadata, `PlanTooComplexError`, `CAST_NOT_NULL` internal op, `isNonStrictGroupBy` conformance, `hasEmptyGroup()` deprecation. `RexSimplify` heavily modified (+310/-37). `Parser.jj` +56/-4. |
| 1.41 → 1.42 | ~350+ | **General decorrelation algorithm** (`RelDecorrelator` rewritten — CALCITE-7031), `ConditionalCorrelate` rel node, `SELECT * EXCLUDE/REPLACE`, `:` path operator, `RelShuttle` new visit overloads (CALCITE-7511 — implementors must add methods). **Avatica 1.27 → 1.28**, Kotlin 1.9.22 → 2.3.20, Hadoop 2.10 → 3.4.3, commons-lang3 dropped. |

## Recommended strategy: four incremental PRs

Do **not** jump straight to 1.42. Each release is a checkpoint where the build must compile and tests must pass before advancing. This bounds the blast radius of any single merge and makes `git blame` / review tractable.

### PR1 — 1.38 → 1.39
### PR2 — 1.39 → 1.40
### PR3 — 1.40 → 1.41
### PR4 — 1.41 → 1.42

PR4 is the riskiest (decorrelator rewrite, new `RelShuttle` methods, Kotlin 2.x). Expect it to be 2-3x the effort of any of the others.

## Per-PR workflow

For each PR, follow this exact sequence (drawn from the original upgrade process doc and adapted to this repo):

### Step 1 — Bump version & dependency scan
1. Open `BodoSQL/calcite_sql/bodosql-calcite-application/pom.xml`.
2. Update `<org.apache.calcite-version>` (and `<org.apache.calcite.avatica-version>` when crossing 1.39 or 1.42).
3. Cross-reference upstream `gradle.properties` for that release against our `pom.xml` deps. Only bump a dep if Calcite's new version requires it and we explicitly declare it. Notably:
   - 1.39: avatica 1.25→1.26
   - 1.40: janino 3.1.10→3.1.12, quidem 0.11→0.12 (we likely don't declare quidem — skip)
   - 1.41: joou (new, for unsigned types), commons-text replaces commons-lang `StringEscapeUtils`
   - 1.42: avatica 1.27→1.28, Jackson 2.18.4.1→2.18.6, Kotlin 1.9.0→2.3.20 (we pin `kotlin.version` at `pom.xml:30`), Hadoop 3.3.3→3.4.3
4. Build: `pixi run bsql`. Expect failure — that's the signal to start propagating diffs.

### Step 2 — First pass: filter the trivial files
1. Open the GitHub compare URL for the release pair (e.g. `https://github.com/apache/calcite/compare/calcite-1.38.0...calcite-1.39.0`), **Files changed** view.
2. For each changed upstream file, check whether it exists at the mirrored path under `BodoSQL/calcite_sql/bodosql-calcite-application/src/main/java/org/apache/calcite/...`.
3. Maintain a **local checklist file** (e.g. `calcite-upgrade-1.39.txt`) of files still to process — GitHub doesn't preserve scroll position.
4. Trivial changes (import reorder, whitespace, unrelated sections) → apply directly, **matching upstream formatting exactly** so future diffs stay clean.
5. Defer the four problematic files (`Parser.jj`, `SqlValidatorImpl.java`, `SqlToRelConverter.java`, `RexSimplify.java`) plus any file with a `// Bodo Change:` in the touched region to the second pass.

### Step 3 — Second pass: `Bodo Change:` regions
For each deferred file where the upstream diff touches a `// Bodo Change:` block:
1. Find the upstream PR(s) responsible (linked from the release notes / `CALCITE-XXXX` JIRAs in the diff stats above).
2. `git blame` the `// Bodo Change:` lines **in this repo**. If the change predates the gemini split, check `Bodo-archive` (mention this to the reviewer — worst case the old `BodoSQL` repo).
3. Decide alignment:
   - Upstream fix supersedes ours → **delete** the `// Bodo Change:` block and remove the file from the shade excludes if no other Bodo changes remain.
   - Upstream fix is orthogonal → rebase our block onto the new upstream code, preserving the `// Bodo Change:` comment.
   - Upstream and Bodo overlap partially → split into the minimal surviving `// Bodo Change:` and document the reasoning in the PR description.
4. Flag every such decision in the PR for explicit reviewer sign-off.

### Step 4 — Problematic files (special handling)
- **`Parser.jj`** (`src/main/codegen/templates/Parser.jj`): we have removed sections that don't match our dialect (commented out, not deleted, to keep alignment). Upstream additions in 1.39 (`SqlSetOption` configurability), 1.40 (lateral column alias, quadratic-time fix), 1.41 (`IntervalWithoutQualifier`, `UNSIGNED`), 1.42 (`SELECT * EXCLUDE/REPLACE`, `:`, `ROW(*)`) all need careful merge. New syntax we don't want → comment out and add a `// Bodo Change:` explaining why.
- **`SqlValidatorImpl.java`**, **`SqlToRelConverter.java`**: line numbers will drift heavily. Use 3-way merge tooling (`git merge-file` against a checkout of the upstream tag) rather than manual patching.
- **`RexSimplify.java`**: 1.41 adds +310 lines (SEARCH simplification, `simplifyTrim`, LIKE folding). Our `BodoRexSimplify.kt` wrapper may need corresponding signature updates.

### Step 5 — New & removed files
- **New upstream files we need to shadow:** if a new Calcite class is referenced by one of our `// Bodo Change:` files and we can't avoid the reference, copy the file into our tree, add `// Bodo Change:` where modified, and register every generated class (including `$1`, `$2`, nested) in the `<excludes>` block of `pom.xml:386+`. List classes via `ls target/classes/org/apache/calcite/...` after a build.
- **Removed upstream files:** if an upstream deletion makes a `// Bodo Change:` obsolete, delete our copy and remove its excludes. **Diff our file against the upstream tag before deletion** to confirm no unmarked changes survive.
- **Removed dependencies** (e.g. 1.42 drops `commons-lang3`): verify we don't transitively rely on it before dropping.

### Step 6 — Test overrides
- Calcite ships tests in `calcite-testkit`. Our overridden copies live under `src/test/java/org/apache/calcite/{sql/test, test, test/catalog}`. Re-sync each against upstream.
- For new upstream tests that don't fit our dialect (common for parser), override with:
  ```java
  @Disabled
  @Test
  void testXxx() {}
  ```
  mirroring the existing pattern in `BodoParserTest.java`.
- If the test *API* changes (fixture constructors, `RelOptFixture` etc.), the BodoSQL Customer Tests repo may need a coordinated PR — flag this early.

### Step 7 — Build & test
1. `pixi run bsql` — must produce a clean JAR.
2. Java tests: `cd BodoSQL/calcite_sql && mvn test` (or the pixi equivalent). Investigate every failure; most will be either plan-shape changes (verify triviality) or API breaks.
3. BodoSQL customer Java tests (separate repo) — coordinate if test API changed.
4. Full Python suite: `mpiexec -n 2 pytest BodoSQL/bodosql/tests/ -v` plus `bodo/tests/` smoke. Watch for plan-cost drift from upstream cardinality/simplifier improvements — these are expected and should be verified as trivial (cost number change, pruned redundant node) before accepting.

## Specific high-risk items to flag for reviewers

| Release | Item | Why it's risky |
|---|---|---|
| 1.39 | `RelDecorrelator` made configurable; `SubQueryRemoveRule` new | We have `com/bodosql/calcite/application/logicalRules/SubQueryRemoveRule.java` — collision / dedup needed. |
| 1.40 | `ExpandDisjunctionHelper` in `RexUtil` | Our `RexUtil.java` has Bodo changes; verify interaction. |
| 1.41 | `Combine` rel node, `RelShuttle`-adjacent `RexNodeAndFieldIndex` | May require new `RelShuttle`/`RexShuttle` methods in our copies (`RexShuttle.java`). |
| 1.41 | `hasEmptyGroup()` deprecates `getGroupCount()` | Any of our `SqlOperatorBinding` consumers must migrate. |
| 1.42 | **General decorrelator rewrite** (`RelDecorrelator` near-rewrite, `ConditionalCorrelate`) | Our `RelDecorrelator.java` + `BodoRelDecorrelator.java` likely need rethinking, not just rebasing. Budget extra time. |
| 1.42 | New `RelShuttle.visit(X)` overloads (CALCITE-7511) | All our `RelShuttle` consumers must add methods or fail to compile. |
| 1.42 | Kotlin 2.3.20 | We pin `kotlin.version=1.9.0` (`pom.xml:30`); K2 compiler may surface warnings/errors in our 5 `.kt` files. |
| 1.42 | Hadoop 3.3.3→3.4.3 | Affects Iceberg connector too — coordinate with iceberg team. |

## Opportunities to bank (per the upgrade doc's "Noting Interesting Changes")

Maintain a running list in the upgrade PRs:
- **1.39**: `DpHyp` optimal join enumeration — may let us drop custom join-order rules.
- **1.40**: New `IntersectToSemiJoinRule` / `MinusToAntiJoinRule` — check if we can remove Bodo equivalents under `com/bodosql/calcite/application/logicalRules/`.
- **1.41**: Functional-dependency metadata (`RelMdFunctionalDependency`) — may improve our cardinality estimates.
- **1.41**: `simplifyTrim`, improved LIKE folding in `RexSimplify` — may obsolete Bodo simplifier tweaks.
- **1.42**: `AggregateReduceFunctionsOnGroupKeysRule` / `AggregateRemoveDuplicateKeysRule` — candidate plan-pruning wins.

## Estimated effort

| PR | Effort | Driver |
|---|---|---|
| PR1 (1.39) | ~3-5 eng-days | avatica bump, VARIANT/UUID touches validator, `SubQueryRemoveRule` collision |
| PR2 (1.40) | ~2-4 eng-days | Smallest Java diff; mostly rules + `RexUtil` |
| PR3 (1.41) | ~5-8 eng-days | Largest file count, unsigned types, `Combine`, heavy `RexSimplify` churn |
| PR4 (1.42) | ~8-12 eng-days | Decorrelator rewrite, `RelShuttle` API break, Kotlin 2.x, Hadoop 4.x |
| **Total** | **~18-29 eng-days** | Plus review bandwidth; recommend two reviewers per PR |

---

## PR1 progress log — 1.38 → 1.39

Status: **in progress**

### Step 1 — Version bump & dependency scan
- [x] Bump `<org.apache.calcite-version>` 1.38.0 → 1.39.0 in `pom.xml`
- [x] Bump `<org.apache.calcite.avatica-version>` 1.23.0 → 1.26.0 in `pom.xml`
- [x] Build `pixi run bsql` — initial build had 8 compile errors (all resolved)

### Compile errors fixed (Step 2-3)
1. **`SqlValidatorImpl.java` + `TableNamespace.java`**: `mustFilterFields` → `FilterRequirement` refactor. Replaced `getMustFilterFields()`/`mustFilterFields` with `getFilterRequirement()`/`filterRequirement`. Added `validateMustFilterRequirements()`, `purgeForBypassFields()`, `toQualifieds()`, `qualifiedMatchesIdentifier()` methods. Added `Stream` import.
2. **`RelOptTableImpl.java`**: Added `subSchemas()` override to `MySchemaPlus` (new abstract method on `SchemaPlus` in 1.39).
3. **`Parser.jj`**: Replaced removed `RESOURCE.illegalFromEmpty()` with `BODO_SQL_RESOURCE.genericTrimError()` (TRIM grammar restructured in 1.39, `illegalFromEmpty` removed).
4. **`SqlKind.java`**: Added 8 new enum values: `CONVERT_ORACLE`, `CHECKED_PLUS`, `CHECKED_MINUS`, `CHECKED_TIMES`, `CHECKED_DIVIDE`, `CHECKED_MINUS_PREFIX`, `ARRAY_SLICE` (already had `SUBSTRING_INDEX`). Updated `BINARY_ARITHMETIC`, `SYMMETRICAL_SAME_ARG_TYPE`, `EXPRESSION`, `FUNCTION` sets and `getFunctionKind()`. Added `CHECKED_ARITHMETIC` set.
5. **`ReturnTypes.java`**: Added `CHAR_NULLABLE_IF_ARGS_NULLABLE`, `VARIANT`, `VARCHAR_FORCE_NULLABLE`, `VARBINARY_FORCE_NULLABLE`.
6. **`RelOptUtil.java`**: Added `eqUpToNullability()` method and `SqlTypeUtil` import.
7. **`SqlStdOperatorTable.java`**: Added `CHECKED_PLUS`, `CHECKED_MINUS`, `CHECKED_MULTIPLY`, `CHECKED_DIVIDE`, `CHECKED_DIVIDE_INTEGER`, `CHECKED_UNARY_MINUS` operators. Added `TYPEOF` and `VARIANTNULL` functions. **Key gotcha**: checked operators must be `SqlBinaryOperator` (not `SqlMonotonicBinaryOperator`) to match upstream field type descriptors — mismatched types cause `NoSuchFieldError` at runtime.

### Test results after compile fixes
- BodoParserTest: 635 tests, 0 failures, 1 error (`testConvertOracle` — disabled as Oracle CONVERT not in our dialect)
- SimplificationTest: 51 tests, 0 failures, 0 errors
- Remaining: ~220 errors + 69 failures across plan codegen and snowflake tests (needs investigation — likely plan-shape changes and more missing fields/methods in forked files)

### Step 2 — First pass: trivial files
- Compare URL: https://github.com/apache/calcite/compare/calcite-1.38.0...calcite-1.39.0
- [x] `RelOptTableImpl.java` — `subSchemas()` added
- [x] `Parser.jj` — `illegalFromEmpty` replaced
- [ ] Walk remaining Files-changed tab; apply trivial diffs to other forked files

### Step 3 — `Bodo Change:` regions touched by upstream
- [x] `SqlValidatorImpl.java` — `mustFilterFields` → `FilterRequirement`
- [x] `TableNamespace.java` — `mustFilterFields` → `FilterRequirement`
- [x] `SqlKind.java` — new enum values + sets
- [x] `SqlStdOperatorTable.java` — checked operators + TYPEOF/VARIANTNULL
- [x] `ReturnTypes.java` — new return types
- [x] `RelOptUtil.java` — `eqUpToNullability` + `isPureLimit`/`isOffset` changes
- [ ] `RexSimplify.java` — checked arithmetic handling (diff identified, not yet applied)
- [ ] `RexUtil.java` — `isLosslessCast` in `isEffectivelyNotNull`, CHECKED ops in canonize
- [ ] `StandardConvertletTable.java` — ROW comparison cast fix
- [ ] `SqlToRelConverter.java` — Union via RelBuilder, OFFSET in subqueries, `convertUsing` fix
- [ ] `RelDecorrelator.java` — configurable rules, LIMIT 1 fix
- [ ] `VolcanoPlanner.java` — RelDecorrelator import

### Step 4 — Problematic files
- [ ] `Parser.jj` — `SqlSetOption` parsing made configurable via `${parser.setOptionParserMethod!default.parser.setOptionParserMethod}(Span.of(), null)`

### Step 5 — New & removed files
- [ ] `SubQueryRemoveRule` (new upstream, `core/.../rel/rules/SubQueryRemoveRule.java`, 492 lines) — **collision** with `com/bodosql/calcite/application/logicalRules/SubQueryRemoveRule.java`. Audit for overlap.
- [ ] `SqlCallFactory` / `SqlCallFactories` — new utility; may be referenced by checked-arithmetic changes in `SqlStdOperatorTable`. Copy if referenced.
- [ ] `ConvertToChecked` — new; likely not needed unless we expose checked arithmetic.
- [ ] `SqlUuidLiteral` — new; only needed if we adopt UUID type.
- [ ] `DpHyp` / `DphypJoinReorderRule` — new join enumeration; note as opportunity, not required.

### Step 6 — Test overrides
- [x] Disabled `testConvertOracle` in `BodoParserTest.java`
- [ ] Re-sync `src/test/java/org/apache/calcite/{sql/test/SqlTestFactory.java, test/SqlToRelFixture.java, test/SqlValidatorFixture.java, test/catalog/MockCatalogReader*.java}`
- [ ] Audit `BodoParserTest.java` for other new upstream parser tests to disable

### Step 7 — Build & test
- [x] `pixi run bsql` clean (JAR builds successfully)
- [x] BodoParserTest: 635/636 pass (1 disabled)
- [x] SimplificationTest: 51/51 pass
- [ ] Full Java test suite — ~220 errors + 69 failures remain (plan codegen tests, snowflake tests)
- [ ] `mpiexec -n 2 pytest BodoSQL/bodosql/tests/ -v`
- [ ] `mpiexec -n 2 pytest bodo/tests/ -m smoke -v`

### Notes for reviewer
- avatica jumps two minor versions (1.23 → 1.26) — verify no API breaks in our `avatica`-touching code
- `SubQueryRemoveRule` name collision is the highest-risk item; may need rename or dedup
- **Key lesson**: when adding new fields/methods to forked files that shadow the calcite JAR, the field **types** must exactly match upstream (e.g. `SqlBinaryOperator` vs `SqlMonotonicBinaryOperator`) or non-forked JAR code will fail with `NoSuchFieldError` at runtime due to field descriptor mismatch
- Remaining ~220 errors are likely plan-shape changes and/or more missing fields in forked files — systematic diff of each forked file against upstream 1.39 needed
