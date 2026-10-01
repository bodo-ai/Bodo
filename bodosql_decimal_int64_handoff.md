# Handoff: Int64-Backed Decimals for the BodoSQL C++ Backend

## Implementation status (updated after this session)

Implemented in this session, with one design deviation (see "Representation
deviation" below):

- `decimal_is_int64(precision)` helper added in `bodo/libs/_bodo_common.h`
  (`DECIMAL_INT64_MAX_PRECISION = 18`), selecting int64 fast paths.
- **Expressions** (step 3): new `decimal_int64_binary_op` in
  `bodo/libs/_decimal_ext.cpp/.h`, wired into `do_arrow_compute_binary`
  (`bodo/pandas/physical/expression.cpp`). Covers add/subtract/multiply and
  all six comparisons for decimal-decimal and decimal-integer operands
  (arrays and scalars, with null propagation). Rows whose values fit int64
  (checked per row, so any declared precision works) use int64 loads and
  `__int128` arithmetic; other rows fall back to the existing 128-bit scalar
  utilities with identical semantics. For result precision <= 38 the exact
  result is produced directly in the Snowflake result type, skipping the
  Arrow compute call and the trailing result-type cast (this removes the
  per-row `CastFunctor`/`RescaleWouldCauseDataLoss` work seen in profiles).
  For result precision > 38 the gandiva multiply rounding (half away from
  zero) and FitsInPrecision(38) overflow check are replicated; overflow
  throws "Decimal overflow in operation <op>" (the legacy array utility
  silently appends undefined values on overflow, so this is a deliberate,
  stricter deviation). Division still uses the generic path.
- **Aggregates** (step 4): `sum_decimal128_int64` fast accumulation (one
  `__int128` add per row + the same precision-38 check) used by groupby
  sum/mean in `_groupby_do_apply_to_column.cpp` when the input column has
  precision <= 18; min/max use direct int64 comparisons of the low 8 bytes.
  Mean's final per-group divide is unchanged. var/std/prod untouched.
- **Escape hatch**: `BODO_DISABLE_DECIMAL_INT64_FASTPATH=1` reverts to the
  128-bit paths for A/B testing.
- **In-place evaluation**: a second, array-level kernel
  (`decimal_int64_binary_op_arrays`) evaluates decimal binary operations
  directly on Bodo `array_info` buffers with output allocated from Bodo's
  pool (no Arrow Datum round trip); it is the primary path in
  `do_arrow_compute_binary` at the `ExprResult` level, with the Datum-level
  kernel kept for fallbacks. Groupby sum/mean/min/max use a per-row int64
  fit check (not just the declared precision), so columns like
  decimal(31,4) whose values fit int64 take the fast path.
- **TPC-H SF10 A/B (iceberg filesystem catalog, 1 worker, warmup + 1 iter,
  `BODO_DATAFRAME_LIBRARY_RUN_PARALLEL=0`, fastpath on vs
  `BODO_DISABLE_DECIMAL_INT64_FASTPATH=1`)**: 22/22 queries pass in both
  runs and all 22 result files are byte-identical. Total 373.5s vs
  408.8s (**1.09x**); biggest wins on decimal-heavy queries: Q21 1.28x,
  Q20 1.24x, Q22 1.24x, Q1 1.11x, Q5/Q18 1.11x. Remaining Q1 profile:
  parquet FLBA(16)->Decimal128 decode dominates (FromBigEndian ~8% +
  RawBytesToDecimalBytes ~5% + dictionary decode ~3.5%), which only the
  write-side INT64 parquet companion (or reading INT64 physical columns)
  removes; the fast kernel itself is ~7% and groupby machinery ~7%.
- **Correctness**:
  - New C++ differential suite `bodo/tests/test_decimal_int64.cpp`
    (registered in `CMakeLists.txt`) compares the fast path against Arrow
    compute (result precision <= 38) and against the gandiva utilities
    (result precision 38) across precisions, scales, scalars, nulls and
    rounding/overflow cases.
  - `BodoSQL/bodosql/tests/test_decimal.py` (144 tests) and the df_lib
    decimal tests in `bodo/tests/test_df_lib/test_end_to_end.py` pass.
  - End-to-end differential run over 21 BodoSQL queries (arithmetic,
    mixed-integer ops, comparisons, filters, groupby sum/avg/min/max/count,
    chained expressions) produces byte-identical output with the fast path
    enabled and disabled.
- **Not done (unchanged from the plan below)**: the read path stays
  zero-copy 16-byte (its decode cost lives inside Arrow's scanner), the
  scanner filter pushdown (step 2) still evaluates Decimal128 comparisons,
  and the write-side INT64 parquet companion is not implemented. All three
  depend on the write-side companion (or a scanner schema override) to see
  int64 columns; they remain the follow-up work below.

### Representation deviation

The plan proposed storing decimal(p<=18) as 8-byte int64 buffers. That
representation is threaded through ~100 generic sizing/copy/shuffle/hash
sites (`numpy_item_size[DECIMAL]`), which is not safely reviewable in one
change. Instead, buffers keep the 16-byte Arrow-compatible layout (values
sign-extended into the high 8 bytes, so zero-copy Arrow conversions and all
generic infrastructure are untouched) and the int64 backing is a *compute*
property: hot kernels load the low 8 bytes and compute in int64/`__int128`,
which is where the profiling showed the time going (expressions and
aggregates). The 8-byte storage optimization can be layered on later; the
`decimal_is_int64` gate and the kernel structure are already in place.

---

## Problem

TPC-H Q1 at SF10 takes 30.5s single-worker in BodoSQL; DuckDB runs the same
query on the same files in 2.75s (11x). Profiling (perf, 999Hz) shows ~97% of
BodoSQL's post-read time is Decimal128 arithmetic, and the read path adds more
of the same. All evidence in
`bodosql_iceberg_io_singlethread_report.md`; this doc is the implementation
plan for the fix.

## Evidence (measured on this box, WSL2, 1 worker)

- Q1 SF10 op breakdown: read 16.9s (now ~4s after the decode-pool/string
  fixes), projection 3.1 + 4.9s, aggregate 5.3s. At SF100 the two projections
  plus aggregate are 229.6s of a 235.4s query.
- perf attribution of the compute: `BasicDecimal128::operator*=`,
  `DecimalDivide<BasicDecimal128>`, `Abs`, `operator*`, `FitsInPrecision`,
  `RescaleWouldCauseDataLoss<BasicDecimal128>`,
  `CastFunctor<Decimal128,Decimal128>`, `multiply_decimal_scalars_util`,
  `apply_to_column<__int128, DECIMAL>` — together ~33% of total process
  samples in a decode-fast run (~50% of the main thread).
- Q6's pushed-down filter inside the Arrow scanner costs ~25% of its scan:
  decimal comparisons go through the generic
  `ScalarBinary<BooleanType, Decimal128, Decimal128>` kernel plus per-row
  rescale casts (`CastFunctor`, `RescaleWouldCauseDataLoss`).
- Root cause: Arrow's Decimal128 is a 16-byte sign-magnitude value, so every
  op is multi-word arithmetic plus precision/rescale validation per element.
  DuckDB instead maps precision to physical width (vendored source,
  `src/common/types.cpp:105-115`: <=4 int16, <=9 int32, <=18 int64, <=38
  int128) and verified at runtime for our data:
  `l_quantity`/`l_extendedprice` DECIMAL(15,2) -> int64;
  `l_extendedprice * (1 - l_discount)` -> DECIMAL(18,4) -> still int64;
  `sum(...)` -> DECIMAL(38,2) -> int128 accumulator; `avg` -> double.
  Every per-row operation in DuckDB's Q1 is plain int64 math.

## Current decimal flow in the C++ backend

1. Parquet stores DECIMAL(p,s) as FIXED_LEN_BYTE_ARRAY(16), big-endian.
2. Arrow scanner decodes to `decimal128` arrays; pushed filters are Arrow
   expressions over Decimal128 (with implicit rescale casts).
3. `arrow_array_to_bodo` produces a Bodo `array_info` of type
   `NULLABLE_INT_BOOL`/DECIMAL whose data buffer is raw 16-byte values
   (precision/scale in `array_info::precision/scale`).
4. Expressions (projection, filter, aggregate inputs) evaluate via Arrow
   compute (`PhysicalExpression`, `apply_to_column<__int128, DECIMAL>`).
5. Aggregates consume the decimal arrays (sum/avg/min/max) in 128-bit.

## Proposed change

Represent decimal(p, s) with **p <= 18 as int64** (scaled by 10^s) end-to-end
in the C++ backend's data plane, mirroring DuckDB:

1. **Read path** (`arrow_decimal_array_to_bodo`, `bodo/libs/_bodo_to_arrow.cpp`):
   for p <= 18, convert each FLBA(16) big-endian value to int64 during decode.
   p <= 18 guarantees the value fits in int64, so this is "take the low 8
   bytes, big-endian load, sign-extend" — vectorizable, no 128-bit object.
   Keep precision/scale in `array_info` as today. For p > 18 keep the current
   16-byte path. Dictionary-encoded decimals: convert the dictionary once,
   then gather int64s by index (replaces per-row `FromBigEndian`).
2. **Filter pushdown** (`generate_expr_filter` /
   `bodo::arrow_py_compat::scanner_from_py_dataset`): build the pushed
   expression over the int64 representation (comparison against
   `scalar * 10^s` as int64). This removes the per-row rescale casts and
   128-bit comparisons from the scanner, which is ~25% of Q6's scan.
3. **Expressions** (`bodo/pandas/physical/expression.*`): for decimal inputs
   that are int64-backed, evaluate +,-,* as int64 ops with DuckDB-style
   result-type rules (add/sub: max(s1,s2), precision max(p1,p2)+1; mul:
   p1+p2, s1+s2) with overflow checks only where the result precision
   requires them. Division: widen to int128 or use double like DuckDB
   (`avg` is double). If an expression result exceeds p=18, materialize the
   16-byte form for that node only.
4. **Aggregates** (`bodo/libs/groupby*` / `PhysicalAggregate`): sum on
   int64-backed decimals accumulates in int128 (one add/row), final rescale
   once; avg accumulates double; min/max/count unchanged int64.
5. **Boundaries**: any operator that must return decimal to the user converts
   int64 -> Decimal128 (or string) at the sink only.

## Scope decision (recommended)

Do this **only for the C++ backend data plane first** (steps 1-5 above). The
JIT (`@bodo.jit`) path, shuffle serialization, GPU/cudf path, and
pandas-interop (`pd.Decimal` conversions) keep the 16-byte representation and
convert at boundaries where they meet the C++ backend. This keeps the change
reviewable: the C++ backend is where TPC-H/BodoSQL time goes, and its decimal
handling is concentrated in the files listed above.

## Write-side companion (separable, cheap)

Emit DECIMAL(p<=18) as parquet **INT64** physical type instead of
FLBA(16) (parquet-standard). Then step 1's conversion disappears entirely
(native little-endian load). Files should compress at least as well. Keep
FLBA(16) only for p > 18.

## Correctness plan

- Byte-identical results on the SF10 suite (all 22 answers) vs current build;
  the suite runs in ~12 min (`benchmarks/tpch/bodo_sql/bodosql_queries.py`,
  `BODO_DATAFRAME_LIBRARY_RUN_PARALLEL=0`).
- Focused tests: overflow at precision boundaries (multiplication that
  exceeds p=18 must match current Decimal128 behavior, including error
  semantics), negative values and rounding on divide/avg, null handling,
  dictionary-encoded decimal columns, schema evolution with mixed widths.
- The profiling harness and metrics (`BODO_TRACING_OUTPUT_DIR`) quantify
  per-op time before/after; `io_profile/` has the harness.

## Expected impact (from measurements)

- Q1 SF10: 30.5s -> ~5s projected (read ~4s + int64 compute ~1s; DuckDB does
  the whole query in 2.75s with the same data and one thread).
- Q6 SF10: 10.3s -> ~2s (filter becomes int64 compares).
- Suite-wide: the decimal-compute share is query-dependent; decimal-heavy
  queries (Q1-Q10, Q14-Q22) should improve 2-5x; the l_comment-style probes
  are unaffected.

## Risks / open questions

- Arrow compute kernels used elsewhere may implicitly expect 16-byte decimal
  buffers (e.g. casts between decimal precisions, decimal->varchar
  formatting for output). Audit `PhysicalExpression` op coverage.
- `RescaleWouldCauseDataLoss` semantics on the int64 path must match Arrow's
  (same errors, same rounding) or results diverge.
- Two decimal representations coexisting (int64-backed and 16-byte) means
  every consumer of DECIMAL arrays needs a width check; consider a
  `decimal_is_int64()` helper next to `array_info::precision`.
- GPU path (`gpu_read_iceberg`, cudf) untouched; conversions at boundaries
  if mixed plans appear.
- Overflow-check strategy: DuckDB checks per-op and promotes; decide between
  always-promote (simpler, slower) vs static result-precision analysis
  (matches DuckDB binder behavior).

## Related work already in tree (this session)

- Zero-copy string conversion in `arrow_string_binary_array_to_bodo`
  (widened offsets / char views; `BODO_TEST_DISABLE_STRING_ZERO_COPY=1`
  reverts).
- Scanner batch decoupling prototype (`BODO_SCANNER_BATCH_SIZE`).
- Experimental decode-executor knobs: `BODO_INLINE_SCANNER_DECODE`
  (deadlocks with the scan pipeline - do not ship),
  `BODO_FREE_RUNNING_DECODE` (no gain, deadlocks at scanner boundaries -
  do not ship). Both should be deleted or kept only as documentation of the
  dead ends.
