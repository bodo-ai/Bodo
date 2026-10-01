# Handoff: Physical Int64 Decimals End-to-End (BodoSQL C++ backend data plane)

Companion to `bodosql_decimal_int64_handoff.md`. That doc's compute phase
(int64 arithmetic in expressions/aggregates) is done; this doc covers making
decimal(p <= 18) **physically 8 bytes** through the whole C++ backend data
plane: parquet/Iceberg read -> expressions -> aggregates -> shuffle/exchange
-> sink.

## Where we are (measured on this box, WSL2, 1 worker)

The compute phase landed: int64-backed decimal arithmetic/comparisons
(`decimal_int64_binary_op_arrays`, in-place on Bodo buffers), int64 groupby
sum/mean/min/max with per-row fit checks, `decimal_is_int64()` helper, and an
A/B escape hatch (`BODO_DISABLE_DECIMAL_INT64_FASTPATH=1`).

TPC-H SF10 iceberg A/B (fastpath on vs off): **373.5s vs 408.8s = 1.09x**,
22/22 queries, all 22 result files byte-identical. Biggest wins: Q21 1.28x,
Q20 1.24x, Q22 1.24x, Q1/Q5/Q18 1.11x.

Q1 profile after the compute phase (perf, ~30s query):
- parquet decimal decode ~17%: `Decimal128::FromBigEndian` 8.0%,
  `RawBytesToDecimalBytes<Decimal128>` 5.4%, dictionary byte-array decode
  ~3.5%.
- decimal expression kernel ~7% (mostly the 16-byte sign-extended store and
  per-batch output allocation).
- groupby machinery ~9% (`apply_to_column` 4.8%, `get_group_info` 3.7%).
- BLAS thread spinning ~3.9% (openblas `blas_thread_server` idle wakeups;
  unrelated to decimals, cheap win: cap OMP/BLAS threads).
- rest: string group keys/hash, snappy + bit unpacking, JVM ~2%, sink
  conversions, spread thin.

**Key insight: decimal arithmetic is no longer the bottleneck.** The
handoff's original projection (Q1 ~5s) assumed it was. The remaining
decimal-specific costs are the 16-byte *decode* and *storage*; everything
else is data-plane infrastructure.

## Goal

`decimal(p <= 18)` is stored as 8-byte little-endian int64 (scaled by 10^s)
in Bodo `array_info` buffers everywhere inside the C++ backend data plane.
`decimal(p > 18)` and all boundary crossings (user output, JIT, GPU,
serialization) keep the 16-byte form. Width is derived from precision, which
already travels everywhere (`array_info::precision`, `bodo::DataType`
Serialize/Deserialize, Iceberg/Arrow schemas).

Important consequence of keeping Snowflake result-type rules (required for
byte-identical results): Q1's hot multiply `l_extendedprice *
(1 - l_discount)` has result decimal(31,4) -> its *output* must stay 16-byte.
E2E 8-byte storage therefore mostly benefits decode, comparisons, small
results (p <= 18), memory footprint, and the write-side companion - not
wide-result intermediates.

## Work items

### 1. Width-aware sizing (the audit; the main risk)
Only DECIMAL changes width, so replace dtype-only sizing with a
width-aware helper wherever DECIMAL can flow:
- Add `bodo_array_item_size(const array_info&)` (returns
  `decimal_item_bytes(precision) = precision <= 18 ? 8 : 16` for DECIMAL,
  else `numpy_item_size[dtype]`) next to `decimal_is_int64` in
  `bodo/libs/_bodo_common.h`.
- Audit/convert the ~103 `numpy_item_size[...]` sites (counts from grep):
  `_array_utils.cpp` (18), `_bodo_common.cpp` (17, alloc/copy/like),
  `_array.cpp` (11), `_array_build_buffer.{h,cpp}` (16),
  `_chunked_table_builder.cpp` (7), `streaming/_shuffle.{h,cpp}` (8),
  `groupby/_groupby_common.cpp` (5), `_array_utils.h` (4),
  `_array_hash.{h,cpp}` (5, join-key hashing!), `_shuffle.cpp` (3),
  `_quantile_alg.cpp` (3), `groupby/_groupby_update.cpp` (3),
  `_bodo_to_arrow.cpp` (3), plus `io/arrow_reader.cpp`,
  `_javascript_udf.h`. Many are dtype-generic templates where DECIMAL can
  arrive; gate each by "is this a DECIMAL array here?".
- Shuffle receive buffers are sized from schema `DataType`; width is
  derivable from the serialized precision (already transmitted).

### 2. Read path (`bodo/libs/_bodo_to_arrow.cpp`)
`arrow_decimal_array_to_bodo`: for p <= 18 decode each Arrow Decimal128
value to int64 (little-endian load of the low 8 bytes; high bytes are the
sign extension by construction) into an 8-byte buffer; keep precision/scale
in `array_info`. Dictionary decimals: convert the dictionary once, gather
int64 by index. Keep the current 16-byte zero-copy path for p > 18. Gate
with a parameter so non-C++-backend callers (JIT, snowflake connector)
can keep 16-byte until their boundaries are updated.

### 3. Write-side INT64 parquet companion (separable, cheap-ish)
Emit DECIMAL(p <= 18) as parquet INT64 physical + decimal logical type
(parquet-standard) in `bodo/io/parquet_write.cpp` and
`bodo/io/iceberg_parquet_write.cpp`. Files compress at least as well.
**Open question**: Arrow's parquet reader maps decimal logical types to
Decimal128Array regardless of physical width, so INT64-physical files alone
do not remove the reader's int64->Decimal128 widening (though it becomes a
cheap LE load + sign-extend instead of 16-byte big-endian byte swapping).
Options, in order of preference:
a. If Arrow can hand back the raw INT64 for decimal columns (check
   `ArrowReaderProperties` / dataset projection casts), use it.
b. Bypass: read the column through a schema that omits the decimal logical
   type for Bodo-managed tables (Bodo knows decimal-ness from the Iceberg
   schema), reconstruct decimal-ness on the Bodo side.
c. Keep the widening (still ~3x cheaper than FLBA(16) decode).
Note (b) affects external readers only if the parquet loses the decimal
logical annotation - prefer keeping the annotation and solving it on the
read side.

### 4. Scanner filter pushdown over int64
`generate_expr_filter` (bodo/io/iceberg/read_parquet.py) +
`bodo::arrow_py_compat::scanner_from_py_dataset`: once the scanner sees
int64 columns (option 3a/3b), build pushed predicates over the int64
representation (literal scaled by 10^s as int64). Removes the per-row
rescale casts and 128-bit compares (~25% of Q6's scan). Without 3, this
item is blocked.

### 5. Kernels
- `decimal_int64_binary_op_arrays`: when the result precision is <= 18,
  write an 8-byte output (no sign-extended 16-byte store); promote to
  16-byte only for p > 18 results. Comparisons already load 8 bytes.
- Groupby sum/mean accumulate from 8-byte inputs directly; min/max compare
  int64; output width follows the output type (sum's precision-38 output
  stays 16-byte).
- `_array_hash.h`: hash 8-byte decimal keys as int64 (join/groupby keys).

### 6. Boundaries
- Sink: `bodo_array_to_arrow` widens 8 -> 16 bytes when building
  Decimal128Array (or, better, hands Arrow an int64 array + lets the
  Iceberg/parquet writer encode INT64 directly).
- JIT, GPU/cudf, pandas-interop: keep 16-byte; convert where they meet the
  C++ backend (unchanged from the phase-1 decision).
- `estimated item size` / memory-budget code paths in `_bodo_common.cpp`.

## Correctness plan

- Byte-identical results on the SF10 suite (22 answers) vs the current
  build; harness: `benchmarks/tpch/bodo_sql/bodosql_queries.py`,
  `BODO_DATAFRAME_LIBRARY_RUN_PARALLEL=0 BODO_NUM_WORKERS=1`, compare
  `--output_path` files by md5.
- The e2e differential script pattern (21 BodoSQL queries over decimal
  columns, fastpath on/off, diff outputs) used in this session - commit it
  under `benchmarks/` or `bodo/tests/` scripts.
- Extend `bodo/tests/test_decimal_int64.cpp` with 8-byte-representation
  cases (arrays kernel already differential-tested vs the Datum kernel).
- Focused: overflow at precision boundaries, negatives, rounding on
  rescale, nulls, dictionary decimals, schema evolution with mixed widths
  (8-byte and 16-byte columns in one table), shuffle round-trip of 8-byte
  columns, sink widening.

## Expected impact (from the measured profile)

- Q1 SF10: ~28s -> ~23-24s (decode ~17% -> ~0-3%; kernel store ~1-2%).
  The write-side companion is the single biggest piece (~5s of Q1).
- Q6 SF10: ~10.5s -> ~8s (scanner filter on int64).
- Suite-wide: 373s -> roughly 330-345s; cumulative vs the pre-project
  baseline (409s) about 1.2-1.25x. Decimal-heavy queries with narrow
  results and filters benefit most; queries dominated by wide decimal
  intermediates (Q1's multiply) or by non-decimal costs gain less.

## What this does NOT get us

DuckDB runs Q1 SF10 in 2.75s on the same files. After this project the gap
is no longer decimal-width-related: the remaining ~20s of Q1 is data-plane
infrastructure - full materialization of scan output and projection
intermediates, per-batch buffer allocation/refcounting, string group-key
handling, aggregate machinery, JVM/plan overhead, BLAS thread spinning.
Reaching DuckDB-class performance needs a separate effort: late
materialization + fused scan/filter/project/aggregate pipelines with
selection vectors, buffer reuse across batches, dictionary-propagating
group keys, and single-threaded BLAS/allocator tuning. Recommend a separate
profiling-driven handoff for that once this lands.

## Risks / open questions

- The ~103-site sizing audit is the core risk; mitigate with the helper,
  per-file review, the differential suites, and shipping behind
  `BODO_DECIMAL_INT64_STORAGE=1` (default off) for one release.
- Arrow reader semantics for INT64-physical decimals (work item 3).
- Iceberg spec compliance for external readers if the logical annotation is
  ever dropped (avoid).
- Dictionary-encoded decimals and mixed-width schema evolution.
- Two coexisting widths means every consumer needs the width check; the
  `decimal_is_int64`/`bodo_array_item_size` helpers are the contract.
- GPU path untouched; convert at boundaries if mixed plans appear.
