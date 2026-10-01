# Handoff: 8-byte Decimal Storage — Remaining Bugs After the Width-Aware Audit

Companion to `bodosql_decimal_int64_e2e_handoff.md` (storage phase plan) and
`bodosql_decimal_int64_handoff.md` (compute phase). Read those first for
project context. Results of the A/B benchmarking session that produced this
doc are in `/home/isaac/Bodo/ab_dec64/RESULTS.md` (not committed).

## State of the project after this session

Three-part optimization for decimal(p <= 18) in the BodoSQL C++ backend:

1. **Compute fastpath** (landed earlier): int64 arithmetic on the low 8 bytes
   of decimal buffers, in-place array kernel, scalar support, aggregate
   fastpaths. Gate: `BODO_DISABLE_DECIMAL_INT64_FASTPATH=1`. Correct and
   verified; worth ~1.1x on decimal-arithmetic-heavy queries only.
2. **INT64-physical parquet I/O** (landed this session): Bodo's writer now
   emits decimal(p <= 9) as INT32 and decimal(p <= 18) as INT64 physical types
   (`enable_store_decimal_as_integer()` on the `WriterProperties` builder in
   `bodo/io/parquet_write.cpp` — this is Iceberg-spec and Parquet-spec
   compliant and matches Spark/Iceberg-Java defaults). The iceberg reader maps
   decimal(p <= 18) to `pa.decimal64` in the pyarrow dataset read schema
   (`schema_with_decimal64` in `bodo/io/iceberg/read_parquet.py`), and the C++
   conversion handles `arrow::Decimal64Array`
   (`arrow_decimal64_array_to_bodo` in `bodo/libs/_bodo_to_arrow.cpp`).
   `EvolveArray` (bodo/io/iceberg_parquet_reader.cpp) accepts decimal32/64/128
   variants with matching precision/scale. **Verified: SF10 suite on
   INT64-physical files, default mode: 22/22, all outputs identical to the
   FLBA baseline, 333.4s -> 285.4s = 1.17x.**
3. **8-byte storage** (`BODO_DECIMAL_INT64_STORAGE=1`, default off): buffers
   store decimal(p <= 18) as 8 bytes instead of 16. Adds ~1.09x on top of (2)
   (262.5s vs 285.4s) and halves decimal memory footprint. **Keep the flag
   default OFF (2026-09-30): with the SF10 battery fixes below, flag OFF is
   fully correct (SF10 22/22 identical to the FLBA baseline, JIT correct),
   but flag ON still corrupts `@bodo.jit` decimal workloads (numba's decimal
   type model assumes 16-byte buffers). Promoting requires making the numba
   decimal type model width-aware first.**

## SF10 promotion-gate findings (2026-09-30, after bug 1 + bug 2 fixes)

Two crashes/corruptions found and fixed when re-running the SF10 battery:

1. **Segfault (null pool):** `arrow_decimal64_array_to_bodo`'s 8-byte branch
   (bodo/libs/_bodo_to_arrow.cpp ~1184) passed the raw `pool` through to
   `AllocateBodoBuffer`; the iceberg reader passes `pool = nullptr` (see the
   "TODO Pass BufferPool" at bodo/io/iceberg_parquet_reader.cpp:913/1029),
   and `NRT_MemInfo_alloc_safe_aligned_pool` dereferences the null pool.
   Fixed with the same `pool != nullptr ? pool : BufferPool::DefaultPtr()`
   fallback the 16-byte branch (and the Decimal128 counterpart) already had.
2. **Flag-OFF corruption:** the scan sites passed a literal `true` for
   `int64_storage` (bodo/io/iceberg_parquet_reader.cpp:915/1031,
   bodo/io/parquet_reader.cpp:284), so with `BODO_DECIMAL_INT64_STORAGE`
   unset the scan produced 8-byte decimal buffers while downstream
   (JIT typing, join codegen) assumed the flag-off 16-byte layout: SF10
   storage-off runs had wrong decimal aggregations (q01/q02/q05/q08/q09/
   q10/q11/q18 vs the FLBA baseline), a shape change in q11, and Q22 failed
   with "General Join Conditions with 'DecimalArrayType(38, 8)' column type
   and 'Decimal128Type(38, 8)' data type not supported" (bodo/ir/join.py).
   Fixed: the scan sites now pass `decimal_int64_storage_enabled()` so the
   layout is consistent with downstream in both modes. (This supersedes the
   bug-1 note "only the C++-backend scan sites pass true" — literal `true`
   is only consistent when the env flag is on.)

Post-fix status:
- SF10 battery (FLBA warehouse, 1 worker): 22/22 in BOTH modes, all outputs
  data-identical (row-sorted) to the FLBA baseline (`ab_dec64/int64wh_16b`
  and `ab_dec64/baseline` output sets).
- 9-query battery identical across storage on/off x 1/8 workers.
- `bodo/tests/test_decimal_int64.cpp` 9/9 in both modes;
  `test_decimal.py` 144 passed flag-off (1 worker); df_lib decimal tests
  11 passed both modes.
- **Remaining flag-ON blocker:** `@bodo.jit` decimal workloads produce wrong
  results/crash with 8-byte buffers because numba's decimal type model still
  assumes 16-byte layout: `/tmp/dec_repro/minonly.py` (JIT groupby-min)
  returns 0.00 instead of 1.00 with the flag on, and
  `test_decimal_int_multiply_vector` dies (SIGKILL) ~20 tests into a
  flag-on `test_decimal.py` run. Fixing this means threading the layout
  through the numba decimal type model (or always widening at JIT
  boundaries); do that before promoting the flag to default.

## Bug 1 (RESOLVED): sort/limit chain zeroes pass-through decimal values

**Root cause (found 2026-09-30).** Decimal buffer width (8 vs 16 bytes) was
inferred from `precision` + the global `BODO_DECIMAL_INT64_STORAGE` env flag
(`decimal_item_bytes(precision)`), but 16-byte-layout decimal arrays
legitimately exist even when the flag is on:

- `arrow_array_to_bodo` / `arrow_table_to_bodo` default to
  `decimal_int64_storage=false` (16-byte buffers, required by the JIT/numba
  boundary), and
- `plan_optimizer.arrow_to_cpp_table` (`df_to_cpp_table`, used by
  `gatherv_nojit` and Python-UDF plumbing in `bodo/spawn/utils.py`) converts
  result DataFrames back into C++ tables through that default, so 16-byte
  buffers flowed into C++ pipelines and MPI serialization.

Any width inference from the env flag then disagreed with the actual buffer
layout. The final `bodo_array_to_arrow` read 16-byte-stride buffers at
8-byte stride (values re-appear on even rows, odd rows zero — exactly the
"0.00" symptom and the "first half correct, rest zeroed" pattern), and MPI
serialization paths computed 8-byte message sizes for 16-byte data (the
"message truncated" 2x stride mismatch and glibc "free(): chunks corrupted"
at >1 worker).

**Fix.** Decimal storage width is now self-describing on `array_info`:

- `array_info` gained a `decimal_int64_layout` bool (constructor default =
  env flag, so all C++-internal allocations keep their previous behavior).
- `decimal_item_bytes(precision, int64_layout)` takes the layout explicitly;
  `bodo_array_item_size(arr)` and `bodo_array_to_arrow` use the array's
  flag instead of the global env flag.
- `arrow_decimal_array_to_bodo` / `arrow_decimal64_array_to_bodo` set the
  flag from their `int64_storage` param (and no longer consult the env flag
  internally).
- `pyarrow_to_cpp_table` / `pyarrow_array_to_cpp_table` (bodo/pandas/
  _plan.cpp) now pass `decimal_int64_storage_enabled()` so Python→C++
  pipeline-boundary tables use pipeline layout (JIT/numba-boundary sites
  keep the default false).
- `ChunkedTableArrayBuilder::UnsafeAppendRows`, `ChunkedTableArrayBuilder`
  alloc/Finalize sizing (`get_nullable_arr_alloc_sizes`), and
  `ArrayBuildBuffer::UnsafeAppendBatch` are mixed-width safe (both strides
  computed independently, values sign-extended via
  `decimal_load_wide`/`decimal_store_wide`).
- Legacy shuffle `fill_send_array` gained a 8-byte decimal64 variant
  (`fill_send_array_inner_decimal64` in bodo/libs/_shuffle.cpp); the old
  `fill_send_array_inner_decimal` hardcoded 16-byte strides and overflowed
  width-aware send buffers (root cause of the 8-worker
  DISTINCT/groupby-key heap corruption).

**Verification.** `bodo/tests/test_decimal_int64.cpp` 9/9 with and without
`BODO_DECIMAL_INT64_STORAGE=1` (the round-trip test now covers both the
default 16-byte JIT-boundary conversion and explicit int64 conversion).
The `/tmp/dec_repro/repro.py` 9-query battery is data-identical between
storage on/off at BOTH 1 and 8 workers (row-sorted frame equality, 36/36
comparisons). `BodoSQL/bodosql/tests/test_decimal.py` 144 passed (default
mode). SF10 battery still recommended before unblocking item 3.

The `BODO_DEC_DEBUG=1` env var enables decimal checksum/stride tracing
prints (`[DEC-ARROW2BODO]`, `[DEC-TOARROW-8B]`, `[DEC-APPENDROWS]`,
`[DEC-FINALIZE]`) which were used to localize this bug and are safe to
leave in place for future debugging.

**Width bugs already fixed this session (same class, fixed and verified —
check these are kept):**
1. `bodo/io/arrow_reader.cpp:458` PrimitiveBuilder fallback: copied Arrow
   Decimal128 with 16-byte stride into width-aware 8-byte output. Fixed with
   separate in/out strides (`copy_data` gained an `in_dtype_size` param).
2. `bodo/libs/_chunked_table_builder.h:276` index-based `UnsafeAppendRows`:
   `dtype_to_type<DECIMAL> = __int128_t` forced a 16-byte element copy into
   8-byte buffers. Fixed with a DECIMAL width-aware byte-copy template
   (`dtype != DECIMAL` added to the T-based templates' requires clauses).
3. `bodo/libs/streaming/_shuffle.cpp:927` `AsyncShuffleSendState::addArray`:
   allocated the send buffer without precision -> 16-byte sizing + 2x MPI
   messages. Fixed by passing `in_arr->precision`/`scale` (and setting them
   explicitly after alloc).
4. `bodo/libs/groupby/_groupby_common.cpp:559`
   `aggfunc_output_initialize_kernel`: memset at `numpy_item_size[DECIMAL]`
   over 8-byte MIN/MAX outputs. Fixed with `bodo_dtype_item_size`.
5. JIT boundary leak: 8-byte decimals reached numba-generated code (assumes
   16-byte) -> double-free at teardown. Fixed by threading an explicit
   `decimal_int64_storage` flag through `arrow_array_to_bodo` /
   `arrow_table_to_bodo` / `arrow_recordbatch_to_bodo` (default false =
   16-byte); only the C++-backend scan sites (parquet_reader.cpp:284,
   iceberg_parquet_reader.cpp:890/1005) pass true. Do not revert this to a
   global default — the JIT path must keep 16-byte buffers.

## Bug 2 (RESOLVED): p>38 decimal multiply hangs

**Root cause (found 2026-09-30).** Not an infinite loop. The check-path
kernels raised a data-dependent C++ exception (`Decimal overflow in operation
multiply`) on only the ranks holding overflowing rows. Those ranks exited the
JIT'd query, skipped its remaining MPI collectives, and reached the
post-execution allreduce in `bodo/spawn/worker.py:481`, while the ranks
without overflowing rows kept executing and blocked in collectives (e.g.
`dist_exscan` in the JIT'd `bodosql_impl`). MPICH busy-polls in `MPIR_Wait`,
so every rank spun at 100% CPU forever: a rank-divergent-error deadlock, not
a compute loop. A kernel-level allreduce of the overflow flag does NOT fix it
(tries were reverted): expression kernels run per batch, so ranks with no
overflowing rows — or no data at all — never reach the kernel and the
allreduce itself deadlocks.

This is a general Bodo limitation, not decimal-specific: the same deadlock is
reproducible on mainline behavior with `SELECT CAST(S AS INT)` on a string
column whose bad rows land on only a subset of ranks
(`/tmp/dec_repro/bug2_cast4.py`, 4 rows / 8 workers), and with
`test_decimal_moment_functions_overflow` (groupby-sum overflow error) at 8
workers — both hang identically with `BODO_DISABLE_DECIMAL_INT64_FASTPATH=1`.

**Fix.** Row-level overflow in the `check_max_precision` path can never be
raised safely (it is always rank-divergent), so the array paths of
`decimal_int64_binary_op` (bodo/pandas/physical/expression.cpp callers) and
`decimal_int64_binary_op_arrays` (bodo/libs/_decimal_ext.cpp) now match the
legacy fastpath-off behavior: gandiva leaves its zero-initialized result on
overflow, the kernels store that 0, and no error is raised. Overflowing
values now produce 0 instead of hanging, and outputs are identical between
fastpath on/off. Scalar-scalar overflow is rank-uniform (scalars are
replicated) and still raises. NOTE: this means p>38 decimal arithmetic
silently produces 0 on overflow (pre-existing legacy behavior, also what
gandiva-without-check does); raising a proper error needs the general
rank-divergent-error fix (error aggregation at a rank-uniform point), which
is out of scope here.

**Verification.**
- `bodo/tests/test_decimal_int64.cpp` 9/9 with and without
  `BODO_DECIMAL_INT64_STORAGE=1`;
  `test_decimal_int64_max_precision_overflow` now covers the 2xdecimal(38,0)
  multirow case (positive and negative overflow -> 0, null operand row, and a
  non-overflowing row keeping its value) through both the array-level and
  Datum-level kernels.
- `/tmp/dec_repro/bug2.py` (the 2x38-bit repro) and `/tmp/dec_repro/bug2_partial.py`
  (100 rows, 2 overflowing, data on all ranks) complete instead of hanging;
  outputs match fastpath-off.
- 9-query battery data-identical (row-sorted) between storage on/off at BOTH
  1 and 8 workers.
- `BodoSQL/bodosql/tests/test_decimal.py` 144 passed (`BODO_NUM_WORKERS=1`;
  note the 8-worker run hangs in the pre-existing groupby-overflow
  rank-divergence described above, unrelated to this fix).
- df_lib decimal tests (`test_end_to_end.py -k decimal`) 11 passed.

## Benchmarking state (all in /home/isaac/Bodo/ab_dec64/, RESULTS.md has details)

- SF10 FS-catalog warehouses: FLBA baseline
  `~/Bodo/tpch_data/iceberg_sf10_fs`, INT64-physical rewrite
  `~/Bodo/tpch_data/iceberg_sf10_fs_int64` (pyiceberg SqlCatalog + namespace
  `TPCH`; loading it through Bodo's `FileSystemCatalog` needs
  `BODO_FS_CATALOG_SCHEMA=TPCH` and per-table
  `metadata/version-hint.text` + `v1.metadata.json` — generated by
  `/tmp/dec_repro/rewrite_warehouse.py`; beware pyiceberg commits losing
  snapshots when interleaving `set_properties` transactions with appends).
- Driver: `ab_dec64/run_ab.sh TAG [ENV=V ...]` (full suite, 1 worker),
  focused variant with warmup+n_iters inline in RESULTS.md.
- Measured (1 worker, cold, 1 iter): FLBA baseline 333.4s; FLBA + 8-byte
  storage 328.4s (broken); INT64-physical 16-byte 285.4s (1.17x, correct);
  INT64-physical + 8-byte storage 262.5s (1.27x, broken by bug 1).
- perf profile of Q1 (INT64-physical, 16-byte): `FromBigEndian` is gone from
  the top of the profile after the I/O change (was 7.7%); remaining time is
  data-plane infrastructure (groupby machinery, string keys, BLAS threads).

## Correctness gates to run before promoting anything

1. `pixi run mpiexec -n 1 pytest bodo/tests/test_decimal_int64.cpp` (9 tests;
   with and without `BODO_DECIMAL_INT64_STORAGE=1`).
2. `/tmp/dec_repro/repro.py` battery (9 operator queries) — outputs must be
   data-identical (row-sorted frame equality) between storage on/off, at BOTH
   1 and 8 workers. Note the 8-worker run needs the heap to survive; glibc
   aborts are the signal for bug 1.
3. Full SF10 battery: `ab_dec64/run_ab.sh` with and without the storage flag;
   outputs must be data-identical to the FLBA baseline (md5 of files is NOT
   reliable; compare row-sorted frames).
4. `BodoSQL/bodosql/tests/test_decimal.py` and the df_lib decimal tests in
   `bodo/tests/test_df_lib/test_end_to_end.py`.
