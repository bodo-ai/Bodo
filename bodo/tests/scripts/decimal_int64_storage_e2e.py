"""Differential smoke test for BODO_DECIMAL_INT64_STORAGE (8-byte decimal
storage in the C++ backend data plane).

Runs a set of decimal-heavy operations through the DataFrame library twice,
once with 8-byte storage enabled and once with the default 16-byte storage,
and compares the outputs element-wise (including nulls).

Usage (no mpiexec, spawn mode):
    BODO_NUM_WORKERS=2 python bodo/tests/scripts/decimal_int64_storage_e2e.py
"""

import os
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa

import bodo.pandas as bd

mode = os.environ.get("BODO_DECIMAL_INT64_STORAGE", "0")
out_path = os.environ.get("DECIMAL_E2E_OUT")
if out_path is None:
    out_path = f"/tmp/decimal_e2e_{mode}.txt"

rng = np.random.default_rng(42)


def make_decimal_series(precision, scale, n, with_nulls=True):
    q = Decimal(10) ** scale
    vals = []
    for _ in range(n):
        if with_nulls and rng.random() < 0.15:
            vals.append(None)
        else:
            v = int(
                rng.integers(
                    -(10 ** (precision - scale - 1)), 10 ** (precision - scale - 1)
                )
            )
            vals.append(Decimal(v) / q)
    return pd.Series(vals, dtype=pd.ArrowDtype(pa.decimal128(precision, scale)))


n = 200
df = pd.DataFrame(
    {
        "d15_2": make_decimal_series(15, 2, n),
        "d10_0": make_decimal_series(10, 0, n),
        "d20_4": make_decimal_series(20, 4, n),  # p > 18: stays 16-byte
        "k": pd.Series(rng.integers(0, 8, n), dtype="int32"),
    }
)
bdf = bd.from_pandas(df)

results = []

# Comparisons and filters
results.append(("cmp", bdf[bdf.d15_2 > Decimal("0.5")].d15_2.execute_plan()))
results.append(("cmp_mixed", bdf[bdf.d15_2 > bdf.d20_4].k.execute_plan()))

# Arithmetic producing narrow results
results.append(("add", (bdf.d15_2 + bdf.d10_0).execute_plan()))
results.append(("sub", (bdf.d15_2 - Decimal("1.5")).execute_plan()))
results.append(("mul_narrow", (bdf.d10_0 * Decimal("3")).execute_plan()))

# Wide result (stays 16-byte)
results.append(("mul_wide", (bdf.d15_2 * bdf.d15_2).execute_plan()))

# Groupby aggregations
results.append(("grp_sum", bdf.groupby("k").d15_2.sum().execute_plan()))
results.append(("grp_mean", bdf.groupby("k").d15_2.mean().execute_plan()))
results.append(("grp_min", bdf.groupby("k").d15_2.min().execute_plan()))
results.append(("grp_max", bdf.groupby("k").d15_2.max().execute_plan()))
results.append(("grp_count", bdf.groupby("k").d15_2.count().execute_plan()))

# Group by the decimal column itself (hash/equality on 8-byte keys)
results.append(("grp_by_dec", bdf.groupby("d10_0").k.sum().execute_plan()))

# Lead/lag and sort preserve values
results.append(("shift", bdf.d15_2.shift(1).execute_plan()))
results.append(
    ("sort", bdf.sort_values("d15_2").d15_2.execute_plan().reset_index(drop=True))
)

with open(out_path, "w") as f:
    for name, out in results:
        f.write(f"==== {name} ====\n")
        f.write(out.to_string() + "\n")

print(f"WROTE {out_path}")
