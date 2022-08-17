#!/usr/bin/python3
# coding: utf-8

import sys

# Standard imports
import time

import numba

# Non-standard imports
import numpy as np
import pandas as pd

# Local imports
from tools import search

import bodo
from bodo import prange


@bodo.jit(cache=True)
def search_all(categories, strategies, df_prod, grids, results_file):
    t1 = time.time()
    # Since it's a simulation-type application we do a scatterv
    # here manually. This is ok since it's not a particularly
    # data intensive application.
    categories = bodo.scatterv(categories)
    strategies = bodo.scatterv(strategies)
    # The program does have communication though since
    # the parallel loop has a concatenation reduction
    # (dataframe append for each iteration).
    n_idx = len(categories)
    df_rec = pd.DataFrame()
    for i in prange(n_idx):
        idx = (categories[i], strategies[i])
        res = search(df_prod, idx, "STRATEGY_SET_MEMBER_ID", grids)
        df_rec = df_rec.append(res)

    df_rec = df_rec.sort_values(by="STRATEGY").reset_index(drop=True)
    df_rec.to_csv(results_file, index=False)
    print("Execution time: ", time.time() - t1)


if __name__ == "__main__":
    require_cache = False
    if len(sys.argv) > 3:
        require_cache = bool(sys.argv[3])
    t1 = time.time()
    bucket_name = sys.argv[1]
    results_file = sys.argv[2]

    input_file = bucket_name + "/test_set_2.csv"
    grids_file = bucket_name + "/grids.csv"

    df_prod = pd.read_csv(input_file)
    grids = pd.read_csv(grids_file).PRICE.values

    STRATEGY_MIN = -0.5
    STRATEGY_MAX = 1
    STRATEGY_STEP = 0.01
    strategy_points = np.arange(STRATEGY_MIN, STRATEGY_MAX, STRATEGY_STEP)

    list_cat = list(df_prod.CAT_L3.unique())
    index = pd.MultiIndex.from_product(
        [list_cat, strategy_points], names=["CAT", "STRATEGY"]
    )
    list_index = list(pd.DataFrame(index=index).index.unique())

    categories = np.array([v[0] for v in list_index], object)
    strategies = np.array([v[1] for v in list_index], np.float64)

    search_all(categories, strategies, df_prod, grids, results_file)

    if require_cache and isinstance(search_all, numba.core.dispatcher.Dispatcher):
        assert (
            search_all._cache_hits[search_all.signatures[0]] == 1
        ), "ERROR: Bodo did not load from cache"
