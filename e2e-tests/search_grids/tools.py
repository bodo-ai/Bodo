#!/usr/bin/env python3
# coding: utf-8
# original code is here https://github.com/Bodo-inc/search_grids

import math

import numpy as np
import pandas as pd

import bodo

# We specify distributed=False for these utility functions since they are called
# inside a parallel loop (prange) (in search_bodo.py), so should be distributed
# (otherwise MPI calls hang).


@bodo.jit(distributed=False)
def perceived(price, eta=0.02):
    measure_of_unit = (
        0.01,
        0.10,
        0.25,
        0.50,
        1.00,
        5.00,
        10.00,
        20.00,
        50.00,
        100.00,
        500.00,
        1000.00,
    )
    units = [mu for mu in measure_of_unit if mu < price]
    perceived_price = 0.00
    for unit in reversed(units):
        n_unit = math.floor(price / unit)
        gamma = 1 + eta * math.log(unit / units[-1])
        perceived_price += gamma * (n_unit * unit)
        price -= n_unit * unit
    perceived_price = round(perceived_price, 2)

    return perceived_price


@bodo.jit(distributed=False)
def filter(df_pg, pg_type, strategy_set_valid_prices):
    df_pg_tmp = df_pg.copy()
    df_pg_tmp["VALID_PRICES"] = df_pg_tmp.PRICE

    mesh = pd.DataFrame({"KEY": 1, pg_type: list(df_pg[pg_type].unique())})
    tmp = pd.DataFrame({"KEY": 1, "VALID_PRICES": strategy_set_valid_prices})
    mesh = mesh.merge(tmp, on="KEY").drop(columns="KEY")

    dtmp = df_pg[[pg_type, "PRICE"]].rename(columns={"PRICE": "VALID_PRICES"})
    mesh = mesh.append(dtmp).sort_values(by=[pg_type, "VALID_PRICES"])

    df_pg = df_pg.merge(mesh, on=pg_type)

    lower_half = df_pg[
        (df_pg.P_MIN <= df_pg.VALID_PRICES)
        & (df_pg.VALID_PRICES <= df_pg.LOWER_V_PRICE)
    ]
    upper_half = df_pg[
        (df_pg.UPPER_V_PRICE <= df_pg.VALID_PRICES)
        & (df_pg.VALID_PRICES <= df_pg.P_MAX)
    ]
    df_pg = lower_half.append(upper_half)

    df_pg = df_pg.append(df_pg_tmp)
    df_pg = df_pg.reset_index(drop=True)

    return df_pg


@bodo.jit(distributed=False)
def strategic(df_pg, pg_type, strategy):
    df_pg["price_diff"] = np.abs(df_pg.VALID_PRICES - df_pg.PRICE)
    df_pg["DELTA"] = 1
    df_pg.DELTA = df_pg.DELTA.where(df_pg.price_diff > 1e-6, 0)
    US_max = df_pg[["UNITS", "US_REF"]].max(axis=1)
    PI = (df_pg.US_REF - df_pg.UNITS) / US_max
    WT = df_pg.REF_PRICE * df_pg.US_REF * PI
    PROFIT = df_pg.UNITS * (df_pg.VALID_PRICES - df_pg.COST)
    df_pg["SV"] = PROFIT - strategy * WT - df_pg.ACTIVITY_COST * df_pg.DELTA

    cols = [pg_type, "VALID_PRICES", "SV"]
    df_pg = df_pg[cols].groupby([pg_type, "VALID_PRICES"]).sum().reset_index()

    return df_pg


@bodo.jit(distributed=False)
def unit(df_pg):
    df_pg["Perc_Price"] = df_pg.VALID_PRICES.apply(perceived)
    Perc_Ref_Price = df_pg.REF_PRICE.apply(perceived)

    x = (df_pg.Perc_Price - df_pg.util) / df_pg.REF_PRICE
    x = np.exp(-df_pg.eps * x)
    theta = x / (1 - x) ** 2
    df_pg["UNITS"] = df_pg.Q_is * theta

    return df_pg


@bodo.jit(distributed=False)
def search(df_opt, idx, pg_type, strategy_set_valid_prices):
    cat, strategy = idx[0], idx[1]
    df_pg = df_opt[df_opt.CAT_L3 == cat].copy()

    cols = [pg_type, "PRICE", "PRICE_LOCK"]
    df_rec = (
        df_pg[df_pg.PRICE_LOCK == True][cols]
        .rename(columns={"PRICE": "VALID_PRICES"})
        .drop_duplicates()
    )

    df_pg = df_pg[df_pg.PRICE_LOCK == False]
    if len(df_pg) > 0:
        df_pg = filter(df_pg, pg_type, strategy_set_valid_prices)
        df_pg = unit(df_pg)
        df_pg = strategic(df_pg, pg_type, strategy)

        idxmax = df_pg.groupby(pg_type)["SV"].idxmax().values
        df_tmp = df_pg.iloc[idxmax][[pg_type, "VALID_PRICES"]]
        df_tmp["PRICE_LOCK"] = False
        df_rec = df_rec.append(df_tmp)

    df_rec = df_rec.sort_values(by=pg_type).reset_index(drop=True)
    df_rec["STRATEGY"] = strategy
    df_rec = df_rec[["STRATEGY", pg_type, "VALID_PRICES", "PRICE_LOCK"]]
    df_rec.rename(columns={"VALID_PRICES": "RECOMMENDED_PRICE"}, inplace=True)

    return df_rec
