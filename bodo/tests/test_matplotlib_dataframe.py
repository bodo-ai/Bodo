# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
    Tests for Matplotlib support inside Bodo. Matplotlib
    generally requires comparing visuals, so we write all
    results to images,
"""

from .test_matplotlib import bodo_check_figures_equal

try:
    import matplotlib  # pragma: no cover
    from matplotlib.testing.decorators import (
        check_figures_equal,  # pragma: no cover
    )

    matplotlib_import_failed = False
except ImportError:
    matplotlib_import_failed = True

import numpy as np
import pandas as pd
import pytest

import bodo
from bodo.tests.utils import pytest_pandas

pytestmark = pytest.mark.skip if matplotlib_import_failed else pytest_pandas


@pytest.mark.weekly
@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_df_plot_simple(fig_test, fig_ref):
    """
    Tests a basic example for df.plot replicated.
    """

    df = pd.DataFrame(
        {"year": [2013, 2014, 2015], "sales": [7941243, 9135482, 9536887]}
    )

    def impl(input_fig, df):
        ax = input_fig.subplots()
        df.plot(x="year", y="sales", ax=ax, figsize=(10, 20))

    impl(fig_ref, df)
    bodo.jit(impl)(fig_test, df)


@pytest.mark.weekly
@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_df_plot_labels(fig_test, fig_ref):
    """
    Tests an example for df.plot with xlabel, ylabel, title
    """

    df = pd.DataFrame(
        {"year": [2013, 2014, 2015], "sales": [7941243, 9135482, 9536887]}
    )

    def impl(input_fig, df):
        ax = input_fig.subplots()
        df.plot(
            x="year",
            y="sales",
            xlabel="time",
            ylabel="revenue",
            title="Revenue Timeline",
            ax=ax,
        )

    impl(fig_ref, df)
    bodo.jit(impl)(fig_test, df)


@pytest.mark.weekly
@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_df_plot_ticks(fig_test, fig_ref):
    """
    Tests an example for df.plot with xlabel, ylabel, title
    """

    df = pd.DataFrame(
        {"year": [2013, 2014, 2015], "sales": [7941243, 9135482, 9536887]}
    )

    def impl(input_fig, df):
        ax = input_fig.subplots()
        df.plot(
            x="year",
            y="sales",
            xticks=(2010, 2012, 2016),
            yticks=(7600000, 8600000, 9600000),
            fontsize=18,
            ax=ax,
        )

    impl(fig_ref, df)
    bodo.jit(impl)(fig_test, df)


@pytest.mark.weekly
@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_df_plot_simple_scatter(fig_test, fig_ref):
    """
    Tests a basic example for df.plot(scatter)
    replicated
    """

    df = pd.DataFrame(
        {"year": [2013, 2014, 2015], "sales": [7941243, 9135482, 9536887]}
    )

    def impl(input_fig, df):
        ax = input_fig.subplots()
        df.plot(kind="scatter", x="year", y="sales", ax=ax, figsize=(5, 5))

    impl(fig_ref, df)
    bodo.jit(impl)(fig_test, df)


@pytest.mark.weekly
@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_df_plot_labels_scatter(fig_test, fig_ref):
    """
    Tests an example for df.plot(scatter) with xlabel, ylabel, title
    replicated
    """

    df = pd.DataFrame(
        {"year": [2013, 2014, 2015], "sales": [7941243, 9135482, 9536887]}
    )

    def impl(input_fig, df):
        ax = input_fig.subplots()
        df.plot(
            kind="scatter",
            x="year",
            y="sales",
            xlabel="time",
            ylabel="revenue",
            title="Revenue Timeline",
            ax=ax,
        )

    impl(fig_ref, df)
    bodo.jit(impl)(fig_test, df)


@pytest.mark.weekly
@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_df_plot_ticks_scatter(fig_test, fig_ref):
    """
    Tests an example for df.plot(scatter) with xlabel, ylabel, title
    replicated
    """

    df = pd.DataFrame(
        {"year": [2013, 2014, 2015], "sales": [7941243, 9135482, 9536887]}
    )

    def impl(input_fig, df):
        ax = input_fig.subplots()
        df.plot(
            kind="scatter",
            x="year",
            y="sales",
            xticks=(2010, 2012, 2016),
            yticks=(7600000, 8600000, 9600000),
            fontsize=18,
            ax=ax,
        )

    impl(fig_ref, df)
    bodo.jit(impl)(fig_test, df)


@pytest.mark.weekly
@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_df_plot_simple_dist(fig_test, fig_ref):
    """
    Tests a basic example for df.plot distributed.
    """

    def impl(input_fig):
        df = pd.DataFrame(
            {
                "year": np.arange(2010, 2015),
                "sales": np.arange(7000000, 9500000, 500000),
            }
        )
        ax = input_fig.subplots()
        df.plot(x="year", y="sales", ax=ax, figsize=(10, 20))

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@pytest.mark.weekly
@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_df_plot_labels_dist(fig_test, fig_ref):
    """
    Tests an example for df.plot with xlabel, ylabel, title
    distributed
    """

    def impl(input_fig):
        df = pd.DataFrame(
            {
                "year": np.arange(2010, 2015),
                "sales": np.arange(7000000, 9500000, 500000),
            }
        )
        ax = input_fig.subplots()
        df.plot(
            x="year",
            y="sales",
            xlabel="time",
            ylabel="revenue",
            title="Revenue Timeline",
            ax=ax,
        )

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@pytest.mark.weekly
@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_df_plot_ticks_dist(fig_test, fig_ref):
    """
    Tests an example for df.plot with xlabel, ylabel, title
    distributed
    """

    def impl(input_fig):
        df = pd.DataFrame(
            {
                "year": np.arange(2010, 2015),
                "sales": np.arange(7000000, 9500000, 500000),
            }
        )
        ax = input_fig.subplots()
        df.plot(
            x="year",
            y="sales",
            xticks=(2010, 2012, 2016, 2018),
            yticks=(6600000, 7600000, 8600000, 9600000),
            fontsize=18,
            ax=ax,
        )

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@pytest.mark.weekly
@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_df_plot_simple_scatter_dist(fig_test, fig_ref):
    """
    Tests a basic example for df.plot(scatter)
    distributed
    """

    def impl(input_fig):
        df = pd.DataFrame(
            {
                "year": np.arange(2010, 2015),
                "sales": np.arange(7000000, 9500000, 500000),
            }
        )
        ax = input_fig.subplots()
        df.plot(kind="scatter", x="year", y="sales", ax=ax, figsize=(5, 5))

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@pytest.mark.weekly
@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_df_plot_labels_scatter_dist(fig_test, fig_ref):
    """
    Tests an example for df.plot(scatter) with xlabel, ylabel, title
    distributed
    """

    def impl(input_fig):
        df = pd.DataFrame(
            {
                "year": np.arange(2010, 2015),
                "sales": np.arange(7000000, 9500000, 500000),
            }
        )
        ax = input_fig.subplots()
        df.plot(
            kind="scatter",
            x="year",
            y="sales",
            xlabel="time",
            ylabel="revenue",
            title="Revenue Timeline",
            ax=ax,
        )

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@pytest.mark.weekly
@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_df_plot_ticks_scatter_dist(fig_test, fig_ref):
    """
    Tests an example for df.plot(scatter) with xlabel, ylabel, title
    distributed
    """

    def impl(input_fig):
        df = pd.DataFrame(
            {
                "year": np.arange(2010, 2015),
                "sales": np.arange(7000000, 9500000, 500000),
            }
        )
        ax = input_fig.subplots()
        df.plot(
            kind="scatter",
            x="year",
            y="sales",
            xticks=(2010, 2012, 2016, 2018),
            yticks=(6600000, 7600000, 8600000, 9600000),
            fontsize=18,
            ax=ax,
        )

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@pytest.mark.weekly
@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_df_plot_x_y_none_distributed(fig_test, fig_ref):
    """
    Tests a basic example for df.plot where x and y are None.
    distributed.
    """

    def impl(input_fig):
        df = pd.DataFrame(
            {
                "year": np.arange(2010, 2015),
                "sales": np.arange(7000000, 9500000, 500000),
                "count": np.arange(1000000, 3500000, 500000),
            }
        )
        ax = input_fig.subplots()
        # df.plot(x="year", y="sales", ax=ax, figsize=(10, 20))
        df.plot(ax=ax)

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@pytest.mark.weekly
@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_df_plot_x_none_distributed(fig_test, fig_ref):
    """
    Tests a basic example for df.plot where x is None.
    distributed.
    """

    def impl(input_fig):
        df = pd.DataFrame(
            {
                "year": np.arange(2010, 2015),
                "sales": np.arange(7000000, 9500000, 500000),
                "count": np.arange(1000000, 3500000, 500000),
            }
        )
        ax = input_fig.subplots()
        df.plot(y="sales", ax=ax)

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@pytest.mark.weekly
@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_df_plot_y_none_distributed(fig_test, fig_ref):
    """
    Tests a basic example for df.plot where x is None.
    distributed.
    """

    def impl(input_fig):
        df = pd.DataFrame(
            {
                "month": ["Jan", "Feb", "March", "April", "May"],
                "year": np.arange(2010, 2015),
                "sales": np.arange(7000000, 9500000, 500000),
                "count": np.arange(1000000, 3500000, 500000),
            }
        )
        ax = input_fig.subplots()
        df.plot(x="year", ax=ax)

    impl(fig_ref)
    bodo.jit(impl)(fig_test)
