"""
Tests for Matplotlib support inside Bodo. Matplotlib
generally requires comparing visuals, so we write all
results to images,
"""

from .test_matplotlib import bodo_check_figures_equal

try:
    import matplotlib  # noqa: F401  # pragma: no cover
    from matplotlib.testing.decorators import (
        check_figures_equal,  # noqa: F401  # pragma: no cover
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
@bodo_check_figures_equal(tol=0.1)
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
@bodo_check_figures_equal(tol=0.1)
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
@bodo_check_figures_equal(tol=0.1)
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
@bodo_check_figures_equal(tol=0.1)
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
@bodo_check_figures_equal(tol=0.1)
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
@bodo_check_figures_equal(tol=0.1)
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
@bodo_check_figures_equal(tol=0.1)
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
@bodo_check_figures_equal(tol=0.1)
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
@bodo_check_figures_equal(tol=0.1)
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
@bodo_check_figures_equal(tol=0.1)
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
@bodo_check_figures_equal(tol=0.1)
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
@bodo_check_figures_equal(tol=0.1)
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
@bodo_check_figures_equal(tol=0.1)
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
@bodo_check_figures_equal(tol=0.1)
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
@bodo_check_figures_equal(tol=0.1)
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


@pytest.mark.slow
def test_df_plot_args(memory_leak_check):
    """
    Error checking for types/values of df.plot supported arguments
    """
    from bodo.utils.typing import BodoError

    def impl1():
        x = np.arange(100)
        y = np.arange(100)
        df = pd.DataFrame({"A": x, "B": y})
        df.plot(x=1.2, y="B")

    err_msg = "x must be a constant column name, constant integer, or None"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl1)()

    err_msg = "x must be a constant column name, constant integer, or None"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl1)()

    def impl2():
        x = np.arange(100)
        y = np.arange(100)
        df = pd.DataFrame({"A": x, "B": y})
        df.plot(y="m")

    err_msg = "column not found"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl2)()

    def impl3():
        x = np.arange(100)
        y = np.arange(100)
        df = pd.DataFrame({"A": x, "B": y})
        df.plot(x="A", y=12)

    err_msg = "is out of bounds"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl3)()

    def impl4():
        x = np.arange(100)
        y = np.arange(100)
        df = pd.DataFrame({"A": x, "B": y})
        df.plot(kind="pie")

    err_msg = "pie plot is not supported"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl4)()

    def impl5():
        x = np.arange(100)
        y = np.arange(100)
        df = pd.DataFrame({"A": x, "B": y})
        df.plot(figsize=10)

    err_msg = "figsize must be a constant numeric tuple"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl5)()

    def impl6():
        x = np.arange(100)
        y = np.arange(100)
        df = pd.DataFrame({"A": x, "B": y})
        df.plot(title=True)

    err_msg = "title must be a constant string"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl6)()

    def impl7():
        x = np.arange(100)
        y = np.arange(100)
        df = pd.DataFrame({"A": x, "B": y})
        df.plot(legend="X")

    err_msg = "legend must be a boolean type"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl7)()

    def impl8():
        x = np.arange(100)
        y = np.arange(100)
        df = pd.DataFrame({"A": x, "B": y})
        df.plot(legend="X")

    err_msg = "legend must be a boolean type"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl8)()

    def impl9():
        x = np.arange(100)
        y = np.arange(100)
        df = pd.DataFrame({"A": x, "B": y})
        df.plot(xticks=3)

    err_msg = "xticks must be a constant tuple"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl9)()

    def impl10():
        x = np.arange(100)
        y = np.arange(100)
        df = pd.DataFrame({"A": x, "B": y})
        df.plot(yticks=2)

    err_msg = "yticks must be a constant tuple"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl10)()

    def impl11():
        x = np.arange(100)
        y = np.arange(100)
        df = pd.DataFrame({"A": x, "B": y})
        df.plot(fontsize=3.4)

    err_msg = "fontsize must be an integer"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl11)()

    def impl12():
        x = np.arange(100)
        y = np.arange(100)
        df = pd.DataFrame({"A": x, "B": y})
        df.plot(xlabel=10)

    err_msg = "xlabel must be a constant string"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl12)()

    def impl13():
        x = np.arange(100)
        y = np.arange(100)
        df = pd.DataFrame({"A": x, "B": y})
        df.plot(ylabel=10)

    err_msg = "ylabel must be a constant string"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl13)()

    def impl14():
        x = np.arange(100)
        y = np.arange(100)
        df = pd.DataFrame({"A": x, "B": y})
        df.plot(kind="scatter")

    err_msg = "requires an x and y column"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl14)()

    def impl15():
        x = np.arange(100)
        y = np.arange(100)
        df = pd.DataFrame({"A": x, "B": y})
        df.plot(kind="scatter", y="B")

    err_msg = "x column is missing."
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl15)()

    def impl16():
        x = np.arange(100)
        y = np.arange(100)
        df = pd.DataFrame({"A": x, "B": y})
        df.plot(kind="scatter", x="A")

    err_msg = "y column is missing."
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl16)()
