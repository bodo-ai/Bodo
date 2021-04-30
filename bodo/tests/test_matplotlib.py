# Copyright (C) 2021 Bodo Inc. All rights reserved.
"""
    Tests for Matplotlib support inside Bodo. Matplotlib
    generally requires comparing visuals, so we write all
    results to images,
"""
import matplotlib
import numpy as np
import pytest
from matplotlib.testing.decorators import check_figures_equal

import bodo
from bodo.utils.typing import BodoError


# TODO: Replace with a more general testing framework in the future.
# TODO: Determine a reasonable value for tol
@check_figures_equal(extensions=["png"], tol=0.1)
def test_usage_example(fig_test, fig_ref):
    """
    Tests a basic example from the matplotlib user guide.
    """

    def impl(input_fig):
        x = np.linspace(0, 2, 100)
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.plot(x, x, label="linear")  # Plot some data on the axes.
        ax.plot(x, x ** 2, label="quadratic")  # Plot more data on the axes...
        ax.plot(x, x ** 3, label="cubic")  # ... and some more.
        ax.set_xlabel("x label")  # Add an x-label to the axes.
        ax.set_ylabel("y label")  # Add a y-label to the axes.
        ax.set_title("Simple Plot")  # Add a title to the axes.
        ax.legend()  # Add a legend.
        # Add a barrier so rank 0 has always written the file.
        bodo.barrier()

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@check_figures_equal(extensions=["png"], tol=0.1)
def test_usage_axes_example(fig_test, fig_ref):
    """
    Tests a basic example from the matplotlib user guide with multiple axes.
    """

    def impl(input_fig):
        x = np.linspace(0, 2, 100)
        axes = input_fig.subplots(nrows=4, ncols=2)  # Create a figure and an axes.
        axes[0][1].plot(x, x, label="linear")  # Plot some data on the axes.
        axes[1][0].plot(x, x ** 2, label="quadratic")  # Plot more data on the axes...
        axes[1][0].plot(x, x ** 3, label="cubic")  # ... and some more.
        axes[1][0].set_xlabel("x label")  # Add an x-label to the axes.
        axes[1][0].set_ylabel("y label")  # Add a y-label to the axes.
        axes[1][0].set_title("Simple Plot")  # Add a title to the axes.
        axes[1][0].legend()  # Add a legend.
        # Add a barrier so rank 0 has always written the file.
        bodo.barrier()

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@check_figures_equal(extensions=["png"], tol=0.1)
def test_usage_replicated_example(fig_test, fig_ref):
    """
    Tests a basic example from the matplotlib user guide with replicated data.
    """

    def impl(x, input_fig):
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.plot(x, x, label="linear")  # Plot some data on the axes.
        ax.plot(x, x ** 2, label="quadratic")  # Plot more data on the axes...
        ax.plot(x, x ** 3, label="cubic")  # ... and some more.
        ax.set_xlabel("x label")  # Add an x-label to the axes.
        ax.set_ylabel("y label")  # Add a y-label to the axes.
        ax.set_title("Simple Plot")  # Add a title to the axes.
        ax.legend()  # Add a legend.
        # Add a barrier so rank 0 has always written the file.
        bodo.barrier()

    x = np.linspace(0, 2, 100)
    impl(x, fig_ref)
    bodo.jit(impl)(x, fig_test)


def test_mpl_subplots_const_error(memory_leak_check):
    """
    Tests that subplots requires constants for nrows and ncols.
    """

    def impl1():
        return matplotlib.pyplot.subplots(0, 1)

    def impl2():
        return matplotlib.pyplot.subplots(1, 0)

    def impl3(lst):
        return matplotlib.pyplot.subplots(lst[3], 1)

    def impl4(lst):
        return matplotlib.pyplot.subplots(1, lst[3])

    err_msg = "matplotlib.pyplot.subplots.* must be a constant integer >= 1"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl1)()
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl2)()

    lst = [-1, 2, 5, 1]
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl3)(lst)
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl4)(lst)


def test_fig_subplots_const_error(memory_leak_check):
    """
    Tests that subplots requires constants for nrows and ncols.
    """

    def impl1(fig):
        return fig.subplots(0, 1)

    def impl2(fig):
        return fig.subplots(1, 0)

    def impl3(fig, lst):
        return fig.subplots(lst[3], 1)

    def impl4(fig, lst):
        return fig.subplots(1, lst[3])

    fig = matplotlib.pyplot.gcf()

    err_msg = "matplotlib.figure.Figure.subplots.* must be a constant integer >= 1"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl1)(fig)
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl2)(fig)

    lst = [-1, 2, 5, 1]
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl3)(fig, lst)
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl4)(fig, lst)
