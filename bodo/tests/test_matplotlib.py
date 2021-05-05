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


def bodo_check_figures_equal(*, extensions=("png", "pdf", "svg"), tol=0):
    """
    Bodo decorate around check_figures_equal that only compares
    values on rank 0.

    Example usage: @bodo_check_figures_equal(extensions=["png"], tol=0.1)
    """
    if bodo.get_rank() == 0:
        return check_figures_equal(extensions=extensions, tol=tol)
    else:
        # If we aren't on rank 0, we want to run the same code but not
        # generate any files
        def decorator(func):
            def wrapper(*args, request, **kwargs):
                # Generate a fake fig_test and fig_ref to match mpl decorator
                # behavior
                fig_test = matplotlib.pyplot.figure()
                fig_ref = matplotlib.pyplot.figure()
                # Wrap the function call in a try so we can close the figures.
                try:
                    func(*args, fig_test=fig_test, fig_ref=fig_ref, **kwargs)
                finally:
                    matplotlib.pyplot.close(fig_test)
                    matplotlib.pyplot.close(fig_ref)

            return wrapper

        return decorator


# TODO: Determine a reasonable value for tol
@bodo_check_figures_equal(extensions=["png"], tol=0.1)
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

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
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

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
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
