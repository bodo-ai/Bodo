# Copyright (C) 2022 Bodo Inc. All rights reserved.
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
        x = np.arange(10)
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.plot(x, x, label="linear")  # Plot some data on the axes.
        ax.plot(x, x**2, label="quadratic")  # Plot more data on the axes...
        ax.plot(x, x**3, label="cubic")  # ... and some more.
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
        x = np.arange(10)
        axes = input_fig.subplots(nrows=4, ncols=2)  # Create a figure and an axes.
        axes[0][1].plot(x, x, label="linear")  # Plot some data on the axes.
        axes[1][0].plot(x, x**2, label="quadratic")  # Plot more data on the axes...
        axes[1][0].plot(x, x**3, label="cubic")  # ... and some more.
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
        ax.plot(x, x**2, label="quadratic")  # Plot more data on the axes...
        ax.plot(x, x**3, label="cubic")  # ... and some more.
        ax.set_xlabel("x label")  # Add an x-label to the axes.
        ax.set_ylabel("y label")  # Add a y-label to the axes.
        ax.set_title("Simple Plot")  # Add a title to the axes.
        ax.legend()  # Add a legend.

    x = np.arange(10)
    impl(x, fig_ref)
    bodo.jit(impl)(x, fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_scatter_example(fig_test, fig_ref):
    """
    Tests a basic example of scatter with distributed data.
    """

    def impl(input_fig):
        x = np.arange(50)
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.scatter(x, x, alpha=0.2)  # Plot some data on the axes.

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_scatter_replicated_example(fig_test, fig_ref):
    """
    Tests a basic example of scatter with replicated data.
    """

    def impl(x, input_fig):
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.scatter(x, x, alpha=0.2)  # Plot some data on the axes.

    x = np.arange(50)
    impl(x, fig_ref)
    bodo.jit(impl)(x, fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_bar_example(fig_test, fig_ref):
    """
    Tests a basic example of bar with distributed data.
    """

    def impl(input_fig):
        x = np.arange(50)
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.bar(x, x, width=0.4)  # Plot some data on the axes.

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_bar_replicated_example(fig_test, fig_ref):
    """
    Tests a basic example of bar with replicated data.
    """

    def impl(x, input_fig):
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.bar(x, x, width=0.4)  # Plot some data on the axes.

    x = np.arange(50)
    impl(x, fig_ref)
    bodo.jit(impl)(x, fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_contour_example(fig_test, fig_ref):
    """
    Tests a basic example of contour with distributed data.
    """

    def impl(input_fig):
        z = np.arange(50).reshape(5, 10)
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.contour(z)  # Plot some data on the axes.

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_contour_replicated_example(fig_test, fig_ref):
    """
    Tests a basic example of contour with replicated data.
    """

    def impl(x, input_fig):
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.contour(x)  # Plot some data on the axes.

    z = np.arange(50).reshape(5, 10)
    impl(z, fig_ref)
    bodo.jit(impl)(z, fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_contourf_example(fig_test, fig_ref):
    """
    Tests a basic example of contourf with distributed data.
    """

    def impl(input_fig):
        z = np.arange(50).reshape(5, 10)
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.contourf(z)  # Plot some data on the axes.

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_contourf_replicated_example(fig_test, fig_ref):
    """
    Tests a basic example of contourf with replicated data.
    """

    def impl(x, input_fig):
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.contourf(z)  # Plot some data on the axes.

    z = np.arange(50).reshape(5, 10)
    impl(z, fig_ref)
    bodo.jit(impl)(z, fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_quiver_example(fig_test, fig_ref):
    """
    Tests a basic example of quiver with distributed data.
    """

    def impl(input_fig):
        x = np.arange(50)
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.quiver(x, x)  # Plot some data on the axes.

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_quiver_replicated_example(fig_test, fig_ref):
    """
    Tests a basic example of quiver with replicated data.
    """

    def impl(x, input_fig):
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.quiver(x, x)  # Plot some data on the axes.

    x = np.arange(50)
    impl(x, fig_ref)
    bodo.jit(impl)(x, fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_pie_example(fig_test, fig_ref):
    """
    Tests a basic example of pie with distributed data.
    """

    def impl(input_fig):
        x = np.arange(50)
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.pie(x)  # Plot some data on the axes.

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_pie_replicated_example(fig_test, fig_ref):
    """
    Tests a basic example of pie with replicated data.
    """

    def impl(x, input_fig):
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.pie(x)  # Plot some data on the axes.

    x = np.arange(50)
    impl(x, fig_ref)
    bodo.jit(impl)(x, fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_fill_example(fig_test, fig_ref):
    """
    Tests a basic example of fill with distributed data.
    """

    def impl(input_fig):
        # Allocate the array manually because np.hstack doesn't preserve order
        # yet in Bodo.
        x = np.empty(20, dtype=np.int64)
        y = np.empty(20, dtype=np.int64)
        for i in range(2):
            for j in range(10):
                x[10 * i + j] = j
                y[10 * i + j] = i
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.fill(x, y, "b")  # Plot some data on the axes.

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_fill_replicated_example(fig_test, fig_ref):
    """
    Tests a basic example of fill with replicated data.
    """

    def impl(x, y, input_fig):
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.fill(x, y, "b")  # Plot some data on the axes.

    x = np.hstack((np.arange(10), np.arange(10)))
    y = np.hstack((np.full(10, 0, dtype=np.int64), np.full(10, 1, dtype=np.int64)))
    impl(x, y, fig_ref)
    bodo.jit(impl)(x, y, fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_fill_between_example(fig_test, fig_ref):
    """
    Tests a basic example of fill_between with distributed data.
    """

    def impl(input_fig):
        x = np.arange(50)
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.fill_between(x, x)  # Plot some data on the axes.

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_fill_between_replicated_example(fig_test, fig_ref):
    """
    Tests a basic example of fill_between with replicated data.
    """

    def impl(x, input_fig):
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.fill_between(x, x)  # Plot some data on the axes.

    x = np.arange(50)
    impl(x, fig_ref)
    bodo.jit(impl)(x, fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_step_example(fig_test, fig_ref):
    """
    Tests a basic example of step with distributed data.
    """

    def impl(input_fig):
        x = np.arange(50)
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.step(x, x)  # Plot some data on the axes.

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_step_replicated_example(fig_test, fig_ref):
    """
    Tests a basic example of step with replicated data.
    """

    def impl(x, input_fig):
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.step(x, x)  # Plot some data on the axes.

    x = np.arange(50)
    impl(x, fig_ref)
    bodo.jit(impl)(x, fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_errorbar_example(fig_test, fig_ref):
    """
    Tests a basic example of errorbar with distributed data.
    """

    def impl(input_fig):
        x = np.arange(50)
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.errorbar(x, x)  # Plot some data on the axes.

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_errorbar_replicated_example(fig_test, fig_ref):
    """
    Tests a basic example of errorbar with replicated data.
    """

    def impl(x, input_fig):
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.errorbar(x, x)  # Plot some data on the axes.

    x = np.arange(50)
    impl(x, fig_ref)
    bodo.jit(impl)(x, fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_barbs_example(fig_test, fig_ref):
    """
    Tests a basic example of barbs with distributed data.
    """

    def impl(input_fig):
        x = np.arange(50)
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.barbs(x, x)  # Plot some data on the axes.

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_barbs_replicated_example(fig_test, fig_ref):
    """
    Tests a basic example of barbs with replicated data.
    """

    def impl(x, input_fig):
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.barbs(x, x)  # Plot some data on the axes.

    x = np.arange(50)
    impl(x, fig_ref)
    bodo.jit(impl)(x, fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_eventplot_example(fig_test, fig_ref):
    """
    Tests a basic example of eventplot with distributed data.
    """

    def impl(input_fig):
        x = np.arange(50)
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.eventplot(x)  # Plot some data on the axes.

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_eventplot_replicated_example(fig_test, fig_ref):
    """
    Tests a basic example of eventplot with replicated data.
    """

    def impl(x, input_fig):
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.eventplot(x)  # Plot some data on the axes.

    x = np.arange(50)
    impl(x, fig_ref)
    bodo.jit(impl)(x, fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_hexbin_example(fig_test, fig_ref):
    """
    Tests a basic example of hexbin with distributed data.
    """

    def impl(input_fig):
        x = np.arange(50)
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.hexbin(x, x)  # Plot some data on the axes.

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_hexbin_replicated_example(fig_test, fig_ref):
    """
    Tests a basic example of hexbin with replicated data.
    """

    def impl(x, input_fig):
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.hexbin(x, x)  # Plot some data on the axes.

    x = np.arange(50)
    impl(x, fig_ref)
    bodo.jit(impl)(x, fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_imshow_example(fig_test, fig_ref):
    """
    Tests a basic example of imshow with distributed data.
    """

    def impl(input_fig):
        x = np.arange(10000).reshape(100, 100)
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.imshow(x)  # Plot some data on the axes.

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_imshow_replicated_example(fig_test, fig_ref):
    """
    Tests a basic example of imshow with replicated data.
    """

    def impl(x, input_fig):
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.imshow(x)  # Plot some data on the axes.

    x = np.arange(10000).reshape(100, 100)
    impl(x, fig_ref)
    bodo.jit(impl)(x, fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_xcorr_example(fig_test, fig_ref):
    """
    Tests a basic example of xcorr with distributed data.
    """

    def impl(input_fig):
        x = np.arange(100).astype(np.float64)
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.xcorr(x, x)  # Plot some data on the axes.

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_xcorr_example_usevlines_false(fig_test, fig_ref):
    """
    Tests a basic example of xcorr with distributed data and usevlines=False.
    """

    def impl(input_fig):
        x = np.arange(100).astype(np.float64)
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.xcorr(x, x, usevlines=False)  # Plot some data on the axes.

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_xcorr_replicated_example(fig_test, fig_ref):
    """
    Tests a basic example of imshow with replicated data.
    """

    def impl(x, input_fig):
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.xcorr(x, x)  # Plot some data on the axes.

    x = np.arange(100).astype(np.float64)
    impl(x, fig_ref)
    bodo.jit(impl)(x, fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_text_example(fig_test, fig_ref):
    """
    Tests a basic example of text.
    """

    def impl(input_fig):
        ax = input_fig.subplots()  # Create a figure and an axes.
        ax.text(0.5, 0.5, "Bodo.ai")  # Plot some data on the axes.

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_axes_set_off(fig_test, fig_ref):
    """
    Tests a series of axes functions followed by turning
    the axis off.
    """

    def impl(input_fig):
        x = np.arange(10)
        ax = input_fig.subplots()
        ax.plot(x, x, label="linear")
        ax.plot(x, x**2, label="quadratic")
        ax.plot(x, x**3, label="cubic")
        ax.set_xscale("symlog")
        ax.set_yscale("log")
        ax.set_xlim(left=1.0, right=2.2)
        ax.set_ylim(1.0, top=10.0)
        ax.set_xticks((1.0, 1.2, 1.4), minor=False)
        ax.set_xticklabels(("Ready", "set", "go"))
        ax.set_yticks((1.1, 2.2, 7.4), minor=True)
        ax.set_yticklabels(("Fighting", "Irish"))
        ax.grid(True, which="major", axis="x")
        ax.annotate("BigPoint", (1, 1))
        ax.set_axis_off()
        input_fig.suptitle("Go Team!")
        input_fig.tight_layout(pad=2.0)

    impl(fig_ref)
    bodo.jit(impl)(fig_test)


@bodo_check_figures_equal(extensions=["png"], tol=0.1)
def test_axes_set_on(fig_test, fig_ref):
    """
    Tests a series of axes functions followed by turning
    the axis off and then on. This should maintain all of
    the original changes.
    """

    def impl(input_fig):
        x = np.arange(10)
        ax = input_fig.subplots()
        ax.plot(x, x, label="linear")
        ax.plot(x, x**2, label="quadratic")
        ax.plot(x, x**3, label="cubic")
        ax.set_xscale("symlog")
        ax.set_yscale("log")
        ax.set_xlim(left=1.0, right=2.2)
        ax.set_ylim(1.0, top=10.0)
        ax.set_xticks((1.0, 1.2, 1.4), minor=False)
        ax.set_xticklabels(("Ready", "set", "go"))
        ax.set_yticks((1.1, 2.2, 7.4), minor=True)
        ax.set_yticklabels(("Fighting", "Irish"))
        ax.grid(True, which="major", axis="x")
        ax.annotate("BigPoint", (1, 1))
        ax.set_axis_off()
        ax.set_axis_on()
        input_fig.suptitle("Go Team!")
        input_fig.tight_layout(pad=2.0)

    impl(fig_ref)

    bodo.jit(impl)(fig_test)


def test_mpl_xcorr_errors(memory_leak_check):
    """
    Tests that xcorr requires a constant boolean input.
    """

    def impl1():
        x = np.arange(100)
        ax = matplotlib.pyplot.gca()
        return ax.xcorr(x, x, usevlines=None)

    def impl2():
        x = np.arange(100)
        return matplotlib.pyplot.xcorr(x, x, usevlines=None)

    def impl3(lst):
        x = np.arange(100)
        ax = matplotlib.pyplot.gca()
        return ax.xcorr(x, x, usevlines=lst[0])

    def impl4(lst):
        x = np.arange(100)
        return matplotlib.pyplot.xcorr(x, x, usevlines=lst[0])

    err_msg = "xcorr.*: usevlines must be a constant boolean"
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl1)()
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl2)()

    lst = [False, True, True, False]
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl3)(lst)
    with pytest.raises(BodoError, match=err_msg):
        bodo.jit(impl4)(lst)


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
