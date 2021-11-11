.. _data_visualization:

Data Visualization
------------------
Bodo supports Matplotlib visualization natively inside JIT functions.
This section specifies the supported Matplotlib APIs and classes.
In general, these APIs support all arguments except for the restrictions specified in each section.

Plotting APIs
~~~~~~~~~~~~~

Currently, Bodo automatically supports the following plotting APIs.

  * :func:`matplotlib.pyplot.plot`
  * :func:`matplotlib.pyplot.scatter`
  * :func:`matplotlib.pyplot.bar`
  * :func:`matplotlib.pyplot.contour`
  * :func:`matplotlib.pyplot.contourf`
  * :func:`matplotlib.pyplot.quiver`
  * :func:`matplotlib.pyplot.pie` (``autopct`` must be a constant boolean or omitted)
  * :func:`matplotlib.pyplot.fill`
  * :func:`matplotlib.pyplot.fill_between`
  * :func:`matplotlib.pyplot.step`
  * :func:`matplotlib.pyplot.errorbar`
  * :func:`matplotlib.pyplot.barbs`
  * :func:`matplotlib.pyplot.eventplot`
  * :func:`matplotlib.pyplot.hexbin`
  * :func:`matplotlib.pyplot.xcorr` (``autopct`` must be a constant boolean or omitted)
  * :func:`matplotlib.pyplot.imshow`
  * :func:`matplotlib.pyplot.plot`
  * :func:`matplotlib.pyplot.scatter`
  * :func:`matplotlib.pyplot.bar`
  * :meth:`matplotlib.axes.Axes.contour`
  * :meth:`matplotlib.axes.Axes.contourf`
  * :meth:`matplotlib.axes.Axes.quiver`
  * :meth:`matplotlib.axes.Axes.pie` (``usevlines`` must be a constant boolean or omitted)
  * :meth:`matplotlib.axes.Axes.fill`
  * :meth:`matplotlib.axes.Axes.fill_between`
  * :meth:`matplotlib.axes.Axes.step`
  * :meth:`matplotlib.axes.Axes.errorbar`
  * :meth:`matplotlib.axes.Axes.barbs`
  * :meth:`matplotlib.axes.Axes.eventplot`
  * :meth:`matplotlib.axes.Axes.hexbin`
  * :meth:`matplotlib.axes.Axes.xcorr` (``usevlines`` must be a constant boolean or omitted)
  * :meth:`matplotlib.axes.Axes.imshow`


These APIs have the following restrictions:

  * The data being plotted must be Numpy arrays and not Pandas data structures.
  * Use of lists is not currently supported. If you need to plot multiple arrays
    use a tuple or a 2D Numpy array.

These functions work by automatically gathering all of the
data onto one machine and then plotting the data. If there is not enough
memory on your machine, a sample of the data can be selected. The
example code below demonstrates calling plot with a sample of the data:

.. code:: ipython3

    import matplotlib.pyplot as plt

    %matplotlib inline

    @bodo.jit
    def dist_plot(n):
        X = np.arange(n)
        Y = np.exp(-X/3.0)
        plt.plot(X[::10], Y[::10]) # gather every 10th element
        plt.show()

    dist_plot(100)



.. parsed-literal::

    [output:0]


.. image:: ../bodo_tutorial_files/bodo_tutorial_83_1.png
   :align: center

Formatting APIs
~~~~~~~~~~~~~~~
In addition to plotting, we also support a variety of formatting APIs to modify your figures.

  * :func:`matplotlib.pyplot.gca`
  * :func:`matplotlib.pyplot.gcf`
  * :func:`matplotlib.pyplot.text`
  * :func:`matplotlib.pyplot.subplots` (``nrows`` and ``ncols`` must be constant integers)
  * :func:`matplotlib.pyplot.suptitle`
  * :func:`matplotlib.pyplot.tight_layout`
  * :func:`matplotlib.pyplot.savefig`
  * :func:`matplotlib.pyplot.draw`
  * :func:`matplotlib.pyplot.show` (Output is only displayed on rank 0)
  * :meth:`matplotlib.figure.Figure.suptitle`
  * :meth:`matplotlib.figure.Figure.tight_layout`
  * :meth:`matplotlib.figure.Figure.subplots` (``nrows`` and ``ncols`` must be constant integers)
  * :meth:`matplotlib.figure.Figure.show` (Output is only displayed on rank 0)
  * :meth:`matplotlib.axes.Axes.annotate`
  * :meth:`matplotlib.axes.Axes.text`
  * :meth:`matplotlib.axes.Axes.set_xlabel`
  * :meth:`matplotlib.axes.Axes.set_ylabel`
  * :meth:`matplotlib.axes.Axes.set_xscale`
  * :meth:`matplotlib.axes.Axes.set_yscale`
  * :meth:`matplotlib.axes.Axes.set_xticklabels`
  * :meth:`matplotlib.axes.Axes.set_yticklabels`
  * :meth:`matplotlib.axes.Axes.set_xlim`
  * :meth:`matplotlib.axes.Axes.set_ylim`
  * :meth:`matplotlib.axes.Axes.set_xticks`
  * :meth:`matplotlib.axes.Axes.set_yticks`
  * :meth:`matplotlib.axes.Axes.set_axis_on`
  * :meth:`matplotlib.axes.Axes.set_axis_off`
  * :meth:`matplotlib.axes.Axes.draw`
  * :meth:`matplotlib.axes.Axes.set_title`
  * :meth:`matplotlib.axes.Axes.legend`
  * :meth:`matplotlib.axes.Axes.grid`




In general these APIs support all arguments except for the restrictions specified.
In addition, APIs have the following restrictions:

    * Use of lists is not currently supported. If you need to provide a list, please use a tuple
      instead.
    * Formatting functions execute on all ranks by default. If you need to execute further Matplotlib
      code on all of your processes, please close any figures you opened inside Bodo.


.. _matplotlib_classes:

Matplotlib Classes
~~~~~~~~~~~~~~~~~~
Bodo supports the following Matplotlib classes when used with
the previously mentioned APIs:

  * :class:`matplotlib.figure.Figure`
  * :class:`matplotlib.axes.Axes`
  * :class:`matplotlib.text.Text`
  * :class:`matplotlib.text.Annotation`
  * :class:`matplotlib.lines.Line2D`
  * :class:`matplotlib.collections.PathCollection`
  * :class:`matplotlib.container.BarContainer`
  * :class:`matplotlib.contour.QuadContourSet`
  * :class:`matplotlib.quiver.Quiver`
  * :class:`matplotlib.patches.Wedge`
  * :class:`matplotlib.patches.Polygon`
  * :class:`matplotlib.collections.PolyCollection`
  * :class:`matplotlib.image.AxesImage`
  * :class:`matplotlib.container.ErrorbarContainer`
  * :class:`matplotlib.quiver.Barbs`
  * :class:`matplotlib.collections.EventCollection`
  * :class:`matplotlib.collections.LineCollection`


Working with Unsupported APIs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For other visualization functions, you can call them from regular Python and manually gather the data.
If the data does not fit in a single machine's memory, you may need to take a sample. The example code below demonstrates
gathering a portion of data in Bodo and calling polar (which Bodo doesn't support yet)
in regular Python::

    import bodo
    import numpy as np
    import matplotlib.pyplot as plt

    @bodo.jit()
    def dist_gather_test(n):
        X = np.arange(n)
        Y = 3 - np.cos(X)
        return bodo.gatherv(X[::10]), bodo.gatherv(Y[::10])  # gather every 10th element

    X_Sample, Y_Sample = dist_gather_test(1000)
    if bodo.get_rank() == 0:
        plt.polar(X_Sample, Y_Sample)
        plt.show()
