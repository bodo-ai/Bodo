# Copyright (C) 2021 Bodo Inc. All rights reserved.
"""
Implements support for matplotlib extensions such as pyplot.plot.
"""

import matplotlib
import matplotlib.pyplot as plt
import numba
import numpy as np
from numba.core import ir_utils, types
from numba.core.typing.templates import (
    AbstractTemplate,
    AttributeTemplate,
    bound_function,
    infer_global,
    signature,
)
from numba.extending import (
    NativeValue,
    box,
    infer_getattr,
    models,
    overload,
    overload_method,
    register_model,
    typeof_impl,
    unbox,
)

import bodo
from bodo.utils.typing import (
    BodoError,
    gen_objmode_func_overload,
    gen_objmode_method_overload,
    get_overload_const_int,
    is_overload_constant_int,
    raise_bodo_error,
)
from bodo.utils.utils import unliteral_all


# Define matplot lib types as void* pointers because they will be used only inside objmode
class MplFigure(types.Opaque):
    """
    Type for figure in matplotlib:
    for example fig from:
    `fig, ax = plt.subplots()`
    """

    def __init__(self):
        super(MplFigure, self).__init__(name="MplFigure")


mpl_figure_type = MplFigure()
types.mpl_figure_type = mpl_figure_type
register_model(MplFigure)(models.OpaqueModel)


@typeof_impl.register(matplotlib.figure.Figure)
def typeof_mpl_figure(val, c):
    return mpl_figure_type


class MplAxes(types.Opaque):
    """
    Type for axes in matplotlib:
    for example ax from:
    `fig, ax = plt.subplots()`
    """

    def __init__(self):
        super(MplAxes, self).__init__(name="MplAxes")


mpl_axes_type = MplAxes()
types.mpl_axes_type = mpl_axes_type
register_model(MplAxes)(models.OpaqueModel)


@typeof_impl.register(matplotlib.axes.Axes)
def typeof_mpl_axes(val, c):
    return mpl_axes_type


class MplText(types.Opaque):
    """
    Type for Text in matplotlib.
    """

    def __init__(self):
        super(MplText, self).__init__(name="MplText")


mpl_text_type = MplText()
types.mpl_text_type = mpl_text_type
register_model(MplText)(models.OpaqueModel)


@typeof_impl.register(matplotlib.text.Text)
def typeof_mpl_text(val, c):
    return mpl_text_type


class MplAnnotation(types.Opaque):
    """
    Type for Annotation in matplotlib:
    for example res from:
    `res = ax.annotate(str, point)`
    """

    def __init__(self):
        super(MplAnnotation, self).__init__(name="MplAnnotation")


mpl_annotation_type = MplAnnotation()
types.mpl_annotation_type = mpl_annotation_type
register_model(MplAnnotation)(models.OpaqueModel)


@typeof_impl.register(matplotlib.text.Annotation)
def typeof_mpl_annotation(val, c):
    return mpl_annotation_type


@unbox(MplFigure)
@unbox(MplAxes)
@unbox(MplText)
@unbox(MplAnnotation)
def unbox_mpl_obj(typ, val, c):
    # just return the Python object pointer
    c.pyapi.incref(val)
    return NativeValue(val)


@box(MplFigure)
@box(MplAxes)
@box(MplText)
@box(MplAnnotation)
def box_mpl_obj(typ, val, c):
    # just return the Python object pointer
    c.pyapi.incref(val)
    return val


def generate_matplotlib_signature(return_typ, args, kws, obj_typ=None):
    """
    Helper function for generating a signature for a matplotlib function
    that uses args and kwargs.
    """
    kws = dict(kws)
    # add dummy default value for kws to avoid errors
    arg_names = ", ".join(f"e{i}" for i in range(len(args)))
    if arg_names:
        arg_names += ", "
    kw_names = ", ".join(f"{a} = ''" for a in kws.keys())
    obj_name = "matplotlib_obj, " if obj_typ is not None else ""
    func_text = f"def mpl_stub({obj_name} {arg_names} {kw_names}):\n"
    func_text += "    pass\n"
    loc_vars = {}
    exec(func_text, {}, loc_vars)
    mpl_stub = loc_vars["mpl_stub"]
    pysig = numba.core.utils.pysignature(mpl_stub)
    arg_types = ((obj_typ,) if obj_typ is not None else ()) + args + tuple(kws.values())
    return signature(return_typ, *unliteral_all(arg_types)).replace(pysig=pysig)


def generate_axes_typing(mod_name, nrows, ncols):
    # axes can be an np.array, but we will use a tuple instead
    const_err_msg = "{}.subplots(): {} must be a constant integer >= 1"
    if not is_overload_constant_int(nrows):
        raise_bodo_error(const_err_msg.format(mod_name, "nrows"))
    if not is_overload_constant_int(ncols):
        raise_bodo_error(const_err_msg.format(mod_name, "ncols"))
    nrows_const = get_overload_const_int(nrows)
    ncols_const = get_overload_const_int(ncols)
    if nrows_const < 1:
        raise BodoError(const_err_msg.format(mod_name, "nrows"))
    if ncols_const < 1:
        raise BodoError(const_err_msg.format(mod_name, "ncols"))

    if nrows_const == 1 and ncols_const == 1:
        output_type = mpl_axes_type
    else:
        # output type is np.array, but we will use tuples instead
        if ncols_const == 1:
            row_type = mpl_axes_type
        else:
            row_type = types.Tuple([mpl_axes_type] * ncols_const)
        output_type = types.Tuple([row_type] * nrows_const)
    return output_type


# Define a signature for the plt.plot function because it uses *args and **kwargs.
@infer_global(plt.plot)
class PlotTyper(AbstractTemplate):
    def generic(self, args, kws):
        # plot doesn't return anything
        return generate_matplotlib_signature(types.none, args, kws)


# Define a signature for the plt.gca function because it uses **kwargs.
@infer_global(plt.gca)
class GCATyper(AbstractTemplate):
    def generic(self, args, kws):
        # gca returns mpl_axes_type
        return generate_matplotlib_signature(mpl_axes_type, args, kws)


# Define a signature for the plt.suptitle function because it uses **kwargs.
@infer_global(plt.suptitle)
class SuptitleTyper(AbstractTemplate):
    def generic(self, args, kws):
        # suptitle returns mpl_text_type
        return generate_matplotlib_signature(mpl_text_type, args, kws)


# Define a signature for the plt.tight_layout function because it uses **kwargs.
@infer_global(plt.tight_layout)
class TightLayoutTyper(AbstractTemplate):
    def generic(self, args, kws):
        # tight_layout doesn't return anything
        return generate_matplotlib_signature(types.none, args, kws)


# Define a signature for the plt.subplots function because it uses *args and **kwargs.
@infer_global(plt.subplots)
class SubplotsTyper(AbstractTemplate):
    def generic(self, args, kws):
        # subplots returns a tuple of figure and axes
        nrows = args[0] if len(args) > 0 else kws.get("nrows", types.literal(1))
        ncols = args[1] if len(args) > 1 else kws.get("ncols", types.literal(1))
        axes_type = generate_axes_typing("matplotlib.pyplot", nrows, ncols)

        return generate_matplotlib_signature(
            types.Tuple([mpl_figure_type, axes_type]),
            args,
            kws,
        )


SubplotsTyper._no_unliteral = True


# Define signatures for figure methods that contain kwargs
@infer_getattr
class MatplotlibFigureKwargsAttribute(AttributeTemplate):
    key = MplFigure

    @bound_function("fig.suptitle", no_unliteral=True)
    def resolve_suptitle(self, fig_typ, args, kws):
        return generate_matplotlib_signature(mpl_text_type, args, kws, obj_typ=fig_typ)

    @bound_function("fig.tight_layout", no_unliteral=True)
    def resolve_tight_layout(self, fig_typ, args, kws):
        return generate_matplotlib_signature(types.none, args, kws, obj_typ=fig_typ)


# Define signatures for axes methods that contain kwargs
@infer_getattr
class MatplotlibAxesKwargsAttribute(AttributeTemplate):
    key = MplAxes

    @bound_function("ax.annotate", no_unliteral=True)
    def resolve_annotate(self, ax_typ, args, kws):
        return generate_matplotlib_signature(types.none, args, kws, obj_typ=ax_typ)

    @bound_function("ax.grid", no_unliteral=True)
    def resolve_grid(self, ax_typ, args, kws):
        return generate_matplotlib_signature(types.none, args, kws, obj_typ=ax_typ)

    @bound_function("ax.plot", no_unliteral=True)
    def resolve_plot(self, ax_typ, args, kws):
        return generate_matplotlib_signature(types.none, args, kws, obj_typ=ax_typ)

    @bound_function("ax.set_xlabel", no_unliteral=True)
    def resolve_set_xlabel(self, ax_typ, args, kws):
        return generate_matplotlib_signature(types.none, args, kws, obj_typ=ax_typ)

    @bound_function("ax.set_xticklabels", no_unliteral=True)
    def resolve_set_xticklabels(self, ax_typ, args, kws):
        return generate_matplotlib_signature(
            types.List(mpl_text_type), args, kws, obj_typ=ax_typ
        )

    @bound_function("ax.set_yticklabels", no_unliteral=True)
    def resolve_set_yticklabels(self, ax_typ, args, kws):
        return generate_matplotlib_signature(
            types.List(mpl_text_type), args, kws, obj_typ=ax_typ
        )

    @bound_function("ax.set_ylabel", no_unliteral=True)
    def resolve_set_ylabel(self, ax_typ, args, kws):
        return generate_matplotlib_signature(types.none, args, kws, obj_typ=ax_typ)

    @bound_function("ax.set_xscale", no_unliteral=True)
    def resolve_set_xscale(self, ax_typ, args, kws):
        return generate_matplotlib_signature(types.none, args, kws, obj_typ=ax_typ)

    @bound_function("ax.set_yscale", no_unliteral=True)
    def resolve_set_yscale(self, ax_typ, args, kws):
        return generate_matplotlib_signature(types.none, args, kws, obj_typ=ax_typ)

    @bound_function("ax.set_title", no_unliteral=True)
    def resolve_set_title(self, ax_typ, args, kws):
        return generate_matplotlib_signature(types.none, args, kws, obj_typ=ax_typ)

    @bound_function("ax.legend", no_unliteral=True)
    def resolve_legend(self, ax_typ, args, kws):
        return generate_matplotlib_signature(types.none, args, kws, obj_typ=ax_typ)


@overload(plt.savefig, no_unliteral=True)
def overload_savefig(
    fname,
    dpi=None,
    facecolor="w",
    edgecolor="w",
    orientation="portrait",
    format=None,
    transparent=False,
    bbox_inches=None,
    pad_inches=0.1,
    metadata=None,
):
    """
    Overloads plt.subplots. Note we can't use gen_objmode_func_overload
    because the matplotlib implementation uses *args and **kwargs (even though
    it doesn't need to), which fails assertion checks in
    gen_objmode_func_overload.
    """
    # Note: We omit papertype and frameon because these arguments are deprecated and will be removed in 2 minor releases.
    def impl(
        fname,
        dpi=None,
        facecolor="w",
        edgecolor="w",
        orientation="portrait",
        format=None,
        transparent=False,
        bbox_inches=None,
        pad_inches=0.1,
        metadata=None,
    ):  # pragma: no cover
        with bodo.objmode():
            plt.savefig(
                fname=fname,
                dpi=dpi,
                facecolor=facecolor,
                edgecolor=edgecolor,
                orientation=orientation,
                format=format,
                transparent=transparent,
                bbox_inches=bbox_inches,
                pad_inches=pad_inches,
                metadata=metadata,
            )

    return impl


@overload_method(MplFigure, "subplots", no_unliteral=True)
def overload_subplots(
    fig,
    nrows=1,
    ncols=1,
    sharex=False,
    sharey=False,
    squeeze=True,
    subplot_kw=None,
    gridspec_kw=None,
):
    """
    Overloads fig.subplots. Note we can't use gen_objmode_method_overload
    because the output type depends on nrows and ncols.
    """
    axes_type = generate_axes_typing("matplotlib.figure.Figure", nrows, ncols)

    # workaround objmode string type name requirement by adding the type to types module
    # TODO: fix Numba's object mode to take type refs
    type_name = str(axes_type)
    if not hasattr(types, type_name):
        type_name = f"objmode_type{ir_utils.next_label()}"
        setattr(types, type_name, axes_type)

    # if axes is np.array, we convert to nested tuples
    func_text = f"""def impl(
        fig,
        nrows=1,
        ncols=1,
        sharex=False,
        sharey=False,
        squeeze=True,
        subplot_kw=None,
        gridspec_kw=None,
    ):
        with numba.objmode(axes="{type_name}"):
            axes = fig.subplots(
                nrows=nrows,
                ncols=ncols,
                sharex=sharex,
                sharey=sharey,
                squeeze=squeeze,
                subplot_kw=subplot_kw,
                gridspec_kw=gridspec_kw,
            )
            if isinstance(axes, np.ndarray):
                axes = tuple([tuple(elem) if isinstance(elem, np.ndarray) else elem for elem in axes])
        return axes
    """
    loc_vars = {}
    exec(func_text, {"numba": numba, "np": np}, loc_vars)
    impl = loc_vars["impl"]
    return impl


gen_objmode_func_overload(plt.show, output_type=types.none, single_rank=True)
gen_objmode_func_overload(plt.draw, output_type=types.none, single_rank=True)
gen_objmode_func_overload(plt.gcf, output_type=types.mpl_figure_type)
gen_objmode_method_overload(
    MplFigure,
    "show",
    matplotlib.figure.Figure.show,
    output_type=types.none,
    single_rank=True,
)
gen_objmode_method_overload(
    MplAxes,
    "set_xlim",
    matplotlib.axes.Axes.set_xlim,
    output_type=types.UniTuple(types.float64, 2),
)
gen_objmode_method_overload(
    MplAxes,
    "set_ylim",
    matplotlib.axes.Axes.set_ylim,
    output_type=types.UniTuple(types.float64, 2),
)
gen_objmode_method_overload(
    MplAxes, "set_xticks", matplotlib.axes.Axes.set_xticks, output_type=types.none
)
gen_objmode_method_overload(
    MplAxes, "set_yticks", matplotlib.axes.Axes.set_yticks, output_type=types.none
)
gen_objmode_method_overload(
    MplAxes, "draw", matplotlib.axes.Axes.draw, output_type=types.none, single_rank=True
)
gen_objmode_method_overload(
    MplAxes, "set_axis_on", matplotlib.axes.Axes.set_axis_on, output_type=types.none
)
gen_objmode_method_overload(
    MplAxes, "set_axis_off", matplotlib.axes.Axes.set_axis_off, output_type=types.none
)
