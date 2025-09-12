import cloudpickle
from numba.core.dispatcher import Dispatcher  # noqa TID253

import bodo


def test_reduce_pipeline_class():
    """
    Test that pickling / unpickling preserves the pipeline_class of a
    Dispatcher
    """

    @bodo.jit
    def f(x):
        return x + 1

    pickled_f = cloudpickle.dumps(f)

    # remove function from cache to force a rebuild
    del Dispatcher._memo[f._uuid]

    unpickle_f = cloudpickle.loads(pickled_f)

    assert unpickle_f._compiler.pipeline_class == bodo.compiler.BodoCompiler, (
        "expected f to use BodoCompiler as it's pipeline"
    )
