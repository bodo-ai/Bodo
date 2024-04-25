import llvmlite.binding as ll
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.core.extending import intrinsic, overload

import bodo
from bodo.ext import s3_reader
from bodo.io.fs_io import ArrowFs
from bodo.libs.str_ext import unicode_to_utf8

ll.add_symbol(
    "create_s3_fs_instance_py_entry", s3_reader.create_s3_fs_instance_py_entry
)


@intrinsic
def _create_s3_fs_instance(typingctx, region, anonymous, credentials_provider):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
                lir.LiteralStructType([lir.IntType(8).as_pointer(), lir.IntType(1)]),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="create_s3_fs_instance_py_entry"
        )
        args = (args[0], args[1], args[2])
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return (
        ArrowFs()(types.voidptr, types.boolean, types.optional(credentials_provider)),
        codegen,
    )


def create_s3_fs_instance(region="", anonymous=False, credentials_provider=None):
    pass


@overload(create_s3_fs_instance)
def overload_create_s3_fs_instance(
    region="", anonymous=False, credentials_provider=None
):
    """
    Create a S3 filesystem instance.
    args:
        region: str
            The region to use, if not specified, automatically detected.
        anonymous: bool
            Whether to use anonymous credentials.
        credentials_provider: an AWS credentials provider pointer
    """

    def impl(region="", anonymous=False, credentials_provider=None):
        return _create_s3_fs_instance(
            unicode_to_utf8(region), anonymous, credentials_provider
        )

    return impl
