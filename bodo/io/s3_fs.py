import llvmlite.binding as ll
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.core.extending import intrinsic, models, overload, register_model

import bodo
from bodo.ext import s3_reader
from bodo.io.fs_io import ArrowFs
from bodo.libs.str_ext import gen_std_str_to_unicode, unicode_to_utf8

ll.add_symbol(
    "create_s3_fs_instance_py_entry", s3_reader.create_s3_fs_instance_py_entry
)
ll.add_symbol(
    "get_region_from_creds_provider_py_entry",
    s3_reader.get_region_from_creds_provider_py_entry,
)


class IcebergAwsCredentialsProviderType(types.Type):
    """Type for C++ Iceberg REST AWS Credentials Provider"""

    def __init__(self):  # pragma: no cover
        super(IcebergAwsCredentialsProviderType, self).__init__(
            name=f"IcebergAwsCredentialsProvider()"
        )


@intrinsic
def _create_iceberg_aws_credentials_provider(
    typingctx, catalog_uri, bearer_token, warehouse, schema, table
):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module,
            fnty,
            name="create_iceberg_aws_credentials_provider_py_entry",
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return (
        IcebergAwsCredentialsProviderType()(
            types.voidptr, types.voidptr, types.voidptr, types.voidptr, types.voidptr
        ),
        codegen,
    )


def create_iceberg_aws_credentials_provider(
    catalog_uri: str, bearer_token: str, warehouse: str, schema: str, table: str
):
    pass


@overload(create_iceberg_aws_credentials_provider)
def overload_create_iceberg_aws_credentials_provider(
    catalog_uri: str, bearer_token: str, warehouse: str, schema: str, table: str
):
    """
    Create a C++ Iceberg AWS Credentials Provider
    Creates an Iceberg REST AWS Credentials Provider object if all the parameters
    have values, otherwise creates a default AWS Credentials Provider object.
    @param catalog_uri: Iceberg REST Catalog URI
    @param bearer_token: Bearer token for authentication to the REST server
    @param warehouse: Iceberg warehouse
    @param schema: database schema
    @param table: table name
    @return: Iceberg AWS Credentials Provider object
    """

    def impl(
        catalog_uri: str, bearer_token: str, warehouse: str, schema: str, table: str
    ):
        return _create_iceberg_aws_credentials_provider(
            unicode_to_utf8(catalog_uri),
            unicode_to_utf8(bearer_token),
            unicode_to_utf8(warehouse),
            unicode_to_utf8(schema),
            unicode_to_utf8(table),
        )

    return impl


@intrinsic
def _destroy_iceberg_aws_credentials_provider(typingctx, provider):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(8).as_pointer()])
        fn_tp = cgutils.get_or_insert_function(
            builder.module,
            fnty,
            name="destroy_iceberg_aws_credentials_provider_py_entry",
        )
        builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

    return types.void(IcebergAwsCredentialsProviderType()), codegen


def destroy_iceberg_aws_credentials_provider(provider):
    pass


@overload(destroy_iceberg_aws_credentials_provider)
def overload_destroy_iceberg_rest_aws_credentials_provider(provider):
    """
    Destroy a C++ Iceberg REST AWS Credentials Provider
    """

    def impl(provider):
        if provider is not None:
            return _destroy_iceberg_aws_credentials_provider(provider)

    return impl


@intrinsic
def _get_region_from_creds_provider(typingctx, provider):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(), [lir.IntType(8).as_pointer()]
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module,
            fnty,
            name="get_region_from_creds_provider_py_entry",
        )
        str = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        ret = gen_std_str_to_unicode(context, builder, str, True)
        return ret

    return types.unicode_type(IcebergAwsCredentialsProviderType()), codegen


def get_region_from_creds_provider(provider: IcebergAwsCredentialsProviderType):
    """
    Extracts the region attribute from an IcebergRestAWSCredentialsProvider
    """
    pass


@overload(get_region_from_creds_provider)
def overload_get_region_from_creds_provider(provider):
    assert isinstance(provider, IcebergAwsCredentialsProviderType)

    def impl(provider):
        return _get_region_from_creds_provider(provider)

    return impl


register_model(IcebergAwsCredentialsProviderType)(models.OpaqueModel)


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
        ArrowFs()(
            types.voidptr,
            types.boolean,
            types.Optional(IcebergAwsCredentialsProviderType()),
        ),
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
