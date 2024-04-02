import bodo


def test_query_profile_collection_compiles():
    """Check that all query profile collector functions compile"""

    @bodo.jit
    def impl():
        bodo.libs.query_profile_collector.init()
        bodo.libs.query_profile_collector.start_pipeline(1)
        bodo.libs.query_profile_collector.end_pipeline(1, 10)
        bodo.libs.query_profile_collector.finalize()
        return

    impl()
