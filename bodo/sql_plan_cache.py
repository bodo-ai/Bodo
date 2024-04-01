import contextlib
import hashlib
import os
import tempfile
import uuid
from pathlib import Path

import numba

import bodo


@contextlib.contextmanager
def _open_for_write(filepath: str):
    """
    Shamelessly copied from Numba
    (https://github.com/numba/numba/blob/9ce83ef5c35d7f68a547bf2fd1266b9a88d3a00d/numba/core/caching.py#L553).
    Open *filepath* for writing in a race condition-free way (hopefully).
    uuid4 is used to try and avoid name collisions on a shared filesystem.
    """
    uid = uuid.uuid4().hex[:16]  # avoid long paths
    tmpname = "%s.tmp.%s" % (filepath, uid)
    try:
        with open(tmpname, "w") as f:
            yield f
        os.replace(tmpname, filepath)
    except Exception:
        # In case of error, remove dangling tmp file
        try:
            os.unlink(tmpname)
        except OSError:
            pass
        raise


class BodoSqlPlanCache:
    """
    A simple class that provides functionality for caching
    BodoSQL Plans generated during compilation.
    """

    @staticmethod
    def ensure_cache_dir_writable(path: str):
        """
        Similar to Numba's ensure_cache_path
        (https://github.com/numba/numba/blob/9ce83ef5c35d7f68a547bf2fd1266b9a88d3a00d/numba/core/caching.py#L103).
        Create the leaf directory (and all intermediate ones)
        if they don't already exist. Verify that the directory
        is writable by writing a temporary file to it.

        Args:
            path (str): Path to the directory.
        """
        os.makedirs(path, exist_ok=True)
        # Ensure the directory is writable by trying to write a temporary file
        tempfile.TemporaryFile(dir=path).close()

    @staticmethod
    def _get_sql_plan_cache_fname(sql_query: str) -> str:
        """
        Helper function to generate a deterministic filename
        for the plan based on the query text and bodo version.

        Args:
            sql_query (str): SQL Query text whose plan we're
                trying to cache.

        Returns:
            str: Filename for the cache file.
        """
        query_hash = hashlib.md5(sql_query.encode()).hexdigest()
        bodo_version = bodo.__version__
        return f"bodo_{bodo_version}_{query_hash}_plan.txt"

    @staticmethod
    def get_cache_loc(sql_query: str) -> str | None:
        """
        Get the location where the plan for a query should
        be cached. It returns None if caching is not setup
        (i.e. bodo.sql_plan_cache_loc is None or "") or if
        the that directory is not writable for some reason.
        This can also be used to retrieve the location of
        a previously cached plan. However, this doesn't
        guarantee that the plan exists at the location.

        Args:
            sql_query (str): SQL query text for which we want
                to cache the plan.

        Returns:
            str | None: Path where the plan should be cached.
        """
        if bodo.sql_plan_cache_loc:
            try:
                BodoSqlPlanCache.ensure_cache_dir_writable(bodo.sql_plan_cache_loc)
                return Path(
                    bodo.sql_plan_cache_loc
                ) / BodoSqlPlanCache._get_sql_plan_cache_fname(sql_query)
            except Exception:
                pass
        return None

    @staticmethod
    def cache_bodosql_plan(sql_plan: str, sql_query: str):
        """
        Cache a BodoSQL plan. This is called in the typing pass
        after generating the plan for a user query.
        The plan is only saved to disk by rank 0.

        Args:
            sql_plan (str): SQL plan to cache.
            sql_query (str): The SQL query that the plan corresponds to.
        """
        try:
            if bodo.get_rank() == 0:
                cache_loc = BodoSqlPlanCache.get_cache_loc(sql_query)
                if (not cache_loc) and numba.core.config.DEBUG_CACHE:
                    print(
                        f"[cache] sql plan caching not configured. Set the BODO_SQL_PLAN_CACHE_DIR environment variable to enable."
                    )
                if cache_loc:
                    with _open_for_write(cache_loc) as f:
                        f.write(sql_plan)
                    if numba.core.config.DEBUG_CACHE:
                        print(f"[cache] sql plan saved to {cache_loc}.")
        except Exception as e:
            if numba.core.config.DEBUG_CACHE:
                print("[cache] failed to cache BodoSQL Plan.")
                if numba.core.config.DEVELOPER_MODE:
                    print(f"[cache] error: {str(e)}")
