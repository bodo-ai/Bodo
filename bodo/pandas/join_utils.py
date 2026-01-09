import bodo
from bodo.pandas.utils import (
    BodoLibNotImplementedException,
    cpp_table_to_df,
    df_to_cpp_table,
)

if bodo.gpu_enabled:
    import cudf
    import cupy
    import pyarrow
    import pylibcudf as plc
    import rmm
    import rmm.mr

    from bodo.ext import libcudf_hash_join


def new_stream():
    return cupy.cuda.Stream(non_blocking=True)


def new_event():
    # default flags are fine; use cupy.cuda.Event() which is recordable on streams
    return cupy.cuda.Event()


def record_event_on_stream(event, stream):
    # record event on given stream
    with stream:
        event.record()


def stream_wait_event(stream, event):
    # make stream wait for event (so subsequent work on stream starts after event)
    stream.wait_event(event)


def event_done(event):
    # non-blocking query
    return event.done


def cpp_table_to_cudf(cpp_table_ptr):
    """
    cpp_table_ptr: pointer to C++ table_info (or equivalent)
    Returns: cuDF DataFrame
    """

    bodo_df = cpp_table_to_df(cpp_table_ptr, use_arrow_dtypes=True, delete_input=False)
    return cudf.DataFrame(bodo_df)


def cpp_table_to_pylibcudf_table(cpp_table_ptr, stream):
    """
    cpp_table_ptr: pointer to C++ table_info (or equivalent)
    Returns: cuDF DataFrame
    """
    if stream is None:
        print("cpp_table_to_pylibcudf_table using default stream")
        stream = rmm.pylibrmm.stream.Stream(plc.utils.CUDF_DEFAULT_STREAM)

    bodo_df = cpp_table_to_df(cpp_table_ptr, use_arrow_dtypes=True, delete_input=False)
    if len(bodo_df) == 0:
        return None, bodo_df.columns
    return plc.Table.from_arrow(
        pyarrow.Table.from_pandas(bodo_df), stream=stream
    ), bodo_df.columns


def cpp_table_to_hash_join(cpp_table_ptr, nkeys):
    """
    cpp_table_ptr: pointer to C++ table_info (or equivalent)
    Returns: cuDF DataFrame
    """
    # TO-DO
    # Pass in build_kept_cols and then filter out unused columns before
    # transfering to GPU.

    df = cpp_table_to_cudf(cpp_table_ptr)
    # HashJoin only uses the key fields.
    # Those are the first columns of our dataframe in order.
    df_keys = df.iloc[:, :nkeys]
    # Convert to arrow to be able to convert to pylibcudf.Table
    arrow_df = df.to_arrow()
    arrow_df_keys = df_keys.to_arrow()
    # Convert to pylibcudf.
    plc_df = plc.Table.from_arrow(arrow_df)
    plc_df_keys = plc.Table.from_arrow(arrow_df_keys)
    # Build hash for keys of build table.
    hash_joined_table = libcudf_hash_join.HashJoin(
        plc_df_keys,
        libcudf_hash_join.NullEquality.EQUAL,
        None,  # libcudf_hash_join.NullableJoin.NO,
        0.5,
        plc.utils.CUDF_DEFAULT_STREAM,
    )
    # plc_df_keys to make sure that it stays alive until HashJoin is done
    return (arrow_df, hash_joined_table, plc_df, plc_df_keys, df.columns)


def select_columns(plc_table, column_indices):
    return plc.Table([plc_table.columns()[i] for i in column_indices])


def cpp_table_to_hash_join_async(cpp_table_ptr, nkeys):
    """
    cpp_table_ptr: pointer to C++ table_info (or equivalent)
    Returns: cuDF DataFrame
    """
    print("hash_join_async")

    build_stream = new_stream()
    build_event = new_event()

    # TO-DO
    # Pass in build_kept_cols and then filter out unused columns before
    # transfering to GPU.

    with build_stream:
        rmm_stream = rmm.pylibrmm.stream.Stream(build_stream)
        plc_df, df_columns = cpp_table_to_pylibcudf_table(
            cpp_table_ptr, stream=rmm_stream
        )
        assert plc_df != None
        plc_df_key_columns = select_columns(plc_df, range(nkeys))
        # Build hash for keys of build table.
        hash_joined_table = libcudf_hash_join.HashJoin(
            # HashJoin only uses the key fields.
            plc_df_key_columns,
            libcudf_hash_join.NullEquality.EQUAL,
            None,  # libcudf_hash_join.NullableJoin.NO,
            0.5,
            rmm_stream,
        )
        build_event.record()

    return {
        "plc_df": plc_df,
        "hash_joined_table": hash_joined_table,
        "build_stream": build_stream,
        "build_event": build_event,
        "df_columns": df_columns,
        "rmm_stream": rmm_stream,
        "plc_df_key_columns": plc_df_key_columns,
        # To be filled in by cudf_probe_hash_join_async below.
        # Will be a list of tuples (probe_stream, probe_event).
        "probe_tables": [],
        "compute_streams": [],
    }


def cudf_probe_hash_join(
    table_hash_join,  # 2-tuple of full build table and HashJoin object
    probe_cpp_table,
    nkeys,
    build_kept_cols,
    probe_kept_cols,
    join_kind,
    is_last,
):
    """
    table_hash_join: result of cpp_table_to_hash_join (full build side plus HashJoin)
    probe_cpp_table: pandas DataFrame (probe side)
    nkeys: number of join keys.  both input have to have keys in
        the same order at the beginning of the columns
    build_kept_cols: columns to keep from the build side
    probe_kept_cols: columns to keep from the probe side
    join_kind: join type ("inner", "left", "right", "outer")
    is_last: the last time this will be called for this join
    """

    # Extract the saved parts returned by cpp_table_to_hash_join above.
    full_build_table, build_hash_join, plc_build_table, _, build_table_columns = (
        table_hash_join
    )
    probe_df = cpp_table_to_df(
        probe_cpp_table, use_arrow_dtypes=True, delete_input=False
    )
    # TO-DO
    # Filter out unused columns before transfering to GPU.
    probe_gdf = cudf.DataFrame(probe_df)

    # HashJoin only use keys fields.
    probe_gdf_keys = probe_gdf.iloc[:, :nkeys]
    # Convert to arrow to convert to pylibcudf.
    arrow_df = probe_gdf.to_arrow()
    arrow_df_keys = probe_gdf_keys.to_arrow()
    # Convert to pylibcudf.
    plc_df = plc.Table.from_arrow(arrow_df)
    plc_df_keys = plc.Table.from_arrow(arrow_df_keys)

    cudfstream = plc.utils.CUDF_DEFAULT_STREAM
    dmr = rmm.mr.get_current_device_resource()

    if join_kind == "inner":
        result = build_hash_join.inner_join(plc_df_keys, None, cudfstream, dmr)
    else:
        raise BodoLibNotImplementedException(
            "hash_join other than inner not yet supported."
        )

    if not isinstance(result, tuple) or len(result) != 2:
        raise Exception("hash_join result not a 2-tuple")

    # result is 2-tuple of columns containing row indices.
    probe_idx_col, build_idx_col = result

    # Equivalent of pd.DataFrame.take.  Grab rows from row indices.
    probe_rows = plc.copying.gather(
        plc_df,
        probe_idx_col,
        plc.copying.OutOfBoundsPolicy.NULLIFY,
        cudfstream,
        dmr,
    )
    build_rows = plc.copying.gather(
        plc_build_table,
        build_idx_col,
        plc.copying.OutOfBoundsPolicy.NULLIFY,
        cudfstream,
        dmr,
    )
    # Drop columns not needed in the output.
    kept_probe_rows = [probe_rows.columns()[i] for i in probe_kept_cols]
    kept_build_rows = [build_rows.columns()[i] for i in build_kept_cols]
    result_plc_table = plc.Table(kept_probe_rows + kept_build_rows)
    result_arrow_table = result_plc_table.to_arrow()
    # Add column_names to arrow otherwise conversion to cudf drops
    # all but one unnamed column.
    column_names = [probe_df.columns[i] for i in probe_kept_cols] + [
        build_table_columns[i] for i in build_kept_cols
    ]
    result_arrow_table = result_arrow_table.rename_columns(column_names)
    result_gdf = cudf.DataFrame.from_arrow(result_arrow_table)
    result_df = result_gdf.to_arrow().to_pandas()
    out_ptr, _ = df_to_cpp_table(result_df)
    return out_ptr, 2


def add_probe_table(cudf_join_data, probe_cpp_table):
    # Add a probe table we haven't seen before to the list of work to be done.
    if probe_cpp_table != 0:
        probe_stream = new_stream()
        probe_event = new_event()
        with probe_stream:
            probe_rmm_stream = rmm.pylibrmm.stream.Stream(probe_stream)
            plc_df, probe_columns = cpp_table_to_pylibcudf_table(
                probe_cpp_table, stream=probe_rmm_stream
            )
            if plc_df is not None:
                probe_event.record()
                cudf_join_data["probe_tables"].append(
                    {
                        "probe_stream": probe_stream,
                        "probe_event": probe_event,
                        "probe_rmm_stream": probe_rmm_stream,
                        "plc_df": plc_df,
                        "probe_columns": probe_columns,
                    }
                )
            else:
                print("probe table empty")
    else:
        print("probe table missing")


def check_for_completed_compute(
    cudf_join_data, is_last, build_kept_cols, probe_kept_cols, cudfstream, dmr
):
    print("check_for_completed_compute", len(cudf_join_data["compute_streams"]))

    if len(cudf_join_data["compute_streams"]) > 0:
        compute_ready = False
        if is_last:
            # If this is the last time through then we have to return data.
            cudf_join_data["compute_streams"][0]["compute_event"].synchronize()
            compute_ready = True
        else:
            if event_done(cudf_join_data["compute_streams"][0]["compute_event"]):
                compute_ready = True

        if compute_ready:
            compute_data = cudf_join_data["compute_streams"].pop(0)

            if (
                not isinstance(compute_data["result"], tuple)
                or len(compute_data["result"]) != 2
            ):
                raise Exception("hash_join result not a 2-tuple")

            # result is 2-tuple of columns containing row indices.
            probe_idx_col, build_idx_col = compute_data["result"]

            # Equivalent of pd.DataFrame.take.  Grab rows from row indices.
            probe_rows = plc.copying.gather(
                compute_data["plc_df"],
                probe_idx_col,
                plc.copying.OutOfBoundsPolicy.NULLIFY,
                cudfstream,
                dmr,
            )
            build_rows = plc.copying.gather(
                cudf_join_data["plc_df"],
                build_idx_col,
                plc.copying.OutOfBoundsPolicy.NULLIFY,
                cudfstream,
                dmr,
            )
            # Drop columns not needed in the output.
            kept_probe_rows = [probe_rows.columns()[i] for i in probe_kept_cols]
            kept_build_rows = [build_rows.columns()[i] for i in build_kept_cols]
            result_plc_table = plc.Table(kept_probe_rows + kept_build_rows)
            result_arrow_table = result_plc_table.to_arrow()
            # Add column_names to arrow otherwise conversion to cudf drops
            # all but one unnamed column.
            probe_columns = compute_data["probe_data"]["probe_columns"]
            column_names = [probe_columns[i] for i in probe_kept_cols] + [
                cudf_join_data["df_columns"][i] for i in build_kept_cols
            ]
            result_arrow_table = result_arrow_table.rename_columns(column_names)
            result_gdf = cudf.DataFrame.from_arrow(result_arrow_table)
            result_df = result_gdf.to_arrow().to_pandas()
            out_ptr, _ = df_to_cpp_table(result_df)
            return out_ptr, 1 if (
                len(cudf_join_data["probe_tables"]) > 0
                or len(cudf_join_data["compute_streams"]) > 0
            ) else 2

    return None


def check_for_completed_probe(cudf_join_data, is_last, nkeys, join_kind, dmr):
    print("check_for_completed_probe", len(cudf_join_data["probe_tables"]))

    # See if any outstanding probe table transfers.
    if len(cudf_join_data["probe_tables"]) > 0:
        probe_ready = False  # Assume no probe is ready.
        if is_last:
            # If this is the last time through then we have to return data.
            # Can't imagine this will ever happen but if build side still
            # not done then wait on it here.
            cudf_join_data["build_event"].synchronize()
            cudf_join_data["probe_tables"][0]["probe_event"].synchronize()
            # If last time through we wait so probe must be ready.
            probe_ready = True
        else:
            if event_done(cudf_join_data["build_event"]) and event_done(
                cudf_join_data["probe_tables"][0]["probe_event"]
            ):
                # Probe ready if build and probe event have triggered.
                probe_ready = True

        print("probe_ready:", probe_ready)
        # We know we have a built and probe table ready for compute here.
        if probe_ready:
            compute_stream = new_stream()
            compute_event = new_event()
            with compute_stream:
                compute_rmm_stream = rmm.pylibrmm.stream.Stream(compute_stream)
                # Remove completed probe side from waiting probe list.
                probe_data = cudf_join_data["probe_tables"].pop(0)
                stream_wait_event(compute_stream, cudf_join_data["build_event"])
                stream_wait_event(compute_stream, probe_data["probe_event"])
                plc_df = probe_data["plc_df"]
                plc_df_keys = select_columns(plc_df, range(nkeys))

                if join_kind == "inner":
                    result = cudf_join_data["hash_joined_table"].inner_join(
                        plc_df_keys, None, compute_rmm_stream, dmr
                    )
                else:
                    raise BodoLibNotImplementedException(
                        "hash_join other than inner not yet supported."
                    )
                compute_event.record()
                cudf_join_data["compute_streams"].append(
                    {
                        "compute_stream": compute_stream,
                        "compute_event": compute_event,
                        "probe_data": probe_data,
                        "result": result,
                        "plc_df": plc_df,
                        "compute_rmm_stream": compute_rmm_stream,
                    }
                )


def cudf_probe_hash_join_async(
    cudf_join_data,  # dict of vars from cpp_table_to_hash_join_async
    probe_cpp_table,
    nkeys,
    build_kept_cols,
    probe_kept_cols,
    join_kind,
    is_last,
):
    """
    cudf_join_data: result of cpp_table_to_hash_join (full build side plus HashJoin)
    probe_cpp_table: pandas DataFrame (probe side)
    nkeys: number of join keys.  both input have to have keys in
        the same order at the beginning of the columns
    build_kept_cols: columns to keep from the build side
    probe_kept_cols: columns to keep from the probe side
    join_kind: join type ("inner", "left", "right", "outer")
    is_last: the last time this will be called for this join
    """

    cupy_def_stream = cupy.cuda.Stream.null
    cudfstream = rmm.pylibrmm.stream.Stream(cupy_def_stream)
    dmr = rmm.mr.get_current_device_resource()

    print("cudf_probe_hash_join_async", is_last, len(cudf_join_data["probe_tables"]))
    # TO-DO
    # Filter out unused columns before transfering to GPU.

    # Start probe table transfer to GPU if we have a valid probe table we havne't seen before.
    add_probe_table(cudf_join_data, probe_cpp_table)

    print(
        "cudf_probe_hash_join_async after probe batch processing",
        len(cudf_join_data["probe_tables"]),
    )
    # If there are probe tables that we started transferring before, see
    # if the transfers are done and we can launch the join kernel.
    check_for_completed_probe(cudf_join_data, is_last, nkeys, join_kind, dmr)
    # If there are completed compute kernels then start the transfer back from the device.
    results = check_for_completed_compute(
        cudf_join_data, is_last, build_kept_cols, probe_kept_cols, cudfstream, dmr
    )
    if results is not None:
        return results

    # General fallback if nothing to be returned.
    return 0, 0
