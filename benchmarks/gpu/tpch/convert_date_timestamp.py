"""
Rewrite Parquet files and convert date columns to timestamp.
Supports local paths and S3 paths.
"""

import argparse
import os

import pyarrow as pa
import pyarrow.fs as fs
import pyarrow.parquet as pq


def get_filesystem_and_path(path: str):
    """
    Returns (filesystem, path_without_scheme)
    """
    if path.startswith("s3://"):
        region = os.environ.get("AWS_REGION", "us-east-2")
        filesystem = fs.S3FileSystem(region=region)
        path = path[len("s3://") :]
    else:
        filesystem = fs.LocalFileSystem()
    return filesystem, path.rstrip("/")


def ensure_parent_dir(filesystem: fs.FileSystem, path: str) -> None:
    """
    Create the parent directory for `path` if needed.
    """
    parent = path.rsplit("/", 1)[0] if "/" in path else ""
    if parent:
        filesystem.create_dir(parent, recursive=True)


def rewrite_table(src_path: str, dst_path: str) -> None:
    """
    Read Parquet files and convert date32 columns to timestamp[ns] while
    preserving the number of files and row groups.
    """
    src_fs, src_path = get_filesystem_and_path(src_path)
    dst_fs, dst_path = get_filesystem_and_path(dst_path)

    def convert_parquet_file(src_file: str, dst_file: str) -> None:
        ensure_parent_dir(dst_fs, dst_file)

        with src_fs.open_input_file(src_file) as f:
            pf = pq.ParquetFile(f)
            schema = pf.schema_arrow

            # Replace date32 fields with timestamp[ns]
            new_fields = []
            for field in schema:
                if pa.types.is_date32(field.type):
                    new_fields.append(
                        pa.field(
                            field.name, pa.timestamp("ns"), nullable=field.nullable
                        )
                    )
                else:
                    new_fields.append(field)

            new_schema = pa.schema(new_fields)

            # Reuse compression from the first row group where possible
            compression_map = {}
            if pf.metadata.num_row_groups > 0:
                rg = pf.metadata.row_group(0)
                for i in range(rg.num_columns):
                    col = rg.column(i)
                    codec = col.compression
                    if codec is not None:
                        compression_map[col.path_in_schema] = str(codec).lower()

            with dst_fs.open_output_stream(dst_file) as out:
                with pq.ParquetWriter(
                    out,
                    new_schema,
                    compression=compression_map or "snappy",
                ) as writer:
                    for i in range(pf.num_row_groups):
                        table = pf.read_row_group(i)

                        new_arrays = []
                        for field, col in zip(table.schema, table.columns):
                            if pa.types.is_date32(field.type):
                                col = col.cast(pa.timestamp("ns"))
                            new_arrays.append(col)

                        new_table = pa.Table.from_arrays(
                            new_arrays, names=new_schema.names
                        )
                        writer.write_table(new_table)

    info = src_fs.get_file_info(src_path)

    if info.type == fs.FileType.File:
        convert_parquet_file(src_path, dst_path)
        return

    if info.type == fs.FileType.Directory:
        # Make the dataset/table directory itself up front
        dst_fs.create_dir(dst_path, recursive=True)

        selector = fs.FileSelector(src_path, allow_not_found=False, recursive=False)
        for file_info in src_fs.get_file_info(selector):
            if file_info.type != fs.FileType.File:
                continue
            if not file_info.path.endswith(".pq") and not file_info.path.endswith(
                ".parquet"
            ):
                continue

            fname = file_info.path.rsplit("/", 1)[-1]
            print("Converting file:", fname)

            src_file = file_info.path
            dst_file = f"{dst_path}/{fname}"
            convert_parquet_file(src_file, dst_file)
        return

    raise FileNotFoundError(f"Source path not found: {src_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir",
        type=str,
        default="../data/tpch/SF1",
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        default="../data/tpch_dask/SF1",
    )
    args = parser.parse_args()

    for table in [
        "lineitem.pq",
        "orders.pq",
        "customer.pq",
        "nation.pq",
        "region.pq",
        "supplier.pq",
    ]:
        print(f"Converting table {table}...")
        rewrite_table(
            f"{args.src_dir.rstrip('/')}/{table}",
            f"{args.dst_dir.rstrip('/')}/{table}",
        )


if __name__ == "__main__":
    main()
