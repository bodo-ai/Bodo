"""
Rewrite Parquet files and convert date columns to timestamp.
Supports local paths and S3 paths.
"""

import argparse
import os

import pyarrow as pa
import pyarrow.fs as fs
import pyarrow.parquet as pq


def get_filesystem_and_path(path):
    """
    Returns (filesystem, path_without_scheme)
    """
    if path.startswith("s3://"):
        region = os.environ.get("AWS_REGION", "us-east-2")
        filesystem = fs.S3FileSystem(region=region)
        path = path.replace("s3://", "")
    else:
        filesystem = fs.LocalFileSystem()
    return filesystem, path


def rewrite_table(src_path, dst_path):
    """
    Read Parquet files and convert date columns to timestamp while preserving
    number of files and row groups.
    """

    src_fs, src_path = get_filesystem_and_path(src_path)
    dst_fs, dst_path = get_filesystem_and_path(dst_path)

    def convert_parquet_file(src_file, dst_file):
        with src_fs.open_input_file(src_file) as f:
            pf = pq.ParquetFile(f)

            schema = pf.schema_arrow

            # Replace date32 types with timestamp[ns]
            new_fields = []
            for field in schema:
                if pa.types.is_date32(field.type):
                    new_fields.append(
                        pa.field(field.name, pa.timestamp("ns"), field.nullable)
                    )
                else:
                    new_fields.append(field)

            new_schema = pa.schema(new_fields)

            rg = pf.metadata.row_group(0)

            compression_map = {}
            for i in range(rg.num_columns):
                col = rg.column(i)
                compression_map[col.path_in_schema] = col.compression.lower()

            with dst_fs.open_output_stream(dst_file) as out:
                with pq.ParquetWriter(
                    out, new_schema, compression=compression_map
                ) as writer:
                    for i in range(pf.num_row_groups):
                        table = pf.read_row_group(i)

                        columns = []
                        for col in table.columns:
                            if pa.types.is_date32(col.type):
                                col = col.cast(pa.timestamp("ns"))
                            columns.append(col)

                        new_table = pa.Table.from_arrays(columns, schema=new_schema)
                        writer.write_table(new_table)

    info = src_fs.get_file_info(src_path)
    if info.type == fs.FileType.File:
        convert_parquet_file(src_path, dst_path)

    elif info.type == fs.FileType.Directory:
        selector = fs.FileSelector(src_path, allow_not_found=False, recursive=False)
        for file_info in src_fs.get_file_info(selector):
            if not file_info.path.endswith(".pq"):
                continue

            fname = os.path.basename(file_info.path)
            print("Converting file:", fname)

            src_file = file_info.path
            dst_file = f"{dst_path}/{fname}"

            convert_parquet_file(src_file, dst_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir",
        type=str,
        default="../data/tpch/SF1/",
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        default="../data/tpch_dask/SF1/",
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
            f"{args.src_dir}/{table}",
            f"{args.dst_dir}/{table}",
        )


if __name__ == "__main__":
    main()
