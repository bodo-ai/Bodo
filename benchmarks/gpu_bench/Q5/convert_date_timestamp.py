"""
Rewrite Parquet files in a directory and convert date columns to timestamp.
"""

import argparse
import os

import pyarrow as pa
import pyarrow.parquet as pq


def rewrite_table(src_dir, dst_dir):
    """
    Read Parquet files *src_dir* a directory and convert date columns to timestamp.
    Write converted Parquet files to *dst_dir*.
    """

    def convert_parquet_file(src_file, dst_file):
        pf = pq.ParquetFile(src_file)
        schema = pf.schema_arrow

        # Build new schema
        new_fields = []
        for field in schema:
            if pa.types.is_date32(field.type):
                new_fields.append(
                    pa.field(field.name, pa.timestamp("ns"), field.nullable)
                )
            else:
                new_fields.append(field)

        rg = pf.metadata.row_group(0)

        compression_map = {}
        for i in range(rg.num_columns):
            col = rg.column(i)
            compression_map[col.path_in_schema] = col.compression.lower()

        new_schema = pa.schema(new_fields)

        with pq.ParquetWriter(
            dst_file, new_schema, compression=compression_map
        ) as writer:
            # Preserve row groups exactly
            for i in range(pf.num_row_groups):
                table = pf.read_row_group(i)

                # Cast only date columns
                columns = []
                for col in table.columns:
                    if pa.types.is_date32(col.type):
                        col = col.cast(pa.timestamp("ns"))
                    columns.append(col)

                new_table = pa.Table.from_arrays(columns, schema=new_schema)
                writer.write_table(new_table)

    if os.path.isfile(src_dir):
        convert_parquet_file(src_dir, dst_dir)

    elif os.path.isdir(src_dir):
        os.makedirs(dst_dir, exist_ok=True)

        for fname in sorted(os.listdir(src_dir)):
            if not fname.endswith(".pq"):
                continue
            print("Converting file:", fname)

            src_file = os.path.join(src_dir, fname)
            dst_file = os.path.join(dst_dir, fname)

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
            os.path.join(args.src_dir, table), os.path.join(args.dst_dir, table)
        )


if __name__ == "__main__":
    main()
