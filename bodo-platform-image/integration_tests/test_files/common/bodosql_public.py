import time
import bodo
import sys

try:
    import bodosql
except:
    print(" ############## bodosql not Installed ##################### ")
    sys.exit(1)


@bodo.jit()
def use_case_1():
    start = time.time()
    bc = bodosql.BodoSQLContext(
        {
            "lineitem": bodosql.TablePath(
                "s3://bodo-example-data/tpch/SF1/lineitem.pq", "parquet"
            ),
        }
    )

    fdf = bc.sql(
        """select
                l_returnflag,
                l_linestatus,
                sum(l_quantity) as sum_qty,
                sum(l_extendedprice) as sum_base_price,
                sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
                sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
                avg(l_quantity) as avg_qty,
                avg(l_extendedprice) as avg_price,
                avg(l_discount) as avg_disc,
                count(*) as count_order
            from
                lineitem
            where
                l_shipdate <= date '1998-12-01' - interval '90' day
            group by
                l_returnflag,
                l_linestatus
            order by
                l_returnflag,
                l_linestatus"""
    )

    print(f"query took {time.time()-start} seconds", fdf)
    return fdf


if __name__ == "__main__":
    df1 = use_case_1()
    print(df1)
