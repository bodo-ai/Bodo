import sys
import time

import numba
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import bodo

"""
    SELECT
        ss.ss_customer_sk AS cid,
        CAST( count(CASE WHEN i.i_class_id=1  THEN 1 ELSE NULL END) AS DOUBLE ) AS id1,
        CAST( count(CASE WHEN i.i_class_id=2  THEN 1 ELSE NULL END) AS DOUBLE ) AS id2,
        CAST( count(CASE WHEN i.i_class_id=3  THEN 1 ELSE NULL END) AS DOUBLE ) AS id3,
        CAST( count(CASE WHEN i.i_class_id=4  THEN 1 ELSE NULL END) AS DOUBLE ) AS id4,
        CAST( count(CASE WHEN i.i_class_id=5  THEN 1 ELSE NULL END) AS DOUBLE ) AS id5,
        CAST( count(CASE WHEN i.i_class_id=6  THEN 1 ELSE NULL END) AS DOUBLE ) AS id6,
        CAST( count(CASE WHEN i.i_class_id=7  THEN 1 ELSE NULL END) AS DOUBLE ) AS id7,
        CAST( count(CASE WHEN i.i_class_id=8  THEN 1 ELSE NULL END) AS DOUBLE ) AS id8,
        CAST( count(CASE WHEN i.i_class_id=9  THEN 1 ELSE NULL END) AS DOUBLE ) AS id9,
        CAST( count(CASE WHEN i.i_class_id=10 THEN 1 ELSE NULL END) AS DOUBLE ) AS id10,
        CAST( count(CASE WHEN i.i_class_id=11 THEN 1 ELSE NULL END) AS DOUBLE ) AS id11,
        CAST( count(CASE WHEN i.i_class_id=12 THEN 1 ELSE NULL END) AS DOUBLE ) AS id12,
        CAST( count(CASE WHEN i.i_class_id=13 THEN 1 ELSE NULL END) AS DOUBLE ) AS id13,
        CAST( count(CASE WHEN i.i_class_id=14 THEN 1 ELSE NULL END) AS DOUBLE ) AS id14,
        CAST( count(CASE WHEN i.i_class_id=15 THEN 1 ELSE NULL END) AS DOUBLE ) AS id15
    FROM store_sales ss
    INNER JOIN item i
    ON
    (
        ss.ss_item_sk = i.i_item_sk
        AND i.i_category IN ('Books')
        AND ss.ss_customer_sk IS NOT NULL
    )
    GROUP BY ss.ss_customer_sk
    HAVING count(ss.ss_item_sk) > 5
    ORDER BY cid
"""


@bodo.jit(distributed=["customer_i_class"], cache=True)
def tpcx_bb_q26_etl():
    print("Start...")
    print("Reading tables...")
    t0 = time.time()
    ss_file = "s3://bodotest-customer-data/tpcx-bb/q26/store_sales.pq"
    i_file = "s3://bodotest-customer-data/tpcx-bb/q26/item.pq"
    store_sales = pd.read_parquet(ss_file)
    item = pd.read_parquet(i_file)
    t1 = time.time()
    print("Read time: ", t1 - t0)

    print("Starting computation...")
    t0 = time.time()

    item2 = item[item["i_category"] == "Books"]
    sale_items = pd.merge(
        store_sales, item2, left_on="ss_item_sk", right_on="i_item_sk"
    )
    count1 = sale_items.groupby("ss_customer_sk")["ss_item_sk"].count()
    gp1 = sale_items.groupby("ss_customer_sk")["i_class_id"]

    def id1(x):
        return (x == 1).sum()

    def id2(x):
        return (x == 2).sum()

    def id3(x):
        return (x == 3).sum()

    def id4(x):
        return (x == 4).sum()

    def id5(x):
        return (x == 5).sum()

    def id6(x):
        return (x == 6).sum()

    def id7(x):
        return (x == 7).sum()

    def id8(x):
        return (x == 8).sum()

    def id9(x):
        return (x == 9).sum()

    def id10(x):
        return (x == 10).sum()

    def id11(x):
        return (x == 11).sum()

    def id12(x):
        return (x == 12).sum()

    def id13(x):
        return (x == 13).sum()

    def id14(x):
        return (x == 14).sum()

    def id15(x):
        return (x == 15).sum()

    customer_i_class = gp1.agg(
        (
            id1,
            id2,
            id3,
            id4,
            id5,
            id6,
            id7,
            id8,
            id9,
            id10,
            id11,
            id12,
            id13,
            id14,
            id15,
        )
    )
    customer_i_class["ss_item_count"] = count1
    customer_i_class = customer_i_class[customer_i_class.ss_item_count > 5]
    customer_i_class = customer_i_class.drop(["ss_item_count"], axis=1)
    customer_i_class = customer_i_class.sort_values("ss_customer_sk")
    res = customer_i_class.values.astype(np.float64).sum()
    print("checksum", res)
    t1 = time.time()
    print("Execution time: ", t1 - t0)
    return customer_i_class


@bodo.jit(distributed=["training_df"], cache=True)
def run_clustering(training_df):
    print("run_clustering: Start...")
    t0 = time.time()
    model = KMeans(
        n_clusters=8,
        max_iter=20,
        random_state=np.random.randint(0, 500),
        init="k-means++",
        n_init=5,  # Run 5 times and get the best model of those
    )
    model.fit(training_df)
    with numba.objmode(score="float64"):
        score = model.inertia_
    t1 = time.time()
    print("run_clustering: Execution time: ", t1 - t0)
    return model, score


def tpcx_bb_q26_ml(kmeans_input_df):
    model, sse = run_clustering(kmeans_input_df)
    return {
        "cid_labels": model.labels_,
        "wssse": model.inertia_,
        "cluster_centers": model.cluster_centers_,
        "nclusters": 8,
    }


def main():
    result_etl = tpcx_bb_q26_etl()
    ml_result_dict = tpcx_bb_q26_ml(result_etl)
    return ml_result_dict


if __name__ == "__main__":
    require_cache = False
    if len(sys.argv) > 1:
        require_cache = bool(sys.argv[1])
    ml_result_dict = main()
    if bodo.get_rank() == 0:
        print("ml_result_dict:\n", ml_result_dict)
    with open("wssse.txt", "w") as f:
        f.write(str(ml_result_dict["wssse"]))
    if require_cache and isinstance(run_clustering, numba.core.dispatcher.Dispatcher):
        assert run_clustering._cache_hits[run_clustering.signatures[0]] == 1, (
            "ERROR: Bodo did not load from cache"
        )
    if require_cache and isinstance(tpcx_bb_q26_etl, numba.core.dispatcher.Dispatcher):
        assert tpcx_bb_q26_etl._cache_hits[tpcx_bb_q26_etl.signatures[0]] == 1, (
            "ERROR: Bodo did not load from cache"
        )
