import bodo
import numpy as np
import pandas as pd
import sys
import time


@bodo.jit
def q26(ss_file, i_file, category, item_count):
    t1 = time.time()
    store_sales = pd.read_parquet(ss_file)
    item = pd.read_parquet(i_file)

    item2 = item[item["i_category"] == category]
    sale_items = pd.merge(
        store_sales, item2, left_on="ss_item_sk", right_on="i_item_sk")

    count1 = sale_items.groupby("ss_customer_sk")["ss_item_sk"].count()
    gp1 = sale_items.groupby("ss_customer_sk")["i_class_id"]

    def id1(x): return (x==1).sum()
    def id2(x): return (x==2).sum()
    def id3(x): return (x==3).sum()
    def id4(x): return (x==4).sum()
    def id5(x): return (x==5).sum()
    def id6(x): return (x==6).sum()
    def id7(x): return (x==7).sum()
    def id8(x): return (x==8).sum()
    def id9(x): return (x==9).sum()
    def id10(x): return (x==10).sum()
    def id11(x): return (x==11).sum()
    def id12(x): return (x==12).sum()
    def id13(x): return (x==13).sum()
    def id14(x): return (x==14).sum()
    def id15(x): return (x==15).sum()

    customer_i_class = gp1.agg((id1, id2, id3, id4, id5, id6, id7, id8, id9,
        id10, id11, id12, id13, id14, id15))

    customer_i_class["ss_item_count"] = count1

    customer_i_class = customer_i_class[
        customer_i_class.ss_item_count > item_count]
    res = customer_i_class.values.astype(np.float64).sum()
    print("checksum", res)
    print("exec time", time.time() - t1)


if __name__ == "__main__":

    ss_file = "/Users/ehsan/dev/sw/data/store_sales_10.parquet"
    i_file =  "/Users/ehsan/dev/sw/data/item_10.parquet"
    if len(sys.argv) == 3:
        ss_file = sys.argv[1]
        i_file =  sys.argv[2]

    q26_i_category_IN = "Books"
    q26_count_ss_item_sk = 5
    q26(ss_file, i_file, q26_i_category_IN, q26_count_ss_item_sk)
