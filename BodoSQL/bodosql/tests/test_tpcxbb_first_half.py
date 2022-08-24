# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of TPCxBB Benchmark on BodoSQL

Some of these queries should be set with variables. These variables and their values can
be seen in the TPCxBB document,
http://tpc.org/tpc_documents_current_versions/pdf/tpcx-bb_v1.5.0.pdf. For now we set most
of these variables according to the reference query.
"""

import pytest
from bodosql.tests.utils import check_query
from pyspark.sql.types import IntegerType, StructField, StructType


@pytest.mark.skip("[BS-633] Resolve nightly issues for TPCx-BB")
@pytest.mark.slow
def test_tpcxbb_q01(tpcxbb_data, spark_info):
    # TODO: re add memory_leak_check, see BS-574
    tpcxbb_query1 = """
        SELECT DISTINCT ss_item_sk, ss_ticket_number
        FROM store_sales s, item i
        WHERE s.ss_item_sk = i.i_item_sk
        AND i.i_category_id IN (1, 2, 3)
        AND s.ss_store_sk IN (10, 20, 33, 40, 50)
    """
    tpcxbb_data_dict, pyspark_schemas = tpcxbb_data
    result = check_query(
        tpcxbb_query1,
        tpcxbb_data_dict,
        spark_info,
        check_dtype=False,
        pyspark_schemas=pyspark_schemas,
        return_seq_dataframe=True,
    )

    output_df_dict = {"distinct_table": result["output_df"]}
    # Supply schema in case the output is empty.
    # TODO: Update table subset so the result is never empty
    result_schemas = {
        "distinct_table": StructType(
            [
                StructField("ss_item_sk", IntegerType(), True),
                StructField("ss_ticket_number", IntegerType(), True),
            ]
        )
    }

    # Spark uses String for its string name and BodoSQL still uses
    # varchar, so we need separate queries
    tpcxbb_query2 = """
        SELECT item_sk_1, item_sk_2, COUNT(*) AS cnt
        FROM
        (
            SELECT CAST(t1.ss_item_sk as BIGINT) AS item_sk_1,
                CAST(t2.ss_item_sk AS BIGINT) AS item_sk_2
            FROM distinct_table t1
            INNER JOIN distinct_table t2
            ON t1.ss_ticket_number = t2.ss_ticket_number
            WHERE t1.ss_item_sk < t2.ss_item_sk
        )
        GROUP BY item_sk_1, item_sk_2
        HAVING  COUNT(*) > 50
        ORDER BY cnt DESC, CAST(item_sk_1 AS VARCHAR),
                    CAST(item_sk_2 AS VARCHAR)
        LIMIT 100
    """
    tpcxbb_spark_query2 = tpcxbb_query2.replace("VARCHAR", "string")
    check_query(
        tpcxbb_query2,
        output_df_dict,
        spark_info,
        equivalent_spark_query=tpcxbb_spark_query2,
        pyspark_schemas=result_schemas,
        check_dtype=False,
    )


@pytest.mark.skip("[BS-633] Resolve nightly issues for TPCx-BB")
@pytest.mark.slow
def test_tpcxbb_q06(tpcxbb_data, spark_info):
    # TODO: re add memory_leak_check, see BS-574
    q06_LIMIT = 100
    q06_YEAR = 2001
    tpcxbb_query = f"""
        WITH temp_table_1 as
        (
            SELECT ss_customer_sk AS customer_sk,
                sum( case when (d_year = {q06_YEAR}) THEN (((ss_ext_list_price-ss_ext_wholesale_cost-ss_ext_discount_amt)+ss_ext_sales_price)/2.0) ELSE 0.0 END)
                    AS first_year_total,
                sum( case when (d_year = {q06_YEAR + 1}) THEN (((ss_ext_list_price-ss_ext_wholesale_cost-ss_ext_discount_amt)+ss_ext_sales_price)/2.0) ELSE 0.0 END)
                    AS second_year_total
            FROM store_sales,
                date_dim
            WHERE ss_sold_date_sk = d_date_sk
            AND   d_year BETWEEN {q06_YEAR} AND {q06_YEAR + 1}
            GROUP BY ss_customer_sk
            -- first_year_total is an aggregation, rewrite all sum () statement
            HAVING sum( case when (d_year = {q06_YEAR}) THEN (((ss_ext_list_price-ss_ext_wholesale_cost-ss_ext_discount_amt)+ss_ext_sales_price)/2.0) ELSE 0.0 END) > 0.0
        ),
        temp_table_2 AS
        (
            SELECT ws_bill_customer_sk AS customer_sk ,
                sum( case when (d_year = {q06_YEAR}) THEN (((ws_ext_list_price-ws_ext_wholesale_cost-ws_ext_discount_amt)+ws_ext_sales_price)/2.0) ELSE 0.0 END)
                    AS first_year_total,
                sum( case when (d_year = {q06_YEAR + 1}) THEN (((ws_ext_list_price-ws_ext_wholesale_cost-ws_ext_discount_amt)+ws_ext_sales_price)/2.0) ELSE 0.0 END)
                    AS second_year_total
            FROM web_sales,
                 date_dim
            WHERE ws_sold_date_sk = d_date_sk
            AND   d_year BETWEEN {q06_YEAR} AND {q06_YEAR + 1}
            GROUP BY ws_bill_customer_sk
            -- required to avoid division by 0, because later we will divide by this value
            HAVING sum( case when (d_year = {q06_YEAR}) THEN (((ws_ext_list_price-ws_ext_wholesale_cost-ws_ext_discount_amt)+ws_ext_sales_price)/2.0)ELSE 0.0 END) > 0.0
        )
        -- MAIN QUERY
        SELECT
            CAST( (web.second_year_total / web.first_year_total) AS DOUBLE) AS web_sales_increase_ratio,
            c_customer_sk,
            c_first_name,
            c_last_name,
            c_preferred_cust_flag,
            c_birth_country,
            c_login,
            c_email_address
        FROM temp_table_1 store,
            temp_table_2 web,
            customer c
        WHERE store.customer_sk = web.customer_sk
        AND  web.customer_sk = c_customer_sk
        -- if customer has sales in first year for both store and websales,
        -- select him only if web second_year_total/first_year_total
        -- ratio is bigger then his store second_year_total/first_year_total ratio.
        AND (web.second_year_total / web.first_year_total) >
            (store.second_year_total / store.first_year_total)
        ORDER BY
            web_sales_increase_ratio DESC,
            c_customer_sk,
            c_first_name,
            c_last_name,
            c_preferred_cust_flag,
            c_birth_country,
            c_login
        LIMIT {q06_LIMIT}
        """
    tpcxbb_data_dict, pyspark_schemas = tpcxbb_data

    result = check_query(
        tpcxbb_query,
        tpcxbb_data_dict,
        spark_info,
        check_dtype=False,
        pyspark_schemas=pyspark_schemas,
    )


@pytest.mark.skip("[BS-633] Resolve nightly issues for TPCx-BB")
@pytest.mark.skip(reason="Failed:AssertionError: Sequential Python Test Failed, BS-560")
def test_tpcxbb_q09(tpcxbb_data, spark_info, memory_leak_check):
    q09_year = 2001
    q09_part1_ca_country = "United States"
    q09_part1_ca_state_IN = "'KY', 'GA', 'NM'"
    q09_part1_net_profit_min = 0
    q09_part1_net_profit_max = 2000
    q09_part1_education_status = "4 yr Degree"
    q09_part1_marital_status = "M"
    q09_part1_sales_price_min = 100
    q09_part1_sales_price_max = 150
    q09_part2_ca_country = "United States"
    q09_part2_ca_state_IN = "'MT', 'OR', 'IN'"
    q09_part2_net_profit_min = 150
    q09_part2_net_profit_max = 3000
    q09_part2_education_status = "4 yr Degree"
    q09_part2_marital_status = "M"
    q09_part2_sales_price_min = 50
    q09_part2_sales_price_max = 200
    q09_part3_ca_country = "United States"
    q09_part3_ca_state_IN = "'WI', 'MO', 'WV'"
    q09_part3_net_profit_min = 50
    q09_part3_net_profit_max = 25000
    q09_part3_education_status = "4 yr Degree"
    q09_part3_marital_status = "M"
    q09_part3_sales_price_min = 150
    q09_part3_sales_price_max = 200

    tpcxbb_query = f"""
        SELECT SUM(ss1.ss_quantity)
        FROM store_sales ss1,
            date_dim dd,customer_address ca1,
            store s,
            customer_demographics cd
        -- select date range
        WHERE ss1.ss_sold_date_sk = dd.d_date_sk
        AND dd.d_year = {q09_year}
        AND ss1.ss_addr_sk = ca1.ca_address_sk
        AND s.s_store_sk = ss1.ss_store_sk
        AND cd.cd_demo_sk = ss1.ss_cdemo_sk
        AND
        (
            (
                cd.cd_marital_status = '{q09_part1_marital_status}'
                AND cd.cd_education_status = '{q09_part1_education_status}'
                AND {q09_part1_sales_price_min} <= ss1.ss_sales_price
                AND ss1.ss_sales_price <= {q09_part1_sales_price_max}
            )
            OR
            (
                cd.cd_marital_status = '{q09_part2_marital_status}'
                AND cd.cd_education_status = '{q09_part2_education_status}'
                AND {q09_part2_sales_price_min} <= ss1.ss_sales_price
                AND ss1.ss_sales_price <= {q09_part2_sales_price_max}
            )
            OR
            (
                cd.cd_marital_status = '{q09_part3_marital_status}'
                AND cd.cd_education_status = '{q09_part3_education_status}'
                AND {q09_part3_sales_price_min} <= ss1.ss_sales_price
                AND ss1.ss_sales_price <= {q09_part3_sales_price_max}
            )
        )
        AND
        (
            (
                ca1.ca_country = '{q09_part1_ca_country}'
                AND ca1.ca_state IN ({q09_part1_ca_state_IN})
                AND {q09_part1_net_profit_min} <= ss1.ss_net_profit
                AND ss1.ss_net_profit <= {q09_part1_net_profit_max}
            )
            OR
            (
                ca1.ca_country = '{q09_part2_ca_country}'
                AND ca1.ca_state IN ({q09_part2_ca_state_IN})
                AND {q09_part2_net_profit_min} <= ss1.ss_net_profit
                AND ss1.ss_net_profit <= {q09_part2_net_profit_max}
            )
            OR
            (
                ca1.ca_country = '{q09_part3_ca_country}'
                AND ca1.ca_state IN ({q09_part3_ca_state_IN})
                AND {q09_part3_net_profit_min} <= ss1.ss_net_profit
                AND ss1.ss_net_profit <= {q09_part3_net_profit_max}
            )
        )
    """

    tpcxbb_data_dict, pyspark_schemas = tpcxbb_data

    result = check_query(
        tpcxbb_query,
        tpcxbb_data_dict,
        spark_info,
        check_dtype=False,
        pyspark_schemas=pyspark_schemas,
        check_names=False,
    )


@pytest.mark.skip("[BS-633] Resolve nightly issues for TPCx-BB")
@pytest.mark.slow
def test_tpcxbb_q11(tpcxbb_data, spark_info, memory_leak_check):
    tpcxbb_query = """
        WITH p AS
        (
            SELECT
                pr_item_sk,
                count(pr_item_sk) AS r_count,
                AVG( CAST(pr_review_rating AS DOUBLE) ) avg_rating
            FROM product_reviews
            WHERE pr_item_sk IS NOT NULL
            GROUP BY pr_item_sk
        ), s AS
        (
            SELECT
                ws_item_sk
            FROM web_sales ws
            INNER JOIN date_dim d ON ws.ws_sold_date_sk = d.d_date_sk
            WHERE ws_item_sk IS NOT null
            AND CAST(d.d_date AS DATE) >= DATE '2003-01-02'
            AND CAST(d.d_date AS DATE) <= DATE '2003-02-02'
            GROUP BY ws_item_sk
        )
        SELECT p.r_count    AS x,
            p.avg_rating AS y
        FROM s INNER JOIN p ON p.pr_item_sk = s.ws_item_sk
    """

    tpcxbb_data_dict, pyspark_schemas = tpcxbb_data

    result = check_query(
        tpcxbb_query,
        tpcxbb_data_dict,
        spark_info,
        check_dtype=False,
        pyspark_schemas=pyspark_schemas,
    )


@pytest.mark.skip("[BS-633] Resolve nightly issues for TPCx-BB")
@pytest.mark.slow
def test_tpcxbb_q12(tpcxbb_data, spark_info):
    # TODO: re add memory leak check, BS-574
    q12_i_category_IN = "'Books', 'Electronics'"
    tpcxbb_query = f"""
        SELECT DISTINCT wcs_user_sk
        FROM
        (
            SELECT DISTINCT
                wcs_user_sk,
                wcs_click_date_sk
            FROM web_clickstreams, item
            WHERE wcs_click_date_sk BETWEEN 37134 AND 37164
            AND i_category IN ({q12_i_category_IN})
            AND wcs_item_sk = i_item_sk
            AND wcs_user_sk IS NOT NULL
            AND wcs_sales_sk IS NULL
        ) webInRange,
        (
            SELECT DISTINCT
                ss_customer_sk,
                ss_sold_date_sk
            FROM store_sales, item
            WHERE ss_sold_date_sk BETWEEN 37134 AND 37224
            AND i_category IN ({q12_i_category_IN}) -- filter given category
            AND ss_item_sk = i_item_sk
            AND ss_customer_sk IS NOT NULL
        ) storeInRange
        WHERE wcs_user_sk = ss_customer_sk
        AND wcs_click_date_sk < ss_sold_date_sk
        ORDER BY wcs_user_sk
    """

    tpcxbb_data_dict, pyspark_schemas = tpcxbb_data

    result = check_query(
        tpcxbb_query,
        tpcxbb_data_dict,
        spark_info,
        check_dtype=False,
        pyspark_schemas=pyspark_schemas,
    )


@pytest.mark.skip("[BS-633] Resolve nightly issues for TPCx-BB")
@pytest.mark.skip(reason="Empty object array passed to Bodo, BS-559")
def test_tpcxbb_q14(tpcxbb_data, spark_info, memory_leak_check):
    tpcxbb_query = """
		SELECT CASE WHEN pmc > 0.0 THEN CAST (amc AS DOUBLE) / CAST (pmc AS DOUBLE) ELSE -1.0 END AS am_pm_ratio
		FROM
		(
			SELECT SUM(amc1) AS amc, SUM(pmc1) AS pmc
			FROM
			(
				SELECT
					CASE WHEN t_hour BETWEEN 7 AND 8 THEN COUNT(1) ELSE 0 END AS amc1,
					CASE WHEN t_hour BETWEEN 19 AND 20 THEN COUNT(1) ELSE 0 END AS pmc1
				FROM web_sales ws
				JOIN household_demographics hd ON (hd.hd_demo_sk = ws.ws_ship_hdemo_sk and hd.hd_dep_count = 5)
				JOIN web_page wp ON (wp.wp_web_page_sk = ws.ws_web_page_sk and wp.wp_char_count BETWEEN 5000 AND 6000)
				JOIN time_dim td ON (td.t_time_sk = ws.ws_sold_time_sk and td.t_hour IN (7,8,19,20))
				GROUP BY t_hour
			) cnt_am_pm
		) sum_am_pm
	"""
    tpcxbb_data_dict, pyspark_schemas = tpcxbb_data
    result = check_query(
        tpcxbb_query,
        tpcxbb_data_dict,
        spark_info,
        check_dtype=False,
        pyspark_schemas=pyspark_schemas,
    )


@pytest.mark.skip("[BS-633] Resolve nightly issues for TPCx-BB")
@pytest.mark.slow
def test_tpcxbb_q15(tpcxbb_data, spark_info):
    # TODO: re add memory_leak_check, BS-574
    q15_startDate = "2001-09-02"
    q15_endDate = "2002-09-02"
    q15_store_sk = 10
    tpcxbb_query = f"""
        SELECT *
        FROM
        (
            SELECT
                cat,
                ( (count(x) * SUM(xy) - SUM(x) * SUM(y)) / (count(x) * SUM(xx) - SUM(x) * SUM(x)) )  AS slope,
                (SUM(y) - ((count(x) * SUM(xy) - SUM(x) * SUM(y)) / (count(x) * SUM(xx) - SUM(x)*SUM(x)) ) * SUM(x)) / count(x) AS intercept
            FROM
            (
                SELECT
                    i.i_category_id AS cat,
                    s.ss_sold_date_sk AS x,
                    CAST(SUM(s.ss_net_paid) AS DOUBLE) AS y,
                    CAST(s.ss_sold_date_sk * SUM(s.ss_net_paid) AS DOUBLE) AS xy,
                    CAST(s.ss_sold_date_sk * s.ss_sold_date_sk AS DOUBLE) AS xx
                FROM store_sales s
                INNER JOIN item i ON s.ss_item_sk = i.i_item_sk
                INNER JOIN date_dim d ON s.ss_sold_date_sk = d.d_date_sk
                WHERE s.ss_store_sk = {q15_store_sk}
                AND i.i_category_id IS NOT NULL
                AND CAST(d.d_date AS DATE) >= DATE '{q15_startDate}'
                AND   CAST(d.d_date AS DATE) <= DATE '{q15_endDate}'
                GROUP BY i.i_category_id, s.ss_sold_date_sk
            ) temp
            GROUP BY cat
        ) regression
        WHERE slope <= 0.0
        ORDER BY cat
    """
    tpcxbb_data_dict, pyspark_schemas = tpcxbb_data
    result = check_query(
        tpcxbb_query,
        tpcxbb_data_dict,
        spark_info,
        check_dtype=False,
        pyspark_schemas=pyspark_schemas,
    )
