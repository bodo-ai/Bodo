# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of TPCxBB Benchmark on BodoSQL
"""
import pytest
from bodosql.tests.utils import check_query


@pytest.mark.skip("[BS-633] Resolve nightly issues for TPCx-BB")
@pytest.mark.slow
def test_tpcxbb_q20(tpcxbb_data, spark_info, memory_leak_check):
    N_CLUSTERS = 8
    CLUSTER_ITERATIONS = 20
    N_ITER = 5
    tpcxbb_query = """
        SELECT
            ss_customer_sk AS user_sk,
            round(CASE WHEN ((returns_count IS NULL) OR (orders_count IS NULL)
                OR ((returns_count / orders_count) IS NULL) ) THEN 0.0
                ELSE (returns_count / orders_count) END, 7) AS orderRatio,
            round(CASE WHEN ((returns_items IS NULL) OR (orders_items IS NULL)
                OR ((returns_items / orders_items) IS NULL) ) THEN 0.0
                ELSE (returns_items / orders_items) END, 7) AS itemsRatio,
            round(CASE WHEN ((returns_money IS NULL) OR (orders_money IS NULL)
                OR ((returns_money / orders_money) IS NULL) ) THEN 0.0
                ELSE (returns_money / orders_money) END, 7) AS monetaryRatio,
            round(CASE WHEN ( returns_count IS NULL) THEN 0.0
                ELSE returns_count END, 0) AS frequency
        FROM
        (
            SELECT
                ss_customer_sk,
                -- return order ratio
                CAST (COUNT(distinct(ss_ticket_number)) AS DOUBLE)
                    AS orders_count,
                -- return ss_item_sk ratio
                CAST (COUNT(ss_item_sk) AS DOUBLE) AS orders_items,
                -- return monetary amount ratio
                CAST(SUM( ss_net_paid ) AS DOUBLE) AS orders_money
            FROM store_sales s
            GROUP BY ss_customer_sk
        ) orders
        LEFT OUTER JOIN
        (
            SELECT
                sr_customer_sk,
                -- return order ratio
                CAST(count(distinct(sr_ticket_number)) AS DOUBLE)
                    AS returns_count,
                -- return ss_item_sk ratio
                CAST (COUNT(sr_item_sk) AS DOUBLE) AS returns_items,
                -- return monetary amount ratio
                CAST( SUM( sr_return_amt ) AS DOUBLE) AS returns_money
            FROM store_returns
            GROUP BY sr_customer_sk
        ) returned ON ss_customer_sk=sr_customer_sk
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
def test_tpcxbb_q21(tpcxbb_data, spark_info, memory_leak_check):
    tpcxbb_query = """
		SELECT
			part_i.i_item_id AS i_item_id,
            part_i.i_item_desc AS i_item_desc,
            part_s.s_store_id AS s_store_id,
            part_s.s_store_name AS s_store_name,
            CAST(SUM(part_ss.ss_quantity) AS BIGINT) AS store_sales_quantity,
            CAST(SUM(part_sr.sr_return_quantity) AS BIGINT) AS store_returns_quantity,
            CAST(SUM(part_ws.ws_quantity) AS BIGINT) AS web_sales_quantity
		FROM
		(
			SELECT
				sr_item_sk,
				sr_customer_sk,
				sr_ticket_number,
				sr_return_quantity
			FROM
				store_returns sr,
				date_dim d2
			WHERE d2.d_year = 2003
			AND d2.d_moy BETWEEN 1 AND 7 --which were returned in the next six months
			AND sr.sr_returned_date_sk = d2.d_date_sk
		) part_sr
		INNER JOIN
		(
			SELECT
				ws_item_sk,
				ws_bill_customer_sk,
				ws_quantity
			FROM
				web_sales ws,
				date_dim d3
			-- in the following three years (re-purchased by the returning customer afterwards through the web sales channel)
			WHERE d3.d_year BETWEEN 2003 AND 2005
			AND ws.ws_sold_date_sk = d3.d_date_sk
		) part_ws ON
		(
			part_sr.sr_item_sk = part_ws.ws_item_sk
			AND part_sr.sr_customer_sk = part_ws.ws_bill_customer_sk
		) INNER JOIN
		(
			SELECT
				ss_item_sk,
				ss_store_sk,
				ss_customer_sk,
				ss_ticket_number,
				ss_quantity
			FROM
				store_sales ss,
				date_dim d1
			WHERE d1.d_year = 2003
			AND d1.d_moy = 1
			AND ss.ss_sold_date_sk = d1.d_date_sk
		) part_ss ON
		(
			part_ss.ss_ticket_number = part_sr.sr_ticket_number
			AND part_ss.ss_item_sk = part_sr.sr_item_sk
			AND part_ss.ss_customer_sk = part_sr.sr_customer_sk
		)
		INNER JOIN store part_s ON
		(
			part_s.s_store_sk = part_ss.ss_store_sk
		)
		INNER JOIN item part_i ON
		(
			part_i.i_item_sk = part_ss.ss_item_sk
		)
		GROUP BY
			part_i.i_item_id,
			part_i.i_item_desc,
			part_s.s_store_id,
			part_s.s_store_name
		ORDER BY
			part_i.i_item_id,
			part_i.i_item_desc,
			part_s.s_store_id,
			part_s.s_store_name
		LIMIT 100
	"""
    tpcxbb_data_dict, pyspark_schemas = tpcxbb_data
    check_query(
        tpcxbb_query,
        tpcxbb_data_dict,
        spark_info,
        check_dtype=False,
        pyspark_schemas=pyspark_schemas,
    )


@pytest.mark.skip("[BS-633] Resolve nightly issues for TPCx-BB")
@pytest.mark.skip(reason="Undefined function: 'timestampdiff', BS-563")
def test_tpcxbb_q22(tpcxbb_data, spark_info, memory_leak_check):
    q22_date = "2001-05-08"
    q22_i_current_price_min = "0.98"
    q22_i_current_price_max = "1.5"
    tpcxbb_query = f"""
        SELECT
            w_warehouse_name,
            i_item_id,
            SUM(CASE WHEN timestampdiff(DAY, timestamp '{q22_date} 00:00:00', CAST(d_date || ' 00:00:00' AS timestamp))
                / 1000000 < 0 THEN inv_quantity_on_hand ELSE 0 END) AS inv_before,
            SUM(CASE WHEN timestampdiff(DAY, timestamp '{q22_date} 00:00:00', CAST(d_date || ' 00:00:00' AS timestamp))
                / 1000000 >= 0 THEN inv_quantity_on_hand ELSE 0 END) AS inv_after
        FROM
            inventory inv,
            item i,
            warehouse w,
            date_dim d
        WHERE i_current_price BETWEEN {q22_i_current_price_min} AND {q22_i_current_price_max}
        AND i_item_sk        = inv_item_sk
        AND inv_warehouse_sk = w_warehouse_sk
        AND inv_date_sk      = d_date_sk
        AND timestampdiff(DAY, timestamp '{q22_date} 00:00:00', CAST(d_date || ' 00:00:00' AS timestamp)) / 1000000 >= -30
        AND timestampdiff(DAY, timestamp '{q22_date} 00:00:00', CAST(d_date || ' 00:00:00' AS timestamp)) / 1000000 <= 30
        GROUP BY w_warehouse_name, i_item_id
        HAVING SUM(CASE WHEN timestampdiff(DAY, timestamp '{q22_date}', CAST(d_date || ' 00:00:00' AS timestamp))
            / 1000000 < 0 THEN inv_quantity_on_hand ELSE 0 END) > 0
        AND
        (
            CAST(
            SUM (CASE WHEN timestampdiff(DAY, timestamp '{q22_date} 00:00:00', CAST(d_date || ' 00:00:00' AS timestamp)) / 1000000 >= 0 THEN inv_quantity_on_hand ELSE 0 END) AS DOUBLE)
            / CAST( SUM(CASE WHEN timestampdiff(DAY, timestamp '{q22_date} 00:00:00', CAST(d_date || ' 00:00:00' AS timestamp)) / 1000000 < 0 THEN inv_quantity_on_hand ELSE 0 END)
            AS DOUBLE) >= 0.666667
        )
        AND
        (
            CAST(
            SUM(CASE WHEN timestampdiff(DAY, timestamp '{q22_date} 00:00:00', CAST(d_date || ' 00:00:00' AS timestamp)) / 1000000 >= 0 THEN inv_quantity_on_hand ELSE 0 END) AS DOUBLE)
            / CAST ( SUM(CASE WHEN timestampdiff(DAY, timestamp '{q22_date} 00:00:00', CAST(d_date || ' 00:00:00' AS timestamp)) / 1000000 < 0 THEN inv_quantity_on_hand ELSE 0 END)
         AS DOUBLE) <= 1.50
        )
        ORDER BY w_warehouse_name, i_item_id
        LIMIT 100
    """

    tpcxbb_data_dict, pyspark_schemas = tpcxbb_data
    check_query(
        tpcxbb_query,
        tpcxbb_data_dict,
        spark_info,
        check_dtype=False,
        pyspark_schemas=pyspark_schemas,
    )


@pytest.mark.skip("[BS-633] Resolve nightly issues for TPCx-BB")
@pytest.mark.skip(reason="Output mismatch in expected columns, BS-564")
@pytest.mark.slow
def test_tpcxbb_q24(tpcxbb_data, spark_info, memory_leak_check):
    tpcxbb_query = """
		WITH temp_table as
		(
			SELECT
				i_item_sk,
				imp_sk,
				(imp_competitor_price - i_current_price) / i_current_price AS price_change,
				imp_start_date,
				(imp_end_date - imp_start_date) AS no_days_comp_price
			FROM item i ,item_marketprices imp
			WHERE i.i_item_sk = imp.imp_item_sk
			AND i.i_item_sk = 10000
			ORDER BY i_item_sk, imp_sk, imp_start_date
		)
		SELECT ws_item_sk,
		-- avg ( (current_ss_quant + current_ws_quant - prev_ss_quant - prev_ws_quant) / ((prev_ss_quant + prev_ws_quant) * ws.price_change) ) -- single node
			sum( (current_ss_quant+current_ws_quant-prev_ss_quant-prev_ws_quant) / (prev_ss_quant*ws.price_change+prev_ws_quant*ws.price_change) )
			/ count( (current_ss_quant + current_ws_quant - prev_ss_quant - prev_ws_quant) / ((prev_ss_quant + prev_ws_quant) * ws.price_change) ) AS cross_price_elasticity
		FROM
		(
			SELECT
				ws_item_sk,
				imp_sk,
				price_change,
				SUM( CASE WHEN ( (ws_sold_date_sk >= c.imp_start_date) AND (ws_sold_date_sk < (c.imp_start_date + c.no_days_comp_price))) THEN ws_quantity ELSE 0 END ) AS current_ws_quant,
				SUM( CASE WHEN ( (ws_sold_date_sk >= (c.imp_start_date - c.no_days_comp_price)) AND (ws_sold_date_sk < c.imp_start_date)) THEN ws_quantity ELSE 0 END ) AS prev_ws_quant
			FROM web_sales ws
			JOIN temp_table c ON ws.ws_item_sk = c.i_item_sk
			GROUP BY ws_item_sk, imp_sk, price_change
		) ws JOIN
		(
			SELECT
				ss_item_sk,
				imp_sk,
				price_change,
				SUM( CASE WHEN ((ss_sold_date_sk >= c.imp_start_date) AND (ss_sold_date_sk < (c.imp_start_date + c.no_days_comp_price))) THEN ss_quantity ELSE 0 END) AS current_ss_quant,
				SUM( CASE WHEN ((ss_sold_date_sk >= (c.imp_start_date - c.no_days_comp_price)) AND (ss_sold_date_sk < c.imp_start_date)) THEN ss_quantity ELSE 0 END) AS prev_ss_quant
			FROM store_sales ss
			JOIN temp_table c ON c.i_item_sk = ss.ss_item_sk
			GROUP BY ss_item_sk, imp_sk, price_change
		) ss
		ON (ws.ws_item_sk = ss.ss_item_sk and ws.imp_sk = ss.imp_sk)
		GROUP BY  ws.ws_item_sk
  	"""

    tpcxbb_data_dict, pyspark_schemas = tpcxbb_data
    check_query(
        tpcxbb_query,
        tpcxbb_data_dict,
        spark_info,
        check_dtype=False,
        pyspark_schemas=pyspark_schemas,
    )


@pytest.mark.skip("[BS-633] Resolve nightly issues for TPCx-BB")
@pytest.mark.skip(reason="Segfaults, BS-565")
def test_tpcxbb_q25(tpcxbb_data, spark_info, memory_leak_check):
    q25_date = "2002-01-02"
    tpcxbb_query = f"""
        WITH concat_table AS
        (
            (
                SELECT
                    ss_customer_sk AS cid,
                    count(distinct ss_ticket_number) AS frequency,
                    max(ss_sold_date_sk) AS most_recent_date,
                    CAST( SUM(ss_net_paid) AS DOUBLE) AS amount
                FROM store_sales ss
                JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk
                WHERE CAST(d.d_date AS DATE) > DATE '{q25_date}'
                AND ss_customer_sk IS NOT NULL
                GROUP BY ss_customer_sk
            ) union all
            (
                SELECT
                    ws_bill_customer_sk AS cid,
                    count(distinct ws_order_number) AS frequency,
                    max(ws_sold_date_sk)   AS most_recent_date,
                    CAST( SUM(ws_net_paid) AS DOUBLE) AS amount
                FROM web_sales ws
                JOIN date_dim d ON ws.ws_sold_date_sk = d.d_date_sk
                WHERE CAST(d.d_date AS DATE) > DATE '{q25_date}'
                AND ws_bill_customer_sk IS NOT NULL
                GROUP BY ws_bill_customer_sk
            )
        )
        SELECT
            cid AS cid,
            CASE WHEN 37621 - max(most_recent_date) < 60 THEN 1.0
                ELSE 0.0 END AS recency, -- 37621 == 2003-01-02
            CAST( SUM(frequency) AS BIGINT) AS frequency, --total frequency
            CAST( SUM(amount) AS DOUBLE)    AS amount --total amount
        FROM concat_table
        GROUP BY cid
        ORDER BY cid
    """
    tpcxbb_data_dict, pyspark_schemas = tpcxbb_data
    check_query(
        tpcxbb_query,
        tpcxbb_data_dict,
        spark_info,
        check_dtype=False,
        pyspark_schemas=pyspark_schemas,
    )


@pytest.mark.skip("[BS-633] Resolve nightly issues for TPCx-BB")
@pytest.mark.slow
def test_tpcxbb_q26(tpcxbb_data, spark_info):
    # TODO: re add memory leak check, see BS-574
    tpcxbb_query = """
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
    tpcxbb_data_dict, pyspark_schemas = tpcxbb_data
    check_query(
        tpcxbb_query,
        tpcxbb_data_dict,
        spark_info,
        check_dtype=False,
        pyspark_schemas=pyspark_schemas,
    )
