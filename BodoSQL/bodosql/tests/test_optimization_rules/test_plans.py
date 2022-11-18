# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Test correctness of generated plans for certain queries
"""

import bodosql
import pandas as pd
import pytest


@pytest.fixture()
def engage3_ctx():
    """The schemas of the tables used inPOC1. Schemas manually
    recreated by observing the schema on Snowflake and creating DataFrames
    with identical datatypes."""
    return {
        "p_ret_price_history_denorm": pd.DataFrame(
            {
                "uuid": pd.Series(["foo"]),
                "ret_product_id": pd.Series([42], dtype=pd.Int64Dtype()),
                "store_id": pd.Series([42], dtype=pd.Int64Dtype()),
                "price_collecter_id": pd.Series(["foo"]),
                "sku": pd.Series(["foo"]),
                "upc": pd.Series(["foo"]),
                "description": pd.Series(["foo"]),
                "brand": pd.Series(["foo"]),
                "size_uofm": pd.Series(["foo"]),
                "url": pd.Series(["foo"]),
                "image_url": pd.Series(["foo"]),
                "category_l1": pd.Series(["foo"]),
                "category_l2": pd.Series(["foo"]),
                "category_l3": pd.Series(["foo"]),
                "category_l4": pd.Series(["foo"]),
                "category_l5": pd.Series(["foo"]),
                "category_l6": pd.Series(["foo"]),
                "category_l7": pd.Series(["foo"]),
                "category_l8": pd.Series(["foo"]),
                "reg_multiple": pd.Series([42], dtype=pd.Int64Dtype()),
                "reg_price": pd.Series([-1.234]),
                "reg_ppu": pd.Series([-1.234]),
                "promo_multiple": pd.Series([42], dtype=pd.Int64Dtype()),
                "promo_price": pd.Series([-1.234]),
                "promo_ppu": pd.Series([-1.234]),
                "loyalty_multiple": pd.Series([42], dtype=pd.Int64Dtype()),
                "loyalty_price": pd.Series([-1.234]),
                "loyalty_ppu": pd.Series([-1.234]),
                "last_seen": pd.Series([pd.Timestamp("2022-10-26")]),
                "notes": pd.Series(["foo"]),
                "loyalty_exp_date": pd.Series([pd.Timestamp("2022-10-26")]),
                "promo_exp_date": pd.Series([pd.Timestamp("2022-10-26")]),
                "longitude": pd.Series([-1.234]),
                "latitude": pd.Series([-1.234]),
                "regular_price_correction": pd.Series([-1.234]),
                "regular_price_reason": pd.Series(["foo"]),
                "promo_price_correction": pd.Series([-1.234]),
                "promo_price_reason": pd.Series(["foo"]),
                "loyalty_price_correction": pd.Series([-1.234]),
                "loyalty_price_reason": pd.Series(["foo"]),
                "banner_id": pd.Series([42], dtype=pd.Int64Dtype()),
                "postal_code": pd.Series(["foo"]),
                "f_flag_value": pd.Series([-1.234]),
                "c_flag_value": pd.Series([-1.234]),
                "d_flag_value": pd.Series([-1.234]),
                "l_flag_value": pd.Series([-1.234]),
                "loy_flag_value": pd.Series([-1.234]),
                "pro_flag_value": pd.Series([-1.234]),
                "j_flag_value": pd.Series([-1.234]),
                "z_flag_value": pd.Series([-1.234]),
                "y_flag_value": pd.Series([-1.234]),
                "q_flag_value": pd.Series([-1.234]),
                "w_flag_value": pd.Series([-1.234]),
                "n_flag_value": pd.Series([-1.234]),
                "t_flag_value": pd.Series([-1.234]),
                "u_flag_value": pd.Series([-1.234]),
                "v_flag_value": pd.Series([-1.234]),
                "onefortyeight_flag_value": pd.Series([-1.234]),
                "zone_indicator": pd.Series(["foo"]),
                "client_descr": pd.Series(["foo"]),
                "item_size": pd.Series(["foo"]),
                "item_uom": pd.Series(["foo"]),
                "price_status_id": pd.Series([42], dtype=pd.Int64Dtype()),
                "raw_data_json": pd.Series(["foo"]),
                "created_by": pd.Series(["foo"]),
                "created_date": pd.Series([pd.Timestamp("2022-10-26")]),
                "last_updated_by": pd.Series(["foo"]),
                "last_update_date": pd.Series([pd.Timestamp("2022-10-26")]),
                "order_number": pd.Series(["foo"]),
                "product_set_member_id": pd.Series(["foo"]),
                "qa_flags": pd.Series(["foo"]),
                "shopping_list_item_id": pd.Series([42], dtype=pd.Int64Dtype()),
                "in_stock": pd.Series([True], dtype=pd.BooleanDtype()),
                "user_mission_xref_id": pd.Series([42], dtype=pd.Int64Dtype()),
                "user_id": pd.Series([42], dtype=pd.Int64Dtype()),
                "user_email": pd.Series(["foo"]),
                "user_client_id": pd.Series(["foo"]),
                "user_client_name": pd.Series(["foo"]),
                "etl_audit": pd.Series(["foo"]),
                "scannar_request_item_id": pd.Series([42], dtype=pd.Int64Dtype()),
                "ppu_unit": pd.Series(["foo"]),
                "delete_flag": pd.Series(["foo"]),
                "delete_flag_date": pd.Series([pd.Timestamp("2022-10-26")]),
            }
        ),
        "s_ret_store": pd.DataFrame(
            {
                "store_id": pd.Series([42], dtype=pd.Int64Dtype()),
                "banner_id": pd.Series([42], dtype=pd.Int64Dtype()),
                "store_name": pd.Series(["foo"]),
                "store_code": pd.Series(["foo"]),
                "store_url": pd.Series(["foo"]),
                "address_line_1": pd.Series(["foo"]),
                "address_line_2": pd.Series(["foo"]),
                "city": pd.Series(["foo"]),
                "state": pd.Series(["foo"]),
                "postal_code": pd.Series(["foo"]),
                "country": pd.Series(["foo"]),
                "longitude": pd.Series([-1.234]),
                "latitude": pd.Series([-1.234]),
                "active_flag": pd.Series([True], dtype=pd.BooleanDtype()),
                "public_flag": pd.Series([True], dtype=pd.BooleanDtype()),
                "last_validated": pd.Series([pd.Timestamp("2022-10-26")]),
                "online_store_code": pd.Series(["foo"]),
                "created_by": pd.Series([42], dtype=pd.Int64Dtype()),
                "created_dt": pd.Series([pd.Timestamp("2022-10-26")]),
                "last_update": pd.Series(["foo"]),
                "last_update_by": pd.Series([42], dtype=pd.Int64Dtype()),
                "last_update_dt": pd.Series([pd.Timestamp("2022-10-26")]),
                "replaced_by": pd.Series([42], dtype=pd.Int64Dtype()),
            }
        ),
        "s_ret_banner": pd.DataFrame(
            {
                "banner_id": pd.Series([42], dtype=pd.Int64Dtype()),
                "chain_id": pd.Series([42], dtype=pd.Int64Dtype()),
                "banner_name": pd.Series(["foo"]),
                "banner_url": pd.Series(["foo"]),
                "created_dt": pd.Series([pd.Timestamp("2022-10-26")]),
                "created_by": pd.Series([42], dtype=pd.Int64Dtype()),
                "last_update_by": pd.Series([42], dtype=pd.Int64Dtype()),
                "last_update_dt": pd.Series([pd.Timestamp("2022-10-26")]),
                "shop_url": pd.Series(["foo"]),
            }
        ),
        "p_ret_product": pd.DataFrame(
            {
                "ret_product_id": pd.Series([42], dtype=pd.Int64Dtype()),
                "banner_id": pd.Series([42], dtype=pd.Int64Dtype()),
                "mst_product_id": pd.Series([42], dtype=pd.Int64Dtype()),
                "sku": pd.Series(["foo"]),
                "upc": pd.Series(["foo"]),
                "description": pd.Series(["foo"]),
                "brand": pd.Series(["foo"]),
                "size_uofm": pd.Series(["foo"]),
                "url": pd.Series(["foo"]),
                "image_url": pd.Series(["foo"]),
                "last_seen": pd.Series([pd.Timestamp("2022-10-26")]),
                "public_flag": pd.Series([True], dtype=pd.BooleanDtype()),
                "raw_data": pd.Series(["foo"]),
                "replaced_by": pd.Series([42], dtype=pd.Int64Dtype()),
                "active_flag": pd.Series([True], dtype=pd.BooleanDtype()),
                "upc_bak": pd.Series(["foo"]),
                "created_by": pd.Series([42], dtype=pd.Int64Dtype()),
                "created_dt": pd.Series([pd.Timestamp("2022-10-26")]),
                "last_update": pd.Series(["foo"]),
                "last_update_by": pd.Series([42], dtype=pd.Int64Dtype()),
                "last_update_dt": pd.Series([pd.Timestamp("2022-10-26")]),
                "price": pd.Series([-1.234]),
                "unit_size": pd.Series([-1.234]),
                "pack_size": pd.Series([-1.234]),
                "unit_uom": pd.Series(["foo"]),
            }
        ),
        "p_mst_product": pd.DataFrame(
            {
                "mst_product_id": pd.Series([42], dtype=pd.Int64Dtype()),
                "gtin": pd.Series(["foo"]),
                "plu": pd.Series([42], dtype=pd.Int64Dtype()),
                "mph_node_id": pd.Series([42], dtype=pd.Int64Dtype()),
                "mph_node_id_cb": pd.Series([42], dtype=pd.Int64Dtype()),
                "mph_node_id_dt": pd.Series([pd.Timestamp("2022-10-26")]),
                "description": pd.Series(["foo"]),
                "description_cb": pd.Series([42], dtype=pd.Int64Dtype()),
                "description_dt": pd.Series([pd.Timestamp("2022-10-26")]),
                "pack_size": pd.Series(["foo"]),
                "pack_size_cb": pd.Series([42], dtype=pd.Int64Dtype()),
                "pack_size_dt": pd.Series([pd.Timestamp("2022-10-26")]),
                "unit_size": pd.Series(["foo"]),
                "unit_size_cb": pd.Series([42], dtype=pd.Int64Dtype()),
                "unit_size_dt": pd.Series([pd.Timestamp("2022-10-26")]),
                "image_size": pd.Series(["foo"]),
                "image_size_cb": pd.Series([42], dtype=pd.Int64Dtype()),
                "image_size_dt": pd.Series([pd.Timestamp("2022-10-26")]),
                "total_size": pd.Series(["foo"]),
                "total_size_cb": pd.Series([42], dtype=pd.Int64Dtype()),
                "total_size_dt": pd.Series([pd.Timestamp("2022-10-26")]),
                "replaced_by": pd.Series([42], dtype=pd.Int64Dtype()),
                "gtin_bak": pd.Series(["foo"]),
                "gtin_plu_cb": pd.Series([42], dtype=pd.Int64Dtype()),
                "gtin_plu_dt": pd.Series([pd.Timestamp("2022-10-26")]),
                "last_update_by": pd.Series([42], dtype=pd.Int64Dtype()),
                "last_update_dt": pd.Series([pd.Timestamp("2022-10-26")]),
            }
        ),
        "p_v_mph_flat": pd.DataFrame(
            {
                "mph_node_id": pd.Series([42], dtype=pd.Int64Dtype()),
                "mph_node_name": pd.Series(["foo"]),
                "mph_node_level": pd.Series([42], dtype=pd.Int64Dtype()),
                "leaf_node": pd.Series([True], dtype=pd.BooleanDtype()),
                "l1_id": pd.Series([42], dtype=pd.Int64Dtype()),
                "l1_name": pd.Series(["foo"]),
                "l2_id": pd.Series([42], dtype=pd.Int64Dtype()),
                "l2_name": pd.Series(["foo"]),
                "l3_id": pd.Series([42], dtype=pd.Int64Dtype()),
                "l3_name": pd.Series(["foo"]),
                "l4_id": pd.Series([42], dtype=pd.Int64Dtype()),
                "l4_name": pd.Series(["foo"]),
                "l5_id": pd.Series([42], dtype=pd.Int64Dtype()),
                "l5_name": pd.Series(["foo"]),
                "l6_id": pd.Series([42], dtype=pd.Int64Dtype()),
                "l6_name": pd.Series(["foo"]),
                "l7_id": pd.Series([42], dtype=pd.Int64Dtype()),
                "l7_name": pd.Series(["foo"]),
                "l8_id": pd.Series([42], dtype=pd.Int64Dtype()),
                "l8_name": pd.Series(["foo"]),
                "l9_id": pd.Series([42], dtype=pd.Int64Dtype()),
                "l9_name": pd.Series(["foo"]),
                "l10_id": pd.Series([42], dtype=pd.Int64Dtype()),
                "l10_name": pd.Series(["foo"]),
                "l11_id": pd.Series([42], dtype=pd.Int64Dtype()),
                "l11_name": pd.Series(["foo"]),
                "l12_id": pd.Series([42], dtype=pd.Int64Dtype()),
                "l12_name": pd.Series(["foo"]),
                "created_dt": pd.Series([pd.Timestamp("2022-10-26")]),
                "created_by": pd.Series([42], dtype=pd.Int64Dtype()),
                "last_update_by": pd.Series([42], dtype=pd.Int64Dtype()),
                "last_update_dt": pd.Series([pd.Timestamp("2022-10-26")]),
                "branch": pd.Series(["foo"]),
            }
        ),
    }


@pytest.mark.parametrize(
    "plan_tests",
    [
        pytest.param(
            ("plan_data/tpch_q1.sql", "plan_data/tpch_q1_expected.txt", "TPCH"),
            id="tpch_q1",
        ),
        pytest.param(
            ("plan_data/tpch_q2.sql", "plan_data/tpch_q2_expected.txt", "TPCH"),
            id="tpch_q2",
        ),
        pytest.param(
            ("plan_data/tpch_q3.sql", "plan_data/tpch_q3_expected.txt", "TPCH"),
            id="tpch_q3",
        ),
        pytest.param(
            ("plan_data/tpch_q4.sql", "plan_data/tpch_q4_expected.txt", "TPCH"),
            id="tpch_q4",
        ),
        pytest.param(
            ("plan_data/tpch_q5.sql", "plan_data/tpch_q5_expected.txt", "TPCH"),
            id="tpch_q5",
        ),
        pytest.param(
            ("plan_data/tpch_q6.sql", "plan_data/tpch_q6_expected.txt", "TPCH"),
            id="tpch_q6",
        ),
        pytest.param(
            ("plan_data/tpch_q7.sql", "plan_data/tpch_q7_expected.txt", "TPCH"),
            id="tpch_q7",
        ),
        pytest.param(
            ("plan_data/tpch_q8.sql", "plan_data/tpch_q8_expected.txt", "TPCH"),
            id="tpch_q8",
        ),
        pytest.param(
            ("plan_data/tpch_q9.sql", "plan_data/tpch_q9_expected.txt", "TPCH"),
            id="tpch_q9",
        ),
        pytest.param(
            ("plan_data/tpch_q10.sql", "plan_data/tpch_q10_expected.txt", "TPCH"),
            id="tpch_q10",
        ),
        pytest.param(
            ("plan_data/tpch_q11.sql", "plan_data/tpch_q11_expected.txt", "TPCH"),
            id="tpch_q11",
        ),
        pytest.param(
            ("plan_data/tpch_q12.sql", "plan_data/tpch_q12_expected.txt", "TPCH"),
            id="tpch_q12",
        ),
        pytest.param(
            ("plan_data/tpch_q13.sql", "plan_data/tpch_q13_expected.txt", "TPCH"),
            id="tpch_q13",
        ),
        pytest.param(
            ("plan_data/tpch_q14.sql", "plan_data/tpch_q14_expected.txt", "TPCH"),
            id="tpch_q14",
        ),
        pytest.param(
            ("plan_data/tpch_q15.sql", "plan_data/tpch_q15_expected.txt", "TPCH"),
            id="tpch_q15",
        ),
        pytest.param(
            ("plan_data/tpch_q16.sql", "plan_data/tpch_q16_expected.txt", "TPCH"),
            id="tpch_q16",
        ),
        pytest.param(
            ("plan_data/tpch_q17.sql", "plan_data/tpch_q17_expected.txt", "TPCH"),
            id="tpch_q17",
        ),
        pytest.param(
            ("plan_data/tpch_q18.sql", "plan_data/tpch_q18_expected.txt", "TPCH"),
            id="tpch_q18",
        ),
        pytest.param(
            ("plan_data/tpch_q19.sql", "plan_data/tpch_q19_expected.txt", "TPCH"),
            id="tpch_q19",
        ),
        pytest.param(
            ("plan_data/tpch_q20.sql", "plan_data/tpch_q20_expected.txt", "TPCH"),
            id="tpch_q20",
        ),
        pytest.param(
            ("plan_data/tpch_q21.sql", "plan_data/tpch_q21_expected.txt", "TPCH"),
            id="tpch_q21",
        ),
        pytest.param(
            ("plan_data/tpch_q22.sql", "plan_data/tpch_q22_expected.txt", "TPCH"),
            id="tpch_q22",
        ),
        pytest.param(
            ("plan_data/engage3.sql", "plan_data/engage3_expected.txt", "engage3"),
            id="engage3",
        ),
    ],
)
def test_plan(plan_tests, tpch_data_schema_only, engage3_ctx, datapath):
    """Tests that the sql files in the plan_data directory generate the same
    plan string as the reference solutions."""
    query_path, expected_path, ctx_str = plan_tests

    # Extract the relevent context dictionary based on the name
    if ctx_str == "TPCH":
        ctx = tpch_data_schema_only[0]
    elif ctx_str == "engage3":
        ctx = engage3_ctx
    else:
        pytest.skip(f"CTX name unknown: {ctx_str}")
    with open(datapath(query_path), "r") as f:
        query = f.read()
    with open(datapath(expected_path), "r") as f:
        expected = f.read()
    bc = bodosql.BodoSQLContext(ctx)
    plan = bc.generate_plan(query)
    assert plan == expected
