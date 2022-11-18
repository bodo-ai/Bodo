WITH temp1 as
        (
        SELECT rphd.UUID                                                                            AS uuid
            , 'eb001209-33d7-4d3f-bcaa-9dd6b8cf08b0'                                                AS client_id
            , rp.ret_product_id                                                                     AS ret_product_id
            , mp.mst_product_id                                                                     AS mst_product_id
            , rs.store_id                                                                           AS comp_store_id
            , DATE_TRUNC( 'WEEK', rphd.last_seen )                                                  AS weekdate
            , rphd.created_date                                                                     AS created_date
            ,CAST(( YEAROFWEEKISO( rphd.last_seen )||LPAD( WEEKISO( rphd.last_seen ), 2, '0' ) ) as INT)    AS client_week
            , COALESCE( MPH.L1_NAME, TRIM( rphd.CATEGORY_L1 ) )                                     AS COMP_CATEGORY_L1
            , COALESCE( MPH.L2_NAME, TRIM( rphd.CATEGORY_L2 ) )                                     AS COMP_CATEGORY_L2
            , COALESCE( MPH.L3_NAME, TRIM( rphd.CATEGORY_L3 ) )                                     AS COMP_CATEGORY_L3
            , COALESCE( MPH.L4_NAME, TRIM( rphd.CATEGORY_L4 ) )                                     AS COMP_CATEGORY_L4
            , COALESCE( MPH.L5_NAME, TRIM( rphd.CATEGORY_L5 ) )                                     AS COMP_CATEGORY_L5
            , COALESCE( MPH.L6_NAME, TRIM( rphd.CATEGORY_L6 ) )                                     AS COMP_CATEGORY_L6
            , COALESCE( MPH.L7_NAME, TRIM( rphd.CATEGORY_L7 ) )                                     AS COMP_CATEGORY_L7
            , COALESCE( MPH.L8_NAME, TRIM( rphd.CATEGORY_L8 ) )                                     AS COMP_CATEGORY_L8
            , COALESCE( rp.description, rphd.description )                                          AS comp_description
            , mp.pack_size                                                                          AS comp_pack_size
            , mp.unit_size                                                                          AS comp_unit_size
            , rphd.sku                                                                              AS comp_sku
            , COALESCE( mp.gtin, CAST(mp.plu as VARCHAR), rphd.upc )                                           AS comp_upc
            , rb.banner_name                                                                        AS banner_name
            , rb.banner_id                                                                          AS banner_id
            , rs.address_line_1                                                                     AS address
            , rs.city                                                                               AS city
            , rs.state                                                                              AS state
            , rs.postal_code                                                                        AS postal_code
            , rs.latitude                                                                           AS latitude
            , rs.longitude                                                                          AS longitude
            , ROUND( rphd.reg_price   / GREATEST( COALESCE( rphd.reg_multiple, 0 ), 1 ), 2 )      AS comp_regular_price_normalized,
            CASE
                WHEN ROUND( rphd.promo_price / GREATEST( COALESCE( rphd.promo_multiple, 0 ), 1 ), 2 ) <
                     ROUND( rphd.reg_price   / GREATEST( COALESCE( rphd.reg_multiple, 0 ), 1 ), 2 )
                     THEN ROUND( rphd.promo_price / GREATEST( COALESCE( rphd.promo_multiple, 0 ), 1 ), 2 )
                ELSE NULL
              END                                                                                   AS comp_promo_price_normalized,
              CASE
                WHEN ROUND( rphd.promo_price / GREATEST( COALESCE( rphd.promo_multiple, 0 ), 1 ), 2 ) <
                     ROUND( rphd.reg_price   / GREATEST( COALESCE( rphd.reg_multiple, 0 ), 1 ), 2 )
                     THEN ROUND( rphd.promo_price / GREATEST( COALESCE( rphd.promo_multiple, 0 ), 1 ), 2 )
                ELSE ROUND( rphd.reg_price   / GREATEST( COALESCE( rphd.reg_multiple, 0 ), 1 ), 2 )
              END                                                                                   AS comp_best_price_normalized,
    'Market Insights Data'                     AS notes,
    rphd.last_seen as rhpd_last_seen

        FROM p_ret_price_history_denorm         rphd
        INNER JOIN s_ret_store                  rs
            ON rphd.store_id = rs.store_id
        INNER JOIN s_ret_banner rb
            ON rs.banner_id = rb.banner_id
        INNER JOIN p_ret_product rp
            ON rphd.ret_product_id = rp.ret_product_id
        INNER JOIN p_mst_product mp
            ON rp.mst_product_id = mp.mst_product_id
        LEFT JOIN p_v_mph_flat mph
            ON mp.mph_node_id = mph.mph_node_id
        WHERE rphd.ret_product_id     IS NOT NULL
            AND COALESCE( rphd.reg_price, 1 )   > 0
            AND COALESCE( rphd.promo_price, 1 ) > 0
), temp2 as (select
   uuid,
   client_id,
   ret_product_id,
   mst_product_id,
   comp_store_id,
   weekdate,
   created_date,
   client_week,
   COMP_CATEGORY_L1,
    COMP_CATEGORY_L2,
    COMP_CATEGORY_L3,
    COMP_CATEGORY_L4,
    COMP_CATEGORY_L5,
    COMP_CATEGORY_L6,
    COMP_CATEGORY_L7,
    COMP_CATEGORY_L8,
    comp_description,
    comp_pack_size,
    comp_unit_size,
    comp_sku,
    comp_upc,
    banner_name,
    banner_id,
    address,
    city,
    state,
    postal_code,
    latitude,
    longitude,
    comp_regular_price_normalized,
    comp_promo_price_normalized,
    comp_best_price_normalized,
    notes,
    rhpd_last_seen,
    ROW_NUMBER() OVER(PARTITION BY client_week , comp_store_id, ret_product_id ORDER BY rhpd_last_seen DESC) as row_num
    FROM temp1
    WHERE comp_best_price_normalized      > 0
) select
   uuid,
   client_id,
   ret_product_id,
   mst_product_id,
   comp_store_id,
   weekdate,
   created_date,
   client_week,
   comp_category_l1,
    comp_category_l2,
    comp_category_l3,
    comp_category_l4,
    comp_category_l5,
    comp_category_l6,
    comp_category_l7,
    comp_category_l8,
    comp_description,
    comp_pack_size,
    comp_unit_size,
    comp_sku,
    comp_upc,
    banner_name as comp_banner,
    banner_id as comp_banner_id,
    address as comp_address,
    city as comp_city,
    state as comp_state,
    postal_code as comp_postal_code,
    latitude as comp_latitude,
    longitude as comp_longitude,
    comp_regular_price_normalized,
    comp_promo_price_normalized,
    comp_best_price_normalized,
    notes
    FROM temp2
    WHERE row_num = 1
        ORDER BY rhpd_last_seen, created_date
