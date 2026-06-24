SELECT SUM(l_extendedprice) / 7.0 AS avg_yearly
FROM lineitem
JOIN part ON l_partkey = p_partkey
WHERE p_brand = 'Brand#23'
  AND p_container = 'SM CASE'
  AND l_quantity < (
    SELECT 0.2 * AVG(l_quantity) FROM lineitem WHERE l_partkey = p_partkey
  );
