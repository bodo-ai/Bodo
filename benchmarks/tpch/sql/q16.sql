SELECT
  p_brand,
  p_type,
  p_size,
  COUNT(DISTINCT ps_suppkey) AS supplier_cnt
FROM part
JOIN partsupp ON p_partkey = ps_partkey
WHERE p_brand <> 'Brand#45'
  AND p_type NOT LIKE 'MEDIUM%'
  AND p_size IN (49, 14, 23)
GROUP BY p_brand, p_type, p_size
ORDER BY supplier_cnt DESC, p_brand, p_type, p_size;
