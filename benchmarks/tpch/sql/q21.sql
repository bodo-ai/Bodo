SELECT
  s_name,
  s_address,
  s_phone,
  SUM(l_extendedprice * (1 - l_discount)) AS revenue
FROM supplier
JOIN lineitem ON s_suppkey = l_suppkey
JOIN part ON l_partkey = p_partkey
WHERE p_name LIKE '%steel%'
  AND l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
GROUP BY s_name, s_address, s_phone
ORDER BY revenue DESC
LIMIT 100;
