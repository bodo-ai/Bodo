SELECT
  s_name,
  s_address,
  s_phone,
  SUM(l_extendedprice * (1 - l_discount)) AS revenue
FROM supplier
JOIN lineitem ON s_suppkey = l_suppkey
JOIN orders ON l_orderkey = o_orderkey
JOIN nation ON s_nationkey = n_nationkey
WHERE o_orderdate >= DATE '1995-01-01'
  AND o_orderdate < DATE '1996-01-01'
  AND l_returnflag = 'R'
GROUP BY s_name, s_address, s_phone
ORDER BY revenue DESC
LIMIT 100;
