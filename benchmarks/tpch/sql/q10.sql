SELECT
  c_custkey,
  c_name,
  SUM(l_extendedprice * (1 - l_discount)) AS revenue,
  c_acctbal,
  c_address,
  c_phone,
  c_comment
FROM customer
JOIN orders ON c_custkey = o_custkey
JOIN lineitem ON l_orderkey = o_orderkey
WHERE o_orderdate >= DATE '1993-10-01'
  AND o_orderdate < DATE '1994-01-01'
GROUP BY c_custkey, c_name, c_acctbal, c_address, c_phone, c_comment
ORDER BY revenue DESC
LIMIT 20;
