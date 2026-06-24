SELECT
  c_name,
  c_custkey,
  o_orderkey,
  o_orderdate,
  o_totalprice,
  SUM(l_quantity) AS order_qty
FROM customer
JOIN orders ON c_custkey = o_custkey
JOIN lineitem ON o_orderkey = l_orderkey
JOIN nation ON c_nationkey = n_nationkey
WHERE n_name = 'UNITED STATES'
  AND o_orderdate >= DATE '1996-01-01'
  AND o_orderdate < DATE '1997-01-01'
GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice
ORDER BY o_totalprice DESC, o_orderdate
LIMIT 100;
