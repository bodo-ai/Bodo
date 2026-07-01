SELECT
  l_shipmode,
  SUM(CASE WHEN o_orderpriority = '1-URGENT' THEN 1 ELSE 0 END) AS high_line_count,
  SUM(CASE WHEN o_orderpriority <> '1-URGENT' THEN 1 ELSE 0 END) AS low_line_count
FROM orders
JOIN lineitem ON o_orderkey = l_orderkey
WHERE o_orderdate >= DATE '1992-01-01'
  AND o_orderdate < DATE '1993-01-01'
  AND l_commitdate < l_receiptdate
GROUP BY l_shipmode
ORDER BY l_shipmode;
