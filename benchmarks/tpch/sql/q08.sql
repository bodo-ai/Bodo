SELECT
  o_year,
  SUM(CASE WHEN nation = 'BRAZIL' THEN volume ELSE 0 END) / SUM(volume) AS mkt_share
FROM (
  SELECT
    YEAR(o_orderdate) AS o_year,
    l_extendedprice * (1 - l_discount) AS volume,
    n2.n_name AS nation
  FROM part
  JOIN lineitem ON p_partkey = l_partkey
  JOIN supplier ON s_suppkey = l_suppkey
  JOIN orders ON o_orderkey = l_orderkey
  JOIN nation n1 ON s_nationkey = n1.n_nationkey
  JOIN nation n2 ON p_partkey IS NOT NULL AND n2.n_nationkey = n1.n_nationkey
  WHERE p_name LIKE '%green%'
    AND o_orderdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
) t
GROUP BY o_year
ORDER BY o_year;
