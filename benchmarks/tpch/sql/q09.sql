SELECT
  nation,
  SUM(amount) AS total_revenue
FROM (
  SELECT
    n_name AS nation,
    l_extendedprice * (1 - l_discount) AS amount
  FROM part
  JOIN lineitem ON p_partkey = l_partkey
  JOIN supplier ON s_suppkey = l_suppkey
  JOIN nation ON s_nationkey = n_nationkey
  WHERE p_name LIKE '%green%'
) t
GROUP BY nation
ORDER BY total_revenue DESC;
