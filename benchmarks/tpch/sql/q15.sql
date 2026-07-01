SELECT
  s.s_suppkey,
  s_name,
  s_address,
  s_phone,
  total_revenue
FROM (
  SELECT
    s_suppkey,
    SUM(l_extendedprice * (1 - l_discount)) AS total_revenue
  FROM supplier
  JOIN lineitem ON s_suppkey = l_suppkey
  GROUP BY s_suppkey
) rev
JOIN supplier s ON rev.s_suppkey = s.s_suppkey
ORDER BY total_revenue DESC
LIMIT 100;
