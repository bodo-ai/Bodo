SELECT
  c_count,
  COUNT(*) AS custdist
FROM (
  SELECT c_custkey, COUNT(o_orderkey) AS c_count
  FROM customer
  LEFT JOIN orders ON c_custkey = o_custkey
  GROUP BY c_custkey
) t
GROUP BY c_count
ORDER BY custdist DESC, c_count DESC;
