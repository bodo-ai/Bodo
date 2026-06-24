SELECT SUM(l_extendedprice * (1 - l_discount)) AS revenue
FROM lineitem
JOIN part ON l_partkey = p_partkey
WHERE (
  p_brand = 'Brand#12' AND p_container IN ('SM CASE', 'SM BOX') AND l_quantity BETWEEN 1 AND 11
) OR (
  p_brand = 'Brand#23' AND p_container IN ('MED BAG', 'MED BOX') AND l_quantity BETWEEN 10 AND 20
) OR (
  p_brand = 'Brand#34' AND p_container IN ('LG CASE', 'LG BOX') AND l_quantity BETWEEN 20 AND 30
);
